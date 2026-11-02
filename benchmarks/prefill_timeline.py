"""Profiler traces and process-held-out Prefill cost validation, with no core edits."""

import argparse
import atexit
import csv
from collections import defaultdict
from contextlib import ExitStack
import ctypes
import ctypes.util
import gzip
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import random
from statistics import fmean, stdev
import subprocess
import sys
import time
import traceback

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.metrics import percentile
from benchmarks.mixed_load_bench import find_model, gpu_snapshot
from benchmarks.prefill_cost_model import evaluate_models
from benchmarks.prefill_profile import (ProfileCase, TimingProbe, prepare_case, prompt_ids,
                                        require_released, sample_case, validate_samples)
from benchmarks.reliability import write_csv


CASES = (ProfileCase(16, 0, 1), ProfileCase(64, 0, 1), ProfileCase(256, 0, 1),
         ProfileCase(16, 256, 1), ProfileCase(256, 256, 1),
         ProfileCase(64, 0, 4), ProfileCase(256, 0, 4))


class Telemetry:
    """Read-only NVML through ctypes; no package install or clock/power changes."""

    def __init__(self):
        self.lib = None
        self.error = None
        try:
            library = ctypes.util.find_library("nvidia-ml") or "libnvidia-ml.so.1"
            self.lib = ctypes.CDLL(library)
            self.lib.nvmlInit_v2.restype = ctypes.c_int
            self.lib.nvmlShutdown.restype = ctypes.c_int
            self.lib.nvmlDeviceGetHandleByIndex_v2.argtypes = [ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p)]
            self.lib.nvmlDeviceGetHandleByIndex_v2.restype = ctypes.c_int
            self.handle = ctypes.c_void_p()
            if self.lib.nvmlInit_v2() or self.lib.nvmlDeviceGetHandleByIndex_v2(0, ctypes.byref(self.handle)):
                raise RuntimeError("NVML initialization/device lookup failed")
            for name in ("nvmlDeviceGetClockInfo", "nvmlDeviceGetTemperature"):
                fn = getattr(self.lib, name)
                fn.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.POINTER(ctypes.c_uint)]
                fn.restype = ctypes.c_int
            self.lib.nvmlDeviceGetPowerUsage.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint)]
            self.lib.nvmlDeviceGetPowerUsage.restype = ctypes.c_int
        except (OSError, AttributeError, RuntimeError) as exc:
            self.lib = None
            self.error = str(exc)

    def capture(self):
        if self.lib is None:
            return dict(source="nvidia-smi fallback", nvml_error=self.error, **gpu_snapshot())
        result = dict(source="NVML ctypes", perf_counter=time.perf_counter())
        for key, name, selector, scale in (
                ("sm_clock_mhz", "nvmlDeviceGetClockInfo", 1, 1),
                ("temperature_c", "nvmlDeviceGetTemperature", 0, 1),
                ("power_w", "nvmlDeviceGetPowerUsage", None, .001)):
            value = ctypes.c_uint()
            fn = getattr(self.lib, name)
            status = (fn(self.handle, ctypes.byref(value)) if selector is None else
                      fn(self.handle, selector, ctypes.byref(value)))
            result[key] = value.value * scale if status == 0 else None
            if status:
                result[key + "_error"] = f"NVML status {status}"
        return result

    def close(self):
        if self.lib is not None:
            self.lib.nvmlShutdown()


def merge_intervals(intervals):
    merged = []
    for start, end in sorted(intervals):
        if end < start:
            raise ValueError("negative activity duration")
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return merged


def interval_stats(values):
    return dict(count=len(values), mean_ms=fmean(values) if values else None,
                p50_ms=percentile(values, .5), p95_ms=percentile(values, .95),
                max_ms=max(values) if values else None)


def analyze_trace(trace):
    """Chrome/Kineto timestamps are us. Gaps describe observed activity, not global GPU idle."""
    events = [e for e in trace["traceEvents"] if e.get("ph") == "X" and "dur" in e]
    cpu_scopes = [e for e in events if e.get("cat") == "user_annotation"]
    # Kineto can mirror record_function annotations onto the GPU timeline.
    # They are not additional CPU scopes and must not be counted twice.
    models = [e for e in cpu_scopes if e.get("name") == "prefill::model"]
    steps = [e for e in cpu_scopes if e.get("name") == "prefill::step"]
    if len(models) != 1 or len(steps) != 1:
        raise ValueError("trace must contain exactly one target model and step annotation")
    model, step = models[0], steps[0]
    def inside(event, scope):
        return scope["ts"] <= event["ts"] <= scope["ts"] + scope["dur"]
    runtime = {e.get("args", {}).get("correlation"): e for e in events
               if e.get("cat") in ("cuda_runtime", "cuda_driver") and
               e.get("args", {}).get("correlation") is not None}
    external = {e.get("args", {}).get("External id"): e for e in events
                if e.get("cat") == "cpu_op" and e.get("args", {}).get("External id") is not None}
    scopes = [e for e in cpu_scopes if e.get("name", "").startswith("prefill_ops::")]
    kernels, unmatched = [], 0
    for event in events:
        if event.get("cat") != "kernel":
            continue
        args = event.get("args", {})
        launcher = runtime.get(args.get("correlation")) or external.get(args.get("External id"))
        if launcher is None:
            unmatched += 1
            continue
        if not inside(launcher, model):
            continue
        containing = [scope for scope in scopes if inside(launcher, scope)]
        category = (min(containing, key=lambda e: e["dur"])["name"].split("::", 1)[1]
                    if containing else "Other")
        kernels.append(dict(name=event["name"], start_us=event["ts"], end_us=event["ts"] + event["dur"],
                            duration_us=event["dur"], stream=args.get("stream"), category=category,
                            launch_start_us=launcher["ts"], launch_duration_us=launcher["dur"]))
    category_rows = []
    for name in ("Attention", "Linear", "RMSNorm", "LMHead", "Other"):
        selected = [e for e in kernels if e["category"] == name]
        cpu = [e for e in scopes if e["name"] == "prefill_ops::" + name]
        category_rows.append(dict(category=name, cpu_scope_ms=sum(e["dur"] for e in cpu) / 1000,
                                  cuda_kernel_sum_ms=sum(e["duration_us"] for e in selected) / 1000,
                                  kernel_count=len(selected)))
    result = dict(model_cpu_scope_ms=model["dur"] / 1000, step_cpu_scope_ms=step["dur"] / 1000,
                  model_kernel_count=len(kernels), unmatched_kernel_count=unmatched,
                  step_kernel_count=sum(e.get("cat") == "kernel" for e in events),
                  categories=category_rows, kernels=kernels, streams=[])
    if not kernels:
        result.update(cuda_trace_available=False, cuda_trace_error="no correlated model kernels; CUDA/CUPTI trace unavailable")
        return result
    merged = merge_intervals([(k["start_us"], k["end_us"]) for k in kernels])
    low, high = merged[0][0], merged[-1][1]
    active = sum(b - a for a, b in merged)
    observed = [(max(low, e["ts"]), min(high, e["ts"] + e["dur"])) for e in events
                if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")
                and e["ts"] < high and e["ts"] + e["dur"] > low]
    observed_union = sum(b - a for a, b in merge_intervals(observed))
    result.update(cuda_trace_available=True, model_kernel_sum_ms=sum(k["duration_us"] for k in kernels) / 1000,
                  model_kernel_active_union_ms=active / 1000, model_kernel_span_ms=(high - low) / 1000,
                  model_kernel_free_ms=(high - low - active) / 1000,
                  observed_gpu_activity_free_ms=(high - low - observed_union) / 1000)
    groups = defaultdict(list)
    for kernel in kernels:
        groups[kernel["stream"]].append(kernel)
    for stream, rows in groups.items():
        intervals = merge_intervals([(k["start_us"], k["end_us"]) for k in rows])
        gaps = [(b[0] - a[1]) / 1000 for a, b in zip(intervals, intervals[1:])]
        launches = sorted(rows, key=lambda k: k["launch_start_us"])
        launch_gaps = [max(0, b["launch_start_us"] - a["launch_start_us"] - a["launch_duration_us"]) / 1000
                       for a, b in zip(launches, launches[1:])]
        result["streams"].append(dict(stream=stream, kernel_gaps=interval_stats(gaps),
                                     cpu_launch_gaps=interval_stats(launch_gaps)))
    return result


class OperatorScopes:
    """Profiler-only Python wrappers; all source files and model operations stay intact."""

    def __init__(self, engine, torch, probe):
        self.engine, self.torch, self.probe = engine, torch, probe
        self.restore = ExitStack()

    def replace(self, obj, name, method):
        if name in vars(obj):
            self.restore.callback(setattr, obj, name, vars(obj)[name])
        else:
            self.restore.callback(delattr, obj, name)
        setattr(obj, name, method)

    def __enter__(self):
        for module in self.engine.model_runner.model.modules():
            name = type(module).__name__
            category = ("Attention" if name == "Attention" else "RMSNorm" if name == "RMSNorm" else
                        "LMHead" if name == "ParallelLMHead" else
                        "Linear" if name.endswith("Linear") else None)
            if category is None:
                continue
            original = module.forward
            def wrapped(*args, _original=original, _category=category, **kwargs):
                with self.torch.profiler.record_function("prefill_ops::" + _category):
                    return _original(*args, **kwargs)
            self.replace(module, "forward", wrapped)
        original_model = self.engine.model_runner.run_model
        def model(*args, **kwargs):
            if self.probe.capture is None:
                return original_model(*args, **kwargs)
            with self.torch.profiler.record_function("prefill::model"):
                return original_model(*args, **kwargs)
        self.replace(self.engine.model_runner, "run_model", model)
        return self

    def __exit__(self, *exc):
        self.restore.close()


def profile_case(engine, case, prompts, params, probe, torch, counters, directory):
    # No prefix work or allocation is inside the target Profiler trace.
    prepare_case(engine, case, prompts, params)
    torch.cuda.synchronize()
    supported = torch.profiler.supported_activities()
    activities = [torch.profiler.ProfilerActivity.CPU]
    if torch.profiler.ProfilerActivity.CUDA in supported:
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    frames = counters["frames"]["total"]
    probe.capture = {}
    try:
        with torch.profiler.profile(activities=activities, record_shapes=False,
                                    with_stack=False, profile_memory=False) as profiler:
            with torch.profiler.record_function("prefill::step"):
                started = time.perf_counter()
                outputs, count = engine.step()
                stopped = time.perf_counter()
            torch.cuda.synchronize()
    finally:
        captured = probe.capture
        probe.capture = None
    if counters["frames"]["total"] != frames:
        raise RuntimeError(f"compilation occurred inside Profiler trace: {case.case_id}")
    if (count != case.q or captured["q_each"] != [case.q_per_request] * case.batch_size
            or captured["c_each"] != [case.c] * case.batch_size or len(outputs) != case.batch_size):
        raise RuntimeError("Profiler target q/c/batch differs from request")
    require_released(engine)
    path = directory / f"{case.case_id}.trace.json"
    profiler.export_chrome_trace(str(path))
    trace = json.loads(path.read_text(encoding="utf-8"))
    analysis = analyze_trace(trace)
    compressed = path.with_suffix(path.suffix + ".gz")
    with gzip.open(compressed, "wt", encoding="utf-8") as file:
        json.dump(trace, file)
    path.unlink()
    operators = [dict(name=event.key, calls=event.count,
                      self_cpu_ms=event.self_cpu_time_total / 1000,
                      inclusive_cpu_ms=event.cpu_time_total / 1000,
                      self_cuda_ms=event.self_device_time_total / 1000,
                      inclusive_cuda_ms=event.device_time_total / 1000)
                 for event in profiler.key_averages()]
    write_csv(directory / f"{case.case_id}.operators.csv", operators)
    analysis.update(**case.to_dict(), instrumentation="torch_profiler", trace_file=str(compressed),
                    profiled_step_wall_ms=(stopped - started) * 1000,
                    profiled_cuda_event_ms=probe.events[0].elapsed_time(probe.events[1]),
                    operators=operators, compile_frames=0)
    return analysis


def telemetry_fields(before, after):
    result = dict(telemetry_before=before, telemetry_after=after)
    for key in ("sm_clock_mhz", "temperature_c", "power_w"):
        result[key + "_before"] = before.get(key)
        result[key + "_after"] = after.get(key)
    return result


def run_worker(manifest_path, kind, process_id, output_path):
    manifest = json.loads(Path(manifest_path).read_text())
    options = manifest["options"]
    cases = [ProfileCase(row["q"], row["c"], row["batch_size"]) for row in manifest["cases"]]
    role = "train" if process_id % 2 == 0 else "holdout"
    rows, timelines, environment, warmups, prompts = [], [], {}, [], {}
    error = None
    try:
        import torch
        from torch._dynamo.utils import counters
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; no synthetic performance fallback")
        from nanovllm import LLM, SamplingParams
        torch.manual_seed(options["seed"])
        torch.cuda.manual_seed_all(options["seed"])
        engine = LLM(options["model"], tensor_parallel_size=1, enforce_eager=True,
                     max_model_len=1024, max_num_batched_tokens=256, max_num_seqs=4,
                     gpu_memory_utilization=.65, scheduling_policy="prefill_first",
                     decode_queue_policy="head_first")
        def close_engine():
            atexit.unregister(engine.exit)
            engine.exit()
        with ExitStack() as cleanup:
            cleanup.callback(close_engine)
            telemetry = Telemetry()
            cleanup.callback(telemetry.close)
            environment = dict(gpu_name=torch.cuda.get_device_name(0), torch=torch.__version__,
                               cuda=torch.version.cuda, dtype=str(engine.model_runner.config.hf_config.dtype),
                               python=sys.version, kv_blocks=len(engine.scheduler.block_manager.blocks),
                               supported_profiler_activities=[str(v) for v in torch.profiler.supported_activities()])
            for name in ("transformers", "triton", "flash-attn"):
                environment[name] = importlib.metadata.version(name)
            prompts = {case.case_id: prompt_ids(engine.tokenizer, case, options["seed"]) for case in cases}
            events = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
            events[0].record()
            events[1].record()
            events[1].synchronize()
            params = SamplingParams(max_tokens=1, temperature=.6, ignore_eos=True)
            rng = random.Random(options["seed"] + process_id)
            ordered = list(cases)
            rng.shuffle(ordered)
            with TimingProbe(engine, events) as probe:
                for order, case in enumerate(ordered):
                    print(f"{kind} process {process_id}: warming {case.case_id}", flush=True)
                    for repeat in range(options["warmup_rounds"]):
                        warmups.append(sample_case(engine, case, prompts[case.case_id], params, probe, torch,
                                                   counters, "warmup", repeat, order))
                if kind == "measurement":
                    for repeat in range(options["repeats"]):
                        ordered = list(cases)
                        rng.shuffle(ordered)
                        for order, case in enumerate(ordered):
                            before = telemetry.capture()
                            row = sample_case(engine, case, prompts[case.case_id], params, probe, torch,
                                              counters, "measure", repeat, order)
                            after = telemetry.capture()
                            rows.append(dict(**row, process_id=process_id, role=role,
                                             instrumentation="none", **telemetry_fields(before, after)))
                        print(f"process {process_id}: measurement round {repeat+1}/{options['repeats']}", flush=True)
                else:
                    for case in ordered:
                        control = sample_case(engine, case, prompts[case.case_id], params, probe, torch,
                                              counters, "measure", 0, 0)
                        with OperatorScopes(engine, torch, probe):
                            # Warm Python/module scopes too, outside the Profiler trace.
                            for repeat in range(2):
                                sample_case(engine, case, prompts[case.case_id], params, probe, torch,
                                            counters, "warmup", repeat, 0)
                            directory = Path(output_path).parent / "traces"
                            directory.mkdir(exist_ok=True)
                            timeline = profile_case(engine, case, prompts[case.case_id], params, probe,
                                                    torch, counters, directory)
                        timeline["unprofiled_control_step_wall_ms"] = control["step_wall_ms"]
                        timelines.append(timeline)
                        print(f"profiled {case.case_id}: kernels={timeline['model_kernel_count']}", flush=True)
            resource = require_released(engine)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
        resource = None
    payload = dict(kind=kind, process_id=process_id, role=role if kind == "measurement" else None,
                   status="failed" if error else "ok", error=error, environment=environment,
                   measurements=rows, timelines=timelines, warmups=warmups, prompt_ids=prompts,
                   resources=resource)
    Path(output_path).write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                                 encoding="utf-8")
    return 1 if error else 0


def export_csv(path, rows):
    # Preserve nested values as JSON, rather than Python dictionary repr.
    write_csv(path, [{key: json.dumps(value, sort_keys=True) if isinstance(value, dict) else value
                      for key, value in row.items()} for row in rows])


def summarize_measurements(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["process_id"], row["case_id"])].append(row)
    per_process = []
    for (process_id, case_id), values in sorted(groups.items()):
        result = dict(process_id=process_id, role=values[0]["role"], case_id=case_id,
                      q=values[0]["q"], c=values[0]["c"], batch_size=values[0]["batch_size"], samples=len(values))
        for field in ("step_wall_ms", "cuda_model_stream_ms", "model_submit_wall_ms", "prepare_wall_ms"):
            numbers = [row[field] for row in values]
            result.update({field + "_mean": fmean(numbers), field + "_p50": percentile(numbers, .5),
                           field + "_p95": percentile(numbers, .95), field + "_max": max(numbers)})
        per_process.append(result)
    by_case = defaultdict(list)
    for row in per_process:
        by_case[row["case_id"]].append(row)
    cases = []
    for case_id, values in sorted(by_case.items()):
        result = dict(case_id=case_id, processes=len(values), samples=sum(v["samples"] for v in values))
        for field in ("step_wall_ms", "cuda_model_stream_ms"):
            numbers = [v[field + "_mean"] for v in values]
            result.update({field + "_mean": fmean(numbers), field + "_process_mean_sd": stdev(numbers) if len(numbers) > 1 else 0,
                           field + "_process_mean_min": min(numbers), field + "_process_mean_max": max(numbers)})
        cases.append(result)
    return per_process, cases


def validate_measurements(rows, manifest):
    options = manifest["options"]
    cases = {row["case_id"]: row for row in manifest["cases"]}
    expected = {(pid, case_id, repeat) for pid in range(options["processes"])
                for case_id in cases for repeat in range(options["repeats"])}
    seen = set()
    for row in rows:
        key = (row["process_id"], row["case_id"], row["repeat"])
        if key in seen or key not in expected:
            raise ValueError(f"duplicate/unexpected sample: {key}")
        seen.add(key)
        case = cases[row["case_id"]]
        if (row["instrumentation"] != "none" or row["phase"] != "measure" or row["compile_frames"] != 0
                or row["role"] != ("train" if row["process_id"] % 2 == 0 else "holdout")
                or row["q_each"] != [case["q_per_request"]] * case["batch_size"]
                or row["c_each"] != [case["c"]] * case["batch_size"]
                or row["output_tokens"] != case["batch_size"]
                or row["max_num_batched_tokens"] != case["q"]
                or row["kv_used_after"] or row["kv_ref_count_after"]
                or row["kv_free_after"] != row["kv_total_blocks"]):
            raise ValueError(f"measurement integrity failed: {key}")
        for field in ("q", "c", "batch_size"):
            if row[field] != case[field]:
                raise ValueError(f"wrong {field}: {key}")
        for field in ("step_wall_ms", "cuda_model_stream_ms"):
            if not math.isfinite(row[field]) or row[field] <= 0:
                raise ValueError(f"invalid time: {key}")
        elapsed = (row["wall_end_perf_counter"] - row["wall_start_perf_counter"]) * 1000
        if not math.isclose(elapsed, row["step_wall_ms"], rel_tol=1e-8, abs_tol=1e-6):
            raise ValueError(f"inconsistent monotonic time: {key}")
    if seen != expected:
        raise ValueError(f"missing {len(expected - seen)} measurements")
    return True


def verify_results(directory):
    directory = Path(directory)
    payload = json.loads((directory / "timeline.json").read_text())
    manifest = payload["manifest"]
    rows = payload["measurements"]
    options = manifest["options"]
    kinds = ([] if options["profile_only"] else [("measurement", pid) for pid in range(options["processes"])])
    if not options["measure_only"]:
        kinds.append(("profiler", options["processes"]))
    workers = []
    shared_prompts = None
    cases = [ProfileCase(c["q"], c["c"], c["batch_size"]) for c in manifest["cases"]]
    for kind, process_id in kinds:
        worker = json.loads((directory / f"{kind}_{process_id}.json").read_text())
        if worker["status"] != "ok" or worker["kind"] != kind or worker["process_id"] != process_id:
            raise ValueError("worker identity/status differs")
        state = worker["resources"]
        if state["used_blocks"] or state["ref_count_sum"] or state["free_blocks"] != state["total_blocks"]:
            raise ValueError("worker KV blocks not fully released")
        if shared_prompts is None:
            shared_prompts = worker["prompt_ids"]
        if shared_prompts != worker["prompt_ids"]:
            raise ValueError("workload differs across processes")
        for case in cases:
            prompts = worker["prompt_ids"][case.case_id]
            if len(prompts) != case.batch_size or any(len(p) != case.c + case.q_per_request for p in prompts):
                raise ValueError("prompt lengths differ from design")
        if kind == "measurement":
            validate_samples(worker["warmups"] + worker["measurements"], cases,
                             options["repeats"], options["warmup_rounds"], True)
        workers.append(worker)
    if rows != [row for w in workers for row in w["measurements"]]:
        raise ValueError("worker and aggregate measurements differ")
    if payload["timelines"] != [row for w in workers for row in w["timelines"]]:
        raise ValueError("worker and aggregate timelines differ")
    expected_traces = set() if options["measure_only"] else {case.case_id for case in cases}
    if (len(payload["timelines"]) != len(expected_traces)
            or {r["case_id"] for r in payload["timelines"]} != expected_traces):
        raise ValueError("incomplete/duplicate Profiler traces")
    for name in ("prefill_profile.py", "prefill_timeline.py", "prefill_cost_model.py"):
        digest = hashlib.sha256((directory / name).read_bytes()).hexdigest()
        if digest != manifest["source_sha256"]["benchmarks/" + name]:
            raise ValueError("instrumentation source snapshot differs")
    if not manifest["options"]["profile_only"]:
        validate_measurements(rows, manifest)
        process_summary, case_summary = summarize_measurements(rows)
        if process_summary != payload["process_summary"] or case_summary != payload["case_summary"]:
            raise ValueError("summary does not match raw measurements")
        if evaluate_models(rows) != payload["cost_models"]:
            raise ValueError("model results do not match process-held-out data")
        with (directory / "raw_measurements.csv").open(newline="") as file:
            csv_rows = list(csv.DictReader(file))
        if len(csv_rows) != len(rows):
            raise ValueError("CSV sample count differs")
        for expected, actual in zip(rows, csv_rows):
            for key, value in expected.items():
                encoded = (json.dumps(value, sort_keys=True) if isinstance(value, dict) else
                           json.dumps(value) if isinstance(value, list) else "" if value is None else str(value))
                if actual[key] != encoded:
                    raise ValueError(f"CSV differs: {key}")
    for timeline in payload["timelines"]:
        with gzip.open(timeline["trace_file"], "rt", encoding="utf-8") as file:
            analysis = analyze_trace(json.load(file))
        for key, value in analysis.items():
            if timeline[key] != value:
                raise ValueError(f"trace summary differs: {timeline['case_id']}: {key}")
    return payload


def collect_results(directory, manifest, workers):
    rows = [row for worker in workers for row in worker["measurements"]]
    timelines = [row for worker in workers for row in worker["timelines"]]
    if not manifest["options"]["profile_only"]:
        validate_measurements(rows, manifest)
        process_summary, case_summary = summarize_measurements(rows)
        models = evaluate_models(rows)
        export_csv(directory / "raw_measurements.csv", rows)
        export_csv(directory / "process_summary.csv", process_summary)
        export_csv(directory / "case_summary.csv", case_summary)
        export_csv(directory / "holdout_predictions.csv", models["predictions"])
        export_csv(directory / "holdout_metrics.csv", models["holdout_metrics"])
        export_csv(directory / "holdout_process_metrics.csv", models["per_process_metrics"])
        (directory / "cost_models.json").write_text(json.dumps(models, indent=2, allow_nan=False) + "\n")
    else:
        process_summary, case_summary, models = [], [], None
    export_csv(directory / "timeline_summary.csv", [
        {key: value for key, value in row.items() if key not in ("kernels", "operators", "streams", "categories")}
        for row in timelines])
    for timeline in timelines:
        prefix = directory / f"{timeline['case_id']}"
        export_csv(str(prefix) + ".kernels.csv", timeline["kernels"])
        export_csv(str(prefix) + ".categories.csv", timeline["categories"])
        export_csv(str(prefix) + ".streams.csv", timeline["streams"])
    payload = dict(manifest=manifest, measurements=rows, timelines=timelines,
                   process_summary=process_summary, case_summary=case_summary, cost_models=models,
                   worker_resources=[dict(kind=w["kind"], process_id=w["process_id"], resources=w["resources"])
                                     for w in workers])
    (directory / "timeline.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    verify_results(directory)
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model")
    parser.add_argument("--output-dir", default="results_prefill_timeline")
    parser.add_argument("--processes", type=int, default=6)
    parser.add_argument("--repeats", type=int, default=15)
    parser.add_argument("--warmup-rounds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--process-timeout", type=float, default=600)
    parser.add_argument("--profile-only", action="store_true")
    parser.add_argument("--measure-only", action="store_true")
    parser.add_argument("--case-ids", nargs="+")
    parser.add_argument("--verify", metavar="DIRECTORY")
    parser.add_argument("--worker-manifest", help=argparse.SUPPRESS)
    parser.add_argument("--worker-kind", choices=("measurement", "profiler"), help=argparse.SUPPRESS)
    parser.add_argument("--worker-id", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker_manifest:
        return run_worker(args.worker_manifest, args.worker_kind, args.worker_id, args.worker_output)
    if args.verify:
        verified = verify_results(args.verify)
        print(f"Verified {len(verified['measurements'])} samples, {len(verified['timelines'])} traces")
        return 0
    if (args.processes < 4 or args.repeats < 2 or args.warmup_rounds < 1 or args.process_timeout <= 0
            or not math.isfinite(args.process_timeout) or (args.profile_only and args.measure_only)):
        parser.error("need >=4 processes, >=2 repeats, >=1 warmup, positive timeout, mutually exclusive modes")
    cases = [case for case in CASES if not args.case_ids or case.case_id in args.case_ids]
    if not cases or (args.case_ids and set(args.case_ids) != {case.case_id for case in cases}):
        parser.error("unknown/empty case selection")
    if not args.profile_only and len({case.q for case in cases}) < 2:
        parser.error("model comparison requires at least two q values")
    directory = Path(args.output_dir).resolve()
    if directory.exists() and any(directory.iterdir()):
        raise ValueError(f"output directory must be empty: {directory}")
    directory.mkdir(parents=True, exist_ok=True)
    options = dict(model=str(find_model(args.model)), processes=args.processes, repeats=args.repeats,
                   warmup_rounds=args.warmup_rounds, seed=args.seed, process_timeout=args.process_timeout,
                   profile_only=args.profile_only, measure_only=args.measure_only,
                   tensor_parallel_size=1, enforce_eager=True, max_model_len=1024,
                   max_num_seqs=4, gpu_memory_utilization=.65, initial_max_num_batched_tokens=256)
    sources = ("benchmarks/prefill_timeline.py", "benchmarks/prefill_profile.py", "benchmarks/prefill_cost_model.py",
               "nanovllm/engine/scheduler.py", "nanovllm/engine/model_runner.py", "nanovllm/engine/llm_engine.py")
    manifest = dict(options=options, cases=[case.to_dict() for case in cases],
                    split="even process IDs train, odd holdout; alternating in time; fixed before data collection",
                    timestamp_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                    source_sha256={name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in sources},
                    percentile_method="linear interpolation at (n-1)*p (type 7)")
    for name in sources[:3]:
        (directory / Path(name).name).write_bytes(Path(name).read_bytes())
    manifest_path = directory / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    jobs = ([] if args.profile_only else [("measurement", pid) for pid in range(args.processes)])
    if not args.measure_only:
        jobs.append(("profiler", args.processes))
    workers = []
    for kind, process_id in jobs:
        output = directory / f"{kind}_{process_id}.json"
        command = [sys.executable, "-m", "benchmarks.prefill_timeline", "--worker-manifest", str(manifest_path),
                   "--worker-kind", kind, "--worker-id", str(process_id), "--worker-output", str(output)]
        print(f"Starting isolated {kind} process {process_id}", flush=True)
        log_path = directory / f"{kind}_{process_id}.log"
        with log_path.open("w") as log:
            try:
                result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                        timeout=args.process_timeout, check=False)
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError(f"worker timed out; retained log: {log_path}") from exc
        if result.returncode or not output.exists():
            raise RuntimeError(f"worker failed ({result.returncode}); see {log_path}")
        worker = json.loads(output.read_text())
        if worker["status"] != "ok":
            raise RuntimeError(f"worker error: {worker['error']}; see {log_path}")
        workers.append(worker)
    payload = collect_results(directory, manifest, workers)
    print(f"Saved and verified {len(payload['measurements'])} samples, {len(payload['timelines'])} traces: {directory}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
