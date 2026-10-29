"""Small mixed-load benchmark. Run: python -m benchmarks.mixed_load_bench --help.

Instrumentation is confined to this script and restored after each run. Requests
are admitted only between synchronous engine steps, not by an async HTTP server.
"""

import argparse
import atexit
from contextlib import ExitStack
from dataclasses import dataclass
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import random
import subprocess
import sys
import time
import traceback

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.metrics import RequestMetrics, summarize_requests, write_results


POLICIES = {
    "prefill_first": ("prefill_first", 0),
    "decode_first": ("decode_first", 0),
    "decode_first_fair2": ("decode_first", 2),
    "prefill_first_rr": ("prefill_first", 0),
    "decode_first_fair2_rr": ("decode_first", 2),
}
DEFAULT_POLICIES = ("prefill_first", "decode_first", "decode_first_fair2")
ROUND_ROBIN_POLICIES = ("prefill_first_rr", "decode_first_fair2_rr")


@dataclass
class RequestSpec:
    request_id: int
    prompt_token_ids: list[int]
    max_tokens: int
    scheduled_arrival: float
    workload_class: str

    def to_dict(self):
        return {**vars(self), "prompt_tokens": len(self.prompt_token_ids)}


def build_workload(tokenizer, num_requests=12, seed=0, arrival_interval=0.02,
                   initial_burst=4, max_model_len=512):
    if num_requests < 4 or not 0 <= initial_burst <= num_requests:
        raise ValueError("need at least four requests and 0 <= initial_burst <= num_requests")
    if not math.isfinite(arrival_interval) or arrival_interval < 0:
        raise ValueError("arrival_interval must be finite and non-negative")
    rng = random.Random(seed)
    kinds = [("short", "short"), ("short", "long"),
             ("long", "short"), ("long", "long")]
    classes = [kinds[i % 4] for i in range(num_requests)]
    rng.shuffle(classes)
    snippets = [
        "Explain how a scheduler balances latency and throughput. ",
        "A paged cache stores attention keys and values in reusable blocks. ",
        "Compare the behavior of short and long requests under contention. ",
        "Describe admission, prefill, decoding, and request completion. ",
    ]
    requests = []
    for index, (prompt_kind, output_kind) in enumerate(classes):
        target = rng.randint(64, 128) if prompt_kind == "short" else rng.randint(256, 384)
        max_tokens = rng.randint(16, 32) if output_kind == "short" else rng.randint(48, 64)
        if target + max_tokens > max_model_len:
            raise ValueError(f"request {index}: prompt + output exceeds max_model_len={max_model_len}")
        text = f"Request {index}, experiment {rng.randrange(10**9)}. "
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        while len(token_ids) < target:
            text += rng.choice(snippets)
            token_ids = tokenizer.encode(text, add_special_tokens=False)
        # Text is tokenized with the actual model tokenizer, then truncated at
        # a token boundary. Engine admission uses these exact IDs, not estimates.
        token_ids = token_ids[:target]
        arrival = (0.0 if index < initial_burst else
                   (index - initial_burst + 1) * arrival_interval)
        requests.append(RequestSpec(index, token_ids, max_tokens, arrival,
                                    f"{prompt_kind}_prompt_{output_kind}_output"))
    return requests


class EngineObserver:
    """Observe real append_token calls; never infer token generation from a phase."""

    def __init__(self, engine, specs, clock=time.perf_counter):
        self.engine = engine
        self.clock = clock
        self.origin = None
        self.records = {spec.request_id: RequestMetrics(
            spec.request_id, len(spec.prompt_token_ids), spec.max_tokens,
            spec.scheduled_arrival, spec.workload_class) for spec in specs}
        self.sequences = {}
        self.prefill_steps = 0
        self.decode_steps = 0
        self.preemptions = 0
        self.steps = []
        self.active_step = None
        self.pending_request = None
        self.restorations = ExitStack()

    def now(self):
        if self.origin is None:
            raise RuntimeError("measurement clock has not started")
        return self.clock() - self.origin

    def replace(self, obj, name, replacement):
        # Restore the instance dictionary exactly; original methods normally
        # live on the class and must not be left shadowed on the instance.
        if name in vars(obj):
            self.restorations.callback(setattr, obj, name, vars(obj)[name])
        else:
            self.restorations.callback(delattr, obj, name)
        setattr(obj, name, replacement)

    def __enter__(self):
        scheduler = self.engine.scheduler
        original_add = scheduler.add
        original_schedule = scheduler.schedule
        original_preempt = scheduler.preempt
        original_postprocess = scheduler.postprocess
        original_step = self.engine.step

        def add(seq):
            if self.pending_request is None:
                raise RuntimeError("unexpected request outside benchmark admission")
            record = self.records[self.pending_request]
            if record.admission_time is not None:
                raise RuntimeError("request admitted more than once")
            result = original_add(seq)
            record.admission_time = self.now()  # Immediately after the real queue append.
            self.sequences[seq.seq_id] = record
            original_append = seq.append_token

            def append_token(token_id):
                result = original_append(token_id)
                record.add_token(self.now())  # Real append must have succeeded first.
                if seq.num_completion_tokens != len(record.token_times):
                    raise RuntimeError("output token count and observations diverged")
                if self.active_step is not None:
                    self.active_step["emitted_request_ids"].append(record.request_id)
                    self.active_step["output_token_indices"].append(len(record.token_times))
                return result

            self.replace(seq, "append_token", append_token)
            return result

        def schedule():
            seqs, is_prefill = original_schedule()
            if seqs:
                if is_prefill:
                    self.prefill_steps += 1
                else:
                    self.decode_steps += 1
                if self.active_step is not None:
                    if "phase" in self.active_step:
                        raise RuntimeError("more than one non-empty schedule per engine step")
                    self.active_step.update(
                        phase="prefill" if is_prefill else "decode", batch_size=len(seqs),
                        request_ids=[self.sequences[seq.seq_id].request_id for seq in seqs],
                        schedule_time=self.now(), waiting_count=len(scheduler.waiting),
                        running_count=len(scheduler.running))
            return seqs, is_prefill

        def preempt(seq):
            result = original_preempt(seq)
            self.preemptions += 1
            if self.active_step is not None:
                self.active_step["preemptions"] += 1
            return result

        def step():
            if self.active_step is not None:
                raise RuntimeError("nested engine steps are unsupported")
            trace = dict(step_index=len(self.steps), start_time=self.now(),
                         emitted_request_ids=[], output_token_indices=[], preemptions=0)
            self.active_step = trace
            try:
                result = original_step()
                trace["end_time"] = self.now()
                if "phase" not in trace:
                    raise RuntimeError("engine step did not schedule a non-empty batch")
                trace["duration_s"] = trace["end_time"] - trace["start_time"]
                trace["output_tokens"] = len(trace["emitted_request_ids"])
                if trace["phase"] == "decode" and trace["output_tokens"] != trace["batch_size"]:
                    raise RuntimeError("decode batch size and real appended token count differ")
                self.steps.append(trace)
                return result
            finally:
                self.active_step = None

        def postprocess(seqs, token_ids, is_prefill):
            result = original_postprocess(seqs, token_ids, is_prefill)
            for seq in seqs:
                if seq.is_finished:
                    self.sequences[seq.seq_id].complete(self.now())
            return result

        for name, method in (("add", add), ("schedule", schedule),
                             ("preempt", preempt), ("postprocess", postprocess)):
            self.replace(scheduler, name, method)
        self.replace(self.engine, "step", step)
        return self

    def __exit__(self, *exc):
        self.restorations.close()

    def admit(self, spec, sampling_params):
        self.pending_request = spec.request_id
        try:
            self.engine.add_request(spec.prompt_token_ids, sampling_params)
            if self.records[spec.request_id].admission_time is None:
                raise RuntimeError("add_request did not enqueue the request")
        finally:
            self.pending_request = None


def drive_requests(engine, observer, specs, sampling_factory, timeout_seconds=120,
                   max_steps=10000, sleep=time.sleep):
    """Include idle arrival gaps in elapsed time; setup/warmup/export are excluded."""
    if timeout_seconds <= 0 or not math.isfinite(timeout_seconds) or max_steps <= 0:
        raise ValueError("timeout_seconds and max_steps must be positive")
    pending = sorted(specs, key=lambda spec: (spec.scheduled_arrival, spec.request_id))
    # Construct SamplingParams before starting the measurement.
    params = {spec.request_id: sampling_factory(spec.max_tokens) for spec in specs}
    observer.origin = observer.clock()
    index = steps = 0
    while index < len(pending) or not engine.is_finished():
        if observer.now() >= timeout_seconds:
            raise TimeoutError(f"measurement exceeded {timeout_seconds} seconds after {steps} steps")
        while index < len(pending) and pending[index].scheduled_arrival <= observer.now():
            observer.admit(pending[index], params[pending[index].request_id])
            index += 1
        if not engine.is_finished():
            if steps >= max_steps:
                raise RuntimeError(f"measurement exceeded max_steps={max_steps}")
            engine.step()
            steps += 1
        elif index < len(pending):
            sleep(max(0, min(0.01, pending[index].scheduled_arrival - observer.now(),
                             timeout_seconds - observer.now())))
    elapsed = observer.now()
    if elapsed > timeout_seconds:
        raise TimeoutError("final synchronous step exceeded the measurement timeout")
    return elapsed


def warmup(engine, rounds):
    from nanovllm import SamplingParams

    prompts = []
    for index, length in enumerate((64, 128, 256, 384)):
        text = f"Warmup only, case {index}. " + "Cache scheduling warmup context. " * 256
        ids = engine.tokenizer.encode(text, add_special_tokens=False)[:length]
        if len(ids) != length:
            raise RuntimeError("warmup prompt is shorter than expected")
        prompts.append(ids)
    for _ in range(rounds):
        # Four equal short prompts fit one 256-token prefill batch. Unequal
        # output lengths exercise decode batch sizes 4, 3, 2, and 1.
        engine.generate([ids[:64] for ids in prompts],
                        [SamplingParams(temperature=0.6, max_tokens=13 + i, ignore_eos=True)
                         for i in range(len(prompts))], use_tqdm=False)
        engine.generate(prompts, SamplingParams(temperature=0.6, max_tokens=16,
                                               ignore_eos=True), use_tqdm=False)
    if not engine.is_finished() or engine.scheduler.block_manager.used_block_ids:
        raise RuntimeError("warmup did not release all requests and KV blocks")


def run_identity(policy, repeat):
    scheduling_policy, threshold = POLICIES[policy]
    return dict(policy=policy, scheduling_policy=scheduling_policy,
                max_consecutive_decode_steps=threshold, repeat=repeat,
                decode_queue_policy="round_robin" if policy in ROUND_ROBIN_POLICIES else "head_first")


def load_specs(manifest):
    return [RequestSpec(**{key: value for key, value in row.items() if key != "prompt_tokens"})
            for row in manifest["requests"]]


def failed_run(specs, policy, repeat, error):
    records = [RequestMetrics(spec.request_id, len(spec.prompt_token_ids), spec.max_tokens,
                              spec.scheduled_arrival, spec.workload_class) for spec in specs]
    return {"summary": {**run_identity(policy, repeat), "status": "failed", "error": error,
                        "gpu_name": None, "num_kvcache_blocks": None,
                        "prefill_steps": 0, "decode_steps": 0, "preemptions": 0,
                        **summarize_requests(records, None)},
            "requests": [record.to_dict() for record in records]}


def gpu_snapshot():
    """Read only outside the measured window; missing telemetry is explicit metadata."""
    fields = ["temperature.gpu", "clocks.current.sm", "clocks.current.memory",
              "power.draw", "utilization.gpu"]
    command = ["nvidia-smi", "--id=0", "--query-gpu=" + ",".join(fields),
               "--format=csv,noheader,nounits"]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=10, check=True)
        values = result.stdout.strip().split(",")
        if len(values) != len(fields):
            raise ValueError("unexpected nvidia-smi column count")
        return {"perf_counter_absolute_s": time.perf_counter(),
                **{key: value.strip() for key, value in zip(fields, values)}}
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def run_worker(manifest_path, output_path, policy, repeat):
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    specs = load_specs(manifest)
    options = manifest["config"]
    observer = None
    elapsed = None
    gpu_name = num_blocks = None
    compile_frames = None
    environment = {}
    resource_check = None
    error = None
    try:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable in this process; no GPU measurements were collected")
        from nanovllm import LLM, SamplingParams
        from torch._dynamo.utils import counters

        seed = options["seed"] + repeat
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        gpu_name = torch.cuda.get_device_name(0)
        environment = {"gpu_name": gpu_name, "gpu_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
                       "torch": torch.__version__, "cuda": torch.version.cuda,
                       "python": sys.version, "platform": platform.platform()}
        clock_info = time.get_clock_info("perf_counter")
        environment["measurement_clock"] = dict(implementation=clock_info.implementation,
                                                monotonic=clock_info.monotonic,
                                                adjustable=clock_info.adjustable,
                                                resolution_s=clock_info.resolution)
        for package in ("transformers", "triton", "flash-attn"):
            environment[package] = importlib.metadata.version(package)
        scheduling_policy, threshold = POLICIES[policy]
        engine = LLM(options["model"], scheduling_policy=scheduling_policy,
                     decode_queue_policy=run_identity(policy, repeat)["decode_queue_policy"],
                     max_consecutive_decode_steps=threshold, tensor_parallel_size=1,
                     enforce_eager=True, max_model_len=options["max_model_len"],
                     max_num_batched_tokens=options["max_num_batched_tokens"],
                     max_num_seqs=options["max_num_seqs"],
                     gpu_memory_utilization=options["gpu_memory_utilization"])

        def close_engine():
            atexit.unregister(engine.exit)
            engine.exit()

        with ExitStack() as cleanup:
            cleanup.callback(close_engine)
            num_blocks = len(engine.scheduler.block_manager.blocks)
            print("Model initialization complete; starting untimed warmup", flush=True)
            warmup(engine, options["warmup_rounds"])
            torch.cuda.synchronize()
            random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            frames_before = counters["frames"]["total"]
            environment["gpu_before_measurement"] = gpu_snapshot()
            print("Warmup complete; starting measured event loop", flush=True)
            observer = EngineObserver(engine, specs)
            with observer:
                try:
                    elapsed = drive_requests(
                        engine, observer, specs,
                        lambda count: SamplingParams(temperature=0.6, max_tokens=count, ignore_eos=True),
                        options["timeout_seconds"], options["max_steps"])
                finally:
                    if observer.origin is not None and elapsed is None:
                        elapsed = observer.now()
            environment["gpu_after_measurement"] = gpu_snapshot()
            compile_frames = counters["frames"]["total"] - frames_before
            print(f"Measurement complete; torch.compile frames in window: {compile_frames}", flush=True)
            if compile_frames:
                raise RuntimeError("torch.compile ran during measurement; increase warmup coverage and rerun")
            for spec in specs:
                record = observer.records[spec.request_id]
                if record.completion_time is None or len(record.token_times) != spec.max_tokens:
                    raise RuntimeError(f"request {spec.request_id} did not complete at the requested length")
            manager = engine.scheduler.block_manager
            resource_check = dict(waiting_requests=len(engine.scheduler.waiting),
                                  running_requests=len(engine.scheduler.running),
                                  total_blocks=len(manager.blocks), free_blocks=len(manager.free_block_ids),
                                  used_blocks=len(manager.used_block_ids),
                                  block_ref_count_sum=sum(block.ref_count for block in manager.blocks))
            if (not engine.is_finished() or manager.used_block_ids
                    or len(manager.free_block_ids) != len(manager.blocks)
                    or any(block.ref_count for block in manager.blocks)):
                raise RuntimeError("measured run leaked requests or KV blocks")
            resource_check["released"] = True
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
    run = failed_run(specs, policy, repeat, error)
    run["environment"] = environment
    run["resource_check"] = resource_check
    run["summary"].update(status="failed" if error else "ok", error=error,
                           gpu_name=gpu_name, num_kvcache_blocks=num_blocks,
                           torch_compile_frames_during_measurement=compile_frames)
    if observer is not None:
        run["steps"] = observer.steps
        records = list(observer.records.values())
        run["requests"] = [record.to_dict() for record in records]
        run["summary"].update(summarize_requests(records, elapsed),
                               prefill_steps=observer.prefill_steps, decode_steps=observer.decode_steps,
                               preemptions=observer.preemptions)
    Path(output_path).write_text(json.dumps(run, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                                 encoding="utf-8")
    return 1 if error else 0


def find_model(provided):
    if provided:
        path = Path(provided).expanduser().resolve()
    else:
        snapshots = Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots"
        candidates = sorted(snapshots.glob("*/config.json"))
        local = Path.home() / "huggingface/Qwen3-0.6B/config.json"
        if local.is_file():
            candidates.append(local)
        if len(candidates) != 1:
            raise ValueError("pass --model with an explicit local model directory")
        path = candidates[0].parent.resolve()
    if not (path / "config.json").is_file():
        raise ValueError(f"local model config not found: {path}")
    return path


def comparison_table(runs):
    columns = [("Policy", "policy", 1), ("Repeat", "repeat", 1), ("Status", "status", 1),
               ("Done", "completed_requests", 1), ("TTFT mean ms", "ttft_mean_s", 1000),
               ("TTFT P50 ms", "ttft_p50_s", 1000), ("TTFT P95 ms", "ttft_p95_s", 1000),
               ("ITL mean ms", "itl_mean_s", 1000), ("ITL P95 ms", "itl_p95_s", 1000),
               ("Latency P50 ms", "request_latency_p50_s", 1000),
               ("Latency P95 ms", "request_latency_p95_s", 1000),
               ("Admission P95 ms", "admission_delay_p95_s", 1000),
               ("Output tok/s", "output_token_throughput", 1),
               ("Prefill", "prefill_steps", 1), ("Decode", "decode_steps", 1),
               ("Preempt", "preemptions", 1)]
    lines = ["| " + " | ".join(title for title, _, _ in columns) + " |",
             "| " + " | ".join("---" for _ in columns) + " |"]
    for run in runs:
        summary = run["summary"]
        cells = []
        for _, key, scale in columns:
            value = summary[key]
            if summary["status"] != "ok" and (scale == 1000 or key == "output_token_throughput"):
                value = None  # Failed/partial runs are never presented as performance comparisons.
            cells.append("—" if value is None else
                         f"{value * scale:.3f}" if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def positive_int(value):
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def execution_plan(policies, repeats, order_mode):
    plan = []
    for repeat in range(repeats):
        offset = repeat % len(policies) if order_mode == "rotate" else 0
        ordered = policies[offset:] + policies[:offset]
        base_index = len(plan)
        plan.extend(dict(policy=policy, repeat=repeat, execution_index=base_index + index)
                    for index, policy in enumerate(ordered))
    return plan


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", help="local model path; auto-detect one Qwen3-0.6B snapshot if omitted")
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--num-requests", type=positive_int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--arrival-interval", type=float, default=0.02, help="seconds between arrivals after initial burst")
    parser.add_argument("--initial-burst", type=int, default=4)
    parser.add_argument("--policies", nargs="+", choices=list(POLICIES), default=list(DEFAULT_POLICIES))
    parser.add_argument("--repeats", type=positive_int, default=1)
    parser.add_argument("--order-mode", choices=("fixed", "rotate"), default="fixed")
    parser.add_argument("--workload-from", type=Path,
                        help="reuse exact requests from a saved workload; engine config must match")
    parser.add_argument("--warmup-rounds", type=positive_int, default=1)
    parser.add_argument("--max-model-len", type=positive_int, default=512)
    parser.add_argument("--max-num-batched-tokens", type=positive_int, default=256)
    parser.add_argument("--max-num-seqs", type=positive_int, default=4)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.65)
    parser.add_argument("--timeout-seconds", type=float, default=120,
                        help="measured run timeout, checked between synchronous steps")
    parser.add_argument("--process-timeout", type=float, default=300,
                        help="hard wall timeout per worker, including startup and warmup")
    parser.add_argument("--max-steps", type=positive_int, default=10000)
    parser.add_argument("--verify-only", type=Path, help="verify existing CSV/JSON without importing GPU libraries")
    parser.add_argument("--worker-manifest", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker-policy", choices=list(POLICIES), help=argparse.SUPPRESS)
    parser.add_argument("--repeat-index", type=int, default=0, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    for name in ("timeout_seconds", "process_timeout"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive and finite")
    if not 0 < args.gpu_memory_utilization < 1:
        parser.error("--gpu-memory-utilization must be between 0 and 1")
    if len(set(args.policies)) != len(args.policies):
        parser.error("duplicate policies are not allowed")
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.verify_only:
        from benchmarks.metrics import verify_results
        verify_results(args.verify_only)
        print("CSV/JSON consistency verified")
        return 0
    if args.worker_manifest:
        return run_worker(args.worker_manifest, args.worker_output, args.worker_policy, args.repeat_index)
    source = (json.loads(args.workload_from.read_text(encoding="utf-8"))
              if args.workload_from else None)
    model = find_model(args.model or (source["config"]["model"] if source else None))
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory must be empty to avoid mixing runs: {output_dir}")
    config = {key: getattr(args, key) for key in (
        "seed", "warmup_rounds", "max_model_len", "max_num_batched_tokens", "max_num_seqs",
        "gpu_memory_utilization", "timeout_seconds", "max_steps")}
    config.update(model=str(model), tensor_parallel_size=1, enforce_eager=True,
                  temperature=0.6, ignore_eos=True)
    if source:
        if source["config"] != config:
            raise ValueError("--workload-from requires engine/seed/warmup configuration matching the source")
        specs = load_specs(source)
        if (len(specs) != args.num_requests or args.arrival_interval != source["arrival_interval"]
                or args.initial_burst != source["initial_burst"]):
            raise ValueError("request count and arrival settings must match --workload-from")
    else:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(model), local_files_only=True)
        specs = build_workload(tokenizer, args.num_requests, args.seed, args.arrival_interval,
                               args.initial_burst, args.max_model_len)
    workload = [spec.to_dict() for spec in specs]
    workload_hash = hashlib.sha256(json.dumps(workload, sort_keys=True).encode()).hexdigest()
    if source and workload_hash != source["workload_sha256"]:
        raise ValueError("source workload SHA-256 mismatch")
    plan = execution_plan(args.policies, args.repeats, args.order_mode)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    metadata = dict(config=config, seed=args.seed, workload_sha256=workload_hash,
                    git_commit=commit, num_requests=len(specs), arrival_interval=args.arrival_interval,
                    initial_burst=args.initial_burst, repeats=args.repeats, process_timeout=args.process_timeout,
                    timestamp_unit="seconds relative to perf_counter measurement origin",
                    throughput_window="after warmup to final completed step, including idle arrival gaps",
                    itl_weighting="pooled adjacent-token intervals across requests",
                    admission_model="synchronous single-threaded; arrivals injected only between step() calls",
                    token_observation="host timestamp immediately after real Sequence.append_token returns",
                    compilation_check="reject runs if torch._dynamo frames.total increases during measurement",
                    sampling_seed="seed + repeat index; batching can change sampled output tokens")
    metadata.update(order_mode=args.order_mode, execution_order=plan,
                    workload_source=str(args.workload_from.resolve()) if source else None,
                    step_duration="host perf_counter around real engine.step; includes schedule, GPU result transfer and postprocess",
                    telemetry="nvidia-smi snapshots outside each measured window, not a continuous clock trace",
                    benchmark_source_sha256={path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                                             for path in Path(__file__).resolve().parent.glob("*.py")})
    manifest = {**metadata, "requests": workload}
    output_dir.mkdir(parents=True, exist_ok=True)
    run_dir = output_dir / "runs"
    run_dir.mkdir()
    manifest_path = output_dir / "workload.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    runs = []
    environments = []
    for item in plan:
        policy, repeat = item["policy"], item["repeat"]
        name = f"{policy}_repeat{repeat}"
        run_path = run_dir / f"{name}.json"
        log_path = run_dir / f"{name}.log"
        command = [sys.executable, "-m", "benchmarks.mixed_load_bench", "--worker-manifest",
                   str(manifest_path), "--worker-output", str(run_path), "--worker-policy", policy,
                   "--repeat-index", str(repeat)]
        print(f"Running {name} in a fresh process; log: {log_path}", flush=True)
        failure = None
        with log_path.open("w", encoding="utf-8") as log:
            try:
                completed = subprocess.run(command, cwd=Path(__file__).resolve().parents[1],
                                           stdout=log, stderr=subprocess.STDOUT,
                                           timeout=args.process_timeout, check=False)
                if completed.returncode:
                    failure = f"worker exited {completed.returncode}; see {log_path}"
            except subprocess.TimeoutExpired:
                failure = f"worker exceeded hard timeout {args.process_timeout}s; see {log_path}"
        if run_path.exists():
            run = json.loads(run_path.read_text(encoding="utf-8"))
            if failure and run["summary"]["status"] == "ok":
                run["summary"].update(status="failed", error=failure)
        else:
            run = failed_run(specs, policy, repeat, failure or "worker did not produce an output file")
            run_path.write_text(json.dumps(run, indent=2) + "\n", encoding="utf-8")
        if run["summary"]["status"] != "ok":
            print(f"FAILED {name}: {run['summary']['error']}", file=sys.stderr, flush=True)
        runs.append(run)
        environments.append({**run_identity(policy, repeat), **run.get("environment", {})})
        # Keep completed runs and failure diagnostics even if a later worker fails.
        write_results(output_dir, runs, {**metadata, "environments": environments})
        from benchmarks.reliability import write_step_csv
        write_step_csv(output_dir / "raw_steps.csv", runs)
        single_dir = run_dir / name
        write_results(single_dir, [run], {**metadata, "execution": item,
                                        "environment": run.get("environment", {}),
                                        "resource_check": run.get("resource_check")})
        write_step_csv(single_dir / "raw_steps.csv", [run])
    table = comparison_table(runs)
    (output_dir / "comparison.md").write_text(table, encoding="utf-8")
    print(table)
    print(f"Results and raw-timestamp consistency check: {output_dir}")
    return 0 if all(run["summary"]["status"] == "ok" for run in runs) else 1


if __name__ == "__main__":
    raise SystemExit(main())
