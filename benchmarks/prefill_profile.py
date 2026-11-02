"""Controlled Prefill microbenchmark; never changes the scheduling algorithm.

q is total executed query tokens in a round. c is the cached prefix length of
each request in a homogeneous batch; c_total and per-request lengths are saved.
CUDA Events bracket run_model (forward + logits), not a sum of active kernels.
"""

import argparse
import atexit
from collections import Counter, defaultdict
from contextlib import ExitStack
from dataclasses import asdict, dataclass
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import random
from statistics import fmean
import subprocess
import sys
import time
import traceback

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.metrics import PERCENTILE_METHOD, percentile
from benchmarks.mixed_load_bench import find_model, gpu_snapshot
from benchmarks.reliability import write_csv


@dataclass(frozen=True)
class ProfileCase:
    q: int
    c: int
    batch_size: int

    @property
    def case_id(self):
        return f"b{self.batch_size}_q{self.q}_c{self.c}"

    @property
    def q_per_request(self):
        return self.q // self.batch_size

    def to_dict(self):
        return dict(case_id=self.case_id, **asdict(self), q_per_request=self.q_per_request,
                    c_total=self.c * self.batch_size,
                    prompt_tokens_per_request=self.c + self.q_per_request)


def build_cases(q_values, c_values, batch_sizes, max_model_len):
    for name, values, minimum in (("q", q_values, 1), ("c", c_values, 0),
                                   ("batch_size", batch_sizes, 1)):
        if not values or len(set(values)) != len(values):
            raise ValueError(f"{name} values must be non-empty and unique")
        if any(type(value) is not int or value < minimum for value in values):
            raise ValueError(f"invalid {name} values")
    if type(max_model_len) is not int or max_model_len <= 0:
        raise ValueError("max_model_len must be a positive integer")
    cases, skipped = [], []
    for batch_size in sorted(batch_sizes):
        for q in q_values:
            for c in c_values:
                reason = None
                if q % batch_size:
                    reason = "q must divide equally across the homogeneous batch"
                elif c + q // batch_size + 1 > max_model_len:
                    reason = "prompt + one output token exceeds max_model_len"
                if reason:
                    skipped.append(dict(q=q, c=c, batch_size=batch_size, reason=reason))
                else:
                    cases.append(ProfileCase(q, c, batch_size))
    return cases, skipped


def prompt_ids(tokenizer, case, seed):
    prompts = []
    for index in range(case.batch_size):
        text = f"Profile seed {seed}, request {index}. " + (
            "Explain attention, token scheduling and cached context. " * 256)
        ids = tokenizer.encode(text, add_special_tokens=False)[:case.c + case.q_per_request]
        if len(ids) != case.c + case.q_per_request:
            raise ValueError("tokenized profiling prompt is too short")
        prompts.append(ids)
    return prompts


def cache_snapshot(scheduler):
    manager = scheduler.block_manager
    return dict(total_blocks=len(manager.blocks), free_blocks=len(manager.free_block_ids),
                used_blocks=len(manager.used_block_ids),
                ref_count_sum=sum(block.ref_count for block in manager.blocks))


def require_released(engine):
    state = cache_snapshot(engine.scheduler)
    if (not engine.is_finished() or state["used_blocks"] or state["ref_count_sum"]
            or state["free_blocks"] != state["total_blocks"]):
        raise RuntimeError(f"live requests or leaked KV blocks: {state}")
    return state


def prepare_case(engine, case, prompts, sampling_params):
    """Untimed: allocate fresh real blocks, compute real prefixes, then schedule normally.

    No prefix-hit shortcuts: c=64/128 must work inside a 256-token block. The
    prefix forward uses the unchanged runner and postprocess; incomplete chunks
    produce no Sequence output tokens. Allocation is excluded for all c values.
    """
    require_released(engine)
    scheduler = engine.scheduler
    scheduler.max_num_batched_tokens = case.q
    manager = scheduler.block_manager
    seqs = []
    for ids in prompts:
        engine.add_request(ids, sampling_params)
        seq = scheduler.waiting[-1]
        if len(manager.free_block_ids) < seq.num_blocks:
            raise MemoryError(f"insufficient KV blocks for {case.case_id}")
        manager.allocate(seq, 0)
        seqs.append(seq)
    for seq in seqs:
        if case.c:
            seq.num_scheduled_tokens = case.c
            tokens = engine.model_runner.call("run", [seq], True)
            scheduler.postprocess([seq], tokens, True)
        if seq.num_cached_tokens != case.c or seq.num_completion_tokens != 0:
            raise RuntimeError("prefix computation did not leave the requested real cached state")
    if list(scheduler.waiting) != seqs or scheduler.running:
        raise RuntimeError("prefix preparation unexpectedly changed request queue states")
    return seqs


class TimingProbe:
    """Temporary instance wrappers; event creation/initialization is untimed."""

    def __init__(self, engine, events=None):
        self.engine = engine
        self.events = events
        self.capture = None
        self.restore = ExitStack()

    def replace(self, obj, name, method):
        if name in vars(obj):
            self.restore.callback(setattr, obj, name, vars(obj)[name])
        else:
            self.restore.callback(delattr, obj, name)
        setattr(obj, name, method)

    def __enter__(self):
        runner, scheduler = self.engine.model_runner, self.engine.scheduler
        original_schedule = scheduler.schedule
        original_run, original_prepare, original_model = runner.run, runner.prepare_prefill, runner.run_model

        def schedule():
            seqs, is_prefill = original_schedule()
            if self.capture is not None:
                if not is_prefill or not seqs or "q_each" in self.capture:
                    raise RuntimeError("expected exactly one non-empty Prefill schedule")
                self.capture.update(q_each=[seq.num_scheduled_tokens for seq in seqs],
                                    c_each=[seq.num_cached_tokens for seq in seqs],
                                    block_tables=[list(seq.block_table) for seq in seqs],
                                    scheduled_cache=cache_snapshot(scheduler))
            return seqs, is_prefill

        def run(seqs, is_prefill):
            if self.capture is None:
                return original_run(seqs, is_prefill)
            started = time.perf_counter()
            result = original_run(seqs, is_prefill)
            self.capture["runner_wall_ms"] = (time.perf_counter() - started) * 1000
            return result

        def prepare(seqs):
            if self.capture is None:
                return original_prepare(seqs)
            started = time.perf_counter()
            result = original_prepare(seqs)
            self.capture["prepare_wall_ms"] = (time.perf_counter() - started) * 1000
            return result

        def model(input_ids, positions, is_prefill):
            if self.capture is None:
                return original_model(input_ids, positions, is_prefill)
            if not is_prefill or "model_submit_wall_ms" in self.capture:
                raise RuntimeError("expected one Prefill model forward")
            started = time.perf_counter()
            if self.events:
                self.events[0].record()
            result = original_model(input_ids, positions, is_prefill)
            if self.events:
                self.events[1].record()
            self.capture["model_submit_wall_ms"] = (time.perf_counter() - started) * 1000
            return result

        self.replace(scheduler, "schedule", schedule)
        self.replace(runner, "run", run)
        self.replace(runner, "prepare_prefill", prepare)
        self.replace(runner, "run_model", model)
        return self

    def __exit__(self, *exc):
        self.restore.close()


def sample_case(engine, case, prompts, params, probe, torch, counters, phase, repeat, order):
    seqs = prepare_case(engine, case, prompts, params)
    before = cache_snapshot(engine.scheduler)
    torch.cuda.synchronize()
    frames = counters["frames"]["total"]
    probe.capture = {}
    try:
        started = time.perf_counter()
        outputs, scheduled_tokens = engine.step()
        stopped = time.perf_counter()
    finally:
        captured = probe.capture
        probe.capture = None
    compilation = counters["frames"]["total"] - frames
    if phase == "measure" and compilation:
        raise RuntimeError(f"compilation during measured sample {case.case_id}; increase warmup")
    if (captured.get("q_each") != [case.q_per_request] * case.batch_size
            or captured.get("c_each") != [case.c] * case.batch_size or scheduled_tokens != case.q):
        raise RuntimeError(f"actual scheduled q/c/batch differs from target {case.case_id}: {captured}")
    if len(outputs) != case.batch_size or any(len(ids) != 1 for _, ids in outputs):
        raise RuntimeError("target Prefill must finish every request with exactly one real output token")
    after = require_released(engine)
    event_ms = None
    if probe.events:
        # run() already copied sampled tokens to CPU. The model's end marker
        # precedes sampling on this stream and must have completed by now.
        if not probe.events[1].query():
            raise RuntimeError("CUDA end event is not complete after CPU outputs became available")
        event_ms = probe.events[0].elapsed_time(probe.events[1])
        if not math.isfinite(event_ms) or event_ms <= 0:
            raise RuntimeError("invalid CUDA Event interval")
    return dict(**case.to_dict(), phase=phase, repeat=repeat, case_order=order,
                max_num_batched_tokens=engine.scheduler.max_num_batched_tokens,
                wall_start_perf_counter=started, wall_end_perf_counter=stopped,
                step_wall_ms=(stopped - started) * 1000,
                runner_wall_ms=captured["runner_wall_ms"], prepare_wall_ms=captured["prepare_wall_ms"],
                model_submit_wall_ms=captured["model_submit_wall_ms"], cuda_model_stream_ms=event_ms,
                q_each=captured["q_each"], c_each=captured["c_each"],
                block_tables=captured["block_tables"], block_size=engine.scheduler.block_size,
                kv_total_blocks=before["total_blocks"], kv_used_before=before["used_blocks"],
                kv_used_after_schedule=captured["scheduled_cache"]["used_blocks"],
                kv_free_after=after["free_blocks"], kv_used_after=after["used_blocks"],
                kv_ref_count_after=after["ref_count_sum"], output_tokens=len(outputs),
                compile_frames=compilation, attention_kv_path="paged_kv" if case.c else "direct_qkv")


TIME_FIELDS = ("step_wall_ms", "runner_wall_ms", "prepare_wall_ms", "model_submit_wall_ms",
               "cuda_model_stream_ms")


def summarize_samples(samples):
    groups = defaultdict(list)
    for row in samples:
        if row["phase"] == "measure":
            groups[row["case_id"]].append(row)
    summaries = []
    for case_id, rows in groups.items():
        first = rows[0]
        summary = {key: first[key] for key in ("case_id", "q", "c", "batch_size", "q_per_request",
                                               "c_total", "max_num_batched_tokens", "attention_kv_path",
                                               "block_size", "kv_total_blocks", "kv_used_before")}
        summary["samples"] = len(rows)
        for name in TIME_FIELDS:
            values = [row[name] for row in rows if row[name] is not None]
            summary.update({f"{name}_samples": len(values),
                            f"{name}_mean": fmean(values) if values else None,
                            f"{name}_p50": percentile(values, .5),
                            f"{name}_p95": percentile(values, .95)})
        summaries.append(summary)
    return sorted(summaries, key=lambda row: (row["batch_size"], row["c"], row["q"]))


def validate_samples(samples, cases, repeats, warmup_rounds, cuda_events):
    expected = {case.case_id: case for case in cases}
    counts = Counter()
    seen = set()
    for row in samples:
        case = expected[row["case_id"]]
        for name, value in case.to_dict().items():
            if row[name] != value:
                raise ValueError(f"case metadata differs: {case.case_id}, {name}")
        identity = (case.case_id, row["phase"], row["repeat"])
        limit = repeats if row["phase"] == "measure" else warmup_rounds
        if (row["phase"] not in ("measure", "warmup") or identity in seen
                or not 0 <= row["repeat"] < limit):
            raise ValueError("duplicate, unknown or out-of-range trial")
        seen.add(identity)
        counts[(case.case_id, row["phase"])] += 1
        if (row["q_each"] != [case.q_per_request] * case.batch_size
                or row["c_each"] != [case.c] * case.batch_size
                or sum(row["q_each"]) != row["q"] or row["max_num_batched_tokens"] != row["q"]):
            raise ValueError("actual scheduled q/c/batch differs from case")
        if (row["output_tokens"] != case.batch_size or row["kv_used_after"]
                or row["kv_ref_count_after"] or row["kv_free_after"] != row["kv_total_blocks"]):
            raise ValueError("output count or final KV state differs")
        blocks = case.batch_size * ((case.c + case.q_per_request + row["block_size"] - 1) // row["block_size"])
        tables = row["block_tables"]
        if (row["kv_used_before"] != blocks or row["kv_used_after_schedule"] != blocks
                or len(tables) != case.batch_size or sum(map(len, tables)) != blocks
                or len({value for table in tables for value in table}) != blocks):
            raise ValueError("KV block allocation differs from the actual prompts")
        if row["phase"] == "measure" and row["compile_frames"]:
            raise ValueError("compiled frames in a measured sample")
        if not math.isclose(row["step_wall_ms"],
                            (row["wall_end_perf_counter"] - row["wall_start_perf_counter"]) * 1000,
                            rel_tol=1e-10, abs_tol=1e-9):
            raise ValueError("wall duration differs from perf_counter timestamps")
        for name in TIME_FIELDS:
            value = row[name]
            if name == "cuda_model_stream_ms" and not cuda_events:
                if value is not None:
                    raise ValueError("unexpected CUDA Event timing when disabled")
            elif value is None or not math.isfinite(value) or value <= 0:
                raise ValueError(f"invalid timing {name}")
    for case in cases:
        if counts[(case.case_id, "measure")] != repeats or counts[(case.case_id, "warmup")] != warmup_rounds:
            raise ValueError(f"incomplete sampling: {case.case_id}")
    return True


def write_results(output_dir, runs, metadata, cases):
    output_dir = Path(output_dir)
    samples = [row for run in runs for row in run["samples"]]
    summaries = summarize_samples(samples)
    payload = dict(metadata=metadata, workers=[{key: value for key, value in run.items()
                                               if key != "samples"} for run in runs],
                   cases=[case.to_dict() for case in cases], samples=samples, summary=summaries)
    (output_dir / "profile.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2,
                                                        allow_nan=False) + "\n", encoding="utf-8")
    write_csv(output_dir / "raw_prefill.csv", samples)
    write_csv(output_dir / "summary.csv", summaries)
    return payload


def verify_results(output_dir):
    import csv
    output_dir = Path(output_dir)
    payload = json.loads((output_dir / "profile.json").read_text(encoding="utf-8"))
    cases = [ProfileCase(row["q"], row["c"], row["batch_size"]) for row in payload["cases"]]
    options = payload["metadata"]["options"]
    if any(run["status"] != "ok" for run in payload["workers"]):
        raise ValueError("a worker failed; partial results are not a valid complete profile")
    validate_samples(payload["samples"], cases, options["repeats"], options["warmup_rounds"],
                     options["cuda_events"])
    if payload["summary"] != summarize_samples(payload["samples"]):
        raise ValueError("summary JSON differs from raw samples")
    for name, rows in (("raw_prefill.csv", payload["samples"]), ("summary.csv", payload["summary"])):
        with (output_dir / name).open(encoding="utf-8", newline="") as file:
            actual = list(csv.DictReader(file))
        expected = [{key: "" if value is None else json.dumps(value) if isinstance(value, list)
                     else str(value) for key, value in row.items()} for row in rows]
        if actual != expected:
            raise ValueError(f"CSV and JSON differ: {name}")
    return payload


def run_worker(manifest_path, batch_size, output_path):
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    options = manifest["options"]
    cases = [ProfileCase(row["q"], row["c"], row["batch_size"]) for row in manifest["cases"]
             if row["batch_size"] == batch_size]
    random.Random(options["seed"] + batch_size).shuffle(cases)
    samples, environment, prompts_by_case, telemetry = [], {}, {}, []
    error = None
    try:
        import torch
        from torch._dynamo.utils import counters
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; no substitute CPU performance will be generated")
        from nanovllm import LLM, SamplingParams

        torch.manual_seed(options["seed"])
        torch.cuda.manual_seed_all(options["seed"])
        engine = LLM(options["model"], tensor_parallel_size=1, enforce_eager=True,
                     max_model_len=options["max_model_len"], max_num_batched_tokens=256,
                     max_num_seqs=max(options["batch_sizes"]), gpu_memory_utilization=0.65,
                     scheduling_policy="prefill_first", decode_queue_policy="head_first")
        def close_engine():
            atexit.unregister(engine.exit)
            engine.exit()
        with ExitStack() as cleanup:
            cleanup.callback(close_engine)
            if engine.model_runner.config.max_model_len < options["max_model_len"]:
                raise ValueError("model context limit is smaller than the requested profiling limit")
            properties = torch.cuda.get_device_properties(0)
            info = time.get_clock_info("perf_counter")
            environment = dict(gpu_name=properties.name, gpu_memory_bytes=properties.total_memory,
                               torch=torch.__version__, cuda=torch.version.cuda, python=sys.version,
                               dtype=str(engine.model_runner.config.hf_config.dtype),
                               total_kv_blocks=len(engine.scheduler.block_manager.blocks),
                               clock=dict(implementation=info.implementation, monotonic=info.monotonic,
                                          resolution_s=info.resolution))
            for package in ("transformers", "triton", "flash-attn"):
                environment[package] = importlib.metadata.version(package)
            events = None
            if options["cuda_events"]:
                events = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
                events[0].record()
                events[1].record()
                events[1].synchronize()
            params = SamplingParams(max_tokens=1, temperature=0.6, ignore_eos=True)
            with TimingProbe(engine, events) as probe:
                for order, case in enumerate(cases):
                    if case.c + case.q_per_request + 1 > options["max_model_len"]:
                        raise ValueError(f"invalid model length for {case.case_id}")
                    prompts = prompt_ids(engine.tokenizer, case, options["seed"])
                    prompts_by_case[case.case_id] = prompts
                    before = gpu_snapshot()
                    print(f"{case.case_id}: {options['warmup_rounds']} warmups + {options['repeats']} samples", flush=True)
                    for phase, count in (("warmup", options["warmup_rounds"]), ("measure", options["repeats"])):
                        for repeat in range(count):
                            samples.append(sample_case(engine, case, prompts, params, probe, torch,
                                                       counters, phase, repeat, order))
                    telemetry.append(dict(case_id=case.case_id, before=before, after=gpu_snapshot()))
            validate_samples(samples, cases, options["repeats"], options["warmup_rounds"], options["cuda_events"])
            require_released(engine)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
    payload = dict(status="failed" if error else "ok", error=error, batch_size=batch_size,
                   environment=environment, case_order=[case.case_id for case in cases],
                   prompts_by_case=prompts_by_case, telemetry=telemetry, samples=samples)
    Path(output_path).write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                                 encoding="utf-8")
    return 1 if error else 0


def render_report(payload, output_dir):
    summaries = payload["summary"]
    options = payload["metadata"]["options"]
    measured = [row for row in payload["samples"] if row["phase"] == "measure"]
    lines = ["# 第五阶段第一步：Prefill 计算成本 Profiling", "",
             "## 范围与源码路径", "",
             f"基础稳定提交为 `{payload['metadata']['git_commit']}`。本阶段不修改 Scheduler、ModelRunner 或 CUDA Kernel，不新增 GPU 依赖，不 Commit/Push。", "",
             "Scheduler 根据预算设置 num_scheduled_tokens，并使用 num_cached_tokens 计算本轮剩余量。LLMEngine.step() 调度后调用 ModelRunner.run()，返回 CPU Token 列表后调用 postprocess()。prepare_prefill() 从 c 位置取 q_i 个 Token，设置 seqlen_q=q_i、seqlen_k=c+q_i；c>0 使用 KV block table。run_model() 包含完整模型前向与 logits，采样和 .tolist() 在它之后。mixed_load_bench.py 的事件循环和逐 Token 插桩适合服务指标，本工具独立控制微基准输入，不重放在线队列负载。", "",
             "## q、c、批大小和 KV 的定义", "",
             "q 是整轮真实调度的查询 Token 总数，q_i=q/batch_size；本实验采用同质批次，c 是每请求已计算的前缀长度，c_total=batch_size×c。每条原始记录保存 q_each/c_each，不能把 batch_size=4、q=64 误读为每请求64个新Token。", "",
             f"max_model_len={options['max_model_len']}，TP=1、enforce_eager=True、GPU显存比例0.65、max_num_seqs={max(options['batch_sizes'])}。引擎初始批Token预算256，目标轮的真实 Scheduler 预算设为 q 并逐次核对。为覆盖 c=256、q=256、batch=1，且保留一个输出Token，模型长度上限使用1024而不是第三阶段的512。", "",
             "所有 c 值都在计时外用真实 BlockManager.allocate(seq, 0) 预分配完整 Prompt 所需块，禁用 Prefix Cache 命中捷径。c>0 时在计时外真正运行前缀前向，并调用原 postprocess；此时 chunk 未完成、没有追加输出Token。随后原 Scheduler.schedule() 选择完整目标批次，每请求剩余q_i个Token，真实模型执行后生成一个Token并结束，验证全部KV释放。因而 c=64/128 对应真实有效KV，不是伪造缓存计数。", "",
             "这是计算成本微基准：KV分配和前缀预计算不计入目标耗时，不等同于完整在线请求Prefill延迟。c=0为直接QKV路径，c>0为分页KV路径，跨越零点还包含代码路径变化。", "",
             "## 时间窗口", "",
             "- step_wall_ms：perf_counter 包围真实 engine.step，包含调度、准备输入、H2D提交、模型、采样、CPU取回结果和后处理/释放。前后同步排除前序GPU工作。",
             "- runner_wall_ms：包围真实 ModelRunner.run，包含准备、模型、采样与 .tolist 同步。",
             "- prepare_wall_ms：主机包围 prepare_prefill，含Python构造及张量/传输提交；不保证所有异步传输在该窗口结束。",
             "- model_submit_wall_ms：主机包围 run_model 调用，未在调用内部强制同步，不能当作纯CPU耗时。",
             "- cuda_model_stream_ms：CUDA Event包围原 run_model（完整前向+logits），不包含输入准备、采样、CPU传回和后处理。区间可能包含主机提交间隙造成的流空闲，因此不是活跃Kernel执行时间之和。",
             "",
             "CUDA Event在计时前创建并初始化；目标run在取回CPU输出后验证结束Event已完成，再读取elapsed_time。主机和设备各自计算持续时间，不混用时间戳。不能简单用step_wall减去Event时间，推断纯Python或纯CPU开销。", "",
             "## 实验与完整性", "",
             f"batch_size 按1、2、4的顺序分别使用独立进程，进程内随机化(q,c)组合顺序，种子{options['seed']}。每组合先{options['warmup_rounds']}次真实暖机，再{options['repeats']}次采样。只汇总measure行；暖机原始记录同样保留。测量中出现torch.compile帧会报错，不能当作有效性能样本。", "",
             f"有效组合{len(summaries)}个，测量样本{len(measured)}条；分位数使用 `{PERCENTILE_METHOD}`，每个组合的样本量单独记录。", "",
             "| batch | q总数 | c每请求 | q每请求 | KV使用块数 | n | Wall Mean/P50/P95 ms | CUDA Event Mean/P50/P95 ms |",
             "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for row in summaries:
        wall = "/".join(f"{row['step_wall_ms_' + key]:.4f}" for key in ("mean", "p50", "p95"))
        event = "/".join("—" if row["cuda_model_stream_ms_" + key] is None else
                         f"{row['cuda_model_stream_ms_' + key]:.4f}" for key in ("mean", "p50", "p95"))
        lines.append(f"| {row['batch_size']} | {row['q']} | {row['c']} | {row['q_per_request']} | {row['kv_used_before']} | {row['samples']} | {wall} | {event} |")
    lookup = {(row["batch_size"], row["c"], row["q"]): row for row in summaries}
    for metric, title in (("step_wall_ms_mean", "q、c 与主机总耗时均值"),
                           ("cuda_model_stream_ms_mean", "q、c 与 CUDA Event区间均值")):
        lines += ["", f"## {title}", "", "单位ms；每格是同一(q,c,batch)组合多次采样的均值。"]
        for batch in sorted(options["batch_sizes"]):
            lines += ["", f"### batch_size={batch}", "",
                      "| c \\ q | " + " | ".join(str(q) for q in options["q_values"]) + " |",
                      "| --- | " + " | ".join("---" for _ in options["q_values"]) + " |"]
            for c in options["c_values"]:
                cells = []
                for q in options["q_values"]:
                    value = lookup.get((batch, c, q), {}).get(metric)
                    cells.append("—" if value is None else f"{value:.4f}")
                lines.append(f"| {c} | " + " | ".join(cells) + " |")
    lines += ["", "## 主机分段观测与瓶颈边界", "",
              "| batch | 主机Step均值范围 ms | Runner均值范围 ms | 输入准备均值范围 ms | 模型流Event均值范围 ms |",
              "| --- | --- | --- | --- | --- |"]
    for batch in sorted(options["batch_sizes"]):
        group = [row for row in summaries if row["batch_size"] == batch]
        cells = []
        for metric in ("step_wall_ms_mean", "runner_wall_ms_mean", "prepare_wall_ms_mean", "cuda_model_stream_ms_mean"):
            values = [row[metric] for row in group if row[metric] is not None]
            cells.append("—" if not values else f"{min(values):.4f}–{max(values):.4f}")
        lines.append(f"| {batch} | " + " | ".join(cells) + " |")
    lines += ["", "该分段只能定位整体窗口和输入准备的大致量级。模型流区间同时包含线性层、归一化、RoPE、KV写入、attention与LM head；本实验没有逐算子活跃时间或硬件计数器，不能据此断言attention、算力、显存带宽或Python中的某一项是已证实的唯一瓶颈。", "",
              "## 下一步建模可行性与限制", "",
              "这些数据可以作为候选成本模型的训练/验证材料，但本阶段未拟合或验证预测模型。后续至少同时考虑q_i、c_i、batch、q总数、c=0/分页KV路径、Block数及量化/数据类型；不要只用q总数拟合一条直线。应使用独立进程重跑的留出样本检验预测误差，再决定是否可用于调度。", "",
              "同一组合仅10次测量（以实际CLI为准），每个batch只有一个模型进程；组合顺序虽随机化，但未锁频，温度/功耗/后台负载仍可影响结果。CUDA Event包含流内空闲，不是纯Kernel累计时间；没有模拟并发Decode、异步在线接纳、异质批次或KV高压力。q固定而batch增加时，每请求q_i随之减小，不能把它解释成固定每请求长度的批扩展实验。c=0和c>0的代码路径不同，局部非单调或小差异可能来自噪声和离散内核策略，不证明成本模型有效。", "",
              "## 输出与复现", "", "```bash",
              f".venv/bin/python -m benchmarks.prefill_profile --output-dir {output_dir}",
              f".venv/bin/python -m benchmarks.prefill_profile --verify-only {output_dir}",
              ".venv/bin/python -m unittest discover -s tests -v", "git diff --check", "```", "",
              "raw_prefill.csv/profile.json 保存原始暖机和测量记录，summary.csv 保存每组合均值/P50/P95；manifest.json 保存配置、源码hash和实验计划；runs/batch*.json 保存精确Prompt Token ID、环境与遥测。CSV/JSON逐项核对，并重算汇总。原始结果保留本地Git忽略目录，没有提交。"]
    return "\n".join(lines) + "\n"


def positive_int(value):
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model")
    parser.add_argument("--output-dir", type=Path, default=Path("results_prefill_profile"))
    parser.add_argument("--report", type=Path, default=Path("benchmarks/prefill_profile_analysis_zh.md"))
    parser.add_argument("--q-values", nargs="+", type=int, default=[16, 32, 64, 128, 256])
    parser.add_argument("--c-values", nargs="+", type=int, default=[0, 64, 128, 256])
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 2, 4])
    parser.add_argument("--max-model-len", type=positive_int, default=1024)
    parser.add_argument("--warmup-rounds", type=positive_int, default=3)
    parser.add_argument("--repeats", type=positive_int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cuda-events", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--process-timeout", type=float, default=900)
    parser.add_argument("--verify-only", type=Path)
    parser.add_argument("--worker-manifest", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker-batch", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.verify_only:
        payload = verify_results(args.verify_only)
        print(f"Verified {len(payload['summary'])} combinations and all raw samples/CSV/JSON")
        return 0
    if args.worker_manifest:
        return run_worker(args.worker_manifest, args.worker_batch, args.worker_output)
    if not math.isfinite(args.process_timeout) or args.process_timeout <= 0:
        parser.error("process-timeout must be positive and finite")
    cases, skipped = build_cases(args.q_values, args.c_values, args.batch_sizes, args.max_model_len)
    if not cases:
        parser.error("no valid profiling cases")
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"use an empty output directory: {output}")
    options = {key: getattr(args, key) for key in ("q_values", "c_values", "batch_sizes", "max_model_len",
                                                 "warmup_rounds", "repeats", "seed", "cuda_events")}
    options["model"] = str(find_model(args.model))
    metadata = dict(options=options, cases=[case.to_dict() for case in cases], skipped=skipped,
                    git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    percentile_method=PERCENTILE_METHOD, initial_max_num_batched_tokens=256,
                    q_definition="sum of actual scheduled Prefill tokens in target round",
                    c_definition="actual computed prefix tokens per request in homogeneous batch",
                    wall_scope="engine.step; excludes KV allocation and prefix computation",
                    event_scope="run_model forward + logits; stream elapsed including possible launch gaps; not active kernel sum")
    output.mkdir(parents=True, exist_ok=True)
    run_dir = output / "runs"
    run_dir.mkdir()
    manifest = output / "manifest.json"
    manifest.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    runs = []
    for batch in sorted({case.batch_size for case in cases}):
        log_path, run_path = run_dir / f"batch{batch}.log", run_dir / f"batch{batch}.json"
        command = [sys.executable, "-m", "benchmarks.prefill_profile", "--worker-manifest", str(manifest),
                   "--worker-batch", str(batch), "--worker-output", str(run_path)]
        print(f"Starting batch_size={batch}, independent process; log: {log_path}", flush=True)
        failure = None
        with log_path.open("w", encoding="utf-8") as log:
            try:
                result = subprocess.run(command, cwd=Path(__file__).resolve().parents[1],
                                        stdout=log, stderr=subprocess.STDOUT, timeout=args.process_timeout)
                if result.returncode:
                    failure = f"worker exited {result.returncode}; see {log_path}"
            except subprocess.TimeoutExpired:
                failure = f"worker exceeded hard timeout {args.process_timeout}s; see {log_path}"
        if run_path.exists():
            run = json.loads(run_path.read_text(encoding="utf-8"))
            if failure and run["status"] == "ok":
                run.update(status="failed", error=failure)
        else:
            run = dict(status="failed", error=failure or "worker produced no records", batch_size=batch, samples=[])
        runs.append(run)
        write_results(output, runs, metadata, [case for case in cases if case.batch_size <= batch])
        if run["status"] != "ok":
            print(f"FAILED batch={batch}: {run['error']}", file=sys.stderr)
            return 1
    payload = verify_results(output)
    args.report.write_text(render_report(payload, args.output_dir), encoding="utf-8")
    print(f"Validated {len(payload['summary'])} combinations; report: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
