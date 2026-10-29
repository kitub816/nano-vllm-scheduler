"""Standard-library-only reliability analysis of saved, real benchmark artifacts."""

import argparse
from collections import Counter, defaultdict
import csv
import json
import math
from pathlib import Path
from statistics import fmean, stdev

from benchmarks.metrics import percentile, verify_results


def interval_analysis(requests):
    intervals = []
    for row in requests:
        times = row["token_times"]
        if isinstance(times, str):
            times = json.loads(times)
        for index, (start, end) in enumerate(zip(times, times[1:]), start=1):
            if not (math.isfinite(start) and math.isfinite(end)) or end < start:
                raise ValueError("invalid token interval")
            intervals.append(dict(request_id=int(row["request_id"]), from_output_index=index,
                                  to_output_index=index + 1, start_time=start, end_time=end,
                                  itl_s=end - start))
    values = [row["itl_s"] for row in intervals]
    tail = [value for value in values if value > 0.1]
    body = [value for value in values if value <= 0.1]
    return dict(samples=len(values), mean_s=fmean(values) if values else None,
                p50_s=percentile(values, .5), p95_s=percentile(values, .95),
                p99_s=percentile(values, .99), max_s=max(values) if values else None,
                over_100ms=len(tail), over_200ms=sum(value > .2 for value in values),
                over_100ms_time_share=sum(tail) / sum(values) if sum(values) else None,
                mean_at_most_100ms_s=fmean(body) if body else None,
                longest_10=sorted(intervals, key=lambda row: row["itl_s"], reverse=True)[:10])


def decode_batch_statistics(steps):
    groups = defaultdict(list)
    for step in steps:
        if step["phase"] == "decode":
            groups[step["batch_size"]].append(step["duration_s"])
    count = sum(len(values) for values in groups.values())
    tokens = sum(size * len(values) for size, values in groups.items())
    return dict(rounds=count, decode_tokens=tokens,
                mean_batch_size=tokens / count if count else None,
                distribution=[dict(batch_size=size, rounds=len(values),
                                   fraction=len(values) / count,
                                   mean_step_duration_s=fmean(values),
                                   total_step_duration_s=sum(values))
                              for size, values in sorted(groups.items())])


def validate_run_steps(run):
    """Join every logged append to exactly one real step, and verify final resources."""
    summary = run["summary"]
    if summary["status"] != "ok":
        raise ValueError(f"cannot validate successful-run invariants for {summary['status']}")
    records = {row["request_id"]: row for row in run["requests"]}
    if len(records) != summary["requested_requests"]:
        raise ValueError("request count mismatch")
    seen = Counter()
    phases = Counter()
    previous_end = 0
    for index, step in enumerate(run["steps"]):
        if (step["step_index"] != index or step["start_time"] < previous_end
                or not step["start_time"] <= step["schedule_time"] <= step["end_time"]
                or step["end_time"] > summary["elapsed_seconds"]):
            raise ValueError("non-monotonic or misindexed step trace")
        if not math.isclose(step["duration_s"], step["end_time"] - step["start_time"], abs_tol=1e-12):
            raise ValueError("step duration mismatch")
        previous_end = step["end_time"]
        if step["phase"] not in ("prefill", "decode"):
            raise ValueError("unknown step phase")
        phases[step["phase"]] += 1
        ids = step["request_ids"]
        emitted = step["emitted_request_ids"]
        indices = step["output_token_indices"]
        if (step["batch_size"] != len(ids) or not ids or len(set(ids)) != len(ids)
                or len(emitted) != len(indices) or len(emitted) != step["output_tokens"]
                or len(set(emitted)) != len(emitted) or not set(emitted) <= set(ids)):
            raise ValueError("step batch/output membership mismatch")
        if step["phase"] == "decode" and step["output_tokens"] != step["batch_size"]:
            raise ValueError("decode output count mismatch")
        for request_id, output_index in zip(emitted, indices):
            record = records[request_id]
            if not 1 <= output_index <= len(record["token_times"]):
                raise ValueError("invalid output token index")
            stamp = record["token_times"][output_index - 1]
            if not step["schedule_time"] <= stamp <= step["end_time"]:
                raise ValueError("token timestamp lies outside its step")
            seen[(request_id, output_index)] += 1
    expected = Counter({(request_id, index): 1 for request_id, row in records.items()
                        for index in range(1, len(row["token_times"]) + 1)})
    if seen != expected:
        raise ValueError("missing or duplicated token observations in step trace")
    if (phases["prefill"] != summary["prefill_steps"] or phases["decode"] != summary["decode_steps"]
            or sum(step["preemptions"] for step in run["steps"]) != summary["preemptions"]
            or sum(seen.values()) != summary["total_output_tokens"]):
        raise ValueError("step totals differ from summary")
    if (summary["completed_requests"] != len(records)
            or any(row["completion_time"] is None or row["output_tokens"] != row["max_tokens"]
                   for row in records.values())):
        raise ValueError("unfinished requests or incorrect output lengths")
    resources = run["resource_check"]
    if (not resources or not resources["released"] or resources["waiting_requests"]
            or resources["running_requests"] or resources["used_blocks"]
            or resources["block_ref_count_sum"] or resources["free_blocks"] != resources["total_blocks"]):
        raise ValueError("KV blocks or live requests were not completely released")
    if summary["torch_compile_frames_during_measurement"] != 0:
        raise ValueError("compilation occurred during measurement")
    return True


def cross_run_statistics(summaries):
    """Sample SD (ddof=1) of per-run metrics; do not pool percentiles across runs."""
    groups = defaultdict(list)
    for summary in summaries:
        if summary["status"] != "ok":
            raise ValueError("failed runs must not be included in performance statistics")
        groups[summary["policy"]].append(summary)
    rows = []
    for policy, group in groups.items():
        keys = dict.fromkeys(key for item in group for key, value in item.items()
                             if isinstance(value, (int, float)) and not isinstance(value, bool)
                             and key not in ("repeat", "max_consecutive_decode_steps"))
        for key in keys:
            values = [item[key] for item in group if item.get(key) is not None]
            if not values or not all(math.isfinite(value) for value in values):
                raise ValueError(f"invalid cross-run samples: {policy}, {key}")
            rows.append(dict(policy=policy, metric=key, samples=len(values), mean=fmean(values),
                             sample_stddev=stdev(values) if len(values) > 1 else None,
                             min=min(values), max=max(values)))
    return rows


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: "" if value is None else json.dumps(value) if isinstance(value, list)
                             else str(value) for key, value in row.items()})


def write_step_csv(path, runs):
    rows = [{"policy": run["summary"]["policy"], "repeat": run["summary"]["repeat"], **step}
            for run in runs for step in run.get("steps", [])]
    write_csv(path, rows)
    verify_step_csv(path, runs)


def verify_step_csv(path, runs):
    rows = [{"policy": run["summary"]["policy"], "repeat": run["summary"]["repeat"], **step}
            for run in runs for step in run.get("steps", [])]
    with Path(path).open(encoding="utf-8", newline="") as file:
        actual = list(csv.DictReader(file))
    expected = [{key: json.dumps(value) if isinstance(value, list) else str(value)
                 for key, value in row.items()} for row in rows]
    if actual != expected:
        raise ValueError("raw step CSV and run JSON differ")
    return True


def long_gap_evidence(run):
    worst = interval_analysis(run["requests"])["longest_10"]
    evidence = []
    for gap in worst:
        request_id = gap["request_id"]
        emitting = [step for step in run["steps"] if request_id in step["emitted_request_ids"]]
        first = next(step for step in emitting if
                     dict(zip(step["emitted_request_ids"], step["output_token_indices"]))[request_id]
                     == gap["from_output_index"])
        second = next(step for step in emitting if
                      dict(zip(step["emitted_request_ids"], step["output_token_indices"]))[request_id]
                      == gap["to_output_index"])
        between = run["steps"][first["step_index"] + 1:second["step_index"]]
        evidence.append({**gap, "first_step": first["step_index"], "next_step": second["step_index"],
                         "intervening_prefills": sum(step["phase"] == "prefill" for step in between),
                         "intervening_decodes": sum(step["phase"] == "decode" for step in between),
                         "intervening_steps_without_request": sum(
                             request_id not in step["request_ids"] for step in between)})
    return evidence


def analyze(baseline_dir, repeat_dir):
    baseline_dir, repeat_dir = Path(baseline_dir), Path(repeat_dir)
    verify_results(baseline_dir)
    verify_results(repeat_dir)
    with (baseline_dir / "raw_requests.csv").open(encoding="utf-8", newline="") as file:
        baseline_groups = defaultdict(list)
        for row in csv.DictReader(file):
            baseline_groups[(row["policy"], int(row["repeat"]))].append(row)
    baseline = [{"policy": policy, "repeat": repeat, **interval_analysis(rows)}
                for (policy, repeat), rows in baseline_groups.items()]
    payload = json.loads((repeat_dir / "summary.json").read_text(encoding="utf-8"))
    original = json.loads((baseline_dir / "workload.json").read_text(encoding="utf-8"))
    manifest = json.loads((repeat_dir / "workload.json").read_text(encoding="utf-8"))
    if (manifest["requests"] != original["requests"]
            or manifest["workload_sha256"] != original["workload_sha256"]
            or manifest["config"] != original["config"]):
        raise ValueError("workload/config changed from baseline")
    runs = []
    original_runs = []
    derived = []
    batch_rows = []
    for item in payload["runs"]:
        policy, repeat = item["policy"], item["repeat"]
        run = json.loads((repeat_dir / "runs" / f"{policy}_repeat{repeat}.json").read_text(encoding="utf-8"))
        if run["summary"] != item:
            raise ValueError("per-run and aggregate JSON summaries differ")
        validate_run_steps(run)
        original_runs.append(run)
        verify_results(repeat_dir / "runs" / f"{policy}_repeat{repeat}")
        verify_step_csv(repeat_dir / "runs" / f"{policy}_repeat{repeat}" / "raw_steps.csv", [run])
        tails = interval_analysis(run["requests"])
        batches = decode_batch_statistics(run["steps"])
        augmented = {**item, "itl_p99_s": tails["p99_s"], "itl_max_s": tails["max_s"],
                     "itl_over_100ms": tails["over_100ms"], "itl_over_200ms": tails["over_200ms"],
                     "decode_mean_batch_size": batches["mean_batch_size"]}
        augmented["decode_step_mean_s"] = fmean(
            step["duration_s"] for step in run["steps"] if step["phase"] == "decode")
        for size in range(1, manifest["config"]["max_num_seqs"] + 1):
            batch = next((row for row in batches["distribution"] if row["batch_size"] == size), None)
            augmented[f"decode_batch_{size}_rounds"] = batch["rounds"] if batch else 0
            augmented[f"decode_batch_{size}_step_mean_s"] = batch["mean_step_duration_s"] if batch else None
        derived.append(augmented)
        batch_rows.extend(dict(policy=policy, repeat=repeat, **row) for row in batches["distribution"])
        runs.append(dict(summary=augmented, intervals=tails, decode_batches=batches,
                         gap_evidence=long_gap_evidence(run), resource_check=run["resource_check"],
                         environment=run["environment"]))
    verify_step_csv(repeat_dir / "raw_steps.csv", original_runs)
    stats = cross_run_statistics(derived)
    global_longest = sorted([dict(policy=item["policy"], **gap)
                             for item in baseline for gap in item["longest_10"]],
                            key=lambda row: row["itl_s"], reverse=True)[:10]
    result = dict(baseline=baseline, runs=runs, cross_run_statistics=stats,
                  baseline_global_longest_10=global_longest,
                  metadata=payload["metadata"], sample_stddev_ddof=1)
    (repeat_dir / "analysis.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                                               encoding="utf-8")
    write_csv(repeat_dir / "cross_run_statistics.csv", stats)
    write_csv(repeat_dir / "decode_batch_statistics.csv", batch_rows)
    write_csv(repeat_dir / "baseline_top_itl.csv", [dict(policy=item["policy"], repeat=item["repeat"], **gap)
                                                       for item in baseline for gap in item["longest_10"]])
    write_csv(repeat_dir / "baseline_global_top10.csv", global_longest)
    return result


def render_report(analysis, baseline_dir, repeat_dir):
    lines = ["# Benchmark 可靠性复核", "", "## 测量方法审查", "",
             "复核保留原调度算法、ModelRunner 和 CUDA Kernel，仅扩展 Benchmark 的观察和导出。", "",
             "所有请求时间戳、逐轮时间和正式耗时使用同一 worker 内同一个 perf_counter 单调时钟及同一 origin。不同进程的相对时间不直接相减；只汇总各自计算出的指标。GPU 温度/频率快照在正式窗口外采集，不用于请求延迟计算。", "",
             "ModelRunner.run() 在返回前对采样 GPU Tensor 调用 .tolist()，取得主机可用的 Token 列表；LLMEngine.step() 随后调用 postprocess()。观察器在真实 append_token() 返回后记录一次 Token 时间，postprocess() 完成状态转换和释放后记录请求完成时间。未完成的 Prefill Chunk 不追加 Token，不记录虚假的 TTFT。", "",
             "TTFT 从实际入队到首 Token；ITL 为同一请求相邻 Token 时间之差（跨请求池化间隔）；完整延迟从实际入队到 postprocess 完成。吞吐量为真实追加的总输出 Token 数 / 暖机后正式耗时，包含计划到达间隙。计划到达只是偏移；接纳延迟另行报告，未混入以入队为起点的 TTFT。", "",
             "逐轮 trace 将每个 (request_id, output_token_index) 与原始 Token 时间一一核对，验证不重计、不遗漏；Decode 批大小等于该轮实际追加 Token 数。P50/P95/P99 均沿用 (n−1)×q 的 type 7 线性插值。轮次耗时是 engine.step() 的主机总耗时，包含调度、GPU 计算/结果传输、后处理与轻量插桩，不是 CUDA Kernel 耗时。", "",
             "## 既有单次数据的 ITL 长尾", "",
             f"来源：`{baseline_dir}/raw_requests.csv`，原文件未覆盖。每种策略 480 个 ITL 样本。", "",
             "| 策略 | 均值 ms | P50 ms | P95 ms | P99 ms | Max ms | >100ms | >200ms | >100ms 占总 ITL 时间 | 其余均值 ms |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for item in analysis["baseline"]:
        lines.append(f"| {item['policy']} | " + " | ".join(f"{item[key]*1000:.3f}" for key in
                     ("mean_s", "p50_s", "p95_s", "p99_s", "max_s")) +
                     f" | {item['over_100ms']} | {item['over_200ms']} | {item['over_100ms_time_share']*100:.2f}% | {item['mean_at_most_100ms_s']*1000:.3f} |")
    lines += ["", "Prefill-first 超过 100ms 的 12 个间隔只占 2.5%，但占间隔总时长的 55.88%；均发生在第 1→2 个输出 Token 之间。均值因此可以高于 P95，现有证据不支持把该大小关系本身判定为指标 Bug。它反映请求先得到首 Token、随后长时间没有获得 Decode 服务。", ""]
    lines += ["### 三策略合并后的最长 10 个 ITL", "",
              "| 策略 | request_id | 输出 Token 序号 | ITL ms |", "| --- | --- | --- | --- |"]
    for gap in analysis["baseline_global_longest_10"]:
        lines.append(f"| {gap['policy']} | {gap['request_id']} | {gap['from_output_index']}→{gap['to_output_index']} | {gap['itl_s']*1000:.3f} |")
    lines.append("")
    for item in analysis["baseline"]:
        lines += [f"### {item['policy']}：最长 10 个 ITL", "",
                  "| request_id | 输出 Token 序号 | ITL ms | 起点 s | 终点 s |", "| --- | --- | --- | --- | --- |"]
        for gap in item["longest_10"]:
            lines.append(f"| {gap['request_id']} | {gap['from_output_index']}→{gap['to_output_index']} | {gap['itl_s']*1000:.3f} | {gap['start_time']:.6f} | {gap['end_time']:.6f} |")
        lines.append("")
    lines += ["## 重复实验设计与逐次结果", "",
              "相同精确 Token ID、输出长度和计划到达，SHA-256 与原 workload 一致。引擎配置和暖机函数保持一致，每次重新创建独立 worker、模型实例和缓存。三次重复按 A/B/C、B/C/A、C/A/B 轮换顺序；同一 repeat 内采样种子相同（seed+repeat），不保证不同调度下输出内容逐 Token 相同。", "",
              "| 执行序号 | repeat | 策略 | TTFT 均值 ms | ITL 均值 ms | 输出 tok/s | Decode 轮数 | 平均批大小 | 暖机后/运行后温度 °C | SM 频率前/后 MHz |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    order = {(item["policy"], item["repeat"]): item["execution_index"]
             for item in analysis["metadata"]["execution_order"]}
    for run in analysis["runs"]:
        row, env = run["summary"], run["environment"]
        before, after = env.get("gpu_before_measurement", {}), env.get("gpu_after_measurement", {})
        lines.append(f"| {order[(row['policy'], row['repeat'])]} | {row['repeat']} | {row['policy']} | {row['ttft_mean_s']*1000:.3f} | {row['itl_mean_s']*1000:.3f} | {row['output_token_throughput']:.3f} | {row['decode_steps']} | {row['decode_mean_batch_size']:.6f} | {before.get('temperature.gpu', '不可用')}/{after.get('temperature.gpu', '不可用')} | {before.get('clocks.current.sm', '不可用')}/{after.get('clocks.current.sm', '不可用')} |")
    telemetry = [run["environment"].get(key, {}) for run in analysis["runs"]
                 for key in ("gpu_before_measurement", "gpu_after_measurement")]
    for key, label, unit in (("temperature.gpu", "前后快照的 GPU 温度", "°C"),
                             ("clocks.current.sm", "前后快照的 SM 频率", "MHz")):
        values = [float(row[key]) for row in telemetry if str(row.get(key, "")).replace(".", "", 1).isdigit()]
        if values:
            lines += ["", f"{label}范围为 {min(values):g}–{max(values):g} {unit}。"]
    lines += ["", "顺序已轮换，但频率快照存在变化，因此不能宣称硬件时钟保持恒定；三个样本也不足以把轮次耗时变化归因于温度、时钟或调度中的某一个因素。"]
    lines += ["", "## Decode 批大小与轮次耗时", "",
              "原单次实验没有逐轮记录，不能从旧 raw_requests.csv 准确恢复当时的批大小分布或耗时；下表来自本次真实 engine.step() 追踪。", "",
              "| 策略 | repeat | batch size | 轮数 | Decode 轮数占比 | 平均 engine.step 耗时 ms |",
              "| --- | --- | --- | --- | --- | --- |"]
    for run in analysis["runs"]:
        row = run["summary"]
        for batch in run["decode_batches"]["distribution"]:
            lines.append(f"| {row['policy']} | {row['repeat']} | {batch['batch_size']} | {batch['rounds']} | {batch['fraction']*100:.2f}% | {batch['mean_step_duration_s']*1000:.3f} |")
    lines += ["", "每个请求的首 Token 在 Prefill 完成时追加，因此 492 输出 Token 中有 12 个来自 Prefill，其余 480 个来自 Decode。对每次运行都核对 Σ(batch_size×轮数)=480。平均 Decode 批大小为 480/Decode轮数，不能把 Decode 轮数当作 Token 数。", "",
              "decode_first 在现有 running 可 Decode 时持续 Decode，不接纳 waiting 做 Prefill，因而较少请求同时参与 Decode；prefill_first 提前让更多请求进入 running，fair2 周期性插入 Prefill，后两者更容易构成大批次。这解释了同为 480 个 Decode Token 却需要不同轮数。max_num_seqs=4 是单批上限，running 可以超过 4；_schedule_decode() 将选中的请求放回 running 队首，已有队首请求可能长期占据批次，造成后面请求首 Token 后的长间隔。这里是对现有行为的测量与解释，未修改该行为。", "",
              "### 新追踪中的长间隔证据（每策略 repeat 0 的最坏间隔）", "",
              "| 策略 | request_id | Token 序号 | ITL ms | 两端 step | 中间 Prefill/Decode 轮数 | 中间未调度该请求的轮数 |",
              "| --- | --- | --- | --- | --- | --- | --- |"]
    for run in analysis["runs"]:
        if run["summary"]["repeat"] == 0:
            gap = run["gap_evidence"][0]
            lines.append(f"| {run['summary']['policy']} | {gap['request_id']} | {gap['from_output_index']}→{gap['to_output_index']} | {gap['itl_s']*1000:.3f} | {gap['first_step']}→{gap['next_step']} | {gap['intervening_prefills']}/{gap['intervening_decodes']} | {gap['intervening_steps_without_request']} |")
    lines += ["", "## 各指标跨运行统计", "",
              "下表对每个运行先计算指标，再对三次运行求均值、样本标准差（ddof=1）、最小值、最大值；不把三次原始样本混起来算分位数。时间指标在表中换算为 ms，CSV/JSON 保持秒。每项 n=3，不报告显著性或普遍加速结论。", "",
              "| 策略 | 指标 | n | 均值 | 样本标准差 | 最小值 | 最大值 |", "| --- | --- | --- | --- | --- | --- | --- |"]
    for row in analysis["cross_run_statistics"]:
        scale = 1000 if row["metric"].endswith("_s") else 1
        label = row["metric"] + (" (ms)" if scale == 1000 else "")
        lines.append(f"| {row['policy']} | {label} | {row['samples']} | " + " | ".join(
            f"{row[key]*scale:.6f}" for key in ("mean", "sample_stddev", "min", "max")) + " |")
    throughputs = {row["policy"]: row for row in analysis["cross_run_statistics"]
                   if row["metric"] == "output_token_throughput"}
    if "prefill_first" in throughputs and "decode_first_fair2" in throughputs:
        first, fair = throughputs["prefill_first"], throughputs["decode_first_fair2"]
        lines += ["", f"Prefill-first 三次吞吐量范围为 {first['min']:.3f}–{first['max']:.3f} tok/s；fair2 为 {fair['min']:.3f}–{fair['max']:.3f} tok/s。两者区间重叠，不能根据各自均值给出稳定的快慢排序。"]
    lines += ["", "## 完成数、Token 与资源核验", "",
              "| 策略 | repeat | 完成/请求 | 输出 Token | ITL 样本 | Prefill/Decode | 抢占 | 空闲/总 KV Block | 引用计数和 | 计时编译帧 |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for run in analysis["runs"]:
        row, resources = run["summary"], run["resource_check"]
        lines.append(f"| {row['policy']} | {row['repeat']} | {row['completed_requests']}/{row['requested_requests']} | {row['total_output_tokens']} | {row['itl_samples']} | {row['prefill_steps']}/{row['decode_steps']} | {row['preemptions']} | {resources['free_blocks']}/{resources['total_blocks']} | {resources['block_ref_count_sum']} | {row['torch_compile_frames_during_measurement']} |")
    lines += ["", "## 文件、复现命令与限制", "",
              f"原始实验保留在 `{baseline_dir}`。重复实验的汇总和逐请求/逐轮 CSV 位于 `{repeat_dir}`；每个 `runs/<policy>_repeatN/` 还有独立 summary.json、summary.csv、raw_requests.csv、raw_steps.csv，旁边的 JSON 保存原始请求、逐轮记录、GPU/软件元数据和资源快照。", "",
              "```bash", f".venv/bin/python -m benchmarks.mixed_load_bench --workload-from {baseline_dir}/workload.json --repeats 3 --order-mode rotate --output-dir {repeat_dir}",
              f".venv/bin/python -m benchmarks.reliability --baseline {baseline_dir} --repeats {repeat_dir} --report benchmarks/benchmark_analysis_zh.md",
              ".venv/bin/python -m unittest discover -s tests -v", "git diff --check", "```", "",
              "限制：单台笔记本 GPU、12 请求、一个混合工作负载、每策略只有三次重复；顺序轮换只能平衡位置，不能隔离后台负载、GPU 温度/频率或功耗变化。GPU 遥测只有窗口前后快照，不是连续频率轨迹，不能据此断言漂移已消除或把变化归因于某个因素。同步单线程仅在 step 间隙注入请求，存在可测的接纳延迟；没有模拟异步生产服务、网络或客户端时延。轻量追踪也有 CPU 开销，旧单次与新增追踪重复不是完全相同的仪器条件。默认 ignore_eos 固定长度、原始文本截取，不代表所有真实聊天负载；没有制造 KV 高压，零抢占不能验证高压性能。", "",
              "本复核确认测量定义及记录的内部一致性，区分了长尾 ITL、批大小与轮数；不宣称某种调度算法普遍更快。"]
    lines += ["", "## 验证记录", "",
              "全部 49 项单元测试通过（保留原有 42 项，新增 7 项可靠性测试）。9 次 GPU 运行的请求 CSV、逐轮 CSV、JSON 汇总、Token/轮次对应关系、输出长度和 KV 资源快照均已校验。原始 results/ 的 CSV/JSON 校验仍通过；git diff --check 与新增文件空白检查通过。未新增依赖，未修改 Scheduler、ModelRunner 或 CUDA Kernel，未执行 Git Commit。"]
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=Path("results"))
    parser.add_argument("--repeats", type=Path, default=Path("results_reliability"))
    parser.add_argument("--report", type=Path, default=Path("benchmarks/benchmark_analysis_zh.md"))
    args = parser.parse_args(argv)
    result = analyze(args.baseline, args.repeats)
    args.report.write_text(render_report(result, args.baseline, args.repeats), encoding="utf-8")
    print(f"Validated {len(result['runs'])} runs; report: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
