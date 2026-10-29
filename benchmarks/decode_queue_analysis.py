"""Analyze head-first versus round-robin using saved real GPU run records."""

import argparse
from collections import Counter
import json
from pathlib import Path
from statistics import fmean

from benchmarks.metrics import percentile
from benchmarks.reliability import analyze, cross_run_statistics, write_csv


VARIANTS = {"prefill_first": "A", "prefill_first_rr": "B",
            "decode_first_fair2": "C", "decode_first_fair2_rr": "D"}


def first_to_second_statistics(requests):
    rows = [dict(request_id=row["request_id"], first_token_time=row["token_times"][0],
                 second_token_time=row["token_times"][1],
                 first_to_second_s=row["token_times"][1] - row["token_times"][0])
            for row in requests if len(row["token_times"]) >= 2]
    values = [row["first_to_second_s"] for row in rows]
    if any(value < 0 for value in values):
        raise ValueError("first-to-second timestamps must be monotonic")
    return dict(first_to_second_samples=len(values),
                first_to_second_mean_s=fmean(values) if values else None,
                first_to_second_p50_s=percentile(values, .5),
                first_to_second_p95_s=percentile(values, .95),
                first_to_second_p99_s=percentile(values, .99),
                first_to_second_max_s=max(values) if values else None,
                first_to_second_over_100ms=sum(value > .1 for value in values)), rows


def decode_service_gaps(requests, steps):
    """Count actual intervening decode opportunities, separately from wall time."""
    locations = {(request_id, output_index): index for index, step in enumerate(steps)
                 for request_id, output_index in zip(step["emitted_request_ids"], step["output_token_indices"])}
    all_gaps, first_gaps = [], []
    for row in requests:
        for output_index in range(1, len(row["token_times"])):
            start = locations[(row["request_id"], output_index)]
            end = locations[(row["request_id"], output_index + 1)]
            gap = sum(step["phase"] == "decode" for step in steps[start + 1:end])
            all_gaps.append(gap)
            if output_index == 1:
                first_gaps.append(gap)
    return dict(max_intervening_decode_rounds=max(all_gaps, default=0),
                max_first_to_second_intervening_decode_rounds=max(first_gaps, default=0))


def analyze_queue_experiment(baseline, output):
    output = Path(output)
    analysis = analyze(baseline, output)
    counts = Counter(run["summary"]["policy"] for run in analysis["runs"])
    if counts != Counter({key: 3 for key in VARIANTS}):
        raise ValueError("expected exactly three successful runs of each A/B/C/D variant")
    first_rows = []
    for run in analysis["runs"]:
        summary = run["summary"]
        original = json.loads((output / "runs" / f"{summary['policy']}_repeat{summary['repeat']}.json").read_text())
        stats, rows = first_to_second_statistics(original["requests"])
        summary.update(stats)
        summary.update(decode_service_gaps(original["requests"], original["steps"]))
        first_rows.extend(dict(policy=summary["policy"], repeat=summary["repeat"], **row) for row in rows)
    analysis["cross_run_statistics"] = cross_run_statistics([run["summary"] for run in analysis["runs"]])
    analysis["variants"] = VARIANTS
    (output / "queue_analysis.json").write_text(json.dumps(analysis, indent=2, ensure_ascii=False) + "\n",
                                                encoding="utf-8")
    write_csv(output / "queue_cross_run_statistics.csv", analysis["cross_run_statistics"])
    write_csv(output / "first_to_second.csv", first_rows)
    write_csv(output / "queue_run_metrics.csv", [run["summary"] for run in analysis["runs"]])
    return analysis


def render_queue_report(analysis, output, baseline):
    summaries = [run["summary"] for run in analysis["runs"]]
    stats = {(row["policy"], row["metric"]): row for row in analysis["cross_run_statistics"]}
    lines = ["# 第四阶段：running 内部 Decode 轮转公平性", "",
             "第三阶段稳定版本已提交并推送为 `07c86e4`，仅包含代码、测试、报告与整理后的工作负载/汇总；没有上传本地原始结果、.venv 或 wheel。第四阶段改动保持未提交。", "",
             "## 源码变更与状态转换", "",
             "Config 新增 decode_queue_policy，默认 head_first，非法值抛出 ValueError。Scheduler 仅在 _schedule_decode 末尾选择队首/队尾恢复；跨阶段 schedule()、preempt()、postprocess()、Sequence、BlockManager、ModelRunner 和 CUDA Kernel 的逻辑未修改。", "",
             "```python", 'if self.decode_queue_policy == "round_robin":',
             '    self.running.extend(scheduled_seqs)', 'else:',
             '    self.running.extendleft(reversed(scheduled_seqs))', '```', "",
             "选择和抢占期间，成功选中的请求暂存在 scheduled_seqs，直到整个选择循环结束后才恢复。因此同一请求不会被重复选中或成为本轮后续抢占的目标。恢复到队尾后，未选中的 running 在下一轮优先；postprocess 仍从 running 中移除完成请求并释放 KV。当前引擎同步执行 schedule→run→postprocess，队列在 schedule 返回前恢复即可满足下一轮优先级，无需改变 postprocess。", "",
             "6 个 running、max_num_seqs=4 时，前三个 Decode 批次为 [0,1,2,3]、[4,5,0,1]、[2,3,4,5]，每个请求服务两次；head_first 保持原有队首行为。", "",
             "## 实验设计", "",
             "| 组 | 跨阶段策略 | 阈值 | Decode 队列 |", "| --- | --- | --- | --- |",
             "| A | prefill_first | 0 | head_first |", "| B | prefill_first | 0 | round_robin |",
             "| C | decode_first | 2 | head_first |", "| D | decode_first | 2 | round_robin |", "",
             f"使用 `{baseline}/workload.json` 的完全相同 Token ID、长度和到达时间；工作负载 SHA-256 为 `{analysis['metadata']['workload_sha256']}`。RTX 4060 Laptop 8GB、本地 Qwen3-0.6B。12 请求、492 输出 Token；参数仍为 TP=1、eager=True、模型长度512、批Token预算256、max_num_seqs=4、显存比例0.65、ignore_eos=True。", "",
             "每组独立进程、相同显式暖机，三次重复。同一 repeat 用相同采样种子 seed+repeat。执行顺序为 A/B/C/D、B/C/D/A、C/D/A/B；仅三次重复不能完全平衡四个执行位置。", "",
             "计时仍为 perf_counter，Token 在真实 append_token 后记录；未产出 Token 的 Chunk 不记录 TTFT。ITL 按相邻间隔池化，P95/P99 用 (n−1)×q 线性插值。跨运行标准差为样本标准差 ddof=1。逐轮时间是主机 engine.step 总时间。", "",
             "## 每次实验的指标", "",
             "| 组/repeat | TTFT P95 ms | ITL mean ms | ITL P95 ms | ITL P99 ms | ITL max ms | ITL >100ms | 延迟 P95 ms | 输出 tok/s | 平均Decode批大小 |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for row in summaries:
        cells = [f"{VARIANTS[row['policy']]}/{row['repeat']}"]
        cells += [f"{row[key]*1000:.3f}" for key in ("ttft_p95_s", "itl_mean_s", "itl_p95_s", "itl_p99_s", "itl_max_s")]
        cells += [str(row["itl_over_100ms"]), f"{row['request_latency_p95_s']*1000:.3f}",
                  f"{row['output_token_throughput']:.3f}", f"{row['decode_mean_batch_size']:.6f}"]
        lines.append("| " + " | ".join(cells) + " |")
    focus = ("ttft_p95_s", "itl_mean_s", "itl_p95_s", "itl_p99_s", "itl_max_s", "itl_over_100ms",
             "request_latency_p95_s", "output_token_throughput", "decode_mean_batch_size",
             "first_to_second_mean_s", "first_to_second_p50_s", "first_to_second_p95_s",
             "first_to_second_p99_s", "first_to_second_max_s", "first_to_second_over_100ms")
    focus += ("max_intervening_decode_rounds", "max_first_to_second_intervening_decode_rounds")
    lines += ["", "## 跨运行统计（每项 n=3）", "",
              "| 组 | 指标 | 均值 | 样本标准差 | Min | Max |", "| --- | --- | --- | --- | --- | --- |"]
    for policy, label in VARIANTS.items():
        for key in focus:
            row = stats[(policy, key)]
            scale = 1000 if key.endswith("_s") else 1
            unit = " (ms)" if scale == 1000 else ""
            lines.append(f"| {label} | {key}{unit} | " + " | ".join(
                f"{row[name]*scale:.6f}" for name in ("mean", "sample_stddev", "min", "max")) + " |")
    lines += ["", "## 第一个到第二个输出 Token 的独立统计", "",
              "每次运行有 12 个首→第二 Token 间隔；这不同于全体 480 个 ITL 间隔。逐请求记录见 first_to_second.csv。", "",
              "| 组/repeat | Mean ms | P50 ms | P95 ms | P99 ms | Max ms | >100ms |",
              "| --- | --- | --- | --- | --- | --- | --- |"]
    for row in summaries:
        lines.append(f"| {VARIANTS[row['policy']]}/{row['repeat']} | " + " | ".join(
            f"{row[key]*1000:.3f}" for key in ("first_to_second_mean_s", "first_to_second_p50_s",
                                              "first_to_second_p95_s", "first_to_second_p99_s",
                                              "first_to_second_max_s")) + f" | {row['first_to_second_over_100ms']} |")
    lines += ["", "## Decode 批大小、完成数与资源", "",
              "| 组/repeat | Decode 轮数 | BS1/2/3/4 轮数 | 完成/请求 | 输出Token | 空闲/总KV Block | 引用计数和 | 抢占 | 编译帧 |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for run in analysis["runs"]:
        row, blocks = run["summary"], run["resource_check"]
        distribution = {item["batch_size"]: item["rounds"] for item in run["decode_batches"]["distribution"]}
        counts = "/".join(str(distribution.get(i, 0)) for i in range(1, 5))
        lines.append(f"| {VARIANTS[row['policy']]}/{row['repeat']} | {row['decode_steps']} | {counts} | {row['completed_requests']}/{row['requested_requests']} | {row['total_output_tokens']} | {blocks['free_blocks']}/{blocks['total_blocks']} | {blocks['block_ref_count_sum']} | {row['preemptions']} | {row['torch_compile_frames_during_measurement']} |")
    lines += ["", "## 同 repeat 的绝对差值（round_robin − head_first）", "",
              "| 配对 | repeat | ITL P99差 ms | 首→第二P95差 ms | 延迟P95差 ms | 吞吐差 tok/s |",
              "| --- | --- | --- | --- | --- | --- |"]
    by_run = {(row["policy"], row["repeat"]): row for row in summaries}
    for head, rr, name in (("prefill_first", "prefill_first_rr", "B−A"),
                           ("decode_first_fair2", "decode_first_fair2_rr", "D−C")):
        for repeat in range(3):
            old, new = by_run[(head, repeat)], by_run[(rr, repeat)]
            lines.append(f"| {name} | {repeat} | " + " | ".join(
                f"{(new[key]-old[key])*1000:+.3f}" for key in ("itl_p99_s", "first_to_second_p95_s", "request_latency_p95_s")) +
                f" | {new['output_token_throughput']-old['output_token_throughput']:+.3f} |")
    lines += ["", "## 本次数据中的具体取舍", ""]
    for head, rr, name in (("prefill_first", "prefill_first_rr", "B 相对 A"),
                           ("decode_first_fair2", "decode_first_fair2_rr", "D 相对 C")):
        def mean(policy, key):
            return stats[(policy, key)]["mean"]
        lines += [
            f"{name}：ITL P99 的三次均值为 {mean(head, 'itl_p99_s')*1000:.3f}→{mean(rr, 'itl_p99_s')*1000:.3f}ms，ITL Max 为 {mean(head, 'itl_max_s')*1000:.3f}→{mean(rr, 'itl_max_s')*1000:.3f}ms，首→第二 Token P95 为 {mean(head, 'first_to_second_p95_s')*1000:.3f}→{mean(rr, 'first_to_second_p95_s')*1000:.3f}ms。",
            "",
            f"同时 ITL mean 为 {mean(head, 'itl_mean_s')*1000:.3f}→{mean(rr, 'itl_mean_s')*1000:.3f}ms，ITL P95 为 {mean(head, 'itl_p95_s')*1000:.3f}→{mean(rr, 'itl_p95_s')*1000:.3f}ms；请求延迟 P50 为 {mean(head, 'request_latency_p50_s')*1000:.3f}→{mean(rr, 'request_latency_p50_s')*1000:.3f}ms，P95 为 {mean(head, 'request_latency_p95_s')*1000:.3f}→{mean(rr, 'request_latency_p95_s')*1000:.3f}ms。不能把长尾下降解释为所有请求都更快。",
            "",
            f"输出吞吐量均值 {mean(head, 'output_token_throughput'):.3f}→{mean(rr, 'output_token_throughput'):.3f} tok/s；>100ms ITL 数量的每次运行均值为 {mean(head, 'itl_over_100ms'):.3f}→{mean(rr, 'itl_over_100ms'):.3f}。每运行最长相邻 Token 之间的未服务 Decode 轮数均值为 {mean(head, 'max_intervening_decode_rounds'):g}→{mean(rr, 'max_intervening_decode_rounds'):g}。",
            ""]
    lines += ["Prefill-first + round_robin 中保留的最坏长间隔位于连续 Prefill 期间，这符合未改变跨阶段 Prefill 优先级的约束；轮转优化只保证 Decode 机会的相对顺序，不会消除 Prefill 占用时间。", ""]
    lines += ["", "## 性能取舍与限制", "",
              "round_robin 提供的是 running 内部的服务机会轮转，不改变 Prefill/Decode 的跨阶段优先级。等待中的长 Prompt、Chunked Prefill 和 KV 不足仍可能推迟输出；即便轮转，也不保证固定毫秒级 ITL。公平分配可能让原本始终占据队首的长请求更晚完成，因此必须同时观察请求延迟和吞吐，不能仅看首→第二 Token 间隔。实际效果以以上配对差值为准，不预设每项都会改善。", "",
              "仅单台笔记本 GPU、单个12请求混合负载和每组3次重复；无生产服务异步到达/网络时延。温度/时钟仅为窗口前后快照；四组、三轮不完全平衡位置，不能排除热状态、后台负载与频率漂移。输出长度固定并忽略EOS，不代表所有真实聊天场景；低KV压力下的性能不能外推到频繁抢占。追踪有轻量CPU开销，跨阶段旧实验与本次不直接做百分比排名。", "",
              "## 复现与文件", "", "```bash",
              f".venv/bin/python -m benchmarks.mixed_load_bench --workload-from {baseline}/workload.json --policies prefill_first prefill_first_rr decode_first_fair2 decode_first_fair2_rr --repeats 3 --order-mode rotate --output-dir {output}",
              f".venv/bin/python -m benchmarks.decode_queue_analysis --baseline {baseline} --results {output}",
              ".venv/bin/python -m unittest discover -s tests -v", "git diff --check", "```", "",
              f"完整原始结果在被 Git 忽略的 `{output}/`；每次独立 CSV/JSON 位于 runs/ 子目录。queue_cross_run_statistics.csv 包含所有指标均值、标准差、Min/Max，queue_run_metrics.csv 保存逐次指标，first_to_second.csv 保存逐请求首→第二间隔，raw_steps.csv 保存逐轮批大小和时间。源码 Diff 保存在 source.diff。"]
    lines += ["", "## 验证", "",
              "先新增调度测试并确认在缺少配置时失败，再修改实现。全量 65 项单元测试通过，包含原有 49 项；覆盖六请求轮转、默认兼容、完成、Chunked Prefill、跨阶段计数、KV 不足、抢占/恢复、未调度幸存请求优先级、共享 Prefix 引用计数和指标计算。12 次 GPU 运行的请求 CSV、逐轮 CSV、JSON 及资源快照均校验通过。git diff --check 和新增文件空白检查通过。未安装依赖；第四阶段未 Commit/Push。"]
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=Path("results"))
    parser.add_argument("--results", type=Path, default=Path("results_stage4"))
    parser.add_argument("--report", type=Path, default=Path("benchmarks/decode_queue_analysis_zh.md"))
    args = parser.parse_args(argv)
    result = analyze_queue_experiment(args.baseline, args.results)
    args.report.write_text(render_queue_report(result, args.results, args.baseline), encoding="utf-8")
    print(f"Validated {len(result['runs'])} queue experiments; report: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
