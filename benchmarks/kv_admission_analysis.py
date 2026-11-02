"""Reproduce the Chinese reserve comparison from verified real GPU artifacts."""

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
from statistics import fmean, stdev

from benchmarks.kv_pressure_bench import verify_results, export_csv


LABELS = ("prefill_first", "B0", "B4", "B8", "B16", "decode_first_fair2_rr")
DISPLAY = {"prefill_first": "A", "decode_first_fair2_rr": "C"}


def analyze(payload):
    groups = defaultdict(list)
    decomposition = []
    for run in payload["runs"]:
        s = run["summary"]
        groups[s["pressure"], s["policy"]].append(run)
        records = run["requests"]
        queue = fmean(r["first_prefill_time"] - r["admission_time"] for r in records)
        processing = fmean(r["first_token_time"] - r["first_prefill_time"] for r in records)
        outputs = fmean(r["completion_time"] - r["first_token_time"] for r in records)
        if not math.isclose(queue + processing + outputs, s["request_latency_mean_s"], abs_tol=1e-9):
            raise ValueError("request latency decomposition failed")
        blocked_steps = {e["step_index"] for e in run["reserve_deferral_events"]}
        decomposition.append(dict(pressure=s["pressure"], policy=s["policy"], repeat=s["repeat"],
            admission_to_first_prefill_mean_s=queue, first_prefill_to_first_token_mean_s=processing,
            first_token_to_completion_mean_s=outputs, ttft_mean_s=s["ttft_mean_s"],
            request_latency_mean_s=s["request_latency_mean_s"],
            deferral_steps_decode=sum(run["steps"][i]["phase"] == "decode" for i in blocked_steps),
            deferral_steps_prefill=sum(run["steps"][i]["phase"] == "prefill" for i in blocked_steps)))
    stats = []
    for (level, policy), runs in groups.items():
        for key in runs[0]["summary"]:
            values = [r["summary"][key] for r in runs]
            if not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in values):
                continue
            stats.append(dict(pressure=level, policy=policy, metric=key, samples=len(values),
                              mean=fmean(values), sample_stddev=stdev(values) if len(values) > 1 else None,
                              min=min(values), max=max(values)))
    comparisons = []
    request_differences = []
    for level in payload["manifest"]["config"]["pressures"]:
        baseline = groups[level, "B0"]
        if not baseline:
            raise ValueError("B0 is required for comparison")
        for policy in ("B4", "B8", "B16"):
            if not groups[level, policy]:
                raise ValueError(f"missing threshold: {level}/{policy}")
            row = dict(pressure=level, policy=policy)
            for key in ("preemptions", "recomputed_tokens", "ttft_mean_s", "ttft_p50_s", "ttft_p95_s",
                        "itl_p95_s", "request_latency_mean_s", "request_latency_p95_s", "output_token_throughput"):
                base = fmean(r["summary"][key] for r in baseline)
                value = fmean(r["summary"][key] for r in groups[level, policy])
                row[key + "_difference"] = value-base
                row[key + "_change_percent"] = (value/base-1)*100 if base else None
            comparisons.append(row)
            bases = {r["summary"]["repeat"]: {x["request_id"]: x for x in r["requests"]} for r in baseline}
            for run in groups[level, policy]:
                repeat = run["summary"]["repeat"]
                for request in run["requests"]:
                    rid = request["request_id"]
                    base = bases[repeat][rid]
                    request_differences.append(dict(pressure=level, policy=policy, repeat=repeat, request_id=rid,
                        first_prefill_wait_difference_s=(request["first_prefill_time"] - request["admission_time"])
                            - (base["first_prefill_time"] - base["admission_time"]),
                        ttft_difference_s=(request["first_token_time"] - request["admission_time"])
                            - (base["first_token_time"] - base["admission_time"]),
                        output_span_difference_s=(request["completion_time"] - request["first_token_time"])
                            - (base["completion_time"] - base["first_token_time"]),
                        completion_latency_difference_s=(request["completion_time"] - request["admission_time"])
                            - (base["completion_time"] - base["admission_time"])))
    return dict(statistics=stats, comparisons=comparisons, latency_decomposition=decomposition,
                request_differences=request_differences,
                note="Changes are ratios of three-run means, not significance tests or marginal causal costs.")


def write_report(directory, report_path):
    directory = Path(directory)
    payload = verify_results(directory)
    if set(payload["manifest"]["config"]["policies"]) != set(LABELS):
        raise ValueError("report requires all four thresholds and both A/C references")
    config = payload["manifest"]["config"]
    if config["repeats"] != 3 or set(config["pressures"]) != {"low", "medium", "high"}:
        raise ValueError("this report template requires three pressures and three repeats")
    for run in payload["runs"]:
        expected = payload["manifest"]["policy_configurations"][run["summary"]["policy"]]
        for key in ("scheduling_policy", "decode_queue_policy", "max_consecutive_decode_steps",
                    "kv_admission_policy", "kv_reserve_blocks"):
            if run["summary"][key] != expected[key]:
                raise ValueError("measured policy configuration differs from manifest")
    derived = analyze(payload)
    original = Path(config["workload_from"]) if config.get("workload_from") else None
    if original and original.exists():
        previous = json.loads(original.read_text())
        if any(rows != previous["workloads"][level] for level, rows in payload["manifest"]["workloads"].items()):
            raise ValueError("original and measured workload Token IDs differ")
        derived["original_workload_sha256"] = hashlib.sha256(original.read_bytes()).hexdigest()
    (directory / "admission_analysis.json").write_text(json.dumps(derived, indent=2, ensure_ascii=False,
                                                                  allow_nan=False) + "\n")
    for key, rows in derived.items():
        if isinstance(rows, list):
            export_csv(directory / ("admission_" + key + ".csv"), rows)
    source = Path(__file__).read_bytes()
    (directory / "admission_analysis_source.py").write_bytes(source)
    (directory / "admission_analysis_source.sha256").write_text(hashlib.sha256(source).hexdigest() + "\n")
    groups = defaultdict(list)
    for run in payload["runs"]:
        groups[run["summary"]["pressure"], run["summary"]["policy"]].append(run)
    levels = ("low", "medium", "high")
    lines = []
    def add(s=""):
        lines.append(s)
    def table(headers, rows):
        add("| " + " | ".join(headers) + " |")
        add("|" + "|".join("---" for _ in headers) + "|")
        for row in rows:
            add("| " + " | ".join(str(v) for v in row) + " |")
        add()
    def mean(level, policy, key):
        return fmean(r["summary"][key] for r in groups[level, policy])
    def formatted(level, policy, key):
        values = [r["summary"][key] for r in groups[level, policy]]
        if any(v is None for v in values):
            return "—"
        return f"{fmean(values):.3f} ± {stdev(values):.3f}"
    def label(policy):
        return DISPLAY.get(policy, policy)
    unit_log = directory / "unittest.log"
    match = re.search(r"Ran (\d+) tests", unit_log.read_text()) if unit_log.exists() else None
    test_result = f"{match[1]}项通过" if match and "\nOK\n" in unit_log.read_text() else "见单元测试验证日志"
    add("# KV 容量感知接纳与静态余量实验（第六阶段第二步）\n")
    add("日期：2026-11-02（Asia/Shanghai）；分支feat/mixed-load-scheduler。")
    add(f"基础HEAD：`{payload['manifest']['git_head']}`。本阶段未Commit/Push。\n")
    add("## 结论\n")
    add("建议保留为**默认关闭的实验性策略**，不建议据此默认启用或宣布通用最佳阈值。"
        "低/中压力下4/8/16阈值未触发保护；高压力下确有抢占与重算减少，但保留阈值并非"
        "未来扩页的完整预留，峰值仍可达到105块。完整结果包括全部阈值、A/C参考组和三个重复。\n")
    table(["高压配置", "抢占均值", "相对B0减少", "重算Token均值", "相对B0减少", "TTFT mean秒", "完成延迟mean秒"], [
        [p, f"{mean('high',p,'preemptions'):.0f}",
         f"{(1-mean('high',p,'preemptions')/mean('high','B0','preemptions'))*100:.2f}%",
         f"{mean('high',p,'recomputed_tokens'):.0f}",
         f"{(1-mean('high',p,'recomputed_tokens')/mean('high','B0','recomputed_tokens'))*100:.2f}%",
         f"{mean('high',p,'ttft_mean_s'):.3f}", f"{mean('high',p,'request_latency_mean_s'):.3f}"]
        for p in ("B0", "B4", "B8", "B16")])
    add("这些是本固定负载的三次均值比较，不是统计显著性或普遍性能保证；等待转移的分析见后文。\n")
    add("高压相对B0变化（延迟正值表示更慢，吞吐正值表示增加；百分比为三次均值之比）：\n")
    table(["阈值", "TTFT mean变化", "TTFT P50变化", "TTFT P95变化", "ITL P95变化", "完成延迟P95变化", "吞吐变化"], [
        [r["policy"]] + [f"{r[k+'_change_percent']:+.2f}%" for k in
            ("ttft_mean_s", "ttft_p50_s", "ttft_p95_s", "itl_p95_s", "request_latency_p95_s", "output_token_throughput")]
        for r in derived["comparisons"] if r["pressure"] == "high"])
    add("## 源码审查与设计\n")
    add("Config新增kv_admission_policy=none|reserve（默认none），kv_reserve_blocks（默认0）；"
        "非法策略以及负数、浮点数、布尔值、字符串等阈值抛ValueError，原dtype兼容逻辑保留。\n")
    add("Scheduler仅增加_can_admit_prefill和首次allocate前的一次判断。can_allocate先验证"
        "真实容量和可匹配前缀，返回命中的完整块数；余量估算为：\n")
    add("`预计free减少量 = seq.num_blocks − 命中前缀中已经活跃的共享块数`\n\n"
        "`允许首次接纳 ⇔ free_blocks − 预计free减少量 >= kv_reserve_blocks`\n")
    add("活跃共享块只增加引用计数，不减少free；空闲缓存命中重新启用后会离开free，"
        "所以不能简单用num_blocks−所有命中块数。hash链只沿can_allocate已确认的前缀读取，"
        "没有修改BlockManager分配、释放、缓存索引或引用计数，也没有收益评分或扫描后续请求。\n")
    add("物理KV张量在初始化时已预分配105块；这里保护的是池内可用Block槽位，"
        "不是每次请求额外向CUDA申请的显存字节，也没有缩小实际张量容量。\n")
    add("保护仅作用于新请求的首次Block分配；Token预算不够组成同批时原有break先发生，"
        "不将其误计为reserve拒绝。已开始的Chunked Prefill持有完整Prompt块，直接继续；"
        "被抢占请求保留原输出，其num_completion_tokens>0，恢复不受reserve阻挡。"
        "waiting FIFO和running RR顺序不变，批次仍不混合Prefill/Decode。\n")
    add("保护拒绝新请求时，原schedule阶段回退尝试Decode；如果已经选择了Prefill请求，"
        "先完成该纯Prefill批次。running为空时绕过余量，允许至少一个请求前进。"
        "Decode在len(seq)%256==1且无空闲块时仍按原规则抢占未选择的running队尾，"
        "无其他候选则抢占自身；若没有Decode批次，原调度会重试Prefill，此时running为空。"
        "实际容量不够则仍报Cannot schedule，不返回空成功、不绕过BlockManager断言。\n")
    add("该保护是软余量：Decode继续增长、恢复请求及空running前进例外可消费余量，"
        "并不保证运行期间始终free>=reserve。max_num_seqs=4仍是批次上限而非全局running数量上限。\n")
    add("修改文件：nanovllm/config.py、nanovllm/engine/scheduler.py、benchmarks/kv_pressure_bench.py；"
        "新增tests/test_scheduler_admission.py与tests/test_kv_admission_analysis.py、扩展tests/test_kv_pressure.py，新增本离线分析工具与报告，"
        "更新benchmarks/README.md。ModelRunner、CUDA Kernel、BlockManager、Sequence没有修改；"
        "工作区原有llm_engine.py注释、smoke_test.py及个人笔记不属于本次修改。\n")
    add("## 实验设计与测量\n")
    add("A=prefill_first/head_first；B0/B4/B8/B16=prefill_first/round_robin + reserve=0/4/8/16；"
        "C=decode_first/fair2/round_robin。A/C默认none。B0与none行为相同，CPU完整轨迹测试已覆盖。\n")
    add("RTX 4060 Laptop 8GB，Qwen3-0.6B，bfloat16，105个实际KV Block，block_size=256，"
        "TP=1、eager=True、max_model_len=768、max_num_batched_tokens=256、max_num_seqs=4、"
        "gpu_memory_utilization=0.65。torch=2.7.1+cu126/CUDA12.6，未安装/升级依赖。\n")
    add("使用--workload-from复用上一阶段manifest中完全相同的Prompt Token IDs和长度，"
        "低/中/高为16/48/80请求，248/504 Token Prompt交替，固定输出16 Token且ignore_eos=True，"
        "请求1/3共享一个完整256 Token页，所有请求t=0提交。不是缓存命中率优化实验。\n")
    total_requests = sum(r["summary"]["completed_requests"] for r in payload["runs"])
    total_tokens = sum(r["summary"]["total_output_tokens"] for r in payload["runs"])
    add(f"共{len(payload['runs'])}个独立进程，每组合3次；完成{total_requests}请求、{total_tokens}输出Token。"
        "每组相同两轮暖机，清理空闲Prefix元数据而不改物理池/free列表/引用。加载、初始化和暖机"
        "排除计时，所有正式窗口编译计数为0。重新实测A/C，不将旧日期的参考延迟混入本次比较。\n")
    add("进程顺序按重复轮换压力和策略；6策略仅3重复不能完全平衡6个执行位置，"
        "未锁频、未隔离WSL其他主机负载，不能用微小吞吐差异宣称普遍收益。"
        "A/C/B0也使用同一观察器。保留所有54份原始JSON/日志，没有挑选最好运行。\n")
    add("沿用真实append_token后的perf_counter记录；GPU采样结果tolist到CPU后才append。"
        "TTFT从实际入队计起；ITL为相邻输出时间差；完成延迟从入队至完成；吞吐为全部输出/正式窗口。"
        "首段Chunked Prefill未产生输出不记录TTFT。重算只算抢占后执行区间与历史有效KV的交集。"
        "纯边际抢占时间仍不可识别，不使用Token比例或未验证成本模型摊分。\n")
    add("以下延迟为秒、吞吐为Token/s。每个值为3次逐运行统计的均值±样本标准差；"
        "各次P95/P99先按type7排序后(n−1)*p插值，再跨运行求均值，不合并运行样本。"
        "每次TTFT样本16/48/80，ITL样本240/720/1200。min/max保存在CSV/JSON。\n")
    add("## 全部实测：资源与吞吐\n")
    table(["压力", "配置", "抢占", "重算Token", "Prefix复用Token", "峰值块", "Decode平均Batch", "输出Token/s"], [
        [level, label(p)] + [f"{mean(level,p,k):.3f}" if k=="decode_mean_batch_size" else f"{mean(level,p,k):.0f}"
                            for k in ("preemptions","recomputed_tokens","prefix_hit_tokens","peak_used_blocks","decode_mean_batch_size")]
        + [formatted(level,p,"output_token_throughput")] for level in levels for p in LABELS])
    add("## 全部实测：请求与Token延迟\n")
    table(["压力", "配置", "TTFT P50", "TTFT P95", "ITL P50", "ITL P95", "ITL P99", "ITL max", "完成延迟P95"], [
        [level,label(p)] + [formatted(level,p,k) for k in ("ttft_p50_s","ttft_p95_s","itl_p50_s", "itl_p95_s", "itl_p99_s", "itl_max_s", "request_latency_p95_s")]
        for level in levels for p in LABELS])
    add("## Reserve拒绝次数与关联等待\n")
    add("reserve_deferrals是保护函数返回False的尝试次数（重复探测分别计数），不是新请求数量；"
        "每请求首次拒绝至首次真实分配Block合并为一个episode。它包含之后的FIFO、物理容量、"
        "恢复工作与其他调度等待，不是reserve造成的纯边际延迟。总等待是多个请求episode的和，"
        "可重叠，不能当作运行Wall Time。没有拒绝时P50/P95为—而非伪造0时长样本。\n")
    add("直接推迟计数只包含被保护函数检查并拒绝的队首请求。排在其后的FIFO请求也可能间接"
        "等待，但不会重复计作直接reserve拒绝；整体影响由所有请求的首次分配等待、TTFT和"
        "完成延迟体现。另存同重复/同request_id相对B0的逐请求差值，包含不同进程运行波动，"
        "不把每一个微小正差值都判定为reserve因果损失。\n")
    table(["高压配置","直接推迟请求数","拒绝尝试数","关联等待总秒","关联等待P50秒","关联等待P95秒"], [
        [p, f"{mean('high',p,'reserve_deferred_requests'):.0f}", f"{mean('high',p,'reserve_deferrals'):.0f}"]+
        [formatted('high',p,k) for k in ("reserve_wait_total_s","reserve_wait_p50_s","reserve_wait_p95_s")]
        for p in ("B0","B4","B8","B16")])
    add("首次Prefill时间记录首次Block分配完成，早于真正GPU前向开始；它用于划分接纳前队列等待。"
        "低/中压力下各阈值的拒绝次数均为0，和B0产生相同抢占/重算计数；延迟差包含运行波动"
        "与检查开销，不能据此宣称reserve带来收益。\n")
    add("## 是否只是把等待转移到Prefill？\n")
    add("以每请求平均完成延迟作可加分解：TTFT=接纳前等待+首次Prefill至首Token；"
        "完成延迟=TTFT+首Token至完成时间。后一个区间包括Decode服务等待、恢复及其他Prefill干扰，"
        "不是纯Decode GPU时间。下表为三个运行内逐请求均值的均值。\n")
    table(["高压配置","首次分配前等待","首次分配至首Token","首Token至完成","TTFT mean","完成延迟mean"], [
        [p] + [f"{fmean(r[k] for r in derived['latency_decomposition'] if r['pressure']=='high' and r['policy']==p):.3f}"
               for k in ("admission_to_first_prefill_mean_s","first_prefill_to_first_token_mean_s", "first_token_to_completion_mean_s", "ttft_mean_s", "request_latency_mean_s")]
        for p in ("B0","B4","B8","B16")])
    add("必须同时读上述分解和尾延迟表：reserve会让部分请求更晚首次接纳，不能只展示更短ITL。"
        "真实重算Token减少证明它并非只有时间标签变化，但减少了冗余计算不保证所有请求更快、"
        "TTFT更好或吞吐有统计可靠改善。A保留更低中位ITL同时有长尾，C避免抢占但TTFT更晚，"
        "两组参考也没有统一支配全部指标。\n")
    add("## 兼容性、失败路径与验收\n")
    add(f"原96项测试完整保留；当前测试：{test_result}。新增覆盖默认none/非零忽略、reserve=0完整旧轨迹、"
        "阈值边界、活跃共享/空闲命中、首次推迟后Decode推进、无running例外、已持块Chunked Prefill、"
        "抢占恢复及保留输出、扩页抢占兜底、RR顺序、fair2失败尝试不重置计数、真实物理不足报错、"
        "拒绝episode合并和不把物理不足误记为reserve拒绝。\n")
    add("所有运行完成率100%，物理张量/Runner配置/BlockManager均105块，退出前队列空、used=0、"
        "ref_count_sum=0、free=105。每次调度/后处理保留全队列所有权和引用审计；原Token守恒、"
        "重算重放、Prefix计数、真实Token时间与CSV/JSON一致性检查继续通过。"
        "物理容量无法满足时抛RuntimeError；超时120秒、总步数10000、连续100步无新输出/新KV进度"
        "会报错并保存失败状态/已有结果，不把失败数据当成功样本。\n")
    add("reserve大于容量不当作配置错误：无running时允许前进，运行时余量可突破；如果单请求"
        "所需物理块本身超过容量，余量例外也不能绕过真实分配失败。抢占恢复优先于保护新请求，"
        "可能再次消耗余量；这也是其不能保证零抢占的原因。\n")
    add("## 数据与复现\n")
    add(f"正式目录：`{directory}`，由Git忽略。原results_kv_pressure保持原样，试采"
        "results_kv_admission_pilot未混入正式数据。manifest包含旧工作负载精确Token IDs、6组"
        "策略配置、参数、commit与源码hash；保存启动时Config/Scheduler/BlockManager/Runner"
        "源码快照、54份worker JSON/日志及instrumentation_source.py。\n")
    add("每档raw_requests.csv、summary.csv/JSON、steps.csv、preemption_events.csv、allocation_events.csv"
        "继续保留；新增reserve_deferral_events.csv和reserve_wait_events.csv，逐请求统计随raw_requests保存。"
        "总summary.csv、pressure_runs.json、cross_run_statistics.csv保留全部运行。"
        "本脚本另写admission_analysis.json、admission_statistics.csv、admission_comparisons.csv、"
        "admission_latency_decomposition.csv、admission_request_differences.csv和分析源码快照，"
        "可重算比较及等待分解。\n")
    add("```bash\n.venv/bin/python -m benchmarks.kv_pressure_bench \\\n  --workload-from results_kv_pressure/manifest.json --expected-blocks 105 \\\n  --policies prefill_first B0 B4 B8 B16 decode_first_fair2_rr \\\n  --repeats 3 --output-dir results_kv_admission_20261102\n"
        ".venv/bin/python -m benchmarks.kv_pressure_bench --verify results_kv_admission_20261102\n"
        ".venv/bin/python -m unittest discover -s tests -v\n"
        ".venv/bin/python -m benchmarks.kv_admission_analysis --results results_kv_admission_20261102\n"
        "git diff --check\n```\n")
    add("## 限制与保留建议\n")
    add("只有单GPU/模型、固定突发、两个Prompt长度和16输出Token、单个共享前缀对、3重复。"
        "静态4/8/16与本例临界增长需求相关，不推导通用最优阈值。FIFO队首阻塞不扫描后续可能便宜的"
        "请求；未评分Prefix收益或优化缓存命中率。未来输出跨更多页、恢复更频繁、不同突发/到达"
        "模式、共享比例或容量可能效果更差。同步单线程只能在step间注入，不是生产异步服务。\n")
    add("观察器、资源审计、hash读取和拒绝事件记录有开销，包含在实测Wall内；未单独量化或"
        "扣除开销，不把不同进程的微小波动归因于算法。没有纯GPU重算边际耗时或无抢占反事实。"
        "保留none为默认，reserve作为可选实验控制；静态阈值仅部分有效，不自动加入动态规则、"
        "复杂Cost-aware调度或缓存收益评分。未来是否扩展须用新的负载和预先确定的验收指标验证。\n")
    Path(report_path).write_text("\n".join(lines), encoding="utf-8")
    return derived


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="results_kv_admission_20261102")
    parser.add_argument("--report", default="benchmarks/kv_admission_analysis_zh.md")
    args = parser.parse_args()
    write_report(args.results, args.report)
    print(f"Wrote verified analysis: {args.report}")


if __name__ == "__main__":
    main()
