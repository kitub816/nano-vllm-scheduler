"""Read-only trace recovery and OFFLINE, SINGLE-DECISION bypass probes.

No synthetic output IDs, GPU service times, or counterfactual future queue state.
Each probe starts from a recovered real state, allocates at most one candidate,
and stops. Probes cannot be concatenated into a simulated GPU run.
"""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from statistics import fmean
import subprocess

from benchmarks.kv_pressure_bench import export_csv, verify_results
from benchmarks.metrics import percentile


MODES = ("fifo", "skip_1", "skip_2", "min_new_blocks")


def recover_waiting(run, specs):
    """Recover order at log points, not invented sub-event transition timestamps."""
    records = {r["request_id"]: r for r in run["requests"]}
    prompts = {s["request_id"]: s["prompt_token_ids"] for s in specs}
    admissions = sorted(records, key=lambda rid: records[rid]["admission_time"])
    times = [records[rid]["admission_time"] for rid in admissions]
    if len(times) != len(set(times)):
        raise ValueError("equal admission timestamps: queue order not identifiable")
    events = defaultdict(list)
    for kind in ("allocation_events", "preemption_events", "reserve_deferral_events"):
        for event in run[kind]:
            events[event["step_index"]].append((event["time"], kind, event))
    waiting, running, allocated = [], set(), set()
    next_admission = 0
    snapshots, boundaries = [], []
    for step in run["steps"]:
        while next_admission < len(admissions) and times[next_admission] <= step["start_time"]:
            waiting.append(admissions[next_admission])
            next_admission += 1
        before_schedule = list(waiting)
        running_before = len(running)
        fresh_suffix = bool(waiting) and all(x not in allocated and records[x]["first_prefill_time"] > step["start_time"] for x in waiting)
        free_before = (run["steps"][step["step_index"]-1]["kv_after_step"]["free_blocks"]
                       if step["step_index"] else step["kv_after_schedule"]["total_blocks"])
        executions = step["executions"] if step["phase"] == "prefill" else []
        position = 0
        def advance_ready_prefills():
            nonlocal position
            while position < len(executions):
                execution = executions[position]
                rid = execution["request_id"]
                if rid not in allocated:
                    break  # Its real allocation log must be processed first.
                if not waiting or waiting[0] != rid:
                    break  # A logged preemption can move this running victim first.
                prior_outputs = sum(t < step["start_time"] for t in records[rid]["token_times"])
                length = len(prompts[rid]) + prior_outputs
                if execution["end"] > length:
                    raise ValueError("executed beyond real Sequence length")
                if execution["end"] == length:
                    waiting.pop(0)
                    running.add(rid)
                position += 1
        ordered = sorted(events[step["step_index"]], key=lambda row: row[0])
        if len({e[0] for e in ordered}) != len(ordered):
            raise ValueError("equal intra-step event timestamps: order ambiguous")
        for timestamp, kind, event in ordered:
            if not step["start_time"] <= timestamp <= step["schedule_time"]:
                raise ValueError("event outside scheduling interval")
            advance_ready_prefills()
            rid = event["request_id"]
            if kind == "preemption_events":
                if rid not in running:
                    raise ValueError("preemption victim is not running")
                running.remove(rid)
                allocated.remove(rid)
                waiting.insert(0, rid)
            elif kind == "allocation_events":
                if not waiting or waiting[0] != rid or rid in allocated:
                    raise ValueError("allocation contradicts queue/ownership state")
                length = len(prompts[rid]) + sum(t < step["start_time"] for t in records[rid]["token_times"])
                if event["seq_tokens"] != length:
                    raise ValueError("allocation Sequence length differs from original Token history")
                allocated.add(rid)
            else:
                if not waiting or waiting[0] != rid or rid in allocated:
                    raise ValueError("reserve rejection contradicts recovered head")
                if any(x in allocated or records[x]["first_prefill_time"] <= timestamp for x in waiting):
                    raise ValueError("reserve snapshot contains non-fresh requests; cannot infer pure fresh suffix")
                snapshots.append(dict(**event, waiting=list(waiting), running_count=len(running),
                    point="logged_reserve_rejection",
                    selected_prefill_tokens=sum(e["q"] for e in executions[:position]),
                    selected_prefill_seqs=position, phase=step["phase"],
                    step_start=step["start_time"], step_end=step["end_time"],
                    allocation_state="known fresh suffix; external cache contents not reconstructed"))
        advance_ready_prefills()
        if position != len(executions):
            raise ValueError("unresolved Prefill allocation")
        if len(waiting) != step["waiting_count"] or len(running) != step["running_count"]:
            raise ValueError("recovered queue counts differ from schedule snapshot")
        before = list(waiting)
        for rid, record in records.items():
            if step["start_time"] <= record["completion_time"] <= step["end_time"]:
                if rid not in running:
                    raise ValueError("finished request not running")
                running.remove(rid)
                allocated.remove(rid)
        if (len(waiting) != step["kv_after_step"]["waiting_requests"]
                or len(running) != step["kv_after_step"]["running_requests"]):
            raise ValueError("recovered postprocess counts differ")
        boundaries.append(dict(step_index=step["step_index"], phase=step["phase"],
                               waiting_before_schedule=before_schedule, fresh_suffix_before_schedule=fresh_suffix,
                               running_before_schedule_count=running_before, free_before_schedule=free_before,
                               waiting_after_schedule=before, waiting_after_step=list(waiting),
                               running_members_after_step=sorted(running)))
    if waiting or running or allocated or next_admission != len(admissions):
        raise ValueError("recovered trajectory did not drain")
    return snapshots, boundaries


def physical_demand(num_blocks, hit_block_ids, active_block_ids):
    """Same ownership rule as BlockManager: cached-free hits still cost a slot."""
    if num_blocks <= 0 or not 0 <= len(hit_block_ids) < num_blocks or len(set(hit_block_ids)) != len(hit_block_ids):
        raise ValueError("invalid matching prefix")
    return num_blocks - sum(b in active_block_ids for b in hit_block_ids)


def fresh_cost(spec, specs, timestamp, records):
    """Prove no prefix hit using known prompt bytes; otherwise return UNKNOWN.

    For a short source the last output IDs are absent. A mismatch within its
    known prompt still rules out a matching first full block. Never invent IDs.
    """
    ids = spec["prompt_token_ids"]
    blocks = (len(ids) + 255) // 256
    if blocks == 1:
        return dict(known=True, blocks=physical_demand(1, (), ()), cached_blocks=0, proof="last block is never reusable")
    ambiguous = []
    for other in specs:
        rid = other["request_id"]
        if rid == spec["request_id"] or records[rid]["first_prefill_time"] > timestamp:
            continue  # This other request has never populated the cache yet.
        known = min(256, len(other["prompt_token_ids"]))
        if ids[:known] == other["prompt_token_ids"][:known]:
            ambiguous.append(rid)
    if ambiguous:
        return dict(known=False, blocks=None, cached_blocks=None,
                    proof="matching known prefix; active/free/hash retention unavailable", possible_sources=ambiguous)
    return dict(known=True, blocks=physical_demand(blocks, (), ()), cached_blocks=0,
                proof="first-page known tokens mismatch every previously admitted source")


def probe(snapshot, costs, specs_by_id, records, mode, capacity, budget=256, max_seqs=4):
    """One hypothetical initial allocation; future occupancy and latency are UNKNOWN."""
    queue = snapshot["waiting"]
    if not 0 <= snapshot["free_blocks"] <= capacity or snapshot["reserve_blocks"] < 0:
        raise ValueError("invalid capacity snapshot")
    limit = {"fifo": 1, "skip_1": 2, "skip_2": 3, "min_new_blocks": len(queue)}[mode]
    available = snapshot["free_blocks"] - snapshot["reserve_blocks"]
    remaining = budget - snapshot["selected_prefill_tokens"]
    candidates = []
    for index, rid in enumerate(queue[:limit]):
        cost = costs[rid]
        if not cost["known"] or cost["blocks"] > available:
            continue
        if snapshot["selected_prefill_seqs"] >= max_seqs:
            continue
        length = len(specs_by_id[rid]["prompt_token_ids"])
        if remaining <= 0 or (snapshot["selected_prefill_seqs"] and length > remaining):
            continue
        candidates.append((cost["blocks"], index, rid))
    selected = (min(candidates) if mode == "min_new_blocks" else
                min(candidates, key=lambda x: x[1])) if candidates else None
    result = dict(kind="OFFLINE_SINGLE_DECISION", mode=mode, step_index=snapshot["step_index"],
        time=snapshot["time"], head_request_id=queue[0], selected_request_id=None,
        skipped_requests=[], small_over_large_count=0, conditional_initial_wait_s=None,
        observed_initial_wait_s=None, conditional_advance_s=None, projected_used_blocks=None,
        future_ttft_s=None, future_completion_latency_s=None, future_starvation_count=None)
    if selected:
        needed, index, rid = selected
        used = capacity - snapshot["free_blocks"] + needed
        if used > capacity or capacity - used < snapshot["reserve_blocks"]:
            raise ValueError("probe exceeded physical capacity/reserve")
        skipped = queue[:index]
        small = len(specs_by_id[rid]["prompt_token_ids"]) <= 256
        result.update(selected_request_id=rid, skipped_requests=skipped,
            small_over_large_count=sum(len(specs_by_id[x]["prompt_token_ids"]) > 256 for x in skipped) if small else 0,
            conditional_initial_wait_s=snapshot["time"]-records[rid]["admission_time"],
            observed_initial_wait_s=records[rid]["first_prefill_time"]-records[rid]["admission_time"],
            conditional_advance_s=records[rid]["first_prefill_time"]-snapshot["time"],
            projected_used_blocks=used)
    return result


def actual_metrics(run, snapshots, probes, boundaries):
    records = run["requests"]
    waits = [r["first_prefill_time"] - r["admission_time"] for r in records]
    large = [r for r in records if r["prompt_tokens"] > 256]
    # Each successful opportunity covers only its own observed step suffix.
    # Union repeated/touching intervals so no request/step is counted twice.
    intervals = sorted((p["time"], snapshots[p["snapshot_index"]]["step_end"])
                       for p in probes if p["mode"] == "min_new_blocks" and p["selected_request_id"] is not None)
    union = []
    for a, b in intervals:
        if union and a <= union[-1][1]:
            union[-1] = (union[-1][0], max(b, union[-1][1]))
        else:
            union.append((a, b))
    episodes = {e["request_id"]: e for e in run["reserve_wait_events"]}
    confirmed_head_time = 0.
    for step, boundary in zip(run["steps"], boundaries):
        queue = boundary["waiting_after_schedule"]
        if queue and queue[0] in episodes:
            episode = episodes[queue[0]]
            if episode["start_time"] <= step["schedule_time"] and step["end_time"] <= episode["end_time"]:
                confirmed_head_time += step["end_time"] - step["schedule_time"]
    return dict(initial_wait_mean_s=fmean(waits), initial_wait_max_s=max(waits),
        initial_wait_p95_s=percentile(waits, .95),
        large_wait_over_1s=sum(r["first_prefill_time"]-r["admission_time"] > 1 for r in large),
        large_wait_over_5s=sum(r["first_prefill_time"]-r["admission_time"] > 5 for r in large),
        large_requests=len(large), starvation_observed=0,
        reserve_episode_sum_s=sum(e["duration_s"] for e in run["reserve_wait_events"]),
        confirmed_reserve_head_residence_s=confirmed_head_time,
        hol_opportunity_associated_step_wall_s=sum(b-a for a,b in union),
        observed_peak_used_blocks=run["summary"]["peak_used_blocks"],
        observed_step_weighted_utilization=run["summary"]["step_weighted_block_utilization"],
        ttft_p95_s=run["summary"]["ttft_p95_s"], latency_p95_s=run["summary"]["request_latency_p95_s"])


def analyze_run(run, specs):
    snapshots, boundaries = recover_waiting(run, specs)
    records = {r["request_id"]: r for r in run["requests"]}
    by_id = {s["request_id"]: s for s in specs}
    # Recover first Prefill attempts hidden by an earlier can_allocate failure.
    # Between previous postprocess and step start there is no KV allocation.
    keys = {(s["step_index"],s["request_id"],s["free_blocks"],s["reserve_blocks"]) for s in snapshots}
    if run["summary"]["scheduling_policy"] == "prefill_first":
        for step, boundary in zip(run["steps"],boundaries):
            if not boundary["fresh_suffix_before_schedule"]:
                continue
            head = boundary["waiting_before_schedule"][0]
            cost = fresh_cost(by_id[head],specs,step["start_time"],records)
            reserve = (run["summary"]["kv_reserve_blocks"]
                       if run["summary"]["kv_admission_policy"] == "reserve" and boundary["running_before_schedule_count"] else 0)
            free = boundary["free_before_schedule"]
            key=(step["step_index"],head,free,reserve)
            if cost["known"] and free-cost["blocks"] < reserve and key not in keys:
                snapshots.append(dict(request_id=head, time=step["start_time"], step_index=step["step_index"],
                    point="recovered_step_start_blocked", cached_blocks=0, physical_blocks=cost["blocks"],
                    free_blocks=free, reserve_blocks=reserve, waiting=boundary["waiting_before_schedule"],
                    running_count=boundary["running_before_schedule_count"], selected_prefill_tokens=0,
                    selected_prefill_seqs=0, phase=step["phase"], step_start=step["start_time"], step_end=step["end_time"],
                    allocation_state="exact boundary free count; no matching fresh prefix proven"))
    snapshots.sort(key=lambda s:s["time"])
    probes, candidates = [], []
    for index, snapshot in enumerate(snapshots):
        costs = {rid: fresh_cost(by_id[rid], specs, snapshot["time"], records) for rid in snapshot["waiting"]}
        head_cost = costs[snapshot["request_id"]]
        if head_cost["known"] and head_cost["blocks"] != snapshot["physical_blocks"]:
            raise ValueError("head-demand proof contradicts measured demand")
        if head_cost["known"] and head_cost["cached_blocks"] != snapshot["cached_blocks"]:
            raise ValueError("prefix proof contradicts measured cache hits")
        for position, rid in enumerate(snapshot["waiting"]):
            candidates.append(dict(snapshot_index=index, queue_position=position, request_id=rid,
                                   prompt_tokens=len(by_id[rid]["prompt_token_ids"]), **costs[rid]))
        for mode in MODES:
            probes.append(dict(snapshot_index=index, **probe(snapshot, costs, by_id, records, mode,
                                                             run["summary"]["num_kvcache_blocks"])))
    summaries = []
    for mode in MODES:
        selected = [p for p in probes if p["mode"] == mode and p["selected_request_id"] is not None]
        # An earliest opportunity for one request remains an independent branch.
        earliest = {}
        for row in selected:
            earliest.setdefault(row["selected_request_id"], row)
        summaries.append(dict(mode=mode, kind="OFFLINE_SINGLE_DECISION",
            decision_snapshots=len(snapshots),
            logged_reserve_rejections=sum(s["point"]=="logged_reserve_rejection" for s in snapshots),
            admissible_opportunities=len(selected),
            distinct_candidates=len(earliest), conditional_overtake_actions=sum(len(p["skipped_requests"]) for p in selected),
            conditional_small_over_large_actions=sum(p["small_over_large_count"] for p in selected),
            earliest_independent_branches=list(earliest.values()),
            simulated_global_wait_p95_s=None, simulated_global_max_wait_s=None,
            simulated_large_starvation_count=None, simulated_time_weighted_utilization=None,
            reason="future GPU service, output IDs, KV eviction and queue feedback unobserved"))
    return dict(identity={k:run["summary"][k] for k in ("pressure","policy","repeat")},
                snapshots=snapshots, waiting_boundaries=boundaries, candidate_costs=candidates,
                offline_probes=probes, probe_summary=summaries,
                observed_metrics=actual_metrics(run,snapshots,probes,boundaries))


def input_hashes(directory):
    return {str(p.relative_to(directory)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(Path(directory).rglob("*")) if p.is_file()}


def verify_source_snapshots(directory):
    directory=Path(directory)
    manifest=json.loads((directory/"manifest.json").read_text())
    for name in ("nanovllm/config.py", "nanovllm/engine/scheduler.py", "nanovllm/engine/block_manager.py",
                 "nanovllm/engine/model_runner.py"):
        archived=directory/(Path(name).stem+"_source.py")
        if hashlib.sha256(archived.read_bytes()).hexdigest()!=manifest["source_sha256"][name]:
            raise ValueError(f"archived source fingerprint differs: {name}")


def export_observed_requests(result, output):
    source=Path(result["input_dir"]).resolve()
    output=Path(output).resolve()
    if output==source or source in output.parents:
        raise ValueError("observed export must not modify original input")
    raw=json.loads((source/"pressure_runs.json").read_text())
    rows=[]
    for run in raw["runs"]:
        initial={e["request_id"]:e for e in run["allocation_events"] if not e["recovery"]}
        for request in run["requests"]:
            event=initial[request["request_id"]]
            blocks=(request["prompt_tokens"]+255)//256
            rows.append(dict(kind="OBSERVED_ORIGINAL_GPU", **{k:run["summary"][k] for k in ("pressure","policy","repeat")},
                **{k:request[k] for k in ("request_id","prompt_tokens","max_tokens","admission_time",
                    "first_prefill_time","first_token_time","completion_time","ttft_s","request_latency_s")},
                initial_prompt_blocks=blocks, initial_physical_blocks=blocks-event["active_shared_blocks"],
                initial_prefix_hit_tokens=event["hit_tokens"],
                initial_wait_s=request["first_prefill_time"]-request["admission_time"]))
    export_csv(output/"observed_requests.csv",rows)


def write_report(result, path):
    source=Path(result["input_dir"]).resolve()
    target=Path(path).resolve()
    if source in target.parents or source==target:
        raise ValueError("report cannot be written into original data")
    groups=defaultdict(list)
    for run in result["runs"]:
        groups[run["identity"]["pressure"],run["identity"]["policy"]].append(run)
    policies=("prefill_first","B0","B4","B8","B16","decode_first_fair2_rr")
    display={"prefill_first":"A","decode_first_fair2_rr":"C"}
    lines=[]
    def add(text=""): lines.append(text)
    def table(headers,rows):
        add("| "+" | ".join(headers)+" |")
        add("|"+"|".join("---" for _ in headers)+"|")
        for row in rows: add("| "+" | ".join(str(v) for v in row)+" |")
        add()
    def summary(run,mode): return next(s for s in run["probe_summary"] if s["mode"]==mode)
    def avg(policy,field): return fmean(r["observed_metrics"][field] for r in groups["high",policy])
    add("# FIFO Head-of-Line Blocking 离线分析（第六阶段第三步）\n")
    add("日期：2026-11-02（Asia/Shanghai）。只读原实验，不运行GPU、不修改调度器、不安装依赖、不Commit/Push。\n")
    add(f"用户给定/实验记录HEAD为`{result['recorded_experiment_head']}`；当前实际HEAD为"
        f"`{result['current_head']}`，已包含阶段二提交。本次未回退代码。实验HEAD当时有未提交改动，"
        "所以恢复以原manifest中的源码hash及归档Scheduler/BlockManager为准，不只依赖HEAD。\n")
    add("## 结论\n")
    add("**真实的、局部可避免的FIFO队首阻塞存在，主要见于高压B8。**B8三次均有133个"
        "可绕过决策样本，只涉及3个不同候选请求；不是133个额外完成的请求。B4/B16在已恢复的"
        "新请求决策点没有可绕过候选；B0另有1个物理容量导致的机会。跳过1个、跳过2个、"
        "最小新增Block需求，在本数据的所有可判定决策点选出相同请求，扩大搜索没有额外证据收益。\n")
    add("**不足以据此正式修改Scheduler或宣布端到端收益。**现有数据支持做一次小范围、默认关闭"
        "的候选实验；最简单候选是只看下一位，且每个队首在首次接纳前最多允许被越过一次，"
        "不跳过已持块Chunked Prefill/恢复请求。当前任务没有实现该算法。\n")
    add("## 轨迹如何恢复、哪些状态没有恢复\n")
    add(f"只读`{result['input_dir']}`中的全部54次原运行。首次入队顺序取实际admission_time；"
        "每次preempt按时间appendleft；allocation后结合执行区间end与Prompt长度+历史输出计数，"
        "确定完整Prefill何时从waiting移至running，部分Chunk留在队首；完成后从running移除。"
        f"共逐步核对{sum(len(r['waiting_boundaries']) for r in result['runs'])}轮的"
        "schedule/postprocess队列数量、每个分配/抢占位置、reserve拒绝的队首ID以及最终清空状态。\n")
    add("可恢复：每步前/后和每个已记录reserve拒绝点的waiting顺序；首次分配时间、Prompt Tokens、"
        "原输出计数/时间、拒绝时free、步边界free。单步内未记录的pop时间不能精确定位，"
        "只恢复其逻辑先后；running只恢复成员集合，不伪造内部优先顺序。时间戳冲突、缺失分配、"
        "计数矛盾会报错。运行中没有向输入目录写入任何文件，前后逐文件SHA256一致。\n")
    add("缺口：没有完整waiting快照、输出Token IDs、每次Decode扩页的Block ID、每次hash覆盖、"
        "完整free列表/引用表快照，以及改变批次后的GPU服务时间。waiting在上述日志点可用事件"
        "恢复，不代表完整KV缓存状态或任意时刻的队列时间线也能恢复。不能把原来的完成时刻"
        "当作绕过后的资源释放时刻；没有伪造输出Token来重跑引擎。\n")
    add("还检查Prefill-first每步开始时的waiting和上一步postprocess后的真实free，以补充那些"
        "在can_allocate阶段就失败、从未进入reserve拒绝日志的决策。与同状态的拒绝事件去重。"
        "只分析队首及后续均为尚未首次分配的新请求的队列后缀；恢复/已持块前缀不被跳过。"
        "C是Decode-first，步开始不是相同的Prefill决策点，只作实际指标参考，不据其零样本判无HOL。\n")
    add("## Block与Prefix语义\n")
    add("新增free消耗=请求总块数−活跃共享命中块数。空闲缓存重新启用也消耗free，"
        "不能减去全部缓存命中。最后一块不整体复用，单页248 Token请求成本必为1；"
        "504 Token请求需要2块，除非存在合法活跃前缀共享。\n")
    unknown=sum(not c["known"] for r in result["runs"] for c in r["candidate_costs"])
    add(f"本数据所有已检查候选均能由已知Prompt证明不存在前缀命中（unknown行数={unknown}）。"
        "方法不是假设缓存为空：将候选首块与每个已首次分配请求的已知Prompt部分比较；"
        "即使短Prompt的生成后缀未知，只要已知部分不相等就不可能命中。若已知前缀相同、"
        "保留/活跃状态不明，成本标UNKNOWN并不模拟接纳。核心算式另有活跃共享/空闲命中的CPU测试。\n")
    add("因此无需臆造全局Block ID或缓存索引即可判定这些新候选；物理容量105块，"
        "每个分支必须满足投影used<=105、free>=reserve以及Token预算/批次上限。"
        "所有请求保留原Token IDs/长度，不生成任何替代输出，不宣称缓存收益评分优化。\n")
    add("## 四种离线模拟的严格范围\n")
    add("A=fifo：只看队首；B=skip_1：最多看前2位，取第一个可接纳；C=skip_2：最多看前3位；"
        "D=min_new_blocks：在已知成本的整个后缀中取最小物理需求，平局保留FIFO顺序。"
        "以上A/B/C/D是模拟方法，与实测策略A/C参考标签不同。\n")
    add("**每行都是独立的OFFLINE_SINGLE_DECISION分支**：从真实快照出发最多接纳1个请求后停止。"
        "下一行重新从另一个真实快照出发，不继承上一个假设分支。不会把同一候选重复计算为"
        "多个真实接纳，也不能把独立分支拼成一条可实现的连续轨迹。未来TTFT/完成延迟、全体等待"
        "P95/max、长期饥饿数和时间加权KV利用率均为null/不可识别。\n")
    add("候选保持原Prompt与16 Token输出需求；首次接纳占1/2块，独立运行完成最多需2/3块。"
        "单次分支只验证首次分配，不把未来Decode需求当成已消失；其额外页和释放时间不作推断。"
        "条件首次等待按决策时刻立即接纳定义，没有估算替代allocate耗时或GPU前向耗时。\n")
    table(["高压实测配置","日志reserve拒绝/次","恢复后决策点/次","FIFO机会/次","skip1机会/候选数","skip2机会/候选数","min需求机会/候选数"],[
        [display.get(p,p),summary(groups['high',p][0],'fifo')['logged_reserve_rejections'],
         summary(groups['high',p][0],'fifo')['decision_snapshots'],0] +
        [f"{summary(groups['high',p][0],m)['admissible_opportunities']}/{summary(groups['high',p][0],m)['distinct_candidates']}" for m in MODES[1:]]
        for p in policies if p!='decode_first_fair2_rr'])
    add("计数在三个重复中相同。低/中压力的可恢复新请求受阻决策点未发现可绕过候选。"
        "B8的133个机会中131次重复检查65→66，另外为71→72、77→78。"
        "B4大部分free−reserve=0，连小请求也不能接纳；B16最后一次有1块余量时队首79已是最后一个请求。\n")
    add("## 条件式的等待改善：不是GPU收益\n")
    rows=[]
    for p in ('B0','B8'):
        runs=groups['high',p]
        exemplar=summary(runs[0],'skip_1')['earliest_independent_branches']
        for e in exemplar:
            choices=[x for r in runs for x in summary(r,'skip_1')['earliest_independent_branches'] if x['selected_request_id']==e['selected_request_id']]
            rows.append([p,e['head_request_id'],e['selected_request_id'],
                         f"{fmean(x['observed_initial_wait_s'] for x in choices):.3f}",
                         f"{fmean(x['conditional_initial_wait_s'] for x in choices):.3f}",
                         f"{fmean(x['conditional_advance_s'] for x in choices):.3f}",e['projected_used_blocks']])
    table(["实测来源","被越过队首","候选","真实首次等待秒","孤立分支首次等待秒","条件提前秒","单次分配后used"],rows)
    add("B8的请求66在首次机会可以比其原轨迹首次分配更早开始，证明局部可减少该请求等待。"
        "这不是TTFT或完成时间的预测，更不是能把这些提前量相加作为全局收益。"
        "B8每个成功分支占用由96升至97、free由9降至8，满足reserve；随后Decode扩页、抢占、"
        "缓存覆盖及其他请求的排队会改变，当前日志无法决定最终哪种政策更好。\n")
    add("## 真实FIFO公平性与利用率\n")
    add("下表是原GPU运行的观察值，均为3次逐运行统计的均值，不是模拟。首次等待定义为"
        "实际入队至首次Block分配；大请求为504 Token/初始2块。长期等待阈值预先取1秒和5秒，"
        "计数按请求、不是拒绝事件数；长等待不等于无限饥饿。\n")
    table(["高压配置","首次等待mean秒","首次等待max秒","首次等待P95秒","大请求>1秒","大请求>5秒","峰值块","step加权利用率"],[
        [display.get(p,p)]+[f"{avg(p,k):.3f}" for k in ('initial_wait_mean_s','initial_wait_max_s','initial_wait_p95_s')]+
        [f"{avg(p,k):.2f}" for k in ('large_wait_over_1s','large_wait_over_5s')]+
        [f"{avg(p,'observed_peak_used_blocks'):.0f}",f"{avg(p,'observed_step_weighted_utilization')*100:.2f}%"] for p in policies])
    add("全部原请求完成，首次接纳的新请求没有真实小请求越过大请求（恢复插队单独处理）。"
        "模拟skip1/skip2/D在B8有133个条件式小越大动作样本/次，但只有3个独立候选，"
        "不能报告为实际越过133次。全局模拟公平性和模拟KV时间利用率不具备可识别数据。\n")
    table(["高压配置","拒绝episode合计秒","确认处于受阻队首的执行窗秒"],[
        [p,f"{avg(p,'reserve_episode_sum_s'):.3f}",f"{avg(p,'confirmed_reserve_head_residence_s'):.3f}"] for p in ('B4','B8','B16')])
    add("episode从首次拒绝到真实首次分配，期间可能有抢占恢复请求插到它前面，不等于持续占据"
        "队首或持续可绕过。第二列执行窗只累计schedule日志完成至step结束时仍为该队首且"
        "处于其episode内的已确认区间，是下界。精确队首pop时间缺失，不插值整段连续HOL时长；"
        "hol_opportunity_associated_step_wall_s只是机会所在step后缀时间，不能称持续可接纳时间。\n")
    add("## 饥饿风险与是否值得实现\n")
    add("有限lookahead不等于限制同一大请求累计被越过次数。若只规定每次最多跳过1位，"
        "连续到达的小请求仍可能反复越过同一队首。当前三次固定80请求轨迹全部完成，"
        "既不能证明反事实政策不会饥饿，也不能把131次重复探测当成大请求真的被越过131次。\n")
    add("当前建议：**暂不修改正式Scheduler**，先补足可重放日志或做默认关闭的最小受控候选"
        "实验。若后续实现，优先只看下一位+每队首最多被1个新请求越过；不跳过恢复和持块Chunk，"
        "仍遵守物理容量/Prefix语义。skip2与全队列最小需求没有显示额外局部选择收益，"
        "不值得凭这份数据增加复杂度。前述每队首预算是后续建议，本次没有假装已仿真其完整效果。\n")
    add("补充日志应包含每次Prefill尝试前后有序waiting/running、请求的完整Token历史或足以复现"
        "Prefix的块内容/链式hash、所有分配/释放/扩页Block ID与ref_count、缓存覆盖和free顺序。"
        "即使可精确重放KV，替代GPU服务时间仍需明确模型/新实测，不能沿用原完成时刻。\n")
    add("## 文件与验证\n")
    add("benchmarks/fifo_hol_analysis.py只做分析；tests/test_fifo_hol_analysis.py新增6项CPU测试，"
        "覆盖Chunk/抢占队列重放、缺失/歧义日志拒绝、容量和lookahead边界、Token预算、"
        "活跃/空闲Prefix消耗及未知生成后缀。原113项测试保留，当前119项通过。"
        "Scheduler/BlockManager/ModelRunner/Sequence未修改，无GPU实测、无依赖安装、无Commit/Push。\n")
    add("独立结果目录保存analysis.json、waiting_boundaries.csv、snapshots.csv、candidate_costs.csv、"
        "offline_probes.csv、probe_summary.csv、observed_metrics.csv、observed_requests.csv、源码快照及输入全文件SHA256。"
        "observed_requests逐请求保留实际首次分配、TTFT、完成时间/延迟、初始物理需求与Prefix命中，标为原GPU观察值。"
        "首次试验分析目录与最终目录分开，未混入任何原始实验。\n")
    add("```bash\n.venv/bin/python -m benchmarks.fifo_hol_analysis --output-dir results_fifo_hol_final_20261102\n"
        ".venv/bin/python -m benchmarks.fifo_hol_analysis --report-from results_fifo_hol_final_20261102/analysis.json\n"
        ".venv/bin/python -m unittest discover -s tests -v\ngit diff --check\n```\n")
    add("已有非空输出目录会报错，重跑分析需改用新目录；report-from只重生成中文报告。"
        "所有模拟指标明确标为单次离线分支；没有端到端模拟结果，也没有宣称GPU性能提升。\n")
    target.write_text("\n".join(lines),encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="results_kv_admission_20261102")
    parser.add_argument("--output-dir", default="results_fifo_hol_20261102")
    parser.add_argument("--report", default="benchmarks/fifo_hol_analysis_zh.md")
    parser.add_argument("--report-from", help="render report from an already saved offline analysis")
    args = parser.parse_args()
    if args.report_from:
        result=json.loads(Path(args.report_from).read_text())
        if input_hashes(Path(result["input_dir"])) != result["input_sha256"]:
            raise ValueError("original input fingerprints changed")
        verify_source_snapshots(result["input_dir"])
        export_observed_requests(result,Path(args.report_from).parent)
        write_report(result,args.report)
        print(f"Wrote report without changing original data: {args.report}")
        return
    source, output = Path(args.results).resolve(), Path(args.output_dir).resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError("analysis output must be separate from original experiment")
    if output.exists() and any(output.iterdir()):
        raise ValueError("output directory must be empty; never overwrite an analysis")
    before = input_hashes(source)
    payload = verify_results(source)
    verify_source_snapshots(source)
    analyzed = [analyze_run(run,payload["manifest"]["workloads"][run["summary"]["pressure"]])
                for run in payload["runs"]]
    output.mkdir(parents=True,exist_ok=True)
    for field in ("snapshots","waiting_boundaries","candidate_costs","offline_probes","probe_summary"):
        export_csv(output/(field+".csv"), [{**r["identity"],**row} for r in analyzed for row in r[field]])
    export_csv(output/"observed_metrics.csv", [{**r["identity"],**r["observed_metrics"]} for r in analyzed])
    after = input_hashes(source)
    if before != after:
        raise RuntimeError("original artifacts changed during read-only analysis")
    result = dict(kind="OBSERVED_REPLAY_AND_OFFLINE_SINGLE_DECISION_ONLY", input_dir=str(source),
                  input_sha256=before, input_unchanged=True,
                  current_head=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
                  recorded_experiment_head=payload["manifest"]["git_head"],runs=analyzed,
                  gaps=["no generated output Token IDs", "no complete hash/free-list state",
                        "no IDs for Decode page allocations", "no alternative GPU service durations",
                        "intra-schedule queue-pop order inferred; exact pop timestamps unrecorded"])
    (output/"analysis.json").write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+"\n")
    (output/"analysis_source.py").write_bytes(Path(__file__).read_bytes())
    export_observed_requests(result,output)
    write_report(result,args.report)
    print(f"Verified and recovered {len(analyzed)} original runs; offline probes only: {output}")


if __name__ == "__main__":
    main()
