"""Native-capacity KV contention benchmark; no scheduler/runner/kernel changes.

Pressure comes from concurrent page-boundary requests, never from editing the
free list or pretending the GPU tensor has a different capacity.
"""

import argparse
import atexit
from collections import Counter, defaultdict
from contextlib import ExitStack
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.metrics import percentile, summarize_requests, write_results
from benchmarks.mixed_load_bench import (EngineObserver, RequestSpec, drive_requests, find_model,
                                         gpu_snapshot, run_identity)
from benchmarks.reliability import (interval_analysis, validate_run_steps, write_csv,
                                    cross_run_statistics, decode_batch_statistics)


POLICIES = ("prefill_first", "prefill_first_rr", "decode_first_fair2_rr")
ADMISSION_RESERVES = {"B0": 0, "B4": 4, "B8": 8, "B16": 16, "B8_bypass": 8}
PRESSURES = {"low": 16, "medium": 48, "high": 80}


def pressure_identity(policy, repeat):
    base = "prefill_first_rr" if policy in ADMISSION_RESERVES else policy
    return dict(**{**run_identity(base, repeat), "policy": policy},
                kv_admission_policy="reserve" if policy in ADMISSION_RESERVES else "none",
                kv_reserve_blocks=ADMISSION_RESERVES.get(policy, 0),
                kv_bounded_bypass=policy == "B8_bypass")


def make_workload(tokenizer, count, seed):
    if type(count) is not int or count < 4:
        raise ValueError("need at least four requests")
    rng = random.Random(seed)
    specs = []
    for i in range(count):
        length = 248 if i % 2 == 0 else 504
        unit = (f"KV request {i}, nonce {rng.randrange(10**12)}. "
                "Explain page boundaries, memory contention and cache recovery. ")
        text = unit * 150
        ids = tokenizer.encode(text, add_special_tokens=False)[:length]
        if len(ids) != length:
            raise ValueError("tokenized prompt is too short")
        specs.append(RequestSpec(i, ids, 16, 0., f"page_boundary_{length}"))
    # Exactly one deliberately shared full block; unique suffixes remain intact.
    specs[3].prompt_token_ids[:256] = specs[1].prompt_token_ids[:256]
    return specs


def validate_workload(specs, capacity, block_size, max_model_len):
    if capacity <= 0 or block_size != 256 or not specs:
        raise ValueError("need positive native capacity, 256-token blocks, and requests")
    if len({s.request_id for s in specs}) != len(specs):
        raise ValueError("duplicate request IDs")
    for spec in specs:
        # The final output is not fed back to the model, hence -1 for KV demand.
        demand = (len(spec.prompt_token_ids) + spec.max_tokens - 2) // block_size + 1
        if (not spec.prompt_token_ids or spec.max_tokens <= 0
                or len(spec.prompt_token_ids) + spec.max_tokens > max_model_len
                or demand > capacity or spec.scheduled_arrival != 0.):
            raise ValueError(f"request {spec.request_id}: capacity/context/arrival constraint failed "
                             f"(standalone_blocks={demand}, capacity={capacity})")
    return True


def resource_snapshot(scheduler):
    """Check actual queue ownership, every reference, and the full block partition."""
    manager = scheduler.block_manager
    live = list(scheduler.waiting) + list(scheduler.running)
    if len({s.seq_id for s in live}) != len(live):
        raise RuntimeError("duplicate live sequence in queues")
    refs = Counter(b for seq in live for b in seq.block_table)
    capacity = len(manager.blocks)
    if any(b < 0 or b >= capacity for b in refs):
        raise RuntimeError("out-of-range block ID")
    used = {b for b, count in refs.items() if count}
    free = list(manager.free_block_ids)
    if (len(free) != len(set(free)) or set(free) != set(range(capacity)) - used
            or manager.used_block_ids != used
            or any(block.ref_count != refs[i] for i, block in enumerate(manager.blocks))):
        raise RuntimeError("KV block ownership/reference/free-list mismatch")
    if (any(s.status.name != "WAITING" for s in scheduler.waiting)
            or any(s.status.name != "RUNNING" for s in scheduler.running)):
        raise RuntimeError("queue/status mismatch")
    return dict(total_blocks=capacity, used_blocks=len(used), free_blocks=len(free),
                ref_count_sum=sum(refs.values()), waiting_requests=len(scheduler.waiting),
                running_requests=len(scheduler.running))


def clear_idle_prefix_cache(engine):
    state = resource_snapshot(engine.scheduler)
    if state["used_blocks"] or not engine.is_finished():
        raise RuntimeError("cannot reset prefix metadata while requests are live")
    manager = engine.scheduler.block_manager
    manager.hash_to_block_id.clear()
    for block in manager.blocks:
        block.hash = -1
        block.token_ids = []
    # No change to tensor capacity, block count, free-list contents/order, or refs.
    if resource_snapshot(engine.scheduler) != state:
        raise RuntimeError("idle cache reset changed resources")


class PressureObserver(EngineObserver):
    def __init__(self, engine, specs, **kwargs):
        super().__init__(engine, specs, **kwargs)
        self.request_stats = {s.request_id: dict(preemptions=0, recomputed_tokens=0,
            prefill_tokens=0, prefix_hit_tokens=0, recovery_prefix_hit_tokens=0,
            valid_highwater=0, reserve_deferrals=0, reserve_wait_s=0.,
            first_prefill_time=None) for s in specs}
        self.preemption_events = []
        self.allocation_events = []
        self.peak_used_blocks = 0
        self.cache_evictions = 0
        self.failed_can_append = 0
        self.blocked_request = None
        self.owner_sequences = {}
        self.reserve_deferral_events = []
        self.reserve_wait_events = []
        self.pending_reserve_waits = {}
        self.bypass_events = []

    def request_id(self, seq):
        return self.sequences[seq.seq_id].request_id

    def observe_peak(self):
        self.peak_used_blocks = max(self.peak_used_blocks, len(self.engine.scheduler.block_manager.used_block_ids))

    def __enter__(self):
        super().__enter__()
        scheduler = self.engine.scheduler
        manager = scheduler.block_manager
        original_allocate, original_new = manager.allocate, manager._allocate_block
        original_append = manager.can_append
        original_preempt, original_schedule = scheduler.preempt, scheduler.schedule
        original_postprocess = scheduler.postprocess
        original_promote = scheduler._promote_bypass

        def promote(head, candidate):
            waiting_before = [self.request_id(seq) for seq in scheduler.waiting]
            allocation = self.allocation_events[-1]
            if (scheduler.waiting[0] is not head or scheduler.waiting[1] is not candidate
                    or head.block_table or head.num_completion_tokens or candidate.num_completion_tokens
                    or head.seq_id in scheduler._bypassed_before_admission
                    or allocation["request_id"] != self.request_id(candidate) or allocation["recovery"]):
                raise RuntimeError("invalid bounded bypass state")
            result = original_promote(head, candidate)
            self.bypass_events.append(dict(event_index=len(self.bypass_events),
                step_index=len(self.steps), time=self.now(), head_request_id=self.request_id(head),
                candidate_request_id=self.request_id(candidate), waiting_before=waiting_before,
                waiting_after=[self.request_id(seq) for seq in scheduler.waiting],
                allocation_event_index=allocation["event_index"],
                candidate_physical_blocks=len(candidate.block_table)-allocation["active_shared_blocks"],
                free_after=len(manager.free_block_ids), reserve_blocks=scheduler.kv_reserve_blocks))
            return result

        def new_block():
            block = manager.blocks[manager.free_block_ids[0]]
            evicted = block.hash != -1 and manager.hash_to_block_id.get(block.hash) == block.block_id
            result = original_new()
            self.cache_evictions += int(evicted)
            self.observe_peak()
            return result

        def allocate(seq, hit_blocks):
            request_id = self.request_id(seq)
            before_used = set(manager.used_block_ids)
            result = original_allocate(seq, hit_blocks)
            active = sum(b in before_used for b in seq.block_table[:hit_blocks])
            recovery = self.request_stats[request_id]["preemptions"] > 0
            hit_tokens = hit_blocks * manager.block_size
            stats = self.request_stats[request_id]
            if not recovery and stats["first_prefill_time"] is None:
                stats["first_prefill_time"] = self.now()
                if request_id in self.pending_reserve_waits:
                    episode = self.pending_reserve_waits.pop(request_id)
                    episode.update(end_time=stats["first_prefill_time"],
                                   duration_s=stats["first_prefill_time"] - episode["start_time"])
                    stats["reserve_wait_s"] = episode["duration_s"]
                    self.reserve_wait_events.append(episode)
            stats["prefix_hit_tokens"] += hit_tokens
            stats["recovery_prefix_hit_tokens"] += hit_tokens if recovery else 0
            stats["valid_highwater"] = max(stats["valid_highwater"], seq.num_cached_tokens)
            event = dict(event_index=len(self.allocation_events), request_id=request_id,
                         step_index=len(self.steps), time=self.now(), recovery=recovery,
                         hit_blocks=hit_blocks, hit_tokens=hit_tokens,
                         active_shared_blocks=active, free_retained_blocks=hit_blocks-active,
                         block_table=list(seq.block_table), seq_tokens=len(seq))
            self.allocation_events.append(event)
            if self.active_step is not None:
                self.active_step["prefix_hit_tokens"] = self.active_step.get("prefix_hit_tokens", 0) + hit_tokens
            self.observe_peak()
            return result

        def can_append(seq):
            result = original_append(seq)
            if not result:
                self.failed_can_append += 1
                self.blocked_request = self.request_id(seq)
            return result

        def preempt(seq):
            request_id = self.request_id(seq)
            event = dict(event_index=len(self.preemption_events), request_id=request_id,
                         step_index=len(self.steps), time=self.now(), trigger_request_id=self.blocked_request,
                         valid_tokens_before=seq.num_cached_tokens, seq_tokens=len(seq),
                         output_tokens=seq.num_completion_tokens, block_table_before=list(seq.block_table),
                         free_before=len(manager.free_block_ids), used_before=len(manager.used_block_ids),
                         resume_schedule_time=None, restored_time=None)
            result = original_preempt(seq)
            self.request_stats[request_id]["preemptions"] += 1
            event.update(waiting_after=seq in scheduler.waiting, block_table_after=list(seq.block_table),
                         free_after=len(manager.free_block_ids), used_after=len(manager.used_block_ids))
            if not event["waiting_after"] or event["block_table_after"] or seq.num_cached_tokens:
                raise RuntimeError("preemption did not restore WAITING/deallocated state")
            self.preemption_events.append(event)
            return result

        def schedule():
            if self.active_step is not None:
                self.active_step["reserve_deferrals"] = 0
            seqs, is_prefill = original_schedule()
            executions = []
            for seq in seqs:
                rid = self.request_id(seq)
                self.owner_sequences[rid] = seq
                stats = self.request_stats[rid]
                start, end = seq.num_cached_tokens, seq.num_cached_tokens + seq.num_scheduled_tokens
                # Only execute-overlap with historically valid KV after preemption.
                # Fresh chunks and the last not-yet-fed output are excluded.
                repeated = (max(0, min(end, stats["valid_highwater"]) - start)
                            if is_prefill and stats["preemptions"] else 0)
                executions.append(dict(request_id=rid, start=start, end=end, q=end-start,
                                       valid_highwater_before=stats["valid_highwater"],
                                       recomputed_tokens=repeated, recovery=is_prefill and stats["preemptions"] > 0))
                if is_prefill:
                    for event in reversed(self.preemption_events):
                        if event["request_id"] == rid:
                            if event["resume_schedule_time"] is None:
                                event["resume_schedule_time"] = self.now()
                            break
            self.observe_peak()
            if self.active_step is not None:
                self.active_step.update(executions=executions,
                    prefill_tokens=sum(e["q"] for e in executions) if is_prefill else 0,
                    recomputed_tokens=sum(e["recomputed_tokens"] for e in executions),
                    prefix_hit_tokens=self.active_step.get("prefix_hit_tokens", 0),
                    kv_after_schedule=resource_snapshot(scheduler))
            return seqs, is_prefill

        def postprocess(seqs, token_ids, is_prefill):
            result = original_postprocess(seqs, token_ids, is_prefill)
            for execution, seq in zip(self.active_step["executions"], seqs):
                stats = self.request_stats[execution["request_id"]]
                stats["valid_highwater"] = max(stats["valid_highwater"], execution["end"])
                if is_prefill:
                    stats["prefill_tokens"] += execution["q"]
                    stats["recomputed_tokens"] += execution["recomputed_tokens"]
                    for event in reversed(self.preemption_events):
                        if event["request_id"] == execution["request_id"]:
                            if event["restored_time"] is None and execution["end"] >= event["valid_tokens_before"]:
                                event["restored_time"] = self.now()
                            break
            self.active_step["kv_after_step"] = resource_snapshot(scheduler)
            return result

        for obj, name, method in ((manager, "_allocate_block", new_block), (manager, "allocate", allocate),
                                 (manager, "can_append", can_append), (scheduler, "preempt", preempt),
                                 (scheduler, "schedule", schedule), (scheduler, "postprocess", postprocess),
                                 (scheduler, "_promote_bypass", promote)):
            self.replace(obj, name, method)
        original_admit = scheduler._can_admit_prefill
        def can_admit(seq, cached_blocks):
            allowed = original_admit(seq, cached_blocks)
            if not allowed:
                rid = self.request_id(seq)
                now = self.now()
                self.request_stats[rid]["reserve_deferrals"] += 1
                if rid not in self.pending_reserve_waits:
                    self.pending_reserve_waits[rid] = dict(request_id=rid, start_time=now, attempts=0)
                self.pending_reserve_waits[rid]["attempts"] += 1
                physical_blocks = seq.num_blocks
                h = -1
                for i in range(cached_blocks):
                    h = manager.compute_hash(seq.block(i), h)
                    physical_blocks -= manager.hash_to_block_id[h] in manager.used_block_ids
                self.reserve_deferral_events.append(dict(request_id=rid, time=now, step_index=len(self.steps),
                    cached_blocks=cached_blocks, physical_blocks=physical_blocks,
                    free_blocks=len(manager.free_block_ids), reserve_blocks=scheduler.kv_reserve_blocks))
                if self.active_step is not None:
                    self.active_step["reserve_deferrals"] += 1
            return allowed
        self.replace(scheduler, "_can_admit_prefill", can_admit)
        return self


def run_requests(engine, observer, specs, sampling_factory, timeout, max_steps, no_progress_steps):
    if no_progress_steps <= 0:
        raise ValueError("no_progress_steps must be positive")
    original_step = engine.step
    stale = 0
    progress = 0
    def checked_step():
        nonlocal stale, progress
        result = original_step()
        current = (sum(len(r.token_times) for r in observer.records.values()) +
                   sum(r["valid_highwater"] for r in observer.request_stats.values()))
        stale = stale + 1 if current == progress else 0
        progress = current
        if stale >= no_progress_steps:
            raise RuntimeError(f"no new output/cache progress for {stale} steps; possible preemption thrashing")
        return result
    observer.replace(engine, "step", checked_step)
    return drive_requests(engine, observer, specs, sampling_factory, timeout, max_steps)


def pressure_summary(observer, elapsed):
    records = list(observer.records.values())
    summary = summarize_requests(records, elapsed)
    intervals = interval_analysis([r.to_dict() for r in records])
    recovery = [s for s in observer.steps if s["recomputed_tokens"] > 0]
    pure = [s for s in recovery if s["recomputed_tokens"] == s["prefill_tokens"]]
    total = len(observer.engine.scheduler.block_manager.blocks)
    waits = [e["duration_s"] for e in observer.reserve_wait_events]
    first_prefill_waits = [stats["first_prefill_time"] - observer.records[rid].admission_time
                          for rid, stats in observer.request_stats.items() if stats["first_prefill_time"] is not None]
    return dict(**summary, completion_rate=summary["completed_requests"] / len(records) if records else None,
        itl_p99_s=intervals["p99_s"], itl_max_s=intervals["max_s"],
        peak_used_blocks=observer.peak_used_blocks, peak_block_utilization=observer.peak_used_blocks / total,
        step_weighted_block_utilization=(sum(s["kv_after_schedule"]["used_blocks"] * s["duration_s"]
                                             for s in observer.steps) / (total * sum(s["duration_s"] for s in observer.steps))
                                         if observer.steps else None),
        steps_at_capacity=sum(s["kv_after_schedule"]["used_blocks"] == total for s in observer.steps),
        prefill_steps=observer.prefill_steps, decode_steps=observer.decode_steps, preemptions=observer.preemptions,
        failed_can_append=observer.failed_can_append, cache_evictions=observer.cache_evictions,
        executed_prefill_tokens=sum(s["prefill_tokens"] for s in observer.steps),
        recomputed_tokens=sum(s["recomputed_tokens"] for s in observer.steps),
        prefix_hit_tokens=sum(e["hit_tokens"] for e in observer.allocation_events),
        prefix_hit_blocks=sum(e["hit_blocks"] for e in observer.allocation_events),
        prefix_active_shared_blocks=sum(e["active_shared_blocks"] for e in observer.allocation_events),
        prefix_free_retained_blocks=sum(e["free_retained_blocks"] for e in observer.allocation_events),
        recovery_prefix_hit_tokens=sum(e["hit_tokens"] for e in observer.allocation_events if e["recovery"]),
        recompute_affected_steps=len(recovery),
        recompute_affected_step_wall_s=sum(s["duration_s"] for s in recovery),
        recompute_only_step_wall_s=sum(s["duration_s"] for s in pure),
        marginal_preemption_overhead_s=None,
        reserve_deferrals=len(observer.reserve_deferral_events),
        reserve_deferred_requests=sum(s["reserve_deferrals"] > 0 for s in observer.request_stats.values()),
        reserve_deferred_steps=len({e["step_index"] for e in observer.reserve_deferral_events}),
        reserve_wait_total_s=sum(waits), reserve_wait_p50_s=percentile(waits, .5),
        reserve_wait_p95_s=percentile(waits, .95),
        admission_to_first_prefill_p50_s=percentile(first_prefill_waits, .5),
        admission_to_first_prefill_mean_s=sum(first_prefill_waits) / len(first_prefill_waits) if first_prefill_waits else None,
        admission_to_first_prefill_p95_s=percentile(first_prefill_waits, .95),
        actual_bypasses=len(observer.bypass_events),
        decode_mean_batch_size=decode_batch_statistics(observer.steps)["mean_batch_size"])


def warmup_pressure(engine, rounds):
    from nanovllm import SamplingParams
    # Disjoint text, exercising partial prefills, paged KV and boundary decodes.
    for _ in range(rounds):
        for length in (248, 504):
            ids = engine.tokenizer.encode("Warmup KV pressure only. " * 200,
                                          add_special_tokens=False)[:length]
            engine.generate([ids], SamplingParams(max_tokens=16, ignore_eos=True), use_tqdm=False)
        engine.generate([ids[:64]] * 4,
                        [SamplingParams(max_tokens=13+i, ignore_eos=True) for i in range(4)], use_tqdm=False)
    clear_idle_prefix_cache(engine)


def run_worker(manifest_path, output, pressure, policy, repeat):
    manifest = json.loads(Path(manifest_path).read_text())
    config = manifest["config"]
    specs = [RequestSpec(**{k: v for k, v in r.items() if k != "prompt_tokens"})
             for r in manifest["workloads"][pressure]]
    identity = dict(**pressure_identity(policy, repeat), pressure=pressure)
    observer = None
    elapsed = None
    environment, resources = {}, None
    error = None
    native_blocks = compile_frames = None
    try:
        import torch
        from torch._dynamo.utils import counters
        from nanovllm import LLM, SamplingParams
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; no fabricated performance fallback")
        seed = config["seed"] + repeat
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        engine = LLM(config["model"], tensor_parallel_size=1, enforce_eager=True,
                     max_model_len=768, max_num_batched_tokens=256, max_num_seqs=4,
                     gpu_memory_utilization=.65, scheduling_policy=identity["scheduling_policy"],
                     max_consecutive_decode_steps=identity["max_consecutive_decode_steps"],
                     decode_queue_policy=identity["decode_queue_policy"],
                     kv_admission_policy=identity["kv_admission_policy"], kv_reserve_blocks=identity["kv_reserve_blocks"],
                     kv_bounded_bypass=identity["kv_bounded_bypass"])
        def close():
            atexit.unregister(engine.exit)
            engine.exit()
        with ExitStack() as cleanup:
            cleanup.callback(close)
            native_blocks = len(engine.scheduler.block_manager.blocks)
            physical_blocks = engine.model_runner.kv_cache.shape[2]
            if native_blocks != physical_blocks or native_blocks != engine.model_runner.config.num_kvcache_blocks:
                raise RuntimeError("physical and managed KV capacities differ")
            if config.get("expected_blocks") is not None and native_blocks != config["expected_blocks"]:
                raise RuntimeError(f"expected {config['expected_blocks']} native KV blocks, got {native_blocks}")
            validate_workload(specs, native_blocks, engine.scheduler.block_size,
                              engine.model_runner.config.max_model_len)
            environment.update(gpu_name=torch.cuda.get_device_name(0), torch=torch.__version__,
                               cuda=torch.version.cuda, python=sys.version,
                               physical_kv_shape=list(engine.model_runner.kv_cache.shape),
                               physical_blocks=physical_blocks, gpu_before_warmup=gpu_snapshot())
            print(f"native KV capacity={native_blocks}; warming {pressure}/{policy}", flush=True)
            warmup_pressure(engine, config["warmup_rounds"])
            torch.cuda.synchronize()
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            frames = counters["frames"]["total"]
            environment["gpu_before_measurement"] = gpu_snapshot()
            observer = PressureObserver(engine, specs)
            with observer:
                try:
                    elapsed = run_requests(engine, observer, specs,
                        lambda n: SamplingParams(max_tokens=n, temperature=.6, ignore_eos=True),
                        config["timeout_seconds"], config["max_steps"], config["no_progress_steps"])
                finally:
                    if observer.origin is not None and elapsed is None:
                        elapsed = observer.now()
                    resources = resource_snapshot(engine.scheduler)
            compile_frames = counters["frames"]["total"] - frames
            environment["gpu_after_measurement"] = gpu_snapshot()
            if compile_frames:
                raise RuntimeError(f"{compile_frames} compiled frames in measured window; expand warmup")
            if (not engine.is_finished() or resources["used_blocks"] or resources["ref_count_sum"]
                    or any(len(r.token_times) != r.max_tokens or r.completion_time is None
                           for r in observer.records.values())):
                raise RuntimeError("requests incomplete or KV not released")
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
    run = dict(summary=dict(**identity, status="failed" if error else "ok", error=error,
                            num_kvcache_blocks=native_blocks, torch_compile_frames_during_measurement=compile_frames),
               environment=environment, resource_check=resources, requests=[], steps=[],
               preemption_events=[], allocation_events=[])
    if observer is not None:
        run["summary"].update(pressure_summary(observer, elapsed))
        run["requests"] = [{**r.to_dict(), **observer.request_stats[r.request_id]}
                           for r in observer.records.values()]
        run.update(steps=observer.steps, preemption_events=observer.preemption_events,
                   allocation_events=observer.allocation_events,
                   reserve_deferral_events=observer.reserve_deferral_events,
                   reserve_wait_events=observer.reserve_wait_events,
                   bypass_events=observer.bypass_events)
        if resources:
            resources.update(released=not resources["used_blocks"] and not resources["waiting_requests"]
                                      and not resources["running_requests"],
                             block_ref_count_sum=resources["ref_count_sum"])
    Path(output).write_text(json.dumps(run, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(f"{pressure}/{policy}/{repeat}: {run['summary']}", flush=True)
    return int(error is not None)


def validate_pressure_run(run, specs):
    validate_run_steps(run)
    summary = run["summary"]
    if summary["num_kvcache_blocks"] != run["environment"]["physical_blocks"]:
        raise ValueError("physical/managed capacity mismatch")
    if {r["request_id"] for r in run["requests"]} != {s.request_id for s in specs}:
        raise ValueError("request workload differs")
    for row, spec in zip(run["requests"], specs):
        if (row["prompt_tokens"] != len(spec.prompt_token_ids) or row["max_tokens"] != spec.max_tokens
                or row["request_id"] != spec.request_id):
            raise ValueError("request lengths/IDs differ")
    if len(run["preemption_events"]) != summary["preemptions"]:
        raise ValueError("preemption count differs")
    hwm, preemptions, recomputed = Counter(), Counter(), Counter()
    prefix, executed = Counter(), Counter()
    by_step_alloc, by_step_preempt = defaultdict(list), defaultdict(list)
    for event in run["allocation_events"]:
        if event["hit_tokens"] != 256 * event["hit_blocks"] or event["active_shared_blocks"] + event["free_retained_blocks"] != event["hit_blocks"]:
            raise ValueError("prefix reuse accounting differs")
        by_step_alloc[event["step_index"]].append(event)
    for event in run["preemption_events"]:
        if not event["waiting_after"] or event["block_table_after"] or event["restored_time"] is None:
            raise ValueError("preemption never recovered")
        by_step_preempt[event["step_index"]].append(event)
    for step in run["steps"]:
        for event in by_step_preempt[step["step_index"]]:
            preemptions[event["request_id"]] += 1
        for event in by_step_alloc[step["step_index"]]:
            hwm[event["request_id"]] = max(hwm[event["request_id"]], event["hit_tokens"])
            prefix[event["request_id"]] += event["hit_tokens"]
        total_recomputed = total_prefill = 0
        for execution in step["executions"]:
            rid = execution["request_id"]
            if execution["valid_highwater_before"] != hwm[rid]:
                raise ValueError("historical valid prefix differs")
            expected = (max(0, min(execution["end"], hwm[rid]) - execution["start"])
                        if step["phase"] == "prefill" and preemptions[rid] else 0)
            if expected != execution["recomputed_tokens"] or execution["q"] != execution["end"] - execution["start"]:
                raise ValueError("recomputation interval accounting differs")
            recomputed[rid] += expected
            total_recomputed += expected
            if step["phase"] == "prefill":
                total_prefill += execution["q"]
                executed[rid] += execution["q"]
            hwm[rid] = max(hwm[rid], execution["end"])
        if total_recomputed != step["recomputed_tokens"] or total_prefill != step["prefill_tokens"]:
            raise ValueError("step recomputation/prefill totals differ")
        if step["kv_after_schedule"]["total_blocks"] != summary["num_kvcache_blocks"]:
            raise ValueError("step capacity changed")
    for row in run["requests"]:
        rid = row["request_id"]
        for key, expected in (("preemptions", preemptions[rid]), ("recomputed_tokens", recomputed[rid]),
                              ("prefix_hit_tokens", prefix[rid]), ("prefill_tokens", executed[rid])):
            if row[key] != expected:
                raise ValueError(f"request {rid}: {key} differs")
    if (summary["recomputed_tokens"] != sum(recomputed.values())
            or summary["prefix_hit_tokens"] != sum(prefix.values())
            or summary["executed_prefill_tokens"] != sum(executed.values())
            or summary["completion_rate"] != 1. or summary["marginal_preemption_overhead_s"] is not None):
        raise ValueError("summary pressure accounting differs")
    # Independent conservation check: every prompt and non-final output must
    # be fed exactly once, except initial prefix reuse and repeated KV work.
    initial_hits = sum(e["hit_tokens"] for e in run["allocation_events"] if not e["recovery"])
    actual_model_tokens = sum(e["q"] for s in run["steps"] for e in s["executions"])
    minimal_model_tokens = sum(len(s.prompt_token_ids) + s.max_tokens - 1 for s in specs) - initial_hits
    if actual_model_tokens != minimal_model_tokens + summary["recomputed_tokens"]:
        raise ValueError("executed Token conservation failed")
    recovery = [s for s in run["steps"] if s["recomputed_tokens"]]
    pure = [s for s in recovery if s["recomputed_tokens"] == s["prefill_tokens"]]
    for key, expected in (
            ("recompute_affected_step_wall_s", sum(s["duration_s"] for s in recovery)),
            ("recompute_only_step_wall_s", sum(s["duration_s"] for s in pure))):
        if summary[key] != expected:
            raise ValueError("recomputation-associated timing differs")
    if "reserve_deferral_events" in run:
        attempts = Counter(e["request_id"] for e in run["reserve_deferral_events"])
        episodes = {e["request_id"]: e for e in run["reserve_wait_events"]}
        if len(episodes) != len(run["reserve_wait_events"]) or set(episodes) != set(attempts):
            raise ValueError("reserve wait episode missing/duplicated")
        records = {r["request_id"]: r for r in run["requests"]}
        for event in run["reserve_deferral_events"]:
            if (event["free_blocks"] - event["physical_blocks"] >= event["reserve_blocks"]
                    or not records[event["request_id"]]["admission_time"] <= event["time"] <= records[event["request_id"]]["first_prefill_time"]):
                raise ValueError("invalid reserve deferral")
        for rid, row in records.items():
            if (row["reserve_deferrals"] != attempts[rid]
                    or row["first_prefill_time"] is None
                    or not row["admission_time"] <= row["first_prefill_time"] <= row["first_token_time"]):
                raise ValueError("invalid initial Prefill/defer accounting")
            if rid in episodes:
                event = episodes[rid]
                if (event["end_time"] != row["first_prefill_time"] or event["attempts"] != attempts[rid]
                        or event["duration_s"] != event["end_time"] - event["start_time"]
                        or event["duration_s"] != row["reserve_wait_s"]):
                    raise ValueError("reserve duration differs")
            elif row["reserve_wait_s"] != 0.:
                raise ValueError("unexplained reserve wait")
        if (summary["reserve_deferrals"] != sum(attempts.values())
                or summary["reserve_deferred_requests"] != len(attempts)
                or summary["reserve_wait_total_s"] != sum(e["duration_s"] for e in run["reserve_wait_events"])):
            raise ValueError("reserve summary differs")
    if "bypass_events" in run:
        heads = set()
        records = {r["request_id"]: r for r in run["requests"]}
        for event in run["bypass_events"]:
            head, candidate = event["head_request_id"], event["candidate_request_id"]
            step = run["steps"][event["step_index"]]
            allocation = run["allocation_events"][event["allocation_event_index"]]
            rejected = [e for e in run["reserve_deferral_events"]
                        if e["request_id"] == head and e["step_index"] == event["step_index"]
                        and e["time"] <= event["time"]]
            if (not summary.get("kv_bounded_bypass") or head in heads
                    or event["waiting_before"][:2] != [head, candidate]
                    or event["waiting_after"] != [candidate, head] + event["waiting_before"][2:]
                    or step["phase"] != "prefill" or candidate not in step["request_ids"]
                    or allocation["request_id"] != candidate or allocation["recovery"]
                    or allocation["step_index"] != event["step_index"] or not rejected
                    or event["candidate_physical_blocks"] != len(allocation["block_table"])-allocation["active_shared_blocks"]
                    or event["free_after"] < event["reserve_blocks"]
                    or not allocation["time"] <= event["time"] <= step["schedule_time"]
                    or records[head]["first_prefill_time"] <= event["time"]):
                raise ValueError("invalid or repeated bounded bypass event")
            heads.add(head)
        if summary["actual_bypasses"] != len(run["bypass_events"]):
            raise ValueError("bypass summary differs")
    return True


def export_csv(path, rows):
    write_csv(path, [{k: json.dumps(v, sort_keys=True) if isinstance(v, dict) else v
                      for k, v in row.items()} for row in rows])


def assert_csv_matches(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    expected = [{k: "" if row.get(k) is None else
                 json.dumps(row[k], sort_keys=True) if isinstance(row[k], dict) else
                 json.dumps(row[k]) if isinstance(row[k], list) else str(row[k]) for k in fields}
                for row in rows]
    with Path(path).open(newline="") as file:
        actual = list(csv.DictReader(file))
    if actual != expected:
        raise ValueError(f"raw CSV/JSON differs: {path}")


def analyze_run(run):
    summary = run["summary"]
    events = run["preemption_events"]
    lost = sum(e["valid_tokens_before"] for e in events)
    reused = summary["recovery_prefix_hit_tokens"]
    if lost - reused != summary["recomputed_tokens"]:
        raise ValueError("lost/reused/recomputed Token conservation failed")
    requests = {r["request_id"]: r for r in run["requests"]}
    recovery_tokens = sum(e["q"] for s in run["steps"] for e in s["executions"] if e["recovery"])
    waits = [e["resume_schedule_time"] - e["time"] for e in events]
    restores = [e["restored_time"] - e["time"] for e in events]
    return dict(pressure=summary["pressure"], policy=summary["policy"], repeat=summary["repeat"],
        preempted_requests=sum(r["preemptions"] > 0 for r in run["requests"]),
        preemptions_per_request_histogram=dict(Counter(r["preemptions"] for r in run["requests"])),
        max_request_preemptions=max((r["preemptions"] for r in run["requests"]), default=0),
        victim_prompt_histogram=dict(Counter(requests[e["request_id"]]["prompt_tokens"] for e in events)),
        lost_valid_kv_tokens=lost, recovered_by_prefix_tokens=reused,
        theoretically_full_block_reusable_tokens=sum(e["valid_tokens_before"] // 256 * 256 for e in events),
        lost_partial_block_tokens=sum(e["valid_tokens_before"] % 256 for e in events),
        recovery_prefill_tokens=recovery_tokens,
        recovery_fresh_tokens=recovery_tokens-summary["recomputed_tokens"],
        resume_wait_p50_s=percentile(waits, .5), resume_wait_p95_s=percentile(waits, .95),
        restore_elapsed_p50_s=percentile(restores, .5), restore_elapsed_p95_s=percentile(restores, .95),
        decode_batch_statistics=decode_batch_statistics(run["steps"]))


def verify_results(directory):
    from benchmarks.metrics import verify_results as verify_metrics
    directory = Path(directory)
    payload = json.loads((directory / "pressure_runs.json").read_text())
    manifest = payload["manifest"]
    expected = {(level, policy, repeat) for level in manifest["config"]["pressures"]
                for policy in manifest["config"]["policies"] for repeat in range(manifest["config"]["repeats"])}
    seen = set()
    capacities = set()
    for run in payload["runs"]:
        s = run["summary"]
        identity = (s["pressure"], s["policy"], s["repeat"])
        if identity not in expected or identity in seen:
            raise ValueError("duplicate/unexpected experiment")
        seen.add(identity)
        specs = [RequestSpec(**{k: v for k, v in r.items() if k != "prompt_tokens"})
                 for r in manifest["workloads"][s["pressure"]]]
        validate_pressure_run(run, specs)
        capacities.add(s["num_kvcache_blocks"])
        path = directory / "runs" / f"{s['pressure']}_{s['policy']}_{s['repeat']}.json"
        if json.loads(path.read_text()) != run:
            raise ValueError("worker/aggregate data differs")
    if seen != expected or len(capacities) != 1:
        raise ValueError("missing experiment or unequal native KV capacities")
    for level in manifest["config"]["pressures"]:
        verify_metrics(directory / level)
        selected = [r for r in payload["runs"] if r["summary"]["pressure"] == level]
        identities = ("policy", "scheduling_policy", "max_consecutive_decode_steps", "repeat", "decode_queue_policy")
        assert_csv_matches(directory / level / "raw_requests.csv", [
            {**{k: r["summary"][k] for k in identities}, **row}
            for r in selected for row in r["requests"]])
        extra_fields = ("reserve_deferral_events", "reserve_wait_events") if any("reserve_deferral_events" in r for r in selected) else ()
        if any("bypass_events" in r for r in selected):
            extra_fields += ("bypass_events",)
        for field in ("steps", "preemption_events", "allocation_events") + extra_fields:
            expected_rows = [{"policy": r["summary"]["policy"], "repeat": r["summary"]["repeat"], **row}
                             for r in selected for row in r[field]]
            assert_csv_matches(directory / level / f"{field}.csv", expected_rows)
    summaries = [r["summary"] for r in payload["runs"]]
    assert_csv_matches(directory / "summary.csv", summaries)
    stats = cross_run_statistics([{**s, "policy": s["pressure"] + "/" + s["policy"]} for s in summaries])
    if payload["cross_run_statistics"] != stats:
        raise ValueError("cross-run statistics differ")
    assert_csv_matches(directory / "cross_run_statistics.csv", stats)
    analysis = [analyze_run(r) for r in payload["runs"]]
    if json.loads((directory / "pressure_analysis.json").read_text()) != json.loads(json.dumps(analysis)):
        raise ValueError("derived event analysis differs")
    assert_csv_matches(directory / "event_analysis.csv", analysis)
    if (hashlib.sha256((directory / "instrumentation_source.py").read_bytes()).hexdigest()
            != manifest["source_sha256"]["benchmarks/kv_pressure_bench.py"]):
        raise ValueError("instrumentation source snapshot differs")
    return payload


def collect_results(directory, manifest, runs):
    for level in manifest["config"]["pressures"]:
        selected = [r for r in runs if r["summary"]["pressure"] == level]
        subdir = directory / level
        write_results(subdir, selected, dict(config=manifest["config"], pressure=level))
        extra_fields = ("reserve_deferral_events", "reserve_wait_events") if any("reserve_deferral_events" in r for r in selected) else ()
        if any("bypass_events" in r for r in selected):
            extra_fields += ("bypass_events",)
        for field in ("steps", "preemption_events", "allocation_events") + extra_fields:
            export_csv(subdir / f"{field}.csv", [{"policy": r["summary"]["policy"],
                "repeat": r["summary"]["repeat"], **row} for r in selected for row in r[field]])
    export_csv(directory / "summary.csv", [r["summary"] for r in runs])
    stats = cross_run_statistics([{**r["summary"], "policy": r["summary"]["pressure"] + "/" + r["summary"]["policy"]}
                                  for r in runs])
    export_csv(directory / "cross_run_statistics.csv", stats)
    analysis = [analyze_run(run) for run in runs]
    (directory / "pressure_analysis.json").write_text(json.dumps(analysis, indent=2, allow_nan=False) + "\n")
    export_csv(directory / "event_analysis.csv", analysis)
    (directory / "pressure_runs.json").write_text(json.dumps(dict(manifest=manifest, runs=runs,
        cross_run_statistics=stats), indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    verify_results(directory)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model")
    parser.add_argument("--output-dir", default="results_kv_pressure")
    parser.add_argument("--pressures", nargs="+", choices=tuple(PRESSURES), default=list(PRESSURES))
    parser.add_argument("--policies", nargs="+", choices=POLICIES + tuple(ADMISSION_RESERVES), default=list(POLICIES))
    parser.add_argument("--workload-from", help="reuse exact saved stage-six manifest Token IDs")
    parser.add_argument("--expected-blocks", type=int)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup-rounds", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timeout-seconds", type=float, default=120.)
    parser.add_argument("--max-steps", type=int, default=10000)
    parser.add_argument("--no-progress-steps", type=int, default=100)
    parser.add_argument("--verify")
    parser.add_argument("--worker-manifest", help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", help=argparse.SUPPRESS)
    parser.add_argument("--worker-pressure", help=argparse.SUPPRESS)
    parser.add_argument("--worker-policy", help=argparse.SUPPRESS)
    parser.add_argument("--worker-repeat", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker_manifest:
        return run_worker(args.worker_manifest, args.worker_output, args.worker_pressure,
                          args.worker_policy, args.worker_repeat)
    if args.verify:
        result = verify_results(args.verify)
        print(f"Verified {len(result['runs'])} real runs")
        return 0
    if (args.repeats < 1 or args.warmup_rounds < 1 or args.timeout_seconds <= 0
            or not math.isfinite(args.timeout_seconds) or args.max_steps <= 0 or args.no_progress_steps <= 0
            or (args.expected_blocks is not None and args.expected_blocks <= 0)
            or len(args.pressures) != len(set(args.pressures)) or len(args.policies) != len(set(args.policies))):
        parser.error("invalid repetitions/warmup/limits or duplicate levels/policies")
    directory = Path(args.output_dir).resolve()
    if directory.exists() and any(directory.iterdir()):
        raise ValueError(f"output directory must be empty: {directory}")
    directory.mkdir(parents=True, exist_ok=True)
    model = str(find_model(args.model))
    if args.workload_from:
        saved = json.loads(Path(args.workload_from).read_text())
        workloads = {level: saved["workloads"][level] for level in args.pressures}
        if saved["config"]["model"] != model or saved["config"]["seed"] != args.seed:
            raise ValueError("saved workload model/seed differs")
        if any(len(workloads[level]) != PRESSURES[level] for level in args.pressures):
            raise ValueError("saved workload request count differs")
    else:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True)
        full = make_workload(tokenizer, max(PRESSURES.values()), args.seed)
        workloads = {level: [s.to_dict() for s in full[:PRESSURES[level]]] for level in args.pressures}
    config = dict(model=model, seed=args.seed, repeats=args.repeats, warmup_rounds=args.warmup_rounds,
                  timeout_seconds=args.timeout_seconds, max_steps=args.max_steps,
                  no_progress_steps=args.no_progress_steps, pressures=args.pressures, policies=args.policies,
                  max_model_len=768, max_num_batched_tokens=256, max_num_seqs=4,
                  tensor_parallel_size=1, enforce_eager=True, gpu_memory_utilization=.65,
                  kvcache_block_size=256, capacity_mode="native automatic; no capacity override",
                  expected_blocks=args.expected_blocks,
                  workload_from=str(Path(args.workload_from).resolve()) if args.workload_from else None)
    source_names = ("benchmarks/kv_pressure_bench.py", "benchmarks/mixed_load_bench.py", "nanovllm/config.py",
                    "nanovllm/engine/scheduler.py", "nanovllm/engine/block_manager.py",
                    "nanovllm/engine/model_runner.py")
    manifest = dict(config=config, workloads=workloads,
                    policy_configurations={p: pressure_identity(p, 0) for p in args.policies},
                    timestamp_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                    source_sha256={n: hashlib.sha256(Path(n).read_bytes()).hexdigest() for n in source_names},
                    timing="perf_counter after real append; synchronous admission between steps",
                    percentile_method="linear interpolation (n-1)*p, type 7")
    manifest_path = directory / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    (directory / "instrumentation_source.py").write_bytes(Path(__file__).read_bytes())
    for name in source_names[2:]:
        (directory / (Path(name).stem + "_source.py")).write_bytes(Path(name).read_bytes())
    (directory / "runs").mkdir()
    runs, capacities = [], set()
    for repeat in range(args.repeats):
        levels = args.pressures[repeat % len(args.pressures):] + args.pressures[:repeat % len(args.pressures)]
        for level in levels:
            policies = args.policies[repeat % len(args.policies):] + args.policies[:repeat % len(args.policies)]
            for policy in policies:
                output = directory / "runs" / f"{level}_{policy}_{repeat}.json"
                command = [sys.executable, "-m", "benchmarks.kv_pressure_bench", "--worker-manifest", str(manifest_path),
                           "--worker-output", str(output), "--worker-pressure", level, "--worker-policy", policy,
                           "--worker-repeat", str(repeat)]
                log_path = output.with_suffix(".log")
                print(f"Starting isolated {level}/{policy}, repeat={repeat}", flush=True)
                with log_path.open("w") as log:
                    try:
                        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                                timeout=args.timeout_seconds + 180, check=False)
                    except subprocess.TimeoutExpired as exc:
                        raise RuntimeError(f"worker timeout; retained partial results/log: {log_path}") from exc
                if result.returncode or not output.exists():
                    raise RuntimeError(f"worker failed ({result.returncode}); retained results: {output}; log: {log_path}")
                run = json.loads(output.read_text())
                specs = [RequestSpec(**{k: v for k, v in r.items() if k != "prompt_tokens"}) for r in workloads[level]]
                validate_pressure_run(run, specs)
                capacities.add(run["summary"]["num_kvcache_blocks"])
                if len(capacities) != 1:
                    raise RuntimeError(f"native capacities differ: {capacities}; results cannot be compared")
                runs.append(run)
                s = run["summary"]
                print(f"done={s['completed_requests']}, peak={s['peak_used_blocks']}/{s['num_kvcache_blocks']}, "
                      f"preemptions={s['preemptions']}, recomputed={s['recomputed_tokens']}", flush=True)
    collect_results(directory, manifest, runs)
    print(f"Saved and verified {len(runs)} runs: {directory}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
