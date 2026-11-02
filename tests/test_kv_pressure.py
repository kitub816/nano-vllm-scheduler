"""Real CPU queues/BlockManager; no GPU, model weights, or fabricated performance."""

from copy import deepcopy
import unittest

from benchmarks.kv_pressure_bench import (PressureObserver, make_workload, resource_snapshot,
                                          validate_workload, run_requests, clear_idle_prefix_cache,
                                          pressure_summary, validate_pressure_run, pressure_identity)
from benchmarks.mixed_load_bench import RequestSpec, run_identity
from test_scheduler_policy import SchedulerTestCase, Sequence, SamplingParams


class CpuEngine:
    def __init__(self, scheduler):
        self.scheduler = scheduler

    def add_request(self, ids, params):
        self.scheduler.add(Sequence(ids, params))

    def is_finished(self):
        return self.scheduler.is_finished()

    def step(self):
        seqs, prefill = self.scheduler.schedule()
        tokens = sum(s.num_scheduled_tokens for s in seqs) if prefill else -len(seqs)
        self.scheduler.postprocess(seqs, [10000 + s.seq_id for s in seqs], prefill)
        return [(s.seq_id, s.completion_token_ids) for s in seqs if s.is_finished], tokens


class PressureTests(SchedulerTestCase):
    def execute(self, count=6, blocks=8, policy="prefill_first", queue="round_robin", shared=False):
        scheduler = self.make_scheduler(num_kvcache_blocks=blocks, max_num_batched_tokens=256,
                                        max_num_seqs=4, scheduling_policy=policy, decode_queue_policy=queue,
                                        max_consecutive_decode_steps=2 if policy == "decode_first" else 0)
        specs = []
        for i in range(count):
            ids = [i + 10] * 248
            if shared:
                ids = [1] * 256 + [i + 10] * 248
            specs.append(RequestSpec(i, ids, 16, 0., "cpu_resource_test"))
        engine = CpuEngine(scheduler)
        observer = PressureObserver(engine, specs)
        with observer:
            elapsed = run_requests(engine, observer, specs,
                                   lambda n: SamplingParams(max_tokens=n, ignore_eos=True), 10., 2000, 100)
        state = resource_snapshot(scheduler)
        self.assertEqual(state["used_blocks"], 0)
        self.assertEqual(state["ref_count_sum"], 0)
        self.assertEqual(state["free_blocks"], blocks)
        self.assertTrue(all(r.completion_time is not None for r in observer.records.values()))
        self.assertTrue(all(len(r.token_times) == 16 for r in observer.records.values()))
        return engine, observer, elapsed

    def test_low_pressure_has_no_preemption_or_recomputation(self):
        _, observer, _ = self.execute(count=2, blocks=8)
        self.assertEqual(observer.preemptions, 0)
        self.assertEqual(sum(r["recomputed_tokens"] for r in observer.request_stats.values()), 0)
        self.assertGreater(sum(s["prefill_tokens"] for s in observer.steps), 0)

    def test_pressure_preempts_recovers_and_releases_for_three_policies(self):
        for policy, queue in (("prefill_first", "head_first"), ("prefill_first", "round_robin"),
                              ("decode_first", "round_robin")):
            with self.subTest(policy=policy, queue=queue):
                _, observer, _ = self.execute(policy=policy, queue=queue, count=10, blocks=4)
                self.assertGreater(observer.preemptions, 0)
                self.assertEqual(observer.preemptions, len(observer.preemption_events))
                self.assertEqual(observer.preemptions,
                                 sum(r["preemptions"] for r in observer.request_stats.values()))
                recomputed = sum(s["recomputed_tokens"] for s in observer.steps)
                self.assertGreater(recomputed, 0)
                self.assertEqual(recomputed, sum(r["recomputed_tokens"] for r in observer.request_stats.values()))
                self.assertTrue(all(e["waiting_after"] and not e["block_table_after"]
                                    for e in observer.preemption_events))

    def test_prefix_reuse_is_counted_at_successful_allocation(self):
        _, observer, _ = self.execute(count=3, blocks=16, shared=True)
        self.assertGreater(sum(e["hit_blocks"] for e in observer.allocation_events), 0)
        self.assertGreater(sum(e["active_shared_blocks"] for e in observer.allocation_events), 0)
        self.assertEqual(sum(r["recomputed_tokens"] for r in observer.request_stats.values()), 0)
        self.assertTrue(any(s["prefix_hit_tokens"] > 0 for s in observer.steps))

    def test_reference_mismatch_and_duplicate_queue_entry_rejected(self):
        scheduler = self.make_scheduler()
        seq = self.make_seq()
        scheduler.add(seq)
        scheduler.schedule()
        self.assertGreater(resource_snapshot(scheduler)["used_blocks"], 0)
        scheduler.block_manager.blocks[seq.block_table[0]].ref_count += 1
        with self.assertRaises(RuntimeError):
            resource_snapshot(scheduler)
        scheduler.block_manager.blocks[seq.block_table[0]].ref_count -= 1
        scheduler.running.append(seq)
        with self.assertRaises(RuntimeError):
            resource_snapshot(scheduler)

    def test_capacity_and_context_failure_are_explicit(self):
        spec = RequestSpec(7, [1] * 504, 16, 0., "too_big")
        with self.assertRaisesRegex(ValueError, "request 7"):
            validate_workload([spec], 2, 256, 768)
        with self.assertRaises(ValueError):
            validate_workload([spec], 3, 256, 512)
        self.assertTrue(validate_workload([spec], 3, 256, 768))

    def test_cold_cache_reset_requires_idle_and_keeps_capacity(self):
        engine, _, _ = self.execute(count=2, blocks=8, shared=True)
        before = resource_snapshot(engine.scheduler)
        clear_idle_prefix_cache(engine)
        self.assertEqual(resource_snapshot(engine.scheduler), before)
        self.assertEqual(engine.scheduler.block_manager.hash_to_block_id, {})
        engine.add_request([1] * 8, SamplingParams(max_tokens=2))
        with self.assertRaises(RuntimeError):
            clear_idle_prefix_cache(engine)

    def test_hooks_restore_after_exception(self):
        engine = CpuEngine(self.make_scheduler())
        before = dict(vars(engine.scheduler))
        with self.assertRaisesRegex(RuntimeError, "sentinel"):
            with PressureObserver(engine, []):
                raise RuntimeError("sentinel")
        self.assertEqual(before, vars(engine.scheduler))
        self.assertNotIn("step", vars(engine))

    def test_accounting_replay_rejects_counting_all_prefill_as_recomputation(self):
        engine, observer, elapsed = self.execute(count=10, blocks=4)
        resources = resource_snapshot(engine.scheduler)
        resources.update(released=True, block_ref_count_sum=0)
        run = dict(summary=dict(**run_identity("prefill_first_rr", 0), **pressure_summary(observer, elapsed),
                                status="ok", num_kvcache_blocks=4, torch_compile_frames_during_measurement=0),
                   requests=[{**r.to_dict(), **observer.request_stats[r.request_id]}
                             for r in observer.records.values()],
                   steps=observer.steps, resource_check=resources,
                   environment=dict(physical_blocks=4), preemption_events=observer.preemption_events,
                   allocation_events=observer.allocation_events)
        specs = [RequestSpec(i, [i+10]*248, 16, 0., "cpu_resource_test") for i in range(10)]
        self.assertTrue(validate_pressure_run(run, specs))
        altered = deepcopy(run)
        altered["steps"][0]["executions"][0]["recomputed_tokens"] = 248
        with self.assertRaises(ValueError):
            validate_pressure_run(altered, specs)

    def test_no_progress_watchdog_raises_and_restores_step(self):
        engine = CpuEngine(self.make_scheduler())
        observer = PressureObserver(engine, [])
        # Simulate a stuck loop to isolate the watchdog; real-resource recovery
        # is exercised by the other tests above.
        with observer:
            observer.replace(engine, "is_finished", lambda: False)
            observer.replace(engine, "step", lambda: None)
            with self.assertRaisesRegex(RuntimeError, "no new output/cache progress"):
                run_requests(engine, observer, [], lambda n: None, 10, 100, 3)
        self.assertNotIn("step", vars(engine))
        self.assertNotIn("is_finished", vars(engine))

    def test_reserve_deferral_episode_closes_at_first_real_allocation(self):
        scheduler = self.make_scheduler(num_kvcache_blocks=4, max_num_seqs=4,
            kv_admission_policy="reserve", kv_reserve_blocks=3, decode_queue_policy="round_robin")
        specs = [RequestSpec(i, [i+1]*8, 4, 0., "reserve_metric_test") for i in range(2)]
        engine = CpuEngine(scheduler)
        observer = PressureObserver(engine, specs)
        with observer:
            elapsed = run_requests(engine, observer, specs,
                lambda n: SamplingParams(max_tokens=n, ignore_eos=True), 10, 100, 10)
        stats = pressure_summary(observer, elapsed)
        self.assertGreater(stats["reserve_deferrals"], 0)
        self.assertEqual(stats["reserve_deferred_requests"], 1)
        self.assertEqual(len(observer.reserve_wait_events), 1)
        self.assertEqual(observer.pending_reserve_waits, {})
        event = observer.reserve_wait_events[0]
        self.assertEqual(event["request_id"], 1)
        self.assertEqual(event["attempts"], len(observer.reserve_deferral_events))
        self.assertEqual(event["duration_s"], event["end_time"] - event["start_time"])
        self.assertEqual(event["end_time"], observer.request_stats[1]["first_prefill_time"])
        self.assertEqual(event["duration_s"], observer.request_stats[1]["reserve_wait_s"])
        self.assertEqual(resource_snapshot(scheduler)["free_blocks"], 4)

    def test_zero_reserve_never_logs_false_deferrals(self):
        _, observer, elapsed = self.execute(count=10, blocks=4)
        stats = pressure_summary(observer, elapsed)
        self.assertGreater(stats["preemptions"], 0)
        self.assertEqual(stats["reserve_deferrals"], 0)
        self.assertEqual(stats["reserve_wait_total_s"], 0)
        self.assertIsNone(stats["reserve_wait_p95_s"])

    def test_actual_bypass_events_match_allocations_and_reject_duplicates(self):
        scheduler = self.make_scheduler(num_kvcache_blocks=4, max_num_seqs=4,
            kv_admission_policy="reserve", kv_reserve_blocks=2, kv_bounded_bypass=True,
            decode_queue_policy="round_robin")
        specs = [RequestSpec(i, [i+1]*length, 4, 0., "bypass_metric_test")
                 for i, length in enumerate((8, 504, 8, 8))]
        engine = CpuEngine(scheduler)
        observer = PressureObserver(engine, specs)
        with observer:
            elapsed = run_requests(engine, observer, specs,
                lambda n: SamplingParams(max_tokens=n, ignore_eos=True), 10, 100, 10)
        self.assertEqual(len(observer.bypass_events), 1)
        event = observer.bypass_events[0]
        self.assertEqual((event["head_request_id"], event["candidate_request_id"]), (1, 2))
        resources = resource_snapshot(scheduler)
        resources.update(released=True, block_ref_count_sum=0)
        run = dict(summary=dict(**pressure_identity("B8_bypass", 0), **pressure_summary(observer, elapsed),
                                status="ok", num_kvcache_blocks=4, torch_compile_frames_during_measurement=0),
                   requests=[{**r.to_dict(), **observer.request_stats[r.request_id]}
                             for r in observer.records.values()],
                   steps=observer.steps, resource_check=resources, environment=dict(physical_blocks=4),
                   preemption_events=observer.preemption_events, allocation_events=observer.allocation_events,
                   reserve_deferral_events=observer.reserve_deferral_events,
                   reserve_wait_events=observer.reserve_wait_events, bypass_events=observer.bypass_events)
        self.assertTrue(validate_pressure_run(run, specs))
        altered = deepcopy(run)
        altered["bypass_events"].append(deepcopy(event))
        altered["summary"]["actual_bypasses"] = 2
        with self.assertRaisesRegex(ValueError, "bypass"):
            validate_pressure_run(altered, specs)


class WorkloadTests(unittest.TestCase):
    def test_bypass_label_is_only_independent_switch(self):
        fifo = pressure_identity("B8", 0)
        bypass = pressure_identity("B8_bypass", 0)
        self.assertFalse(fifo.pop("kv_bounded_bypass"))
        self.assertTrue(bypass.pop("kv_bounded_bypass"))
        fifo.pop("policy")
        bypass.pop("policy")
        self.assertEqual(fifo, bypass)

    def test_admission_labels_preserve_rr_and_explicit_configuration(self):
        for label, reserve in (("B0", 0), ("B4", 4), ("B8", 8), ("B16", 16)):
            config = pressure_identity(label, 2)
            self.assertEqual((config["scheduling_policy"], config["decode_queue_policy"]),
                             ("prefill_first", "round_robin"))
            self.assertEqual((config["kv_admission_policy"], config["kv_reserve_blocks"]), ("reserve", reserve))
            self.assertEqual(config["repeat"], 2)

    def test_seed_ids_lengths_and_shared_prefix(self):
        class Tokenizer:
            def encode(self, text, add_special_tokens=False):
                return list(text.encode())
        one = make_workload(Tokenizer(), 16, 0)
        two = make_workload(Tokenizer(), 16, 0)
        self.assertEqual([s.to_dict() for s in one], [s.to_dict() for s in two])
        self.assertEqual([len(s.prompt_token_ids) for s in one[:2]], [248, 504])
        self.assertEqual(one[1].prompt_token_ids[:256], one[3].prompt_token_ids[:256])
        self.assertNotEqual(one[1].prompt_token_ids[256:], one[3].prompt_token_ids[256:])


if __name__ == "__main__":
    unittest.main()
