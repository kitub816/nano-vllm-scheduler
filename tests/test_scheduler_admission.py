"""CPU-only admission tests using real queues, Sequence and BlockManager."""

from test_scheduler_policy import SchedulerTestCase, SequenceStatus


class AdmissionTests(SchedulerTestCase):
    def scheduler(self, **options):
        defaults = dict(num_kvcache_blocks=8, max_num_batched_tokens=512, max_num_seqs=4,
                        decode_queue_policy="round_robin", kv_admission_policy="reserve", kv_reserve_blocks=2)
        defaults.update(options)
        return self.make_scheduler(**defaults)

    def start(self, scheduler, seq):
        scheduler.add(seq)
        batch, prefill = scheduler.schedule()
        self.assertTrue(prefill)
        scheduler.postprocess(batch, [17] * len(batch), prefill)
        self.assert_blocks_consistent(scheduler)

    def test_config_defaults_valid_and_invalid(self):
        config = self.make_config()
        self.assertEqual((config.kv_admission_policy, config.kv_reserve_blocks), ("none", 0))
        self.make_config(kv_admission_policy="reserve", kv_reserve_blocks=16)
        for field, values in (("kv_admission_policy", (None, "unknown", True)),
                              ("kv_reserve_blocks", (-1, 1.5, 0., True, False, "4", None))):
            for value in values:
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    self.make_config(**{field: value})

    def test_zero_and_none_preserve_complete_legacy_trace(self):
        traces = []
        for policy, reserve in (("none", 0), ("none", 16), ("reserve", 0)):
            s = self.scheduler(num_kvcache_blocks=4, max_num_batched_tokens=256,
                               kv_admission_policy=policy, kv_reserve_blocks=reserve)
            requests = [self.make_seq(248, token=i+1, max_tokens=16, ignore_eos=True) for i in range(10)]
            indices = {seq.seq_id: i for i, seq in enumerate(requests)}
            for seq in requests:
                s.add(seq)
            trace = []
            for _ in range(1000):
                if s.is_finished():
                    break
                batch, prefill = s.schedule()
                trace.append((prefill, [indices[x.seq_id] for x in batch],
                              [x.num_scheduled_tokens for x in batch], len(s.block_manager.used_block_ids)))
                s.postprocess(batch, [17] * len(batch), prefill)
                self.assert_blocks_consistent(s)
            self.assertTrue(s.is_finished())
            traces.append(trace)
        self.assertEqual(traces[0], traces[1])
        self.assertEqual(traces[0], traces[2])

    def test_threshold_boundary_defers_prefill_and_advances_decode(self):
        for reserve, expected_prefill in ((0, True), (5, True), (6, False), (100, False)):
            with self.subTest(reserve=reserve):
                s = self.scheduler(kv_reserve_blocks=reserve)
                running = self.make_seq(8, token=1, max_tokens=4, ignore_eos=True)
                self.start(s, running)
                new = self.make_seq(504, token=2, max_tokens=4, ignore_eos=True)
                s.add(new)
                before = running.num_completion_tokens
                batch, prefill = s.schedule()
                self.assertEqual(prefill, expected_prefill)
                if not prefill:
                    self.assertEqual(batch, [running])
                    self.assertEqual(new.block_table, [])
                    self.assertEqual(list(s.waiting), [new])
                s.postprocess(batch, [17] * len(batch), prefill)
                if not prefill:
                    self.assertEqual(running.num_completion_tokens, before + 1)
                self.finish_all(s, [running, new], limit=100)

    def test_running_empty_bypasses_impossible_reserve_and_drains(self):
        s = self.scheduler(num_kvcache_blocks=3, kv_reserve_blocks=100)
        requests = [self.make_seq(504, token=i+1, max_tokens=16, ignore_eos=True) for i in range(4)]
        for seq in requests:
            s.add(seq)
        self.finish_all(s, requests, limit=300)

    def test_active_shared_prefix_reduces_physical_cost(self):
        s = self.scheduler(num_kvcache_blocks=4, kv_reserve_blocks=1)
        owner = self.make_seq(257, token=1, max_tokens=4, ignore_eos=True)
        self.start(s, owner)
        new = self.make_seq(257, token=1, max_tokens=4, ignore_eos=True)
        s.add(new)
        before = len(s.block_manager.free_block_ids)
        batch, prefill = s.schedule()
        self.assertTrue(prefill)
        self.assertEqual(batch, [new])
        self.assertEqual(before - len(s.block_manager.free_block_ids), 1)
        self.assertEqual(new.num_cached_tokens, 256)
        shared = owner.block_table[0]
        self.assertEqual(s.block_manager.blocks[shared].ref_count, 2)
        s.postprocess(batch, [17], prefill)
        self.finish_all(s, [owner, new])

    def test_free_retained_prefix_still_consumes_free_capacity(self):
        s = self.scheduler(num_kvcache_blocks=4, kv_reserve_blocks=2)
        completed = self.make_seq(257, token=1, max_tokens=1, ignore_eos=True)
        self.start(s, completed)
        running = self.make_seq(8, token=2, max_tokens=4, ignore_eos=True)
        self.start(s, running)
        new = self.make_seq(257, token=1, max_tokens=4, ignore_eos=True)
        self.assertEqual(s.block_manager.can_allocate(new), 1)
        s.add(new)
        batch, prefill = s.schedule()
        self.assertFalse(prefill)
        self.assertEqual(batch, [running])
        self.assertEqual(new.block_table, [])
        s.postprocess(batch, [17], prefill)
        self.finish_all(s, [running, new])

    def test_started_chunked_prefill_is_never_reserve_blocked(self):
        s = self.scheduler(num_kvcache_blocks=4, max_num_batched_tokens=256, kv_reserve_blocks=100)
        chunk = self.make_seq(504, token=1, max_tokens=4, ignore_eos=True)
        self.start(s, chunk)
        self.assertEqual(chunk.num_cached_tokens, 256)
        self.assertTrue(chunk.block_table)
        # Add a valid running owner to exercise the guard with held chunk blocks.
        owner = self.make_seq(8, token=2, max_tokens=4, ignore_eos=True)
        s.block_manager.allocate(owner, 0)
        owner.status = SequenceStatus.RUNNING
        owner.num_scheduled_tokens = 8
        s.running.append(owner)
        s.postprocess([owner], [17], True)
        batch, prefill = s.schedule()
        self.assertTrue(prefill)
        self.assertEqual(batch, [chunk])
        s.postprocess(batch, [17], prefill)
        self.finish_all(s, [chunk, owner])

    def test_recovery_bypasses_reserve_without_losing_outputs(self):
        s = self.scheduler(num_kvcache_blocks=4, kv_reserve_blocks=0)
        owner = self.make_seq(248, token=1, max_tokens=16, ignore_eos=True)
        victim = self.make_seq(248, token=2, max_tokens=16, ignore_eos=True)
        for seq in (owner, victim):
            self.start(s, seq)
        s.running.remove(victim)
        s.preempt(victim)
        previous = list(victim.completion_token_ids)
        s.kv_reserve_blocks = 100
        batch, prefill = s.schedule()
        self.assertTrue(prefill)
        self.assertEqual(batch, [victim])
        s.postprocess(batch, [17], prefill)
        self.assertEqual(victim.completion_token_ids[:len(previous)], previous)
        self.finish_all(s, [owner, victim], limit=300)

    def test_decode_expansion_preemption_remains_progress_fallback(self):
        s = self.scheduler(num_kvcache_blocks=2, kv_reserve_blocks=0)
        requests = [self.make_seq(256, token=i+1, max_tokens=2, ignore_eos=True) for i in range(2)]
        for seq in requests:
            s.add(seq)
        batch, prefill = s.schedule()
        s.postprocess(batch, [17] * len(batch), prefill)
        new = self.make_seq(8, token=3, max_tokens=2, ignore_eos=True)
        s.add(new)
        s.kv_reserve_blocks = 2
        batch, prefill = s.schedule()
        self.assertFalse(prefill)
        self.assertEqual(len(batch), 1)
        self.assertTrue(any(x.num_completion_tokens for x in s.waiting))
        s.postprocess(batch, [17], prefill)
        self.finish_all(s, requests + [new])

    def test_rr_order_remains_fair_after_admission(self):
        s = self.scheduler(num_kvcache_blocks=16, kv_reserve_blocks=4)
        requests = [self.make_seq(8, token=i+1, max_tokens=8, ignore_eos=True) for i in range(6)]
        for seq in requests:
            s.add(seq)
        while s.waiting:
            batch, prefill = s.schedule()
            self.assertTrue(prefill)
            s.postprocess(batch, [17] * len(batch), prefill)
        first, prefill = s.schedule()
        self.assertFalse(prefill)
        self.assertEqual(first, requests[:4])
        s.postprocess(first, [17] * len(first), False)
        second, _ = s.schedule()
        self.assertEqual(second, requests[4:] + requests[:2])
        s.postprocess(second, [17] * len(second), False)
        self.finish_all(s, requests)

    def test_unrecoverable_capacity_raises_instead_of_spinning(self):
        s = self.scheduler(num_kvcache_blocks=2, kv_reserve_blocks=100)
        s.add(self.make_seq(513, max_tokens=2))
        with self.assertRaisesRegex(RuntimeError, "Cannot schedule"):
            s.schedule()
        self.assert_blocks_consistent(s)

    def test_failed_fair_prefill_does_not_reset_decode_batch_counter(self):
        s = self.scheduler(scheduling_policy="decode_first", max_consecutive_decode_steps=2,
                           kv_reserve_blocks=100)
        owner = self.make_seq(8, token=1, max_tokens=4, ignore_eos=True)
        self.start(s, owner)
        new = self.make_seq(8, token=2, max_tokens=4, ignore_eos=True)
        s.add(new)
        s.consecutive_decode_steps = 2
        batch, prefill = s.schedule()
        self.assertFalse(prefill)
        self.assertEqual(batch, [owner])
        self.assertEqual(s.consecutive_decode_steps, 3)
        s.postprocess(batch, [17], False)
        self.finish_all(s, [owner, new])
