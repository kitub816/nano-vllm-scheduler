"""Bounded bypass regression tests: real queues, allocations and references."""

from test_scheduler_policy import SchedulerTestCase, SequenceStatus


class BoundedBypassTests(SchedulerTestCase):
    def scheduler(self, **options):
        defaults = dict(num_kvcache_blocks=4, max_num_batched_tokens=512, max_num_seqs=4,
                        decode_queue_policy="round_robin", kv_admission_policy="reserve",
                        kv_reserve_blocks=2, kv_bounded_bypass=True)
        defaults.update(options)
        return self.make_scheduler(**defaults)

    def start(self, s, seq):
        s.add(seq)
        batch, prefill = s.schedule()
        self.assertTrue(prefill)
        s.postprocess(batch, [17] * len(batch), prefill)
        self.assert_blocks_consistent(s)

    def blocked_pair(self, **options):
        s = self.scheduler(**options)
        owner = self.make_seq(8, token=1, max_tokens=5, ignore_eos=True)
        self.start(s, owner)
        head = self.make_seq(504, token=2, max_tokens=3, ignore_eos=True)
        second = self.make_seq(8, token=3, max_tokens=3, ignore_eos=True)
        s.add(head)
        s.add(second)
        return s, owner, head, second

    def test_config_default_and_exact_boolean_validation(self):
        self.assertIs(self.make_config().kv_bounded_bypass, False)
        self.make_config(kv_bounded_bypass=True)
        for value in (0, 1, "true", None, 0., [], {}):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.make_config(kv_bounded_bypass=value)

    def test_disabled_matches_original_fifo_full_trace(self):
        traces = []
        # Omitted option and explicit False must produce identical real traces.
        for options in ({}, {"kv_bounded_bypass": False}):
            s = self.make_scheduler(num_kvcache_blocks=4, max_num_batched_tokens=256,
                max_num_seqs=4, kv_admission_policy="reserve", kv_reserve_blocks=2,
                decode_queue_policy="round_robin", **options)
            seqs = [self.make_seq(248 if i % 2 == 0 else 504, token=i+1,
                                 max_tokens=16, ignore_eos=True) for i in range(8)]
            for seq in seqs:
                s.add(seq)
            trace = []
            for _ in range(500):
                if s.is_finished():
                    break
                batch, prefill = s.schedule()
                trace.append((prefill, [seqs.index(x) for x in batch],
                              [x.num_scheduled_tokens for x in batch],
                              [seqs.index(x) for x in s.waiting],
                              [seqs.index(x) for x in s.running],
                              [b.ref_count for b in s.block_manager.blocks]))
                s.postprocess(batch, [17] * len(batch), prefill)
                self.assert_blocks_consistent(s)
            self.assertTrue(s.is_finished())
            self.assert_blocks_consistent(s, *seqs)
            traces.append(trace)
        self.assertEqual(traces[0], traces[1])

    def test_reserve_rejected_head_allows_second_once_then_drains(self):
        s, owner, head, second = self.blocked_pair()
        third = self.make_seq(8, token=4, max_tokens=3, ignore_eos=True)
        s.add(third)
        batch, prefill = s.schedule()
        self.assertTrue(prefill)
        self.assertEqual(batch, [second])
        self.assertEqual(list(s.waiting), [head, third])
        self.assertEqual(len(s.block_manager.free_block_ids), 2)
        s.postprocess(batch, [17], prefill)
        # Even when another small request fits, this head's allowance is spent.
        s.kv_reserve_blocks = 1
        batch, prefill = s.schedule()
        self.assertFalse(prefill)
        self.assertEqual(batch, [owner, second])
        self.assertFalse(third.block_table)
        s.postprocess(batch, [17] * len(batch), prefill)
        self.finish_all(s, [owner, head, second, third], limit=200)
        self.assertFalse(s._bypassed_before_admission)

    def test_only_second_is_considered_never_third(self):
        s, owner, head, second = self.blocked_pair()
        s.waiting.pop()
        second = self.make_seq(504, token=3, ignore_eos=True)
        s.add(second)
        third = self.make_seq(8, token=4, ignore_eos=True)
        s.add(third)
        batch, prefill = s.schedule()
        self.assertFalse(prefill)
        self.assertEqual(batch, [owner])
        self.assertEqual(list(s.waiting), [head, second, third])
        s.postprocess(batch, [17], prefill)
        self.finish_all(s, [owner, head, second, third], limit=200)

    def test_failed_candidate_does_not_spend_allowance(self):
        s, owner, head, second = self.blocked_pair(kv_reserve_blocks=3)
        batch, prefill = s.schedule()
        self.assertFalse(prefill)
        self.assertFalse(s._bypassed_before_admission)
        s.postprocess(batch, [17], prefill)
        s.kv_reserve_blocks = 2
        batch, prefill = s.schedule()
        self.assertTrue(prefill)
        self.assertEqual(batch, [second])
        s.postprocess(batch, [17], prefill)
        self.finish_all(s, [owner, head, second])

    def test_held_chunk_head_cannot_be_bypassed(self):
        s = self.scheduler(max_num_batched_tokens=128, kv_reserve_blocks=100)
        head = self.make_seq(504, token=1, ignore_eos=True)
        self.start(s, head)
        second = self.make_seq(8, token=2, ignore_eos=True)
        s.add(second)
        batch, prefill = s.schedule()
        self.assertTrue(prefill)
        self.assertEqual(batch, [head])
        self.assertFalse(second.block_table)
        s.postprocess(batch, [17], prefill)
        self.finish_all(s, [head, second])

    def test_bypassing_chunk_keeps_priority_until_chunk_completed(self):
        s, owner, head, second = self.blocked_pair(max_num_batched_tokens=4)
        # start() consumed only a chunk; finish owner prefill before testing reserve.
        while owner in s.waiting:
            batch, prefill = s.schedule()
            s.postprocess(batch, [17] * len(batch), prefill)
        batch, prefill = s.schedule()
        self.assertTrue(prefill)
        self.assertEqual(batch, [second])
        self.assertEqual(list(s.waiting), [second, head])
        s.postprocess(batch, [17], prefill)
        batch, prefill = s.schedule()
        self.assertTrue(prefill)
        self.assertEqual(batch, [second])
        self.assertEqual(second.num_scheduled_tokens, 4)
        s.postprocess(batch, [17], prefill)
        self.finish_all(s, [owner, head, second], limit=1000)

    def test_preempted_head_and_second_are_never_bypassed(self):
        for recovery_at_head in (True, False):
            s = self.scheduler(kv_reserve_blocks=0)
            owner = self.make_seq(8, token=1, max_tokens=5, ignore_eos=True)
            victim = self.make_seq(248, token=2, max_tokens=5, ignore_eos=True)
            self.start(s, owner)
            self.start(s, victim)
            s.running.remove(victim)
            s.preempt(victim)
            new = self.make_seq(504, token=3, ignore_eos=True)
            if recovery_at_head:
                s.add(new)
            else:
                s.waiting.appendleft(new)
            s.kv_reserve_blocks = 2
            batch, prefill = s.schedule()
            self.assertEqual(prefill, recovery_at_head)
            self.assertEqual(batch, [victim] if recovery_at_head else [owner])
            s.postprocess(batch, [17] * len(batch), prefill)
            self.finish_all(s, [owner, victim, new], limit=200)

    def test_active_prefix_cost_allows_second_and_refs_release(self):
        s = self.scheduler(num_kvcache_blocks=5, kv_reserve_blocks=2)
        owner = self.make_seq(257, token=1, max_tokens=4, ignore_eos=True)
        self.start(s, owner)
        head = self.make_seq(504, token=2, ignore_eos=True)
        second = self.make_seq(257, token=1, ignore_eos=True)
        s.add(head)
        s.add(second)
        batch, prefill = s.schedule()
        self.assertTrue(prefill)
        self.assertEqual(batch, [second])
        self.assertEqual(second.num_cached_tokens, 256)
        self.assertEqual(len(s.block_manager.free_block_ids), 2)
        self.assertEqual(s.block_manager.blocks[owner.block_table[0]].ref_count, 2)
        s.postprocess(batch, [17], prefill)
        self.finish_all(s, [owner, head, second])

    def test_idle_cached_prefix_still_costs_free_blocks(self):
        s = self.scheduler(num_kvcache_blocks=5, kv_reserve_blocks=3)
        completed = self.make_seq(257, token=1, max_tokens=1, ignore_eos=True)
        self.start(s, completed)
        owner = self.make_seq(8, token=2, max_tokens=4, ignore_eos=True)
        self.start(s, owner)
        head = self.make_seq(504, token=3, ignore_eos=True)
        second = self.make_seq(257, token=1, ignore_eos=True)
        self.assertEqual(s.block_manager.can_allocate(second), 1)
        s.add(head)
        s.add(second)
        batch, prefill = s.schedule()
        self.assertFalse(prefill)
        self.assertEqual(batch, [owner])
        self.assertFalse(second.block_table)
        s.postprocess(batch, [17], prefill)
        self.finish_all(s, [owner, head, second])

    def test_physical_failure_never_triggers_bypass_and_impossible_raises(self):
        s, owner, head, second = self.blocked_pair(num_kvcache_blocks=2, kv_reserve_blocks=1)
        batch, prefill = s.schedule()
        self.assertFalse(prefill)
        self.assertEqual(batch, [owner])
        self.assertFalse(second.block_table)
        s.postprocess(batch, [17], prefill)
        self.finish_all(s, [owner, head, second], limit=200)
        s = self.scheduler(num_kvcache_blocks=1)
        s.add(self.make_seq(257))
        s.add(self.make_seq(8, token=2))
        with self.assertRaisesRegex(RuntimeError, "Cannot schedule"):
            s.schedule()
        self.assert_blocks_consistent(s)

    def test_candidate_token_budget_and_sequence_limit(self):
        for limit in (1, 4):
            s = self.scheduler(num_kvcache_blocks=5, max_num_batched_tokens=16,
                               max_num_seqs=limit)
            owner = self.make_seq(8, token=1, max_tokens=4, ignore_eos=True)
            self.start(s, owner)
            first = self.make_seq(8, token=2, ignore_eos=True)
            head = self.make_seq(504, token=3, ignore_eos=True)
            second = self.make_seq(16, token=4, ignore_eos=True)
            for seq in (first, head, second):
                s.add(seq)
            batch, prefill = s.schedule()
            self.assertTrue(prefill)
            self.assertEqual(batch, [first])
            self.assertLessEqual(sum(x.num_scheduled_tokens for x in batch), 16)
            self.assertFalse(second.block_table)
            self.assertFalse(s._bypassed_before_admission)
            s.postprocess(batch, [17], prefill)
            self.finish_all(s, [owner, first, head, second], limit=300)

    def test_rr_fair2_and_preemption_recovery_still_drain(self):
        for fairness in (0, 2):
            s = self.scheduler(num_kvcache_blocks=3, kv_reserve_blocks=1,
                max_num_batched_tokens=256, scheduling_policy="decode_first" if fairness else "prefill_first",
                max_consecutive_decode_steps=fairness, max_num_seqs=2)
            requests = [self.make_seq(256 if i % 2 else 248, token=i+1,
                                     max_tokens=16, ignore_eos=True) for i in range(8)]
            victims = []
            original = s.preempt
            def preempt(seq):
                victims.append(seq)
                return original(seq)
            s.preempt = preempt
            for seq in requests:
                s.add(seq)
            self.finish_all(s, requests, limit=1000)
            self.assertTrue(victims)
            self.assertFalse(s._bypassed_before_admission)

    def test_cached_head_fits_token_budget_but_candidate_cannot_chunk_second(self):
        s = self.scheduler(num_kvcache_blocks=8, max_num_batched_tokens=16, kv_reserve_blocks=4)
        cached = self.make_seq(513, token=1, max_tokens=1, ignore_eos=True)
        s.add(cached)
        self.finish_all(s, [cached])
        owner = self.make_seq(8, token=2, max_tokens=4, ignore_eos=True)
        self.start(s, owner)
        first = self.make_seq(8, token=3, ignore_eos=True)
        head = self.make_seq(513, token=1, ignore_eos=True)
        candidate = self.make_seq(16, token=4, ignore_eos=True)
        for seq in (first, head, candidate):
            s.add(seq)
        # Head has q=1 after two idle cache hits but still costs three free
        # blocks. The candidate costs one block, yet q=16 exceeds remaining=8.
        self.assertEqual(s.block_manager.can_allocate(head), 2)
        batch, prefill = s.schedule()
        self.assertTrue(prefill)
        self.assertEqual(batch, [first])
        self.assertEqual(list(s.waiting), [head, candidate])
        self.assertFalse(candidate.block_table)
        self.assertFalse(s._bypassed_before_admission)
        s.postprocess(batch, [17], prefill)
        # With a fresh token budget the very same candidate can bypass once.
        batch, prefill = s.schedule()
        self.assertTrue(prefill)
        self.assertEqual(batch, [candidate])
        self.assertEqual(candidate.num_scheduled_tokens, 16)
        s.postprocess(batch, [17], prefill)
        self.finish_all(s, [owner, first, head, candidate])

    def test_enabled_does_not_change_rr_or_fair2_batch_order(self):
        s = self.scheduler(num_kvcache_blocks=16, kv_reserve_blocks=1, max_num_seqs=4,
                           scheduling_policy="decode_first", max_consecutive_decode_steps=2)
        requests = [self.make_seq(8, token=i+1, max_tokens=8, ignore_eos=True) for i in range(6)]
        for seq in requests:
            s.add(seq)
        batch, prefill = s.schedule()
        self.assertEqual(batch, requests[:4])
        s.postprocess(batch, [17]*4, prefill)
        for _ in range(2):
            batch, prefill = s.schedule()
            self.assertFalse(prefill)
            self.assertEqual(batch, requests[:4])
            s.postprocess(batch, [17]*4, prefill)
        batch, prefill = s.schedule()
        self.assertTrue(prefill)
        self.assertEqual(batch, requests[4:])
        self.assertEqual(s.consecutive_decode_steps, 0)
        s.postprocess(batch, [17]*2, prefill)
        batch, prefill = s.schedule()
        self.assertEqual(batch, requests[:4])
        s.postprocess(batch, [17]*4, prefill)
        batch, prefill = s.schedule()
        self.assertEqual(batch, requests[4:] + requests[:2])
        s.postprocess(batch, [17]*4, prefill)
        self.finish_all(s, requests)
