"""Decode queue policy tests using real Scheduler/Sequence/BlockManager state."""

import unittest

from test_scheduler_policy import SchedulerTestCase, SequenceStatus


class DecodeQueuePolicyTests(SchedulerTestCase):
    def make_running(self, policy="round_robin", max_tokens=6, **kwargs):
        options = dict(max_num_seqs=4, num_kvcache_blocks=16, decode_queue_policy=policy)
        options.update(kwargs)
        scheduler = self.make_scheduler(**options)
        requests = [self.make_seq(token=i + 1, max_tokens=max_tokens) for i in range(6)]
        for seq in requests:
            scheduler.add(seq)
        while scheduler.waiting:
            seqs, is_prefill = scheduler.schedule()
            self.assertTrue(is_prefill)
            scheduler.postprocess(seqs, [17] * len(seqs), is_prefill)
            self.assert_blocks_consistent(scheduler)
        self.assertEqual(list(scheduler.running), requests)
        return scheduler, requests

    def test_default_head_first_and_invalid_policies(self):
        self.assertEqual(self.make_config().decode_queue_policy, "head_first")
        for value in ("head_first", "round_robin"):
            self.assertEqual(self.make_config(decode_queue_policy=value).decode_queue_policy, value)
        for value in (None, "", "fifo", "ROUND_ROBIN", 0, True, []):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "decode_queue_policy"):
                    self.make_config(decode_queue_policy=value)

    def test_six_running_requests_rotate_in_batches_of_four(self):
        scheduler, requests = self.make_running()
        expected = [(requests[:4], requests[4:] + requests[:4]),
                    (requests[4:] + requests[:2], requests[2:4] + requests[4:] + requests[:2]),
                    (requests[2:4] + requests[4:], requests)]
        served = {seq.seq_id: 0 for seq in requests}
        for batch, running in expected:
            seqs, is_prefill = scheduler.schedule()
            self.assertFalse(is_prefill)
            self.assertEqual(seqs, batch)
            self.assertEqual(list(scheduler.running), running)
            self.assert_blocks_consistent(scheduler)
            scheduler.postprocess(seqs, [17] * len(seqs), is_prefill)
            for seq in seqs:
                served[seq.seq_id] += 1
            self.assert_blocks_consistent(scheduler)
        self.assertEqual(set(served.values()), {2})
        self.finish_all(scheduler, requests)

    def test_explicit_and_default_head_first_preserve_queue_priority(self):
        for explicit in (False, True):
            with self.subTest(explicit=explicit):
                options = {"decode_queue_policy": "head_first"} if explicit else {}
                scheduler = self.make_scheduler(max_num_seqs=4, **options)
                requests = [self.make_seq(token=i + 1, max_tokens=5) for i in range(6)]
                for seq in requests:
                    scheduler.add(seq)
                while scheduler.waiting:
                    seqs, is_prefill = scheduler.schedule()
                    scheduler.postprocess(seqs, [17] * len(seqs), is_prefill)
                for _ in range(2):
                    seqs, is_prefill = scheduler.schedule()
                    self.assertFalse(is_prefill)
                    self.assertEqual(seqs, requests[:4])
                    self.assertEqual(list(scheduler.running), requests)
                    scheduler.postprocess(seqs, [17] * len(seqs), is_prefill)
                    self.assert_blocks_consistent(scheduler)
                self.assertEqual([seq.num_completion_tokens for seq in requests[4:]], [1, 1])
                self.finish_all(scheduler, requests)

    def test_completed_requests_are_removed_from_rotated_tail(self):
        scheduler, requests = self.make_running(max_tokens=2)
        seqs, is_prefill = scheduler.schedule()
        self.assertEqual(seqs, requests[:4])
        self.assertEqual(list(scheduler.running), requests[4:] + requests[:4])
        scheduler.postprocess(seqs, [17] * len(seqs), is_prefill)
        self.assertEqual(list(scheduler.running), requests[4:])
        self.assert_blocks_consistent(scheduler, *requests[:4])
        self.finish_all(scheduler, requests)

    def test_each_request_is_scheduled_at_most_once_per_batch(self):
        scheduler = self.make_scheduler(decode_queue_policy="round_robin", max_num_seqs=4)
        seq = self.make_seq(max_tokens=4)
        scheduler.add(seq)
        trace = self.finish_all(scheduler, [seq])
        self.assertEqual([prefill for prefill, _ in trace], [True, False, False, False])
        self.assertTrue(all(ids == (seq.seq_id,) for _, ids in trace))

    def test_waiting_prefill_priority_is_unchanged(self):
        scheduler, requests = self.make_running()
        waiting = self.make_seq(token=99)
        scheduler.add(waiting)
        seqs, is_prefill = scheduler.schedule()
        self.assertTrue(is_prefill)
        self.assertEqual(seqs, [waiting])
        self.assertEqual(list(scheduler.running), requests + [waiting])
        scheduler.postprocess(seqs, [17], is_prefill)
        self.finish_all(scheduler, requests + [waiting])

    def test_decode_first_fair2_keeps_cross_phase_counter_semantics(self):
        scheduler, requests = self.make_running()
        # The six requests have been admitted through real Prefill scheduling.
        scheduler.scheduling_policy = "decode_first"
        scheduler.max_consecutive_decode_steps = 2
        waiting = self.make_seq(token=99)
        scheduler.add(waiting)
        for count, expected in [(1, requests[:4]), (2, requests[4:] + requests[:2])]:
            seqs, is_prefill = scheduler.schedule()
            self.assertFalse(is_prefill)
            self.assertEqual(seqs, expected)
            self.assertEqual(scheduler.consecutive_decode_steps, count)
            self.assertEqual(list(scheduler.waiting), [waiting])
            scheduler.postprocess(seqs, [17] * len(seqs), is_prefill)
            self.assert_blocks_consistent(scheduler)
        seqs, is_prefill = scheduler.schedule()
        self.assertTrue(is_prefill)
        self.assertEqual(seqs, [waiting])
        self.assertEqual(scheduler.consecutive_decode_steps, 0)
        scheduler.postprocess(seqs, [17], is_prefill)
        self.finish_all(scheduler, requests + [waiting])

    def test_chunked_prefill_and_round_robin_both_drain(self):
        scheduler = self.make_scheduler(decode_queue_policy="round_robin",
                                        max_num_batched_tokens=128, max_num_seqs=4)
        requests = [self.make_seq(length=300, token=1), self.make_seq(length=64, token=2)]
        for seq in requests:
            scheduler.add(seq)
        trace = self.finish_all(scheduler, requests)
        self.assertEqual([prefill for prefill, _ in trace[:3]], [True, True, True])
        self.assertEqual(requests[0].num_completion_tokens, 3)

    def test_kv_pressure_preempts_tail_once_then_recovers(self):
        for queue_policy in ("head_first", "round_robin"):
            with self.subTest(queue_policy=queue_policy):
                scheduler = self.make_scheduler(decode_queue_policy=queue_policy,
                                                max_num_seqs=4, num_kvcache_blocks=6)
                requests = [self.make_seq(length=256, token=i + 1, max_tokens=3) for i in range(6)]
                for seq in requests:
                    scheduler.add(seq)
                while scheduler.waiting:
                    seqs, is_prefill = scheduler.schedule()
                    scheduler.postprocess(seqs, [17] * len(seqs), is_prefill)
                seqs, is_prefill = scheduler.schedule()
                self.assertFalse(is_prefill)
                self.assertEqual(seqs, requests[:3])
                self.assertEqual(list(scheduler.waiting), requests[3:])
                self.assertEqual(list(scheduler.running), requests[:3])
                for seq in requests[3:]:
                    self.assertEqual(seq.status, SequenceStatus.WAITING)
                    self.assertEqual(seq.block_table, [])
                    self.assertEqual(seq.num_cached_tokens, 0)
                    self.assertEqual(seq.completion_token_ids, [17])
                    self.assertTrue(seq.is_prefill)
                self.assert_blocks_consistent(scheduler)
                scheduler.postprocess(seqs, [17] * len(seqs), is_prefill)
                trace = self.finish_all(scheduler, requests)
                self.assertTrue(any(prefill for prefill, _ in trace))

    def test_failed_self_preemption_retains_request_and_releases_blocks(self):
        scheduler = self.make_scheduler(decode_queue_policy="round_robin", num_kvcache_blocks=1)
        seq = self.make_seq(length=256)
        scheduler.add(seq)
        seqs, is_prefill = scheduler.schedule()
        scheduler.postprocess(seqs, [17], is_prefill)
        with self.assertRaisesRegex(RuntimeError, "Cannot schedule any request"):
            scheduler.schedule()
        self.assertEqual(list(scheduler.waiting), [seq])
        self.assertEqual(list(scheduler.running), [])
        self.assertTrue(seq.is_prefill)
        self.assertEqual(seq.completion_token_ids, [17])
        self.assert_blocks_consistent(scheduler)

    def test_preemption_with_unselected_survivor_keeps_survivor_ahead(self):
        scheduler = self.make_scheduler(decode_queue_policy="round_robin",
                                        max_num_seqs=4, num_kvcache_blocks=6)
        requests = [self.make_seq(length=256 if i == 0 else 8, token=i + 1, max_tokens=4)
                    for i in range(6)]
        for seq in requests:
            scheduler.add(seq)
        while scheduler.waiting:
            seqs, is_prefill = scheduler.schedule()
            scheduler.postprocess(seqs, [17] * len(seqs), is_prefill)
        seqs, is_prefill = scheduler.schedule()
        self.assertFalse(is_prefill)
        self.assertEqual(seqs, requests[:4])
        self.assertEqual(list(scheduler.waiting), [requests[5]])
        self.assertEqual(list(scheduler.running), [requests[4]] + requests[:4])
        self.assertEqual(requests[5].block_table, [])
        self.assertEqual(requests[5].num_cached_tokens, 0)
        self.assert_blocks_consistent(scheduler)
        scheduler.postprocess(seqs, [17] * len(seqs), is_prefill)
        # The waiting victim cannot yet be allocated. Decode falls back to
        # the surviving unselected request before last round's successful batch.
        seqs, is_prefill = scheduler.schedule()
        self.assertFalse(is_prefill)
        self.assertEqual(seqs, [requests[4]] + requests[:3])
        self.assertEqual(list(scheduler.running), [requests[3], requests[4]] + requests[:3])
        self.assert_blocks_consistent(scheduler)
        scheduler.postprocess(seqs, [17] * len(seqs), is_prefill)
        self.finish_all(scheduler, requests)

    def test_shared_prefix_reference_counts_survive_rotation_and_completion(self):
        scheduler = self.make_scheduler(decode_queue_policy="round_robin",
                                        max_num_seqs=1, num_kvcache_blocks=3)
        first = self.make_seq(length=300, max_tokens=3)
        second = self.make_seq(length=300, max_tokens=3)
        scheduler.add(first)
        scheduler.add(second)
        for _ in range(2):
            seqs, is_prefill = scheduler.schedule()
            self.assertTrue(is_prefill)
            scheduler.postprocess(seqs, [17], is_prefill)
        shared = first.block_table[0]
        self.assertEqual(second.block_table[0], shared)
        self.assertEqual(scheduler.block_manager.blocks[shared].ref_count, 2)
        for seq in (first, second, first, second):
            seqs, is_prefill = scheduler.schedule()
            self.assertFalse(is_prefill)
            self.assertEqual(seqs, [seq])
            scheduler.postprocess(seqs, [17], is_prefill)
            self.assert_blocks_consistent(scheduler)
        self.assertTrue(scheduler.is_finished())
        self.assert_blocks_consistent(scheduler, first, second)


if __name__ == "__main__":
    unittest.main()
