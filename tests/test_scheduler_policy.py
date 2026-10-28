"""CPU scheduler tests using real queues and KV block bookkeeping.

Only package initialization and Hugging Face config loading are isolated:
nanovllm.__init__ eagerly imports the CUDA inference stack. No scheduler,
sequence, or block manager methods are mocked.
"""

import importlib
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import torch


with patch.dict(sys.modules):
    package = ModuleType("nanovllm")
    package.__path__ = [str(Path(__file__).resolve().parents[1] / "nanovllm")]
    sys.modules["nanovllm"] = package
    Config = importlib.import_module("nanovllm.config").Config
    Scheduler = importlib.import_module("nanovllm.engine.scheduler").Scheduler
    sequence_module = importlib.import_module("nanovllm.engine.sequence")
    Sequence = sequence_module.Sequence
    SequenceStatus = sequence_module.SequenceStatus
    SamplingParams = importlib.import_module("nanovllm.sampling_params").SamplingParams


class SchedulerTestCase(unittest.TestCase):
    def setUp(self):
        model_dir = TemporaryDirectory()
        self.addCleanup(model_dir.cleanup)
        self.model_dir = model_dir.name
        # Match LLMEngine's initialization without constructing an engine.
        previous_block_size = Sequence.block_size
        Sequence.block_size = 256
        self.addCleanup(setattr, Sequence, "block_size", previous_block_size)

    def make_config(self, **kwargs):
        with patch.object(Config.__post_init__.__globals__["AutoConfig"],
                          "from_pretrained", return_value=SimpleNamespace(
                              dtype=torch.float32, max_position_embeddings=4096)):
            return Config(self.model_dir, **kwargs)

    def make_scheduler(self, **kwargs):
        options = dict(num_kvcache_blocks=8, max_num_batched_tokens=1024,
                       max_num_seqs=8, eos=999)
        options.update(kwargs)
        return Scheduler(self.make_config(**options))

    def make_seq(self, length=8, token=1, max_tokens=3, ignore_eos=False):
        return Sequence([token] * length, SamplingParams(
            max_tokens=max_tokens, ignore_eos=ignore_eos))

    def assert_blocks_consistent(self, scheduler, *finished):
        manager = scheduler.block_manager
        live = list(scheduler.waiting) + list(scheduler.running)
        self.assertEqual(len(live), len({seq.seq_id for seq in live}))
        refs = [0] * len(manager.blocks)
        for seq in live:
            for block_id in seq.block_table:
                refs[block_id] += 1
        used = {i for i, count in enumerate(refs) if count}
        free = set(range(len(refs))) - used
        self.assertEqual(manager.used_block_ids, used)
        self.assertEqual(set(manager.free_block_ids), free)
        self.assertEqual(len(manager.free_block_ids), len(free))
        self.assertEqual([block.ref_count for block in manager.blocks], refs)
        for seq in scheduler.waiting:
            self.assertEqual(seq.status, SequenceStatus.WAITING)
        for seq in scheduler.running:
            self.assertEqual(seq.status, SequenceStatus.RUNNING)
        for seq in finished:
            self.assertEqual(seq.status, SequenceStatus.FINISHED)
            self.assertEqual(seq.block_table, [])
            self.assertEqual(seq.num_cached_tokens, 0)
            self.assertNotIn(seq, live)

    def finish_all(self, scheduler, requests, limit=100):
        trace = []
        for _ in range(limit):
            if scheduler.is_finished():
                break
            seqs, is_prefill = scheduler.schedule()
            self.assertTrue(seqs, "Pending requests must make progress")
            trace.append((is_prefill, tuple(seq.seq_id for seq in seqs)))
            self.assert_blocks_consistent(scheduler)
            scheduler.postprocess(seqs, [17] * len(seqs), is_prefill)
            self.assert_blocks_consistent(scheduler)
        self.assertTrue(scheduler.is_finished(), "Scheduler did not drain")
        self.assert_blocks_consistent(scheduler, *requests)
        for seq in requests:
            self.assertEqual(seq.num_completion_tokens, seq.max_tokens)
        return trace


class PrefillFirstRegressionTests(SchedulerTestCase):
    def test_original_prefill_fifo_and_sequence_limit(self):
        scheduler = self.make_scheduler(max_num_seqs=2)
        requests = [self.make_seq(token=i) for i in range(1, 4)]
        for seq in requests:
            scheduler.add(seq)
        seqs, is_prefill = scheduler.schedule()
        self.assertTrue(is_prefill)
        self.assertEqual(seqs, requests[:2])
        self.assertEqual(list(scheduler.running), requests[:2])
        self.assertEqual(list(scheduler.waiting), requests[2:])
        self.assertEqual([seq.num_scheduled_tokens for seq in seqs], [8, 8])
        self.assert_blocks_consistent(scheduler)
        scheduler.postprocess(seqs, [11, 12], is_prefill)
        self.finish_all(scheduler, requests)

    def test_waiting_prefill_precedes_existing_running_decode(self):
        scheduler = self.make_scheduler()
        running = self.make_seq()
        waiting = self.make_seq(token=2)
        scheduler.add(running)
        seqs, is_prefill = scheduler.schedule()
        scheduler.postprocess(seqs, [11], is_prefill)
        scheduler.add(waiting)
        seqs, is_prefill = scheduler.schedule()
        self.assertTrue(is_prefill)
        self.assertEqual(seqs, [waiting])
        self.assertEqual(running.num_completion_tokens, 1)
        self.assertEqual(list(scheduler.running), [running, waiting])
        self.assertEqual(list(scheduler.waiting), [])
        self.assert_blocks_consistent(scheduler)
        scheduler.postprocess(seqs, [12], is_prefill)
        self.finish_all(scheduler, [running, waiting])

    def test_chunked_prefill_retains_blocks_and_only_samples_at_end(self):
        scheduler = self.make_scheduler(max_num_batched_tokens=256)
        seq = self.make_seq(length=600)
        scheduler.add(seq)
        block_table = None
        for chunk, cached in [(256, 256), (256, 512), (88, 600)]:
            seqs, is_prefill = scheduler.schedule()
            self.assertTrue(is_prefill)
            self.assertEqual(seqs, [seq])
            self.assertEqual(seq.num_scheduled_tokens, chunk)
            if block_table is None:
                block_table = list(seq.block_table)
            self.assertEqual(seq.block_table, block_table)
            self.assertEqual(len(block_table), 3)
            scheduler.postprocess(seqs, [17], is_prefill)
            self.assertEqual(seq.num_cached_tokens, cached)
            self.assertEqual(seq.num_completion_tokens, int(cached == 600))
            self.assertEqual(list(scheduler.waiting), [] if cached == 600 else [seq])
            self.assertEqual(list(scheduler.running), [seq] if cached == 600 else [])
            self.assert_blocks_consistent(scheduler)
        self.finish_all(scheduler, [seq])

    def test_only_first_sequence_in_batch_can_be_chunked(self):
        scheduler = self.make_scheduler(max_num_batched_tokens=256)
        first = self.make_seq(length=100)
        second = self.make_seq(length=200, token=2)
        scheduler.add(first)
        scheduler.add(second)
        seqs, is_prefill = scheduler.schedule()
        self.assertTrue(is_prefill)
        self.assertEqual(seqs, [first])
        self.assertEqual(second.block_table, [])
        self.assertEqual(list(scheduler.waiting), [second])
        self.assert_blocks_consistent(scheduler)
        scheduler.postprocess(seqs, [17], is_prefill)
        self.finish_all(scheduler, [first, second])

    def test_decode_preempts_tail_when_kv_cache_is_full_and_recovers(self):
        scheduler = self.make_scheduler(num_kvcache_blocks=2)
        first = self.make_seq(length=256, token=1)
        second = self.make_seq(length=256, token=2)
        scheduler.add(first)
        scheduler.add(second)
        seqs, is_prefill = scheduler.schedule()
        scheduler.postprocess(seqs, [11, 12], is_prefill)
        first_block = first.block_table[0]
        second_block = second.block_table[0]
        self.assertEqual(len(scheduler.block_manager.free_block_ids), 0)
        seqs, is_prefill = scheduler.schedule()
        self.assertFalse(is_prefill)
        self.assertEqual(seqs, [first])
        self.assertEqual(list(scheduler.running), [first])
        self.assertEqual(list(scheduler.waiting), [second])
        self.assertTrue(second.is_prefill)
        self.assertEqual(second.block_table, [])
        self.assertEqual(second.num_cached_tokens, 0)
        self.assertEqual(second.num_completion_tokens, 1)
        self.assertEqual(first.block_table, [first_block, second_block])
        self.assert_blocks_consistent(scheduler)
        scheduler.postprocess(seqs, [17], is_prefill)
        trace = self.finish_all(scheduler, [first, second])
        self.assertIn((True, (second.seq_id,)), trace)

    def test_completion_releases_blocks_for_max_tokens_and_eos(self):
        for token_id, max_tokens in [(17, 1), (999, 5)]:
            with self.subTest(token_id=token_id):
                scheduler = self.make_scheduler(num_kvcache_blocks=1)
                seq = self.make_seq(max_tokens=max_tokens)
                scheduler.add(seq)
                seqs, is_prefill = scheduler.schedule()
                self.assertEqual(len(scheduler.block_manager.used_block_ids), 1)
                scheduler.postprocess(seqs, [token_id], is_prefill)
                self.assertTrue(scheduler.is_finished())
                self.assertEqual(seq.completion_token_ids, [token_id])
                self.assert_blocks_consistent(scheduler, seq)

    def test_completed_prefix_blocks_can_be_reused(self):
        scheduler = self.make_scheduler(num_kvcache_blocks=3)
        first = self.make_seq(length=300, max_tokens=1)
        scheduler.add(first)
        seqs, is_prefill = scheduler.schedule()
        prefix_block = first.block_table[0]
        scheduler.postprocess(seqs, [17], is_prefill)
        self.assert_blocks_consistent(scheduler, first)
        second = self.make_seq(length=300, max_tokens=1)
        scheduler.add(second)
        seqs, is_prefill = scheduler.schedule()
        self.assertTrue(is_prefill)
        self.assertEqual(second.num_cached_tokens, 256)
        self.assertEqual(second.num_scheduled_tokens, 44)
        self.assertEqual(second.block_table[0], prefix_block)
        self.assert_blocks_consistent(scheduler)
        scheduler.postprocess(seqs, [17], is_prefill)
        self.assert_blocks_consistent(scheduler, first, second)


class SchedulingPolicyTests(SchedulerTestCase):
    policies = ("prefill_first", "decode_first")

    def test_config_default_and_supported_policies(self):
        self.assertEqual(self.make_config().scheduling_policy, "prefill_first")
        for policy in self.policies:
            with self.subTest(policy=policy):
                config = self.make_config(scheduling_policy=policy)
                self.assertEqual(config.scheduling_policy, policy)
                self.assertEqual(Scheduler(config).scheduling_policy, policy)

    def test_config_rejects_invalid_policies(self):
        for policy in ("", "fifo", "DECODE_FIRST", None, []):
            with self.subTest(policy=policy):
                with self.assertRaisesRegex(ValueError, "Unsupported scheduling policy"):
                    self.make_config(scheduling_policy=policy)

    def test_policy_execution_order_and_complete_drain(self):
        for policy in self.policies:
            with self.subTest(policy=policy):
                scheduler = self.make_scheduler(scheduling_policy=policy)
                first = self.make_seq(max_tokens=3)
                second = self.make_seq(token=2, max_tokens=2)
                scheduler.add(first)
                seqs, is_prefill = scheduler.schedule()
                self.assertTrue(is_prefill)
                self.assertEqual(seqs, [first])
                scheduler.postprocess(seqs, [17], is_prefill)
                scheduler.add(second)
                self.assertEqual(list(scheduler.running), [first])
                self.assertEqual(list(scheduler.waiting), [second])
                trace = self.finish_all(scheduler, [first, second])
                if policy == "prefill_first":
                    expected = [(True, (second.seq_id,)),
                                (False, (first.seq_id, second.seq_id)),
                                (False, (first.seq_id,))]
                else:
                    expected = [(False, (first.seq_id,)),
                                (False, (first.seq_id,)),
                                (True, (second.seq_id,)),
                                (False, (second.seq_id,))]
                self.assertEqual(trace, expected)

    def test_empty_queues_return_empty_decode_batch(self):
        for policy in self.policies:
            with self.subTest(policy=policy):
                scheduler = self.make_scheduler(scheduling_policy=policy)
                self.assertEqual(scheduler.schedule(), ([], False))
                self.assertTrue(scheduler.is_finished())
                self.assert_blocks_consistent(scheduler)

    def test_waiting_only_then_running_only(self):
        for policy in self.policies:
            with self.subTest(policy=policy):
                scheduler = self.make_scheduler(scheduling_policy=policy)
                seq = self.make_seq(max_tokens=3)
                scheduler.add(seq)
                trace = self.finish_all(scheduler, [seq])
                self.assertEqual(trace, [(True, (seq.seq_id,)),
                                         (False, (seq.seq_id,)),
                                         (False, (seq.seq_id,))])

    def test_chunked_prefill_and_homogeneous_batches_for_both_policies(self):
        for policy in self.policies:
            with self.subTest(policy=policy):
                scheduler = self.make_scheduler(scheduling_policy=policy,
                                                max_num_batched_tokens=128)
                long = self.make_seq(length=600)
                short = self.make_seq(length=17, token=2)
                scheduler.add(long)
                scheduler.add(short)
                trace = self.finish_all(scheduler, [long, short])
                self.assertEqual(trace, [(True, (long.seq_id,))] * 4 +
                                 [(True, (long.seq_id, short.seq_id))] +
                                 [(False, (long.seq_id, short.seq_id))] * 2)

    def test_blocked_prefill_falls_back_to_decode_and_releases_capacity(self):
        for policy in self.policies:
            with self.subTest(policy=policy):
                scheduler = self.make_scheduler(scheduling_policy=policy,
                                                num_kvcache_blocks=1)
                first = self.make_seq(max_tokens=2)
                second = self.make_seq(token=2, max_tokens=2)
                scheduler.add(first)
                seqs, is_prefill = scheduler.schedule()
                scheduler.postprocess(seqs, [17], is_prefill)
                scheduler.add(second)
                seqs, is_prefill = scheduler.schedule()
                self.assertFalse(is_prefill)
                self.assertEqual(seqs, [first])
                self.assertEqual(list(scheduler.waiting), [second])
                self.assertEqual(second.block_table, [])
                self.assert_blocks_consistent(scheduler)
                scheduler.postprocess(seqs, [17], is_prefill)
                self.assert_blocks_consistent(scheduler, first)
                trace = self.finish_all(scheduler, [first, second])
                self.assertEqual(trace[0], (True, (second.seq_id,)))

    def test_preempted_requests_recompute_and_finish_for_both_policies(self):
        for policy in self.policies:
            with self.subTest(policy=policy):
                scheduler = self.make_scheduler(scheduling_policy=policy,
                                                num_kvcache_blocks=2)
                first = self.make_seq(length=256)
                second = self.make_seq(length=256, token=2)
                scheduler.add(first)
                scheduler.add(second)
                seqs, is_prefill = scheduler.schedule()
                scheduler.postprocess(seqs, [17, 18], is_prefill)
                seqs, is_prefill = scheduler.schedule()
                self.assertFalse(is_prefill)
                self.assertEqual(seqs, [first])
                self.assertEqual(list(scheduler.waiting), [second])
                self.assertEqual(second.block_table, [])
                self.assertEqual(second.num_cached_tokens, 0)
                self.assertEqual(second.completion_token_ids, [18])
                self.assertTrue(second.is_prefill)
                self.assert_blocks_consistent(scheduler)
                scheduler.postprocess(seqs, [17], is_prefill)
                trace = self.finish_all(scheduler, [first, second])
                self.assertIn((True, (second.seq_id,)), trace)
                self.assertEqual(second.completion_token_ids, [18, 17, 17])

    def test_shared_prefix_reference_counts_survive_completion(self):
        for policy in self.policies:
            with self.subTest(policy=policy):
                scheduler = self.make_scheduler(scheduling_policy=policy,
                                                num_kvcache_blocks=3)
                first = self.make_seq(length=300, max_tokens=2)
                second = self.make_seq(length=300, max_tokens=3)
                scheduler.add(first)
                scheduler.add(second)
                seqs, is_prefill = scheduler.schedule()
                self.assertTrue(is_prefill)
                self.assertEqual(seqs, [first])
                scheduler.postprocess(seqs, [17], is_prefill)
                # Explicit prefill-first admission is necessary here to observe
                # live sharing; decode-first drains first before admitting second.
                if policy == "decode_first":
                    seqs, is_prefill = scheduler.schedule()
                    self.assertFalse(is_prefill)
                    scheduler.postprocess(seqs, [17], is_prefill)
                    self.assert_blocks_consistent(scheduler, first)
                seqs, is_prefill = scheduler.schedule()
                self.assertTrue(is_prefill)
                self.assertEqual(seqs, [second])
                prefix_block = second.block_table[0]
                self.assertEqual(second.num_cached_tokens, 256)
                expected_refs = 2 if policy == "prefill_first" else 1
                self.assertEqual(scheduler.block_manager.blocks[prefix_block].ref_count,
                                 expected_refs)
                self.assert_blocks_consistent(scheduler)
                scheduler.postprocess(seqs, [17], is_prefill)
                if policy == "prefill_first":
                    seqs, is_prefill = scheduler.schedule()
                    scheduler.postprocess(seqs, [17] * len(seqs), is_prefill)
                    self.assert_blocks_consistent(scheduler, first)
                    self.assertEqual(scheduler.block_manager.blocks[prefix_block].ref_count, 1)
                self.finish_all(scheduler, [first, second])

    def test_oversized_prompt_raises_instead_of_returning_empty_pending_batch(self):
        for policy in self.policies:
            with self.subTest(policy=policy):
                scheduler = self.make_scheduler(scheduling_policy=policy,
                                                num_kvcache_blocks=1)
                seq = self.make_seq(length=257)
                scheduler.add(seq)
                with self.assertRaisesRegex(RuntimeError, "Cannot schedule any request"):
                    scheduler.schedule()
                self.assertEqual(list(scheduler.waiting), [seq])
                self.assertEqual(seq.block_table, [])
                self.assertEqual(seq.num_completion_tokens, 0)
                self.assert_blocks_consistent(scheduler)

    def test_self_preemption_reports_unrecoverable_capacity_without_leaking(self):
        for policy in self.policies:
            with self.subTest(policy=policy):
                scheduler = self.make_scheduler(scheduling_policy=policy,
                                                num_kvcache_blocks=1)
                seq = self.make_seq(length=256)
                scheduler.add(seq)
                seqs, is_prefill = scheduler.schedule()
                scheduler.postprocess(seqs, [17], is_prefill)
                with self.assertRaisesRegex(RuntimeError, "Cannot schedule any request"):
                    scheduler.schedule()
                self.assertEqual(list(scheduler.waiting), [seq])
                self.assertEqual(list(scheduler.running), [])
                self.assertEqual(seq.num_cached_tokens, 0)
                self.assertEqual(seq.block_table, [])
                self.assertTrue(seq.is_prefill)
                self.assertEqual(seq.completion_token_ids, [17])
                self.assertFalse(scheduler.is_finished())
                self.assert_blocks_consistent(scheduler)


if __name__ == "__main__":
    unittest.main()
