"""CPU-only profile planning/integrity tests; fake forward never supplies GPU data."""

from collections import Counter, defaultdict
from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

from benchmarks.prefill_profile import (ProfileCase, TimingProbe, build_cases, prepare_case,
                                        require_released, sample_case, summarize_samples,
                                        validate_samples, verify_results, write_results)
from test_scheduler_policy import SchedulerTestCase, Sequence, SamplingParams


class ProfileDataTests(unittest.TestCase):
    def row(self, case, phase, repeat, duration):
        stop = 10 + duration / 1000
        blocks_each = (case.c + case.q_per_request + 255) // 256
        blocks = blocks_each * case.batch_size
        return dict(**case.to_dict(), phase=phase, repeat=repeat, case_order=0,
                    max_num_batched_tokens=case.q, attention_kv_path="paged_kv" if case.c else "direct_qkv",
                    q_each=[case.q_per_request] * case.batch_size, c_each=[case.c] * case.batch_size,
                    block_tables=[list(range(i * blocks_each, (i + 1) * blocks_each))
                                  for i in range(case.batch_size)], block_size=256, kv_total_blocks=16,
                    kv_used_before=blocks, kv_used_after_schedule=blocks, kv_free_after=16,
                    kv_used_after=0, kv_ref_count_after=0, output_tokens=case.batch_size, compile_frames=0,
                    wall_start_perf_counter=10, wall_end_perf_counter=stop,
                    step_wall_ms=(stop - 10) * 1000, runner_wall_ms=duration * .9,
                    prepare_wall_ms=duration * .1, model_submit_wall_ms=duration * .5,
                    cuda_model_stream_ms=duration * .6)

    def test_default_grid_q_is_total_and_model_limits(self):
        cases, skipped = build_cases([16, 32, 64, 128, 256], [0, 64, 128, 256], [1, 2, 4], 1024)
        self.assertEqual(len(cases), 60)
        self.assertEqual(skipped, [])
        self.assertEqual(ProfileCase(16, 64, 4).q_per_request, 4)
        self.assertEqual(ProfileCase(16, 64, 4).to_dict()["c_total"], 256)
        cases, skipped = build_cases([16, 32, 64, 128, 256], [0, 64, 128, 256], [1, 2, 4], 512)
        self.assertEqual(len(cases), 59)
        self.assertEqual((skipped[0]["q"], skipped[0]["c"], skipped[0]["batch_size"]), (256, 256, 1))

    def test_invalid_or_indivisible_grid_is_explicit(self):
        for args in (([0], [0], [1], 1024), ([16], [-1], [1], 1024),
                     ([16], [0], [0], 1024), ([16, 16], [0], [1], 1024)):
            with self.assertRaises(ValueError):
                build_cases(*args)
        cases, skipped = build_cases([17], [0], [4], 1024)
        self.assertEqual(cases, [])
        self.assertIn("divide", skipped[0]["reason"])

    def test_summary_excludes_warmup_and_uses_linear_p95(self):
        case = ProfileCase(32, 64, 2)
        rows = [self.row(case, "warmup", 0, 100), self.row(case, "measure", 0, 1),
                self.row(case, "measure", 1, 3)]
        self.assertTrue(validate_samples(rows, [case], 2, 1, True))
        summary = summarize_samples(rows)[0]
        self.assertEqual(summary["samples"], 2)
        self.assertAlmostEqual(summary["step_wall_ms_mean"], 2)
        self.assertAlmostEqual(summary["step_wall_ms_p50"], 2)
        self.assertAlmostEqual(summary["step_wall_ms_p95"], 2.9)

    def test_integrity_rejects_duplicates_wrong_q_cache_leaks_and_compilation(self):
        case = ProfileCase(32, 64, 2)
        rows = [self.row(case, "warmup", 0, 2), self.row(case, "measure", 0, 1)]
        for change in ("duplicate", "q", "cache", "compilation", "timing"):
            with self.subTest(change=change):
                altered = deepcopy(rows)
                if change == "duplicate":
                    altered.append(deepcopy(altered[-1]))
                elif change == "q":
                    altered[-1]["q_each"] = [32, 32]
                elif change == "cache":
                    altered[-1]["kv_used_after"] = 1
                elif change == "compilation":
                    altered[-1]["compile_frames"] = 1
                else:
                    altered[-1]["step_wall_ms"] += 1
                with self.assertRaises(ValueError):
                    validate_samples(altered, [case], 1, 1, True)

    def test_csv_json_round_trip_and_tamper_detection(self):
        case = ProfileCase(16, 0, 1)
        rows = [self.row(case, "warmup", 0, 2), self.row(case, "measure", 0, 1)]
        metadata = {"options": dict(repeats=1, warmup_rounds=1, cuda_events=True)}
        with TemporaryDirectory() as directory:
            write_results(directory, [dict(status="ok", samples=rows)], metadata, [case])
            self.assertEqual(len(verify_results(directory)["summary"]), 1)
            path = Path(directory) / "profile.json"
            payload = json.loads(path.read_text())
            payload["summary"][0]["step_wall_ms_mean"] += 1
            path.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "summary JSON"):
                verify_results(directory)

    def test_optional_event_time_stays_missing(self):
        case = ProfileCase(16, 0, 1)
        rows = [self.row(case, "warmup", 0, 2), self.row(case, "measure", 0, 1)]
        for row in rows:
            row["cuda_model_stream_ms"] = None
        self.assertTrue(validate_samples(rows, [case], 1, 1, False))
        self.assertIsNone(summarize_samples(rows)[0]["cuda_model_stream_ms_mean"])


class PrefixPreparationTests(SchedulerTestCase):
    def engine(self, num_blocks=16):
        scheduler = self.make_scheduler(num_kvcache_blocks=num_blocks, max_num_seqs=4)
        class CpuRunner:
            def __init__(self):
                self.calls = 0
            def call(self, name, *args):
                self.calls += 1
                return getattr(self, name)(*args)
            def prepare_prefill(self, seqs):
                return [], []
            def run_model(self, ids, positions, is_prefill):
                return None
            def run(self, seqs, is_prefill):
                ids, positions = self.prepare_prefill(seqs)
                self.run_model(ids, positions, is_prefill)
                return [17] * len(seqs)
        class CpuEngine:
            def __init__(self):
                self.scheduler = scheduler
                self.model_runner = CpuRunner()
            def is_finished(self):
                return self.scheduler.is_finished()
            def add_request(self, ids, params):
                self.scheduler.add(Sequence(ids, params))
            def step(self):
                seqs, is_prefill = self.scheduler.schedule()
                count = sum(seq.num_scheduled_tokens for seq in seqs)
                ids = self.model_runner.call("run", seqs, is_prefill)
                self.scheduler.postprocess(seqs, ids, is_prefill)
                return [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished], count
        return CpuEngine()

    def test_real_prefix_postprocess_retains_waiting_and_no_output(self):
        engine = self.engine()
        case = ProfileCase(32, 64, 2)
        params = SamplingParams(max_tokens=1, ignore_eos=True)
        seqs = prepare_case(engine, case, [[1] * 80, [2] * 80], params)
        self.assertEqual(engine.model_runner.calls, 2)
        self.assertEqual([seq.num_cached_tokens for seq in seqs], [64, 64])
        self.assertEqual([seq.num_completion_tokens for seq in seqs], [0, 0])
        self.assertEqual(list(engine.scheduler.waiting), seqs)
        self.assert_blocks_consistent(engine.scheduler)
        outputs, count = engine.step()
        self.assertEqual(count, 32)
        self.assertEqual(len(outputs), 2)
        require_released(engine)
        self.assert_blocks_consistent(engine.scheduler, *seqs)

    def test_timing_probe_captures_actual_schedule_and_restores_methods(self):
        engine = self.engine()
        case = ProfileCase(16, 0, 4)
        fake_torch = SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda: None))
        with TimingProbe(engine) as probe:
            row = sample_case(engine, case, [[i] * 4 for i in range(4)],
                              SamplingParams(max_tokens=1, ignore_eos=True), probe, fake_torch,
                              defaultdict(Counter), "measure", 0, 0)
        self.assertEqual(row["q_each"], [4] * 4)
        self.assertEqual(row["c_each"], [0] * 4)
        self.assertEqual(row["kv_used_before"], 4)
        self.assertIsNone(row["cuda_model_stream_ms"])
        self.assertNotIn("schedule", vars(engine.scheduler))
        for name in ("run", "prepare_prefill", "run_model"):
            self.assertNotIn(name, vars(engine.model_runner))

    def test_kv_capacity_error_is_not_silently_skipped(self):
        engine = self.engine(num_blocks=1)
        case = ProfileCase(256, 256, 1)
        with self.assertRaisesRegex(MemoryError, "insufficient KV"):
            prepare_case(engine, case, [[1] * 512], SamplingParams(max_tokens=1))


if __name__ == "__main__":
    unittest.main()
