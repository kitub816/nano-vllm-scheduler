"""Pure Python reliability checks with test-only synthetic clocks/records."""

from copy import deepcopy
from tempfile import TemporaryDirectory
from pathlib import Path
from types import SimpleNamespace
import unittest
import csv

from benchmarks.metrics import summarize_requests
from benchmarks.mixed_load_bench import EngineObserver, RequestSpec, execution_plan
from benchmarks.mixed_load_bench import drive_requests
from benchmarks.reliability import (cross_run_statistics, decode_batch_statistics,
                                    interval_analysis, long_gap_evidence, validate_run_steps,
                                    verify_step_csv, write_step_csv)
from test_benchmark_metrics import FakeClock, FakeEngine


class ReliabilityTests(unittest.TestCase):
    def make_run(self):
        clock = FakeClock()
        engine = FakeEngine(clock)
        specs = [RequestSpec(0, [1] * 64, 2, 0, "short")]
        with EngineObserver(engine, specs, clock) as observer:
            elapsed = drive_requests(engine, observer, specs,
                                     lambda count: SimpleNamespace(max_tokens=count), sleep=clock.advance)
        return dict(summary={"policy": "test_only", "repeat": 0, "status": "ok",
                             "prefill_steps": observer.prefill_steps, "decode_steps": observer.decode_steps,
                             "preemptions": 0, "torch_compile_frames_during_measurement": 0,
                             **summarize_requests(list(observer.records.values()), elapsed)},
                    requests=[record.to_dict() for record in observer.records.values()],
                    steps=observer.steps,
                    resource_check=dict(released=True, waiting_requests=0, running_requests=0,
                                        used_blocks=0, total_blocks=1, free_blocks=1, block_ref_count_sum=0))

    def test_trace_matches_real_appends_and_chunk_without_output(self):
        run = self.make_run()
        self.assertTrue(validate_run_steps(run))
        self.assertEqual([step["output_tokens"] for step in run["steps"]], [0, 1, 1])
        self.assertEqual(run["steps"][1]["output_token_indices"], [1])
        self.assertEqual(run["steps"][2]["output_token_indices"], [2])
        self.assertAlmostEqual(run["steps"][2]["duration_s"], 1.01)
        evidence = long_gap_evidence(run)[0]
        self.assertEqual((evidence["first_step"], evidence["next_step"]), (1, 2))

    def test_trace_rejects_missing_duplicate_or_mistimed_tokens(self):
        original = self.make_run()
        for alteration in ("duplicate", "missing", "timestamp", "resources", "compile"):
            with self.subTest(alteration=alteration):
                run = deepcopy(original)
                if alteration == "duplicate":
                    run["steps"][-1]["output_token_indices"] = [1]
                elif alteration == "missing":
                    run["steps"][-1]["emitted_request_ids"] = []
                elif alteration == "timestamp":
                    run["requests"][0]["token_times"][-1] += 10
                elif alteration == "resources":
                    run["resource_check"]["used_blocks"] = 1
                else:
                    run["summary"]["torch_compile_frames_during_measurement"] = 1
                with self.assertRaises(ValueError):
                    validate_run_steps(run)

    def test_mean_can_exceed_p95_due_to_small_tail(self):
        stamps = [0]
        for gap in [0.01] * 99 + [2.0]:
            stamps.append(stamps[-1] + gap)
        result = interval_analysis([dict(request_id=9, token_times=stamps)])
        self.assertEqual(result["samples"], 100)
        self.assertEqual(result["over_100ms"], 1)
        self.assertEqual(result["over_200ms"], 1)
        self.assertGreater(result["mean_s"], result["p95_s"])
        self.assertEqual(result["longest_10"][0]["request_id"], 9)
        self.assertEqual(result["longest_10"][0]["to_output_index"], 101)

    def test_decode_batch_distribution_is_weighted_by_rounds(self):
        result = decode_batch_statistics([
            dict(phase="prefill", batch_size=4, duration_s=9),
            dict(phase="decode", batch_size=1, duration_s=.1),
            dict(phase="decode", batch_size=4, duration_s=.2),
            dict(phase="decode", batch_size=4, duration_s=.3)])
        self.assertEqual(result["rounds"], 3)
        self.assertEqual(result["decode_tokens"], 9)
        self.assertEqual(result["mean_batch_size"], 3)
        self.assertEqual(result["distribution"][1]["rounds"], 2)
        self.assertAlmostEqual(result["distribution"][1]["mean_step_duration_s"], .25)

    def test_cross_run_sample_standard_deviation(self):
        result = cross_run_statistics([dict(policy="test", status="ok", repeat=i, value=value)
                                       for i, value in enumerate((1, 2, 3))])
        self.assertEqual(result, [dict(policy="test", metric="value", samples=3,
                                       mean=2, sample_stddev=1, min=1, max=3)])
        with self.assertRaises(ValueError):
            cross_run_statistics([dict(policy="test", status="failed", value=1)])

    def test_rotating_order_balances_positions_without_changing_repeat_ids(self):
        plan = execution_plan(["A", "B", "C"], 3, "rotate")
        self.assertEqual([row["execution_index"] for row in plan], list(range(9)))
        self.assertEqual([row["policy"] for row in plan], list("ABCBCACAB"))
        for repeat in range(3):
            self.assertEqual({row["policy"] for row in plan if row["repeat"] == repeat}, {"A", "B", "C"})
        self.assertEqual([row["policy"] for row in execution_plan(["A", "B"], 2, "fixed")], list("ABAB"))

    def test_step_csv_retains_per_request_and_token_indices(self):
        run = self.make_run()
        with TemporaryDirectory() as directory:
            path = Path(directory) / "steps.csv"
            write_step_csv(path, [run])
            text = path.read_text()
            self.assertIn("output_token_indices", text)
            self.assertIn("test_only", text)
            self.assertEqual(len(text.splitlines()), 4)
            self.assertTrue(verify_step_csv(path, [run]))
            with path.open(newline="") as file:
                reader = csv.DictReader(file)
                fields, rows = reader.fieldnames, list(reader)
            rows[-1]["batch_size"] = "4"
            with path.open("w", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=fields, lineterminator="\n")
                writer.writeheader()
                writer.writerows(rows)
            with self.assertRaisesRegex(ValueError, "step CSV and run JSON differ"):
                verify_step_csv(path, [run])


if __name__ == "__main__":
    unittest.main()
