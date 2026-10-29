"""Pure Python tests: synthetic clocks/engines only, no torch or model imports."""

import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

from benchmarks.metrics import (RequestMetrics, percentile, summarize_requests,
                                verify_results, write_results)
from benchmarks.mixed_load_bench import (EngineObserver, RequestSpec, build_workload,
                                         drive_requests, failed_run)


class MetricsTests(unittest.TestCase):
    def test_linear_percentiles_and_empty_samples(self):
        self.assertIsNone(percentile([], 0.95))
        self.assertEqual(percentile([7], 0.95), 7)
        self.assertEqual(percentile([0, 10, 20, 30], 0.5), 15)
        self.assertAlmostEqual(percentile([0, 10, 20, 30], 0.95), 28.5)
        self.assertEqual(percentile([1, 3], 0), 1)
        self.assertEqual(percentile([1, 3], 1), 3)
        for values, q in [([1], -0.1), ([1], 1.1), ([float("nan")], 0.5)]:
            with self.assertRaises(ValueError):
                percentile(values, q)

    def make_records(self):
        first = RequestMetrics(0, 64, 3, 0.5, admission_time=1.0)
        for timestamp in (1.2, 1.5, 2.0):
            first.add_token(timestamp)
        first.complete(2.05)
        second = RequestMetrics(1, 300, 1, 1.9, admission_time=2.0)
        second.add_token(2.1)
        second.complete(2.2)
        return [first, second]

    def test_metrics_use_actual_admission_and_pool_token_intervals(self):
        records = self.make_records()
        summary = summarize_requests(records, 4.0)
        self.assertAlmostEqual(records[0].ttft, 0.2)
        self.assertAlmostEqual(records[0].to_dict()["admission_delay_s"], 0.5)
        self.assertAlmostEqual(summary["ttft_mean_s"], 0.15)
        self.assertAlmostEqual(summary["ttft_p95_s"], 0.195)
        self.assertAlmostEqual(summary["itl_mean_s"], 0.4)
        self.assertAlmostEqual(summary["itl_p95_s"], 0.49)
        self.assertAlmostEqual(summary["request_latency_p50_s"], 0.625)
        self.assertEqual(summary["ttft_samples"], 2)
        self.assertEqual(summary["itl_samples"], 2)
        self.assertEqual(summary["completed_requests"], 2)
        self.assertEqual(summary["total_output_tokens"], 4)
        self.assertEqual(summary["output_token_throughput"], 1)

    def test_pending_and_single_token_requests_have_no_itl(self):
        records = [RequestMetrics(0, 64, 2, 0), RequestMetrics(1, 64, 1, 0, admission_time=0)]
        records[1].add_token(1)
        summary = summarize_requests(records, 2)
        self.assertEqual(summary["completed_requests"], 0)
        self.assertEqual(summary["admitted_requests"], 1)
        self.assertEqual(summary["ttft_samples"], 1)
        self.assertEqual(summary["itl_samples"], 0)
        self.assertIsNone(summary["itl_p95_s"])
        self.assertIsNone(summary["request_latency_p50_s"])
        self.assertIsNone(summarize_requests(records, None)["output_token_throughput"])

    def test_invalid_timestamps_and_duration_are_rejected(self):
        record = RequestMetrics(0, 64, 2, 1)
        with self.assertRaises(ValueError):
            record.add_token(2)
        record.admission_time = 2
        for timestamp in (1, float("inf"), float("nan")):
            with self.assertRaises(ValueError):
                record.add_token(timestamp)
        record.add_token(3)
        with self.assertRaises(ValueError):
            record.complete(2)
        record.complete(3)
        with self.assertRaises(ValueError):
            record.add_token(4)
        with self.assertRaises(ValueError):
            RequestMetrics(1, 64, 1, 2, admission_time=1).validate()
        for duration in (0, -1, float("inf"), float("nan")):
            with self.assertRaises(ValueError):
                summarize_requests([record], duration)

    def make_run(self, records):
        return {"summary": {"policy": "unit_test", "scheduling_policy": "decode_first",
                            "max_consecutive_decode_steps": 2, "repeat": 0, "status": "ok",
                            **summarize_requests(records, 4.0)},
                "requests": [record.to_dict() for record in records]}

    def test_csv_json_export_round_trip_and_tamper_detection(self):
        records = self.make_records()
        with TemporaryDirectory() as directory:
            write_results(directory, [self.make_run(records)], {"test_only": True})
            self.assertTrue(verify_results(directory))
            path = Path(directory) / "summary.json"
            payload = json.loads(path.read_text())
            payload["runs"][0]["ttft_p95_s"] += 1
            path.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "summaries differ"):
                verify_results(directory)

    def test_raw_token_timestamps_are_checked_against_summary(self):
        with TemporaryDirectory() as directory:
            write_results(directory, [self.make_run(self.make_records())], {})
            path = Path(directory) / "raw_requests.csv"
            with path.open(newline="") as file:
                reader = csv.DictReader(file)
                fields, rows = reader.fieldnames, list(reader)
            rows[0]["token_times"] = "[1.3, 1.5, 2.0]"
            with path.open("w", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
            with self.assertRaisesRegex(ValueError, "token timestamps"):
                verify_results(directory)

    def test_unstarted_failure_exports_no_fabricated_performance(self):
        spec = RequestSpec(0, [1] * 64, 16, 0.0, "short")
        run = failed_run([spec], "decode_first", 0, "CUDA unavailable (unit test)")
        with TemporaryDirectory() as directory:
            write_results(directory, [run], {"test_only": True})
            self.assertTrue(verify_results(directory))
            summary = json.loads((Path(directory) / "summary.json").read_text())["runs"][0]
            self.assertEqual(summary["status"], "failed")
            self.assertEqual(summary["completed_requests"], 0)
            self.assertEqual(summary["ttft_samples"], 0)
            self.assertIsNone(summary["elapsed_seconds"])
            self.assertIsNone(summary["output_token_throughput"])
            self.assertIsNone(summary["ttft_p95_s"])


class FakeClock:
    def __init__(self):
        self.value = 100.0

    def __call__(self):
        return self.value

    def advance(self, seconds):
        # A real perf_counter advances while the loop runs; this fake clock
        # needs a minimum tick to avoid rounding tiny sleeps back to zero.
        self.value += max(seconds, 1e-9) if seconds > 0 else 0


class FakeSequence:
    def __init__(self, seq_id, max_tokens, clock):
        self.seq_id = seq_id
        self.max_tokens = max_tokens
        self.clock = clock
        self.num_completion_tokens = 0
        self.chunks = 0

    @property
    def is_finished(self):
        return self.num_completion_tokens == self.max_tokens

    def append_token(self, token_id):
        self.clock.advance(0.01)
        self.num_completion_tokens += 1


class FakeScheduler:
    def __init__(self):
        self.queue = []

    def add(self, seq):
        self.queue.append(seq)

    @property
    def waiting(self):
        return [seq for seq in self.queue if seq.chunks < 2]

    @property
    def running(self):
        return [seq for seq in self.queue if seq.chunks >= 2]

    def schedule(self):
        return [self.queue[0]], self.queue[0].chunks < 2

    def preempt(self, seq):
        pass

    def postprocess(self, seqs, token_ids, is_prefill):
        seq = seqs[0]
        if is_prefill:
            seq.chunks += 1
            if seq.chunks < 2:
                return  # A partial chunk has no real output token.
        seq.append_token(token_ids[0])
        if seq.is_finished:
            self.queue.remove(seq)


class FakeEngine:
    def __init__(self, clock):
        self.clock = clock
        self.scheduler = FakeScheduler()
        self.next_id = 0

    def add_request(self, prompt, params):
        seq = FakeSequence(self.next_id, params.max_tokens, self.clock)
        self.next_id += 1
        self.scheduler.add(seq)

    def is_finished(self):
        return not self.scheduler.queue

    def step(self):
        seqs, is_prefill = self.scheduler.schedule()
        self.clock.advance(1)
        self.scheduler.postprocess(seqs, [17], is_prefill)


class BenchmarkObservationTests(unittest.TestCase):
    def test_chunk_does_not_record_ttft_and_hooks_restore_after_exception(self):
        clock = FakeClock()
        engine = FakeEngine(clock)
        spec = RequestSpec(0, [1] * 300, 2, 0, "long")
        observer = EngineObserver(engine, [spec], clock)
        with self.assertRaisesRegex(RuntimeError, "test interruption"):
            with observer:
                observer.origin = clock()
                observer.admit(spec, SimpleNamespace(max_tokens=2))
                seq = engine.scheduler.queue[0]
                engine.step()
                self.assertEqual(observer.prefill_steps, 1)
                self.assertEqual(observer.records[0].token_times, [])
                self.assertIsNone(observer.records[0].ttft)
                engine.step()
                self.assertEqual(seq.num_completion_tokens, 1)
                self.assertAlmostEqual(observer.records[0].ttft, 2.01)
                engine.scheduler.preempt(seq)
                self.assertEqual(observer.preemptions, 1)
                engine.step()
                self.assertEqual(observer.decode_steps, 1)
                self.assertIsNotNone(observer.records[0].completion_time)
                raise RuntimeError("test interruption")
        self.assertNotIn("append_token", vars(seq))
        self.assertNotIn("step", vars(engine))
        for name in ("add", "schedule", "preempt", "postprocess"):
            self.assertNotIn(name, vars(engine.scheduler))

    def test_arrival_during_step_is_admitted_at_next_boundary(self):
        clock = FakeClock()
        engine = FakeEngine(clock)
        specs = [RequestSpec(0, [1] * 64, 2, 0, "short"),
                 RequestSpec(1, [2] * 300, 1, 0.5, "long")]
        with EngineObserver(engine, specs, clock) as observer:
            elapsed = drive_requests(engine, observer, specs,
                                     lambda count: SimpleNamespace(max_tokens=count), sleep=clock.advance)
        first, second = observer.records.values()
        self.assertEqual(first.admission_time, 0)
        self.assertEqual(second.admission_time, 1)
        self.assertAlmostEqual(second.to_dict()["admission_delay_s"], 0.5)
        self.assertEqual(observer.prefill_steps, 4)
        self.assertEqual(observer.decode_steps, 1)
        self.assertAlmostEqual(elapsed, 5.03)

    def test_idle_arrival_time_is_included_in_throughput_window(self):
        clock = FakeClock()
        engine = FakeEngine(clock)
        specs = [RequestSpec(0, [1] * 64, 1, 0.05, "short")]
        with EngineObserver(engine, specs, clock) as observer:
            elapsed = drive_requests(engine, observer, specs,
                                     lambda count: SimpleNamespace(max_tokens=count), sleep=clock.advance)
        self.assertAlmostEqual(observer.records[0].admission_time, 0.05)
        self.assertAlmostEqual(elapsed, 2.06)

    def test_step_limit_and_timeout_raise_errors(self):
        for timeout, steps, expected in [(120, 1, RuntimeError), (0.5, 100, TimeoutError)]:
            with self.subTest(timeout=timeout, steps=steps):
                clock = FakeClock()
                engine = FakeEngine(clock)
                specs = [RequestSpec(0, [1] * 64, 2, 0, "short")]
                with EngineObserver(engine, specs, clock) as observer:
                    with self.assertRaises(expected):
                        drive_requests(engine, observer, specs,
                                       lambda count: SimpleNamespace(max_tokens=count),
                                       timeout_seconds=timeout, max_steps=steps, sleep=clock.advance)


class WorkloadTests(unittest.TestCase):
    class Tokenizer:
        def encode(self, text, add_special_tokens=False):
            return [ord(char) for char in text]

    def test_workload_reproducibility_four_classes_and_context_limits(self):
        first = build_workload(self.Tokenizer(), seed=42)
        second = build_workload(self.Tokenizer(), seed=42)
        different = build_workload(self.Tokenizer(), seed=43)
        self.assertEqual(first, second)
        self.assertNotEqual(first, different)
        self.assertEqual(len({spec.workload_class for spec in first}), 4)
        self.assertEqual(len(first), 12)
        self.assertEqual([spec.scheduled_arrival for spec in first[:4]], [0] * 4)
        for spec in first:
            self.assertLessEqual(len(spec.prompt_token_ids) + spec.max_tokens, 512)
            low, high = (64, 128) if spec.workload_class.startswith("short") else (256, 384)
            self.assertTrue(low <= len(spec.prompt_token_ids) <= high)
            self.assertTrue(16 <= spec.max_tokens <= 64)

    def test_workload_rejects_invalid_arrivals_and_context_limits(self):
        for kwargs in ({"num_requests": 3}, {"initial_burst": -1},
                       {"arrival_interval": -0.1}, {"arrival_interval": float("nan")},
                       {"max_model_len": 64}):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    build_workload(self.Tokenizer(), **kwargs)


if __name__ == "__main__":
    unittest.main()
