"""Pure Python trace interpretation and process-held-out model validation."""

import unittest

from benchmarks.prefill_cost_model import error_metrics, evaluate_models, fit_model, predict
from benchmarks.prefill_timeline import analyze_trace, merge_intervals
from benchmarks.prefill_timeline import validate_measurements, summarize_measurements
from benchmarks.prefill_profile import ProfileCase


def event(name, ts, dur, cat="user_annotation", **args):
    return dict(name=name, ts=ts, dur=dur, ph="X", cat=cat, args=args)


class TraceTests(unittest.TestCase):
    def trace(self):
        # GPU execution can finish after the model's CPU submission scope ends.
        return dict(traceEvents=[event("prefill::step", 0, 100), event("prefill::model", 10, 40),
                                 event("prefill_ops::Linear", 11, 20),
                                 event("cudaLaunchKernel", 15, 2, "cuda_runtime", correlation=1),
                                 event("cudaLaunchKernel", 35, 2, "cuda_runtime", correlation=2),
                                 event("gemm", 30, 10, "kernel", correlation=1, stream=7),
                                 event("other", 60, 20, "kernel", correlation=2, stream=7)])

    def test_correlated_gpu_work_outlives_cpu_scope(self):
        trace = self.trace()
        trace["traceEvents"].extend([event("prefill::model", 30, 50, "gpu_user_annotation"),
                                     event("prefill::step", 30, 50, "gpu_user_annotation"),
                                     event("prefill_ops::Linear", 30, 10, "gpu_user_annotation")])
        result = analyze_trace(trace)
        self.assertTrue(result["cuda_trace_available"])
        self.assertEqual(result["model_kernel_count"], 2)
        self.assertAlmostEqual(result["model_kernel_active_union_ms"], .030)
        self.assertAlmostEqual(result["model_kernel_span_ms"], .050)
        self.assertAlmostEqual(result["model_kernel_free_ms"], .020)
        self.assertEqual(result["categories"][1]["kernel_count"], 1)
        self.assertAlmostEqual(result["categories"][1]["cpu_scope_ms"], .020)
        self.assertEqual(result["categories"][-1]["kernel_count"], 1)
        self.assertAlmostEqual(result["streams"][0]["kernel_gaps"]["max_ms"], .020)
        self.assertAlmostEqual(result["streams"][0]["cpu_launch_gaps"]["max_ms"], .018)

    def test_overlap_and_copy_are_not_double_counted_as_idle(self):
        trace = self.trace()
        trace["traceEvents"].extend([event("overlap", 35, 10, "kernel", correlation=1, stream=8),
                                     event("copy", 45, 10, "gpu_memcpy")])
        result = analyze_trace(trace)
        self.assertAlmostEqual(result["model_kernel_sum_ms"], .040)
        self.assertAlmostEqual(result["model_kernel_active_union_ms"], .035)
        self.assertAlmostEqual(result["observed_gpu_activity_free_ms"], .005)

    def test_missing_cuda_is_explicit_and_does_not_invent_times(self):
        trace = self.trace()
        trace["traceEvents"] = [e for e in trace["traceEvents"] if e["cat"] != "kernel"]
        result = analyze_trace(trace)
        self.assertFalse(result["cuda_trace_available"])
        self.assertNotIn("model_kernel_span_ms", result)
        self.assertEqual(result["model_kernel_count"], 0)

    def test_missing_or_duplicate_scope_is_rejected(self):
        for change in ("missing", "duplicate"):
            trace = self.trace()
            if change == "missing":
                trace["traceEvents"] = trace["traceEvents"][1:]
            else:
                trace["traceEvents"].append(event("prefill::model", 20, 10))
            with self.assertRaises(ValueError):
                analyze_trace(trace)

    def test_interval_union_handles_overlaps_and_rejects_negative(self):
        self.assertEqual(merge_intervals([(5, 7), (1, 4), (3, 5), (9, 10)]), [(1, 7), (9, 10)])
        with self.assertRaises(ValueError):
            merge_intervals([(3, 2)])


class CostModelTests(unittest.TestCase):
    def rows(self):
        return [dict(instrumentation="none", process_id=pid, role="train" if pid % 2 == 0 else "holdout",
                     q=q, case_id=f"q{q}", repeat=0, step_wall_ms=10 + q / 16,
                     cuda_model_stream_ms=8 + q / 16)
                for pid in range(6) for q in (16, 64, 256)]

    def test_constant_is_mae_optimal_median_and_q_fit_predicts(self):
        rows = self.rows()[:3]
        self.assertEqual(fit_model(rows, "step_wall_ms", "constant")["intercept_ms"], 14)
        linear = fit_model(rows, "step_wall_ms", "q_linear")
        self.assertAlmostEqual(predict(linear, {"q": 128}), 18)

    def test_holdout_never_influences_fit(self):
        rows = self.rows()
        original = evaluate_models(rows)
        for row in rows:
            if row["role"] == "holdout":
                row["step_wall_ms"] += 1000
                row["cuda_model_stream_ms"] += 1000
        altered = evaluate_models(rows)
        self.assertEqual(original["models"], altered["models"])
        self.assertEqual(original["training_process_cv"], altered["training_process_cv"])
        self.assertNotEqual(original["holdout_metrics"], altered["holdout_metrics"])
        for fold in original["training_process_cv"]:
            self.assertNotIn(fold["validation_process_id"], fold["training_process_ids"])
            self.assertTrue(set(fold["training_process_ids"]) <= {0, 2, 4})

    def test_process_overlap_and_profiler_measurements_rejected(self):
        for change in ("overlap", "profiler", "no_holdout", "unknown_role"):
            rows = self.rows()
            if change == "overlap":
                rows[-1]["process_id"] = 0
            elif change == "profiler":
                rows[-1]["instrumentation"] = "torch_profiler"
            elif change == "unknown_role":
                rows[-1]["role"] = "unknown"
            else:
                rows = [row for row in rows if row["role"] == "train"]
            with self.assertRaises(ValueError):
                evaluate_models(rows)

    def test_error_distribution_and_linear_p95(self):
        result = error_metrics([1, 2, 5, 12], [1, 1, 1, 1])
        self.assertEqual(result["mae_ms"], 4)
        self.assertAlmostEqual(result["p95_absolute_error_ms"], 9.95)
        self.assertEqual(result["absolute_error_lt_1ms"], 1)
        self.assertEqual(result["absolute_error_1_to_2ms"], 1)
        self.assertEqual(result["absolute_error_2_to_5ms"], 1)
        self.assertEqual(result["absolute_error_ge_10ms"], 1)

    def test_model_rejects_degenerate_q_and_nonfinite_targets(self):
        rows = self.rows()[:1]
        with self.assertRaises(ValueError):
            fit_model(rows, "step_wall_ms", "q_linear")
        rows[0]["step_wall_ms"] = float("nan")
        with self.assertRaises(ValueError):
            fit_model(rows, "step_wall_ms", "constant")


class MeasurementIntegrityTests(unittest.TestCase):
    def fixture(self):
        case = ProfileCase(16, 256, 1)
        row = dict(**case.to_dict(), process_id=0, repeat=0, role="train", phase="measure",
                   instrumentation="none", compile_frames=0, q_each=[16], c_each=[256],
                   output_tokens=1, max_num_batched_tokens=16, kv_used_after=0, kv_ref_count_after=0,
                   kv_free_after=8, kv_total_blocks=8, wall_start_perf_counter=1,
                   wall_end_perf_counter=1.01, step_wall_ms=10., cuda_model_stream_ms=9.,
                   model_submit_wall_ms=8., prepare_wall_ms=.2)
        return [row], dict(options=dict(processes=1, repeats=1), cases=[case.to_dict()])

    def test_missing_duplicate_wrong_phase_compile_and_leak_rejected(self):
        rows, manifest = self.fixture()
        self.assertTrue(validate_measurements(rows, manifest))
        with self.assertRaises(ValueError):
            validate_measurements([], manifest)
        with self.assertRaises(ValueError):
            validate_measurements(rows + rows, manifest)
        for field, bad in (("phase", "warmup"), ("instrumentation", "torch_profiler"),
                           ("compile_frames", 1), ("kv_ref_count_after", 1), ("c_each", [0]),
                           ("step_wall_ms", 11.), ("role", "holdout"), ("output_tokens", 2)):
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_measurements([{**rows[0], field: bad}], manifest)

    def test_cross_process_sd_uses_process_means(self):
        rows, _ = self.fixture()
        rows.append({**rows[0], "process_id": 1, "role": "holdout", "step_wall_ms": 14.})
        process, cases = summarize_measurements(rows)
        self.assertEqual(len(process), 2)
        self.assertEqual(cases[0]["step_wall_ms_mean"], 12.)
        self.assertAlmostEqual(cases[0]["step_wall_ms_process_mean_sd"], 8 ** .5)


if __name__ == "__main__":
    unittest.main()
