"""Synthetic arithmetic fixtures only; never GPU benchmark measurements."""

import unittest

from benchmarks.kv_admission_analysis import analyze, LABELS


class AdmissionAnalysisTests(unittest.TestCase):
    def payload(self):
        runs = []
        for level in ("low", "medium", "high"):
            for policy in LABELS:
                for repeat in range(3):
                    preemptions = (4 if policy == "B0" else 2) if level == "high" else 0
                    summary = dict(pressure=level, policy=policy, repeat=repeat,
                        preemptions=preemptions, recomputed_tokens=preemptions*2,
                        ttft_mean_s=1., ttft_p50_s=1., ttft_p95_s=1., itl_p95_s=.1,
                        request_latency_mean_s=3., request_latency_p95_s=3., output_token_throughput=10.)
                    records = [dict(request_id=0, admission_time=0., first_prefill_time=.5,
                                    first_token_time=1., completion_time=3.)]
                    runs.append(dict(summary=summary, requests=records, steps=[], reserve_deferral_events=[]))
        return dict(runs=runs, manifest=dict(config=dict(pressures=["low", "medium", "high"])))

    def test_all_thresholds_zero_baseline_and_wait_decomposition(self):
        result = analyze(self.payload())
        self.assertEqual(len(result["comparisons"]), 9)
        for row in result["comparisons"]:
            if row["pressure"] == "high":
                self.assertEqual(row["preemptions_change_percent"], -50.)
                self.assertEqual(row["recomputed_tokens_change_percent"], -50.)
            else:
                self.assertIsNone(row["preemptions_change_percent"])
                self.assertIsNone(row["recomputed_tokens_change_percent"])
        self.assertEqual(len(result["latency_decomposition"]), 54)
        row = result["latency_decomposition"][0]
        self.assertEqual(row["admission_to_first_prefill_mean_s"], .5)
        self.assertEqual(row["first_prefill_to_first_token_mean_s"], .5)
        self.assertEqual(row["first_token_to_completion_mean_s"], 2.)
        self.assertEqual({r["policy"] for r in result["statistics"]}, set(LABELS))
        self.assertEqual(len(result["request_differences"]), 27)
        self.assertTrue(all(r["first_prefill_wait_difference_s"] == 0 for r in result["request_differences"]))

    def test_bad_decomposition_is_rejected(self):
        payload = self.payload()
        payload["runs"][0]["requests"][0]["completion_time"] = 4.
        with self.assertRaisesRegex(ValueError, "decomposition"):
            analyze(payload)


if __name__ == "__main__":
    unittest.main()
