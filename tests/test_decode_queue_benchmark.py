"""Pure Python checks for stage-four variant labels and first-token intervals."""

import unittest

from benchmarks.decode_queue_analysis import decode_service_gaps, first_to_second_statistics
from benchmarks.mixed_load_bench import parse_args, run_identity


class DecodeQueueBenchmarkTests(unittest.TestCase):
    def test_original_default_benchmark_and_queue_variants(self):
        self.assertEqual(parse_args([]).policies,
                         ["prefill_first", "decode_first", "decode_first_fair2"])
        for policy, expected in (("prefill_first", "head_first"),
                                 ("prefill_first_rr", "round_robin"),
                                 ("decode_first_fair2", "head_first"),
                                 ("decode_first_fair2_rr", "round_robin")):
            row = run_identity(policy, 0)
            self.assertEqual(row["decode_queue_policy"], expected)
        self.assertEqual(run_identity("decode_first_fair2_rr", 0)["max_consecutive_decode_steps"], 2)

    def test_first_to_second_is_not_all_token_intervals(self):
        summary, rows = first_to_second_statistics([
            dict(request_id=0, token_times=[1, 1.1, 3]),
            dict(request_id=1, token_times=[2, 2.3, 8]),
            dict(request_id=2, token_times=[4])])
        self.assertEqual(summary["first_to_second_samples"], 2)
        self.assertAlmostEqual(summary["first_to_second_mean_s"], .2)
        self.assertAlmostEqual(summary["first_to_second_max_s"], .3)
        self.assertEqual([row["request_id"] for row in rows], [0, 1])

    def test_missing_second_token_has_no_interval(self):
        summary, rows = first_to_second_statistics([dict(request_id=0, token_times=[])])
        self.assertEqual(summary["first_to_second_samples"], 0)
        self.assertIsNone(summary["first_to_second_mean_s"])
        self.assertEqual(rows, [])
        with self.assertRaises(ValueError):
            first_to_second_statistics([dict(request_id=1, token_times=[2, 1])])

    def test_service_gap_counts_decode_rounds_not_prefills(self):
        requests = [dict(request_id=0, token_times=[0, 4])]
        steps = [dict(phase="prefill", emitted_request_ids=[0], output_token_indices=[1]),
                 dict(phase="prefill", emitted_request_ids=[], output_token_indices=[]),
                 dict(phase="decode", emitted_request_ids=[1], output_token_indices=[2]),
                 dict(phase="decode", emitted_request_ids=[1], output_token_indices=[3]),
                 dict(phase="decode", emitted_request_ids=[0], output_token_indices=[2])]
        self.assertEqual(decode_service_gaps(requests, steps),
                         dict(max_intervening_decode_rounds=2,
                              max_first_to_second_intervening_decode_rounds=2))


if __name__ == "__main__":
    unittest.main()
