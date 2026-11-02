"""Pure Python checks of analysis arithmetic, using synthetic CPU fixtures."""

from copy import deepcopy
import unittest

from benchmarks.kv_bypass_analysis import wait_metrics, paired_rows, analyze_run


class BypassAnalysisTests(unittest.TestCase):
    def test_allocation_wait_uses_enqueue_not_scheduled_arrival(self):
        rows = [dict(admission_time=2., first_prefill_time=2.+x, scheduled_arrival=0.)
                for x in (0., 1., 2., 13.)]
        metrics = wait_metrics(rows)
        self.assertEqual(metrics["allocation_wait_mean_s"], 4.)
        self.assertAlmostEqual(metrics["allocation_wait_p95_s"], 11.35)
        self.assertEqual(metrics["allocation_wait_max_s"], 13.)
        self.assertEqual(metrics["allocation_wait_over_10s"], 1)
        with self.assertRaises(ValueError):
            wait_metrics([dict(admission_time=2., first_prefill_time=1.)])

    def test_request_pairs_keep_delayed_heads_and_negative_improvements(self):
        rows = [dict(request_id=i, prompt_tokens=504 if i == 0 else 248,
                     max_tokens=16, admission_time=1., first_prefill_time=4.,
                     ttft_s=3.1, request_latency_s=8.) for i in range(2)]
        fifo = dict(summary=dict(repeat=0), requests=rows)
        bypass = dict(requests=deepcopy(rows), bypass_events=[dict(head_request_id=0, candidate_request_id=1)])
        bypass["requests"][0]["first_prefill_time"] += .5
        bypass["requests"][1]["first_prefill_time"] -= 2.
        result = paired_rows(fifo, bypass)
        self.assertEqual(result[0]["allocation_wait_delta_s"], .5)
        self.assertEqual(result[0]["overtaken_count"], 1)
        self.assertEqual(result[1]["allocation_wait_delta_s"], -2.)
        self.assertEqual(result[1]["promoted_count"], 1)
        self.assertEqual(rows[0]["first_prefill_time"], 4.)

    def test_group_metrics_never_hide_large_requests(self):
        requests = [dict(prompt_tokens=248 if i % 2 == 0 else 504,
                         admission_time=0., first_prefill_time=1. if i % 2 == 0 else 9.,
                         ttft_s=10., request_latency_s=12.) for i in range(80)]
        run = dict(summary=dict(step_weighted_block_utilization=.8, num_kvcache_blocks=105),
                   requests=requests, bypass_events=[])
        result = analyze_run(run)
        self.assertEqual(result["allocation_wait_mean_s"], 5.)
        self.assertEqual(result["large_wait_mean_s"], 9.)
        self.assertEqual(result["small_wait_mean_s"], 1.)
        self.assertEqual(result["step_weighted_mean_used_blocks"], 84.)
        run["bypass_events"] = [dict(head_request_id=1)] * 2
        with self.assertRaisesRegex(ValueError, "overtaken"):
            analyze_run(run)
