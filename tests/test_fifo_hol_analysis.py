"""CPU replay and independent-decision simulation, never synthetic GPU results."""

from copy import deepcopy
import unittest

from benchmarks.fifo_hol_analysis import recover_waiting, fresh_cost, physical_demand, probe
from benchmarks.kv_pressure_bench import PressureObserver, RequestSpec, run_requests
from test_kv_pressure import CpuEngine
from test_scheduler_policy import SchedulerTestCase, SamplingParams


class ReplayTests(SchedulerTestCase):
    def fixture(self):
        scheduler = self.make_scheduler(num_kvcache_blocks=4, max_num_batched_tokens=256,
            max_num_seqs=2, decode_queue_policy="round_robin", kv_admission_policy="reserve", kv_reserve_blocks=1)
        specs = [RequestSpec(i, [i+10]*(248 if i%2==0 else 504), 16, 0., "cpu_fixture") for i in range(8)]
        engine = CpuEngine(scheduler)
        observer = PressureObserver(engine, specs)
        with observer:
            run_requests(engine, observer, specs, lambda n:SamplingParams(max_tokens=n,ignore_eos=True), 10, 2000, 100)
        return dict(requests=[{**r.to_dict(), **observer.request_stats[r.request_id]} for r in observer.records.values()],
                    steps=observer.steps, preemption_events=observer.preemption_events,
                    allocation_events=observer.allocation_events, reserve_deferral_events=observer.reserve_deferral_events), [s.to_dict() for s in specs]

    def test_real_chunk_preemption_and_queue_replay_drain(self):
        run, specs = self.fixture()
        snapshots, boundaries = recover_waiting(run, specs)
        self.assertTrue(run["preemption_events"])
        self.assertEqual(len(boundaries), len(run["steps"]))
        self.assertEqual(boundaries[-1]["waiting_after_step"], [])
        self.assertEqual(boundaries[-1]["running_members_after_step"], [])
        for snapshot in snapshots:
            self.assertEqual(snapshot["waiting"][0], snapshot["request_id"])

    def test_missing_allocation_or_ambiguous_admission_raises(self):
        run, specs = self.fixture()
        broken = deepcopy(run)
        broken["allocation_events"].pop(0)
        with self.assertRaises(ValueError):
            recover_waiting(broken, specs)
        broken = deepcopy(run)
        broken["requests"][1]["admission_time"] = broken["requests"][0]["admission_time"]
        with self.assertRaisesRegex(ValueError, "ambiguous|identifiable"):
            recover_waiting(broken, specs)


class ProbeTests(unittest.TestCase):
    def fixture(self):
        snapshot = dict(waiting=[0,1,2], free_blocks=9, reserve_blocks=8, time=2., step_index=1,
                        selected_prefill_tokens=0, selected_prefill_seqs=0)
        specs = {i:dict(prompt_token_ids=[i+1]*(504 if i<2 else 248)) for i in range(3)}
        costs = {i:dict(known=True,blocks=2 if i<2 else 1) for i in range(3)}
        records = {i:dict(admission_time=0.,first_prefill_time=5.) for i in range(3)}
        return snapshot,costs,specs,records

    def test_bounds_capacity_and_independent_state(self):
        s,c,p,r = self.fixture()
        before = deepcopy(s)
        self.assertIsNone(probe(s,c,p,r,"fifo",105)["selected_request_id"])
        self.assertIsNone(probe(s,c,p,r,"skip_1",105)["selected_request_id"])
        for mode in ("skip_2","min_new_blocks"):
            selected = probe(s,c,p,r,mode,105)
            self.assertEqual(selected["selected_request_id"],2)
            self.assertEqual(selected["projected_used_blocks"],97)
            self.assertEqual(selected["small_over_large_count"],2)
            self.assertIsNone(selected["future_ttft_s"])
            self.assertIsNone(selected["future_starvation_count"])
        self.assertEqual(s,before)

    def test_budget_and_unknown_prefix_never_bypassed(self):
        s,c,p,r = self.fixture()
        s.update(selected_prefill_seqs=1,selected_prefill_tokens=10)
        self.assertIsNone(probe(s,c,p,r,"skip_2",105)["selected_request_id"])
        s.update(selected_prefill_seqs=0,selected_prefill_tokens=0)
        c[2] = dict(known=False,blocks=None)
        self.assertIsNone(probe(s,c,p,r,"min_new_blocks",105)["selected_request_id"])

    def test_active_share_and_free_cached_reference_semantics(self):
        self.assertEqual(physical_demand(2,[3],{3}),1)
        self.assertEqual(physical_demand(2,[3],set()),2)
        with self.assertRaises(ValueError):
            physical_demand(2,[3,3],{3})

    def test_missing_generated_ids_make_possible_prefix_unknown(self):
        specs=[dict(request_id=0,prompt_token_ids=[1]*248),dict(request_id=1,prompt_token_ids=[1]*256+[2]*248)]
        records={0:dict(first_prefill_time=1.),1:dict(first_prefill_time=3.)}
        result=fresh_cost(specs[1],specs,2.,records)
        self.assertFalse(result["known"])
        specs[1]["prompt_token_ids"][0]=3
        self.assertTrue(fresh_cost(specs[1],specs,2.,records)["known"])


if __name__ == "__main__":
    unittest.main()
