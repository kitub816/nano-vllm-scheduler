"""Analyze the six real high-pressure B8 FIFO/bypass runs, without GPU work.

Per-run quantiles use type 7; cross-run SD uses ddof=1. Request pairing describes
observed changes, not a counterfactual estimate of the isolated bypass cost.
"""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from statistics import fmean

from benchmarks.kv_pressure_bench import verify_results, export_csv, assert_csv_matches
from benchmarks.metrics import percentile
from benchmarks.reliability import cross_run_statistics, decode_batch_statistics


POLICIES = ("B8", "B8_bypass")


def wait_metrics(requests, prefix="allocation_wait"):
    waits = [row["first_prefill_time"] - row["admission_time"] for row in requests]
    if not waits or any(value < 0 for value in waits):
        raise ValueError("missing/negative first allocation waits")
    return {f"{prefix}_mean_s": fmean(waits), f"{prefix}_p50_s": percentile(waits, .5),
            f"{prefix}_p95_s": percentile(waits, .95), f"{prefix}_max_s": max(waits),
            f"{prefix}_over_10s": sum(value > 10. for value in waits)}


def analyze_run(run):
    summary = run["summary"]
    rows = run["requests"]
    if Counter(r["prompt_tokens"] for r in rows) != {248: 40, 504: 40}:
        raise ValueError("expected the original 40 short + 40 long prompts")
    metrics = {**summary, **wait_metrics(rows),
               **wait_metrics([r for r in rows if r["prompt_tokens"] == 504], "large_wait"),
               **wait_metrics([r for r in rows if r["prompt_tokens"] == 248], "small_wait")}
    heads = Counter(e["head_request_id"] for e in run["bypass_events"])
    if heads and max(heads.values()) > 1:
        raise ValueError("a head was overtaken more than once")
    metrics["max_overtakes_per_request"] = max(heads.values(), default=0)
    metrics["step_weighted_mean_used_blocks"] = summary["step_weighted_block_utilization"] * summary["num_kvcache_blocks"]
    metrics["large_ttft_p95_s"] = percentile([r["ttft_s"] for r in rows if r["prompt_tokens"] == 504], .95)
    metrics["large_latency_p95_s"] = percentile([r["request_latency_s"] for r in rows if r["prompt_tokens"] == 504], .95)
    return metrics


def paired_rows(fifo, bypass):
    originals = {r["request_id"]: r for r in fifo["requests"]}
    candidates = {r["request_id"]: r for r in bypass["requests"]}
    if originals.keys() != candidates.keys():
        raise ValueError("paired requests differ")
    heads = Counter(e["head_request_id"] for e in bypass["bypass_events"])
    promoted = Counter(e["candidate_request_id"] for e in bypass["bypass_events"])
    rows = []
    for rid, first in originals.items():
        second = candidates[rid]
        if (first["prompt_tokens"], first["max_tokens"]) != (second["prompt_tokens"], second["max_tokens"]):
            raise ValueError("paired token demands differ")
        a = first["first_prefill_time"] - first["admission_time"]
        b = second["first_prefill_time"] - second["admission_time"]
        rows.append(dict(repeat=fifo["summary"]["repeat"], request_id=rid,
            prompt_tokens=first["prompt_tokens"], overtaken_count=heads[rid], promoted_count=promoted[rid],
            fifo_allocation_wait_s=a, bypass_allocation_wait_s=b, allocation_wait_delta_s=b-a,
            fifo_ttft_s=first["ttft_s"], bypass_ttft_s=second["ttft_s"],
            ttft_delta_s=second["ttft_s"]-first["ttft_s"],
            fifo_latency_s=first["request_latency_s"], bypass_latency_s=second["request_latency_s"],
            latency_delta_s=second["request_latency_s"]-first["request_latency_s"]))
    return rows


def build_analysis(payload):
    manifest = payload["manifest"]
    if (manifest["config"]["pressures"] != ["high"]
            or manifest["config"]["policies"] != list(POLICIES)
            or manifest["config"]["repeats"] != 3):
        raise ValueError("expected high pressure, B8/B8_bypass, three repeats")
    runs = payload["runs"]
    if len(runs) != 6:
        raise ValueError("expected all six experiments")
    grouped = {}
    metrics = []
    for run in runs:
        s = run["summary"]
        key = (s["policy"], s["repeat"])
        if key in grouped or s["num_kvcache_blocks"] != 105:
            raise ValueError("duplicate experiment or changed physical capacity")
        if (s["scheduling_policy"], s["decode_queue_policy"], s["max_consecutive_decode_steps"],
                s["kv_admission_policy"], s["kv_reserve_blocks"], s["kv_bounded_bypass"]) != (
                "prefill_first", "round_robin", 0, "reserve", 8, s["policy"] == "B8_bypass"):
            raise ValueError("experiment configurations differ beyond bypass")
        grouped[key] = run
        metrics.append(analyze_run(run))
    paired, changes = [], []
    excluded = {"repeat", "max_consecutive_decode_steps", "kv_reserve_blocks", "num_kvcache_blocks"}
    for repeat in range(3):
        fifo, bypass = grouped[("B8", repeat)], grouped[("B8_bypass", repeat)]
        paired.extend(paired_rows(fifo, bypass))
        a, b = analyze_run(fifo), analyze_run(bypass)
        for key, value in a.items():
            if key not in excluded and type(value) in (int, float) and type(b.get(key)) in (int, float):
                changes.append(dict(repeat=repeat, metric=key, fifo=value, bypass=b[key],
                                    delta=b[key]-value, relative_percent=(100*(b[key]-value)/value if value else None)))
    return dict(data_kind="observed real GPU measurements; no offline simulated performance",
        percentile_method="linear interpolation at (n-1)*p (type 7)",
        cross_run_method="mean/sample SD (ddof=1)/min/max of three per-run metrics; no pooled quantiles",
        large_request_definition="504 prompt tokens (40 requests/run); short = 248 tokens (40/run); all output=16",
        wait_definition="first successful BlockManager.allocate return timestamp minus actual enqueue timestamp",
        mean_occupancy_definition="sum(after-schedule used blocks * step wall duration) / sum(step wall duration); step approximation",
        long_wait_definition="over 10 seconds, descriptive count; not a starvation proof",
        starvation_scope="finite 80-request closed batch; completion and one-overtake limit cannot prove open-arrival wall-time fairness",
        runs=metrics, cross_run_statistics=cross_run_statistics(metrics), request_pairs=paired,
        paired_metric_changes=changes,
        decode_batch_statistics=[dict(policy=r["summary"]["policy"], repeat=r["summary"]["repeat"],
                                      **decode_batch_statistics(r["steps"])) for r in runs],
        environment=[dict(policy=r["summary"]["policy"], repeat=r["summary"]["repeat"],
                          **r["environment"]) for r in runs],
        input_manifest_sha256=hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="results_kv_bypass_20261102")
    parser.add_argument("--output-dir")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    root = Path(args.results)
    payload = verify_results(root)
    result = build_analysis(payload)
    directory = Path(args.output_dir) if args.output_dir else root / "bypass_analysis"
    fields = {"per_run.csv": "runs", "cross_run_statistics.csv": "cross_run_statistics",
              "request_pairs.csv": "request_pairs", "paired_metric_changes.csv": "paired_metric_changes"}
    if args.verify:
        if json.loads((directory / "analysis.json").read_text()) != result:
            raise ValueError("analysis JSON differs from raw data")
        for name, field in fields.items():
            assert_csv_matches(directory / name, result[field])
        print("Verified all six GPU runs, event/resource accounting and analysis CSV/JSON")
        return 0
    if directory.exists() and any(directory.iterdir()):
        raise ValueError(f"analysis output must be empty: {directory}")
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "analysis.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)+"\n")
    for name, field in fields.items():
        export_csv(directory / name, result[field])
    print(f"Saved six-run B8 analysis: {directory}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
