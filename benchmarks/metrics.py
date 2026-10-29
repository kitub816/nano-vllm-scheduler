"""Standard-library-only timing metrics and CSV/JSON exports.

All timestamps are seconds relative to the measured run's perf_counter origin.
Percentiles use linear interpolation at (n - 1) * q (Hyndman-Fan type 7).
ITL statistics pool individual adjacent-token intervals across requests.
"""

import csv
from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from statistics import fmean


PERCENTILE_METHOD = "linear interpolation at (n - 1) * q (Hyndman-Fan type 7)"


def percentile(values, q):
    if not 0 <= q <= 1:
        raise ValueError("q must be between 0 and 1")
    ordered = sorted(values)
    if not ordered:
        return None
    if not all(math.isfinite(value) for value in ordered):
        raise ValueError("samples must be finite")
    position = (len(ordered) - 1) * q
    low = math.floor(position)
    high = math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


@dataclass
class RequestMetrics:
    request_id: int
    prompt_tokens: int
    max_tokens: int
    scheduled_arrival: float
    workload_class: str = ""
    admission_time: float | None = None
    token_times: list[float] = field(default_factory=list)
    completion_time: float | None = None

    def validate(self):
        if self.prompt_tokens <= 0 or self.max_tokens <= 0:
            raise ValueError("prompt_tokens and max_tokens must be positive")
        stamps = [self.scheduled_arrival]
        if self.admission_time is not None:
            stamps.append(self.admission_time)
        elif self.token_times or self.completion_time is not None:
            raise ValueError("tokens/completion require admission")
        stamps.extend(self.token_times)
        if self.completion_time is not None:
            if not self.token_times:
                raise ValueError("completion requires at least one output token")
            stamps.append(self.completion_time)
        if any(not math.isfinite(t) or t < 0 for t in stamps):
            raise ValueError("timestamps must be finite and non-negative")
        if any(a > b for a, b in zip(stamps, stamps[1:])):
            raise ValueError("timestamps must be monotonic and follow admission")
        if len(self.token_times) > self.max_tokens:
            raise ValueError("output exceeds max_tokens")

    def add_token(self, timestamp):
        if self.admission_time is None or self.completion_time is not None:
            raise ValueError("token requires an admitted, unfinished request")
        previous = self.token_times[-1] if self.token_times else self.admission_time
        if not math.isfinite(timestamp) or timestamp < previous:
            raise ValueError("token timestamps must be finite and monotonic")
        if len(self.token_times) >= self.max_tokens:
            raise ValueError("output exceeds max_tokens")
        self.token_times.append(timestamp)

    def complete(self, timestamp):
        if self.completion_time is not None:
            raise ValueError("request already completed")
        if not self.token_times or not math.isfinite(timestamp) or timestamp < self.token_times[-1]:
            raise ValueError("completion must follow the final output token")
        self.completion_time = timestamp

    @property
    def ttft(self):
        return self.token_times[0] - self.admission_time if self.token_times else None

    @property
    def itls(self):
        return [b - a for a, b in zip(self.token_times, self.token_times[1:])]

    def to_dict(self):
        self.validate()
        return dict(
            request_id=self.request_id, workload_class=self.workload_class,
            prompt_tokens=self.prompt_tokens, max_tokens=self.max_tokens,
            output_tokens=len(self.token_times), scheduled_arrival=self.scheduled_arrival,
            admission_time=self.admission_time,
            first_token_time=self.token_times[0] if self.token_times else None,
            completion_time=self.completion_time, ttft_s=self.ttft,
            admission_delay_s=(None if self.admission_time is None else
                               self.admission_time - self.scheduled_arrival),
            request_latency_s=(None if self.completion_time is None else
                               self.completion_time - self.admission_time),
            token_times=list(self.token_times), itl_s=self.itls,
            status=("completed" if self.completion_time is not None else
                    "pending" if self.admission_time is None else "incomplete"),
        )


def summarize_requests(records, elapsed_seconds):
    if elapsed_seconds is not None and (not math.isfinite(elapsed_seconds) or elapsed_seconds <= 0):
        raise ValueError("elapsed_seconds must be positive and finite, or None for an unstarted run")
    rows = [record.to_dict() for record in records]
    tokens = sum(row["output_tokens"] for row in rows)
    summary = dict(
        requested_requests=len(rows),
        admitted_requests=sum(row["admission_time"] is not None for row in rows),
        completed_requests=sum(row["status"] == "completed" for row in rows),
        total_output_tokens=tokens, elapsed_seconds=elapsed_seconds,
        output_token_throughput=(tokens / elapsed_seconds if elapsed_seconds is not None else None),
    )
    samples = {
        "ttft": [row["ttft_s"] for row in rows if row["ttft_s"] is not None],
        "itl": [interval for row in rows for interval in row["itl_s"]],
        "request_latency": [row["request_latency_s"] for row in rows
                            if row["request_latency_s"] is not None],
        "admission_delay": [row["admission_delay_s"] for row in rows
                            if row["admission_delay_s"] is not None],
    }
    for name, values in samples.items():
        summary.update({f"{name}_samples": len(values),
                        f"{name}_mean_s": fmean(values) if values else None,
                        f"{name}_p50_s": percentile(values, 0.5),
                        f"{name}_p95_s": percentile(values, 0.95)})
    return summary


def _csv_value(value):
    if value is None:
        return ""
    if isinstance(value, list):
        return json.dumps(value, allow_nan=False)
    return str(value)


def write_results(output_dir, runs, metadata):
    """Runs contain a flat summary plus per-request dictionaries; no aggregation across repeats."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = [run["summary"] for run in runs]
    payload = {"metadata": {**metadata, "percentile_method": PERCENTILE_METHOD}, "runs": summaries}
    (output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as file:
        fields = list(dict.fromkeys(key for summary in summaries for key in summary))
        writer = csv.DictWriter(file, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for summary in summaries:
            writer.writerow({key: _csv_value(summary.get(key)) for key in fields})
    rows = []
    for run in runs:
        identity = {key: run["summary"][key] for key in
                    ("policy", "scheduling_policy", "max_consecutive_decode_steps", "repeat")}
        rows.extend({**identity, **row} for row in run["requests"])
    with (output_dir / "raw_requests.csv").open("w", newline="", encoding="utf-8") as file:
        fields = list(rows[0]) if rows else ["policy", "request_id"]
        writer = csv.DictWriter(file, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(value) for key, value in row.items()})
    verify_results(output_dir)


def verify_results(output_dir):
    """Recompute metrics from raw CSV and compare them to both summary formats."""
    output_dir = Path(output_dir)
    payload = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    with (output_dir / "summary.csv").open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        fields = reader.fieldnames
        summaries_csv = list(reader)
    if len(summaries_csv) != len(payload["runs"]):
        raise ValueError("CSV/JSON summary row count differs")
    with (output_dir / "raw_requests.csv").open(newline="", encoding="utf-8") as file:
        raw = list(csv.DictReader(file))
    groups = {}
    for row in raw:
        identity = (row["policy"], int(row["repeat"]))
        record = RequestMetrics(
            request_id=int(row["request_id"]), prompt_tokens=int(row["prompt_tokens"]),
            max_tokens=int(row["max_tokens"]), scheduled_arrival=float(row["scheduled_arrival"]),
            workload_class=row["workload_class"],
            admission_time=float(row["admission_time"]) if row["admission_time"] else None,
            token_times=json.loads(row["token_times"]),
            completion_time=float(row["completion_time"]) if row["completion_time"] else None)
        canonical = record.to_dict()
        for key, value in canonical.items():
            observed = (row[key] if isinstance(value, str) else
                        None if not row[key] else json.loads(row[key]))
            if observed != value:
                raise ValueError(f"raw CSV field differs from token timestamps: {identity}, {key}")
        groups.setdefault(identity, []).append(record)
    seen = set()
    for row, summary in zip(summaries_csv, payload["runs"]):
        if row != {key: _csv_value(summary.get(key)) for key in fields}:
            raise ValueError("CSV/JSON summaries differ")
        identity = (summary["policy"], summary["repeat"])
        if identity in seen:
            raise ValueError("duplicate policy/repeat in summary")
        seen.add(identity)
        records = groups.pop(identity, [])
        if len({record.request_id for record in records}) != len(records):
            raise ValueError("duplicate request IDs within a run")
        computed = summarize_requests(records, summary["elapsed_seconds"])
        for key, value in computed.items():
            if summary[key] != value:
                raise ValueError(f"raw CSV/summary mismatch: {identity}, {key}")
    if groups:
        raise ValueError("raw CSV contains runs missing from summary")
    return True
