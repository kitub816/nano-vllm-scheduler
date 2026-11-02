"""Offline, standard-library-only models; split by process, never by sample."""

from collections import defaultdict
import math
from statistics import fmean, median

from benchmarks.metrics import percentile


def fit_model(rows, target, kind):
    if not rows or any(row.get("instrumentation") != "none" for row in rows):
        raise ValueError("models require non-empty, unprofiled measurements")
    values = [row[target] for row in rows]
    if not all(math.isfinite(value) and value > 0 for value in values):
        raise ValueError("invalid model target")
    if kind == "constant":
        # The median is the MAE-optimal constant, a deliberately strong baseline.
        intercept, slope = median(values), 0.0
    elif kind == "q_linear":
        xs = [row["q"] / 256 for row in rows]
        x_mean, y_mean = fmean(xs), fmean(values)
        variance = sum((x - x_mean) ** 2 for x in xs)
        if not variance:
            raise ValueError("q-linear model needs distinct q values")
        slope = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, values)) / variance
        intercept = y_mean - slope * x_mean
    else:
        raise ValueError(f"unknown model: {kind}")
    return dict(kind=kind, target=target, intercept_ms=intercept, slope_per_256_tokens_ms=slope,
                training_process_ids=sorted({row["process_id"] for row in rows}),
                training_samples=len(rows))


def predict(model, row):
    return max(0.0, model["intercept_ms"] + model["slope_per_256_tokens_ms"] * row["q"] / 256)


def error_metrics(actual, predictions):
    if not actual or len(actual) != len(predictions):
        raise ValueError("error metrics need equal, non-empty arrays")
    signed = [prediction - value for value, prediction in zip(actual, predictions)]
    absolute = [abs(value) for value in signed]
    if not all(math.isfinite(value) for value in absolute):
        raise ValueError("errors must be finite")
    return dict(samples=len(absolute), mae_ms=fmean(absolute),
                p50_absolute_error_ms=percentile(absolute, .5),
                p95_absolute_error_ms=percentile(absolute, .95),
                max_absolute_error_ms=max(absolute), signed_bias_ms=fmean(signed),
                underpredict_count=sum(value < 0 for value in signed),
                absolute_error_lt_1ms=sum(value < 1 for value in absolute),
                absolute_error_1_to_2ms=sum(1 <= value < 2 for value in absolute),
                absolute_error_2_to_5ms=sum(2 <= value < 5 for value in absolute),
                absolute_error_5_to_10ms=sum(5 <= value < 10 for value in absolute),
                absolute_error_ge_10ms=sum(value >= 10 for value in absolute))


def evaluate_models(rows):
    if any(row.get("instrumentation") != "none" for row in rows):
        raise ValueError("Profiler data cannot be used for performance prediction")
    if any(row.get("role") not in ("train", "holdout") for row in rows):
        raise ValueError("every measurement needs an explicit train/holdout role")
    train = [row for row in rows if row["role"] == "train"]
    test = [row for row in rows if row["role"] == "holdout"]
    train_ids = {row["process_id"] for row in train}
    test_ids = {row["process_id"] for row in test}
    if not train_ids or not test_ids or train_ids & test_ids:
        raise ValueError("train and holdout must be non-empty and process-disjoint")
    fitted, heldout, per_process, predictions, cv = [], [], [], [], []
    for target in ("step_wall_ms", "cuda_model_stream_ms"):
        for kind in ("constant", "q_linear"):
            model = fit_model(train, target, kind)
            fitted.append(model)
            estimated = [predict(model, row) for row in test]
            identity = dict(model=kind, target=target)
            heldout.append({**identity, **error_metrics([row[target] for row in test], estimated)})
            groups = defaultdict(list)
            for row, prediction in zip(test, estimated):
                predictions.append(dict(**identity, process_id=row["process_id"], case_id=row["case_id"],
                                        repeat=row["repeat"], actual_ms=row[target], predicted_ms=prediction,
                                        signed_error_ms=prediction - row[target],
                                        absolute_error_ms=abs(prediction - row[target])))
                groups[row["process_id"]].append((row[target], prediction))
            for process_id, values in groups.items():
                per_process.append(dict(**identity, process_id=process_id,
                                        **error_metrics([v[0] for v in values], [v[1] for v in values])))
            # Inner validation uses training processes only; holdout is never fit.
            for validation_id in sorted(train_ids):
                inner_train = [row for row in train if row["process_id"] != validation_id]
                inner_test = [row for row in train if row["process_id"] == validation_id]
                if not inner_train:
                    continue
                inner = fit_model(inner_train, target, kind)
                cv.append(dict(**identity, validation_process_id=validation_id,
                               training_process_ids=inner["training_process_ids"],
                               **error_metrics([row[target] for row in inner_test],
                                               [predict(inner, row) for row in inner_test])))
    return dict(models=fitted, holdout_metrics=heldout, per_process_metrics=per_process,
                predictions=predictions, training_process_cv=cv,
                train_process_ids=sorted(train_ids), holdout_process_ids=sorted(test_ids),
                complex_model="not fitted: c/256 equals paged-KV indicator; batch=4 has only c=0; design is not identifiable")
