"""Mergeable token statistics with bounded-memory percentile upper bounds."""
import bisect
import math

from harborrl.trajectories.native import finite

# Logarithmic buckets retain sub-micro precision near zero, with no token all-gather.
EDGES = [0.0, *[2 ** (-24 + i / 8) for i in range(321)]]


def statistics(rows, *, threshold=None, worst=8):
    result = {"count": 0, "signed_sum": 0.0, "abs_sum": 0.0, "max_abs_delta": 0.0,
              "min_log_ratio": None, "max_log_ratio": None, "bad_count": 0,
              "histogram": [0] * (len(EDGES) + 1), "worst": []}
    for row_index, (current, behavior, mask) in enumerate(rows):
        if len(current) != len(behavior) or len(mask) != len(current) or not len(current):
            raise ValueError("Probability/mask length mismatch")
        if any(flag not in (0, 1) for flag in mask):
            raise ValueError("Loss mask must be binary")
        for position, (p, q, flag) in enumerate(zip(current, behavior, mask, strict=True)):
            if not finite(p) or not finite(q) or p > 0 or q > 0:
                raise ValueError("Log probabilities must be finite and nonpositive")
            if not flag:
                continue
            delta = float(p) - float(q)
            absolute = abs(delta)
            if not math.isfinite(delta):
                raise ValueError("Nonfinite log-probability difference")
            result["count"] += 1
            result["signed_sum"] += delta
            result["abs_sum"] += absolute
            result["max_abs_delta"] = max(result["max_abs_delta"], absolute)
            result["min_log_ratio"] = delta if result["min_log_ratio"] is None else min(result["min_log_ratio"], delta)
            result["max_log_ratio"] = delta if result["max_log_ratio"] is None else max(result["max_log_ratio"], delta)
            result["bad_count"] += int(threshold is not None and absolute > threshold)
            result["histogram"][bisect.bisect_left(EDGES, absolute)] += 1
            if worst:
                result["worst"].append({"row": row_index, "position": position, "delta": delta})
                result["worst"] = sorted(result["worst"], key=lambda x: abs(x["delta"]), reverse=True)[:worst]
    if not result["count"]:
        raise ValueError("No valid response tokens")
    if not math.isfinite(result["abs_sum"]) or not math.isfinite(result["signed_sum"]):
        raise ValueError("Probability statistics overflow")
    return result


def merge(parts):
    if not parts:
        raise ValueError("No contributing ranks")
    count = sum(p["count"] for p in parts)
    bins = [sum(p["histogram"][i] for p in parts) for i in range(len(EDGES) + 1)]
    maximum = max(p["max_abs_delta"] for p in parts)
    def percentile(fraction):
        target, seen = math.ceil(count * fraction), 0
        for index, n in enumerate(bins):
            seen += n
            if seen >= target:
                return min(EDGES[index], maximum) if index < len(EDGES) else maximum
        raise ValueError("Corrupt histogram")
    return {"token_count": count, "signed_mean_delta": math.fsum(p["signed_sum"] for p in parts) / count,
            "mean_abs_delta": math.fsum(p["abs_sum"] for p in parts) / count,
            "max_abs_delta": maximum, "p95_abs_delta_upper": percentile(.95), "p99_abs_delta_upper": percentile(.99),
            "min_log_ratio": min(p["min_log_ratio"] for p in parts),
            "max_log_ratio": max(p["max_log_ratio"] for p in parts),
            "bad_fraction": sum(p["bad_count"] for p in parts) / count,
            "percentile_semantics": "logarithmic-histogram upper bound; not exact quantiles",
            "gap_semantics": "fixed-prefix token log-ratio; not full-sequence KL"}
