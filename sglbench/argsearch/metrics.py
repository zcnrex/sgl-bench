"""Canonical metric vocabulary shared by the bench transport and the objective.

The concrete BenchClient ([[RFC-0001:C-MEASUREMENT]]) emits these keys; the Pareto/SLO
objective ([[RFC-0001:C-OBJECTIVE]]) consumes them. Throughput selection prefers the tail
steady-state decode rate over OSL-averaged throughput; TTFT and per-token selection prefer
percentile keys over the single-sample baseline anchor ([[RFC-0001:C-BASELINE-ANCHOR]]).
"""

from __future__ import annotations

TTFT_P95_MS = "ttft_p95_ms"
TTFT_MS = "ttft_ms"
PER_TOKEN_P95_MS = "per_token_p95_ms"
PER_TOKEN_MS = "per_token_ms"
DECODE_THROUGHPUT = "decode_throughput_tok_s"
OUTPUT_THROUGHPUT = "output_throughput_tok_s"
OVERALL_THROUGHPUT = "overall_throughput_tok_s"
INPUT_THROUGHPUT = "input_throughput_tok_s"
E2E_LATENCY_S = "e2e_latency_s"

TTFT_KEYS = (TTFT_P95_MS, TTFT_MS)
PER_TOKEN_KEYS = (PER_TOKEN_P95_MS, PER_TOKEN_MS)
THROUGHPUT_KEYS = (DECODE_THROUGHPUT, OUTPUT_THROUGHPUT, OVERALL_THROUGHPUT)


def metric_value(metrics: dict, key: str, stat: str = "median") -> float | None:
    """Read one metric, accepting either a raw float or an aggregated {stat: value}."""
    if key not in metrics:
        return None
    v = metrics[key]
    if isinstance(v, dict):
        if stat in v:
            return float(v[stat])
        if "mean" in v:
            return float(v["mean"])
        return None
    return float(v)


def select(metrics: dict, keys, stat: str = "median") -> float | None:
    """First present metric among `keys`, in priority order."""
    for k in keys:
        val = metric_value(metrics, k, stat)
        if val is not None:
            return val
    return None


def ttft_ms(metrics: dict, stat: str = "median") -> float | None:
    return select(metrics, TTFT_KEYS, stat)


def per_token_ms(metrics: dict, stat: str = "median") -> float | None:
    return select(metrics, PER_TOKEN_KEYS, stat)


def throughput(metrics: dict, stat: str = "median") -> float | None:
    return select(metrics, THROUGHPUT_KEYS, stat)
