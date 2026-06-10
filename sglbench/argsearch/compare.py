"""Per-concurrency comparison of every OFAT candidate against the baseline config.

Companion to the Pareto frontier ([[RFC-0001:C-OBJECTIVE]], `objective.py`): the frontier
answers "which configs are non-dominated", this answers "how far does each single-arg change
move decode throughput / latency / accuracy relative to the baseline reference point". The
baseline is the OFAT origin — `generate_ofat(branch)[0]` — so the reference is resolved from
the config, not hard-coded.
"""

from __future__ import annotations

import argparse
import sys

from . import metrics as M
from .generate import DEFAULT_RESULTS, generate_ofat, load_config
from .objective import load_results


def _by_config_workload(records: list[dict]) -> dict[tuple[str, int], dict]:
    """Index records by (config_hash, concurrency); last write wins on duplicates."""
    return {
        (r.get("config_hash", ""), int(r.get("workload", {}).get("concurrency", -1))): r
        for r in records
    }


def _accuracy(record: dict, metric: str) -> float | None:
    return (record.get("accuracy") or {}).get(metric)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="Compare each OFAT candidate against the baseline, per concurrency."
    )
    p.add_argument("--config", required=True, help="Versioned search config YAML")
    p.add_argument("--branch", required=True, help="Precision branch name")
    p.add_argument("--results", default=DEFAULT_RESULTS, help="Measured results JSONL")
    p.add_argument("--stat", choices=["median", "mean"], default="median")
    a = p.parse_args(argv)

    cfg = load_config(a.config)
    branch = cfg.branch(a.branch)
    metric = cfg.quality_gate.metric if cfg.quality_gate else "accuracy"

    points = generate_ofat(branch)
    baseline_hash = points[0].config_hash
    label = {pt.config_hash: pt.label for pt in points}
    order = [pt.config_hash for pt in points]

    records = [r for r in load_results(a.results) if r.get("branch") == a.branch]
    indexed = _by_config_workload(records)
    concurrencies = sorted({c for _, c in indexed})

    width = max((len(lbl) for lbl in label.values()), default=8)
    print(f"baseline = {label.get(baseline_hash, baseline_hash)} ({baseline_hash})")
    print(f"branch = {a.branch}   stat = {a.stat}   metric = {metric}\n")

    for conc in concurrencies:
        base = indexed.get((baseline_hash, conc))
        base_thr = M.throughput(base.get("metrics", {}), a.stat) if base else None
        header = (
            f"{'config':{width}} {'decode tok/s':>12} {'Δ vs base':>9} "
            f"{'ptok ms':>8} {'ttft ms':>9} {metric:>7}"
        )
        print(f"# concurrency = {conc}")
        print(header)
        for h in order:
            r = indexed.get((h, conc))
            if r is None:
                continue
            m = r.get("metrics", {})
            thr = M.throughput(m, a.stat)
            ptok = M.per_token_ms(m, a.stat)
            ttft = M.ttft_ms(m, a.stat)
            acc = _accuracy(r, metric)
            delta = (
                "baseline"
                if h == baseline_hash
                else ("n/a" if base_thr in (None, 0) or thr is None else f"{(thr / base_thr - 1) * 100:+.1f}%")
            )
            thr_s = "n/a" if thr is None else f"{thr:.1f}"
            ptok_s = "n/a" if ptok is None else f"{ptok:.2f}"
            ttft_s = "n/a" if ttft is None else f"{ttft:.1f}"
            acc_s = "n/a" if acc is None else f"{acc}"
            print(f"{label.get(h, h):{width}} {thr_s:>12} {delta:>9} {ptok_s:>8} {ttft_s:>9} {acc_s:>7}")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
