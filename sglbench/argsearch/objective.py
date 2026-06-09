"""Decode-first Pareto frontier under an SLO over the measured results.

Implements [[RFC-0001:C-OBJECTIVE]]: the SLO gate is decode per-token latency (ITL); TTFT
is report-only and never excludes a point. Survivors form the decode-throughput-versus-ITL
Pareto frontier (decode throughput up, ITL down), never ranked by raw throughput alone.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from . import metrics as M
from .generate import load_config
from .schema import SLO

DEFAULT_RESULTS = "out/results.jsonl"
DEFAULT_FRONTIER = "out/frontier.jsonl"


@dataclass
class FrontierEntry:
    config_hash: str
    branch: str
    label: str
    workload: dict
    context_length: int
    throughput: float
    per_token_ms: float
    ttft_ms: float | None


def load_results(path) -> list[dict]:
    text = Path(path).read_text()
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def _context_length(workload: dict) -> int:
    return int(workload.get("isl", 0)) + int(workload.get("osl", 0))


def passes_slo(record: dict, slo: SLO, stat: str = "median") -> bool:
    """True when the decode per-token latency (ITL) is within the SLO gate for the record's
    workload ([[RFC-0001:C-OBJECTIVE]])."""
    metrics = record.get("metrics", {})
    per_token = M.per_token_ms(metrics, stat)
    if per_token is None:
        return False
    workload = record.get("workload", {})
    bound = slo.gate_per_token_ms(int(workload.get("isl", -1)), int(workload.get("osl", -1)))
    return per_token <= bound


def to_entry(record: dict, stat: str = "median") -> FrontierEntry | None:
    """Build a frontier entry, or None when decode throughput/latency metrics are absent."""
    metrics = record.get("metrics", {})
    thr = M.throughput(metrics, stat)
    per_token = M.per_token_ms(metrics, stat)
    if thr is None or per_token is None:
        return None
    workload = record.get("workload", {})
    return FrontierEntry(
        config_hash=record.get("config_hash", ""),
        branch=record.get("branch", ""),
        label=record.get("label", ""),
        workload=workload,
        context_length=_context_length(workload),
        throughput=thr,
        per_token_ms=per_token,
        ttft_ms=M.ttft_ms(metrics, stat),
    )


def _dominates(a: FrontierEntry, b: FrontierEntry) -> bool:
    """`a` dominates `b`: no worse on both axes (throughput up, per-token down), strictly
    better on at least one."""
    no_worse = a.throughput >= b.throughput and a.per_token_ms <= b.per_token_ms
    strictly = a.throughput > b.throughput or a.per_token_ms < b.per_token_ms
    return no_worse and strictly


def pareto_frontier(entries: list[FrontierEntry]) -> list[FrontierEntry]:
    """Non-dominated set: entries not dominated by any other entry."""
    return [
        e
        for i, e in enumerate(entries)
        if not any(_dominates(o, e) for j, o in enumerate(entries) if i != j)
    ]


def build_frontier(
    records: list[dict],
    slo: SLO,
    *,
    branch: str | None = None,
    stat: str = "median",
) -> tuple[list[FrontierEntry], list[FrontierEntry]]:
    """Return (SLO-passing entries, ranked Pareto frontier) for the chosen branch."""
    selected = [
        r
        for r in records
        if (branch is None or r.get("branch") == branch) and passes_slo(r, slo, stat)
    ]
    entries = [e for e in (to_entry(r, stat) for r in selected) if e is not None]
    frontier = pareto_frontier(entries)
    frontier.sort(key=lambda e: (-e.throughput, e.per_token_ms))
    return entries, frontier


def write_frontier(frontier: list[FrontierEntry], out_path) -> Path:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for e in frontier:
            f.write(json.dumps(asdict(e), default=str) + "\n")
    return path


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="Pareto frontier under an SLO over measured results (RFC-0001:C-OBJECTIVE)."
    )
    p.add_argument("--config", required=True, help="Versioned search config YAML (holds the SLO)")
    p.add_argument("--results", default=DEFAULT_RESULTS, help="Measured results JSONL")
    p.add_argument("--branch", default=None, help="Restrict to one precision branch")
    p.add_argument("--stat", choices=["median", "mean"], default="median")
    p.add_argument("--out", default=DEFAULT_FRONTIER, help="Frontier output JSONL")
    p.add_argument("--no-save", action="store_true", help="Print only; do not write --out")
    a = p.parse_args(argv)

    cfg = load_config(a.config)
    if cfg.slo is None:
        p.error("config defines no 'slo'; an SLO must be defined before the search (C-OBJECTIVE)")

    records = load_results(a.results)
    passing, frontier = build_frontier(records, cfg.slo, branch=a.branch, stat=a.stat)

    ttft_target = "none" if cfg.slo.ttft_p95_ms is None else f"{cfg.slo.ttft_p95_ms}ms"
    print(
        f"slo gate: decode ptok<= {cfg.slo.per_token_ms}ms"
        f"  (ttft target {ttft_target}, report-only)"
        f"  (+{len(cfg.slo.overrides)} per-pair override(s))"
    )
    print(f"records={len(records)}  slo_passing={len(passing)}  frontier={len(frontier)}")
    for rank, e in enumerate(frontier, 1):
        wl = e.workload
        ttft = "n/a" if e.ttft_ms is None else f"{e.ttft_ms:.0f}ms"
        print(
            f"{rank:>2}. {e.config_hash} {e.branch}/{e.label}  "
            f"isl{wl.get('isl')}-osl{wl.get('osl')}-c{wl.get('concurrency')} L={e.context_length}  "
            f"decode={e.throughput:.1f}tok/s  ptok={e.per_token_ms:.1f}ms  ttft~{ttft}"
        )

    if not a.no_save:
        out = write_frontier(frontier, a.out)
        print(f"wrote frontier to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
