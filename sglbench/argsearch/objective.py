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
from .generate import _assemble, config_hash, load_config
from .schema import SLO, PrecisionBranch, QualityGate

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
    accuracy: dict | None = None
    quality_pass: bool | None = None


def load_results(path) -> list[dict]:
    text = Path(path).read_text()
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def _context_length(workload: dict) -> int:
    return int(workload.get("isl", 0)) + int(workload.get("osl", 0))


def record_quality_pass(record: dict, gate: QualityGate) -> bool:
    """Gate verdict for a record: trust a stamped `quality_pass`, else recompute from the
    recorded accuracy; a record with no accuracy score fails ([[RFC-0001:C-QUALITY-GATE]])."""
    flag = record.get("quality_pass")
    if isinstance(flag, bool):
        return flag
    accuracy = record.get("accuracy") or {}
    return gate.passes(accuracy.get(gate.metric))


def passes_quality_gate(record: dict, gate: QualityGate | None) -> bool:
    """True when the record clears the accuracy gate, or when no gate is defined."""
    if gate is None:
        return True
    return record_quality_pass(record, gate)


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
        accuracy=record.get("accuracy"),
        quality_pass=record.get("quality_pass"),
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
    gate: QualityGate | None = None,
) -> tuple[list[FrontierEntry], list[FrontierEntry]]:
    """Return (eligible entries, ranked Pareto frontier) for the chosen branch.

    Eligibility requires clearing both the decode SLO and the accuracy gate; a
    gate-failing config is excluded from the acceptable results but remains in the
    record stream ([[RFC-0001:C-QUALITY-GATE]]).
    """
    selected = [
        r
        for r in records
        if (branch is None or r.get("branch") == branch)
        and passes_slo(r, slo, stat)
        and passes_quality_gate(r, gate)
    ]
    entries = [e for e in (to_entry(r, stat) for r in selected) if e is not None]
    frontier = pareto_frontier(entries)
    frontier.sort(key=lambda e: (-e.throughput, e.per_token_ms))
    return entries, frontier


def gate_failed_pins(
    branch: PrecisionBranch, records: list[dict], gate: QualityGate
) -> list[dict]:
    """Pins whose OFAT configuration failed the accuracy gate ([[RFC-0001:C-QUALITY-GATE]]).

    A non-gridded argument MUST NOT be pinned to a value whose OFAT config failed the gate.
    Returns one entry per offending pin (arg, value, config_hash) for the caller to reject.
    """
    fg = branch.focused_grid
    if fg is None:
        return []
    by_hash = {r.get("config_hash"): r for r in records}
    offenders: list[dict] = []
    for arg, value in fg.pins.items():
        cfg = _assemble(branch, {arg: value})
        h = config_hash(cfg)
        rec = by_hash.get(h)
        if rec is not None and not record_quality_pass(rec, gate):
            offenders.append({"arg": arg, "value": value, "config_hash": h})
    return offenders


def gate_status(entry: FrontierEntry, gate: QualityGate | None) -> tuple[str, float | None]:
    """Inspection-view gate verdict for a ranked entry ([[RFC-0001:C-QUALITY-GATE]]).

    Returns ("PASS" | "FAIL" | "n-a", score). Unlike the eligibility check, an entry whose
    accuracy was never measured is reported as `n-a` rather than folded into `FAIL`, so the
    speed-versus-quality trade-off stays legible for a human override decision.
    """
    if gate is None:
        return "n-a", None
    score = (entry.accuracy or {}).get(gate.metric)
    if isinstance(entry.quality_pass, bool):
        return ("PASS" if entry.quality_pass else "FAIL"), score
    if score is None:
        return "n-a", None
    return ("PASS" if gate.passes(score) else "FAIL"), score


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
    p.add_argument("--out", default=None, help=f"Frontier output JSONL (default: {DEFAULT_FRONTIER})")
    p.add_argument("--no-save", action="store_true", help="Print only; do not write --out")
    p.add_argument(
        "--inspect",
        action="store_true",
        help="Read-only inspection view: rank gate-failing/unmeasured configs too, annotate "
        "each row with its gate status, and refuse --out (never acceptable results).",
    )
    a = p.parse_args(argv)

    if a.inspect and a.out is not None:
        p.error(
            "--inspect and --out are mutually exclusive; the inspection view includes "
            "gate-failing configs and is never written to the acceptable-results artifact "
            "(C-QUALITY-GATE). Redirect stdout to persist it."
        )
    out_path = a.out if a.out is not None else DEFAULT_FRONTIER

    cfg = load_config(a.config)
    if cfg.slo is None:
        p.error("config defines no 'slo'; an SLO must be defined before the search (C-OBJECTIVE)")

    gate = cfg.quality_gate
    records = load_results(a.results)
    effective_gate = None if a.inspect else gate
    passing, frontier = build_frontier(
        records, cfg.slo, branch=a.branch, stat=a.stat, gate=effective_gate
    )

    in_scope = [r for r in records if a.branch is None or r.get("branch") == a.branch]
    slo_only = [r for r in in_scope if passes_slo(r, cfg.slo, a.stat)]
    quality_excluded = [r for r in slo_only if not passes_quality_gate(r, gate)]

    ttft_target = "none" if cfg.slo.ttft_p95_ms is None else f"{cfg.slo.ttft_p95_ms}ms"
    print(
        f"slo gate: decode ptok<= {cfg.slo.per_token_ms}ms"
        f"  (ttft target {ttft_target}, report-only)"
        f"  (+{len(cfg.slo.overrides)} per-pair override(s))"
    )
    if gate is None:
        print("quality gate: none defined (C-QUALITY-GATE: a gate SHOULD be defined before the search)")
    else:
        bound = "min" if gate.direction == "higher" else "max"
        print(f"quality gate: {gate.metric} on {gate.dataset} ({bound} {gate.threshold})")

    if a.inspect:
        print(
            "!! INSPECTION VIEW — includes gate-FAILING / unmeasured configs; "
            "NOT acceptable results (RFC-0001:C-QUALITY-GATE)"
        )
        print(
            f"records={len(records)}  slo_passing={len(slo_only)}  "
            f"eligible(ignoring gate)={len(passing)}  frontier={len(frontier)}"
        )
    else:
        print(
            f"records={len(records)}  slo_passing={len(slo_only)}  "
            f"quality_excluded={len(quality_excluded)}  eligible={len(passing)}  frontier={len(frontier)}"
        )

    for rank, e in enumerate(frontier, 1):
        wl = e.workload
        ttft = "n/a" if e.ttft_ms is None else f"{e.ttft_ms:.0f}ms"
        line = (
            f"{rank:>2}. {e.config_hash} {e.branch}/{e.label}  "
            f"isl{wl.get('isl')}-osl{wl.get('osl')}-c{wl.get('concurrency')} L={e.context_length}  "
            f"decode={e.throughput:.1f}tok/s  ptok={e.per_token_ms:.1f}ms  ttft~{ttft}"
        )
        if a.inspect:
            status, score = gate_status(e, gate)
            score_s = "n/a" if score is None else f"{score}"
            line += f"  gate={status}({gate.metric if gate else 'accuracy'}={score_s})"
        print(line)

    if a.inspect:
        return 0

    if quality_excluded:
        print(f"\nquality-excluded (SLO-passing but below the accuracy gate; recorded, not eligible):")
        for r in quality_excluded:
            acc = (r.get("accuracy") or {}).get(gate.metric) if gate else None
            acc_s = "n/a" if acc is None else f"{acc}"
            print(f"   {r.get('config_hash')} {r.get('branch')}/{r.get('label')}  {gate.metric if gate else 'accuracy'}={acc_s}")

    branches = [a.branch] if a.branch else [b.name for b in cfg.precision_branches]
    for bn in branches:
        bad_pins = gate_failed_pins(cfg.branch(bn), in_scope, gate) if gate else []
        if bad_pins:
            detail = ", ".join(f"{o['arg']}={o['value']}" for o in bad_pins)
            print(
                f"branch {bn}: no acceptable configuration (C-QUALITY-GATE) — "
                f"focused-grid pin(s) failed the accuracy gate: {detail}"
            )
        elif not any(e.branch == bn for e in frontier):
            print(f"branch {bn}: no acceptable configuration (C-QUALITY-GATE)")

    if not a.no_save:
        out = write_frontier(frontier, out_path)
        print(f"wrote frontier to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
