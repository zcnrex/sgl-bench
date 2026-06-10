"""CLI that drives the live outer/inner search against a real SGLang server.

Wires the generator, the concrete SGLang adapter, and the driver: generate
restart-required configs ([[RFC-0001:C-CONFIG-SOURCE]]), launch each against the server and
sweep the workload axes ([[RFC-0001:C-LOOP-STRUCTURE]]), and write provenance records
([[RFC-0001:C-MEASUREMENT]]). This is the perf path; accuracy gating is out of scope here.
"""

from __future__ import annotations

import argparse
import sys

from .driver import run_search, workload_points, write_results
from .generate import accuracy_invariant_search, generate_grid, generate_ofat, load_config
from .objective import build_frontier
from .sglang_adapter import (
    GSM8KEvaluator,
    SGLangServerManager,
    build_bench_cmd,
    build_launch_cmd,
)

DEFAULT_OUT_DIR = "out"


def select_points(branch, mode: str, limit: int):
    """Generator points for the run, optionally truncated to the first `limit`."""
    points = generate_ofat(branch) if mode == "ofat" else generate_grid(branch)
    return points[:limit] if limit and limit > 0 else points


def select_workload(axes: dict, role: str, concurrency, isl_osl, limit: int):
    """Workload points for the run, with optional smoke-run axis overrides and truncation."""
    if concurrency or isl_osl:
        axes = dict(axes)
        if concurrency:
            axes["concurrency"] = list(concurrency)
        if isl_osl:
            pairs = [[int(a), int(b)] for a, b in (p.split("x") for p in isl_osl)]
            key = "report_isl_osl_pairs" if role == "report" else "isl_osl_pairs"
            axes[key] = pairs
    points = workload_points(axes, role)
    return points[:limit] if limit and limit > 0 else points


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="Run the live SGLang server-arg search (RFC-0001)."
    )
    p.add_argument("--config", required=True, help="Versioned search config YAML")
    p.add_argument("--branch", required=True, help="Precision branch name")
    p.add_argument("--mode", choices=["ofat", "grid"], default="ofat")
    p.add_argument("--limit-configs", type=int, default=0, help="Run only the first N configs (0=all)")
    p.add_argument("--role", choices=["iter", "report"], default="iter")
    p.add_argument("--concurrency", type=int, nargs="+", default=None, help="Override workload concurrency list")
    p.add_argument("--isl-osl", nargs="+", default=None, help="Override workload pairs as ISLxOSL")
    p.add_argument("--limit-workload", type=int, default=0, help="Run only the first N workload points (0=all)")
    p.add_argument("--model", default=None, help="Model path (default: branch.checkpoint or cfg.model)")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=30000)
    p.add_argument("--repeats", type=int, default=2, help="Measured repeats per workload point (>=2)")
    p.add_argument("--gsm8k-examples", type=int, default=0, help="Run gsm8k accuracy gate per config with N examples (0=disabled)")
    p.add_argument("--gsm8k-threads", type=int, default=32, help="Concurrent threads for the gsm8k eval")
    p.add_argument("--launch-timeout", type=float, default=1800.0, help="Seconds to wait for /health")
    p.add_argument("--out", default=DEFAULT_OUT_DIR, help="Output dir for results.jsonl")
    p.add_argument("--frontier", action="store_true", help="Build and print the frontier after the run")
    p.add_argument("--dry-run", action="store_true", help="Print launch/bench commands and exit")
    a = p.parse_args(argv)

    cfg = load_config(a.config)
    branch = cfg.branch(a.branch)
    model = a.model or branch.checkpoint or cfg.model
    points = select_points(branch, a.mode, a.limit_configs)
    workload = select_workload(cfg.workload_axes, a.role, a.concurrency, a.isl_osl, a.limit_workload)

    print(
        f"model={model}  configs={len(points)}  workload_points={len(workload)}  "
        f"repeats={a.repeats}  role={a.role}"
    )

    if a.dry_run:
        base_url = f"http://{a.host}:{a.port}"
        for cp in points:
            print(f"\n# config {cp.config_hash} {cp.branch}/{cp.label}")
            print(" ".join(build_launch_cmd(model, a.host, a.port, cp.args)))
            for wp in workload:
                print(" ".join(build_bench_cmd(base_url, wp, f"{a.out}/{cp.config_hash}-{wp.label}.jsonl")))
        return 0

    manager = SGLangServerManager(model, host=a.host, port=a.port, launch_timeout_s=a.launch_timeout)

    gate = None
    evaluate = None
    if a.gsm8k_examples > 0:
        if cfg.quality_gate is None:
            p.error("--gsm8k-examples set but config defines no quality_gate (C-QUALITY-GATE)")
        gate = cfg.quality_gate
        base_url = f"http://{a.host}:{a.port}"
        evaluator = GSM8KEvaluator(
            base_url,
            metric=gate.metric,
            num_examples=a.gsm8k_examples,
            num_threads=a.gsm8k_threads,
        )
        evaluate = lambda session: evaluator.evaluate()
        print(f"accuracy gate: gsm8k x{a.gsm8k_examples}  ({gate.metric} >= {gate.threshold})")

    evaluate_hashes = None
    if gate is not None and accuracy_invariant_search(branch, points):
        evaluate_hashes = {points[0].config_hash, points[-1].config_hash}
        print(
            f"  accuracy-invariant search: per-config eval skipped; "
            f"evaluating baseline + spot-check ({len(evaluate_hashes)} config(s))"
        )

    results = run_search(
        points, workload, manager,
        repeats=a.repeats, gate=gate, evaluate=evaluate, evaluate_hashes=evaluate_hashes,
    )
    out = write_results(results, a.out)
    print(f"wrote {len(results)} records to {out}")

    if a.frontier and cfg.slo is not None:
        passing, frontier = build_frontier(
            [r.to_record() for r in results], cfg.slo, branch=a.branch, gate=cfg.quality_gate
        )
        print(f"eligible={len(passing)}  frontier={len(frontier)}")
        for rank, e in enumerate(frontier, 1):
            print(
                f"{rank:>2}. {e.config_hash} {e.label}  "
                f"decode={e.throughput:.1f}tok/s  ptok={e.per_token_ms:.1f}ms"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
