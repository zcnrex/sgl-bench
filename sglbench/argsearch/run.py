"""CLI that drives the live outer/inner search against a real SGLang server.

Wires the generator, the concrete SGLang adapter, and the driver: generate
restart-required configs ([[RFC-0001:C-CONFIG-SOURCE]]), launch each against the server and
sweep the workload axes ([[RFC-0001:C-LOOP-STRUCTURE]]), and write provenance records
([[RFC-0001:C-MEASUREMENT]]). When `--gsm8k-examples` is set it also wires the per-branch,
baseline-relative accuracy gate ([[RFC-0001:C-QUALITY-GATE]]).
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone

from pathlib import Path

from .driver import result_line, run_search, workload_points
from .generate import accuracy_invariant_search, generate_grid, generate_ofat, load_config
from .measure import capture_environment, environment_digest, model_slug
from .objective import build_frontier
from .sglang_adapter import (
    BenchServingClient,
    GSM8KEvaluator,
    SGLangServerManager,
    build_bench_cmd,
    build_launch_cmd,
)

DEFAULT_OUT_DIR = "out"

TRANSPORT_TOOL = {"one-batch": "bench_one_batch_server", "serving": "bench_serving"}


def run_dir(base, model: str, transport: str, env_digest: str) -> Path:
    """Result-set directory for one (model, transport, environment) ([[ADR-0007]])."""
    tool = TRANSPORT_TOOL[transport]
    return Path(base) / model_slug(model) / "runs" / tool / env_digest


def read_skip_keys(results_path: Path) -> set[tuple[str, str]]:
    """`(config_hash, label)` of already-recorded measurements ([[RFC-0001:C-RUN-OUTPUT]])."""
    keys: set[tuple[str, str]] = set()
    if not results_path.exists():
        return keys
    for line in results_path.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        keys.add((rec.get("config_hash", ""), rec.get("label", "")))
    return keys


def select_points(branch, mode: str, limit: int, only_config: str | None = None):
    """Generator points for the run; optionally restrict to configs matching `only_config`
    (substring of config_hash or label) and/or truncate to the first `limit`."""
    points = generate_ofat(branch) if mode == "ofat" else generate_grid(branch)
    if only_config:
        points = [p for p in points if only_config in p.config_hash or only_config in p.label]
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
    p.add_argument("--only-config", default=None, help="Run only configs matching this config_hash/label substring")
    p.add_argument("--role", choices=["iter", "report"], default="iter")
    p.add_argument("--concurrency", type=int, nargs="+", default=None, help="Override workload concurrency list")
    p.add_argument("--isl-osl", nargs="+", default=None, help="Override workload pairs as ISLxOSL")
    p.add_argument("--limit-workload", type=int, default=0, help="Run only the first N workload points (0=all)")
    p.add_argument("--model", default=None, help="Served model checkpoint (default: cfg.model)")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=30000)
    p.add_argument("--transport", choices=["one-batch", "serving"], default="one-batch",
                   help="Bench transport")
    p.add_argument("--serving-num-prompts", type=int, default=0, help="Prompts for serving transport (0=auto)")
    p.add_argument("--repeats", type=int, default=2, help="Measured repeats per workload point (>=2)")
    p.add_argument("--gsm8k-examples", type=int, default=0, help="Run gsm8k accuracy gate per config with N examples (0=disabled)")
    p.add_argument("--gsm8k-threads", type=int, default=32, help="Concurrent threads for the gsm8k eval")
    p.add_argument("--launch-timeout", type=float, default=1800.0, help="Seconds to wait for /health")
    p.add_argument("--out", default=DEFAULT_OUT_DIR, help="Base output dir; results land under <out>/<model>/runs/<transport>/<env>/")
    p.add_argument("--force", action="store_true", help="Re-measure all points, ignoring any existing results.jsonl in the run dir")
    p.add_argument("--frontier", action="store_true", help="Build and print the frontier after the run")
    p.add_argument("--dry-run", action="store_true", help="Print launch/bench commands and exit")
    a = p.parse_args(argv)

    cfg = load_config(a.config)
    branch = cfg.branch(a.branch)
    model = a.model or cfg.model
    points = select_points(branch, a.mode, a.limit_configs, a.only_config)
    if not points:
        p.error(f"no configs matched --only-config {a.only_config!r}")
    workload = select_workload(cfg.workload_axes, a.role, a.concurrency, a.isl_osl, a.limit_workload)

    environment = capture_environment()
    env_digest = environment_digest(environment)
    target_dir = run_dir(a.out, model, a.transport, env_digest)
    out_path = target_dir / "results.jsonl"

    print(
        f"model={model}  configs={len(points)}  workload_points={len(workload)}  "
        f"repeats={a.repeats}  role={a.role}  env={env_digest}"
    )

    if a.dry_run:
        base_url = f"http://{a.host}:{a.port}"
        for cp in points:
            print(f"\n# config {cp.config_hash} {cp.branch}/{cp.label}")
            print(" ".join(build_launch_cmd(model, a.host, a.port, cp.args)))
            for wp in workload:
                print(" ".join(build_bench_cmd(base_url, wp, str(target_dir / f"{cp.config_hash}-{wp.label}.jsonl"))))
        return 0

    bench_factory = None
    if a.transport == "serving":
        nump = a.serving_num_prompts or None
        bench_factory = lambda url: BenchServingClient(url, tokenizer=model, num_prompts=nump)
    manager = SGLangServerManager(
        model, host=a.host, port=a.port, launch_timeout_s=a.launch_timeout,
        bench_client_factory=bench_factory,
    )

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
        print(
            f"accuracy gate: gsm8k x{a.gsm8k_examples}  "
            f"({gate.metric} within {gate.tolerance} of the branch baseline)"
        )

    evaluate_hashes = None
    if gate is not None and accuracy_invariant_search(branch, points):
        evaluate_hashes = {points[0].config_hash, points[-1].config_hash}
        print(
            f"  accuracy-invariant search: per-config eval skipped; "
            f"evaluating baseline + spot-check ({len(evaluate_hashes)} config(s))"
        )

    target_dir.mkdir(parents=True, exist_ok=True)
    skip_keys = set() if a.force else read_skip_keys(out_path)
    if a.force and out_path.exists():
        out_path.unlink()
    elif skip_keys:
        print(f"  reusing {len(skip_keys)} recorded measurement(s) in {out_path}")

    manifest = {
        "config": str(Path(a.config)),
        "branch": a.branch,
        "mode": a.mode,
        "transport": a.transport,
        "bench_tool": TRANSPORT_TOOL[a.transport],
        "model": model,
        "role": a.role,
        "repeats": a.repeats,
        "force": a.force,
        "workload": [{"isl": wp.isl, "osl": wp.osl, "concurrency": wp.concurrency} for wp in workload],
        "gate": (
            {"dataset": gate.dataset, "metric": gate.metric, "tolerance": gate.tolerance,
             "direction": gate.direction, "gsm8k_examples": a.gsm8k_examples}
            if gate is not None else None
        ),
        "environment": environment,
        "measured_at": datetime.now(timezone.utc).isoformat(),
    }
    (target_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    with (target_dir / "manifests.jsonl").open("a") as mf:
        mf.write(json.dumps(manifest, default=str) + "\n")

    written = 0
    with out_path.open("a") as rf:
        def sink(res):
            nonlocal written
            rf.write(result_line(res) + "\n")
            rf.flush()
            written += 1
            print(f"  [{res.config_hash} {res.label}] -> {out_path} ({written})")

        results = run_search(
            points, workload, manager,
            repeats=a.repeats, gate=gate, evaluate=evaluate,
            evaluate_hashes=evaluate_hashes, skip_keys=skip_keys, on_result=sink,
        )
    print(f"wrote {written} records to {out_path} (streamed per workload point)")

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
