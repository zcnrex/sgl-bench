"""Outer/inner search driver for the SGLang server-arg search.

Implements [[RFC-0001:C-LOOP-STRUCTURE]]: the outer loop iterates over generator
ConfigPoints (restart-required server configuration) and relaunches the server once per
config; the inner loop sweeps the workload axes (isl/osl pairs x concurrency) against
that single live server without relaunching. Workload axes are expanded only into the
inner loop, never into the outer restart-required loop. Each inner point is measured by
the WI-004 unit ([[RFC-0001:C-MEASUREMENT]]).

Server lifecycle is behind the ServerManager / ServerSession protocols so the
orchestration runs without a GPU; concrete implementations wrap the real SGLang launch
and benchmark transport.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Iterable, Protocol, runtime_checkable

from .generate import ConfigPoint
from .measure import (
    MIN_REPEATS,
    BenchClient,
    MeasurementResult,
    WorkloadPoint,
    measure_point,
)
from .schema import QualityGate


@runtime_checkable
class ServerSession(Protocol):
    """A launched, ready server bound to a benchmark client for the inner sweep."""

    @property
    def client(self) -> BenchClient: ...

    def shutdown(self) -> None: ...


@runtime_checkable
class ServerManager(Protocol):
    """Launches a server for one restart-required config and waits until it is ready."""

    def launch(self, args: dict) -> ServerSession: ...


ITER_PAIRS_KEY = "isl_osl_pairs"
REPORT_PAIRS_KEY = "report_isl_osl_pairs"


def workload_points(axes: dict, role: str = "iter") -> list[WorkloadPoint]:
    """Expand the inner-loop workload axes into concrete points.

    Cross-product of isl/osl pairs with concurrency ([[RFC-0001:C-LOOP-STRUCTURE]]). `role`
    ("iter" | "report") picks the cost-role pair set per [[RFC-0001:C-WORKLOAD-STAGING]].
    """
    key = REPORT_PAIRS_KEY if role == "report" else ITER_PAIRS_KEY
    pairs = axes.get(key, [])
    concurrencies = axes.get("concurrency", [])
    return [
        WorkloadPoint(isl=int(isl), osl=int(osl), concurrency=int(c))
        for isl, osl in pairs
        for c in concurrencies
    ]


def run_search(
    points: Iterable[ConfigPoint],
    workload: Iterable[WorkloadPoint],
    manager: ServerManager,
    *,
    repeats: int = MIN_REPEATS,
    gate: QualityGate | None = None,
    evaluate: Callable[[ServerSession], dict[str, float]] | None = None,
    evaluate_hashes: set[str] | None = None,
    on_config: Callable[[ConfigPoint, list[MeasurementResult]], None] | None = None,
    on_result: Callable[[MeasurementResult], None] | None = None,
) -> list[MeasurementResult]:
    """Drive the outer/inner search and return one MeasurementResult per inner point.

    When a `gate` and an `evaluate` callable are supplied, accuracy is scored per launched
    configuration and every inner-point record for that config is stamped with the score
    and the gate verdict ([[RFC-0001:C-QUALITY-GATE]]). When `evaluate_hashes` is given,
    accuracy is scored only for configs whose hash is in the set (the baseline + spot-check
    of an accuracy-invariant search) and the most recent score is reused for the rest.
    """
    workload = list(workload)
    gating = gate is not None and evaluate is not None
    results: list[MeasurementResult] = []
    last_accuracy: dict[str, float] | None = None
    for cp in points:
        session = manager.launch(cp.args)
        config_results: list[MeasurementResult] = []
        try:
            accuracy = None
            quality_pass = None
            if gating:
                if evaluate_hashes is None or cp.config_hash in evaluate_hashes:
                    last_accuracy = evaluate(session)
                accuracy = last_accuracy
                quality_pass = gate.passes(accuracy.get(gate.metric)) if accuracy else None
            for wp in workload:
                res = measure_point(
                    session.client,
                    config_hash=cp.config_hash,
                    branch=cp.branch,
                    point=wp,
                    repeats=repeats,
                )
                res.accuracy = accuracy
                res.quality_pass = quality_pass
                config_results.append(res)
                if on_result is not None:
                    on_result(res)
        finally:
            session.shutdown()
        results.extend(config_results)
        if on_config is not None:
            on_config(cp, config_results)
    return results


def result_line(res: MeasurementResult) -> str:
    """One JSONL record line for a measurement result."""
    return json.dumps(asdict(res), default=str)


def write_results(results: Iterable[MeasurementResult], out_dir: str) -> Path:
    outdir = Path(out_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / "results.jsonl"
    with path.open("w") as f:
        for res in results:
            f.write(result_line(res) + "\n")
    return path
