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
from typing import Iterable, Protocol, runtime_checkable

from .generate import ConfigPoint
from .measure import (
    MIN_REPEATS,
    BenchClient,
    MeasurementResult,
    WorkloadPoint,
    measure_point,
)


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
) -> list[MeasurementResult]:
    """Drive the outer/inner search and return one MeasurementResult per inner point."""
    workload = list(workload)
    results: list[MeasurementResult] = []
    for cp in points:
        session = manager.launch(cp.args)
        try:
            for wp in workload:
                results.append(
                    measure_point(
                        session.client,
                        config_hash=cp.config_hash,
                        branch=cp.branch,
                        point=wp,
                        repeats=repeats,
                    )
                )
        finally:
            session.shutdown()
    return results


def write_results(results: Iterable[MeasurementResult], out_dir: str) -> Path:
    outdir = Path(out_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / "results.jsonl"
    with path.open("w") as f:
        for res in results:
            f.write(json.dumps(asdict(res), default=str) + "\n")
    return path
