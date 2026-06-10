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

from dataclasses import asdict
from pathlib import Path
from typing import Callable, Iterable, Protocol, runtime_checkable

from .generate import ConfigPoint
from .jsonl import json_line, write_jsonl
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


class _AccuracyGater:
    """Per-launched-config accuracy scoring and gate stamping ([[RFC-0001:C-QUALITY-GATE]]).

    Evaluates the live server for configs selected by `evaluate_hashes` (all configs when
    None), reuses the most recent score for the rest, tracks the branch baseline score, and
    returns the (accuracy, quality_pass) stamp for every record of the config. Inactive
    (stamps None/None) unless both a gate and an evaluate callable are supplied.
    """

    def __init__(
        self,
        gate: QualityGate | None,
        evaluate: Callable[[ServerSession], dict[str, float]] | None,
        evaluate_hashes: set[str] | None,
    ) -> None:
        self._gate = gate
        self._evaluate = evaluate
        self._hashes = evaluate_hashes
        self._last_accuracy: dict[str, float] | None = None
        self._baseline_score: float | None = None

    def stamp(
        self, cp: ConfigPoint, session: ServerSession
    ) -> tuple[dict[str, float] | None, bool | None]:
        if self._gate is None or self._evaluate is None:
            return None, None
        if self._hashes is None or cp.config_hash in self._hashes:
            self._last_accuracy = self._evaluate(session)
        accuracy = self._last_accuracy
        score = accuracy.get(self._gate.metric) if accuracy else None
        if cp.label == "baseline" and score is not None:
            self._baseline_score = score
        if self._baseline_score is None:
            return accuracy, None
        return accuracy, self._gate.passes(score, self._baseline_score)


def run_search(
    points: Iterable[ConfigPoint],
    workload: Iterable[WorkloadPoint],
    manager: ServerManager,
    *,
    repeats: int = MIN_REPEATS,
    gate: QualityGate | None = None,
    evaluate: Callable[[ServerSession], dict[str, float]] | None = None,
    evaluate_hashes: set[str] | None = None,
    skip_keys: set[tuple[str, str]] | None = None,
    on_config: Callable[[ConfigPoint, list[MeasurementResult]], None] | None = None,
    on_result: Callable[[MeasurementResult], None] | None = None,
) -> list[MeasurementResult]:
    """Drive the outer/inner search and return one MeasurementResult per inner point.

    When a `gate` and an `evaluate` callable are supplied, accuracy is scored per launched
    configuration and every inner-point record for that config is stamped with the score
    and the gate verdict ([[RFC-0001:C-QUALITY-GATE]]). When `evaluate_hashes` is given,
    accuracy is scored only for configs whose hash is in the set (the baseline + spot-check
    of an accuracy-invariant search) and the most recent score is reused for the rest.

    `skip_keys` is a set of `(config_hash, workload_point.label)` already recorded; matching
    points are not measured and the server is not relaunched for a config whose points are all
    skipped, making an extended search incremental ([[RFC-0001:C-RUN-OUTPUT]]).
    """
    workload = list(workload)
    gater = _AccuracyGater(gate, evaluate, evaluate_hashes)
    skip = skip_keys or set()
    results: list[MeasurementResult] = []
    for cp in points:
        pending = [wp for wp in workload if (cp.config_hash, wp.label) not in skip]
        if not pending:
            continue
        session = manager.launch(cp.args)
        config_results: list[MeasurementResult] = []
        try:
            accuracy, quality_pass = gater.stamp(cp, session)
            for wp in pending:
                res = measure_point(
                    session.client,
                    config_hash=cp.config_hash,
                    branch=cp.branch,
                    point=wp,
                    repeats=repeats,
                    branch_keys=cp.branch_keys,
                    config_label=cp.label,
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
    return json_line(asdict(res))


def write_results(results: Iterable[MeasurementResult], out_dir: str) -> Path:
    return write_jsonl(
        (asdict(res) for res in results), Path(out_dir) / "results.jsonl"
    )
