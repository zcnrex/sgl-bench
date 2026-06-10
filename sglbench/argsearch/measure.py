"""Measurement of a single (config, workload-point) for the arg search.

Implements [[RFC-0001:C-MEASUREMENT]]: every measured configuration is warmed up
before any recorded timing, measured with at least two repeat runs, and recorded with
full reproducibility provenance -- config identity (content hash), the workload point,
the measured metrics, and the execution environment (the hardware target -- accelerator
model and device count -- the SGLang commit, library versions, and the cluster
networking environment variables in effect).

The actual benchmark transport (genai-bench / bench_one_batch_server against a live
server) is abstracted behind the BenchClient protocol so this logic is exercised
without a GPU. A workload point is one inner-loop coordinate
([[RFC-0001:C-LOOP-STRUCTURE]]); sweeping it is the driver's job, not this module's.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import asdict, dataclass, field
from importlib import metadata
from statistics import fmean, median
from typing import Protocol, runtime_checkable

MIN_REPEATS = 2

PROVENANCE_LIBRARIES = (
    "sglang",
    "sgl-kernel",
    "torch",
    "flashinfer-python",
    "flashinfer",
    "transformers",
)

NETWORK_ENV_PREFIXES = (
    "NCCL_",
    "TORCH_NCCL_",
    "NVSHMEM_",
    "GLOO_",
    "UCX_",
    "OMPI_",
    "PMIX_",
    "IB_",
    "RDMA_",
    "RDMAV_",
    "MLX4_",
    "MLX5_",
    "EFA_",
    "FI_",
    "SHARP_",
    "HCOLL_",
    "GDRCOPY_",
    "NVLS_",
    "GPU_",
)

NETWORK_ENV_NAMES = (
    "MASTER_ADDR",
    "MASTER_PORT",
    "WORLD_SIZE",
    "RANK",
    "LOCAL_RANK",
    "LOCAL_WORLD_SIZE",
    "GROUP_RANK",
    "NODE_RANK",
)

NETWORK_ENV_EXTRA_VAR = "SGLBENCH_NET_ENV_EXTRA"

HARDWARE_ENV_VAR = "SGLBENCH_HARDWARE"


@dataclass(frozen=True)
class WorkloadPoint:
    """One inner-loop coordinate: a fixed workload swept against a live server."""

    isl: int
    osl: int
    concurrency: int

    @property
    def label(self) -> str:
        return f"isl{self.isl}-osl{self.osl}-c{self.concurrency}"


@runtime_checkable
class BenchClient(Protocol):
    """Transport to a live server. Implementations wrap the real benchmark tools."""

    def warmup(self, point: WorkloadPoint) -> None:
        """Run an unrecorded pass to prime caches/CUDA graphs before timing."""

    def measure(self, point: WorkloadPoint) -> dict[str, float]:
        """Run one recorded pass and return its metrics."""


@runtime_checkable
class AccuracyEvaluator(Protocol):
    """Accuracy-gate evaluator bound to a live server ([[RFC-0001:C-QUALITY-GATE]]).

    Evaluated once per launched configuration, not per workload point. Implementations
    wrap the real accuracy harness; returns metric -> score on the gate's dataset.
    """

    def evaluate(self) -> dict[str, float]:
        """Score the live server on the gate's evaluation dataset."""


def _bench_tool(client: BenchClient) -> str | None:
    return getattr(client, "tool", None)


@dataclass
class MeasurementResult:
    config_hash: str
    branch: str
    label: str
    workload: dict
    metrics: dict  # per-metric {mean, median, n} aggregates across repeats
    repeats: list[dict]  # raw per-repeat metrics
    environment: dict
    config_label: str = ""
    branch_keys: dict | None = None
    bench_tool: str | None = None
    accuracy: dict | None = None
    quality_pass: bool | None = None

    def to_record(self) -> dict:
        return asdict(self)


def _detect_sglang_commit() -> str | None:
    """SGLang build identity for provenance. Prefer an explicit commit env var, then the
    installed package version; either is a stable cross-run anchor."""
    commit = os.environ.get("SGLANG_COMMIT")
    if commit:
        return commit
    try:
        return metadata.version("sglang")
    except metadata.PackageNotFoundError:
        return None


def _library_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in PROVENANCE_LIBRARIES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    return versions


def _extra_prefixes(env: dict) -> tuple[str, ...]:
    raw = env.get(NETWORK_ENV_EXTRA_VAR, "")
    return tuple(p for p in (s.strip() for s in raw.split(",")) if p)


def _network_env(environ: dict | None = None) -> dict[str, str]:
    env = os.environ if environ is None else environ
    prefixes = NETWORK_ENV_PREFIXES + _extra_prefixes(env)
    return {
        k: v
        for k, v in env.items()
        if k in NETWORK_ENV_NAMES or any(k.startswith(p) for p in prefixes)
    }


def _detect_hardware(environ: dict | None = None) -> dict:
    """Hardware-target branch key for the execution environment: accelerator model and
    device count ([[RFC-0001:C-MEASUREMENT]], [[RFC-0001:C-BRANCH]]).

    Prefers the explicit SGLBENCH_HARDWARE override (a free-form accelerator label), then
    a live CUDA query; either is absent on a GPU-free host and recorded as None.
    """
    env = os.environ if environ is None else environ
    label = env.get(HARDWARE_ENV_VAR)
    accelerator: str | None = label or None
    device_count: int | None = None
    try:
        import torch

        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            if accelerator is None and device_count:
                accelerator = torch.cuda.get_device_name(0)
    except Exception:
        pass
    return {"accelerator": accelerator, "device_count": device_count}


def capture_environment(
    sglang_commit: str | None = None, environ: dict | None = None
) -> dict:
    """Reproducibility provenance for the execution environment ([[RFC-0001:C-MEASUREMENT]])."""
    return {
        "hardware": _detect_hardware(environ),
        "sglang_commit": sglang_commit or _detect_sglang_commit(),
        "library_versions": _library_versions(),
        "network_env": _network_env(environ),
    }


def environment_digest(environment: dict) -> str:
    """Deterministic 8-hex digest of the execution environment ([[RFC-0001:C-RUN-OUTPUT]])."""
    blob = json.dumps(environment, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:8]


_SLUG_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


def model_slug(name: str) -> str:
    """Filesystem-safe single-segment slug of a model/checkpoint name."""
    s = _SLUG_SAFE.sub("-", str(name)).strip("-")
    return s or "model"


def label_slug(s: str) -> str:
    """Filesystem-safe slug of a config label (`=` becomes `__`)."""
    s = str(s).replace("=", "__")
    s = _SLUG_SAFE.sub("-", s).strip("-")
    return s or "x"


def _aggregate(runs: list[dict[str, float]]) -> dict:
    """Mean/median per metric across repeats. Keys present in every run are aggregated."""
    if not runs:
        return {}
    common = set(runs[0])
    for r in runs[1:]:
        common &= set(r)
    agg: dict[str, dict] = {}
    for k in sorted(common):
        vals = [float(r[k]) for r in runs]
        agg[k] = {"mean": fmean(vals), "median": median(vals), "n": len(vals)}
    return agg


def measure_point(
    client: BenchClient,
    *,
    config_hash: str,
    branch: str,
    point: WorkloadPoint,
    repeats: int = MIN_REPEATS,
    environment: dict | None = None,
    branch_keys: dict | None = None,
    config_label: str = "",
) -> MeasurementResult:
    """Warm up, run `repeats` recorded passes, and build a provenance record.

    Warmup always precedes any recorded timing, and at least MIN_REPEATS measured runs
    are required ([[RFC-0001:C-MEASUREMENT]]).
    """
    if repeats < MIN_REPEATS:
        raise ValueError(
            f"repeats={repeats} below the C-MEASUREMENT minimum of {MIN_REPEATS}"
        )

    client.warmup(point)
    runs = [client.measure(point) for _ in range(repeats)]

    env = environment if environment is not None else capture_environment()
    return MeasurementResult(
        config_hash=config_hash,
        branch=branch,
        label=point.label,
        workload=asdict(point),
        metrics=_aggregate(runs),
        repeats=runs,
        environment=env,
        config_label=config_label,
        branch_keys=branch_keys,
        bench_tool=_bench_tool(client),
    )
