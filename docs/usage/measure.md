# Measure — one workload point with provenance

`measure_point` measures a single `(config, workload-point)` against a live server per
RFC-0001:C-MEASUREMENT: it warms up before any recorded timing, runs **≥ 2** repeats, and
returns a record with full reproducibility provenance.

```python
from sglbench.argsearch import measure_point, WorkloadPoint

point = WorkloadPoint(isl=8192, osl=1024, concurrency=32)
result = measure_point(
    client,                      # a BenchClient bound to the live server
    config_hash="d1ede40fd0b0",
    branch="nvfp4",
    point=point,
    repeats=2,                   # MIN_REPEATS; ValueError if < 2
)
```

`measure_point` calls `client.warmup(point)` first (never recorded), then
`client.measure(point)` `repeats` times. It raises `ValueError` if `repeats < 2`.

## The BenchClient contract

The transport to a live server is abstracted so the logic runs without a GPU. Implement two
methods — wrap genai-bench / `bench_one_batch_server` here:

```python
class MyBenchClient:
    def warmup(self, point: WorkloadPoint) -> None:
        ...  # prime caches / CUDA graphs; result discarded

    def measure(self, point: WorkloadPoint) -> dict[str, float]:
        ...  # one recorded pass -> metrics, e.g. {"ttft_p95": ..., "tput": ...}
```

## Result shape

`MeasurementResult.to_record()` yields a JSON-serializable dict:

```python
{
  "config_hash": "d1ede40fd0b0",
  "branch": "nvfp4",
  "label": "isl8192-osl1024-c32",
  "workload": {"isl": 8192, "osl": 1024, "concurrency": 32},
  "metrics": {"ttft_p95": {"mean": ..., "median": ..., "n": 2}, ...},  # aggregated across repeats
  "repeats": [ {...}, {...} ],                                          # raw per-repeat metrics
  "environment": {...},                                                 # see below
}
```

Aggregation reports `mean`/`median`/`n` per metric key present in every repeat; raw repeats
are retained alongside.

## Environment provenance

`capture_environment()` records the execution environment so runs are comparable and
reproducible:

```python
from sglbench.argsearch import capture_environment

env = capture_environment()          # or capture_environment(sglang_commit="<sha>")
# {
#   "sglang_commit": "<SGLANG_COMMIT env, else installed sglang version>",
#   "library_versions": {"sglang": "...", "torch": "...", "flashinfer": "...", ...},
#   "network_env": {"NCCL_DEBUG": "...", "MASTER_ADDR": "...", ...},
# }
```

`network_env` captures the cluster networking variables in effect: known families by prefix
(`NCCL_`, `TORCH_NCCL_`, `NVSHMEM_`, `GLOO_`, `UCX_`, `OMPI_`, `PMIX_`, `IB_`, `RDMA_`,
`RDMAV_`, `MLX4_`, `MLX5_`, `EFA_`, `FI_`, `SHARP_`, `HCOLL_`, `GDRCOPY_`, `NVLS_`, `GPU_`)
plus the torch-distributed rendezvous vars by exact name (`MASTER_ADDR`, `MASTER_PORT`,
`WORLD_SIZE`, `RANK`, `LOCAL_RANK`, `LOCAL_WORLD_SIZE`, `GROUP_RANK`, `NODE_RANK`).

To capture site-specific networking vars outside those families, set
`SGLBENCH_NET_ENV_EXTRA` to a comma-separated list of extra prefixes:

```bash
export SGLBENCH_NET_ENV_EXTRA="MYCLUSTER_NET_,SITE_FABRIC_"
```

Set `SGLANG_COMMIT` in the environment to pin the exact build commit; otherwise the
installed `sglang` package version is recorded as a best-effort anchor.

`measure_point` calls `capture_environment()` automatically unless you pass an explicit
`environment=` dict.
