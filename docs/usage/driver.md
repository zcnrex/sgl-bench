# Driver — outer/inner search loop

`run_search` drives the two nested loops of RFC-0001:C-LOOP-STRUCTURE:

- **Outer loop** — iterate over generator `ConfigPoint`s (restart-required config) and
  relaunch the server **once per config**.
- **Inner loop** — sweep the workload axes (isl/osl pairs × concurrency) against that single
  live server, **without relaunching**, measuring each point.

Workload axes are expanded only into the inner loop, never into the outer restart-required
loop.

```python
from sglbench.argsearch import (
    load_config, generate_ofat, workload_points, run_search, write_results,
)

cfg = load_config("configs/nemotron_v3_ultra.yaml")
branch = cfg.branch("nvfp4")

points = generate_ofat(branch)                       # outer: restart-required configs
workload = workload_points(cfg.workload_axes, role="iter")  # inner: short-output search points
# role="report" (long-output) is run only on the chosen config(s) for the final report
# (RFC-0001:C-WORKLOAD-STAGING).

results = run_search(points, workload, manager, repeats=2)
write_results(results, "out")                         # -> out/results.jsonl
```

`run_search` launches once per `ConfigPoint`, sweeps the full workload list against that
session, then calls `session.shutdown()` (in a `finally`) before the next config launches.
It returns one `MeasurementResult` per `(config, workload-point)` — see [measure.md](measure.md).

## Server lifecycle contract

Server launch is abstracted behind two protocols so the orchestration is testable without a
GPU. A concrete adapter wraps the real `python -m sglang.launch_server` + benchmark
transport:

```python
class ServerManager:
    def launch(self, args: dict) -> ServerSession:
        ...  # start the server with these restart-required args, wait until ready

class ServerSession:
    @property
    def client(self) -> BenchClient:   # bound to this live server
        ...
    def shutdown(self) -> None:
        ...  # tear the server down
```

`launch(args)` receives a `ConfigPoint.args` dict (the same args `--format cli` renders).
The session exposes a `BenchClient` (warmup + measure — see [measure.md](measure.md)) and a
`shutdown()`.

> **Status:** a concrete adapter now ships in `sglang_adapter.py` — `SGLangServerManager`
> launches `python -m sglang.launch_server`, polls `/health` until ready, and tears the
> server down on `shutdown()`; `BenchOneBatchClient` drives `sglang.bench_one_batch_server`
> (the C-BASELINE-ANCHOR transport) and maps its result record to the canonical metric
> vocabulary in `metrics.py`. Command construction and result parsing are unit-tested
> without a GPU. **Before these numbers are relied upon, reconcile them against the stable
> baseline on a live server (RFC-0001:C-BASELINE-ANCHOR);** genai-bench (richer percentile
> metrics, e.g. true p95 TTFT) is a future transport behind the same `BenchClient`.

## Concrete adapter

```python
from sglbench.argsearch import (
    load_config, generate_ofat, workload_points, run_search, write_results,
    SGLangServerManager,
)

cfg = load_config("configs/nemotron_v3_ultra.yaml")
branch = cfg.branch("nvfp4")

manager = SGLangServerManager(branch.checkpoint, host="127.0.0.1", port=30000)
results = run_search(generate_ofat(branch), workload_points(cfg.workload_axes), manager)
write_results(results, "out")                         # -> out/results.jsonl
```

`SGLangServerManager` injects its subprocess launcher, `/health` probe, and bench runner,
so the orchestration is exercised with fakes in tests. The bench transport maps a
`WorkloadPoint` to one `bench_one_batch_server` batch run (`concurrency` → `--batch-size`,
`isl`/`osl` → `--input-len`/`--output-len`) and reads back the `--result-filename` JSONL.

## Shell alternative

If you prefer to drive launches by hand instead of `run_search`, pipe `--format cli`:

```bash
python -m sglbench.argsearch --config configs/nemotron_v3_ultra.yaml --branch nvfp4 \
    --mode ofat --format cli | while read -r line; do
  case "$line" in \#*) echo "$line"; continue;; esac
  python -m sglang.launch_server --model-path "$MODEL" $line --port 8000
  # ... warmup + bench (>=2 repeats) + record provenance + teardown ...
done
```
