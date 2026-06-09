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
workload = workload_points(cfg.workload_axes)        # inner: isl/osl × concurrency

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

> **Status:** the protocols + orchestration ship today and are unit-tested with fakes. A
> concrete `ServerManager`/`BenchClient` that launches SGLang and runs genai-bench /
> `bench_one_batch_server` is the next implementation slice. Until then, supply your own
> adapter implementing the contract above.

## Adapter sketch

```python
import subprocess, time
from sglbench.argsearch import args_to_cli  # from generate

class SglangServerManager:
    def __init__(self, model, port=8000):
        self.model, self.port = model, port

    def launch(self, args):
        proc = subprocess.Popen(
            ["python", "-m", "sglang.launch_server",
             "--model-path", self.model, "--port", str(self.port),
             *args_to_cli(args).split()]
        )
        _wait_until_ready(self.port)         # poll /health
        return SglangSession(proc, self.port)

class SglangSession:
    def __init__(self, proc, port):
        self.proc = proc
        self._client = MyBenchClient(port)   # implements warmup/measure
    @property
    def client(self):
        return self._client
    def shutdown(self):
        self.proc.terminate(); self.proc.wait()
```

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
