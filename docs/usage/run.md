# Run — drive the live search on a server

`argsearch-run` (`python -m sglbench.argsearch.run`) is the end-to-end driver: it generates
restart-required configs, launches an SGLang server per config, sweeps the workload axes,
optionally runs the accuracy gate, streams provenance records, and prints the frontier. It
ties together `generate.py`, `sglang_adapter.py`, `driver.py`, and `objective.py`.

```bash
python -m sglbench.argsearch.run --config configs/nemotron_v3_ultra.yaml --branch nvfp4 \
    --mode ofat --isl-osl 8192x256 --concurrency 1 8 32 --port 40000 --frontier
# or, after install:  argsearch-run --config ... --branch nvfp4 --mode ofat ...
```

Each config is an outer-loop relaunch (RFC-0001:C-LOOP-STRUCTURE); the workload points are
the inner sweep. One `MeasurementResult` per `(config, workload-point)` is **streamed to the
run dir's `results.jsonl` as each point completes** — a mid-run crash keeps the finished
points.

## Output layout & measurement reuse (ADR-0007, RFC-0001:C-RUN-OUTPUT)

Results are isolated per `(model, transport, execution-environment)`:

```
<out>/<model>/runs/<transport>/<env>/{results.jsonl, manifest.json}
```

- `<model>` — filesystem-safe slug of the served checkpoint (branch `checkpoint`, else config
  `model`).
- `<transport>` — the bench tool name: `bench_one_batch_server` for `--transport one-batch`,
  `bench_serving` for `--transport serving`.
- `<env>` — an 8-hex digest of the execution environment (SGLang commit + key library versions
  + cluster networking env). A new build, library set, or NCCL/transport env lands in a new
  `<env>` dir, so non-comparable numbers are never mixed.

`--out` is the **base** dir (default `out`); the `<model>/runs/<transport>/<env>` subpath is
computed. `manifest.json` records the search request (config path, branch, mode, transport,
requested workload, gate, `force`) plus the environment and a `measured_at` timestamp — written
even when zero points are eligible (the timestamp is metadata, never part of the identity).
Every run also **appends** its manifest to `manifests.jsonl` in the same dir, so the record of
each invocation — including a `--force` re-measure that overrode prior results — is durable and
not clobbered by a later run (RFC-0001:C-RUN-OUTPUT).

**Reuse is the default.** Before measuring, the runner reads any existing `results.jsonl` in
the target dir and skips every `(config_hash, workload-point)` already recorded — and does not
relaunch the server for a config whose points are all already present. Re-invoking with more
concurrencies or candidates therefore measures only the missing points and **appends** them.
Pass `--force` to ignore the existing file and re-measure everything (the old `results.jsonl`
is replaced).

## Selecting configs and workload

- `--mode {ofat,grid}` — OFAT sensitivity sweep, or the focused grid (reads the branch's
  `focused_grid`; see [config.md](config.md)).
- `--limit-configs N` — run only the first N generated configs (N=1 is the baseline).
- `--only-config <hash|label-substr>` — run only configs matching a `config_hash` or label
  substring (e.g. characterize one chosen config).
- `--role {iter,report}` — pick the workload cost-role set (RFC-0001:C-WORKLOAD-STAGING):
  `iter` (short-output, drives the search) or `report` (long-output, final characterization).
- `--concurrency C [C ...]` and `--isl-osl ISLxOSL [...]` — override the workload axes for
  the run; `--limit-workload N` truncates to the first N points.

## Bench transport

`--transport {one-batch,serving}`:

- **`one-batch`** (default) — `bench_one_batch_server`, the stable comparability anchor
  (RFC-0001:C-BASELINE-ANCHOR). One batch, single tail decode step.
- **`serving`** — `bench_serving`: many requests at the target concurrency, reporting ITL
  percentiles. The decode rate is derived from median ITL (`concurrency / median_itl`), not
  the OSL-blended `output_throughput`; `--serving-num-prompts` defaults to ≥5× concurrency
  for steady state. Reconcile against the anchor before relying on it (it agrees to ~0.1% on
  decode; see [objective.md](objective.md)).

## Accuracy gate

`--gsm8k-examples N` (with `--gsm8k-threads T`) enables the accuracy gate
(RFC-0001:C-QUALITY-GATE): gsm8k is evaluated once per launched config via `sgl-eval`, and
each record is stamped with the accuracy and a `quality_pass` flag. A config below the
config's `quality_gate.threshold` is excluded from the frontier but kept (flagged) in the
stream. For an **accuracy-invariant-only** search (all varied args marked
`accuracy_invariant`), per-config evaluation is auto-skipped — only the baseline + a
spot-check config are evaluated and that accuracy is reused. Without `--gsm8k-examples`, the
run is perf-only and (when a `quality_gate` is defined) nothing reaches the frontier.

## Other flags

- `--repeats N` (≥2) — measured repeats per point after warmup (RFC-0001:C-MEASUREMENT).
- `--port` (use 40000 on RadixArk devboxes; 30000 is platform-reserved), `--host`,
  `--launch-timeout`, `--model` (default: branch checkpoint / config model).
- `--out DIR` — **base** output dir; results land under `<out>/<model>/runs/<transport>/<env>/`.
- `--force` — re-measure every point, ignoring any existing `results.jsonl` in the run dir.
- `--frontier` — build and print the ranked frontier after the run.
- `--dry-run` — print the launch + bench commands for each config/point and exit (no GPU).

## Running on a RadixArk devbox

Use `scripts/devbox_sweep.sh`, which syncs the repo, launches `argsearch-run` **detached**
(survives ssh/proxy drops), polls reconnect-tolerantly, and fetches results:

```bash
scripts/devbox_sweep.sh --config configs/nemotron_v3_ultra.yaml --branch nvfp4 \
    --mode ofat --isl-osl 8192x256 --concurrency 1 8 32
```

The `run-argsearch-devbox` skill documents the procedure and the devbox traps (reserved
port, `HF_HOME=/scratch/huggingface`, `/sgl-workspace` outputs, launch-once discipline).
