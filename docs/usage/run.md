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
the inner sweep. One `MeasurementResult` per `(config, workload-point)` is **streamed to
`<out>/results.jsonl` as each point completes** — a mid-run crash keeps the finished points.

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
- `--out DIR` — output dir for `results.jsonl`.
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
