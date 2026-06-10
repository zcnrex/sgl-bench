# sgl-bench

**Find the SGLang server-argument configuration that serves a large model fastest under a
latency target — without regressing accuracy.**

A 550B-class model has many restart-required server flags (tensor/expert/data parallelism,
attention and Mamba/SSM kernel backends, quantization, KV-cache dtype, static memory
fraction, CUDA-graph and chunked-prefill sizes). Each distinct combination costs a full
server relaunch, weight load, and warmup before a single measurement — so a brute-force grid
over flags × workloads × precision is infeasible. `sgl-bench` makes the search tractable: it
bounds and buckets the arguments, sweeps them efficiently (sensitivity first, then a small
focused grid), measures **decode** performance with reproducible provenance, and ranks a
throughput-vs-latency frontier under a service-level objective (SLO) — excluding any config
that fails an accuracy gate.

## How it works

```text
define a config            arguments bucketed fixed / candidate / constrained; SLO + accuracy gate
  → generate permutations  OFAT sensitivity sweep, then a focused grid over the few that interact
  → drive outer/inner loop relaunch the server per config (outer); sweep workload axes (inner)
  → measure each point     warmup + repeats, full provenance (commit, lib versions, env)
  → rank the frontier      decode throughput vs per-token latency, SLO-gated, accuracy-gated
```

Glossary:

- **restart-required arg** — a server flag that needs a relaunch to change (e.g. TP size,
  quantization). Contrast with **workload axes** (input/output length, concurrency), which are
  swept against one live server.
- **OFAT** — one-factor-at-a-time: vary a single candidate around a known-good baseline to see
  which args actually move performance, before any joint grid.
- **focused grid** — a small joint grid over only the args that showed sensitivity *and*
  plausibly interact (e.g. coupled pairs like `ep-size`/`moe-a2a-backend`).
- **decode-first objective** — rank by steady-state *decode* throughput vs per-token (inter-token)
  latency at a fixed context length; the SLO gate is decode latency, TTFT is report-only.
- **accuracy gate** — a config must pass an accuracy eval (e.g. gsm8k) to be eligible for the
  ranked results; failing configs are recorded and flagged, never ranked.

The methodology is specified normatively in **RFC-0001** (`docs/rfc/RFC-0001.md`); this repo
implements it.

## What you need

- **This repo, cloned** (you need `configs/` and `scripts/`, not just the package).
- **GPU-free stages** (define / generate / validate / analyze): Python + `pip install -e .`.
- **The live search** (step 4): a **GPU host** (devbox) with SGLang installed and the model in
  cache; the accuracy gate also needs `sgl-eval`. The server launch and benchmarking run *on
  that host* — `argsearch-run` spawns `sglang.launch_server` and benches it over localhost.

```bash
git clone https://github.com/zcnrex/sgl-bench && cd sgl-bench
pip install -e .          # for the CPU-side stages and tests
```

## Quickstart

Steps 1–3 are GPU-free and run anywhere. Step 4 runs on the GPU host (or drive it from your
laptop with `scripts/devbox_sweep.sh`, which syncs the repo to the devbox and runs it there).

```bash
# 1. Validate the search config
python -m sglbench.argsearch.validate configs/nemotron_v3_ultra.yaml

# 2. Generate restart-required configs — OFAT (sensitivity), then the focused grid
python -m sglbench.argsearch --config configs/nemotron_v3_ultra.yaml --branch nvfp4 --mode ofat
python -m sglbench.argsearch --config configs/nemotron_v3_ultra.yaml --branch nvfp4 --mode grid

# 3. (optional) write the generated configs to out/ as <hash>.json + index.jsonl
python -m sglbench.argsearch --config configs/nemotron_v3_ultra.yaml --branch nvfp4 \
    --mode ofat --save

# 4. Run the live search — drives it on the GPU host (here, from a laptop)
DEVBOX=<your-devbox> scripts/devbox_sweep.sh \
    --config configs/nemotron_v3_ultra.yaml --branch nvfp4 --mode ofat \
    --isl-osl 8192x256 --concurrency 1 8 32 --frontier
#   ... or, on the GPU host directly:
#   argsearch-run --config configs/nemotron_v3_ultra.yaml --branch nvfp4 --mode ofat \
#       --isl-osl 8192x256 --concurrency 1 8 32 --port 40000 --frontier
```

`scripts/devbox_sweep.sh` runs `argsearch-run` **detached** on the devbox (survives ssh drops),
streams `out/results.jsonl` one record per workload point, and fetches results back. Add
`--gsm8k-examples N` to run the accuracy gate per config; `--transport serving` to use
`bench_serving` (percentile ITL) instead of the `bench_one_batch_server` anchor; `--dry-run` to
print the launch/bench commands without running. Full flag reference: [docs/usage/run.md](docs/usage/run.md).

Each measured point is one JSONL record with full provenance:

```json
{"config_hash": "d1ede40fd0b0", "branch": "nvfp4", "label": "isl8192-osl256-c1",
 "workload": {"isl": 8192, "osl": 256, "concurrency": 1},
 "metrics": {"decode_throughput_tok_s": {"median": 107.0, ...}, "per_token_ms": {"median": 9.35}, ...},
 "environment": {"sglang_commit": "0.5.12.post1", "library_versions": {...}, "network_env": {...}},
 "accuracy": {"accuracy": 0.975}, "quality_pass": true}
```

`--frontier` (or `argsearch-frontier --config … --results out/results.jsonl`) ranks them:

```text
slo gate: decode ptok<= 40.0ms  (ttft target 5000ms, report-only)
quality gate: accuracy on gsm8k (min 0.95)
records=21  slo_passing=21  quality_excluded=0  eligible=21  frontier=2
 1. d1ede40fd0b0 nvfp4/baseline  isl8192-osl256-c32  decode=1861.7tok/s  ptok=17.2ms  ttft~3000ms
 2. d1ede40fd0b0 nvfp4/baseline  isl8192-osl256-c1   decode=107.0tok/s   ptok=9.3ms   ttft~430ms
```

(Aggregate decode throughput and per-user latency trade off across concurrency, so several
points are non-dominated.)

## The config is yours to adapt

`configs/nemotron_v3_ultra.yaml` is a **seeded example** for one model on a 4-GPU host. To
search a different model/hardware, write your own: bucket each server arg into `fixed`,
`candidate` (≤10), or constraint rules; declare precision as a top-level `branch`; set the
`slo` and `quality_gate`. **The SLO and accuracy-gate thresholds in the example are phase-0
placeholders — set them to your real service targets before trusting the results.** See
[docs/usage/config.md](docs/usage/config.md).

## Documentation

Per-component guides in [`docs/usage/`](docs/usage/README.md):

| Stage | Guide |
| --- | --- |
| Define & validate the search | [config.md](docs/usage/config.md) |
| Generate restart-required configs | [generate.md](docs/usage/generate.md) |
| Drive the outer/inner search loop | [driver.md](docs/usage/driver.md) |
| Measure a workload point | [measure.md](docs/usage/measure.md) |
| Rank the Pareto frontier under the SLO | [objective.md](docs/usage/objective.md) |
| Run the live search (CLI) | [run.md](docs/usage/run.md) |

For running on a devbox there is also a `run-argsearch-devbox` skill capturing the procedure
and host-specific gotchas (port, paths, launch discipline).

## Develop

```bash
python -m unittest discover -s tests
```

The methodology is governed by govctl: RFC-0001 is the normative spec (implemented); RFC-0002
(a performance-improvement opportunity loop) is specified but **not yet implemented**; the
design decisions are in `gov/adr/`. Changes to the methodology go through govctl
(`/discuss` → `/spec` → `/gov`), not direct edits under `gov/`.
