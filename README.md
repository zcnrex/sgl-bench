# sgl-bench

**Find the fastest SGLang server configuration for a large model under a latency target,
without regressing accuracy.**

Large models expose many [restart-required server flags](#restart-required-arg): tensor, expert, and data
parallelism; attention and Mamba/SSM kernel backends; quantization; KV-cache dtype; static
memory fraction; CUDA graph settings; and chunked prefill sizes. Each combination requires a
server relaunch, weight load, and warmup before measurement, so a brute-force grid over
flags, workloads, and precision is usually infeasible.

`sgl-bench` makes the search tractable. It bounds and buckets server arguments, runs an
[OFAT](#ofat) sensitivity sweep before a small [focused grid](#focused-grid), measures
**decode** performance with full provenance, and ranks the throughput-vs-latency frontier
under a service-level objective (SLO). Configs that fail the [accuracy gate](#accuracy-gate)
are recorded, but excluded from rankings.

## How it works

| Step | What happens |
| --- | --- |
| Define a config | Bucket arguments as fixed, candidate, or constrained; set the SLO and [accuracy gate](#accuracy-gate). |
| Generate permutations | Run an [OFAT](#ofat) sensitivity sweep, then a [focused grid](#focused-grid) over interacting args. |
| Drive the search loop | Relaunch the server per config; sweep [workload axes](#workload-axes) against each live server. |
| Measure each point | Run warmup and repeats; record provenance such as commit, library versions, and env. |
| Rank the frontier | Compare decode throughput vs per-token latency after SLO and accuracy gates. |

The live search is a nested loop:

```text
outer loop: for each restart-required config
  launch one SGLang server

  inner loop: for each workload point
    measure ISL x OSL x concurrency against that live server

  stop the server
```

The [outer loop](#outer-loop) changes server args that require a relaunch. The
[inner loop](#inner-loop) changes request shape and concurrency while reusing the same live
server.

## Glossary

| Term | Meaning |
| --- | --- |
| <a id="restart-required-arg"></a>**restart-required arg** | A server flag that needs a relaunch to change, such as TP size or quantization. |
| <a id="workload-axes"></a>**workload axes** | Runtime workload dimensions, such as input length, output length, and concurrency, that are swept against one live server. |
| <a id="outer-loop"></a>**outer loop** | The search loop over restart-required configs. Each iteration launches SGLang with one server-arg combination, measures it, then stops it. |
| <a id="inner-loop"></a>**inner loop** | The workload loop inside one live server run. It measures each input/output length and concurrency point without changing restart-required server args. |
| <a id="ofat"></a>**OFAT** | One-factor-at-a-time: vary a single candidate around a known-good baseline to see which args actually move performance before any joint grid. |
| <a id="focused-grid"></a>**focused grid** | A small joint grid over only the args that showed sensitivity *and* plausibly interact, such as coupled pairs like `ep-size`/`moe-a2a-backend`. |
| <a id="decode-first-objective"></a>**decode-first objective** | Rank by steady-state *decode* throughput vs per-token (inter-token) latency at a fixed context length. The SLO gate is decode latency; TTFT is report-only. |
| <a id="accuracy-gate"></a>**accuracy gate** | A config must pass an accuracy eval, such as GSM8K, to be eligible for ranked results. Failing configs are recorded and flagged, never ranked. |

The methodology is specified normatively in [RFC-0001](docs/rfc/RFC-0001.md); this repo
implements that spec.

## Requirements

- **This repo, cloned** (you need `configs/` and `scripts/`, not just the package).
- **GPU-free stages** (validate, generate, analyze): Python with `pydantic` and `pyyaml`.
  `pip install -e .` is the easiest setup because it installs those dependencies and the
  `argsearch-*` console scripts. If the dependencies are already installed, the
  `python3 -m ...` commands work from a repo checkout without an editable install.
- **Live search**: a **GPU host** (devbox) with SGLang installed and the model in cache. The
  accuracy gate also needs `sgl-eval`. Server launch and benchmarking run *on that host*:
  `argsearch-run` spawns `sglang.launch_server` and benches it over localhost.

```bash
git clone https://github.com/zcnrex/sgl-bench && cd sgl-bench
python3 -m pip install -e .          # for the CPU-side stages and tests
```

## Quickstart

Steps 1–3 are GPU-free and run anywhere. Step 4 runs on the GPU host (or drive it from your
laptop with `scripts/devbox_sweep.sh`, which syncs the repo to the devbox and runs it there).

```bash
# 1. Validate the search config
python3 -m sglbench.argsearch.validate configs/nemotron_v3_ultra_nvfp4.yaml

# 2. Optional: preview restart-required configs
#    This prints config hashes/labels only; the live search regenerates configs in step 4.
python3 -m sglbench.argsearch --config configs/nemotron_v3_ultra_nvfp4.yaml --branch b200-fp8kv --mode ofat
python3 -m sglbench.argsearch --config configs/nemotron_v3_ultra_nvfp4.yaml --branch b200-fp8kv --mode grid

# 3. Optional: write generated configs to out/ as <hash>.json + index.jsonl
python3 -m sglbench.argsearch --config configs/nemotron_v3_ultra_nvfp4.yaml --branch b200-fp8kv \
    --mode ofat --save

# 4. From your laptop, sync the repo to the GPU host and start the live search there
DEVBOX=<your-devbox> scripts/devbox_sweep.sh \
    --config configs/nemotron_v3_ultra_nvfp4.yaml --branch b200-fp8kv --mode ofat \
    --isl-osl 8192x256 --concurrency 1 8 32 --frontier
#   ... or, on the GPU host directly:
#   argsearch-run --config configs/nemotron_v3_ultra_nvfp4.yaml --branch b200-fp8kv --mode ofat \
#       --isl-osl 8192x256 --concurrency 1 8 32 --port 8888 --frontier
```

`scripts/devbox_sweep.sh` runs `argsearch-run` **detached** on the devbox (survives SSH drops),
streams `out/results.jsonl` one record per workload point, and fetches results back. Add
`--gsm8k-examples N` to run the accuracy gate per config; `--transport serving` to use
`bench_serving` (percentile ITL) instead of the `bench_one_batch_server` anchor; `--dry-run` to
print the launch/bench commands without running. Full flag reference: [docs/usage/run.md](docs/usage/run.md).

Each measured point is one JSONL record with full provenance:

```json
{"config_hash": "d1ede40fd0b0", "branch": "b200-fp8kv", "config_label": "baseline",
 "label": "isl8192-osl256-c1", "workload": {"isl": 8192, "osl": 256, "concurrency": 1},
 "metrics": {"decode_throughput_tok_s": {"median": 107.0, ...}, "per_token_ms": {"median": 9.35}, ...},
 "branch_keys": {"hardware": "4xB200 single-node TP4", "kv_cache_precision": "fp8_e4m3"},
 "environment": {"hardware": {"accelerator": "NVIDIA B200", "device_count": 4}, "sglang_commit": "0.5.12.post1", ...},
 "accuracy": {"accuracy": 0.975}, "quality_pass": true}
```

`--frontier` (or `argsearch-frontier --config … --results out/results.jsonl`) ranks measured
points within each branch:

```text
slo gate: decode ptok<= 40.0ms  (ttft target 5000ms, report-only)
quality gate: accuracy on gsm8k (<= 0.02 below the branch baseline)

== branch b200-fp8kv ==
records=21  slo_passing=21  quality_excluded=0  eligible=21  frontier=2
 1. d1ede40fd0b0 b200-fp8kv/baseline  isl8192-osl256-c32  decode=1861.7tok/s  ptok=17.2ms  ttft~3000ms
 2. d1ede40fd0b0 b200-fp8kv/baseline  isl8192-osl256-c1   decode=107.0tok/s   ptok=9.3ms   ttft~430ms
```

(Aggregate decode throughput and per-user latency trade off across concurrency, so several
points are non-dominated.)

## Adapting the Config

`configs/nemotron_v3_ultra_nvfp4.yaml` is a **seeded example** for one checkpoint on a 4-GPU host.
The served `model` is the checkpoint and carries the weight precision; a different weight
precision (e.g. FP8) is a different checkpoint and gets its own config file. To search a
different model or hardware target, write your own config: bucket each server arg into `fixed`,
`candidate` (≤10), or constraint rules; declare one or more within-checkpoint `branches`, each
keyed by its hardware target and KV-cache precision; and set `slo` and `quality_gate`.

**The SLO and accuracy-gate tolerance in the example are phase-0 placeholders. Set them to
your real service targets before trusting the results.** See
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

For devbox runs, the `run-argsearch-devbox` skill captures the procedure and host-specific
gotchas: port, paths, and launch discipline.

## Development

```bash
python3 -m unittest discover -s tests
```

The methodology is governed by govctl: RFC-0001 is the normative spec (implemented); RFC-0002
(a performance-improvement opportunity loop) is specified but **not yet implemented**; the
design decisions are in `gov/adr/`. Changes to the methodology go through govctl
(`/discuss` → `/spec` → `/gov`), not direct edits under `gov/`.
