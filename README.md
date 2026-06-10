# sgl-bench

Systematic SGLang server-arg search and benchmarking.

This repo is governed by govctl. The methodology is normative:

- **RFC-0001** — Large-Model SGLang Server-Arg Search Methodology
- **RFC-0002** — Performance-Improvement Opportunity Loop

See `docs/rfc/` for the rendered specs and `gov/adr/` for the decisions (ADR-0001..0006).

## Install

```bash
pip install git+https://github.com/zcnrex/sgl-bench
```

Or run from the repo root without installing: `python -m sglbench.argsearch ...`.

## Quickstart

The pipeline goes: **define a config → generate restart-required permutations → drive them
through an outer/inner loop → measure each workload point with provenance.**

```bash
# 1. Validate the search config
python -m sglbench.argsearch.validate configs/nemotron_v3_ultra.yaml

# 2. Generate configs — OFAT (sensitivity) first, then the focused grid
python -m sglbench.argsearch --config configs/nemotron_v3_ultra.yaml --branch nvfp4 --mode ofat
python -m sglbench.argsearch --config configs/nemotron_v3_ultra.yaml --branch nvfp4 --mode grid

# 3. Save configs to out/ as <config_hash>.json + index.jsonl
python -m sglbench.argsearch --config configs/nemotron_v3_ultra.yaml --branch nvfp4 \
    --mode ofat --save

# 4. Run the live search on a server (launch -> bench -> frontier)
argsearch-run --config configs/nemotron_v3_ultra.yaml --branch nvfp4 --mode ofat \
    --isl-osl 8192x256 --concurrency 1 8 32 --port 40000 --frontier
```

The focused grid's admitted args + rationale are declared in the config's `focused_grid`
block (not on the CLI). `argsearch-run` drives the whole outer/inner loop against a live
SGLang server — concrete launcher (`SGLangServerManager`) and bench transports
(`bench_one_batch_server` anchor, `bench_serving` percentile-ITL) are implemented and
hardware-validated; it streams `results.jsonl` per point and optionally runs the gsm8k
accuracy gate (`--gsm8k-examples`). See [docs/usage/run.md](docs/usage/run.md). For library
use, `run_search(points, workload, manager)` is the same loop. To run on a RadixArk devbox,
use `scripts/devbox_sweep.sh` (detached, resilient).

## Documentation

Per-component guides live in [`docs/usage/`](docs/usage/README.md):

| Stage | Guide |
| --- | --- |
| Define & validate the search | [docs/usage/config.md](docs/usage/config.md) |
| Generate restart-required configs | [docs/usage/generate.md](docs/usage/generate.md) |
| Drive the outer/inner search loop | [docs/usage/driver.md](docs/usage/driver.md) |
| Measure a workload point | [docs/usage/measure.md](docs/usage/measure.md) |
| Pareto frontier under the SLO | [docs/usage/objective.md](docs/usage/objective.md) |
| Run the live search (CLI) | [docs/usage/run.md](docs/usage/run.md) |

## Develop

```bash
python -m unittest discover -s tests
```

Changes to the methodology go through govctl (`/discuss` → `/spec` → `/gov`), not direct
edits to `gov/`.
