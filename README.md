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

# 2. Generate configs — OFAT (sensitivity) first, then a focused grid
python -m sglbench.argsearch --config configs/nemotron_v3_ultra.yaml --branch nvfp4 --mode ofat
python -m sglbench.argsearch --config configs/nemotron_v3_ultra.yaml --branch nvfp4 \
    --mode grid --grid-args ep-size moe-a2a-backend

# 3. Save configs to out/ as <config_hash>.json + index.jsonl
python -m sglbench.argsearch --config configs/nemotron_v3_ultra.yaml --branch nvfp4 \
    --mode ofat --save
```

Driving the search (outer relaunch per config, inner workload sweep) and measuring each
point:

```python
from sglbench.argsearch import (
    load_config, generate_ofat, workload_points, run_search, write_results,
)

cfg = load_config("configs/nemotron_v3_ultra.yaml")
branch = cfg.branch("nvfp4")
results = run_search(generate_ofat(branch), workload_points(cfg.workload_axes), manager)
write_results(results, "out")   # -> out/results.jsonl
```

`manager` implements the server-lifecycle contract (launch / client / shutdown). The
protocols + loop orchestration ship today; the concrete SGLang launcher + benchmark
transport is the next slice — see the driver guide for the adapter contract and a sketch.

## Documentation

Per-component guides live in [`docs/usage/`](docs/usage/README.md):

| Stage | Guide |
| --- | --- |
| Define & validate the search | [docs/usage/config.md](docs/usage/config.md) |
| Generate restart-required configs | [docs/usage/generate.md](docs/usage/generate.md) |
| Drive the outer/inner search loop | [docs/usage/driver.md](docs/usage/driver.md) |
| Measure a workload point | [docs/usage/measure.md](docs/usage/measure.md) |

## Develop

```bash
python -m unittest discover -s tests
```

Changes to the methodology go through govctl (`/discuss` → `/spec` → `/gov`), not direct
edits to `gov/`.
