# sgl-bench

Systematic SGLang server-arg search and benchmarking.

This repo is governed by govctl. The methodology is normative:

- **RFC-0001** — Large-Model SGLang Server-Arg Search Methodology
- **RFC-0002** — Performance-Improvement Opportunity Loop

See `docs/rfc/` for the rendered specs and `gov/adr/` for the decisions (ADR-0001..0006).

## What's here (Phase 1)

`sglbench/argsearch/` — the versioned config layer and permutation generator that the
outer driver loop will consume. It defines the source of truth for *restart-required*
server configurations and emits only constraint-valid permutations.

- `schema.py` — config model + validation (fixed / candidate / constraint buckets;
  precision as a top-level branch).
- `generate.py` — OFAT and focused-grid generators, constraint-pruned; CLI + content hash.
- `validate.py` — one-command config validation.
- `configs/nemotron_v3_ultra.yaml` — the Nemotron-3-Ultra-550B NVFP4 search config.

## Install

```bash
pip install git+https://github.com/zcnrex/sgl-bench
```

Or just run from the repo root (no install needed): `python -m sglbench.argsearch ...`.

## Usage

### Validate a config

```bash
python -m sglbench.argsearch.validate configs/nemotron_v3_ultra.yaml
# or, after install:  argsearch-validate configs/nemotron_v3_ultra.yaml
```

```
OK configs/nemotron_v3_ultra.yaml
model: nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4
  branch nvfp4: 9 candidates, 4 constraints, 7 OFAT configs
```

Exits non-zero with the error on an invalid config (e.g. a baseline key that isn't a
candidate, a constraint referencing an undeclared arg, >10 candidates, or a spec-decode
arg).

### Generate configs

Two modes, per RFC-0001:C-SEARCH-STRATEGY — start with `ofat` (sensitivity around the
baseline), then run a focused `grid` over the few args that survive and plausibly interact.

```bash
# OFAT: vary one candidate at a time around the baseline
python -m sglbench.argsearch --config configs/nemotron_v3_ultra.yaml --branch nvfp4 --mode ofat

# Focused grid over a named subset of candidates
python -m sglbench.argsearch --config configs/nemotron_v3_ultra.yaml --branch nvfp4 \
    --mode grid --grid-args ep-size moe-a2a-backend
```

`--format` controls stdout output:

- `label` (default) — a compact, human-scannable index, one config per tab-separated line:
  `<config_hash>  <branch>/<mode>  <label>`. The hash is `sha256[:12]` of the server args
  (the config's provenance identity); the label is what differs from the baseline. This is
  an overview, not something you launch.
- `cli` — an SGLang launch arg string per config.
- `json` — one full config object per line, for the driver.

```
# label
d1ede40fd0b0	nvfp4/ofat	baseline
64fd063646d1	nvfp4/ofat	attention-backend=flashinfer
ce798d170e76	nvfp4/ofat	mamba-scheduler-strategy=no_buffer
...

# cli (baseline)
--attention-backend trtllm_mha --chunked-prefill-size 8192 --context-length 131072 \
--cuda-graph-max-bs 256 --dp-size 1 --ep-size 1 --kv-cache-dtype fp8_e4m3 \
--mamba-backend flashinfer --mamba-full-memory-ratio 0.9 \
--mamba-scheduler-strategy extra_buffer_lazy --mamba-ssm-dtype bfloat16 \
--mem-fraction-static 0.95 --quantization modelopt_fp4 --tensor-parallel-size 4 \
--trust-remote-code
```

Pass `--save` to write per-config files to `out/` instead of stdout — the directory is
created automatically (no `mkdir` needed), one `<config_hash>.json` per config plus an
`index.jsonl` manifest. The hash-named files are the config's provenance identity
(RFC-0001:C-MEASUREMENT):

```bash
python -m sglbench.argsearch --config configs/nemotron_v3_ultra.yaml --branch nvfp4 \
    --mode ofat --save
# wrote 7 configs to out/ (index: out/index.jsonl)
```

`out/` is gitignored. Pipe `--format json` into the (future) driver, or `--format cli` to
launch by hand:

```bash
python -m sglbench.argsearch --config configs/nemotron_v3_ultra.yaml --branch nvfp4 \
    --mode ofat --format cli | while read -r line; do
  case "$line" in \#*) echo "$line"; continue;; esac
  python -m sglang.launch_server --model-path "$MODEL" $line --port 8000
  # ... warmup + bench + teardown (driver loop, RFC-0001:C-LOOP-STRUCTURE) ...
done
```

## Config structure

A config has a `model`, descriptive `workload_axes` (the inner loop — *not* permuted), and
one or more `precision_branches`. Within a branch:

| Bucket | Meaning |
| --- | --- |
| `fixed` | known-best args, never varied |
| `candidate` | args to vary (<= 10); each has a `values` list |
| `constraints` | illegal-combination rules (`when` -> `forbid` / `require`) |
| `baseline` | one starting value per candidate; must itself be constraint-valid |

Validation enforces: every candidate in exactly one value-bucket, <= 10 candidates,
speculative-decode args rejected, precision-defining args (`quantization`) fixed not
candidate, baseline covers every candidate with an allowed value, baseline keys are all
candidates, and constraint-referenced args are declared.

### Coupled pairs

Some levers must change together. `ep-size`/`moe-a2a-backend` (EP needs the deepep
all-to-all) and `dp-size`/`enable-dp-attention` (data-parallel attention needs its flag)
are modeled as constraint pairs. OFAT cannot toggle these on with a single change, so the
generator prunes the lone-change attempts — they appear only in a focused grid:

```
python -m sglbench.argsearch --config configs/nemotron_v3_ultra.yaml --branch nvfp4 \
    --mode grid --grid-args ep-size moe-a2a-backend dp-size enable-dp-attention --format label
# 16 raw combos -> 4 valid (each pair on-on or off-off):
#   ep1/none   + dp1/off
#   ep1/none   + dp4/on
#   ep4/deepep + dp1/off
#   ep4/deepep + dp4/on
```

A boolean candidate (e.g. `enable-dp-attention: [false, true]`) renders as a bare flag:
`--enable-dp-attention` is emitted only when the value is `true`, omitted when `false`.

## Develop

```bash
python -m unittest discover -s tests
```

Changes to the methodology go through govctl (`/discuss` -> `/spec` -> `/gov`), not direct
edits to `gov/`.
