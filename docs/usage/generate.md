# Generate — restart-required config permutations

The generator emits only constraint-valid *restart-required* server configs
(RFC-0001:C-CONFIG-SOURCE). Workload axes are the inner loop and are **not** expanded here
(see [driver.md](driver.md)).

Two modes implement the staged search of RFC-0001:C-SEARCH-STRATEGY — start with `ofat`
(sensitivity around the baseline), then run a focused `grid` over the few args that survive
and plausibly interact.

```bash
# OFAT: vary one candidate at a time around the baseline
python -m sglbench.argsearch --config configs/nemotron_v3_ultra.yaml --branch nvfp4 --mode ofat

# Focused grid over a named subset of candidates
python -m sglbench.argsearch --config configs/nemotron_v3_ultra.yaml --branch nvfp4 \
    --mode grid --grid-args ep-size moe-a2a-backend
```

## Output formats

`--format` controls stdout:

- `label` (default) — compact, human-scannable index, one config per tab-separated line:
  `<config_hash>  <branch>/<mode>  <label>`. The hash is `sha256[:12]` of the server args
  (the config's provenance identity); the label is what differs from the baseline. An
  overview, not something you launch.
- `cli` — an SGLang launch arg string per config.
- `json` — one full config object per line, for the driver.

```
# label
d1ede40fd0b0	nvfp4/ofat	baseline
64fd063646d1	nvfp4/ofat	attention-backend=flashinfer
ce798d170e76	nvfp4/ofat	mamba-scheduler-strategy=no_buffer

# cli (baseline)
--attention-backend trtllm_mha --chunked-prefill-size 8192 --context-length 131072 \
--cuda-graph-max-bs 256 --dp-size 1 --ep-size 1 --kv-cache-dtype fp8_e4m3 \
--mamba-backend flashinfer --mamba-full-memory-ratio 0.9 \
--mamba-scheduler-strategy extra_buffer_lazy --mamba-ssm-dtype bfloat16 \
--mem-fraction-static 0.95 --quantization modelopt_fp4 --tensor-parallel-size 4 \
--trust-remote-code
```

## Save to disk

Pass `--save` to write per-config files to `out/` instead of stdout — the directory is
created automatically, one `<config_hash>.json` per config plus an `index.jsonl` manifest.
The hash-named files are the config's provenance identity (RFC-0001:C-MEASUREMENT). `out/`
is gitignored.

```bash
python -m sglbench.argsearch --config configs/nemotron_v3_ultra.yaml --branch nvfp4 \
    --mode ofat --save
# wrote 7 configs to out/ (index: out/index.jsonl)
```

## Coupled-pair example

`ep-size`/`moe-a2a-backend` and `dp-size`/`enable-dp-attention` only appear in a grid:

```
python -m sglbench.argsearch --config configs/nemotron_v3_ultra.yaml --branch nvfp4 \
    --mode grid --grid-args ep-size moe-a2a-backend dp-size enable-dp-attention --format label
# 16 raw combos -> 4 valid (each pair on-on or off-off):
#   ep1/none   + dp1/off
#   ep1/none   + dp4/on
#   ep4/deepep + dp1/off
#   ep4/deepep + dp4/on
```

## As a library

```python
from sglbench.argsearch import load_config, generate_ofat, generate_grid

cfg = load_config("configs/nemotron_v3_ultra.yaml")
branch = cfg.branch("nvfp4")
points = generate_ofat(branch)                       # list[ConfigPoint]
grid = generate_grid(branch, ["ep-size", "moe-a2a-backend"])
```

Each `ConfigPoint` carries `branch`, `mode`, `label`, `args` (dict), `checkpoint`, and
`config_hash`. These feed `run_search` — see [driver.md](driver.md).
