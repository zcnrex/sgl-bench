# Config — define & validate the search

The single source of truth for what gets searched is a version-controlled YAML config
(RFC-0001:C-CONFIG-SOURCE). `configs/nemotron_v3_ultra.yaml` is the seeded example for
Nemotron-3-Ultra-550B NVFP4.

## Structure

A config has a `model`, descriptive `workload_axes` (the inner loop — *not* permuted into
restart-required configs), and one or more `precision_branches`. Within a branch:

| Bucket | Meaning |
| --- | --- |
| `fixed` | known-best args, never varied |
| `candidate` | args to vary (≤ 10); each has a `values` list |
| `constraints` | illegal-combination rules (`when` → `forbid` / `require`) |
| `baseline` | one starting value per candidate; must itself be constraint-valid |

Precision (e.g. `quantization`) is a top-level **branch**, never a candidate axis
(RFC-0001:C-PRECISION-BRANCH) — each branch may have its own checkpoint and baseline.

```yaml
model: nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4
workload_axes:
  isl_osl_pairs: [[8192, 1024], [60000, 20], [8192, 65536]]
  concurrency: [1, 8, 32, 128, 256, 512]
precision_branches:
  - name: nvfp4
    checkpoint: nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4
    fixed:
      quantization: modelopt_fp4
      tensor-parallel-size: 4
    candidate:
      - name: attention-backend
        values: [trtllm_mha, flashinfer]
    baseline:
      attention-backend: trtllm_mha
```

## Validate

```bash
python -m sglbench.argsearch.validate configs/nemotron_v3_ultra.yaml
# or, after install:  argsearch-validate configs/nemotron_v3_ultra.yaml
```

```
OK configs/nemotron_v3_ultra.yaml
model: nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4
  branch nvfp4: 9 candidates, 4 constraints, 7 OFAT configs
```

Exits non-zero with the error on an invalid config. Validation enforces:

- every candidate in exactly one value-bucket;
- ≤ 10 candidates (RFC-0001:C-SCOPE);
- speculative-decode args rejected (out of scope, RFC-0001:C-SCOPE);
- precision-defining args (`quantization`) fixed, not candidate;
- baseline covers every candidate with an allowed value;
- baseline keys are all candidates (a non-candidate key is an error — put constants in `fixed`);
- constraint-referenced args are declared in `fixed` or `candidate`.

## Coupled pairs

Some levers must change together. `ep-size`/`moe-a2a-backend` (EP needs the deepep
all-to-all) and `dp-size`/`enable-dp-attention` (data-parallel attention needs its flag)
are modeled as constraint pairs. OFAT cannot toggle these on with a single change, so the
generator prunes the lone-change attempts — they appear only in a focused grid (see
[generate.md](generate.md)).

A boolean candidate (`enable-dp-attention: [false, true]`) renders as a bare flag:
`--enable-dp-attention` is emitted only when `true`, omitted when `false`.
