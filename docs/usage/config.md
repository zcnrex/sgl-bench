# Config — define & validate the search

The single source of truth for what gets searched is a version-controlled YAML config
(RFC-0001:C-CONFIG-SOURCE). `configs/nemotron_v3_ultra.yaml` is the seeded example for
Nemotron-3-Ultra-550B NVFP4.

## Schema reference

Every field is tagged **consumed** (read by the tooling and affects behavior) or
**informational** (recorded for humans/provenance, never read). The schema itself carries
these one-line descriptions via pydantic `Field(description=...)`, so
`SearchConfig.model_json_schema()` is the machine-readable mirror of this table.

### Top level (`SearchConfig`)

| Field | Type | Required | Consumed? | Meaning |
| --- | --- | --- | --- | --- |
| `model` | str | yes | consumed | Default served model; the `<model>` output slug when a branch sets no `checkpoint`. |
| `workload_axes` | map | no | consumed | Inner-loop axes (`isl_osl_pairs`, `report_isl_osl_pairs`, `concurrency`); never permuted into restart-required configs. |
| `slo` | `SLO` | no | consumed | Decode-first objective SLO; required to build the frontier. |
| `quality_gate` | `QualityGate` | no | consumed | Optional accuracy acceptance gate. |
| `precision_branches` | list[`PrecisionBranch`] | yes (≥1) | consumed | One or more precision branches to search. |

### `PrecisionBranch`

| Field | Type | Required | Consumed? | Meaning |
| --- | --- | --- | --- | --- |
| `name` | str | yes | consumed | Branch identifier, e.g. `nvfp4`. |
| `checkpoint` | str | no | consumed | Served model path for this branch; the default served model and the `<model>` output-dir slug. |
| `hardware` | str | no | **informational** | Recorded for humans/provenance only; **never consumed** by the tooling. |
| `fixed` | map | no | consumed | Known-best args held constant for every config in the branch. |
| `candidate` | list[`CandidateArg`] | no | consumed | Args to vary (≤ 10). |
| `constraints` | list[`Constraint`] | no | consumed | Illegal-combination rules filtering generated configs. |
| `baseline` | map | no | consumed | One starting value per candidate; the OFAT reference point. Must itself be constraint-valid. |
| `focused_grid` | `FocusedGrid` | no | consumed | Optional second-stage joint-grid spec. |

Precision (e.g. `quantization`) is a top-level **branch**, never a candidate axis
(RFC-0001:C-PRECISION-BRANCH) — each branch may have its own checkpoint and baseline.

### `CandidateArg`

| Field | Type | Required | Consumed? | Meaning |
| --- | --- | --- | --- | --- |
| `name` | str | yes | consumed | Server-arg name to vary (without the leading `--`). |
| `values` | list | yes (≥1) | consumed | Distinct values to sweep; must include the baseline value. |
| `accuracy_invariant` | bool | no (default `false`) | consumed | `true` declares the arg changes performance but not outputs, letting its per-config accuracy gate be skipped. |

### `Constraint`

| Field | Type | Required | Consumed? | Meaning |
| --- | --- | --- | --- | --- |
| `name` | str | yes | consumed | Rule identifier for error messages. |
| `description` | str | no | **informational** | Human-readable note on the rule. |
| `when` | map[str, list] | no | consumed | Arg→allowed-values guard; the rule applies only when every clause matches. |
| `forbid` | map[str, list] | no | consumed | Arg→values that are illegal when `when` matches. |
| `require` | map[str, list] | no | consumed | Arg→values that must be present when `when` matches. |

A constraint must set `forbid` or `require` (else it has no effect).

### `SLO` / `SLOOverride`

| Field | Type | Required | Consumed? | Meaning |
| --- | --- | --- | --- | --- |
| `slo.per_token_ms` | float > 0 | yes | consumed | Global decode-ITL gate (ms) — the SLO that filters the frontier. |
| `slo.ttft_p95_ms` | float > 0 | no | consumed (report-only) | Global TTFT p95 target (ms); displayed but never gates. |
| `slo.overrides` | list[`SLOOverride`] | no | consumed | Per-`(isl, osl)` overrides. |
| `override.isl` / `override.osl` | int | yes | consumed | The workload pair the override applies to. |
| `override.per_token_ms` | float > 0 | no | consumed | Per-pair decode-ITL gate; inherits the global value when unset. |
| `override.ttft_p95_ms` | float > 0 | no | consumed (report-only) | Per-pair TTFT target; inherits the global value when unset. |

### `QualityGate`

| Field | Type | Required | Consumed? | Meaning |
| --- | --- | --- | --- | --- |
| `dataset` | str | yes | consumed | Evaluation dataset name passed to the accuracy harness. |
| `metric` | str | no (default `accuracy`) | consumed | Score key compared against `threshold`. |
| `threshold` | float | yes | consumed | Pass/fail bound for the metric. |
| `direction` | `higher`\|`lower` | no (default `higher`) | consumed | `higher`: `score ≥ threshold` passes; `lower`: `score ≤ threshold`. |

### `FocusedGrid`

| Field | Type | Required | Consumed? | Meaning |
| --- | --- | --- | --- | --- |
| `args` | list | yes (≥1) | consumed | Candidate names swept jointly in the focused grid. |
| `rationale` | str | yes | **informational** | Why these args are admitted and plausibly interact (recorded in the grid manifest). |
| `pins` | map | no | consumed | OFAT-best value for each non-gridded candidate, held fixed during the grid. |

```yaml
model: nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4
workload_axes:
  isl_osl_pairs: [[8192, 1024], [60000, 20], [64000, 1024]]   # iter: drives the search
  report_isl_osl_pairs: [[8192, 65536]]                        # report: final chars only
  concurrency: [1, 8, 32, 128, 256, 512]
slo:                          # decode-first objective; gate = per-token ITL (RFC-0001:C-OBJECTIVE)
  per_token_ms: 40            # the GATE (decode ITL)
  ttft_p95_ms: 5000           # report-only target; never excludes a config
  overrides:                  # per-(isl,osl): decode ITL grows with context length
    - {isl: 60000, osl: 20, per_token_ms: 80}
quality_gate:                 # accuracy acceptance gate (RFC-0001:C-QUALITY-GATE)
  dataset: gsm8k              # fast first-layer gate; gpqa_diamond (~2h) is final-only
  metric: accuracy
  threshold: 0.95             # score >= threshold passes (direction: higher | lower)
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

## Workload cost-role staging & SLO

`workload_axes` carries two pair sets (RFC-0001:C-WORKLOAD-STAGING). **`isl_osl_pairs`**
are the *iter* role — short-output, long-context workloads that drive the arg search; their
tail decode window gives steady-state decode throughput at that context length cheaply.
**`report_isl_osl_pairs`** are the *report* role — long-output workloads that are NOT run
across the candidate set and are reserved for characterizing the chosen config(s) at the
end. `workload_points(axes, role="iter"|"report")` selects the set.

The `slo` block defines the **decode-first** objective (RFC-0001:C-OBJECTIVE): the gate is
`per_token_ms` (decode inter-token latency); `ttft_p95_ms` is a report-only target that
never excludes a config. Because decode ITL grows with context length, long-context iter
pairs typically need a relaxed `per_token_ms` override. See [objective.md](objective.md).

The optional `quality_gate` block defines the **accuracy acceptance gate**
(RFC-0001:C-QUALITY-GATE): `dataset` + `threshold` (+ `metric`, `direction`), defined before
the search and evaluated per branch. A config below the bar is excluded from the acceptable
results at every stage but still recorded and flagged (`quality_pass`) for trade-off
inspection — see [objective.md](objective.md).

Each candidate may set **`accuracy_invariant: true`** to declare that its values change
performance but not the model's outputs (scheduling / memory-split / parallelism levers);
the default `false` is the fail-safe (accuracy-active). A search that varies only
accuracy-invariant candidates skips the per-config gate — `argsearch-run` evaluates the
gate on the baseline + a spot-check config and reuses that accuracy for the rest, instead
of re-running gsm8k on every numerically-identical permutation (RFC-0001:C-QUALITY-GATE).

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

## Focused grid

The second search stage (RFC-0001:C-SEARCH-STRATEGY) is a joint grid over only the
candidates that both showed OFAT sensitivity and plausibly interact. That selection is
declared per branch so it is recorded and inspectable, rather than passed ad-hoc on the
command line:

```yaml
focused_grid:
  args: [ep-size, moe-a2a-backend, dp-size, enable-dp-attention]   # swept jointly
  rationale: >-
    Why these args are admitted (e.g. coupled pairs OFAT cannot toggle individually).
  pins:                          # OFAT-best for the surviving, non-gridded candidates
    attention-backend: trtllm_mha
    mamba-backend: flashinfer
```

Validation requires a non-empty `rationale`, that every `args` entry is a candidate, and
that **every non-gridded candidate appears in `pins`** with an allowed value — so the grid
pins survivors to their OFAT-best rather than silently falling back to baseline.
`--mode grid` reads this block; see [generate.md](generate.md).
