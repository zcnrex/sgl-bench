# Compare — candidate-vs-baseline deltas per concurrency

`compare.py` answers a different question than the Pareto frontier. The frontier
([objective.md](objective.md)) tells you *which configs are non-dominated*; `compare.py` tells
you *how far each single-arg OFAT change moves decode throughput, latency, and accuracy relative
to the baseline reference point* — at every concurrency, side by side.

It is the natural read for an **OFAT sweep**: each row is one candidate arg changed from the
baseline, so the Δ column isolates that arg's effect. A candidate that barely beats baseline (or
loses to it) is visible here even when it never reaches the frontier.

## The baseline is resolved from the config

The baseline is the OFAT origin — `generate_ofat(branch)[0]`, the config with every candidate at
its `baseline:` value — so it is read from the versioned config, never hard-coded. Every other
OFAT config is a single-arg change away from it, which is exactly what makes the per-row Δ
meaningful.

## Run it

```bash
argsearch-compare --config configs/nemotron_v3_ultra_nvfp4.yaml --branch b200-fp8kv \
    --results out/<model>/runs/bench_one_batch_server/<env>/results.jsonl
# or: python -m sglbench.argsearch.compare --config ... --branch ... --results ...
```

```
baseline = baseline (4a5953328de8)
branch = nvfp4   stat = median   metric = accuracy

# concurrency = 32
config                                decode tok/s Δ vs base  ptok ms   ttft ms accuracy
baseline                                    1775.2  baseline    18.03   13464.1     1.0
attention-backend=flashinfer                1723.9     -2.9%    18.56   14378.8     1.0
mamba-backend=triton                        1611.1     -9.2%    19.86   13577.1     1.0
mamba-scheduler-strategy=no_buffer          1779.2     +0.2%    17.99   13228.8     1.0
mamba-ssm-dtype=bfloat16                    1861.6     +4.9%    17.19   13449.5   0.975
mamba-full-memory-ratio=0.7                 1776.3     +0.1%    18.02   13461.9     1.0
```

One block per concurrency present in the results. Within a block, configs are listed in OFAT
enumeration order (baseline first). The `Δ vs base` column is the decode-throughput change vs
the baseline at that same concurrency; the baseline row is marked `baseline`. The `accuracy`
column shows the gate metric (from `quality_gate.metric`, default `accuracy`) so a
speed-versus-quality trade-off — e.g. `mamba-ssm-dtype=bfloat16` buying +4.9% decode for a
1.0 → 0.975 accuracy drop — is legible at a glance.

Flags: `--branch` (required), `--stat {median,mean}`, `--results`. A candidate with no record at
a given concurrency is skipped for that block; a missing/null metric prints `n/a`.

## How it differs from `argsearch-frontier`

| | `argsearch-frontier` | `argsearch-compare` |
| --- | --- | --- |
| Question | Which configs are Pareto-optimal under the SLO? | How much does each arg move the needle vs baseline? |
| Rows shown | The non-dominated frontier only | Every OFAT config, grouped by concurrency |
| Reference | None (absolute throughput/ITL) | The baseline config, per concurrency |
| Gate | Filters the acceptable results | Reported per row, never filters |

`compare.py` does **not** apply the SLO or accuracy gate — it is a descriptive read of the
measured results, not the acceptable-results artifact. Use `argsearch-frontier` for the
governed frontier (RFC-0001:C-OBJECTIVE / C-QUALITY-GATE) and `argsearch-compare` to understand
*why* a config did or didn't make it.

## Library

```python
from sglbench.argsearch import load_config, load_results
from sglbench.argsearch.generate import generate_ofat

cfg = load_config("configs/nemotron_v3_ultra_nvfp4.yaml")
branch = cfg.branch("nvfp4")
baseline_hash = generate_ofat(branch)[0].config_hash   # the comparison reference
records = load_results("out/<model>/runs/bench_one_batch_server/<env>/results.jsonl")
```
