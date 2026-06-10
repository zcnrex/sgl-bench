# Objective — decode-first Pareto frontier under an SLO

`objective.py` turns a run's `results.jsonl` into the ranked
**decode-throughput-versus-decode-latency Pareto frontier under an SLO** — the optimization
objective of RFC-0001:C-OBJECTIVE (decisions in ADR-0004 / ADR-0005). It consumes one run
directory's `results.jsonl` (a single transport + environment, per ADR-0007:
`out/<model>/runs/<transport>/<env>/results.jsonl`) or globs across them. Decode is the
first-class citizen:

- The **SLO gate is decode per-token latency (ITL)**. TTFT is captured and displayed but is
  **report-only** — it never excludes a config.
- The **throughput axis is steady-state decode throughput** (the tail decode rate at the run's
  longest context), not an OSL-averaged number, and every point is tagged with its context
  length `L = isl + osl`.
- Configurations are **never ranked by raw throughput alone**: a point appears only if it is
  both decode-SLO-passing *and* non-dominated.

## The SLO lives in the config

Defined **before the search begins**, in the versioned config (ADR-0002):

```yaml
slo:
  per_token_ms: 40            # GATE: decode inter-token latency
  ttft_p95_ms: 5000           # report-only target; never gates
  overrides:                  # decode ITL grows with context length
    - {isl: 60000, osl: 20, per_token_ms: 80}
```

A workload pair with no override inherits the global gate. `ttft_p95_ms` is optional; when
omitted, TTFT is still reported per point but no target is shown.

## The accuracy gate (RFC-0001:C-QUALITY-GATE)

An optional `quality_gate`, also defined before the search, excludes quality-degraded configs
from the **acceptable results** (the frontier) — at every stage, never bought back by speed:

```yaml
quality_gate:
  dataset: gsm8k           # fast first-layer gate; reserve gpqa_diamond (~2h) for final validation
  metric: accuracy
  tolerance: 0.02          # pass if no more than 0.02 below the branch baseline
  direction: higher        # degradation = baseline - score; use "lower" for e.g. perplexity
```

Accuracy is evaluated once per launched config (the driver's `evaluate` callback) and stamped
onto every record as `accuracy` + a `quality_pass` flag. A config that fails the gate is
**dropped from the frontier but kept in `results.jsonl`**, flagged, and listed under
`quality-excluded` so the speed-versus-quality trade-off stays inspectable. If a branch has no
eligible config, it is reported as having **no acceptable configuration** rather than emitting a
degraded one. When no `quality_gate` is defined, the filter is skipped (with a notice).

### Inspecting gate-failing configs (`--inspect`)

To read the perf of gate-failing or accuracy-unmeasured configs (e.g. a perf-only OFAT sweep
where `accuracy` is null), run with `--inspect` (ADR-0008). It ranks **every** SLO-passing
config under the same Pareto semantics, ignoring the gate, and annotates each row with its true
gate status (`PASS` / `FAIL` / `n-a` + the metric value):

```
!! INSPECTION VIEW — includes gate-FAILING / unmeasured configs; NOT acceptable results (RFC-0001:C-QUALITY-GATE)
records=21  slo_passing=21  eligible(ignoring gate)=21  frontier=3
 1. ce798 nvfp4/isl8192-osl256-c32 ...  decode=1868.3tok/s  ptok=17.1ms  ttft~13121ms  gate=n-a(accuracy=n/a)
```

`--inspect` is a **read-only view** and never produces acceptable results: it refuses `--out`
so a gate-failing config can never enter the canonical frontier artifact (C-QUALITY-GATE). To
persist the inspection view, redirect stdout yourself.

## Run it

```bash
argsearch-frontier --config configs/nemotron_v3_ultra_nvfp4.yaml \
    --results out/<model>/runs/bench_one_batch_server/<env>/results.jsonl
# or: python -m sglbench.argsearch.objective --config ... --results ...
```

```
slo gate: decode ptok<= 40.0ms  (ttft target 5000.0ms, report-only)  (+1 per-pair override(s))
records=3  slo_passing=2  frontier=1
 1. aaa nvfp4/baseline  isl8192-osl1024-c32 L=9216  decode=1600.0tok/s  ptok=20.0ms  ttft~3000ms
```

A config with a terrible TTFT but acceptable decode ITL **survives**; a config with the
highest raw throughput but a violated decode ITL is **dropped**. Flags: `--branch`,
`--stat {median,mean}`, `--out`, `--no-save`, `--inspect` (read-only view of gate-failing
configs; see above).

## How it works

1. **Decode SLO filter.** For each record, the decode per-token latency (ITL) for the record's
   `(isl, osl)` must be within the gate. TTFT is ignored for filtering. A record missing the
   decode latency is excluded.
2. **Accuracy gate filter.** When a `quality_gate` is defined, a record must also clear it
   (trusting the stamped `quality_pass`, else recomputed from `accuracy`; a record with no
   accuracy score fails). Gate-failing records are excluded from the frontier but retained,
   flagged, in the record stream.
3. **Pareto frontier.** Over the survivors, a point dominates another when it is no worse on
   both axes (decode throughput up, decode ITL down) and strictly better on at least one. The
   frontier is the non-dominated set, ranked by decode throughput. Because batch size /
   concurrency varies across points, decode throughput (aggregate) and ITL (per-user) form a
   genuine trade-off — higher concurrency raises system throughput but slows each user.

The metric vocabulary is shared with the bench transport via `metrics.py`. Throughput selection
prefers `decode_throughput_tok_s` (the tail steady-state rate, from bench `last_gen_throughput`)
over the OSL-averaged `output_throughput`. TTFT/per-token selection prefer percentile keys
(`ttft_p95_ms`, `per_token_p95_ms`) and fall back to single-sample keys.

> **Measurement caveats (reconcile on a live server, RFC-0001:C-BASELINE-ANCHOR).**
> `bench_one_batch_server` exposes the single tail decode step (`last_gen_throughput`), not a
> multi-step window — the C-OBJECTIVE window requirement is recorded as `decode_window_steps`
> and the ≥2 repeats provide cross-run smoothing; a true window needs a richer transport.
> `argsearch-run --transport serving` provides one: `bench_serving` drives many requests at a
> target concurrency and reports ITL percentiles, from which the decode rate is derived
> (`concurrency / median_itl`) — not its blended `output_throughput`. It reconciles against the
> `bench_one_batch_server` anchor (decode within ~0.1%, TTFT within ~8%) at fixed input length
> (`range_ratio=1.0`). Likewise the anchor's `last_ttft` is a single sample, not a p95, so the
> (report-only) TTFT is approximate. For short-output *iter* workloads at long prefill, every
> decode step sits at ~constant `L`, so the tail rate is a clean steady-state estimate;
> long-output *report* workloads cross many context lengths and only their tail reflects decode
> at the longest `L`.

## Library

```python
from sglbench.argsearch import load_config, load_results, build_frontier

cfg = load_config("configs/nemotron_v3_ultra_nvfp4.yaml")
records = load_results("out/<model>/runs/bench_one_batch_server/<env>/results.jsonl")
passing, frontier = build_frontier(records, cfg.slo, branch="nvfp4", stat="median",
                                   gate=cfg.quality_gate)
```
