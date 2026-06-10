# sgl-bench usage

Component guides for the `sglbench.argsearch` arg-search pipeline. The pipeline mirrors
the RFC-0001 methodology: define a config → generate restart-required permutations → drive
them through an outer/inner loop → measure each workload point with full provenance.

| Stage | Guide | Module |
| --- | --- | --- |
| 1. Define & validate the search | [config.md](config.md) | `schema.py`, `validate.py` |
| 2. Generate restart-required configs | [generate.md](generate.md) | `generate.py` |
| 3. Drive outer/inner search loop | [driver.md](driver.md) | `driver.py`, `sglang_adapter.py` |
| 4. Measure a workload point | [measure.md](measure.md) | `measure.py` |
| 5. Pareto frontier under the SLO | [objective.md](objective.md) | `objective.py`, `metrics.py` |
| 6. Candidate-vs-baseline deltas | [compare.md](compare.md) | `compare.py`, `metrics.py` |
| Run it live on a server (CLI) | [run.md](run.md) | `run.py`, `sglang_adapter.py` |

The accuracy gate (RFC-0001:C-QUALITY-GATE), the bench transports (`bench_one_batch_server`
anchor vs `bench_serving` percentile-ITL), and running on a RadixArk devbox
(`scripts/devbox_sweep.sh`, `run-argsearch-devbox` skill) are covered in
[run.md](run.md) and [objective.md](objective.md).

The normative methodology lives in [`docs/rfc/RFC-0001.md`](../rfc/RFC-0001.md) (search
methodology) and [`docs/rfc/RFC-0002.md`](../rfc/RFC-0002.md) (opportunity loop); the
decisions behind it are in `gov/adr/` (ADR-0001..0006).
