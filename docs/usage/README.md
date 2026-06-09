# sgl-bench usage

Component guides for the `sglbench.argsearch` arg-search pipeline. The pipeline mirrors
the RFC-0001 methodology: define a config → generate restart-required permutations → drive
them through an outer/inner loop → measure each workload point with full provenance.

| Stage | Guide | Module |
| --- | --- | --- |
| 1. Define & validate the search | [config.md](config.md) | `schema.py`, `validate.py` |
| 2. Generate restart-required configs | [generate.md](generate.md) | `generate.py` |
| 3. Drive outer/inner search loop | [driver.md](driver.md) | `driver.py` |
| 4. Measure a workload point | [measure.md](measure.md) | `measure.py` |

The normative methodology lives in [`docs/rfc/RFC-0001.md`](../rfc/RFC-0001.md) (search
methodology) and [`docs/rfc/RFC-0002.md`](../rfc/RFC-0002.md) (opportunity loop); the
decisions behind it are in `gov/adr/` (ADR-0001..0006).
