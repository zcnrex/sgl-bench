"""Pareto-frontier-under-SLO tests ([[RFC-0001:C-OBJECTIVE]]): decode-first, TTFT non-gating.

The accuracy gate is within-branch and baseline-relative ([[RFC-0001:C-QUALITY-GATE]]): a
config passes only if its metric is no more than `tolerance` below its OWN branch baseline.
"""

import tempfile
import unittest
from pathlib import Path

import io
from contextlib import redirect_stdout

from sglbench.argsearch.objective import (
    FrontierEntry,
    branch_baseline_score,
    build_frontier,
    gate_failed_pins,
    gate_status,
    main,
    pareto_frontier,
    passes_quality_gate,
    passes_slo,
    record_quality_pass,
    to_entry,
    write_frontier,
)
from sglbench.argsearch.generate import _assemble, config_hash
from sglbench.argsearch.schema import SLO, QualityGate, SearchConfig, SLOOverride

GATE = QualityGate(dataset="gpqa", metric="accuracy", tolerance=0.15, direction="higher")
BASE = 0.6  # branch-baseline accuracy used as the gate reference in these tests

SLO_GLOBAL = SLO(per_token_ms=40, ttft_p95_ms=5000)
SLO_OVERRIDE = SLO(
    per_token_ms=40,
    ttft_p95_ms=5000,
    overrides=[SLOOverride(isl=60000, osl=20, per_token_ms=80)],
)


def rec(h, isl, osl, c, ttft_ms, per_token_ms, decode_thr, branch="b200-fp8kv",
        label=None, config_label=""):
    def agg(v):
        return {"median": v, "mean": v, "n": 2}

    return {
        "config_hash": h,
        "branch": branch,
        "label": label if label is not None else h,
        "config_label": config_label,
        "workload": {"isl": isl, "osl": osl, "concurrency": c},
        "metrics": {
            "ttft_ms": agg(ttft_ms),
            "per_token_ms": agg(per_token_ms),
            "decode_throughput_tok_s": agg(decode_thr),
        },
    }


def entry(h, thr, ptok, ttft=100.0):
    return FrontierEntry(h, "b200-fp8kv", h, {"isl": 8192, "osl": 1024, "concurrency": 1},
                         9216, thr, ptok, ttft)


class SLOFilterTest(unittest.TestCase):
    def test_within_decode_gate_passes(self):
        self.assertTrue(passes_slo(rec("a", 8192, 1024, 32, 3000, 30, 1600), SLO_GLOBAL))

    def test_per_token_violation_excluded(self):
        self.assertFalse(passes_slo(rec("a", 8192, 1024, 32, 3000, 99, 1600), SLO_GLOBAL))

    def test_ttft_violation_does_not_exclude(self):
        self.assertTrue(passes_slo(rec("a", 8192, 1024, 32, 99000, 30, 1600), SLO_GLOBAL))

    def test_missing_decode_metric_excluded(self):
        r = rec("a", 8192, 1024, 32, 3000, 30, 1600)
        del r["metrics"]["per_token_ms"]
        self.assertFalse(passes_slo(r, SLO_GLOBAL))

    def test_per_pair_per_token_override_applied(self):
        long_ctx = rec("p", 60000, 20, 8, 3000, 70, 900)
        self.assertFalse(passes_slo(long_ctx, SLO_GLOBAL))
        self.assertTrue(passes_slo(long_ctx, SLO_OVERRIDE))


class EntryTest(unittest.TestCase):
    def test_to_entry_missing_throughput_is_none(self):
        r = rec("a", 8192, 1024, 32, 3000, 30, 1600)
        del r["metrics"]["decode_throughput_tok_s"]
        self.assertIsNone(to_entry(r))

    def test_context_length_tagged(self):
        e = to_entry(rec("a", 60000, 20, 8, 3000, 30, 900))
        self.assertEqual(e.context_length, 60020)

    def test_entry_without_ttft_still_built(self):
        r = rec("a", 8192, 1024, 32, 3000, 30, 1600)
        del r["metrics"]["ttft_ms"]
        e = to_entry(r)
        self.assertIsNotNone(e)
        self.assertIsNone(e.ttft_ms)


class ParetoTest(unittest.TestCase):
    def test_dominated_excluded(self):
        a = entry("a", thr=1600, ptok=20)
        b = entry("b", thr=800, ptok=40)
        front = pareto_frontier([a, b])
        self.assertEqual({e.config_hash for e in front}, {"a"})

    def test_tradeoff_both_kept(self):
        a = entry("a", thr=1600, ptok=40)
        b = entry("b", thr=800, ptok=20)
        front = pareto_frontier([a, b])
        self.assertEqual({e.config_hash for e in front}, {"a", "b"})

    def test_equal_points_both_kept(self):
        a = entry("a", thr=1000, ptok=30)
        b = entry("b", thr=1000, ptok=30)
        self.assertEqual(len(pareto_frontier([a, b])), 2)


class BuildFrontierTest(unittest.TestCase):
    def test_branch_filter_and_ranking(self):
        records = [
            rec("a", 8192, 1024, 32, 3000, 20, 1600),
            rec("b", 8192, 1024, 8, 2000, 30, 800),
            rec("c", 8192, 1024, 512, 4000, 99, 5000),
            rec("d", 8192, 1024, 1, 3000, 25, 1200, branch="b200-bf16kv"),
        ]
        passing, frontier = build_frontier(records, SLO_GLOBAL, branch="b200-fp8kv")
        hashes = {e.config_hash for e in passing}
        self.assertNotIn("c", hashes)
        self.assertNotIn("d", hashes)
        self.assertEqual(frontier[0].config_hash, "a")
        self.assertGreaterEqual(frontier[0].throughput, frontier[-1].throughput)

    def test_high_throughput_decode_violator_excluded(self):
        records = [rec("fast-bad", 8192, 1024, 512, 3000, 99, 9000)]
        _, frontier = build_frontier(records, SLO_GLOBAL)
        self.assertEqual(frontier, [])

    def test_high_ttft_decode_winner_kept(self):
        records = [rec("slow-ttft-fast-decode", 8192, 1024, 32, 99000, 20, 2000)]
        _, frontier = build_frontier(records, SLO_GLOBAL)
        self.assertEqual(len(frontier), 1)

    def test_write_frontier(self):
        _, frontier = build_frontier(
            [rec("a", 8192, 1024, 32, 3000, 20, 1600)], SLO_GLOBAL)
        with tempfile.TemporaryDirectory() as d:
            path = write_frontier(frontier, Path(d) / "frontier.jsonl")
            self.assertEqual(len(Path(path).read_text().splitlines()), 1)


def gated_rec(h, *, ptok=20, decode_thr=1600, accuracy=None, quality_pass=None,
              branch="b200-fp8kv", label=None, config_label=""):
    r = rec(h, 8192, 1024, 32, 3000, ptok, decode_thr, branch=branch, label=label,
            config_label=config_label)
    if accuracy is not None:
        r["accuracy"] = {"accuracy": accuracy}
    if quality_pass is not None:
        r["quality_pass"] = quality_pass
    return r


class QualityGateTest(unittest.TestCase):
    def test_passes_within_tolerance_below_baseline(self):
        self.assertTrue(GATE.passes(BASE, BASE))          # at baseline
        self.assertTrue(GATE.passes(0.9, BASE))           # above baseline
        self.assertTrue(GATE.passes(0.45, BASE))          # exactly tolerance below
        self.assertFalse(GATE.passes(0.44, BASE))         # beyond tolerance
        self.assertFalse(GATE.passes(None, BASE))         # missing score
        self.assertFalse(GATE.passes(0.6, None))          # missing baseline

    def test_passes_lower_direction(self):
        ppl = QualityGate(dataset="d", metric="perplexity", tolerance=1.0, direction="lower")
        self.assertTrue(ppl.passes(9.5, 9.0))             # 0.5 worse, within 1.0
        self.assertFalse(ppl.passes(10.5, 9.0))           # 1.5 worse, beyond 1.0

    def test_no_gate_admits_all(self):
        self.assertTrue(passes_quality_gate(gated_rec("a"), None, None))

    def test_stamped_flag_is_trusted(self):
        self.assertTrue(passes_quality_gate(gated_rec("a", quality_pass=True), GATE, BASE))
        self.assertFalse(passes_quality_gate(gated_rec("a", quality_pass=False), GATE, BASE))

    def test_recompute_relative_to_baseline_when_no_flag(self):
        self.assertTrue(record_quality_pass(gated_rec("a", accuracy=0.5), GATE, BASE))
        self.assertFalse(record_quality_pass(gated_rec("a", accuracy=0.3), GATE, BASE))

    def test_missing_accuracy_fails(self):
        self.assertFalse(passes_quality_gate(gated_rec("a"), GATE, BASE))

    def test_missing_baseline_fails_unflagged(self):
        self.assertFalse(passes_quality_gate(gated_rec("a", accuracy=0.6), GATE, None))


class BranchBaselineScoreTest(unittest.TestCase):
    def test_reads_same_branch_baseline_label(self):
        records = [
            gated_rec("h1", accuracy=0.62, config_label="baseline"),
            gated_rec("h2", accuracy=0.30, config_label="attention-backend=flashinfer"),
            gated_rec("o", accuracy=0.99, config_label="baseline", branch="b200-bf16kv"),
        ]
        self.assertEqual(branch_baseline_score(records, "b200-fp8kv", GATE), 0.62)
        self.assertEqual(branch_baseline_score(records, "b200-bf16kv", GATE), 0.99)

    def test_absent_baseline_is_none(self):
        records = [gated_rec("h2", accuracy=0.3, config_label="ofat")]
        self.assertIsNone(branch_baseline_score(records, "b200-fp8kv", GATE))


class GateFrontierTest(unittest.TestCase):
    def test_gate_failing_excluded_via_baseline(self):
        records = [
            gated_rec("baseline", ptok=22, decode_thr=1500, accuracy=0.60, config_label="baseline"),
            gated_rec("good", ptok=20, decode_thr=1600, accuracy=0.55),
            gated_rec("fast-bad", ptok=15, decode_thr=900, accuracy=0.10),
        ]
        _, frontier = build_frontier(records, SLO_GLOBAL, branch="b200-fp8kv", gate=GATE)
        hashes = {e.config_hash for e in frontier}
        self.assertIn("good", hashes)
        self.assertNotIn("fast-bad", hashes)

    def test_no_gate_keeps_both(self):
        records = [
            gated_rec("good", ptok=20, decode_thr=1600, accuracy=0.55),
            gated_rec("fast-bad", ptok=15, decode_thr=900, accuracy=0.10),
        ]
        _, frontier = build_frontier(records, SLO_GLOBAL, gate=None)
        self.assertEqual({e.config_hash for e in frontier}, {"good", "fast-bad"})

    def test_cross_branch_never_co_ranked(self):
        records = [
            gated_rec("fp8kv", ptok=20, decode_thr=1600),
            gated_rec("bf16kv", ptok=10, decode_thr=4000, branch="b200-bf16kv"),
        ]
        _, frontier = build_frontier(records, SLO_GLOBAL, branch="b200-fp8kv")
        self.assertEqual({e.config_hash for e in frontier}, {"fp8kv"})

    def test_all_fail_yields_empty_frontier(self):
        records = [gated_rec("bad", quality_pass=False)]
        _, frontier = build_frontier(records, SLO_GLOBAL, gate=GATE)
        self.assertEqual(frontier, [])

    def test_entry_carries_accuracy_and_flag(self):
        e = to_entry(gated_rec("a", accuracy=0.6, quality_pass=True))
        self.assertEqual(e.accuracy, {"accuracy": 0.6})
        self.assertTrue(e.quality_pass)


class GateFailedPinsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.branch = SearchConfig.model_validate({
            "model": "m",
            "branches": [{
                "name": "b",
                "fixed": {"quantization": "modelopt_fp4"},
                "candidate": [
                    {"name": "g", "values": [1, 2]},
                    {"name": "p", "values": ["lo", "hi"]},
                ],
                "baseline": {"g": 1, "p": "lo"},
                "focused_grid": {"args": ["g"], "rationale": "g interacts", "pins": {"p": "hi"}},
            }],
        }).branch("b")

    def _pin_hash(self):
        return config_hash(_assemble(self.branch, {"p": "hi"}))

    def test_pin_with_failing_ofat_flagged(self):
        records = [{"config_hash": self._pin_hash(), "branch": "b", "label": "p=hi",
                    "quality_pass": False}]
        offenders = gate_failed_pins(self.branch, records, GATE)
        self.assertEqual(len(offenders), 1)
        self.assertEqual(offenders[0]["arg"], "p")
        self.assertEqual(offenders[0]["value"], "hi")

    def test_pin_with_passing_ofat_ok(self):
        records = [{"config_hash": self._pin_hash(), "branch": "b", "label": "p=hi",
                    "quality_pass": True}]
        self.assertEqual(gate_failed_pins(self.branch, records, GATE), [])

    def test_pin_without_record_not_flagged(self):
        self.assertEqual(gate_failed_pins(self.branch, [], GATE), [])


class GateStatusTest(unittest.TestCase):
    def test_no_gate_is_na(self):
        e = to_entry(gated_rec("a", accuracy=0.6))
        self.assertEqual(gate_status(e, None, None), ("n-a", None))

    def test_stamped_pass_and_fail(self):
        passed = to_entry(gated_rec("a", accuracy=0.6, quality_pass=True))
        failed = to_entry(gated_rec("b", accuracy=0.1, quality_pass=False))
        self.assertEqual(gate_status(passed, GATE, BASE), ("PASS", 0.6))
        self.assertEqual(gate_status(failed, GATE, BASE), ("FAIL", 0.1))

    def test_unmeasured_is_na_not_fail(self):
        e = to_entry(gated_rec("a"))
        self.assertEqual(gate_status(e, GATE, BASE), ("n-a", None))

    def test_no_baseline_is_na(self):
        e = to_entry(gated_rec("a", accuracy=0.6))
        self.assertEqual(gate_status(e, GATE, None), ("n-a", None))

    def test_recompute_relative_to_baseline(self):
        good = to_entry(gated_rec("a", accuracy=0.55))
        bad = to_entry(gated_rec("b", accuracy=0.1))
        self.assertEqual(gate_status(good, GATE, BASE)[0], "PASS")
        self.assertEqual(gate_status(bad, GATE, BASE)[0], "FAIL")


CONFIG_YAML = """\
model: m
slo:
  per_token_ms: 40
  ttft_p95_ms: 5000
quality_gate:
  dataset: gpqa
  metric: accuracy
  tolerance: 0.15
  direction: higher
branches:
  - name: b200-fp8kv
"""


class InspectMainTest(unittest.TestCase):
    def _write(self, d, records):
        cfg = Path(d) / "config.yaml"
        cfg.write_text(CONFIG_YAML)
        res = Path(d) / "results.jsonl"
        import json as _json
        res.write_text("\n".join(_json.dumps(r) for r in records) + "\n")
        return str(cfg), str(res)

    def _records(self):
        return [
            gated_rec("baseline", ptok=22, decode_thr=1500, accuracy=0.6, config_label="baseline"),
            gated_rec("fast-bad", ptok=15, decode_thr=900, accuracy=0.1),
            gated_rec("unmeasured", ptok=18, decode_thr=1200),
        ]

    def test_inspect_ranks_failing_and_unmeasured(self):
        with tempfile.TemporaryDirectory() as d:
            cfg, res = self._write(d, self._records())
            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = main(["--config", cfg, "--results", res, "--inspect"])
            out = buf.getvalue()
        self.assertEqual(rc, 0)
        self.assertIn("INSPECTION VIEW", out)
        self.assertIn("eligible(ignoring gate)=3", out)
        for h in ("baseline", "fast-bad", "unmeasured"):
            self.assertIn(h, out)

    def test_inspect_annotates_each_row(self):
        with tempfile.TemporaryDirectory() as d:
            cfg, res = self._write(d, self._records())
            buf = io.StringIO()
            with redirect_stdout(buf):
                main(["--config", cfg, "--results", res, "--inspect"])
            out = buf.getvalue()
        self.assertIn("gate=PASS(accuracy=0.6)", out)
        self.assertIn("gate=FAIL(accuracy=0.1)", out)
        self.assertIn("gate=n-a(accuracy=n/a)", out)

    def test_inspect_refuses_out(self):
        with tempfile.TemporaryDirectory() as d:
            cfg, res = self._write(d, self._records())
            with self.assertRaises(SystemExit):
                main(["--config", cfg, "--results", res, "--inspect",
                      "--out", str(Path(d) / "f.jsonl")])

    def test_default_path_unchanged_excludes_failing(self):
        with tempfile.TemporaryDirectory() as d:
            cfg, res = self._write(d, self._records())
            out_path = Path(d) / "frontier.jsonl"
            buf = io.StringIO()
            with redirect_stdout(buf):
                main(["--config", cfg, "--results", res, "--out", str(out_path)])
            written = out_path.read_text()
        self.assertIn("baseline", written)
        self.assertNotIn("fast-bad", written)
        self.assertNotIn("unmeasured", written)


if __name__ == "__main__":
    unittest.main()
