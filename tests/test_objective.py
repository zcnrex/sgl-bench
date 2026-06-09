"""Pareto-frontier-under-SLO tests ([[RFC-0001:C-OBJECTIVE]]): decode-first, TTFT non-gating."""

import tempfile
import unittest
from pathlib import Path

from sglbench.argsearch.objective import (
    FrontierEntry,
    build_frontier,
    pareto_frontier,
    passes_slo,
    to_entry,
    write_frontier,
)
from sglbench.argsearch.schema import SLO, SLOOverride

SLO_GLOBAL = SLO(per_token_ms=40, ttft_p95_ms=5000)
SLO_OVERRIDE = SLO(
    per_token_ms=40,
    ttft_p95_ms=5000,
    overrides=[SLOOverride(isl=60000, osl=20, per_token_ms=80)],
)


def rec(h, isl, osl, c, ttft_ms, per_token_ms, decode_thr, branch="nvfp4"):
    def agg(v):
        return {"median": v, "mean": v, "n": 2}

    return {
        "config_hash": h,
        "branch": branch,
        "label": h,
        "workload": {"isl": isl, "osl": osl, "concurrency": c},
        "metrics": {
            "ttft_ms": agg(ttft_ms),
            "per_token_ms": agg(per_token_ms),
            "decode_throughput_tok_s": agg(decode_thr),
        },
    }


def entry(h, thr, ptok, ttft=100.0):
    return FrontierEntry(h, "nvfp4", h, {"isl": 8192, "osl": 1024, "concurrency": 1},
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
            rec("d", 8192, 1024, 1, 3000, 25, 1200, branch="fp8"),
        ]
        passing, frontier = build_frontier(records, SLO_GLOBAL, branch="nvfp4")
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


if __name__ == "__main__":
    unittest.main()
