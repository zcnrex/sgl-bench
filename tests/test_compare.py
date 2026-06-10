"""Per-concurrency baseline-comparison view ([[RFC-0001:C-OBJECTIVE]] companion)."""

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from sglbench.argsearch.compare import main
from sglbench.argsearch.generate import generate_ofat, load_config

CONFIG_YAML = """\
model: m
slo:
  per_token_ms: 40
  ttft_p95_ms: 5000
quality_gate:
  dataset: gsm8k
  metric: accuracy
  threshold: 0.95
  direction: higher
precision_branches:
  - name: nvfp4
    baseline:
      mamba-ssm-dtype: float32
    candidate:
      - name: mamba-ssm-dtype
        values: [float32, bfloat16]
"""


def agg(v):
    return {"median": v, "mean": v, "n": 2}


def rec(h, c, decode_thr, ptok=20.0, ttft=400.0, accuracy=None, branch="nvfp4"):
    r = {
        "config_hash": h,
        "branch": branch,
        "label": h,
        "workload": {"isl": 8192, "osl": 256, "concurrency": c},
        "metrics": {
            "ttft_ms": agg(ttft),
            "per_token_ms": agg(ptok),
            "decode_throughput_tok_s": agg(decode_thr),
        },
    }
    if accuracy is not None:
        r["accuracy"] = {"accuracy": accuracy}
    return r


class CompareMainTest(unittest.TestCase):
    def _setup(self, d, records):
        cfg = Path(d) / "config.yaml"
        cfg.write_text(CONFIG_YAML)
        res = Path(d) / "results.jsonl"
        res.write_text("\n".join(json.dumps(r) for r in records) + "\n")
        branch = load_config(str(cfg)).branch("nvfp4")
        points = generate_ofat(branch)
        return str(cfg), str(res), points[0].config_hash, points[1].config_hash

    def _run(self, cfg, res):
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = main(["--config", cfg, "--branch", "nvfp4", "--results", res])
        return rc, buf.getvalue()

    def test_delta_and_baseline_marker(self):
        with tempfile.TemporaryDirectory() as d:
            cfg, res, base_h, cand_h = self._setup(d, [])
            records = [
                rec(base_h, 1, 100.0, accuracy=1.0),
                rec(cand_h, 1, 110.0, accuracy=0.97),
            ]
            Path(res).write_text("\n".join(json.dumps(r) for r in records) + "\n")
            rc, out = self._run(cfg, res)
        self.assertEqual(rc, 0)
        self.assertIn("baseline", out)
        self.assertIn("mamba-ssm-dtype=bfloat16", out)
        self.assertIn("+10.0%", out)
        self.assertIn("0.97", out)

    def test_negative_delta_and_per_concurrency_blocks(self):
        with tempfile.TemporaryDirectory() as d:
            cfg, res, base_h, cand_h = self._setup(d, [])
            records = [
                rec(base_h, 1, 100.0, accuracy=1.0),
                rec(cand_h, 1, 90.0, accuracy=1.0),
                rec(base_h, 8, 800.0, accuracy=1.0),
                rec(cand_h, 8, 840.0, accuracy=0.99),
            ]
            Path(res).write_text("\n".join(json.dumps(r) for r in records) + "\n")
            rc, out = self._run(cfg, res)
        self.assertEqual(rc, 0)
        self.assertIn("concurrency = 1", out)
        self.assertIn("concurrency = 8", out)
        self.assertIn("-10.0%", out)
        self.assertIn("+5.0%", out)

    def test_missing_candidate_at_concurrency_is_skipped(self):
        with tempfile.TemporaryDirectory() as d:
            cfg, res, base_h, cand_h = self._setup(d, [])
            records = [
                rec(base_h, 1, 100.0, accuracy=1.0),
                rec(cand_h, 8, 840.0, accuracy=1.0),
            ]
            Path(res).write_text("\n".join(json.dumps(r) for r in records) + "\n")
            rc, out = self._run(cfg, res)
        self.assertEqual(rc, 0)
        block1 = out.split("concurrency = 8")[0]
        self.assertIn("concurrency = 1", block1)
        self.assertNotIn("mamba-ssm-dtype=bfloat16", block1)


if __name__ == "__main__":
    unittest.main()
