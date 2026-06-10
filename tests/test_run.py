"""Live-search runner CLI tests ([[RFC-0001:C-LOOP-STRUCTURE]]), GPU-free via fakes."""

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from sglbench.argsearch import run
from sglbench.argsearch.generate import load_config

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "nemotron_v3_ultra.yaml"


class FakeSession:
    def __init__(self, args):
        self.args = args

    @property
    def client(self):
        return self

    def warmup(self, point):
        pass

    def measure(self, point):
        return {"decode_throughput_tok_s": 1500.0, "ttft_ms": 3000.0,
                "per_token_ms": 1000.0 * point.concurrency / 1500.0}

    def shutdown(self):
        pass


class FakeManager:
    def __init__(self, model, host="127.0.0.1", port=30000, **kw):
        self.model = model
        self.launches = []

    def launch(self, args):
        self.launches.append(args)
        return FakeSession(args)


class SelectTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.branch = load_config(CONFIG).branch("nvfp4")

    def test_limit_configs_picks_baseline_only(self):
        pts = run.select_points(self.branch, "ofat", 1)
        self.assertEqual(len(pts), 1)
        self.assertEqual(pts[0].label, "baseline")

    def test_workload_override_and_limit(self):
        axes = {"isl_osl_pairs": [[8192, 1024], [60000, 20]], "concurrency": [1, 8, 32]}
        pts = run.select_workload(axes, "iter", concurrency=[1], isl_osl=["8192x1024"], limit=0)
        self.assertEqual(len(pts), 1)
        self.assertEqual((pts[0].isl, pts[0].osl, pts[0].concurrency), (8192, 1024, 1))


class DryRunTest(unittest.TestCase):
    def test_dry_run_prints_commands(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = run.main([
                "--config", str(CONFIG), "--branch", "nvfp4", "--mode", "ofat",
                "--limit-configs", "1", "--concurrency", "1", "--isl-osl", "8192x1024",
                "--dry-run",
            ])
        out = buf.getvalue()
        self.assertEqual(rc, 0)
        self.assertIn("sglang.launch_server", out)
        self.assertIn("sglang.bench_one_batch_server", out)
        self.assertIn("--input-len 8192", out)


class LiveRunTest(unittest.TestCase):
    def test_run_with_fake_manager_writes_results(self):
        orig = run.SGLangServerManager
        run.SGLangServerManager = FakeManager
        try:
            with tempfile.TemporaryDirectory() as d:
                buf = io.StringIO()
                with redirect_stdout(buf):
                    rc = run.main([
                        "--config", str(CONFIG), "--branch", "nvfp4", "--mode", "ofat",
                        "--limit-configs", "1", "--concurrency", "1", "--isl-osl", "8192x1024",
                        "--repeats", "2", "--out", d, "--frontier",
                    ])
                self.assertEqual(rc, 0)
                lines = (Path(d) / "results.jsonl").read_text().splitlines()
                self.assertEqual(len(lines), 1)
                rec = json.loads(lines[0])
                self.assertEqual(rec["branch"], "nvfp4")
                self.assertIn("decode_throughput_tok_s", rec["metrics"])
        finally:
            run.SGLangServerManager = orig


if __name__ == "__main__":
    unittest.main()
