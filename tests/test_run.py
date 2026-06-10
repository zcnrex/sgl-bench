"""Live-search runner CLI tests ([[RFC-0001:C-LOOP-STRUCTURE]]), GPU-free via fakes."""

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from sglbench.argsearch import run
from sglbench.argsearch.generate import load_config
from sglbench.argsearch.measure import capture_environment, environment_digest

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "nemotron_v3_ultra_nvfp4.yaml"
NVFP4_MODEL = "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4"


def results_path(base, model=NVFP4_MODEL, transport="one-batch"):
    return run.run_dir(base, model, transport, environment_digest(capture_environment())) / "results.jsonl"


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
        cls.branch = load_config(CONFIG).branch("b200-fp8kv")

    def test_limit_configs_picks_baseline_only(self):
        pts = run.select_points(self.branch, "ofat", 1)
        self.assertEqual(len(pts), 1)
        self.assertEqual(pts[0].label, "baseline")

    def test_only_config_filters_by_hash_or_label(self):
        baseline = run.select_points(self.branch, "ofat", 0)[0]
        sub = run.select_points(self.branch, "ofat", 0, only_config=baseline.config_hash)
        self.assertEqual([p.config_hash for p in sub], [baseline.config_hash])
        by_label = run.select_points(self.branch, "ofat", 0, only_config="mamba-backend=triton")
        self.assertTrue(all("mamba-backend=triton" in p.label for p in by_label))
        self.assertEqual(len(by_label), 1)

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
                "--config", str(CONFIG), "--branch", "b200-fp8kv", "--mode", "ofat",
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
                        "--config", str(CONFIG), "--branch", "b200-fp8kv", "--mode", "ofat",
                        "--limit-configs", "1", "--concurrency", "1", "--isl-osl", "8192x1024",
                        "--repeats", "2", "--out", d, "--frontier",
                    ])
                self.assertEqual(rc, 0)
                rp = results_path(d)
                lines = rp.read_text().splitlines()
                self.assertEqual(len(lines), 1)
                rec = json.loads(lines[0])
                self.assertEqual(rec["branch"], "b200-fp8kv")
                self.assertIn("decode_throughput_tok_s", rec["metrics"])
                manifest = json.loads((rp.parent / "manifest.json").read_text())
                self.assertEqual(manifest["branch"], "b200-fp8kv")
                self.assertEqual(manifest["bench_tool"], "bench_one_batch_server")
                self.assertIn("measured_at", manifest)
                self.assertIn("environment", manifest)
        finally:
            run.SGLangServerManager = orig


class GateWiringTest(unittest.TestCase):
    """The gate is within-branch and baseline-relative ([[RFC-0001:C-QUALITY-GATE]]): the
    baseline config is the reference (it always passes its own gate); a variant is excluded
    only when it degrades beyond the tolerance from that same-branch baseline."""

    def _run(self, scores, out_dir, limit):
        orig_mgr, orig_ev = run.SGLangServerManager, run.GSM8KEvaluator
        run.SGLangServerManager = FakeManager
        it = iter(scores)

        class FakeEval:
            def __init__(self, base_url, **kw):
                pass

            def evaluate(self):
                return {"accuracy": next(it)}

        run.GSM8KEvaluator = FakeEval
        try:
            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = run.main([
                    "--config", str(CONFIG), "--branch", "b200-fp8kv", "--mode", "ofat",
                    "--limit-configs", str(limit), "--concurrency", "1", "--isl-osl", "8192x256",
                    "--repeats", "2", "--gsm8k-examples", "20", "--out", out_dir, "--frontier",
                ])
            return rc, buf.getvalue()
        finally:
            run.SGLangServerManager, run.GSM8KEvaluator = orig_mgr, orig_ev

    def _records(self, d):
        return [json.loads(l) for l in results_path(d).read_text().splitlines()]

    def test_baseline_passes_its_own_gate(self):
        with tempfile.TemporaryDirectory() as d:
            rc, out = self._run([0.50], d, limit=1)
            self.assertEqual(rc, 0)
            recs = self._records(d)
            self.assertTrue(all(r["config_label"] == "baseline" for r in recs))
            self.assertTrue(all(r["quality_pass"] for r in recs))

    def test_passing_variant_admitted(self):
        with tempfile.TemporaryDirectory() as d:
            rc, out = self._run([0.97, 0.96], d, limit=2)
            self.assertEqual(rc, 0)
            variant = [r for r in self._records(d) if r["config_label"] != "baseline"]
            self.assertTrue(variant and all(r["quality_pass"] for r in variant))

    def test_degraded_variant_excluded(self):
        with tempfile.TemporaryDirectory() as d:
            rc, out = self._run([0.97, 0.50], d, limit=2)
            self.assertEqual(rc, 0)
            recs = self._records(d)
            baseline = [r for r in recs if r["config_label"] == "baseline"]
            variant = [r for r in recs if r["config_label"] != "baseline"]
            self.assertTrue(all(r["quality_pass"] for r in baseline))
            self.assertTrue(variant and all(r["quality_pass"] is False for r in variant))


INVARIANT_DOC = {
    "model": "m",
    "quality_gate": {"dataset": "gsm8k", "metric": "accuracy", "tolerance": 0.05},
    "branches": [{
        "name": "b",
        "fixed": {"quantization": "modelopt_fp4", "kv-cache-dtype": "fp8_e4m3"},
        "candidate": [
            {"name": "ep-size", "values": [1, 4], "accuracy_invariant": True},
            {"name": "dp-size", "values": [1, 4], "accuracy_invariant": True},
        ],
        "baseline": {"ep-size": 1, "dp-size": 1},
    }],
}


class AccuracyInvariantSkipTest(unittest.TestCase):
    """The accuracy-invariant fast path ([[RFC-0001:C-QUALITY-GATE]]): per-config eval is
    skipped, but the baseline and a constructed all-extreme spot-check are gate-evaluated and
    the skip is recorded in the manifest with the spot-check identity and its gate result."""

    def _run(self, cfg_doc, scores, out_dir, extra=(), mode="ofat"):
        import yaml
        cfgp = Path(out_dir) / "inv.yaml"
        cfgp.write_text(yaml.safe_dump(cfg_doc))
        orig_mgr, orig_ev = run.SGLangServerManager, run.GSM8KEvaluator
        run.SGLangServerManager = FakeManager
        it = iter(scores)

        class FakeEval:
            def __init__(self, base_url, **kw):
                pass

            def evaluate(self):
                return {"accuracy": next(it)}

        run.GSM8KEvaluator = FakeEval
        try:
            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = run.main([
                    "--config", str(cfgp), "--branch", "b", "--mode", mode,
                    "--concurrency", "1", "--isl-osl", "8192x256",
                    "--repeats", "2", "--gsm8k-examples", "20", "--out", out_dir, *extra,
                ])
            return rc, buf.getvalue()
        finally:
            run.SGLangServerManager, run.GSM8KEvaluator = orig_mgr, orig_ev

    def _manifest(self, d, model="m"):
        td = run.run_dir(d, model, "one-batch", environment_digest(capture_environment()))
        return json.loads((td / "manifest.json").read_text())

    def test_manifest_records_skip_spotcheck_classification_and_kv_treatment(self):
        with tempfile.TemporaryDirectory() as d:
            rc, out = self._run(INVARIANT_DOC, [0.90, 0.89], d)
            self.assertEqual(rc, 0)
            m = self._manifest(d)
            self.assertEqual(m["kv_cache_precision_treatment"], "branch-key")
            self.assertEqual(
                m["arg_classification"],
                {"ep-size": "accuracy-invariant", "dp-size": "accuracy-invariant"},
            )
            skip = m["accuracy_skip"]
            self.assertTrue(skip["per_config_eval_skipped"])
            self.assertEqual(skip["spot_check"]["args"]["ep-size"], 4)
            self.assertEqual(skip["spot_check"]["args"]["dp-size"], 4)
            gr = skip["spot_check"]["gate_result"]
            self.assertIsNotNone(gr)
            self.assertTrue(gr["quality_pass"])
            self.assertEqual(m["hardware_reconciliation"]["status"], "undetermined")

    def test_spotcheck_failure_recorded_in_gate_result(self):
        with tempfile.TemporaryDirectory() as d:
            rc, out = self._run(INVARIANT_DOC, [0.90, 0.50], d)
            self.assertEqual(rc, 0)
            gr = self._manifest(d)["accuracy_skip"]["spot_check"]["gate_result"]
            self.assertFalse(gr["quality_pass"])

    def test_grid_mode_spotcheck_is_gate_judged(self):
        grid_doc = json.loads(json.dumps(INVARIANT_DOC))
        grid_doc["branches"][0]["focused_grid"] = {
            "args": ["ep-size", "dp-size"],
            "rationale": "both vary jointly",
        }
        with tempfile.TemporaryDirectory() as d:
            rc, out = self._run(grid_doc, [0.90, 0.89], d, mode="grid")
            self.assertEqual(rc, 0)
            m = self._manifest(d)
            gr = m["accuracy_skip"]["spot_check"]["gate_result"]
            self.assertIsNotNone(gr)
            self.assertIsInstance(gr["quality_pass"], bool)
            self.assertTrue(gr["quality_pass"])
            recs = [json.loads(l) for l in results_path(d, model="m").read_text().splitlines()]
            self.assertTrue(any(r["config_label"] == "baseline" for r in recs))

    def test_strict_hardware_aborts_on_mismatch(self):
        doc = json.loads(json.dumps(INVARIANT_DOC))
        doc["branches"][0]["hardware"] = "8xH100"
        orig_cap = run.capture_environment
        run.capture_environment = lambda *a, **k: {
            "hardware": {"accelerator": "NVIDIA B200", "device_count": 4},
            "sglang_commit": "x", "library_versions": {}, "network_env": {},
        }
        try:
            with tempfile.TemporaryDirectory() as d:
                with self.assertRaises(SystemExit):
                    self._run(doc, [0.9, 0.9], d, extra=["--strict-hardware"])
        finally:
            run.capture_environment = orig_cap


class ReuseTest(unittest.TestCase):
    def _run(self, out_dir, extra):
        orig = run.SGLangServerManager
        run.SGLangServerManager = FakeManager
        try:
            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = run.main([
                    "--config", str(CONFIG), "--branch", "b200-fp8kv", "--mode", "ofat",
                    "--limit-configs", "1", "--concurrency", "1", "8",
                    "--isl-osl", "8192x1024", "--repeats", "2", "--out", out_dir, *extra,
                ])
            return rc, buf.getvalue()
        finally:
            run.SGLangServerManager = orig

    def test_reinvoke_reuses_and_measures_only_missing(self):
        with tempfile.TemporaryDirectory() as d:
            rc, _ = self._run(d, ["--concurrency", "1"])
            rp = results_path(d)
            self.assertEqual(len(rp.read_text().splitlines()), 1)
            rc, out = self._run(d, ["--concurrency", "1", "8"])
            self.assertEqual(rc, 0)
            lines = rp.read_text().splitlines()
            self.assertEqual(len(lines), 2)
            keys = {(json.loads(l)["config_hash"], json.loads(l)["label"]) for l in lines}
            self.assertEqual(len(keys), 2)
            self.assertIn("reusing 1 recorded measurement", out)

    def test_force_remeasures_all(self):
        with tempfile.TemporaryDirectory() as d:
            self._run(d, ["--concurrency", "1"])
            rp = results_path(d)
            self.assertEqual(len(rp.read_text().splitlines()), 1)
            rc, out = self._run(d, ["--concurrency", "1", "--force"])
            self.assertEqual(rc, 0)
            self.assertEqual(len(rp.read_text().splitlines()), 1)
            self.assertNotIn("reusing", out)

    def test_manifest_written_even_when_all_reused(self):
        with tempfile.TemporaryDirectory() as d:
            self._run(d, ["--concurrency", "1"])
            rp = results_path(d)
            manifest = rp.parent / "manifest.json"
            manifest.unlink()
            rc, out = self._run(d, ["--concurrency", "1"])
            self.assertEqual(rc, 0)
            self.assertTrue(manifest.exists())
            self.assertEqual(len(rp.read_text().splitlines()), 1)

    def test_force_override_recorded_durably_in_history(self):
        with tempfile.TemporaryDirectory() as d:
            self._run(d, ["--concurrency", "1"])
            self._run(d, ["--concurrency", "1", "--force"])
            self._run(d, ["--concurrency", "1"])
            history = (results_path(d).parent / "manifests.jsonl").read_text().splitlines()
            self.assertEqual(len(history), 3)
            forced = [json.loads(line)["force"] for line in history]
            self.assertEqual(forced, [False, True, False])


class DryRunPathTest(unittest.TestCase):
    def test_dry_run_uses_run_dir_paths(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = run.main([
                "--config", str(CONFIG), "--branch", "b200-fp8kv", "--mode", "ofat",
                "--limit-configs", "1", "--concurrency", "1", "--isl-osl", "8192x1024",
                "--out", "outbase", "--dry-run",
            ])
        self.assertEqual(rc, 0)
        out = buf.getvalue()
        self.assertIn("outbase/nvidia-NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4/runs/bench_one_batch_server/", out)


if __name__ == "__main__":
    unittest.main()
