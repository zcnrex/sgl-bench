"""Driver tests ([[RFC-0001:C-LOOP-STRUCTURE]])."""

import tempfile
import unittest
from pathlib import Path

from sglbench.argsearch.driver import run_search, workload_points, write_results
from sglbench.argsearch.generate import ConfigPoint
from sglbench.argsearch.measure import WorkloadPoint
from sglbench.argsearch.schema import QualityGate

AXES = {
    "isl_osl_pairs": [[8192, 1024], [60000, 20]],
    "concurrency": [1, 32, 256],
}

POINTS = [
    ConfigPoint("nvfp4", "ofat", "baseline", {"attention-backend": "trtllm_mha"}),
    ConfigPoint("nvfp4", "ofat", "attention-backend=flashinfer",
                {"attention-backend": "flashinfer"}),
]


class FakeSession:
    def __init__(self, args, log):
        self.args = args
        self._log = log

    @property
    def client(self):
        return self

    def warmup(self, point):
        pass

    def measure(self, point):
        self._log.append(("measure", self.args["attention-backend"], point.label))
        return {"throughput": 100.0}

    def shutdown(self):
        self._log.append(("shutdown", self.args["attention-backend"]))


class FakeManager:
    def __init__(self):
        self.log: list = []
        self.launches: list[dict] = []

    def launch(self, args):
        self.launches.append(args)
        self.log.append(("launch", args["attention-backend"]))
        return FakeSession(args, self.log)


class WorkloadTest(unittest.TestCase):
    def test_expands_pairs_times_concurrency(self):
        pts = workload_points(AXES)
        self.assertEqual(len(pts), 6)
        self.assertIn(WorkloadPoint(8192, 1024, 1), pts)
        self.assertIn(WorkloadPoint(60000, 20, 256), pts)


class DriverTest(unittest.TestCase):
    def test_one_launch_per_config(self):
        mgr = FakeManager()
        run_search(POINTS, workload_points(AXES), mgr)
        self.assertEqual(len(mgr.launches), len(POINTS))
        self.assertEqual([l["attention-backend"] for l in mgr.launches],
                         ["trtllm_mha", "flashinfer"])

    def test_full_workload_coverage_per_server(self):
        mgr = FakeManager()
        results = run_search(POINTS, workload_points(AXES), mgr)
        self.assertEqual(len(results), len(POINTS) * 6)
        per_config = [r for r in results if r.config_hash == POINTS[0].config_hash]
        self.assertEqual(len(per_config), 6)

    def test_shutdown_before_next_launch(self):
        mgr = FakeManager()
        run_search(POINTS, workload_points(AXES), mgr)
        events = [e[0] for e in mgr.log]
        launch_idx = [i for i, e in enumerate(events) if e == "launch"]
        shutdown_idx = [i for i, e in enumerate(events) if e == "shutdown"]
        # second launch happens only after the first shutdown
        self.assertLess(shutdown_idx[0], launch_idx[1])
        # no relaunch occurs mid-sweep: exactly one launch per config
        self.assertEqual(len(launch_idx), len(POINTS))

    def test_no_relaunch_during_inner_sweep(self):
        mgr = FakeManager()
        run_search(POINTS, workload_points(AXES), mgr)
        # between a launch and its shutdown, only measures occur (no launch)
        segment = []
        depth = 0
        for kind, *_ in mgr.log:
            if kind == "launch":
                depth += 1
                self.assertEqual(depth, 1)
            elif kind == "shutdown":
                depth -= 1
            else:
                segment.append(kind)
        self.assertTrue(all(k == "measure" for k in segment))

    def test_write_results(self):
        mgr = FakeManager()
        results = run_search(POINTS, workload_points(AXES), mgr)
        with tempfile.TemporaryDirectory() as d:
            path = write_results(results, d)
            lines = Path(path).read_text().splitlines()
            self.assertEqual(len(lines), len(results))

    def test_no_gate_leaves_accuracy_unset(self):
        mgr = FakeManager()
        results = run_search(POINTS, workload_points(AXES), mgr)
        self.assertTrue(all(r.accuracy is None and r.quality_pass is None for r in results))


GATE = QualityGate(dataset="gpqa", metric="accuracy", threshold=0.45, direction="higher")


class GateStampingTest(unittest.TestCase):
    def test_evaluates_once_per_config_and_stamps_every_record(self):
        mgr = FakeManager()
        eval_calls = []

        def evaluate(session):
            eval_calls.append(session.args["attention-backend"])
            score = 0.6 if session.args["attention-backend"] == "trtllm_mha" else 0.3
            return {"accuracy": score}

        results = run_search(POINTS, workload_points(AXES), mgr, gate=GATE, evaluate=evaluate)
        self.assertEqual(eval_calls, ["trtllm_mha", "flashinfer"])
        good = [r for r in results if r.config_hash == POINTS[0].config_hash]
        bad = [r for r in results if r.config_hash == POINTS[1].config_hash]
        self.assertTrue(all(r.accuracy == {"accuracy": 0.6} and r.quality_pass for r in good))
        self.assertTrue(all(r.accuracy == {"accuracy": 0.3} and not r.quality_pass for r in bad))


class EvaluateHashesTest(unittest.TestCase):
    def test_only_designated_configs_evaluated_rest_reuse(self):
        mgr = FakeManager()
        eval_calls = []

        def evaluate(session):
            eval_calls.append(session.args["attention-backend"])
            return {"accuracy": 0.97}

        hashes = {POINTS[0].config_hash}
        results = run_search(POINTS, workload_points(AXES), mgr, gate=GATE,
                             evaluate=evaluate, evaluate_hashes=hashes)
        self.assertEqual(eval_calls, ["trtllm_mha"])
        self.assertTrue(all(r.accuracy == {"accuracy": 0.97} for r in results))
        self.assertTrue(all(r.quality_pass for r in results))


if __name__ == "__main__":
    unittest.main()
