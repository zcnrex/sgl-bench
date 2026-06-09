"""Measurement tests ([[RFC-0001:C-MEASUREMENT]])."""

import unittest

from sglbench.argsearch.measure import (
    MIN_REPEATS,
    WorkloadPoint,
    capture_environment,
    measure_point,
)

POINT = WorkloadPoint(isl=8192, osl=1024, concurrency=32)


class FakeClient:
    """Records call order and returns canned per-run metrics."""

    def __init__(self, runs):
        self.runs = list(runs)
        self.calls: list[str] = []
        self._i = 0

    def warmup(self, point):
        self.calls.append("warmup")

    def measure(self, point):
        self.calls.append("measure")
        out = self.runs[self._i]
        self._i += 1
        return out


class MeasureTest(unittest.TestCase):
    def test_warmup_precedes_measurement(self):
        client = FakeClient([{"throughput": 100.0}, {"throughput": 102.0}])
        measure_point(
            client, config_hash="abc", branch="nvfp4", point=POINT, environment={}
        )
        self.assertEqual(client.calls[0], "warmup")
        self.assertEqual(client.calls.count("warmup"), 1)

    def test_default_runs_at_least_two_repeats(self):
        client = FakeClient([{"throughput": 100.0}, {"throughput": 102.0}])
        res = measure_point(
            client, config_hash="abc", branch="nvfp4", point=POINT, environment={}
        )
        self.assertGreaterEqual(len(res.repeats), MIN_REPEATS)
        self.assertEqual(client.calls.count("measure"), MIN_REPEATS)

    def test_repeats_below_minimum_rejected(self):
        client = FakeClient([{"throughput": 100.0}])
        with self.assertRaises(ValueError):
            measure_point(
                client,
                config_hash="abc",
                branch="nvfp4",
                point=POINT,
                repeats=1,
                environment={},
            )

    def test_aggregation_mean_and_median(self):
        client = FakeClient(
            [{"ttft": 10.0}, {"ttft": 20.0}, {"ttft": 30.0}]
        )
        res = measure_point(
            client,
            config_hash="abc",
            branch="nvfp4",
            point=POINT,
            repeats=3,
            environment={},
        )
        self.assertEqual(res.metrics["ttft"]["mean"], 20.0)
        self.assertEqual(res.metrics["ttft"]["median"], 20.0)
        self.assertEqual(res.metrics["ttft"]["n"], 3)

    def test_provenance_completeness(self):
        client = FakeClient([{"throughput": 100.0}, {"throughput": 102.0}])
        env = {
            "sglang_commit": "deadbeef",
            "library_versions": {"torch": "2.5.0"},
            "network_env": {"NCCL_DEBUG": "INFO"},
        }
        res = measure_point(
            client, config_hash="cfg123", branch="nvfp4", point=POINT, environment=env
        )
        rec = res.to_record()
        self.assertEqual(rec["config_hash"], "cfg123")
        self.assertEqual(rec["workload"], {"isl": 8192, "osl": 1024, "concurrency": 32})
        self.assertIn("throughput", rec["metrics"])
        self.assertEqual(rec["environment"], env)

    def test_capture_environment_collects_network_vars(self):
        environ = {
            "NCCL_DEBUG": "INFO",
            "NVSHMEM_IB_ENABLE_IBGDA": "1",
            "TORCH_NCCL_BLOCKING_WAIT": "1",
            "MASTER_ADDR": "10.0.0.1",
            "WORLD_SIZE": "8",
            "LOCAL_RANK": "0",
            "EFA_USE_DEVICE_RDMA": "1",
            "FI_PROVIDER": "efa",
            "IB_HCA": "mlx5_0",
            "HOME": "/root",
            "PATH": "/usr/bin",
        }
        env = capture_environment(sglang_commit="abc123", environ=environ)
        self.assertEqual(env["sglang_commit"], "abc123")
        self.assertEqual(
            set(env["network_env"]),
            {
                "NCCL_DEBUG",
                "NVSHMEM_IB_ENABLE_IBGDA",
                "TORCH_NCCL_BLOCKING_WAIT",
                "MASTER_ADDR",
                "WORLD_SIZE",
                "LOCAL_RANK",
                "EFA_USE_DEVICE_RDMA",
                "FI_PROVIDER",
                "IB_HCA",
            },
        )
        self.assertNotIn("HOME", env["network_env"])
        self.assertNotIn("PATH", env["network_env"])

    def test_capture_environment_honors_operator_extra_prefixes(self):
        environ = {
            "MYCLUSTER_NET_FOO": "1",
            "SGLBENCH_NET_ENV_EXTRA": "MYCLUSTER_NET_",
            "UNRELATED": "x",
        }
        env = capture_environment(sglang_commit="c", environ=environ)
        self.assertIn("MYCLUSTER_NET_FOO", env["network_env"])
        self.assertNotIn("UNRELATED", env["network_env"])


if __name__ == "__main__":
    unittest.main()
