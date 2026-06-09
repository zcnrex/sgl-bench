"""Concrete SGLang adapter tests ([[RFC-0001:C-LOOP-STRUCTURE]], C-MEASUREMENT, C-BASELINE-ANCHOR)."""

import json
import unittest

from sglbench.argsearch import metrics as M
from sglbench.argsearch.driver import ServerManager, ServerSession
from sglbench.argsearch.measure import BenchClient, WorkloadPoint, measure_point
from sglbench.argsearch.sglang_adapter import (
    BenchOneBatchClient,
    SGLangServerManager,
    SGLangSession,
    build_bench_cmd,
    build_launch_cmd,
    parse_result_jsonl,
    record_to_metrics,
    wait_until_ready,
)

POINT = WorkloadPoint(isl=8192, osl=1024, concurrency=32)


def _bench_line(batch_size=32, input_len=8192, output_len=1024, last_ttft=0.5,
                output_throughput=1600.0, overall_throughput=1700.0,
                input_throughput=40000.0, latency=2.0, last_gen_throughput=1280.0):
    return json.dumps({
        "run_name": "default",
        "batch_size": batch_size,
        "input_len": input_len,
        "output_len": output_len,
        "latency": latency,
        "input_throughput": input_throughput,
        "output_throughput": output_throughput,
        "overall_throughput": overall_throughput,
        "last_ttft": last_ttft,
        "last_gen_throughput": last_gen_throughput,
        "acc_length": 1.0,
        "cache_hit_rate": 0.0,
    })


class FakeProc:
    def __init__(self):
        self.terminated = False
        self.killed = False

    def terminate(self):
        self.terminated = True

    def wait(self, timeout=None):
        return 0

    def kill(self):
        self.killed = True


class BuildCmdTest(unittest.TestCase):
    def test_launch_cmd_structure(self):
        cmd = build_launch_cmd("my/model", "127.0.0.1", 30000,
                               {"tensor-parallel-size": 4, "enable-dp-attention": True})
        self.assertEqual(cmd[:3], ["python", "-m", "sglang.launch_server"])
        self.assertIn("--model-path", cmd)
        self.assertEqual(cmd[cmd.index("--model-path") + 1], "my/model")
        self.assertEqual(cmd[cmd.index("--port") + 1], "30000")
        self.assertIn("--tensor-parallel-size", cmd)
        self.assertEqual(cmd[cmd.index("--tensor-parallel-size") + 1], "4")
        self.assertIn("--enable-dp-attention", cmd)

    def test_bench_cmd_maps_workload(self):
        cmd = build_bench_cmd("http://h:1", POINT, "/tmp/r.jsonl")
        self.assertEqual(cmd[cmd.index("--batch-size") + 1], "32")
        self.assertEqual(cmd[cmd.index("--input-len") + 1], "8192")
        self.assertEqual(cmd[cmd.index("--output-len") + 1], "1024")
        self.assertEqual(cmd[cmd.index("--base-url") + 1], "http://h:1")
        self.assertEqual(cmd[cmd.index("--result-filename") + 1], "/tmp/r.jsonl")


class ParseTest(unittest.TestCase):
    def test_picks_matching_record(self):
        text = "\n".join([
            _bench_line(batch_size=1, output_throughput=50.0),
            _bench_line(batch_size=32, output_throughput=1600.0),
        ])
        rec = parse_result_jsonl(text, POINT)
        self.assertEqual(rec["batch_size"], 32)

    def test_last_record_without_point(self):
        text = "\n".join([_bench_line(latency=1.0), _bench_line(latency=9.0)])
        self.assertEqual(parse_result_jsonl(text)["latency"], 9.0)

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            parse_result_jsonl("   \n")


class MetricMapTest(unittest.TestCase):
    def test_canonical_mapping(self):
        rec = json.loads(_bench_line(batch_size=32, last_ttft=0.5,
                                     output_throughput=1600.0, last_gen_throughput=1280.0))
        m = record_to_metrics(rec)
        self.assertAlmostEqual(m[M.TTFT_MS], 500.0)
        self.assertAlmostEqual(m[M.DECODE_THROUGHPUT], 1280.0)
        self.assertAlmostEqual(m[M.OUTPUT_THROUGHPUT], 1600.0)
        self.assertAlmostEqual(m[M.PER_TOKEN_MS], 1000.0 * 32 / 1280.0)
        self.assertAlmostEqual(m[M.E2E_LATENCY_S], 2.0)
        self.assertIn(M.OVERALL_THROUGHPUT, m)
        self.assertIn(M.INPUT_THROUGHPUT, m)

    def test_per_token_falls_back_to_output_throughput(self):
        rec = json.loads(_bench_line(batch_size=16, output_throughput=800.0))
        del rec["last_gen_throughput"]
        m = record_to_metrics(rec)
        self.assertNotIn(M.DECODE_THROUGHPUT, m)
        self.assertAlmostEqual(m[M.PER_TOKEN_MS], 1000.0 * 16 / 800.0)


class WaitReadyTest(unittest.TestCase):
    def test_returns_when_probe_true(self):
        seq = iter([False, False, True])
        slept = []
        wait_until_ready(lambda: next(seq), timeout_s=100, interval_s=1,
                         sleep=slept.append, clock=lambda: 0.0)
        self.assertEqual(len(slept), 2)

    def test_timeout_raises(self):
        clock = iter([0.0, 0.0, 100.0])
        with self.assertRaises(TimeoutError):
            wait_until_ready(lambda: False, timeout_s=10, interval_s=1,
                             sleep=lambda s: None, clock=lambda: next(clock))


class BenchClientTest(unittest.TestCase):
    def test_measure_returns_canonical_metrics(self):
        client = BenchOneBatchClient(
            "http://h:1", run_bench=lambda cmd, path: _bench_line(batch_size=32))
        m = client.measure(POINT)
        self.assertAlmostEqual(m[M.TTFT_MS], 500.0)
        self.assertIn(M.OUTPUT_THROUGHPUT, m)

    def test_warmup_runs(self):
        calls = []
        client = BenchOneBatchClient(
            "http://h:1",
            run_bench=lambda cmd, path: (calls.append(cmd) or _bench_line()))
        client.warmup(POINT)
        self.assertEqual(len(calls), 1)

    def test_tool_identity_recorded_in_provenance(self):
        client = BenchOneBatchClient(
            "http://h:1", run_bench=lambda cmd, path: _bench_line())
        self.assertEqual(client.tool, "bench_one_batch_server")
        result = measure_point(
            client, config_hash="h", branch="nvfp4", point=POINT,
            repeats=2, environment={})
        self.assertEqual(result.bench_tool, "bench_one_batch_server")


class ManagerTest(unittest.TestCase):
    def _manager(self, probe):
        spawned = []
        return SGLangServerManager(
            "my/model",
            popen=lambda cmd: spawned.append(cmd) or FakeProc(),
            probe=probe,
            run_bench=lambda cmd, path: _bench_line(),
            sleep=lambda s: None,
            clock=lambda: 0.0,
        ), spawned

    def test_launch_returns_session(self):
        mgr, spawned = self._manager(lambda: True)
        session = mgr.launch({"tensor-parallel-size": 4})
        self.assertIsInstance(session, SGLangSession)
        self.assertIsInstance(session.client, BenchOneBatchClient)
        self.assertEqual(len(spawned), 1)

    def test_launch_timeout_terminates_proc(self):
        proc_holder = []

        def popen(cmd):
            p = FakeProc()
            proc_holder.append(p)
            return p

        clock = iter([0.0, 0.0, 999.0])
        mgr = SGLangServerManager(
            "my/model", popen=popen, probe=lambda: False,
            launch_timeout_s=10, sleep=lambda s: None, clock=lambda: next(clock))
        with self.assertRaises(TimeoutError):
            mgr.launch({})
        self.assertTrue(proc_holder[0].terminated)

    def test_shutdown_terminates(self):
        proc = FakeProc()
        SGLangSession(proc, object()).shutdown()
        self.assertTrue(proc.terminated)


class ProtocolConformanceTest(unittest.TestCase):
    def test_satisfies_protocols(self):
        mgr = SGLangServerManager("m", probe=lambda: True,
                                  popen=lambda cmd: FakeProc(),
                                  run_bench=lambda cmd, path: _bench_line(),
                                  sleep=lambda s: None, clock=lambda: 0.0)
        self.assertIsInstance(mgr, ServerManager)
        session = mgr.launch({})
        self.assertIsInstance(session, ServerSession)
        self.assertIsInstance(session.client, BenchClient)


if __name__ == "__main__":
    unittest.main()
