"""Concrete SGLang server manager + bench client for the search driver.

Implements the driver's ServerManager / ServerSession / BenchClient protocols against a
real deployment: launch `python -m sglang.launch_server` with a ConfigPoint's
restart-required args, poll `/health` until ready, and tear the server down on shutdown
([[RFC-0001:C-LOOP-STRUCTURE]]). The bench transport wraps `sglang.bench_one_batch_server`
-- the stable comparability anchor of [[RFC-0001:C-BASELINE-ANCHOR]] -- and maps its
result record into the canonical metric vocabulary ([[RFC-0001:C-MEASUREMENT]]).

Command construction and result parsing are pure functions, unit-tested without a GPU.
The subprocess and HTTP transport are injectable so the orchestration is exercised with
fakes; an end-to-end reconciliation against a live server is required before these
numbers are relied upon ([[RFC-0001:C-BASELINE-ANCHOR]]).
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable

from . import metrics as M
from .generate import args_to_cli
from .jsonl import parse_jsonl
from .measure import BenchClient, WorkloadPoint

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8888
DEFAULT_LAUNCH_TIMEOUT_S = 1800.0
DEFAULT_READY_INTERVAL_S = 5.0
HEALTH_PATH = "/health"
BASE_URL_MODEL = "None"


def build_launch_cmd(
    model: str, host: str, port: int, args: dict, extra=()
) -> list[str]:
    """`launch_server` command for a restart-required config ([[RFC-0001:C-LOOP-STRUCTURE]])."""
    cmd = [
        "python",
        "-m",
        "sglang.launch_server",
        "--model-path",
        model,
        "--host",
        host,
        "--port",
        str(port),
    ]
    rendered = args_to_cli(args)
    if rendered:
        cmd += shlex.split(rendered)
    cmd += list(extra)
    return cmd


def build_bench_cmd(
    base_url: str,
    point: WorkloadPoint,
    result_path: str,
    model: str = BASE_URL_MODEL,
    extra=(),
) -> list[str]:
    """`bench_one_batch_server` command mapping a workload point to one batch run.

    Concurrency maps to `--batch-size`; isl/osl to `--input-len`/`--output-len`. Results
    are written as JSONL to `result_path` (the `--result-filename` contract).
    """
    return [
        "python",
        "-m",
        "sglang.bench_one_batch_server",
        "--model",
        model,
        "--base-url",
        base_url,
        "--batch-size",
        str(point.concurrency),
        "--input-len",
        str(point.isl),
        "--output-len",
        str(point.osl),
        "--result-filename",
        result_path,
        *extra,
    ]


def parse_result_jsonl(text: str, point: WorkloadPoint | None = None) -> dict:
    """Parse `bench_one_batch_server` JSONL output into one result record.

    Picks the record matching the workload point's batch/input/output when `point` is
    given, else the last record. Raises ValueError when no record is present.
    """
    records = parse_jsonl(text)
    if not records:
        raise ValueError("bench_one_batch_server produced no result record")
    if point is not None:
        for rec in reversed(records):
            if (
                rec.get("batch_size") == point.concurrency
                and rec.get("input_len") == point.isl
                and rec.get("output_len") == point.osl
            ):
                return rec
    return records[-1]


def record_to_metrics(record: dict) -> dict[str, float]:
    """Map a `bench_one_batch_server` record to the canonical metric vocabulary.

    `last_ttft` is seconds. Steady-state decode throughput ([[RFC-0001:C-OBJECTIVE]]) is the
    tail decode rate `last_gen_throughput` (aggregate tok/s at the longest context), from
    which per-user inter-token latency is `batch_size / last_gen_throughput`; it falls back
    to the OSL-averaged `output_throughput` only when the tail rate is absent.
    """
    out: dict[str, float] = {}
    bs = float(record.get("batch_size", 1)) or 1.0
    if "last_ttft" in record:
        out[M.TTFT_MS] = float(record["last_ttft"]) * 1000.0
    decode_tput = None
    if "last_gen_throughput" in record and float(record["last_gen_throughput"]) > 0:
        decode_tput = float(record["last_gen_throughput"])
        out[M.DECODE_THROUGHPUT] = decode_tput
    if "output_throughput" in record:
        out[M.OUTPUT_THROUGHPUT] = float(record["output_throughput"])
    if decode_tput is None and out.get(M.OUTPUT_THROUGHPUT, 0.0) > 0:
        decode_tput = out[M.OUTPUT_THROUGHPUT]
    if decode_tput is not None and decode_tput > 0:
        out[M.PER_TOKEN_MS] = 1000.0 * bs / decode_tput
    if "overall_throughput" in record:
        out[M.OVERALL_THROUGHPUT] = float(record["overall_throughput"])
    if "input_throughput" in record:
        out[M.INPUT_THROUGHPUT] = float(record["input_throughput"])
    if "latency" in record:
        out[M.E2E_LATENCY_S] = float(record["latency"])
    return out


def _health_ok(base_url: str, path: str = HEALTH_PATH, timeout: float = 5.0) -> bool:
    url = base_url.rstrip("/") + path
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return 200 <= getattr(resp, "status", 200) < 300
    except (urllib.error.URLError, OSError):
        return False


def wait_until_ready(
    probe: Callable[[], bool],
    timeout_s: float,
    interval_s: float,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> None:
    """Block until `probe()` is true or `timeout_s` elapses, raising TimeoutError."""
    start = clock()
    while True:
        if probe():
            return
        if clock() - start >= timeout_s:
            raise TimeoutError(f"server not ready within {timeout_s}s")
        sleep(interval_s)


def _run_cmd(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def _default_run_bench(cmd: list[str], result_path: str) -> str:
    _run_cmd(cmd)
    p = Path(result_path)
    return p.read_text() if p.exists() else ""


class _SubprocessBenchClient:
    """Shared temp-dir -> subprocess -> parse path for the bench transports.

    Subclasses supply the command builder and the result parser; the warmup/measure
    BenchClient surface and the result-file plumbing live here.
    """

    tool: str

    def __init__(
        self, run_bench: Callable[[list[str], str], str] = _default_run_bench
    ) -> None:
        self._run_bench = run_bench

    def _bench_cmd(self, point: WorkloadPoint, result_path: str) -> list[str]:
        raise NotImplementedError

    def _parse(self, text: str, point: WorkloadPoint) -> dict[str, float]:
        raise NotImplementedError

    def _run(self, point: WorkloadPoint) -> dict[str, float]:
        with tempfile.TemporaryDirectory() as d:
            result_path = os.path.join(d, "result.jsonl")
            text = self._run_bench(self._bench_cmd(point, result_path), result_path)
            return self._parse(text, point)

    def warmup(self, point: WorkloadPoint) -> None:
        self._run(point)

    def measure(self, point: WorkloadPoint) -> dict[str, float]:
        return self._run(point)


class BenchOneBatchClient(_SubprocessBenchClient):
    """BenchClient over `sglang.bench_one_batch_server` ([[RFC-0001:C-BASELINE-ANCHOR]])."""

    tool = "bench_one_batch_server"

    def __init__(
        self,
        base_url: str,
        model: str = BASE_URL_MODEL,
        *,
        extra_args=(),
        run_bench: Callable[[list[str], str], str] = _default_run_bench,
    ) -> None:
        super().__init__(run_bench)
        self.base_url = base_url
        self.model = model
        self.extra_args = tuple(extra_args)

    def _bench_cmd(self, point: WorkloadPoint, result_path: str) -> list[str]:
        return build_bench_cmd(
            self.base_url, point, result_path, self.model, self.extra_args
        )

    def _parse(self, text: str, point: WorkloadPoint) -> dict[str, float]:
        return record_to_metrics(parse_result_jsonl(text, point))


def _terminate(proc, timeout: float = 30.0) -> None:
    try:
        proc.terminate()
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


class SGLangSession:
    """A launched server bound to its bench client (ServerSession protocol)."""

    def __init__(self, proc, client: BenchClient, terminate=_terminate) -> None:
        self._proc = proc
        self._client = client
        self._terminate = terminate

    @property
    def client(self) -> BenchClient:
        return self._client

    def shutdown(self) -> None:
        self._terminate(self._proc)


class SGLangServerManager:
    """Launches SGLang per restart-required config (ServerManager protocol)."""

    def __init__(
        self,
        model: str,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        *,
        launch_timeout_s: float = DEFAULT_LAUNCH_TIMEOUT_S,
        ready_interval_s: float = DEFAULT_READY_INTERVAL_S,
        extra_launch_args=(),
        bench_model: str = BASE_URL_MODEL,
        bench_extra_args=(),
        bench_client_factory: Callable[[str], BenchClient] | None = None,
        popen: Callable[[list[str]], object] = None,
        probe: Callable[[], bool] | None = None,
        run_bench: Callable[[list[str], str], str] = _default_run_bench,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.model = model
        self.host = host
        self.port = port
        self.launch_timeout_s = launch_timeout_s
        self.ready_interval_s = ready_interval_s
        self.extra_launch_args = tuple(extra_launch_args)
        self.bench_model = bench_model
        self.bench_extra_args = tuple(bench_extra_args)
        self._bench_client_factory = bench_client_factory
        self._popen = popen
        self._probe = probe
        self._run_bench = run_bench
        self._sleep = sleep
        self._clock = clock

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def _spawn(self, cmd: list[str]):
        if self._popen is not None:
            return self._popen(cmd)
        return subprocess.Popen(cmd, start_new_session=True)

    def launch(self, args: dict) -> SGLangSession:
        cmd = build_launch_cmd(
            self.model, self.host, self.port, args, self.extra_launch_args
        )
        proc = self._spawn(cmd)
        probe = self._probe or (lambda: _health_ok(self.base_url))
        try:
            wait_until_ready(
                probe,
                self.launch_timeout_s,
                self.ready_interval_s,
                self._sleep,
                self._clock,
            )
        except TimeoutError:
            _terminate(proc)
            raise
        if self._bench_client_factory is not None:
            client = self._bench_client_factory(self.base_url)
        else:
            client = BenchOneBatchClient(
                self.base_url,
                self.bench_model,
                extra_args=self.bench_extra_args,
                run_bench=self._run_bench,
            )
        return SGLangSession(proc, client)


GSM8K_DEFAULT_EXAMPLES = 200
GSM8K_DEFAULT_THREADS = 32


def build_gsm8k_cmd(
    base_url: str,
    out_dir: str,
    *,
    name: str = "gsm8k",
    num_examples: int = GSM8K_DEFAULT_EXAMPLES,
    num_threads: int = GSM8K_DEFAULT_THREADS,
    max_tokens: int | None = None,
    temperature: float | None = None,
    model: str | None = None,
    extra=(),
) -> list[str]:
    """`sgl-eval run` command for an accuracy-gate eval ([[RFC-0001:C-QUALITY-GATE]]).

    sgl-eval speaks an OpenAI-compatible endpoint, so the base URL is normalized to end in
    /v1; results land in a timestamped run folder under out_dir as metrics.json.
    """
    url = base_url.rstrip("/")
    if not url.endswith("/v1"):
        url = url + "/v1"
    cmd = [
        "sgl-eval", "run", name,
        "--base-url", url,
        "--num-examples", str(num_examples),
        "--num-threads", str(num_threads),
        "--out-dir", out_dir,
        "--no-dump-predictions",
    ]
    if max_tokens is not None:
        cmd += ["--max-tokens", str(max_tokens)]
    if temperature is not None:
        cmd += ["--temperature", str(temperature)]
    if model is not None:
        cmd += ["--model", model]
    cmd += list(extra)
    return cmd


def parse_gsm8k_metrics(text: str, metric: str = "accuracy") -> dict[str, float]:
    """Map an sgl-eval metrics.json into the gate metric vocabulary.

    The headline accuracy is `aggregate.score`; falls back to aggregate[metric]/mean.
    """
    data = json.loads(text)
    agg = data.get("aggregate", {})
    for k in ("score", metric, "accuracy", "mean"):
        if k in agg:
            return {metric: float(agg[k])}
    raise ValueError("sgl-eval metrics.json has no aggregate score")


def _latest_metrics_json(out_dir: str) -> Path:
    paths = sorted(Path(out_dir).glob("**/metrics.json"), key=lambda p: p.stat().st_mtime)
    if not paths:
        raise FileNotFoundError(f"no metrics.json under {out_dir}")
    return paths[-1]


class GSM8KEvaluator:
    """AccuracyEvaluator over `sgl-eval run gsm8k` ([[RFC-0001:C-QUALITY-GATE]])."""

    tool = "sgl-eval:gsm8k"

    def __init__(
        self,
        base_url: str,
        *,
        metric: str = "accuracy",
        num_examples: int = GSM8K_DEFAULT_EXAMPLES,
        num_threads: int = GSM8K_DEFAULT_THREADS,
        max_tokens: int | None = None,
        temperature: float | None = None,
        model: str | None = None,
        out_dir: str | None = None,
        run_eval: Callable[[list[str]], None] = _run_cmd,
    ) -> None:
        self.base_url = base_url
        self.metric = metric
        self.num_examples = num_examples
        self.num_threads = num_threads
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.model = model
        self.out_dir = out_dir
        self._run_eval = run_eval

    def evaluate(self) -> dict[str, float]:
        out = self.out_dir
        tmp = None
        if out is None:
            tmp = tempfile.TemporaryDirectory()
            out = tmp.name
        try:
            cmd = build_gsm8k_cmd(
                self.base_url, out,
                num_examples=self.num_examples,
                num_threads=self.num_threads,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                model=self.model,
            )
            self._run_eval(cmd)
            return parse_gsm8k_metrics(Path(_latest_metrics_json(out)).read_text(), self.metric)
        finally:
            if tmp is not None:
                tmp.cleanup()


SERVING_DEFAULT_BACKEND = "sglang"


def build_serving_cmd(
    base_url: str,
    point: WorkloadPoint,
    output_file: str,
    *,
    backend: str = SERVING_DEFAULT_BACKEND,
    dataset: str = "random",
    num_prompts: int | None = None,
    request_rate="inf",
    range_ratio: float = 1.0,
    tokenizer: str | None = None,
    extra=(),
) -> list[str]:
    """`bench_serving` command for one workload point ([[RFC-0001:C-OBJECTIVE]] transport).

    Concurrency maps to --max-concurrency; isl/osl to --random-input/output-len; range_ratio
    1.0 fixes sampled lengths to exactly isl/osl. num_prompts defaults to a few multiples of
    concurrency.
    """
    n = num_prompts if num_prompts is not None else max(point.concurrency * 5, 8)
    cmd = [
        "python", "-m", "sglang.bench_serving",
        "--backend", backend,
        "--base-url", base_url,
        "--dataset-name", dataset,
        "--random-input-len", str(point.isl),
        "--random-output-len", str(point.osl),
        "--random-range-ratio", str(range_ratio),
        "--num-prompts", str(n),
        "--max-concurrency", str(point.concurrency),
        "--request-rate", str(request_rate),
        "--output-file", output_file,
    ]
    if tokenizer is not None:
        cmd += ["--tokenizer", tokenizer]
    cmd += list(extra)
    return cmd


def parse_serving_metrics(text: str, point: WorkloadPoint) -> dict[str, float]:
    """Map a `bench_serving` result line into the canonical metric vocabulary.

    Decode per-token latency is the inter-token latency (ITL); steady-state decode
    throughput is derived from ITL (concurrency / ITL), not the OSL-blended
    output_throughput, per [[RFC-0001:C-OBJECTIVE]]. TTFT is carried report-only.
    """
    records = parse_jsonl(text)
    if not records:
        raise ValueError("bench_serving produced no result line")
    rec = records[-1]
    out: dict[str, float] = {}
    itl = rec.get("median_itl_ms")
    if itl is not None and float(itl) > 0:
        out[M.PER_TOKEN_MS] = float(itl)
        out[M.DECODE_THROUGHPUT] = float(point.concurrency) * 1000.0 / float(itl)
    if rec.get("p95_itl_ms") is not None:
        out[M.PER_TOKEN_P95_MS] = float(rec["p95_itl_ms"])
    if rec.get("median_ttft_ms") is not None:
        out[M.TTFT_MS] = float(rec["median_ttft_ms"])
    if rec.get("output_throughput") is not None:
        out[M.OUTPUT_THROUGHPUT] = float(rec["output_throughput"])
    return out


class BenchServingClient(_SubprocessBenchClient):
    """BenchClient over `sglang.bench_serving` — percentile-ITL frontier transport.

    A richer alternative to BenchOneBatchClient; reconcile against it before relying on its
    numbers ([[RFC-0001:C-BASELINE-ANCHOR]]).
    """

    tool = "bench_serving"

    def __init__(
        self,
        base_url: str,
        *,
        backend: str = SERVING_DEFAULT_BACKEND,
        tokenizer: str | None = None,
        num_prompts: int | None = None,
        request_rate="inf",
        dataset: str = "random",
        extra_args=(),
        run_bench: Callable[[list[str], str], str] = _default_run_bench,
    ) -> None:
        super().__init__(run_bench)
        self.base_url = base_url
        self.backend = backend
        self.tokenizer = tokenizer
        self.num_prompts = num_prompts
        self.request_rate = request_rate
        self.dataset = dataset
        self.extra_args = tuple(extra_args)

    def _bench_cmd(self, point: WorkloadPoint, result_path: str) -> list[str]:
        return build_serving_cmd(
            self.base_url, point, result_path,
            backend=self.backend, dataset=self.dataset,
            num_prompts=self.num_prompts, request_rate=self.request_rate,
            tokenizer=self.tokenizer, extra=self.extra_args,
        )

    def _parse(self, text: str, point: WorkloadPoint) -> dict[str, float]:
        return parse_serving_metrics(text, point)
