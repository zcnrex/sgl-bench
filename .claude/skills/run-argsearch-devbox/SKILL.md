---
name: run-argsearch-devbox
description: Run the sgl-bench SGLang server-arg search (argsearch-run) on a RadixArk B200/B300 devbox — sync the repo, launch the search detached so it survives ssh/proxy drops, poll, and tear down. Use when asked to run an OFAT/grid sweep, a perf or accuracy-gated search, or any argsearch-run/bench against a devbox (e.g. chunan-zeng-b300-4gpu). Captures the non-obvious devbox traps (reserved port, paths, kill discipline).
---

# Run the arg search on a RadixArk devbox

`argsearch-run` drives the whole outer/inner search (per-config server launch → bench →
optional gsm8k gate → shutdown). Run it **detached on the devbox** so an `rx`/ssh disruption
never kills the sweep; poll from your side.

## Fast path

```bash
scripts/devbox_sweep.sh --config configs/nemotron_v3_ultra.yaml --branch nvfp4 \
    --mode ofat --isl-osl 8192x256 --concurrency 1 8 32
```
Extra args pass straight to `argsearch-run`. The script syncs the repo, tears down any stray
server, launches detached (`--port 40000`, results to `/sgl-workspace/sweep-out`), polls
reconnect-tolerantly, and fetches results to `out/sweep/`. Env overrides: `DEVBOX`, `PORT`,
`OUT`, `POLL_ITERS`. Re-running the script's poll section is safe — it reattaches.

## Devbox facts & traps (learned the hard way)

- **Port: use 40000, never 30000.** On k8s devboxes (`b300-verda-k8s`) port 30000 is
  platform-reserved; sglang's HTTP bind fails there with `[Errno 98] address already in use`
  *after* a full model load, even on a verified-free port. No process kill frees it.
- **Paths:** model cache at `/scratch/huggingface/hub` → set `HF_HOME=/scratch/huggingface`;
  put logs/outputs under `/sgl-workspace`. Model id `nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4`
  resolves from cache.
- **Connection:** `rx devbox exec` needs a TTY → use `rx devbox run` (one-shot) or
  `rsync`/`ssh -n` via `rx devbox ssh-config <name>`. `ssh -n` keeps detached launches from
  hanging the channel.
- **Detach correctly:** `nohup … > log 2>&1 &` survives an interrupted/rejected tool call —
  so always `pgrep -af sglang` before launching to avoid a double-launch (two 550B servers
  fighting over the GPUs). Record the pid (`echo $! > …pid`).
- **Teardown:** kill by explicit PID + `pkill -TERM -P <master>`; do NOT mass-kill via
  `nvidia-smi`-enumerated PIDs (blocked on shared boxes) and avoid `pkill -f sglang` (self-kill
  trap — your own kill command matches the pattern).

## Manual procedure (if not using the script)

1. `rsync -az --exclude .git --exclude out … ./ <devbox>:/sgl-workspace/sgl-bench/` then
   `ssh -n <devbox> 'pip install -e /sgl-workspace/sgl-bench -q'` (editable; later syncs are live).
2. `ssh -n <devbox> 'cd /sgl-workspace/sgl-bench && HF_HOME=/scratch/huggingface nohup
   python -m sglbench.argsearch.run --config … --branch nvfp4 --mode ofat --port 40000
   --out /sgl-workspace/sweep-out --isl-osl 8192x256 --concurrency 1 8 32 > /sgl-workspace/sweep.log 2>&1 &'`
3. Poll: `ssh -n <devbox> 'tail -5 /sgl-workspace/sweep.log; tail -1 /sgl-workspace/sweep-out/results.jsonl'`
   until `wrote N records`. Results stream per config, so a crash keeps finished configs.
4. Frontier: `ssh -n <devbox> 'cd /sgl-workspace/sgl-bench && python -m sglbench.argsearch.objective
   --config configs/nemotron_v3_ultra.yaml --results /sgl-workspace/sweep-out/results.jsonl --no-save'`.

## Useful argsearch-run flags

- `--mode {ofat,grid}`, `--limit-configs N`, `--isl-osl ISLxOSL …`, `--concurrency C …`
- `--transport {one-batch,serving}` — bench_one_batch_server (anchor) vs bench_serving
  (percentile ITL; uses num_prompts ≥ 5× concurrency for steady state)
- `--gsm8k-examples N` — enable the accuracy gate (per-config; auto-skipped for
  accuracy-invariant-only searches), `--frontier` to print the ranked frontier
- `--dry-run` — print the launch/bench commands without running
