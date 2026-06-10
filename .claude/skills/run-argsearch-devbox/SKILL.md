---
name: run-argsearch-devbox
description: Run the sgl-bench SGLang server-arg search (argsearch-run) on a  B200/B300/GB200 devbox — validate the config, sync the repo, launch the search detached so it survives ssh/proxy drops, poll, and tear down. Use when asked to run an OFAT/grid sweep, a perf or accuracy-gated search, or any argsearch-run/bench against a devbox (e.g. chunan-zeng-b300-4gpu). Captures the non-obvious traps (config pre-validation, per-box HF_HOME, reserved port, shell wrapping, kill discipline).
---

# Run the arg search on a devbox

`argsearch-run` drives the whole outer/inner search (per-config server launch → bench →
optional gsm8k gate → shutdown). Run it **detached on the devbox** so an `rx`/ssh disruption
never kills the sweep; poll from your side.

The flow is always: **(0) validate the config locally → (1) probe the box → (2) launch
detached → (3) poll → (4) frontier**. Steps 0 and 1 are where most time gets wasted if
skipped — do them before any launch.

## 0. Validate the config first (cheap, local, no GPU)

`argsearch-run --dry-run` loads + validates the config and prints the exact launch/bench
commands without running anything. Run it locally before touching a devbox — it catches the
config-shape errors that otherwise only surface after a sync:

```bash
python3 -m sglbench.argsearch.run --config <cfg> --branch <branch> --mode ofat \
    --isl-osl <ISL>x<OSL> --concurrency <C ...> --dry-run | head
```
Common validation failures (all raised at load time by `SearchConfig.model_validate`):
- **`baseline missing candidate '<name>'`** — the `baseline` block must give a value for
  *every* candidate, including ones you only intend to grid. Add the missing key.
- **`non-gridded candidates must be pinned … in focused_grid.pins`** — any candidate NOT in
  `focused_grid.args` must appear in `focused_grid.pins` (pin it to its OFAT-best, or baseline
  value pre-search). Coupled-pair candidates belong in `args`; independent ones get pinned.

The header line (`configs=N workload_points=… repeats=… env=…`) confirms how many configs the
mode actually expands to — OFAT only varies the *independently-toggleable* candidates, so
coupled pairs (e.g. ep/moe-a2a, dp/dp-attention) won't appear until you run `--mode grid`.

**Branch must match the hardware you have.** A branch is keyed by hardware target + KV-cache
precision (C-BRANCH), and that key is recorded in the measurement identity. If the only free
devbox is a different chip than the branch name claims (e.g. branch `gb200-…` but only a B300
box is up), do NOT run the mismatched branch — its numbers would be mislabelled. Add a sibling
branch keyed to the real hardware (copy the lever structure, fix `name`/`hardware`) and sweep
that instead.

## Fast path

```bash
HF_HOME_REMOTE=<box HF_HOME> scripts/devbox_sweep.sh \
    --config <cfg> --branch <branch> --mode ofat --isl-osl <ISL>x<OSL> --concurrency <C ...>
```
Extra args pass straight to `argsearch-run`. The script syncs the repo, tears down any stray
server, launches detached (`--port 8888`, results to `/sgl-workspace/sweep-out`), polls
reconnect-tolerantly, and fetches results to `out/`. Env overrides: `DEVBOX`, `PORT`,
`HF_HOME_REMOTE` (**must match the box — see below; the default may be wrong**), `OUT`,
`REMOTE_DIR`, `POLL_ITERS`, `POLL_SECS`. Re-running the script's poll section is safe — it
reattaches to the detached run.

## 1. Probe the box before launching

Run one combined probe and read it — don't assume paths. **`rx devbox run` execs its argv
directly, so a bare `'a; b'` string fails with `stat … no such file or directory`. Always wrap
in a shell:** `rx devbox run <box> -- bash -lc '<script>'`.

```bash
rx devbox run <box> -- bash -lc '
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
  echo "HF_HOME=$HF_HOME"                     # do NOT assume /scratch/huggingface
  pgrep -af "[l]aunch_server" || echo "no server"   # bracket avoids self-match (see traps)
  ss -ltn | grep :8888 || echo "8888 free"
  ls -d "$HF_HOME"/hub/models--*<model-slug>* 2>/dev/null || echo "model NOT cached"'
```

- **HF_HOME is per-box — discover it, never hardcode.** It varies (`/scratch/huggingface` on
  some boxes, `/cluster-storage/models` on others). Read `$HF_HOME` from the box, point the
  launch at the same value, and look for the model under `$HF_HOME/hub/models--<org>--<name>`.
  Checking the wrong cache path makes a present model look missing.
- **Verify the model is actually cached and complete** before launching: `du -sh
  $HF_HOME/hub/models--…/blobs` should be the real weight size (hundreds of GB for big MoE),
  the snapshot dir should have the expected `*.safetensors` shard count + `config.json` +
  `model.safetensors.index.json`, and `find …/blobs -name '*.incomplete'` should be empty.
  A partial download fails minutes into the launch, not at the start.

## Devbox facts & traps (learned the hard way)

- **`rx devbox run` needs an explicit shell.** `rx devbox run <box> -- bash -lc '<script>'`.
  `rx devbox exec` needs a TTY (don't use it for scripted probes). For sync/launch use
  `rsync` / `ssh -n` against the host from `rx devbox ssh-config <name>` (append it to
  `~/.ssh/config` once). `ssh -n` keeps detached launches from hanging the channel.
- **pgrep self-match trap.** `pgrep -af sglang` (or `… launch_server`) matches *its own*
  command line — including the `bash -lc '…sglang…'` wrapper you ran it in — so it always
  reports a false hit. Use a bracketed pattern (`pgrep -af "[l]aunch_server"`) so the regex
  can't match the literal string in its own argv, and confirm a hit is a real server (check
  the pid's age / GPU memory) before treating the box as occupied.
- **Detach correctly:** `nohup … > log 2>&1 &` survives an interrupted/rejected tool call — so
  always probe for an existing server (bracketed pgrep, above) before launching to avoid a
  double-launch (two big servers fighting over the GPUs). Record the pid (`echo $! > …pid`)
  and confirm `kill -0 <pid>` after launch.
- **Teardown:** kill by explicit PID + `pkill -TERM -P <master>`; do NOT mass-kill via
  `nvidia-smi`-enumerated PIDs (blocked on shared boxes) and avoid `pkill -f sglang` (self-kill
  trap — your own kill command matches the pattern).
- **Paths:** put logs/outputs under `/sgl-workspace`; sync the repo to
  `/sgl-workspace/sgl-bench` (`pip install -e` once, later rsyncs are live).

## Manual procedure (if not using the script)

1. `rsync -az --exclude .git --exclude out … ./ <devbox>:/sgl-workspace/sgl-bench/` then
   `ssh -n <devbox> 'pip install -e /sgl-workspace/sgl-bench -q'` (editable; later syncs are live).
2. `ssh -n <devbox> 'cd /sgl-workspace/sgl-bench && HF_HOME=<box HF_HOME> nohup
   python -m sglbench.argsearch.run --config <cfg> --branch <branch> --mode ofat --port 8888
   --out /sgl-workspace/sweep-out --isl-osl <ISL>x<OSL> --concurrency <C ...> > /sgl-workspace/sweep.log 2>&1 & echo $! > /sgl-workspace/sweep.pid'`
3. Poll: `ssh -n <devbox> 'tail -5 /sgl-workspace/sweep.log; tail -1 /sgl-workspace/sweep-out/results.jsonl'`
   until `wrote N records`. Results stream per config, so a crash keeps finished configs.
4. Frontier: `ssh -n <devbox> 'cd /sgl-workspace/sgl-bench && python -m sglbench.argsearch.objective
   --config <cfg> --results /sgl-workspace/sweep-out/results.jsonl --no-save'`.

## Useful argsearch-run flags

- `--mode {ofat,grid}`, `--limit-configs N`, `--isl-osl ISLxOSL …`, `--concurrency C …`
- `--transport {one-batch,serving}` — bench_one_batch_server (anchor) vs bench_serving
  (percentile ITL; uses num_prompts ≥ 5× concurrency for steady state)
- `--gsm8k-examples N` — enable the accuracy gate (per-config; auto-skipped for
  accuracy-invariant-only searches), `--frontier` to print the ranked frontier
- `--dry-run` — print the launch/bench commands without running (use it in step 0)
