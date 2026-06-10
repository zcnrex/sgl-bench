#!/usr/bin/env bash
# Resilient driver for an sgl-bench arg search on a RadixArk devbox.
#
# The sweep (argsearch-run) runs DETACHED on the devbox and manages the per-config
# server lifecycle itself; this script only syncs, kicks it off, and polls. An ssh / rx
# proxy drop does not touch the running sweep — re-run the poll (or this script's poll
# section) to reattach. Results are streamed to results.jsonl per config, so a mid-sweep
# crash keeps completed configs.
#
# Usage:
#   scripts/devbox_sweep.sh --config configs/nemotron_v3_ultra.yaml --branch nvfp4 \
#       --mode ofat --isl-osl 8192x256 --concurrency 1 8 32
#   (any extra args are passed straight through to `python -m sglbench.argsearch.run`)
#
# Env overrides: DEVBOX, PORT (default 40000 — 30000 is platform-reserved), REMOTE_DIR,
#   HF_HOME_REMOTE (default /scratch/huggingface), OUT, LOG, POLL_ITERS, POLL_SECS.
set -euo pipefail

DEVBOX="${DEVBOX:-chunan-zeng-b300-4gpu}"
PORT="${PORT:-40000}"
REMOTE_DIR="${REMOTE_DIR:-/sgl-workspace/sgl-bench}"
HF_HOME_REMOTE="${HF_HOME_REMOTE:-/scratch/huggingface}"
OUT="${OUT:-/sgl-workspace/sweep-out}"
LOG="${LOG:-/sgl-workspace/sweep.log}"
POLL_ITERS="${POLL_ITERS:-360}"
POLL_SECS="${POLL_SECS:-10}"
LOCAL_DIR="$(cd "$(dirname "$0")/.." && pwd)"

[ "$#" -gt 0 ] || { echo "error: pass argsearch-run args (e.g. --config ... --branch ... --mode ofat)"; exit 2; }

retry() {
  local n=0 max="${RETRIES:-5}"
  until "$@"; do
    n=$((n + 1)); [ "$n" -ge "$max" ] && { echo "[devbox_sweep] gave up after $max tries: $*"; return 1; }
    echo "[devbox_sweep] transient failure (try $n/$max), retrying in 5s: $1 …"; sleep 5
  done
}

echo "[devbox_sweep] sync $LOCAL_DIR -> $DEVBOX:$REMOTE_DIR"
retry rsync -az --exclude '.git' --exclude 'out' --exclude '__pycache__' --exclude '*.egg-info' \
  --exclude '.govctl' "$LOCAL_DIR"/ "$DEVBOX:$REMOTE_DIR/"
ssh -n "$DEVBOX" "pip install -e $REMOTE_DIR -q 2>&1 | tail -1" || true

echo "[devbox_sweep] tear down any stray server/sweep (best-effort)"
ssh -n "$DEVBOX" "bash -lc 'pkill -TERM -f \"sglbench.argsearch.run\" 2>/dev/null || true; M=\$(pgrep -f \"launch_server.*port $PORT\" | head -1); if [ -n \"\$M\" ]; then pkill -TERM -P \$M 2>/dev/null || true; kill -TERM \$M 2>/dev/null || true; fi; sleep 5; true'" || true

echo "[devbox_sweep] launch detached argsearch-run on port $PORT -> $LOG"
ssh -n "$DEVBOX" "cd $REMOTE_DIR && HF_HOME=$HF_HOME_REMOTE nohup python -m sglbench.argsearch.run --port $PORT --out $OUT $* > $LOG 2>&1 & echo \$! > /sgl-workspace/sweep.pid; echo started pid \$(cat /sgl-workspace/sweep.pid)" || echo "[devbox_sweep] launch ssh returned nonzero; verifying the nohup'd run (not relaunching)"
if retry ssh -n "$DEVBOX" "kill -0 \$(cat /sgl-workspace/sweep.pid 2>/dev/null) 2>/dev/null"; then
  echo "[devbox_sweep] sweep process confirmed running"
else
  echo "[devbox_sweep] ERROR: sweep process not running after launch"; exit 1
fi

echo "[devbox_sweep] polling (safe to Ctrl-C / disconnect; the sweep runs detached)"
done=0
for i in $(seq 1 "$POLL_ITERS"); do
  status="$(ssh -n "$DEVBOX" "grep -qaE 'wrote .* records|Traceback|Error:|error while' $LOG 2>/dev/null && echo DONE || echo RUN; tail -1 $LOG 2>/dev/null | cut -c1-100" 2>/dev/null || echo 'RUN (ssh hiccup, retrying)')"
  echo "[poll $i] $status"
  case "$status" in *DONE*) done=1; break;; esac
  sleep "$POLL_SECS"
done

echo "[devbox_sweep] last log lines:"
ssh -n "$DEVBOX" "tail -25 $LOG" 2>/dev/null || true
FETCH_DIR="$LOCAL_DIR/out"
echo "[devbox_sweep] fetch per-model results -> $FETCH_DIR"
mkdir -p "$FETCH_DIR"
rsync -az "$DEVBOX:$OUT/" "$FETCH_DIR/" 2>/dev/null || true
[ "$done" = 1 ] && echo "[devbox_sweep] sweep complete" || echo "[devbox_sweep] poll window ended; sweep may still be running on the devbox (check $LOG)"
