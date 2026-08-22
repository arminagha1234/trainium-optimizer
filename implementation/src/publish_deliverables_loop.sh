#!/usr/bin/env bash
#
# publish_deliverables_loop.sh — durable wrapper around publish_deliverables.py.
#
# Mirrors bank_publish_loop.sh: single-instance flock, sleep ~120s between runs,
# honest logging to a file, deploy-key-only push. The Python does the real work
# (collect verified models -> refresh leaderboard -> commit + push, no-op when
# unchanged). This wrapper just keeps it alive and serialized.
#
# Least privilege: pushes with a repo-scoped SSH deploy key ONLY (never a PAT),
# and the Python only ever stages optimized_models/**, LEADERBOARD.md, and the
# README's marked leaderboard table.
#
# Env overrides (with box defaults):
#   PUB_REPO_DIR              separate publish checkout   (default /home/ubuntu/trainium-optimizer-publish)
#   PUB_DEPLOY_KEY            repo-write SSH deploy key    (default /home/ubuntu/.ssh/gh_optimized_models_deploy)
#   PUB_OPTIMIZED_MODELS_DIR  live optimized_models dir    (default /home/ubuntu/trainium-optimizer/optimized_models)
#   PUB_SSH_URL               git@github.com:arminagha1234/trainium-optimizer.git
#   PUB_BRANCH                main
#   PUB_INTERVAL              seconds between cycles       (default 120)
#   PUB_LOG                   log file                     (default /home/ubuntu/publish_deliverables.log)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INTERVAL="${PUB_INTERVAL:-120}"
LOG="${PUB_LOG:-/home/ubuntu/publish_deliverables.log}"
RUN_LOCK="${PUB_RUN_LOCK:-/tmp/publish_deliverables_loop.lock}"

log() { echo "[loop $(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "$LOG"; }

# Single-instance: refuse to start a second loop.
exec 9>"$RUN_LOCK"
if ! flock -n 9; then
  echo "another publish_deliverables_loop is already running (lock: $RUN_LOCK)" >&2
  exit 0
fi

log "starting publish loop (interval=${INTERVAL}s, repo=${PUB_REPO_DIR:-default})"
while true; do
  if python3 "$HERE/publish_deliverables.py" >>"$LOG" 2>&1; then
    log "cycle ok"
  else
    log "cycle FAILED (see log above) — will retry next interval"
  fi
  sleep "$INTERVAL"
done
