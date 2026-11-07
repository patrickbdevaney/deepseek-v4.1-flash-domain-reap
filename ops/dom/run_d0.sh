#!/usr/bin/env bash
# D0 builder with the Thor memory guard: every heavy step waits until no v41_server runs and MemAvailable >= 40 GiB.
# Usage (detached): setsid nohup ops/dom/run_d0.sh [bucket ...] > ops/logs/dom_run_d0.log 2>&1 < /dev/null &
# No bucket arguments = clone, then every pool in spec order, then corpus assembly.
set -u
cd "$(dirname "$0")/../.."
W=${V41_WORK:-$HOME/v41-work}
export V41_DOM=1 HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$W/dom/hf_datasets_cache}"
PY=.venv/bin/python

gate() {   # block until the box is free for a CPU/RAM-heavy step (coordinator HARD RULE: no GPU window lock,
           # no v41_server, MemAvailable >= 40 GiB)
  while [ -e "$W/GPU_WINDOW.lock" ] || pgrep -x v41_server >/dev/null || pgrep -x llama-server >/dev/null \
        || [ "$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)" -lt 40 ]; do
    sleep 60
  done
}

log() { echo "[$(date -Is)] run_d0: $*"; }

gate; log "clone"
$PY ops/dom/prep.py clone || log "clone returned $?"
gate; log "localrepos (OWNER_HOLDOUT repos, pre-cutoff git only; none in this release)"
$PY ops/dom/prep.py localrepos || log "localrepos returned $?"
if [ -n "${D0_THREADS:-}" ]; then   # e.g. D0_THREADS="gamedev robotics" after THREAD_REPOS changed
  gate; log "threads $D0_THREADS"
  $PY ops/dom/prep.py threads --domains $D0_THREADS || log "threads returned $?"
fi

if [ "$#" -gt 0 ]; then BUCKETS="$*"; else
  BUCKETS=$($PY -c "from ops.dom import corpus_spec_dom as s; print(' '.join(s.BUCKETS))"); fi
for b in $BUCKETS; do
  gate; log "pool $b"
  $PY ops/s02_calib_corpus.py --pools "$b" || log "pool $b FAILED ($?)"
done
if [ "$#" -eq 0 ]; then
  gate; log "assemble"
  $PY ops/s02_calib_corpus.py || log "assembly FAILED ($?)"
fi
log "done"
