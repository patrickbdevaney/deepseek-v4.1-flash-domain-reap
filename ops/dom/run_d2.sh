#!/usr/bin/env bash
# D2 CPU selection over every tier-1 bucket whose D1 pass has completed (p1, then t2, then t3).
# Resumable and incremental: per-layer units already computed are reused; a new pass adds its buckets, and the
# pieces that depend on the set of passes (super-expert union, stability shrink pool, polymath, overlap) are redone.
# CPU only, never touches the GPU; capped at 6 GiB, nice 19, every process stays well under 4 GiB (~0.3 GiB each).
#
#   ops/dom/run_d2.sh                 # run now on whatever passes are complete
#   ops/dom/run_d2.sh --wait t3       # wait until pass t3 is complete (s03 complete + 40 layers), then run
#   setsid nohup ops/dom/run_d2.sh --wait t3 > ops/logs/d2_select.log 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")/../.."
W=${V41_WORK:-$HOME/v41-work}
log() { echo "[$(date -Is)] run_d2: $*"; }
if [ "${1:-}" = --wait ]; then
    P=$2; shift 2
    until grep -q 's03 complete' "$W/dom/s03_$P/s03.log" 2>/dev/null && [ -e "$W/dom/s03_$P/bank/L39.npz" ]; do
        log "waiting for pass $P"; sleep 600; done
fi
log "start"
systemd-run --user --scope -q -p MemoryMax=6G nice -n 19 .venv/bin/python ops/dom/select_dom.py all --procs "${PROCS:-6}" "$@"
rc=$?
log "D2_SELECT_DONE rc=$rc"
exit $rc
