#!/usr/bin/env bash
# P: build the 384-expert K-quant pool (docs/domain/P_POOL.md). Detached, memory-guarded, resumable per layer.
#   1. optional chaining: wait for a pid to exit (--after-pid PID) and/or a line in a log (--after-log FILE REGEX)
#   2. guard (same conditions as run_d1_passes.sh): memguard active, no inference servers,
#      no GPU_WINDOW.lock, no other process over 4 GiB RSS, NVMe >= 90 GB free, MemAvailable >= 108 GiB
#      after drop_caches
#   3. take $V41_WORK/GPU_WINDOW.lock, build the pool corpus (CPU, once; needs every D1 pass complete),
#      run `s07 pool` (GPU units, one layer per process; finished files move to the cold store $V41_COLD), release the lock
# Re-running resumes at the first unfinished layer. Extra args after `--` go to `s07 pool`
# (e.g. -- --stop-after-layer 9 for a bounded window).
#
#   setsid nohup ops/dom/run_pool.sh > ops/logs/dom_pool.log 2>&1 < /dev/null &
#   setsid nohup ops/dom/run_pool.sh --after-log ops/logs/dom_after_d1.log REPLAY_DONE > ops/logs/dom_pool.log 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")/../.."
W=${V41_WORK:-$HOME/v41-work}
PY=.venv/bin/python
export V41_DOM=1 V41_DOM_PASS=pool
log() { echo "[$(date -Is)] run_pool: $*"; }
avail() { awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo; }
reclaim() { sync; echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null 2>&1 || true; }

AFTER_PID="" AFTER_LOG="" AFTER_RE="" EXTRA=()
while [ $# -gt 0 ]; do
    case "$1" in
        --after-pid) AFTER_PID=$2; shift 2;;
        --after-log) AFTER_LOG=$2 AFTER_RE=$3; shift 3;;
        --) shift; EXTRA=("$@"); break;;
        *) echo "unknown arg $1"; exit 2;;
    esac
done

if [ -n "$AFTER_PID" ]; then log "waiting for pid $AFTER_PID"; while kill -0 "$AFTER_PID" 2>/dev/null; do sleep 60; done; fi
if [ -n "$AFTER_LOG" ]; then
    log "waiting for /$AFTER_RE/ in $AFTER_LOG"
    until grep -qE "$AFTER_RE" "$AFTER_LOG" 2>/dev/null; do sleep 300; done
fi
if grep -q 'POOL384 complete' ops/logs/s07_pool.log 2>/dev/null && [ -f "$W/thor/POOL384/thor.json" ] && \
   $PY -c "import json,sys; sys.exit(0 if json.load(open('$W/thor/POOL384/thor.json')).get('complete') else 1)"; then
    log "POOL384 already complete"; log "POOL_DONE"; exit 0
fi

guard() {
    while :; do
        local why=""
        systemctl --user is-active --quiet dsv41-memguard || why="memguard inactive"
        for p in v41_server v41_decode llama-server llama-cli llama-mtmd-cli vllm; do
            pgrep -x "$p" >/dev/null && why="$p running"; done
        [ -e "$W/GPU_WINDOW.lock" ] && why="GPU_WINDOW.lock held: $(cat "$W/GPU_WINDOW.lock")"
        big=$(ps -eo pid=,rss=,comm= | awk '$2 > 4194304 {print $1":"$3}'); [ -n "$big" ] && why="large process $big"
        free_gb=$(df -BG --output=avail "$W" | tail -1 | tr -dc 0-9); [ "$free_gb" -ge 90 ] || why="only ${free_gb} GB NVMe free"
        mountpoint -q "${V41_COLD:-/mnt/v41-cold}" || why="cold store ${V41_COLD:-/mnt/v41-cold} not mounted (pool store)"
        if [ -z "$why" ]; then reclaim; [ "$(avail)" -ge 108 ] && return 0; why="MemAvailable $(avail) GiB"; fi
        log "waiting: $why"; sleep 300
    done
}

guard
echo "$$ $(date -Is) P pool384" > "$W/GPU_WINDOW.lock"
trap 'grep -q "^$$ " "$W/GPU_WINDOW.lock" 2>/dev/null && rm -f "$W/GPU_WINDOW.lock"; reclaim' EXIT
log "lock taken; NVMe free $(df -h --output=avail "$W" | tail -1 | tr -d ' '), MemAvailable $(avail) GiB"

if [ ! -f "$W/dom/corpus_pool/manifest.json" ]; then
    log "building the pool corpus"
    nice -n 10 $PY ops/dom/pool_corpus.py build >> ops/logs/s07_pool.log 2>&1 < /dev/null || { log "pool corpus FAILED (see ops/logs/s07_pool.log)"; exit 1; }
fi
log "s07 pool start ${EXTRA[*]:-}"
$PY ops/s07_quant.py pool "${EXTRA[@]}" >> ops/logs/s07_pool.log 2>&1 < /dev/null
rc=$?
log "s07 pool rc=$rc; NVMe free $(df -h --output=avail "$W" | tail -1 | tr -d ' ')"
if [ $rc -ne 0 ]; then log "POOL FAILED rc=$rc (see ops/logs/s07_pool.log; re-run resumes)"; exit $rc; fi
grep -q 'POOL384 complete' ops/logs/s07_pool.log && log "POOL_DONE" || log "POOL_PARTIAL (stopped early; re-run resumes)"
