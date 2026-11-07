#!/usr/bin/env bash
# D1 driver for an explicit pass order (priority buckets first). Same guards as run_d1.sh, no rebuild wait, no split.
# If an s03 pass started by an earlier driver is still running (--adopt PID PASS), wait for it, release its lock and
# retire it first. Stops after the listed passes; merge is a separate step (passes.py merge sums whatever banks exist).
#
#   setsid nohup ops/dom/run_d1_passes.sh --adopt <PID> p1 > ops/logs/dom_run_d1b.log 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")/../.."
W=${V41_WORK:-$HOME/v41-work}
PY=.venv/bin/python
SEA=${V41_ENGRAM_DIR:?set V41_ENGRAM_DIR to the directory holding the teacher Engram shards (README)}
export V41_DOM=1
log() { echo "[$(date -Is)] run_d1_passes: $*"; }
avail() { awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo; }
reclaim() { sync; echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null 2>&1 || true; }

finish() {   # $1 pass: check, retire
    local P=$1 D=$W/dom/s03_$1
    if ! grep -q 's03 complete' "$D/s03.log"; then log "$P FAILED (see $D/s03.log)"; exit 1; fi
    V41_DOM_PASS=$P $PY ops/s03_saliency.py status | tail -5
    $PY ops/dom/passes.py retire "$P" || { log "retire $P failed"; exit 1; }
    log "$P done; NVMe free $(df -h --output=avail "$W" | tail -1)"
}

if [ "${1:-}" = --adopt ]; then
    APID=$2 AP=$3; shift 3
    log "adopting $AP (pid $APID)"
    while kill -0 "$APID" 2>/dev/null; do sleep 60; done
    rm -f "$W/GPU_WINDOW.lock"; reclaim
    finish "$AP"
fi

guard() {   # same conditions as run_d1.sh
    while :; do
        local why=""
        systemctl --user is-active --quiet dsv41-memguard || why="memguard inactive"
        for p in v41_server v41_decode llama-server llama-cli llama-mtmd-cli vllm; do
            pgrep -x "$p" >/dev/null && why="$p running"; done
        [ -e "$W/GPU_WINDOW.lock" ] && why="GPU_WINDOW.lock held: $(cat "$W/GPU_WINDOW.lock")"
        big=$(ps -eo pid=,rss=,comm= | awk '$2 > 4194304 {print $1":"$3}'); [ -n "$big" ] && why="large process $big"
        free_gb=$(df -BG --output=avail "$W" | tail -1 | tr -dc 0-9); [ "$free_gb" -ge 110 ] || why="only ${free_gb} GB NVMe free"
        if [ -z "$why" ]; then reclaim; [ "$(avail)" -ge 108 ] && return 0; why="MemAvailable $(avail) GiB"; fi
        log "waiting: $why"; sleep 300
    done
}

for P in "$@"; do
    D=$W/dom/s03_$P; mkdir -p "$D"
    if grep -q 's03 complete' "$D/s03.log" 2>/dev/null; then log "$P already complete"; continue; fi
    guard
    echo "$$ $(date -Is) d1 $P" > "$W/GPU_WINDOW.lock"
    log "$P start"
    V41_DOM_PASS=$P $PY ops/s03_saliency.py run --kernels tilelang --engram-dir "$SEA" >> "$D/s03.log" 2>&1 < /dev/null
    rm -f "$W/GPU_WINDOW.lock"; reclaim
    finish "$P"
done
log "PASSES_DONE $*"
