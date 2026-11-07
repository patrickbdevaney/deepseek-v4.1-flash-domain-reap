#!/usr/bin/env bash
# D1 driver: after the post-trace corpus rebuild, split the corpus into passes and run s03 for each pass, strictly one
# at a time, memory-guarded, then merge the pass banks. Holds $V41_WORK/GPU_WINDOW.lock for every pass so no CPU job
# (run_d0, agents) loads beside it. Resumable: a pass whose log already says "s03 complete" is skipped.
#
#   setsid nohup ops/dom/run_d1.sh > ops/logs/dom_run_d1.log 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")/../.."
W=${V41_WORK:-$HOME/v41-work}
PY=.venv/bin/python
SEA=${V41_ENGRAM_DIR:?set V41_ENGRAM_DIR to the directory holding the teacher Engram shards (README)}
export V41_DOM=1
log() { echo "[$(date -Is)] run_d1: $*"; }
avail() { awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo; }
reclaim() { sync; echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null 2>&1 || true; }

# 1. the corpus rebuild after the self-trace window (run_d0.sh logging to $D0_LOG) must have finished cleanly
D0_LOG=${D0_LOG:-ops/logs/dom_run_d0_rebuild.log}
until grep -q 'run_d0: done' "$D0_LOG" 2>/dev/null; do sleep 120; done
if grep -q FAILED "$D0_LOG"; then log "rebuild had FAILED steps; not starting D1"; exit 1; fi
log "corpus rebuilt"

$PY ops/dom/passes.py split || { log "split failed"; exit 1; }
$PY ops/dom/passes.py plan

guard() {   # same conditions as server_window.sh, plus NVMe space for one pass
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

for P in $($PY -c "from ops.dom import corpus_spec_dom as s; print(' '.join(s.PASSES))"); do
    D=$W/dom/s03_$P; mkdir -p "$D"
    if grep -q 's03 complete' "$D/s03.log" 2>/dev/null; then log "$P already complete"; continue; fi
    guard
    echo "$$ $(date -Is) d1 $P" > "$W/GPU_WINDOW.lock"
    log "$P start"
    V41_DOM_PASS=$P $PY ops/s03_saliency.py run --kernels tilelang --engram-dir "$SEA" >> "$D/s03.log" 2>&1 < /dev/null
    rc=$?
    rm -f "$W/GPU_WINDOW.lock"; reclaim
    if ! grep -q 's03 complete' "$D/s03.log"; then log "$P FAILED rc=$rc (see $D/s03.log)"; exit 1; fi
    V41_DOM_PASS=$P $PY ops/s03_saliency.py status | tail -5
    $PY ops/dom/passes.py retire "$P" || { log "retire $P failed"; exit 1; }
    log "$P done; NVMe free $(df -h --output=avail "$W" | tail -1)"
done
$PY ops/dom/passes.py merge && log "D1_DONE" || { log "merge failed"; exit 1; }
