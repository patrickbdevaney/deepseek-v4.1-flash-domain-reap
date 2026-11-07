#!/usr/bin/env bash
# One GPU "server window" for the domain track: start the A192 spine (thor/A192 + NVFP4 dense
# overlay, DSpark K=3), run the requested jobs against it, stop it, reclaim. Memory-guarded:
# refuses to start beside any other model-loading process or with < 108 GiB available, the
# dsv41-memguard unit must be active, and page cache is dropped before and after.
#
#   setsid nohup ops/dom/server_window.sh selftraces > ops/logs/dom_window.log 2>&1 < /dev/null &
#
# Needs the parent repo's CUDA build (cuda/build/v41_server, A192 Thor build, NVFP4 overlay, Engram rows) and its
# dsv41-memguard user unit (ops/memguard.sh); see README.md "Dependencies on the parent repo".
#
# Jobs (in the order given):
#   selftraces  ops/dom/selftraces.py over every bucket with a prompt pool (D0t)
set -u
cd "$(dirname "$0")/../.."
W=${V41_WORK:-$HOME/v41-work}
PORT=${PORT:-8000}
LOG=ops/logs
avail() { awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo; }
reclaim() { sync; echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null 2>&1 || true; }

systemctl --user is-active --quiet dsv41-memguard || { echo "[window] memguard not active; refusing"; exit 1; }
for p in v41_server v41_decode llama-server llama-cli llama-mtmd-cli vllm; do
    pgrep -x "$p" >/dev/null && { echo "[window] $p running; refusing (unified memory)"; exit 1; }
done
# pipeline stages run as python3; check their argv (pgrep -x only, never pgrep -f)
for pid in $(pgrep -x python3; pgrep -x python); do   # venv interpreters are named python
    if tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null | grep -qE 's0[2-9]_|s1[01]_|run_stage|ops/dom/(prep|passes)'; then
        echo "[window] pipeline stage pid $pid running; refusing"; exit 1
    fi
done
# any other process already holding > 4 GiB (venv pythons are not named python3) blocks the window
big=$(ps -eo pid=,rss=,comm= | awk -v me=$$ '$2 > 4194304 && $1 != me {print $1":"$3}')
[ -n "$big" ] && { echo "[window] large process(es) running: $big; refusing"; exit 1; }
# advertise the window: other jobs must check this lock before any RAM-heavy step
LOCK=$W/GPU_WINDOW.lock
echo "$$ $(date -Is) $*" > "$LOCK"; trap 'rm -f "$LOCK"' EXIT
reclaim
a=$(avail); [ "$a" -ge 108 ] || { echo "[window] only $a GiB available; refusing"; exit 1; }

DSV41_LOG_RAW=${DSV41_LOG_RAW:-0} DSV4_NVFP4_OVERLAY=$W/thor/overlay_nvfp4 cuda/build/v41_server --ckpt "$W/thor/A192" \
    --keepset "$W/s04/keepset.json" --engram "$W/engram_rows" --seqmax "${SEQMAX:-65536}" \
    --host 127.0.0.1 --port "$PORT" --spec 3 --model DeepSeek-V4.1-Flash-1017-REAP192 \
    > "$LOG/dom_window_server.log" 2>&1 &
SP=$!
t0=$(date +%s)
for _ in $(seq 1 180); do
    curl -sf "http://127.0.0.1:$PORT/health" >/dev/null && break
    kill -0 "$SP" 2>/dev/null || { echo "[window] server died during load"; tail -20 "$LOG/dom_window_server.log"; exit 1; }
    sleep 5
done
echo "[window] server pid $SP ready after $(( $(date +%s) - t0 )) s; MemAvailable $(avail) GiB"

rc=0
for job in "$@"; do
    case "$job" in
        selftraces)
            mapfile -t B < <(ls "$W/dom/prompts/"*.jsonl 2>/dev/null | xargs -rn1 basename | sed 's/\.jsonl$//')
            if [ "${#B[@]}" -eq 0 ]; then echo "[window] no prompt pools yet; selftraces skipped"; continue; fi
            python3 ops/dom/selftraces.py --url "http://127.0.0.1:$PORT" --buckets "${B[@]}" \
                --tokens-per-bucket "${SELF_TOKENS:-40000}" > "$LOG/dom_selftraces.log" 2>&1
            echo "[window] selftraces rc=$? buckets=${B[*]}" ;;
        py:*)
            python3 "${job#py:}" "http://127.0.0.1:$PORT" > "$LOG/dom_window_$(basename "${job#py:}" .py).log" 2>&1
            echo "[window] ${job#py:} rc=$?" ;;
        *) echo "[window] unknown job $job" ;;
    esac
done

kill "$SP"; wait "$SP" 2>/dev/null
reclaim
echo "[window] done; MemAvailable $(avail) GiB; WINDOW_DONE rc=$rc"
