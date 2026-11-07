#!/usr/bin/env bash
# D0t: wait until no corpus-build step is running, then one memguarded window of self-traces over every prompt pool.
# A refused window (another large process) is retried every 5 min; SELFTRACE_ALL_DONE is written only after a window ran.
cd "$(dirname "$0")/../.."
busy() { for pid in $(pgrep -x python3; pgrep -x python); do
           tr '\0' ' ' < /proc/$pid/cmdline 2>/dev/null | grep -qE 's0[2-9]_|ops/dom/(prep|passes)' && return 0; done; return 1; }
for attempt in $(seq 1 48); do
    while busy; do sleep 60; done
    echo "[$(date -Is)] corpus idle; opening self-trace window (attempt $attempt)"
    out=$(SELF_TOKENS=${SELF_TOKENS:-40000} ops/dom/server_window.sh selftraces 2>&1); echo "$out"
    grep -q 'WINDOW_DONE' <<<"$out" && { echo "SELFTRACE_ALL_DONE"; exit 0; }
    sleep 300
done
echo "SELFTRACE_ALL_GAVE_UP"
