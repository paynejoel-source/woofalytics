#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/joel/woofalytics"
PORT="${WOOF_PORT:-8015}"
OPEN_BROWSER="${WOOF_OPEN_BROWSER:-1}"
LOG_DIR="$ROOT/logs"
LOG_FILE="$LOG_DIR/desktop-launch.log"
PID_FILE="$ROOT/.woofalytics-desktop.pid"
URL="http://127.0.0.1:${PORT}"
PYTHON_BIN="$ROOT/venv/bin/python"
MAIN_SCRIPT="$ROOT/main.py"

mkdir -p "$LOG_DIR"

log() {
    printf '%s %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$1" >>"$LOG_FILE"
}

is_healthy() {
    python3 - <<PY >/dev/null 2>&1
from urllib.request import urlopen
urlopen("${URL}/api/status", timeout=3).read()
PY
}

stop_existing() {
    if [[ -f "$PID_FILE" ]]; then
        pid="$(cat "$PID_FILE" 2>/dev/null || true)"
        if [[ -n "${pid}" ]] && kill -0 "$pid" >/dev/null 2>&1; then
            kill "$pid" >/dev/null 2>&1 || true
            sleep 2
        fi
        rm -f "$PID_FILE"
    fi

    pkill -f "$MAIN_SCRIPT" >/dev/null 2>&1 || true
    sleep 1
}

port_in_use() {
    ss -ltn | awk '{print $4}' | grep -q ":${PORT}$"
}

wait_for_port_release() {
    for _ in $(seq 1 10); do
        if ! port_in_use; then
            return 0
        fi
        sleep 1
    done
    return 1
}

start_server() {
    cd "$ROOT"
    log "Starting Woofalytics on port ${PORT}"
    nohup env WOOF_PORT="$PORT" "$PYTHON_BIN" "$MAIN_SCRIPT" >>"$LOG_FILE" 2>&1 &
    echo $! > "$PID_FILE"
}

wait_until_healthy() {
    for _ in $(seq 1 12); do
        if is_healthy; then
            return 0
        fi
        sleep 1
    done
    return 1
}

if ! is_healthy; then
    log "Existing Woofalytics instance not healthy; restarting"
    stop_existing
    if ! wait_for_port_release; then
        log "Port ${PORT} is still in use after stop attempt"
        exit 1
    fi
    start_server
    if ! wait_until_healthy; then
        log "Woofalytics failed to become healthy"
        exit 1
    fi
fi

if [[ "$OPEN_BROWSER" == "1" ]]; then
    log "Opening ${URL}"
    xdg-open "$URL" >/dev/null 2>&1 &
fi
