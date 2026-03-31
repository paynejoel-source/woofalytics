#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/joel/woofalytics"
PID_FILE="$ROOT/.woofalytics-desktop.pid"
PYTHON_BIN="$ROOT/venv/bin/python"
MAIN_SCRIPT="$ROOT/main.py"
LOG_DIR="$ROOT/logs"
LOG_FILE="$LOG_DIR/desktop-launch.log"

mkdir -p "$LOG_DIR"

log() {
    printf '%s %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$1" >>"$LOG_FILE"
}

if [[ -f "$PID_FILE" ]]; then
    pid="$(cat "$PID_FILE" 2>/dev/null || true)"
    if [[ -n "${pid}" ]] && kill -0 "$pid" >/dev/null 2>&1; then
        log "Stopping Woofalytics pid ${pid}"
        kill "$pid" >/dev/null 2>&1 || true
        sleep 2
        if kill -0 "$pid" >/dev/null 2>&1; then
            kill -9 "$pid" >/dev/null 2>&1 || true
        fi
    fi
    rm -f "$PID_FILE"
fi

pkill -f "$MAIN_SCRIPT" >/dev/null 2>&1 || true
sleep 1
pkill -9 -f "$MAIN_SCRIPT" >/dev/null 2>&1 || true
pkill -f "/usr/bin/ffmpeg -loglevel error -rtsp_transport tcp -i rtsp://127.0.0.1:8554/front_yard -vn -acodec pcm_s16le -ac 1 -ar 16000 -f s16le -" >/dev/null 2>&1 || true
log "Woofalytics stop script finished"
