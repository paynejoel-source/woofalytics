#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/joel/woofalytics"
PID_FILE="$ROOT/.woofalytics-desktop.pid"
MAIN_SCRIPT="$ROOT/main.py"
LOG_DIR="$ROOT/logs"
LOG_FILE="$LOG_DIR/desktop-launch.log"
LOCK_FILE="$ROOT/.woofalytics-desktop.lock"

mkdir -p "$LOG_DIR"
exec 9>"$LOCK_FILE"
flock 9

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
log "Woofalytics stop script finished"
