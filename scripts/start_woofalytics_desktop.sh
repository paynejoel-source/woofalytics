#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/joel/woofalytics"
PORT="${WOOF_PORT:-8015}"
OPEN_BROWSER="${WOOF_OPEN_BROWSER:-1}"
LOG_DIR="$ROOT/logs"
LOG_FILE="$LOG_DIR/desktop-launch.log"
URL="http://127.0.0.1:${PORT}"
UNIT_NAME="woofalytics.service"
UNIT_SOURCE="$ROOT/assets/$UNIT_NAME"
UNIT_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
UNIT_TARGET="$UNIT_DIR/$UNIT_NAME"
MAIN_SCRIPT="$ROOT/main.py"

mkdir -p "$LOG_DIR"
mkdir -p "$UNIT_DIR"

log() {
    printf '%s %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$1" >>"$LOG_FILE"
}

sync_unit() {
    if [[ ! -f "$UNIT_TARGET" ]] || ! cmp -s "$UNIT_SOURCE" "$UNIT_TARGET"; then
        install -m 0644 "$UNIT_SOURCE" "$UNIT_TARGET"
        systemctl --user daemon-reload
        log "Updated user unit ${UNIT_NAME}"
    fi

    systemctl --user enable "$UNIT_NAME" >/dev/null 2>&1 || true
}

is_healthy() {
    python3 - <<PY >/dev/null 2>&1
import json
from urllib.request import urlopen

with urlopen("${URL}/api/status", timeout=3) as response:
    payload = json.load(response)

if not payload.get("worker_alive"):
    raise SystemExit(1)

if not payload.get("capture_healthy"):
    raise SystemExit(1)
PY
}

start_server() {
    pkill -f "$MAIN_SCRIPT" >/dev/null 2>&1 || true
    sleep 1
    pkill -9 -f "$MAIN_SCRIPT" >/dev/null 2>&1 || true
    sleep 1
    log "Starting Woofalytics via systemd user service on port ${PORT}"
    systemctl --user restart "$UNIT_NAME"
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

sync_unit

if ! systemctl --user is-active --quiet "$UNIT_NAME" || ! is_healthy; then
    log "Woofalytics service not healthy; restarting"
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
