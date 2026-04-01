#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$ROOT/logs"
LOG_FILE="$LOG_DIR/desktop-launch.log"
UNIT_NAME="woofalytics.service"
MAIN_SCRIPT="$ROOT/main.py"

mkdir -p "$LOG_DIR"

log() {
    printf '%s %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$1" >>"$LOG_FILE"
}

log "Stopping Woofalytics user service"
systemctl --user stop "$UNIT_NAME" >/dev/null 2>&1 || true
pkill -f "$MAIN_SCRIPT" >/dev/null 2>&1 || true
sleep 1
pkill -9 -f "$MAIN_SCRIPT" >/dev/null 2>&1 || true
log "Woofalytics stop script finished"
