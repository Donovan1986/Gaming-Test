#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"
screen -S foreverqa -p 0 -X stuff $'server shutdown 5\n' 2>/dev/null && log "worldserver: shutdown 5s sent"
sleep 8; pkill -f "$BIN/worldserver" 2>/dev/null || true; pkill -f "$BIN/bnetserver" && log "bnetserver stopped" || true
