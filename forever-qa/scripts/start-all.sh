#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"
brew services start mysql >/dev/null
cd "$BIN"
pgrep -f "$BIN/bnetserver" >/dev/null || { nohup "$BIN/bnetserver" -c "$ETC/bnetserver.conf" >>"$LOGS/bnet.out" 2>&1 & log "bnetserver pid $!"; }
pgrep -f "$BIN/worldserver" >/dev/null && { log "worldserver already running"; exit 0; }
log "worldserver starting in a screen session (attach with: screen -r foreverqa)"
screen -dmS foreverqa bash -c "cd '$BIN' && '$BIN/worldserver' -c '$ETC/worldserver.conf' 2>&1 | tee -a '$LOGS/world.out'"
