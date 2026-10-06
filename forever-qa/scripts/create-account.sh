#!/usr/bin/env bash
# create-account.sh <email> <password> : bnet account + GM level 3 on realm 70 (via the worldserver console)
source "$(dirname "$0")/common.sh"
e="${1:?email}"; p="${2:?password}"
send(){ screen -S foreverqa -p 0 -X stuff "$1"$'\n'; sleep 2; }
send "bnetaccount create $e $p"
send "account set gmlevel ${e%%@*}#1 3 70" 2>/dev/null || true
log "Check $LOGS/world.out. The game account is normally named <bnetid>#1; adjust it if the gmlevel line failed:"
log "  mysql: SELECT id,username FROM auth.account;   then console: account set gmlevel <username> 3 -1"
