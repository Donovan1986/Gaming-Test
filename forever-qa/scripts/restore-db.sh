#!/usr/bin/env bash
# restore-db.sh <backup-dir> [db...]   (takes a fresh safety backup first)
source "$(dirname "$0")/common.sh"
src="${1:?usage: restore-db.sh <backup dir> [db...]}"; shift; [[ -d "$src" ]] || die "no such dir"
pgrep -f "$BIN/worldserver" >/dev/null && die "stop the server first"
read -r -p "Restore ${*:-all DBs} from $src? type YES: " a; [[ "$a" == YES ]] || die "aborted"
"$(dirname "$0")/backup-db.sh"
for d in "${@:-${DBS[@]}}"; do gunzip -c "$src/$d.sql.gz" | mysql_qa "$d"; log "restored $d"; done
