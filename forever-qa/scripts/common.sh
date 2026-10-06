#!/usr/bin/env bash
# Shared settings. Everything lives under $QA (default ~/ForeverQA). Local only.
set -euo pipefail
QA="${FOREVERQA_ROOT:-$HOME/ForeverQA}"
SRC="$QA/source"; BUILD="$QA/build"; SERVER="$QA/server"; DATA="$QA/data"; LOGS="$QA/logs"; BACKUPS="$QA/backups"
BIN="$SERVER/bin"; ETC="$SERVER/etc"
COMMIT_PIN="${FOREVER_COMMIT:-3f77d65e993ab9045c6ea6286f6fe79526f2befb}"
DBUSER=foreverqa; DBS=(auth characters world hotfixes)
KIT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
log(){ echo "[$(date '+%F %T')] $*" | tee -a "$LOGS/qa.log" >&2; }
die(){ log "ERROR: $*"; exit 1; }
mkdir -p "$QA"/{source,build,server,data,database,logs,scripts,backups,notes}
mysql_qa(){ mysql --defaults-extra-file="$QA/database/my.cnf" "$@"; }
