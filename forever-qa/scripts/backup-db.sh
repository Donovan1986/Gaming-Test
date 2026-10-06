#!/usr/bin/env bash
# Timestamped dump; never overwrites an existing backup.
source "$(dirname "$0")/common.sh"
out="$BACKUPS/$(date +%Y%m%d-%H%M%S)"; mkdir "$out"
for d in "${DBS[@]}"; do mysqldump --defaults-extra-file="$QA/database/my.cnf" --single-transaction --routines "$d" | gzip > "$out/$d.sql.gz"; done
log "backup -> $out"; ls -lh "$out"
