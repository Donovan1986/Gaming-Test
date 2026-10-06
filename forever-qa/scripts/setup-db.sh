#!/usr/bin/env bash
# Phase 4: localhost-only MySQL, a dedicated user and empty DBs. worldserver populates them on first start.
source "$(dirname "$0")/common.sh"
brew services start mysql
grep -q "bind-address" "$(brew --prefix)/etc/my.cnf" 2>/dev/null || printf '[mysqld]\nbind-address = 127.0.0.1\nmysqlx-bind-address = 127.0.0.1\n' >> "$(brew --prefix)/etc/my.cnf"
PW_FILE="$QA/database/.pw"; [[ -f "$PW_FILE" ]] || { openssl rand -hex 16 > "$PW_FILE"; chmod 600 "$PW_FILE"; }
PW="$(cat "$PW_FILE")"
printf '[client]\nuser=%s\npassword=%s\nhost=127.0.0.1\n' "$DBUSER" "$PW" > "$QA/database/my.cnf"; chmod 600 "$QA/database/my.cnf"
SQL="CREATE USER IF NOT EXISTS '$DBUSER'@'localhost' IDENTIFIED BY '$PW'; CREATE USER IF NOT EXISTS '$DBUSER'@'127.0.0.1' IDENTIFIED BY '$PW';"
for d in "${DBS[@]}"; do SQL+="CREATE DATABASE IF NOT EXISTS \`$d\` DEFAULT CHARSET utf8mb4 COLLATE utf8mb4_unicode_ci; GRANT ALL ON \`$d\`.* TO '$DBUSER'@'localhost'; GRANT ALL ON \`$d\`.* TO '$DBUSER'@'127.0.0.1';"; done
mysql -uroot -e "$SQL" || die "root login failed (if root has a password: mysql -uroot -p)"
log "DBs ready. Put TDB_full_world_1210.26091_2026_09_09.sql + TDB_full_hotfixes_... (forever release assets) in $BIN"
