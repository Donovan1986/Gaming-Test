#!/usr/bin/env bash
# Phase 6/14: write a local-only config from the .dist files (re-run safe; keeps a .bak).
source "$(dirname "$0")/common.sh"
PW="$(cat "$QA/database/.pw")"; mkdir -p "$ETC"
setc(){ local f=$1 k=$2 v=$3; if grep -q "^$k *=" "$f"; then sed -i '' "s|^$k *=.*|$k = $v|" "$f"; else echo "$k = $v" >> "$f"; fi; }
for s in worldserver bnetserver; do
  f="$ETC/$s.conf"; [[ -f "$f" ]] && cp "$f" "$f.bak"; cp "$ETC/$s.conf.dist" "$f"
  setc "$f" BindIP '"127.0.0.1"'; setc "$f" LogsDir "\"$LOGS\""; setc "$f" SourceDirectory "\"$SRC\""
  setc "$f" MySQLExecutable "\"$(brew --prefix mysql)/bin/mysql\""
  setc "$f" LoginDatabaseInfo "\"127.0.0.1;3306;$DBUSER;$PW;auth\""
done
W="$ETC/worldserver.conf"
setc "$W" WorldDatabaseInfo "\"127.0.0.1;3306;$DBUSER;$PW;world\""
setc "$W" CharacterDatabaseInfo "\"127.0.0.1;3306;$DBUSER;$PW;characters\""
setc "$W" HotfixDatabaseInfo "\"127.0.0.1;3306;$DBUSER;$PW;hotfixes\""
setc "$W" DataDir "\"$DATA\""; setc "$W" RealmID 70; setc "$W" Expansion 0
setc "$W" Network.SkipBuildAuthKeyCheck 1; setc "$W" Network.EnterEncryptedModeRegionGroup 8
setc "$W" MaxPlayerLevel 60; setc "$W" Ra.Enable 0; setc "$W" SOAP.Enabled 0
setc "$W" PacketLogFile '"world.pkt"'; setc "$W" PacketSpoof.Policy 0   # log only
B="$ETC/bnetserver.conf"
setc "$B" Realm.CfgContentSetID 137
setc "$B" LoginREST.ExternalAddress trinity.actual.battle.net; setc "$B" LoginREST.LocalAddress trinity.actual.battle.net
setc "$B" CertificatesFile "\"$QA/database/tls/fullchain.pem\""; setc "$B" PrivateKeyFile "\"$QA/database/tls/key.pem\""
log "config written. Realm address: run once worldserver created realmlist ->"
log "  UPDATE auth.realmlist SET address='127.0.0.1', localAddress='127.0.0.1' WHERE id=70;"
