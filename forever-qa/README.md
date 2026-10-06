# ForeverQA kit (advocaite/TrinityCore `forever`, local only)
Run on the Mac in this order: `scripts/bootstrap.sh` → `setup-db.sh` → `check-client.sh [WoW dir]` → `extract-data.sh`
→ `configure.sh` → (put the TDB_full_*_1210.26091 release SQL into ~/ForeverQA/server/bin) → `start-all.sh`.
Then `backup-db.sh` → `create-account.sh`. Stop: `stop-all.sh`. Notes: `notes/STATE.md`, `notes/recon.md`. Commands: `MAGE-QA.md`.
Copy the notes into `~/ForeverQA/notes` (`cp notes/* ~/ForeverQA/notes/`).
Everything binds to 127.0.0.1. Client login also needs the hosts entry and a local CA/cert from the upstream README (§3, "Login").
