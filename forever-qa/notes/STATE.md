# ForeverQA STATE
SOURCE COMMIT: advocaite/TrinityCore@forever 3f77d65e993ab9045c6ea6286f6fe79526f2befb (2026-10-07)
CLIENT BUILD: UNKNOWN. Needs the user's Mac. README targets 1.60.1.70170; auth SQL also registers 70124, 70205 and 70235.
DATABASE REVISION: TDB_full_world/hotfixes_1210.26091_2026_09_09 (release assets) + sql/custom/* (auto-applied)
BUILD STATUS: NOT STARTED. Recon ran in a Linux cloud container with no access to the Mac.
DATABASE STATUS: NOT STARTED
DATA EXTRACTION STATUS: NOT STARTED (needs local client, product wow_classic_beta)
SERVER STATUS: NOT STARTED
ACCOUNT STATUS: NOT STARTED
MAGE STATUS: NOT STARTED
LEVEL 60 STATUS: BLOCKED IN SOURCE. Hard-coded cap of 30 found (see recon.md §Level cap); QA patch prepared, not compiled
TALENT STATUS: commands exist in source (.learn all talents / .reset talents); not tested
GEAR STATUS: item IDs NOT verified (world DB is a release asset, not in git)
DUNGEON STATUS: 24 classic instance scripts present in source; not tested
ONYXIA STATUS: script + creature 10184 + entrance areatrigger present; not tested
KNOWN PROBLEMS: see recon.md
NEXT ACTION: on the Mac run scripts/bootstrap.sh, then scripts/check-client.sh
