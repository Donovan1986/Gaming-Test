# Recon: advocaite/TrinityCore `forever` @ 3f77d65 (2026-10-07)

## Build/client/data
- README: client **WoW Classic beta 1.60.1.70170**, CASC product `wow_classic_beta`; extractors default to it.
- `sql/custom/auth` registers builds 70009, 70058, 70124, 70170, 70205, 70235. Code comments cite sniffs up to 70235.
  The source *accepts* 70124, but the README-documented build is 70170. Use whatever build the client has, provided it is in that list.
- DB: TrinityCore auto-setup with `TDB_full_world_1210.26091_2026_09_09.sql` + `TDB_full_hotfixes_1210.26091_2026_09_09.sql`
  (GitHub release assets) placed next to worldserver; `Updates.EnableDatabases = 15` applies `sql/updates` + `sql/custom`.
  The VMaNGOS-converted vanilla world is `sql/custom/world/*forever_baseline_0{1,2}.sql` (~65 MB).
  DBs: auth, characters, world, hotfixes (standard master layout).
- Required config: worldserver `RealmID=70`, `Expansion=0`, `Network.SkipBuildAuthKeyCheck=1`,
  `Network.EnterEncryptedModeRegionGroup=8`; bnetserver `Realm.CfgContentSetID=137`,
  `LoginREST.ExternalAddress/LocalAddress=trinity.actual.battle.net` + TLS cert for that name.
- Login: the client connects to `trinity.actual.battle.net` → hosts entry `127.0.0.1`, a self-made local CA with
  nameConstraints, and the Forever Launcher (Windows). **A macOS client is not mentioned anywhere.** The beta client is `_classic_beta_\WowB.exe`.
  You need to confirm the Mac client path.

## Level cap
- `worldserver.conf.dist`: `MaxPlayerLevel = 90` (configurable, Max = MAX_LEVEL).
- **Hard-coded:** `src/server/game/Miscellaneous/SharedDefines.h:109` `GetMaxLevelForExpansion(EXPANSION_CLASSIC) = 30`
  (the retail level-squish table, not a Forever-specific rule). `Player::InitStatsForLevel` (Player.cpp:2427-2432) sends the
  client `MaxLevel = 30` because session expansion = 0 and 30 < MaxPlayerLevel. Also used in Unit.cpp:1782 (armor)
  and DB2Stores.cpp:2270 (scaling).
- Fix: `patches/qa-level60.patch` (one line: Classic → 60) + `MaxPlayerLevel = 60`. The patch is reversible with `git apply -R`.
  NOT compiled or tested. Risk: Unit.cpp armor uses ItemLevelByLevel row 60; `ASSERT_NOTNULL` there will crash if the row is missing (verify in game).

## Content
- Classic scripts: `src/server/scripts/Custom/Classic/instances/`: BFD, BRD, BRS, BWL, DM, Gnomer, Maraudon, MC,
  Naxx, Onyxia, RFD, RFK, AQ20, SM, Scholo, SFK, Strat, ST, Uldaman, WC, ZF, ZG (+ Deadmines).
- Onyxia: `classic_boss_onyxia.cpp` (3 phases at 65%/40%, deep breath), instance script, creature 10184 in baseline SQL,
  areatrigger teleport 1102848 → map 249 (30.89,-54.08,-5.03). `access_requirement` rows were not seen in the custom SQL
  (they may be in the TDB). Use `.go xyz 30.89 -54.08 -5.03 249` or `.tele` to bypass the attunement without changing the DB.
- Custom SQL includes talent reset and dual spec (`2026_09_28_19/20`), trainers, vanilla stats.

## GM commands confirmed present in source (cs_*.cpp)
`.character level`, `.levelup`, `.additem`, `.additem set`, `.learn all talents`, `.learn all/my ...`, `.reset talents`,
`.reset level`, `.tele`, `.tele add`, `.go instance`, `.go xyz`, `.npc add`, `.npc set level`, `.instance unbind`, `.gm on`,
`.gm fly`, `.modify speed`, `.cooldown`, `.die`, `.revive`, `.respawn`, `.appear`, `.summon`, `.damage`,
`.account create/set gmlevel`, `.bnetaccount create`; Forever-specific: `.legacy points`.

## Packet/detection
- `PacketLogFile` (worldserver.conf ~577), `PacketSpoof.Policy` (0 = log only, 1 = log+kick default, 2 = ban).
  QA config sets Policy=0 and enables packet logging to logs/.
