# Forever QA: level-60 Mage and command reference
Every command here was found in the `forever` source (`src/server/scripts/Commands/cs_*.cpp`) @3f77d65.
**None has been run in game yet.** Type them in chat with a `.` prefix, as an account with GM level 3.

| COMMAND | PURPOSE | EXAMPLE | RBAC |
|---|---|---|---|
| `.character level` | set level (target/name) | `.character level Qamage 60` | COMMAND_CHARACTER_LEVEL |
| `.levelup` | +N levels | `.levelup 59` | COMMAND_LEVELUP |
| `.learn all talents` | learn every talent of your class | `.learn all talents` | COMMAND_LEARN_ALL_TALENTS |
| `.reset talents` | refund talents | `.reset talents` | COMMAND_RESET_TALENTS |
| `.learn` | learn spell by ID | `.learn <spellId>` | COMMAND_LEARN |
| `.lookup spell/item/creature/tele` | find real IDs (do not guess) | `.lookup item Netherwind` | COMMAND_LOOKUP_* |
| `.additem` | add item(s) | `.additem <itemId> 1` | COMMAND_ADDITEM |
| `.additem set` | add a whole item set | `.additem set <itemSetId>` | COMMAND_ADDITEMSET |
| `.tele` / `.tele add` | named teleports | `.tele add qa_ony` | COMMAND_TELE(_ADD) |
| `.go xyz` / `.go instance` | coordinate/instance teleport | `.go xyz 30.89 -54.08 -5.03 249` | COMMAND_GO |
| `.npc add` | spawn creature | `.npc add 10184` | COMMAND_NPC_ADD |
| `.respawn` | respawn nearby | `.respawn` | COMMAND_RESPAWN |
| `.instance unbind` | reset lockouts | `.instance unbind all` | COMMAND_INSTANCE_UNBIND |
| `.gm on/off`, `.gm fly on` | GM mode | `.gm on` | COMMAND_GM / GM_FLY |
| `.modify speed` | movement speed | `.modify speed 3` | COMMAND_MODIFY_SPEED |
| `.cooldown`, `.die`, `.revive`, `.damage` | combat control | `.cooldown` | matching COMMAND_* |
| `.appear`, `.summon` | move to/bring player | `.appear Qamage` | COMMAND_APPEAR/SUMMON |
| `.legacy points` | Forever-only command | — | COMMAND_ACHIEVEMENT_ADD |

## Level-60 Mage flow (in game)
1. Create a Mage on the realm, log in, `.gm on`
2. `.character level 60` (needs patch `patches/qa-level60.patch`, which `bootstrap.sh` applies, + `MaxPlayerLevel = 60`)
3. Spells: visit a Mage trainer (the Classic trainers are in `sql/custom/world/2026_09_28_02_world_classic_trainers.sql`)
4. Talents: spend points in the talent UI. Swap builds with `.reset talents` (dual spec SQL exists: `..._20_world_classic_dual_spec.sql`)
5. Gear: `.lookup item <name>` → `.additem <id>`. Record the verified IDs in `notes/gear-ids.md` and then build the gear macros from them.
   (No item IDs are hard-coded. The world DB is a release asset and has not been checked yet.)

## Onyxia (map 249)
Entrance teleport (from areatrigger 1102848): `.go xyz 30.8916 -54.079 -5.02784 249`. This bypasses the attunement without any DB change.
Boss creature 10184 is scripted in `classic_boss_onyxia.cpp`: phase 2 at 65 %, phase 3 at 40 %. Reset with `.instance unbind all`.
Checklist: entry · boss present · P1 abilities · P2 flight + Deep Breath · P3 + whelps/fear · loot · wipe reset · unbind.
