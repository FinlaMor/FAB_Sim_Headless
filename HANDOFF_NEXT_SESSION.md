# Handoff — next session

**Status as of 2026-05-30:** the real Talishar adapter now plays a
**complete game to a winner**. Combat fully works — attacks go on the
combat chain, the defender gets block/defence steps, damage resolves,
health drops, and `PlayerWon` fires with `done=True winner=N reward=±1`.

The breakthrough was a single root cause that had been masquerading as
"the serializer doesn't drive combat": **`Constants.php`'s ~hundreds of
top-level `$CCS_*` / `$CS_*` / `$SET_*` index variables were never
reaching global scope**, because `TalisharBoot::ensureBooted()` `require`s
the engine files from *inside a static method*. Every `$CCS_X = N;`
assignment landed in the method's local scope, so engine functions doing
`global $CCS_AttackTarget; $combatChainState[$CCS_AttackTarget] = ...`
were writing to `$combatChainState[null]` (the `""` key). The attack
target was silently lost → `MISSINGTARGET` → the attack resolved as a
plain non-attack and combat never happened.

## Quick resume

```powershell
$env:ADAPTER_MODE = "real"
docker compose up -d adapter            # code is bind-mounted; restart to reload PHP
curl http://localhost:8000/health       # {"ok":true,...,"mode":"real",...}

# Prove a full game to a winner (controlled: attack in M, never block):
python -m python.examples.diag_controlled_attack
#   -> "GAME OVER at step 108: winner=1 reward=1.0"  final hp=(7,0)

# Combat smoke with random bots on a pure-attack deck (blocks a lot, may not finish):
python -m python.examples.diag_random_attack

# Original drafted-deck smoke (now does real combat; slow blue decks rarely
# finish within the 400-step cap):
python -m python.examples.omn_real_pipeline_smoke
```

IMPORTANT: the adapter is a long-lived `php -S` process. After editing any
PHP under `adapter/` or `talishar/`, run
`docker compose restart adapter` (with `ADAPTER_MODE=real`) or the change
won't load.

## Fixes landed this session

1. **`adapter/lib/TalisharBoot.php`** — after the engine `require`s in
   `ensureBooted()`, promote every engine-defined local into `$GLOBALS`
   (`foreach (get_defined_vars() ...) $GLOBALS[$name] = $value;`). This is
   THE fix that made combat work. Verified: `$GLOBALS['CCS_AttackTarget']`
   is now `16` (was `<<UNSET>>`), `CS_PlayIndex` is `10`.
2. **`adapter/lib/TalisharBoot.php`** — `priority_player` now reads
   `$currentPlayer` instead of the stale `$turn[1]` (FinalizeTurn rewrites
   `$turn[0]/[2]/[3]` but leaves `$turn[1]` holding the previous turn's
   responder, so it was always "1"). The `ActionSerializer` keys off
   `priority_player`, so this also fixed the bot driving the wrong seat.
3. **`adapter/lib/HeadlessShims.php`** — added a no-op
   `logCompletedGameStats()` stub. `PlayerWon()` calls it and only catches
   `Exception`, but a missing function raises `Error`, which crashed the
   winning step.

## Test fixtures added

- `decks/_tmp_attack_smoke/seat{0,1}_deck.json` — 30-card mono-red cost-1
  Lightning attack decks (astral_strike / astral_assault / voltic_impact /
  flowing_stormstrike / beckoning_brilliance / electrolyze). Red attacks
  pitch for 1, so the deck is self-funding (play one, pitch another). No
  cost-0 attacks exist in the OMN cube (min attack cost is 1).
- `python/examples/diag_turnpass.py` — per-step phase/priority/stack dump
  (aggressive: always plays a non-PASS action).
- `python/examples/diag_random_attack.py` — random vs random, logs HP
  changes + winner.
- `python/examples/diag_controlled_attack.py` — deterministic attacker /
  non-blocker; the canonical "does a game finish?" check.

## What's next — in priority order

### 1. Run the full bracket pipeline end-to-end
Now that games complete, wire `omn_real_pipeline_smoke.py` up to the full
7-match bracket (the TODO in its docstring). Random bots on blocky blue
decks rarely finish in 400 steps — either raise the cap, or give the
RandomBot a light heuristic (prefer PASS when defending so attacks land,
prefer ATTACK in M). See item 3.

### 2. Audit other Constants.php globals for correctness
The globals-promotion fix unblocked everything, but double-check that no
engine top-level var we now promote collides badly with a
`seedDefaultGlobals()` default (engine value wins, which is what we want).
The combat-chain init string in `writeInitialGamestate` has one stale
value: index 1 (`CCS_WeaponIndex`) is `0`, canonical is `-1`. Harmless so
far (ResetCombatChainState fixes it on first combat) but worth aligning.

### 3. Bot strategy (`python/gameplay/bots/random_bot.py`)
`RandomBot` blocks/pitches uniformly, which stalls games. A minimal
heuristic — PASS in D/B/INSTANT, ATTACK in M, pitch the cheapest card —
would make random games terminate and produce real win/loss signal for
training.

### 4. `combatChainState` length drift (minor)
The serialized `combatChainState` grows by ~1 element per round-trip
(trailing artifact from `implode`/`GetStringArray`). It doesn't shift the
0..50 semantic indices so it's currently harmless, but it will grow
unbounded over a very long game — worth trimming to
`max(CCS index)+1 = 51` on read if it ever matters.

## Older notes (still valid)

- `bootstrap.php` buffers stdout so PHP warnings never leak into JSON.
- `RngHook::mix` uses CRC32 (no int*int float promotion).
- `lib/HeadlessShims.php` stubs DB/Patreon/Metafy/saved-settings.
- `lib/TalisharBoot.php` mirrors `Start.php` for `createGame`.
- `talishar/CardSetters.php::IncrementClassState` no-ops on NULL `$piece`.
- `python/draft/format.py::BROKEN_CARDS` denylist filters unresolvable cards.
- 114 `include`→`include_once` edits across 46 Talishar files — re-run the
  Python patch (bottom of this file's git history) after a fresh clone.

## Re-applying the Talishar include-once patch

```python
import pathlib, re
root = pathlib.Path("talishar")
target_regexes = [
    r'(?:Libraries/)?(?:[A-Z][A-Za-z]+Libraries|HTTPLibraries|CacheLibraries|SHMOPLibraries|NetworkingLibraries|StatFunctions|UILibraries|CoreLibraries|PlayerSettings|ValidationLibraries|FriendLibraries|BlockedUserLibraries)\.php',
    r'CardDictionary\.php', r'Constants\.php', r'GameTerms\.php',
    r'WriteLog\.php', r'GameLogic\.php', r'CardLogic\.php',
    r'CoreLogic\.php', r'MZLogic\.php', r'CombatChain\.php',
    r'CardGetters\.php', r'CardSetters\.php', r'WeaponLogic\.php',
    r'AllyAbilities\.php', r'AuraAbilities\.php', r'ItemAbilities\.php',
    r'LandmarkAbilities\.php', r'PermanentAbilities\.php',
    r'CharacterAbilities\.php', r'CurrentEffectAbilities\.php',
    r'BuildPlayerInputPopup\.php', r'BuildGameState\.php',
    r'GetNextTurn\.php', r'AI/CombatDummy\.php', r'AI/PlayerMacros\.php',
    r'AI/EncounterAI\.php', r'(?:Classes/)?Card\.php',
    r'Classes/CardObjects/[A-Z]+Cards\.php',
]
combined = r'(?:\./|\.\./)?(?:' + '|'.join(target_regexes) + r')'
pat = re.compile(
    r'^(\s*)include(\s+)(["\'])(' + combined + r')\3\s*;',
    re.MULTILINE,
)
for php in root.rglob("*.php"):
    text = php.read_text(encoding="utf-8", errors="ignore")
    new, n = pat.subn(r'\1include_once\2\3\4\3;', text)
    if n:
        php.write_text(new, encoding="utf-8")
```
