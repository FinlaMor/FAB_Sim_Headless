# Talishar Backend Recon Notes

These notes capture how the headless adapter integrates with the live Talishar
engine. **Do not rewrite or replace any of the Talishar functions referenced
below** — the adapter must remain a thin shim that delegates to them.

Source repo: <https://github.com/Talishar/Talishar>
Pinned in `../talishar/` at clone time.

---

## 1. Main entry points

| File | Role |
| --- | --- |
| `talishar/APIs/CreateGame.php` | Creates `./Games/{gameName}/`, writes `GameFile.txt`, initialises SHMOP cache, then redirects to `JoinGame.php`. Depends on MySQL + session. The adapter inlines the file-creation logic in `routes/new_game.php` and skips the auth/Patreon/Metafy paths. |
| `talishar/ProcessInputAPI.php` | Production HTTP entry that wraps `ProcessInput()`. Reads `$_GET[mode/cardID/buttonInput/chkCount/chk0…/inputText]` and calls `ProcessInput()` after `ParseGamestate.php` runs. Limits execution to 1 s with `set_time_limit(1)` — the adapter overrides that. |
| `talishar/ProcessInput()` (in `GameLogic.php`) | The function we actually invoke for every action. Signature: `ProcessInput($playerID, $mode, $buttonInput, $cardID, $chkCount, $chkInput, $isReplay=false, $inputText="")`. |
| `talishar/ParseGamestate.php` | Loads the entire game state from `./Games/{gameName}/gamestate.txt` (via `ReadCache(GamestateID($gameName))`) into a long list of globals. The adapter calls this before every read/write. |
| `talishar/WriteGamestate.php` | Persists the globals back out. Called after every `ProcessInput`. |
| `talishar/SimulateGame.php` | A replay simulator. Useful pattern reference — we adopt the same `ProcessInput → ProcessMacros → CacheCombatResult → DoGamestateUpdate → WriteGamestate` cadence. |
| `talishar/AI/CombatDummy.php :: CombatDummyAI()` | **Gold.** Already implements an in-process AI driver that calls `ContinueDecisionQueue($opt)` for popups, `ProcessInput($p, 3, "", CharacterPieces(), $idx)` for hero/equip abilities, and `PassInput()` otherwise. We model our `step()` after this loop. |

## 2. Priority / decision system

- `$turn` is a 3-element array: `[0]=phase code`, `[1]=player ID who must respond`, `[2]=comma-separated options`.
- Phase codes include `M` (main), `A` (action), `D` (defence), and many input modes (`BUTTONINPUT`, `YESNO`, `OPT`, `PDECK`, `CHOOSECARD`, `MULTICHOOSE*`, `HANDTOPBOTTOM`, `ORDERTRIGGERS`, …) enumerated in `BuildPlayerInputPopup.php`.
- `$currentPlayer` holds priority (1 or 2).
- `$decisionQueue` is the async queue. When non-empty, the current `$turn[0]` reflects the head of the queue.

## 3. Action input format (mode codes)

Extracted from `BuildPlayerInputPopup.php` and `AI/CombatDummy.php`:

| Mode | Meaning | Args |
| ---: | --- | --- |
| 3 | Activate hero/equipment ability | `cardID=index, chkCount=CharacterPieces(), buttonInput=""` |
| 6 | Pitch a card from hand | `buttonInput=cardID` |
| 7 | `CHOOSENUMBER` / `DYNPITCH` response | `buttonInput=chosen value` |
| 8 / 9 | OPT: send card to top / bottom of deck | `buttonInput=cardID` |
| 11 | `CHOOSEDECK` etc. | `buttonInput=cardID` |
| 12 / 13 | `HANDTOPBOTTOM` (top / bottom) | `buttonInput=cardID` |
| 16 | Generic zone choice (`CHOOSECARDID`, `CHOOSEBANISH`, …) | `buttonInput=cardID` |
| 17 | `BUTTONINPUT` response | `buttonInput=label` |
| 19 | `MULTICHOOSE*` submit | `chk0…chkN=indices` |
| 20 | `YESNO` response | `buttonInput=YES`/`NO` |
| 23 | `CHOOSECARD` / `MAYCHOOSECARD` | `buttonInput=cardID` |
| 27 | Play card from hand by index | `cardID=hand index` (engine reads `$hand[$index]`) |
| 29 | Choose top of opponent deck | `buttonInput=cardID` |
| 99 | OK / pass through queue | `buttonInput=OK` |
| 100015 | Heartbeat (state diff) — ignored by adapter | |
| 105 | Skip all runechant arcane resolves | `buttonInput=0` |

The adapter's `ActionSerializer` enumerates these by inspecting `$turn`, `$decisionQueue`, hand contents, banish/discard contents, etc., and emits one structured `Action` per legal choice. Then `step()` translates the chosen `action_id` back into the underlying `$_GET`-style call to `ProcessInput`.

## 4. State globals (from `ParseGamestate.php`)

Per-player arrays (`p1*` and `p2*`):

`Hand, Deck, CharEquip, Resources, Arsenal, Items, Auras, Discard, Pitch,
 Banish, ClassState, CharacterEffects, Soul, CardStats, TurnStats,
 Allies, Permanents, Settings, Inventory, CardTurnLog`

Each zone has a `*Pieces()` function in `Constants.php` describing the stride
(e.g. `CharacterPieces()=15`, `AuraPieces()=14`, `DiscardPieces()=3`).
The serializer in `serializers/StateSerializer.php` chunks each array using
those strides and emits a structured JSON object.

Shared globals: `$playerHealths, $landmarks, $winner, $firstPlayer,
$currentPlayer, $currentTurn, $turn, $actionPoints, $combatChain,
$combatChainState, $currentTurnEffects, $nextTurnEffects, $decisionQueue,
$dqVars, $dqState, $layers, $layerPriority, $mainPlayer, $defPlayer,
$lastPlayed, $chainLinks, $inGameStatus, $events, $attackQueue`.

## 5. RNG / deterministic shuffle

- Only `Libraries/CoreLibraries.php` references `mt_srand` / `srand`.
- Shuffle goes through `ShuffleDeck()` / `ShuffleDeckAwait()` (in
  `DecisionQueue/AwaitEffects.php`), which uses `shuffle()` → `mt_rand`.
- To make games reproducible we call `mt_srand($seed + $stepCounter)` at the
  top of every adapter request. `RngHook.php` owns this.

## 6. Legal-action enumeration

There is **no single Talishar function that returns "all legal actions"** —
the engine is reactive. The adapter therefore composes the legal-action list
from three sources:

1. **Decision queue / popup state.** If `$turn[0]` matches one of the popup
   phases enumerated in `BuildPlayerInputPopup.php`, the options come from
   `$turn[2]` (split on `,` / `;` / `-`).
2. **Hand playability.** While in main / action phases we walk `$myHand`
   and call `PlayableCardSubset()` (in `CardLogic.php`) to filter to legal
   plays. Mode 27 is emitted for each playable hand index.
3. **Character / equipment activation.** Iterate `$myCharacter` in strides
   of `CharacterPieces()` and emit mode 3 actions for each ability whose
   cost can be paid (same check `CombatDummyAI` uses).
4. **The "pass" action** (mode 99 / `PassInput`) is always emitted unless
   forbidden (e.g. some `BUTTONINPUTNOPASS` phases).

The action enumerator delegates the actual "is this playable" check to
Talishar's own functions — we never re-implement card rules.

## 7. State storage

- File-based: `./Games/{gameName}/gamestate.txt`, plus `GameFile.txt`,
  `gamelog.txt`, `commandfile.txt` (input replay log), and several backups.
- Wrapped behind `ReadCache(GamestateID($gameName))` (in
  `Libraries/CacheLibraries.php`) — under production that hits Redis but
  falls back to the file on a cache miss.
- The adapter runs with `ADAPTER_DISABLE_CACHE=1`; we stub
  `Libraries/CacheLibraries.php` via `lib/CacheStub.php` so reads / writes
  go straight to disk.

## 8. Combat chain handling

- `CombatChain.php` (large) owns chain-link resolution.
- `$combatChain` is a flat array striped by `CombatChainPieces()=12`.
- The serializer emits one struct per link with the fields documented in
  `Constants.php` above `CombatChainPieces()`.
- During combat, priority bounces between players via the decision queue —
  the adapter handles this transparently because it always reads
  `$currentPlayer` after each `ProcessInput` call.

## 9. Bootstrap obstacles on this host

PHP 8.1 (and the shmop/Redis extensions) is **not present** on the Windows
host. Two supported run modes:

1. **`STUB_MODE` (default for the smoke test).** The adapter never includes
   any Talishar PHP files; instead, `lib/StubGame.php` runs a deterministic
   ~30-turn synthetic game whose state shape matches the real serializer
   output. This proves the Python wiring (env → bot → replay → parquet)
   end-to-end without depending on the PHP/MySQL/Redis stack.
2. **`REAL_MODE` (production).** Run the adapter inside the bundled
   `adapter/Dockerfile`-built container which has PHP 8.1, shmop, redis,
   pdo_mysql, and the appropriate `extension=` lines. The adapter then
   includes Talishar files normally; MySQL/Redis/sessions are stubbed out
   by `lib/TalisharBoot.php` so the engine boots without an external DB.

Switching modes is a single env var (`ADAPTER_MODE=stub|real`).

## 10. Reference: order of operations inside `routes/step.php`

```
1. Validate game_id, look up game record.
2. mt_srand(seed + step_counter)      ← RngHook
3. include ParseGamestate.php         ← loads $myHand, $turn, …
4. Resolve adapter action_id → (mode, cardID, buttonInput, chkInput).
5. ProcessInput(...)                  ← the only mutator
6. ProcessMacros(); CacheCombatResult();
7. DoGamestateUpdate(); WriteGamestate();
8. Re-read $winner, $turn, $currentPlayer.
9. Compute reward = +1 / -1 / 0.
10. Return {done, reward, winner, next_state, legal_actions}.
```

The reward function is intentionally minimal at this layer (terminal-only,
zero-sum). Shaping belongs in the Python `replay_buffer` / training code.
