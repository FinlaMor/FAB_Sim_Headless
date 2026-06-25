<?php
/**
 * TalisharBoot
 * ------------------------------------------------------------------
 * Loads the upstream Talishar engine *inside the adapter process* and
 * reproduces the canonical game-init flow from ``talishar/Start.php``
 * (and the equivalent block in ``APIs/SubmitSideboard.php``) without
 * requiring Apache, MySQL, Redis, or the session/auth layer.
 *
 * Key reference points
 *   - talishar/Start.php                 — gold-standard gamestate.txt init
 *   - talishar/MenuFiles/StartHelper.php — initializePlayerState()
 *   - talishar/APIs/JoinGame.php         — 11-line deck-file format
 *   - talishar/ParseGamestate.php        — line-by-line reader (validates
 *                                          the order we write below)
 *
 * Bootstrapping notes
 *   - CacheStub.php must be loaded BEFORE any Talishar file that uses
 *     ReadCache/WriteCache/GetCachePiece/SetCachePiece so the engine
 *     binds to our disk-backed shims (no shmop/redis).
 *   - Card class files for unreleased sets (OMN/AZS/AHA/MPW/AOL/IAR/LGS)
 *     are loaded explicitly so GetClass($cardID) finds them. Without
 *     this, isUnimplemented() returns true and JoinGame rejects the deck.
 *   - We chdir() into talishar/ before each include because Talishar
 *     uses relative includes everywhere.
 */

declare(strict_types=1);

final class TalisharBoot
{
    private static bool $booted = false;

    // Files we have to require to wire up the rules engine. Order matters:
    // Constants/CardDictionary define globals other files use, so they go
    // first. We mirror the order Start.php / ProcessInputAPI.php use.
    private const ENGINE_INCLUDES = [
        'WriteLog.php',
        'GameTerms.php',
        'Constants.php',
        'CardDictionary.php',
        'Libraries/StatFunctions.php',
        'Libraries/UILibraries.php',
        'Libraries/CoreLibraries.php',
        'Libraries/PlayerSettings.php',
        'Libraries/NetworkingLibraries.php',
        'GameLogic.php',
        'AI/CombatDummy.php',
        'MenuFiles/StartHelper.php',
        'MenuFiles/WriteGamefile.php',
    ];

    // Card class files for unreleased sets — see APIs/JoinGame.php top.
    // Anything that's "in development" in Talishar has its Card classes
    // outside the autoload path; we have to include them explicitly.
    private const UNRELEASED_CARD_OBJECTS = [
        'Classes/Card.php',
        'Classes/CardObjects/OMNCards.php',
        'Classes/CardObjects/AZSCards.php',
        'Classes/CardObjects/AHACards.php',
        'Classes/CardObjects/MPWCards.php',
        'Classes/CardObjects/AOLCards.php',
        'Classes/CardObjects/IARCards.php',
        'Classes/CardObjects/LGSCards.php',
    ];

    // ------------------------------------------------------------------
    // Bootstrap
    // ------------------------------------------------------------------
    public static function ensureBooted(): void
    {
        if (self::$booted) return;

        require_once ADAPTER_LIB . '/CacheStub.php';
        CacheStub::register();
        // HeadlessShims must load BEFORE any Talishar file that calls
        // DB/Patreon/Metafy helpers (StartHelper::initializePlayerState
        // calls LoadSavedSettings at line 49). We declare guarded stubs;
        // Talishar's real versions still win if you later opt to load
        // `talishar/includes/functions.inc.php`.
        require_once ADAPTER_LIB . '/HeadlessShims.php';

        // Globals that several Talishar files dereference at file scope.
        self::seedDefaultGlobals();

        $prevDir = getcwd();
        chdir(TALISHAR_ROOT);
        try {
            foreach (self::ENGINE_INCLUDES as $rel) {
                $abs = TALISHAR_ROOT . '/' . $rel;
                if (is_file($abs)) {
                    @require_once $abs;
                }
            }
            // Card class files. Missing files are tolerated — sets are
            // added/removed over time and the adapter shouldn't crash on
            // a stale list.
            foreach (self::UNRELEASED_CARD_OBJECTS as $rel) {
                $abs = TALISHAR_ROOT . '/' . $rel;
                if (is_file($abs)) {
                    @require_once $abs;
                }
            }
        } finally {
            chdir($prevDir);
        }

        // CRITICAL: Constants.php (and other engine files) define HUNDREDS of
        // top-level index variables — $CCS_AttackTarget, $CS_PlayIndex,
        // $SET_*, the *Pieces() strides' backing globals, etc. Production
        // includes these at file (global) scope, so the assignments become
        // globals. We require them from inside this static method, which
        // means every `$CCS_X = N;` landed in THIS method's LOCAL scope and
        // never reached $GLOBALS. Engine functions then do
        // `global $CCS_AttackTarget;` and see null, so e.g.
        // `$combatChainState[$CCS_AttackTarget] = "THEIRCHAR-0"` writes to
        // $combatChainState[null] (the "" key) instead of index 16 — the
        // attack target is silently lost and combat never resolves.
        //
        // Promote every engine-defined local into the global symbol table.
        // (`try` does not introduce scope, so all included files' top-level
        // vars are visible here.) Skip our own loop bookkeeping vars.
        foreach (get_defined_vars() as $name => $value) {
            if ($name === 'rel' || $name === 'abs' || $name === 'prevDir') {
                continue;
            }
            $GLOBALS[$name] = $value;
        }

        self::$booted = true;
    }

    /**
     * Drag every Talishar global into the current method's local scope so
     * any subsequent require()/include() sees them as if running at top
     * level. The cost is a fixed-size global declaration list at the top
     * of each entry-point method; the alternative (running each engine
     * call as a subprocess) would dominate latency.
     *
     * The list is intentionally over-inclusive — adding an unused global
     * costs nothing but missing one causes "Undefined variable" warnings.
     */
    private static function importEngineGlobals(): void
    {
        global $gameName, $filename, $filepath, $playerID;
        global $p1Key, $p2Key, $firstPlayer, $p1IsAI, $p2IsAI;
        global $p1Hand, $p1Deck, $p1CharEquip, $p1Resources, $p1Arsenal, $p1Items, $p1Auras;
        global $p1Discard, $p1Pitch, $p1Banish, $p1ClassState, $p1CharacterEffects, $p1Soul;
        global $p1CardStats, $p1TurnStats, $p1Allies, $p1Permanents, $p1Settings, $p1Inventory;
        global $p2Hand, $p2Deck, $p2CharEquip, $p2Resources, $p2Arsenal, $p2Items, $p2Auras;
        global $p2Discard, $p2Pitch, $p2Banish, $p2ClassState, $p2CharacterEffects, $p2Soul;
        global $p2CardStats, $p2TurnStats, $p2Allies, $p2Permanents, $p2Settings, $p2Inventory;
        global $playerHealths, $landmarks, $winner, $currentPlayer, $currentTurn, $turn;
        global $actionPoints, $combatChain, $combatChainState, $currentTurnEffects;
        global $currentTurnEffectsFromCombat, $nextTurnEffects, $decisionQueue, $dqVars, $dqState;
        global $layers, $layerPriority, $mainPlayer, $defPlayer, $lastPlayed, $chainLinks;
        global $chainLinkSummary, $permanentUniqueIDCounter, $inGameStatus, $animations;
        global $currentPlayerActivity, $p1TotalTime, $p2TotalTime, $lastUpdateTime;
        global $roguelikeGameID, $events, $EffectContext, $p1CardTurnLog, $p2CardTurnLog;
        global $attackQueue, $AIHasInfiniteHP, $afterResolveEffects, $isProcessInput;
        global $mainPlayerGamestateStillBuilt, $mpgBuiltFor, $myStateBuiltFor;
        global $skipWriteGamestate, $makeCheckpoint, $makeBlockBackup, $MakeStartTurnBackup;
        global $animations, $events, $conceded, $randomSeeded;
        // No-op — the declarations themselves are the point. PHP will
        // discard this stack frame as soon as the caller returns.
    }

    /**
     * Provide harmless defaults for globals Talishar reads unconditionally.
     * We use $GLOBALS so the assignments persist across the include()
     * boundary into the engine's file scope.
     */
    private static function seedDefaultGlobals(): void
    {
        $defaults = [
            'p1IsAI'                => '0',
            'p2IsAI'                => '0',
            'p1IsPatron'            => '',
            'p2IsPatron'            => '',
            'p1IsChallengeActive'   => '0',
            'p2IsChallengeActive'   => '0',
            'p1id'                  => '',
            'p2id'                  => '',
            'p1uid'                 => 'Player 1',
            'p2uid'                 => 'Player 2',
            'p1Inventory'           => [],
            'p2Inventory'           => [],
            'p1TotalTime'           => '0',
            'p2TotalTime'           => '0',
            'p1SideboardSubmitted'  => '1',
            'p2SideboardSubmitted'  => '1',
            'p1ContentCreatorID'    => '',
            'p2ContentCreatorID'    => '',
            'p1MetafyTiers'         => '',
            'p2MetafyTiers'         => '',
            'p1MetafyCommunities'   => '',
            'p2MetafyCommunities'   => '',
            'p1DeckLink'            => '',
            'p2DeckLink'            => '',
            'p1deckbuilderID'       => '',
            'p2deckbuilderID'       => '',
            'p1Matchups'            => '',
            'p2Matchups'            => '',
            'p1StartingEquipment'   => '',
            'p2StartingEquipment'   => '',
            'roguelikeGameID'       => '0',
            'AIHasInfiniteHP'       => false,
            'playerID'              => 1,
            'isProcessInput'        => true,
            // The seat-of-pants list grew during boot debugging; keep it
            // here so a future include doesn't re-introduce an undefined
            // notice.
        ];
        foreach ($defaults as $k => $v) {
            if (!array_key_exists($k, $GLOBALS)) {
                $GLOBALS[$k] = $v;
            }
        }
    }

    // ------------------------------------------------------------------
    // Game creation
    // ------------------------------------------------------------------
    public static function createGame(string $gameId, array $req): void
    {
        self::ensureBooted();
        // PHP scopes `global` per function — a helper that declares them
        // does NOT propagate. So we inline a fat declaration list at the
        // top of every entry-point method. Required so require()'d
        // Talishar files see $gameName / $filename / $playerID /
        // $playerHealths / per-zone arrays at their top-level body
        // rather than tripping "Undefined variable" or null indexing.
        global $gameName, $filename, $filepath, $playerID;
        global $p1Key, $p2Key, $firstPlayer, $p1IsAI, $p2IsAI;
        global $p1Inventory, $p2Inventory, $roguelikeGameID;
        global $skipWriteGamestate, $makeCheckpoint, $makeBlockBackup, $MakeStartTurnBackup;
        global $MakeStartGameBackup;
        global $afterResolveEffects, $animations, $events, $randomSeeded, $conceded;
        global $mainPlayerGamestateStillBuilt, $mpgBuiltFor, $myStateBuiltFor;
        global $mainPlayer, $defPlayer, $otherPlayer, $currentPlayer, $currentTurn;
        global $format, $turn, $actionPoints, $combatChain, $combatChainState;
        global $currentTurnEffects, $currentTurnEffectsFromCombat, $nextTurnEffects;
        global $decisionQueue, $dqVars, $dqState, $layers, $layerPriority;
        global $lastPlayed, $chainLinks, $chainLinkSummary, $inGameStatus;
        global $permanentUniqueIDCounter, $currentPlayerActivity;
        global $p1TotalTime, $p2TotalTime, $lastUpdateTime, $EffectContext;
        global $p1CardTurnLog, $p2CardTurnLog, $attackQueue, $AIHasInfiniteHP;
        global $winner, $landmarks, $playerHealths;
        global $p1Hand, $p1Deck, $p1CharEquip, $p1Resources, $p1Arsenal, $p1Items, $p1Auras;
        global $p1Discard, $p1Pitch, $p1Banish, $p1ClassState, $p1CharacterEffects, $p1Soul;
        global $p1CardStats, $p1TurnStats, $p1Allies, $p1Permanents, $p1Settings;
        global $p2Hand, $p2Deck, $p2CharEquip, $p2Resources, $p2Arsenal, $p2Items, $p2Auras;
        global $p2Discard, $p2Pitch, $p2Banish, $p2ClassState, $p2CharacterEffects, $p2Soul;
        global $p2CardStats, $p2TurnStats, $p2Allies, $p2Permanents, $p2Settings;
        global $p1StartingHealth, $p2StartingHealth;
        global $p1StartingEquipment, $p2StartingEquipment;

        $gameDir = TALISHAR_ROOT . '/Games/' . $gameId;
        if (!is_dir($gameDir)) {
            mkdir($gameDir, 0775, true);
        }

        $p1Key = bin2hex(random_bytes(16));
        $p2Key = bin2hex(random_bytes(16));

        // Globals that Start.php / initializePlayerState / StartEffects
        // read. The `global` declarations at the top of createGame()
        // pulled them into method scope; assigning here writes through
        // to both the local view AND the global symbol table.
        $gameName        = $gameId;
        $playerID        = 1;
        $firstPlayer     = 1;
        $mainPlayer      = 1;
        $defPlayer       = 2;
        $otherPlayer     = 2;
        // Format is request-driven, defaulting to "draft" (OMN limited).
        // "draft" makes StartEffects.php run its "Macro for Limited" block
        // (puts the `omens_of_arcana` landmark in play + applies weapon/hero
        // activation restrictions); "cc" (Classic Constructed) deliberately
        // skips that block and plays the standard constructed game. Must agree
        // with the CacheStub field-12 and GameFile.txt format lines below or
        // ParseGamestate reverts it from the cache.
        $format          = $req['format'] ?? 'draft';
        $p1IsAI          = '0';
        $p2IsAI          = '0';
        $p1Inventory     = [];
        $p2Inventory     = [];
        $p1StartingHealth = '';
        $p2StartingHealth = '';
        // StartEffects calls equipModularForPlayer($id, $obj, $zones),
        // which dereferences fields on the object. Empty stdClass is the
        // "no starting equipment" sentinel.
        $p1StartingEquipment = new \stdClass();
        $p2StartingEquipment = new \stdClass();
        $roguelikeGameID = '0';
        $filename        = "./Games/$gameId/gamestate.txt";
        $filepath        = "./Games/$gameId/";

        // GameFile.txt — minimal viable. Order/contents follow
        // MenuFiles/WriteGamefile.php; we write enough fields that
        // ParseGamefile won't choke. Patch in extra fields if upstream
        // adds them.
        self::writeGameFile("$gameDir/GameFile.txt", $req, $p1Key, $p2Key);

        // 11-line deck files (Talishar's actual format).
        self::writeTalisharDeckFile("$gameDir/p1Deck.txt", $req['deck1']);
        self::writeTalisharDeckFile("$gameDir/p2Deck.txt", $req['deck2']);

        file_put_contents("$gameDir/gamelog.txt", '');

        // Cache block: 16 "!"-delimited fields per Start.php line 124.
        $p1Hero = self::extractHero($req['deck1']);
        $p2Hero = self::extractHero($req['deck2']);
        $now    = (string)round(microtime(true) * 1000);
        $cacheVisibility = '0';
        // Field 12 is the format; ParseGamestate reads it back from the
        // cache, so it must match $format above.
        CacheStub::write($gameId, "1!$now!$now!-1!-1!$now!$p1Hero!$p2Hero!$cacheVisibility!0!0!0!$format!1!0!0");

        $prevDir = getcwd();
        chdir(TALISHAR_ROOT);
        try {
            self::writeInitialGamestate($gameId, $p1Key, $p2Key, $p1Hero, $p2Hero);

            require_once TALISHAR_ROOT . '/ParseGamestate.php';
            if (function_exists('ParseGamestate')) ParseGamestate();
            if (is_file(TALISHAR_ROOT . '/StartEffects.php')) {
                require TALISHAR_ROOT . '/StartEffects.php';
            }
            // Persist the post-StartEffects state.
            if (function_exists('DoGamestateUpdate')) DoGamestateUpdate();
            require TALISHAR_ROOT . '/WriteGamestate.php';
        } finally {
            chdir($prevDir);
        }
    }

    // ------------------------------------------------------------------
    // State read
    // ------------------------------------------------------------------
    public static function readState(string $gameId): array
    {
        self::ensureBooted();
        // See createGame for why this fat declaration list is inlined.
        global $gameName, $filename, $filepath, $playerID;
        global $mainPlayerGamestateStillBuilt, $mpgBuiltFor, $myStateBuiltFor;
        global $mainPlayer, $defPlayer, $otherPlayer, $currentPlayer, $currentTurn;
        global $format, $turn, $actionPoints, $combatChain, $combatChainState;
        global $currentTurnEffects, $currentTurnEffectsFromCombat, $nextTurnEffects;
        global $decisionQueue, $dqVars, $dqState, $layers, $layerPriority;
        global $lastPlayed, $chainLinks, $chainLinkSummary, $inGameStatus;
        global $permanentUniqueIDCounter, $currentPlayerActivity;
        global $p1TotalTime, $p2TotalTime, $lastUpdateTime, $EffectContext;
        global $p1CardTurnLog, $p2CardTurnLog, $attackQueue, $AIHasInfiniteHP;
        global $winner, $landmarks, $playerHealths, $firstPlayer;
        global $p1Key, $p2Key, $p1IsAI, $p2IsAI, $p1Inventory, $p2Inventory;
        global $animations, $events;
        global $p1Hand, $p1Deck, $p1CharEquip, $p1Resources, $p1Arsenal, $p1Items, $p1Auras;
        global $p1Discard, $p1Pitch, $p1Banish, $p1ClassState, $p1CharacterEffects, $p1Soul;
        global $p1CardStats, $p1TurnStats, $p1Allies, $p1Permanents, $p1Settings;
        global $p2Hand, $p2Deck, $p2CharEquip, $p2Resources, $p2Arsenal, $p2Items, $p2Auras;
        global $p2Discard, $p2Pitch, $p2Banish, $p2ClassState, $p2CharacterEffects, $p2Soul;
        global $p2CardStats, $p2TurnStats, $p2Allies, $p2Permanents, $p2Settings;

        $prevDir = getcwd();
        chdir(TALISHAR_ROOT);
        try {
            $gameName = $gameId;
            $playerID = 1;
            $filename = "./Games/$gameId/gamestate.txt";
            $filepath = "./Games/$gameId/";
            require_once TALISHAR_ROOT . '/ParseGamestate.php';
            if (function_exists('ParseGamestate')) ParseGamestate();

            return self::collectGlobals();
        } finally {
            chdir($prevDir);
        }
    }

    // ------------------------------------------------------------------
    // Action application
    // ------------------------------------------------------------------
    public static function applyAction(string $gameId, array $action): void
    {
        self::ensureBooted();
        // See createGame for why this fat declaration list is inlined.
        global $gameName, $filename, $filepath, $playerID;
        global $p1Key, $p2Key, $firstPlayer, $p1IsAI, $p2IsAI;
        global $p1Inventory, $p2Inventory, $roguelikeGameID;
        global $skipWriteGamestate, $makeCheckpoint, $makeBlockBackup, $MakeStartTurnBackup;
        global $MakeStartGameBackup;
        global $afterResolveEffects, $animations, $events, $randomSeeded, $conceded;
        global $mainPlayerGamestateStillBuilt, $mpgBuiltFor, $myStateBuiltFor;
        global $mainPlayer, $defPlayer, $otherPlayer, $currentPlayer, $currentTurn;
        global $format, $turn, $actionPoints, $combatChain, $combatChainState;
        global $currentTurnEffects, $currentTurnEffectsFromCombat, $nextTurnEffects;
        global $decisionQueue, $dqVars, $dqState, $layers, $layerPriority;
        global $lastPlayed, $chainLinks, $chainLinkSummary, $inGameStatus;
        global $permanentUniqueIDCounter, $currentPlayerActivity;
        global $p1TotalTime, $p2TotalTime, $lastUpdateTime, $EffectContext;
        global $p1CardTurnLog, $p2CardTurnLog, $attackQueue, $AIHasInfiniteHP;
        global $winner, $landmarks, $playerHealths;
        global $p1Hand, $p1Deck, $p1CharEquip, $p1Resources, $p1Arsenal, $p1Items, $p1Auras;
        global $p1Discard, $p1Pitch, $p1Banish, $p1ClassState, $p1CharacterEffects, $p1Soul;
        global $p1CardStats, $p1TurnStats, $p1Allies, $p1Permanents, $p1Settings;
        global $p2Hand, $p2Deck, $p2CharEquip, $p2Resources, $p2Arsenal, $p2Items, $p2Auras;
        global $p2Discard, $p2Pitch, $p2Banish, $p2ClassState, $p2CharacterEffects, $p2Soul;
        global $p2CardStats, $p2TurnStats, $p2Allies, $p2Permanents, $p2Settings;

        $prevDir = getcwd();
        chdir(TALISHAR_ROOT);
        try {
            $gameName = $gameId;
            $playerID = (int)($action['player_id'] ?? 1);
            $filename = "./Games/$gameId/gamestate.txt";
            $filepath = "./Games/$gameId/";
            // Mirror ProcessInputAPI.php lines 74-111: initialise the
            // per-request mutables that ProcessInput / CardLogic / the
            // decision queue all read. Without these CardLogic.php:125
            // (CopyCurrentTurnEffectsFromAfterResolveEffects) trips on
            // a null $afterResolveEffects, and several other code paths
            // dereference $animations / $events.
            $skipWriteGamestate = false;
            $mainPlayerGamestateStillBuilt = 0;
            $makeCheckpoint = 0;
            $makeBlockBackup = 0;
            $MakeStartTurnBackup = false;
            $conceded = false;
            $randomSeeded = false;
            $afterResolveEffects = [];
            $animations = [];
            $events = [];
            require_once TALISHAR_ROOT . '/ParseGamestate.php';
            if (function_exists('ParseGamestate')) ParseGamestate();

            $mode        = $action['talishar_mode']       ?? 99;
            $buttonInput = $action['talishar_button']     ?? '';
            $cardID      = $action['talishar_card_id']    ?? '';
            $chkCount    = $action['talishar_chk_count']  ?? 0;
            $chkInput    = $action['talishar_chk_input']  ?? [];
            $inputText   = $action['talishar_input_text'] ?? '';

            ProcessInput($playerID, $mode, $buttonInput, $cardID, $chkCount, $chkInput, false, $inputText);

            if (function_exists('ProcessMacros'))     ProcessMacros();
            if (function_exists('CacheCombatResult')) CacheCombatResult();
            if (function_exists('DoGamestateUpdate')) DoGamestateUpdate();

            require TALISHAR_ROOT . '/WriteGamestate.php';

            // Mirror ProcessInputAPI.php's tail: when ProcessInput started a
            // new turn it sets $MakeStartTurnBackup, and the real endpoint
            // flushes it AFTER WriteGamestate so the per-turn snapshot
            // (turn_{player}-{turn}_Gamestate.txt) captures the freshly written
            // gamestate. Without this our headless games only ever saved
            // turn_1-0, so a played game couldn't be replayed turn by turn.
            if ($MakeStartTurnBackup && function_exists('MakeStartTurnBackup')) {
                MakeStartTurnBackup();
            }
        } finally {
            chdir($prevDir);
        }
    }

    // ==================================================================
    // Helpers
    // ==================================================================

    /**
     * Translate our JSON deck file into Talishar's 11-line format.
     *
     * Format (mirrors APIs/JoinGame.php lines 497-507):
     *   Line  1: <hero> <weapon> <equipmentHead> <equipmentChest> <equipmentArms> <equipmentLegs>
     *            (space-separated; only the cards that are present)
     *   Line  2: <deck cards, space-separated>
     *   Lines 3-11: sideboards (headSB / chestSB / armsSB / legsSB / offhandSB /
     *               weaponSB / sideboardCards / quiverSB / modularSB)
     *
     * The line-1 ordering preferred by Talishar is:
     *   hero weapon1 [weapon2] [offhand] [quiver] head chest arms legs
     * but ``initializePlayerState`` only requires the hero to come first
     * (it walks the array and detects the hero by type). We keep the
     * canonical order for cleanliness.
     */
    private static function writeTalisharDeckFile(string $dest, string $deckPath): void
    {
        $json = self::loadDeckJson($deckPath);
        $hero      = (string)($json['hero'] ?? 'unknown_hero');
        $equipment = array_values((array)($json['equipment'] ?? []));
        $deckCards = array_values((array)($json['deck'] ?? []));

        $charLine = trim($hero . ' ' . implode(' ', $equipment));
        $deckLine = implode(' ', $deckCards);

        $lines = [
            $charLine,           // 1
            $deckLine,           // 2
            '', '', '', '', '',  // 3-7: head/chest/arms/legs/offhand sideboards
            '',                  // 8: weapon sideboard
            '',                  // 9: main-deck sideboard
            '',                  // 10: quiver sideboard
            '',                  // 11: modular sideboard
        ];
        file_put_contents($dest, implode("\r\n", $lines));
    }

    private static function loadDeckJson(string $deckPath): array
    {
        $abs = $deckPath;
        if (!is_file($abs)) {
            $abs = PROJECT_ROOT . DIRECTORY_SEPARATOR . $deckPath;
        }
        if (!is_file($abs)) {
            json_error("Deck file not found: $deckPath", 422);
        }
        $json = json_decode((string)file_get_contents($abs), true);
        if (!is_array($json) || !isset($json['deck'])) {
            json_error("Deck file malformed: $deckPath", 422);
        }
        return $json;
    }

    private static function extractHero(string $deckPath): string
    {
        try {
            $json = self::loadDeckJson($deckPath);
            return (string)($json['hero'] ?? '-');
        } catch (\Throwable) {
            return '-';
        }
    }

    /**
     * Write the GameFile.txt with the fields ParseGamefile expects. We
     * keep a wide column count so older parse code doesn't trip over a
     * missing trailing entry.
     */
    private static function writeGameFile(string $path, array $req, string $p1Key, string $p2Key): void
    {
        $p1Hero = self::extractHero($req['deck1']);
        $p2Hero = self::extractHero($req['deck2']);
        $lines = [
            $p1Hero,           // p1 hero
            $p2Hero,           // p2 hero
            $p1Key,            // p1 auth key
            $p2Key,            // p2 auth key
            (string)($req['format'] ?? 'draft'),  // format (see $format in createGame)
            '1',               // first player
            '0',               // first player chooser (unused once both submitted)
            '1',               // p1 sideboard submitted
            '1',               // p2 sideboard submitted
            '0',               // p1IsAI
            '0',               // p2IsAI
            '',                // p1 deck link
            '',                // p2 deck link
            'Player 1',        // p1uid
            'Player 2',        // p2uid
            '',                // p1id
            '',                // p2id
        ];
        file_put_contents($path, implode("\r\n", $lines) . "\r\n");
    }

    /**
     * Write gamestate.txt by literally mimicking talishar/Start.php
     * lines 50-103. This is the canonical reference; if upstream
     * changes any field, sync this block.
     */
    private static function writeInitialGamestate(string $gameId, string $p1Key, string $p2Key,
                                                  string $p1Hero = '', string $p2Hero = ''): void
    {
        $filename = "./Games/$gameId/gamestate.txt";
        $handler  = fopen($filename, 'w');
        if (!$handler) {
            throw new RuntimeException("Cannot open $filename for writing");
        }

        try {
            // 0: playerHealths. Start.php uses CharacterHealth(hero) — NOT a
            // flat 20. This was hardcoded "20 20", which silently gave any
            // hero with non-20 printed life the wrong start (Oscilio is 19).
            $h1 = $h2 = 20;
            if (function_exists('CharacterHealth')) {
                try { if ($p1Hero !== '') $h1 = (int)CharacterHealth($p1Hero); } catch (\Throwable) {}
                try { if ($p2Hero !== '') $h2 = (int)CharacterHealth($p2Hero); } catch (\Throwable) {}
            }
            fwrite($handler, "$h1 $h2\r\n");

            $p1DeckHandler = fopen("./Games/$gameId/p1Deck.txt", 'r');
            if (!$p1DeckHandler) {
                throw new RuntimeException("Cannot open p1Deck.txt for $gameId");
            }
            initializePlayerState($handler, $p1DeckHandler, 1);
            fclose($p1DeckHandler);

            $p2DeckHandler = fopen("./Games/$gameId/p2Deck.txt", 'r');
            if (!$p2DeckHandler) {
                throw new RuntimeException("Cannot open p2Deck.txt for $gameId");
            }
            initializePlayerState($handler, $p2DeckHandler, 2);
            fclose($p2DeckHandler);

            fwrite($handler, "\r\n");                // Landmarks
            fwrite($handler, "0\r\n");               // Winner (0 = none)
            fwrite($handler, "1\r\n");               // First Player
            fwrite($handler, "1\r\n");               // Current Player
            fwrite($handler, "0\r\n");               // Current Turn
            fwrite($handler, "M 1\r\n");             // Phase + active player
            fwrite($handler, "1\r\n");               // Action points
            fwrite($handler, "\r\n");                // Combat Chain
            fwrite($handler,
                "0 0 0 0 0 0 0 GY NA 0 0 0 0 0 0 0 NA 0 0 -1 -1 NA 0 0 0 "
                . "-1 0 0 0 0 - 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 -1 0 0 0\r\n"); // Combat Chain State
            fwrite($handler, "\r\n");                // Current Turn Effects
            fwrite($handler, "\r\n");                // Current Turn Effects From Combat
            fwrite($handler, "\r\n");                // Next Turn Effects
            fwrite($handler, "\r\n");                // Decision Queue
            fwrite($handler, "0\r\n");               // Decision Queue Variables
            fwrite($handler, "0 - - -\r\n");         // Decision Queue State
            fwrite($handler, "\r\n");                // Layers
            fwrite($handler, "\r\n");                // Layer Priority
            fwrite($handler, "1\r\n");               // mainPlayer
            fwrite($handler, "\r\n");                // Last Played Card
            fwrite($handler, "0\r\n");               // Number of prior chain links this turn
            fwrite($handler, "\r\n");                // Chain Link Summaries
            fwrite($handler, $p1Key . "\r\n");
            fwrite($handler, $p2Key . "\r\n");
            fwrite($handler, "0\r\n");               // permanentUniqueIDCounter
            fwrite($handler, "0\r\n");               // inGameStatus (0 = START, 1 = PLAY, 2 = OVER)
            fwrite($handler, "\r\n");                // Animations
            fwrite($handler, "0\r\n");               // Current Player activity status
            fwrite($handler, "0\r\n");               // Player1 Rating
            fwrite($handler, "0\r\n");               // Player2 Rating
            fwrite($handler, "0\r\n");               // p1TotalTime
            fwrite($handler, "0\r\n");               // p2TotalTime
            fwrite($handler, time() . "\r\n");       // lastUpdateTime
            fwrite($handler, "0\r\n");               // roguelikeGameID
            fwrite($handler, "\r\n");                // Events
            fwrite($handler, "-\r\n");               // EffectContext
            fwrite($handler, "\r\n");                // p1 Inventory
            fwrite($handler, "\r\n");                // p2 Inventory
            fwrite($handler, "0\r\n");               // p1IsAI
            fwrite($handler, "0\r\n");               // p2IsAI
        } finally {
            fclose($handler);
        }

        // Push the file into the cache so subsequent ReadCache returns it.
        $gamestate = (string)file_get_contents($filename);
        if (function_exists('WriteGamestateCache')) {
            WriteGamestateCache($gameId, $gamestate);
        } else {
            WriteCache(GamestateID($gameId), $gamestate);
        }
    }

    /** Pull the Talishar globals that ParseGamestate just populated into a flat array. */
    private static function collectGlobals(): array
    {
        $names = [
            'playerHealths', 'landmarks', 'winner', 'firstPlayer', 'currentPlayer',
            'currentTurn', 'turn', 'actionPoints', 'combatChain', 'combatChainState',
            'currentTurnEffects', 'currentTurnEffectsFromCombat', 'nextTurnEffects',
            'decisionQueue', 'dqVars', 'dqState', 'layers', 'layerPriority',
            'mainPlayer', 'defPlayer', 'lastPlayed', 'chainLinks', 'inGameStatus',
            'events', 'attackQueue',
        ];
        $out = [];
        foreach ($names as $n) {
            $out[$n] = $GLOBALS[$n] ?? null;
        }
        $out['players'] = [];
        for ($p = 1; $p <= 2; $p++) {
            $prefix = "p$p";
            $playerOut = ['player_id' => $p];
            foreach (['Hand','Deck','CharEquip','Resources','Arsenal','Items','Auras','Discard',
                      'Pitch','Banish','ClassState','CharacterEffects','Soul','Allies','Permanents',
                      'CardStats','TurnStats','Settings','Inventory'] as $zone) {
                $g = $prefix . $zone;
                $playerOut[strtolower($zone)] = $GLOBALS[$g] ?? [];
            }
            $playerOut['health'] = $out['playerHealths'][$p-1] ?? null;
            // Hero is the first card in the character zone (CharEquip[0]) in both
            // CC and OMN. Surface it explicitly (+ base intellect = hand size)
            // so the serializer/model don't have to infer the hero from the
            // equipment list.
            $charZone = $GLOBALS[$prefix . 'CharEquip'] ?? [];
            $heroId   = (is_array($charZone) && isset($charZone[0])) ? (string)$charZone[0] : '';
            $playerOut['hero']      = $heroId;
            $playerOut['intellect'] = ($heroId !== '' && function_exists('CharacterIntellect'))
                ? (int)CharacterIntellect($heroId) : null;
            $out['players'][]    = $playerOut;
        }
        $out['turn_number']      = $out['currentTurn'];
        $out['phase']            = $out['turn'][0] ?? '';
        $out['current_player']   = (int)$out['currentPlayer'];
        // Priority lives in $currentPlayer. $turn[1] is NOT a reliable
        // priority indicator — FinalizeTurn() rewrites $turn[0]/[2]/[3]
        // on a new turn but leaves $turn[1] holding the previous turn's
        // responder, so it goes stale (always "1" after p1's first turn).
        $out['priority_player']  = (int)$out['currentPlayer'];
        $out['main_player']      = (int)$out['mainPlayer'];
        $out['action_points']    = (int)$out['actionPoints'];
        $out['winner']           = (int)$out['winner'];
        return $out;
    }
}
