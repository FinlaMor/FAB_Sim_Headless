<?php
/**
 * HeadlessShims
 * ------------------------------------------------------------------
 * Tiny stand-ins for the DB / Patreon / Metafy / session helpers that
 * Talishar's gameplay flow calls during boot. The real implementations
 * (in `talishar/includes/functions.inc.php` and friends) cascade into
 * MySQL, SendGrid, Patreon OAuth, etc. — none of which we run in the
 * headless adapter.
 *
 * Every function here is guarded with `function_exists()` so the real
 * version still wins if you ever choose to load `functions.inc.php`.
 *
 * Add new stubs reactively: when `/new_game` or `/step` blows up on
 * "Call to undefined function X", add a guarded stub here that returns
 * the cheapest sensible default for X, restart the adapter, retry.
 */

declare(strict_types=1);

// --- Database -------------------------------------------------------------
if (!function_exists('GetDBConnection')) {
    function GetDBConnection($logKey = 0) { return null; }
}
if (!function_exists('IncrementDBConnectionCount')) {
    function IncrementDBConnectionCount($key): void {}
}
if (!function_exists('GetLocalMySQLConnection')) {
    function GetLocalMySQLConnection() { return null; }
}

// --- Saved-settings ------------------------------------------------------
if (!function_exists('LoadSavedSettings')) {
    function LoadSavedSettings($playerId) { return []; }
}
// ChangeSetting lives in talishar/Libraries/PlayerSettings.php — we load
// that explicitly in TalisharBoot::ENGINE_INCLUDES, so don't shadow it.

// --- User / ban / login --------------------------------------------------
// These live in Talishar's AccountFiles/AccountSessionAPI.php /
// AccountDatabaseAPI.php / BlockedUserLibraries.php / FriendLibraries.php
// — all loaded downstream of StartEffects.php on first use. Shimming
// them here used to collide with the real declarations the moment those
// files were pulled in.

// --- Hero name truncation (called by JoinGame, also referenced elsewhere) -
if (!function_exists('TruncateHeroName')) {
    function TruncateHeroName($cardID) { return function_exists('SetID') ? SetID($cardID) : substr((string)$cardID, 0, 10); }
}

// --- Patreon / Metafy ---------------------------------------------------
// IsPatron and AltArtsDisabled both live in talishar/Libraries/PlayerSettings.php
// which we load explicitly in ENGINE_INCLUDES — stubbing them here would
// race with the real definitions on subsequent process restarts.
if (!function_exists('GetMetafyTiersFromDatabase')) {
    function GetMetafyTiersFromDatabase($uid) { return ''; }
}
if (!function_exists('GetMetafyCommunitiesFromDatabase')) {
    function GetMetafyCommunitiesFromDatabase($uid) { return ''; }
}
if (!function_exists('storeFabraryId')) {
    function storeFabraryId($a, $b): void {}
}
if (!function_exists('GetDeckBuilderId')) {
    function GetDeckBuilderId($a, $b) { return ''; }
}
if (!function_exists('addFavoriteDeck')) {
    function addFavoriteDeck($a, $b, $c, $d, $e): void {}
}

// SaveReplay lives in talishar/ParseGamestate.php which TalisharBoot now
// loads via `_once` so it isn't double-declared across requests.

// --- Replay flag ---------------------------------------------------------
// IsReplay() lives in Libraries/HTTPLibraries.php (which we don't load —
// it's full of web/session/CORS helpers). Card resolution paths such as
// OPT (PassInput) call it to decide whether to append to commandfile.txt.
// The headless adapter never replays, so return false; the engine then
// writes its replay/command log normally under Games/{id}/.
if (!function_exists('IsReplay')) {
    function IsReplay() { return false; }
}

// --- Game-over stats reporting -------------------------------------------
// PlayerWon() calls logCompletedGameStats() (normally in
// includes/functions.inc.php) to write match results to MySQL / ELO /
// leaderboards. It's wrapped in a `catch (Exception)`, but an undefined
// function raises an `Error` (not `Exception`), so a missing definition
// crashes the winning step instead of being swallowed. No-op it.
if (!function_exists('logCompletedGameStats')) {
    function logCompletedGameStats($conceded = false): void {}
}

// --- Misc ----------------------------------------------------------------
if (!function_exists('TransformMatchupsWithTurnOrder')) {
    function TransformMatchupsWithTurnOrder($m) { return $m; }
}
