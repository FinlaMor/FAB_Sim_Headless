<?php
/**
 * GET /GetNextTurn.php?gameName=...&playerID=3&lastUpdate=0
 *   (also reachable as /game/GetNextTurn.php — the FE's dev proxy target)
 *
 * Headless spectator / replay poll. Drives the REAL Talishar
 * BuildGameStateResponse() over one of our persisted Games/{id}/ folders so
 * the genuine Talishar frontend can render a game we played. This is the
 * exact serializer the live site uses — we only feed it a finished game and
 * force spectator visibility. Strictly read-only: never mutates state.
 *
 * Query params mirror the live GetNextTurn.php so the FE needs no changes:
 *   gameName   – numeric game id (a folder under talishar/Games/)
 *   playerID   – 1|2|3; we always serve as spectator (3) for review
 *   lastUpdate – FE's last seen update counter (echoed back as lastUpdate)
 */

declare(strict_types=1);

require_once ADAPTER_LIB . '/TalisharBoot.php';

header('Content-Type: application/json; charset=utf-8');

$gameName = (string)($_GET['gameName'] ?? '');
if ($gameName === '' || !preg_match('/^\d+$/', $gameName)) {
    echo json_encode(['errorMessage' => 'Invalid game name.']);
    return;
}

// Optional: scrub to a specific turn snapshot before serializing. The FE
// replay UI can pass ?turn=turn_3-0 to load that saved gamestate; we copy it
// over gamestate.txt (in a per-request safe way) so ParseGamestate reads it.
$turnSnap = (string)($_GET['turn'] ?? '');

// LIVE mode (play_vs_champion): when ?live=1 we serve the CURRENT gamestate as
// an in-progress game for a real player seat (1|2) instead of a finished-game
// spectator replay. The Python driver mutates the game via /step; each mutation
// rewrites gamestate.txt, so the FE's mtime-driven lastUpdate re-renders. We
// only change visibility + the isReplay flag — never mutate state here.
// Live is requested explicitly (?live=1) OR auto-detected by a marker file the
// play_vs_champion driver drops in the game folder, so the unmodified FE URL
// renders an in-progress game live. Turn-nav (?turn=) always wins -> replay.
$liveMarker = TALISHAR_ROOT . "/Games/$gameName/.headless_live";
$live    = ($turnSnap === '')
        && ((((string)($_GET['live'] ?? '')) === '1') || is_file($liveMarker));
$viewAs  = (int)($_GET['playerID'] ?? 3);
if (!in_array($viewAs, [1, 2, 3], true)) { $viewAs = 3; }

TalisharBoot::ensureBooted();

// BuildGameStateResponse() expects a handful of HTTP/session/settings helpers
// that live in engine library files we deliberately don't boot (they drag in
// DB / SHMOP / Patreon). Each is tiny and irrelevant to spectating a finished
// game, so we provide guarded no-op stubs. function_exists() keeps these safe
// across the per-request re-include and never shadows a real engine symbol.
if (!function_exists('IsGameNameValid')) {
    function IsGameNameValid($gameName): bool { return (bool)preg_match('/^\d+$/', (string)$gameName); }
}
if (!function_exists('IsCasterMode')) {
    function IsCasterMode(): bool { return false; }
}
if (!function_exists('IsHideHandFromFriends')) {
    function IsHideHandFromFriends($player): bool { return false; }
}
if (!function_exists('GamestateUpdated')) {
    function GamestateUpdated($gameName): void {}
}
if (!function_exists('GetActiveSpectators')) {
    function GetActiveSpectators($gameName): array { return ['count' => 0, 'names' => []]; }
}

// The alt-art loops iterate the Patreon/Metafy enums. These dictionary files
// are pure enum definitions (no DB/HTTP side effects), so include them; for
// our games the campaign/community lists are empty and the loops no-op.
if (!class_exists('PatreonCampaign')) {
    @require_once TALISHAR_ROOT . '/Assets/patreon-php-master/src/PatreonDictionary.php';
}
if (!class_exists('MetafyCommunity')) {
    @require_once TALISHAR_ROOT . '/Assets/MetafyDictionary.php';
}

$prevDir = getcwd();
chdir(TALISHAR_ROOT);
try {
    $dir = "./Games/$gameName";
    if (!file_exists("$dir/GameFile.txt")) {
        echo json_encode(['errorMessage' => 'Game no longer exists on the server.']);
        return;
    }

    if ($turnSnap !== '' && preg_match('/^[A-Za-z0-9_\-]+$/', $turnSnap)) {
        $snapFile = "$dir/{$turnSnap}_Gamestate.txt";
        if (is_file($snapFile)) {
            @copy($snapFile, "$dir/gamestate.txt");
        }
    }

    // Any game we recorded is fair game to review: force spectator visibility.
    if (function_exists('SetCachePiece')) {
        SetCachePiece($gameName, 9, '1');
    }

    require_once TALISHAR_ROOT . '/BuildGameState.php';

    // playerID 3 = spectator (review); 1|2 = a live player seat (live=1).
    // Include the initial-load block (heroes, names, decks) so the board
    // renders fully on the first poll.
    $resp = BuildGameStateResponse($gameName, $viewAs, '', [], true, false);

    if (is_string($resp)) {
        echo json_encode(['errorMessage' => $resp]);
    } else {
        // Review games render as a replay (native ReplayPanel turn-nav). A LIVE
        // game must NOT be a replay — the FE then polls for fresh state and
        // shows the in-progress board for the seat we serve.
        if (is_object($resp)) {
            $resp->isReplay = $live ? false : true;
            // Drive the FE's change-detection off the gamestate file mtime so a
            // turn-hop (which rewrites gamestate.txt) is picked up on the next
            // poll and re-rendered.
            $gs = "./Games/$gameName/gamestate.txt";
            $resp->lastUpdate = is_file($gs) ? (int)filemtime($gs) : (int)($resp->lastUpdate ?? 0);
        }
        echo json_encode($resp);
    }
} finally {
    chdir($prevDir);
}
