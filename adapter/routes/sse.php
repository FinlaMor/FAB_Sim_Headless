<?php
/**
 * GET /GetUpdateSSE.php?gameName=...&playerID=3&authKey=spectator&friendsList=[]&userName=
 *   (also reachable as /game/GetUpdateSSE.php via the FE dev proxy)
 *
 * Server-Sent Events live game feed for the REAL Talishar FE. The modern FE
 * dropped GetNextTurn polling and now receives ALL game state over an
 * EventSource on GetUpdateSSE.php (see Talishar-FE app/GameStateHandler.tsx).
 * Without this endpoint the board frame renders but no cards ever arrive.
 *
 * We stream the SAME BuildGameStateResponse payload that spectate.php serves,
 * re-emitting whenever the game's gamestate.txt changes on disk (the
 * play_vs_champion driver / self-play rewrites it every step). The stream is
 * capped at a bounded duration and the EventSource reconnects (retry hint), so
 * the single-threaded `php -S` worker is never held open indefinitely.
 *
 * Read-only: never mutates game state.
 */

declare(strict_types=1);

require_once ADAPTER_LIB . '/TalisharBoot.php';

// --- SSE transport setup -------------------------------------------------
// bootstrap.php wraps the request in ob_start(); drain/disable ALL buffering
// so each event flushes to the client immediately.
while (ob_get_level() > 0) { @ob_end_clean(); }
@ini_set('zlib.output_compression', '0');
@ini_set('output_buffering', '0');
@ini_set('implicit_flush', '1');
@set_time_limit(0);
ignore_user_abort(false);

header('Content-Type: text/event-stream; charset=utf-8');
header('Cache-Control: no-cache, no-transform');
header('Connection: keep-alive');
header('X-Accel-Buffering: no');   // defeat nginx/proxy buffering if fronted

$emit = function (string $event, $payload): bool {
    if ($event !== '') { echo "event: $event\n"; }
    echo 'data: ' . json_encode($payload, JSON_UNESCAPED_SLASHES) . "\n\n";
    @flush();
    return !connection_aborted();
};

echo "retry: 1000\n\n";   // FE reconnects 1s after we close
@flush();

$gameName = (string)($_GET['gameName'] ?? '');
if ($gameName === '' || !preg_match('/^\d+$/', $gameName)) {
    $emit('', ['error' => 'Invalid game name.']);
    return;
}
$viewAs = (int)($_GET['playerID'] ?? 3);
if (!in_array($viewAs, [1, 2, 3], true)) { $viewAs = 3; }

TalisharBoot::ensureBooted();

// Same guarded no-op stubs spectate.php needs for BuildGameStateResponse.
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
if (!class_exists('PatreonCampaign')) {
    @require_once TALISHAR_ROOT . '/Assets/patreon-php-master/src/PatreonDictionary.php';
}
if (!class_exists('MetafyCommunity')) {
    @require_once TALISHAR_ROOT . '/Assets/MetafyDictionary.php';
}

$prevDir = getcwd();
chdir(TALISHAR_ROOT);
try {
    require_once TALISHAR_ROOT . '/BuildGameState.php';

    $dir    = "./Games/$gameName";
    $gsFile = "$dir/gamestate.txt";
    $start  = time();
    $maxDuration = 90;       // close after 90s; EventSource reconnects (retry:1000)
    $lastMtime   = -1;
    $lastPing    = 0;

    while (!connection_aborted() && (time() - $start) < $maxDuration) {
        if (!file_exists("$dir/GameFile.txt")) {
            $emit('', ['error' => 'Game no longer exists on the server.']);
            break;
        }
        clearstatcache(true, $gsFile);
        $m = is_file($gsFile) ? (int)filemtime($gsFile) : 0;

        if ($m !== $lastMtime) {
            $lastMtime = $m;
            // Force spectator visibility (same as spectate.php).
            if (function_exists('SetCachePiece')) { SetCachePiece($gameName, 9, '1'); }
            $resp = BuildGameStateResponse($gameName, $viewAs, '', [], true, false);
            if (is_string($resp)) {
                if (!$emit('', ['error' => $resp])) { break; }
            } else {
                if (is_object($resp)) { $resp->isReplay = false; }
                if (!$emit('', $resp)) { break; }
            }
        } elseif ((time() - $lastPing) >= 2) {
            // Frequent heartbeat so a browser refresh/navigate is detected within
            // ~2s (connection_aborted needs a write), freeing the single php -S
            // worker quickly for the reconnect.
            $lastPing = time();
            echo ": ping\n\n";
            @flush();
            if (connection_aborted()) { break; }
        }

        usleep(400000);   // 0.4s poll of the on-disk gamestate
    }
} finally {
    chdir($prevDir);
}
