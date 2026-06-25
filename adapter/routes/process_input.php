<?php
/**
 * GET /ProcessInput.php?gameName=...&mode=...&cardID=...
 *   (also reachable as /game/ProcessInput.php via the FE dev proxy)
 *
 * The Talishar FE routes ALL game-input through this endpoint. For game
 * REVIEW we only honour the replay turn-hop (the ReplayPanel's Prev / Next /
 * Load-Turn / Return-to-Start buttons dispatch mode=HOP_TO_TURN with the turn
 * number in cardID). Every other mode is a no-op success — a finished game is
 * read-only, so we never mutate it by replaying real plays.
 *
 * A turn-hop copies the saved turn_{player}-{turn}_Gamestate.txt snapshot over
 * gamestate.txt; the FE's next GetNextTurn poll then serialises that turn (the
 * mtime change is what the FE notices). Turn 0 loads origGamestate.txt.
 */

declare(strict_types=1);

header('Content-Type: application/json; charset=utf-8');

const HOP_TO_TURN = 10018;

$gameName = (string)($_GET['gameName'] ?? '');
$mode     = (int)($_GET['mode'] ?? 0);
$cardID   = (string)($_GET['cardID'] ?? '');

if ($gameName === '' || !preg_match('/^\d+$/', $gameName)) {
    echo json_encode(['errorMessage' => 'Invalid game name.']);
    return;
}

$dir = TALISHAR_ROOT . "/Games/$gameName";
if (!is_dir($dir)) {
    echo json_encode(['errorMessage' => 'Game no longer exists on the server.']);
    return;
}

if ($mode !== HOP_TO_TURN) {
    // Read-only review: accept and ignore real plays.
    echo json_encode(['success' => true, 'noop' => true]);
    return;
}

$turn = (int)$cardID;
$gs = "$dir/gamestate.txt";

if ($turn <= 0) {
    $orig = "$dir/origGamestate.txt";
    if (is_file($orig)) {
        @copy($orig, $gs);
        echo json_encode(['success' => true, 'turn' => 0, 'loaded' => 'origGamestate']);
        return;
    }
}

// turn_{player}-{turn}_Gamestate.txt — either player may be the turn-taker.
foreach ([1, 2] as $p) {
    $snap = "$dir/turn_{$p}-{$turn}_Gamestate.txt";
    if (is_file($snap)) {
        @copy($snap, $gs);
        echo json_encode(['success' => true, 'turn' => $turn, 'loaded' => basename($snap)]);
        return;
    }
}

// Not found — report what turns ARE available so the UI can clamp.
$avail = [];
foreach (glob("$dir/turn_*_Gamestate.txt") ?: [] as $f) {
    if (preg_match('/turn_\d+-(\d+)_Gamestate/', basename($f), $m)) {
        $avail[] = (int)$m[1];
    }
}
sort($avail);
echo json_encode(['errorMessage' => "Turn $turn not found.",
                  'availableTurns' => array_values(array_unique($avail))]);
