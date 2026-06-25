<?php
/**
 * GET /game_list?limit=50
 *
 * Lists the most recently played games (folders under talishar/Games/) for
 * the review picker. Pure filesystem scan — cheap, read-only. Returns the
 * two hero ids (GameFile lines 0/1), the format, and a millisecond mtime so
 * the FE/picker can show "most recent first".
 */

declare(strict_types=1);

header('Content-Type: application/json; charset=utf-8');

$limit = (int)($_GET['limit'] ?? 50);
if ($limit < 1)   $limit = 1;
if ($limit > 500) $limit = 500;

$root = TALISHAR_ROOT . DIRECTORY_SEPARATOR . 'Games';
$games = [];

$dh = @opendir($root);
if ($dh !== false) {
    while (($entry = readdir($dh)) !== false) {
        if ($entry === '.' || $entry === '..') continue;
        if (!preg_match('/^\d+$/', $entry)) continue;
        $gf = "$root/$entry/GameFile.txt";
        if (!is_file($gf)) continue;
        $games[] = [$entry, @filemtime($gf) ?: 0, $gf];
    }
    closedir($dh);
}

// Most recent first, then take the top N before the (cheap) detail read.
usort($games, fn($a, $b) => $b[1] <=> $a[1]);
$games = array_slice($games, 0, $limit);

$out = [];
foreach ($games as [$id, $mtime, $gf]) {
    $lines = @file($gf, FILE_IGNORE_NEW_LINES) ?: [];
    $out[] = [
        'gameName' => $id,
        'p1Hero'   => trim($lines[0] ?? ''),
        'p2Hero'   => trim($lines[1] ?? ''),
        'format'   => trim($lines[4] ?? ''),
        'mtime'    => $mtime * 1000,
    ];
}

echo json_encode(['games' => $out]);
