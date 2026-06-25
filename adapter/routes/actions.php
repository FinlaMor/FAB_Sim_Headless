<?php
/** GET /actions?game_id=...  — enumerates legal actions for the priority player. */
declare(strict_types=1);

require_once ADAPTER_ROOT . '/GameAdapter.php';

$gameId = $_GET['game_id'] ?? '';
if ($gameId === '') json_error('game_id query parameter required', 400);

$adapter = new GameAdapter();
$actions = $adapter->getActions((string)$gameId);

json_response([
    'game_id'       => $gameId,
    'legal_actions' => $actions,
    'count'         => count($actions),
]);
