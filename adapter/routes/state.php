<?php
/** GET /state?game_id=...  — returns canonical JSON game state. */
declare(strict_types=1);

require_once ADAPTER_ROOT . '/GameAdapter.php';

$gameId = $_GET['game_id'] ?? '';
if ($gameId === '') json_error('game_id query parameter required', 400);

$adapter = new GameAdapter();
$state   = $adapter->getState((string)$gameId);

json_response(['game_id' => $gameId, 'state' => $state]);
