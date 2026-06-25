<?php
/**
 * POST /reset
 * Body: {"game_id": "..."}
 * Re-runs new_game with the original hero/deck/seed -> identical game.
 */
declare(strict_types=1);

require_once ADAPTER_ROOT . '/GameAdapter.php';

$body   = read_json_body();
$gameId = $body['game_id'] ?? ($_GET['game_id'] ?? '');
if ($gameId === '') json_error('game_id required (body or query)', 400);

$adapter = new GameAdapter();
$result  = $adapter->reset((string)$gameId);

json_response($result);
