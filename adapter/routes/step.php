<?php
/**
 * POST /step
 * Body: {"game_id": "...", "action_id": 7}
 * Response: {"done": bool, "reward": float, "winner": null|1|2, "state": {...}, "legal_actions": [...]}
 */
declare(strict_types=1);

require_once ADAPTER_ROOT . '/GameAdapter.php';

$body = read_json_body();
require_keys($body, ['game_id', 'action_id']);

$adapter = new GameAdapter();
$result  = $adapter->step((string)$body['game_id'], (int)$body['action_id']);

json_response($result);
