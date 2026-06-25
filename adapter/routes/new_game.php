<?php
/**
 * POST /new_game
 * Body: {"hero1": "...", "hero2": "...", "deck1": "...", "deck2": "...", "seed": 12345,
 *        "format": "cc"}   // optional, defaults to "draft" (OMN limited)
 * Response: {"game_id": "...", "state": {...}, "legal_actions": [...], "info": {...}}
 */
declare(strict_types=1);

require_once ADAPTER_ROOT . '/GameAdapter.php';

$body = read_json_body();
require_keys($body, ['hero1', 'hero2', 'deck1', 'deck2', 'seed']);

$adapter = new GameAdapter();
$result  = $adapter->newGame([
    'hero1' => (string)$body['hero1'],
    'hero2' => (string)$body['hero2'],
    'deck1' => (string)$body['deck1'],
    'deck2' => (string)$body['deck2'],
    'seed'  => (int)$body['seed'],
    'format' => isset($body['format']) ? (string)$body['format'] : 'draft',
]);

json_response($result);
