<?php
/**
 * api.php — front controller for the headless Talishar adapter.
 *
 * Routing is intentionally trivial: the path + method maps to a file in
 * routes/. Keeping the dispatcher inline (no Symfony / Laravel) keeps the
 * cold path under 1 ms so we can hit 1000+ games/hr.
 *
 * Endpoints
 *   GET  /health
 *   POST /new_game     {hero1, hero2, deck1, deck2, seed}
 *   GET  /state        ?game_id=...
 *   GET  /actions      ?game_id=...
 *   POST /step         {game_id, action_id}
 *   POST /reset        {game_id}  (no body required if you also pass ?game_id)
 *
 * Run locally inside the adapter container:
 *     php -S 0.0.0.0:8000 -t adapter adapter/api.php
 *
 * The router below also handles routing when api.php is the -t document,
 * so a single PHP built-in server invocation covers every endpoint.
 */

declare(strict_types=1);

require __DIR__ . '/bootstrap.php';

$method = $_SERVER['REQUEST_METHOD'] ?? 'GET';
$uri    = $_SERVER['REQUEST_URI']    ?? '/';
$path   = parse_url($uri, PHP_URL_PATH) ?: '/';
$path   = '/' . trim($path, '/');

// Strip a trailing index entry from the built-in server (`/api.php/...`).
$path = preg_replace('#^/api\.php#', '', $path) ?: '/';

// The Talishar FE's dev proxy targets `.../game/<Endpoint>.php`; accept that
// `/game` prefix transparently so the real frontend needs no patching.
$path = preg_replace('#^/game(?=/)#', '', $path) ?: '/';

$routes = [
    'GET'  => [
        '/'        => 'health.php',
        '/health'  => 'health.php',
        '/state'   => 'state.php',
        '/actions' => 'actions.php',
        // --- Game review / replay surface for the real Talishar FE ---
        '/GetNextTurn.php'  => 'spectate.php',
        '/GetUpdateSSE.php' => 'sse.php',             // live SSE feed for the FE
        '/ProcessInput.php' => 'process_input.php',   // replay turn-hop (GET)
        '/game_list'        => 'game_list.php',
    ],
    'POST' => [
        '/new_game' => 'new_game.php',
        '/step'     => 'step.php',
        '/reset'    => 'reset.php',
    ],
];

if (!isset($routes[$method][$path])) {
    json_error("No route for $method $path", 404, [
        'available' => array_map(fn($v) => array_keys($v), $routes),
    ]);
}

require ADAPTER_ROUTES . DIRECTORY_SEPARATOR . $routes[$method][$path];
