<?php
/** GET /health — liveness probe. Cheap, no game lookup. */
declare(strict_types=1);

json_response([
    'ok'             => true,
    'service'        => 'fab-sim-headless-adapter',
    'mode'           => ADAPTER_MODE,
    'php_version'    => PHP_VERSION,
    'talishar_clone' => is_dir(TALISHAR_ROOT . '/Classes') ? 'present' : 'missing',
    'time'           => microtime(true),
]);
