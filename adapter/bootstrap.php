<?php
/**
 * bootstrap.php
 * ------------------------------------------------------------------
 * Loaded by every adapter HTTP request. Responsibilities:
 *   - Load environment configuration (.env style via getenv()).
 *   - Establish absolute paths to the adapter, talishar/, and games/.
 *   - Switch CWD to talishar/ so Talishar's relative include paths work.
 *   - Register a JSON error handler so HTTP responses never leak PHP HTML.
 *   - Provide tiny helpers (json_response, read_json_body, require_keys).
 *   - Choose run mode: ADAPTER_MODE = "stub" (default) or "real".
 *
 * Production note: this file is intentionally framework-free. Every line
 * matters because it runs for every /step request on the hot path. Stick
 * to plain PHP and let the OPcache amortise it.
 */

declare(strict_types=1);

// -----------------------------------------------------------------------
// Paths
// -----------------------------------------------------------------------
define('ADAPTER_ROOT', __DIR__);
define('PROJECT_ROOT', dirname(__DIR__));
define('TALISHAR_ROOT', PROJECT_ROOT . DIRECTORY_SEPARATOR . 'talishar');
define('GAMES_ROOT',    PROJECT_ROOT . DIRECTORY_SEPARATOR . 'datasets' . DIRECTORY_SEPARATOR . 'games');
define('ADAPTER_LIB',   ADAPTER_ROOT . DIRECTORY_SEPARATOR . 'lib');
define('ADAPTER_SERIALIZERS', ADAPTER_ROOT . DIRECTORY_SEPARATOR . 'serializers');
define('ADAPTER_ROUTES', ADAPTER_ROOT . DIRECTORY_SEPARATOR . 'routes');

if (!is_dir(GAMES_ROOT)) {
    @mkdir(GAMES_ROOT, 0775, true);
}

// -----------------------------------------------------------------------
// Run mode: "stub" exercises the Python plumbing without Talishar PHP.
//           "real" boots Talishar from talishar/ (needs the adapter Docker
//           image with PHP 8.1 + shmop + redis).
// -----------------------------------------------------------------------
define('ADAPTER_MODE', getenv('ADAPTER_MODE') ?: 'stub');
define('ADAPTER_DEBUG', filter_var(getenv('ADAPTER_DEBUG') ?: '0', FILTER_VALIDATE_BOOLEAN));

if (ADAPTER_DEBUG) {
    error_reporting(E_ALL);
    ini_set('display_errors', '1');
} else {
    ini_set('display_errors', '0');
    error_reporting(E_ERROR | E_PARSE);
}

// Production: kill the 1-second timeout that Talishar's ProcessInputAPI sets.
@set_time_limit(0);
@ini_set('max_execution_time', '0');

// Buffer ALL stdout for the duration of the request. Talishar emits PHP
// warnings/notices and occasional echo()s as a side effect of running
// game logic; without buffering, they end up prefixed to our JSON
// response and break the client's parser. `json_response()` calls
// ob_clean() before emitting so the buffer is discarded on the happy
// path. On uncaught errors the shutdown handler dumps the captured
// stderr-equivalent into the `debug` key when ADAPTER_DEBUG=1.
ob_start();

// -----------------------------------------------------------------------
// JSON output helpers
// -----------------------------------------------------------------------
function json_response(array $payload, int $status = 200): void
{
    // Drain anything Talishar / PHP wrote to stdout before this point.
    // Headers haven't been sent yet because the entire response was
    // buffered at the top of bootstrap.php.
    $captured = '';
    while (ob_get_level() > 0) {
        $captured .= ob_get_clean();
    }
    if (ADAPTER_DEBUG && $captured !== '') {
        $payload['_debug_stdout'] = $captured;
    }
    http_response_code($status);
    header('Content-Type: application/json; charset=utf-8');
    header('Cache-Control: no-store');
    echo json_encode($payload, JSON_UNESCAPED_SLASHES | JSON_PRESERVE_ZERO_FRACTION);
    exit;
}

function json_error(string $message, int $status = 400, array $extra = []): void
{
    json_response(array_merge(['error' => $message], $extra), $status);
}

function read_json_body(): array
{
    $raw = file_get_contents('php://input');
    if ($raw === '' || $raw === false) {
        return [];
    }
    $decoded = json_decode($raw, true);
    if (!is_array($decoded)) {
        json_error('Request body must be a JSON object', 400);
    }
    return $decoded;
}

function require_keys(array $arr, array $keys): void
{
    $missing = [];
    foreach ($keys as $k) {
        if (!array_key_exists($k, $arr)) {
            $missing[] = $k;
        }
    }
    if ($missing) {
        json_error('Missing required field(s): ' . implode(', ', $missing), 422);
    }
}

// -----------------------------------------------------------------------
// Fatal error -> JSON. Without this, a PHP fatal would emit a 500 HTML
// page and the Python client would see an opaque JSON decode error.
// -----------------------------------------------------------------------
set_exception_handler(function (\Throwable $e) {
    $payload = ['error' => $e->getMessage(), 'type' => get_class($e)];
    if (ADAPTER_DEBUG) {
        $payload['file']  = $e->getFile();
        $payload['line']  = $e->getLine();
        $payload['trace'] = explode("\n", $e->getTraceAsString());
    }
    json_response($payload, 500);
});

register_shutdown_function(function () {
    $err = error_get_last();
    if ($err && in_array($err['type'], [E_ERROR, E_PARSE, E_CORE_ERROR, E_COMPILE_ERROR], true)) {
        if (!headers_sent()) {
            json_response([
                'error' => $err['message'],
                'file'  => ADAPTER_DEBUG ? $err['file'] : null,
                'line'  => ADAPTER_DEBUG ? $err['line'] : null,
            ], 500);
        }
    }
});

// -----------------------------------------------------------------------
// Autoload adapter classes (PSR-0-ish, no Composer dependency).
// -----------------------------------------------------------------------
spl_autoload_register(function (string $class) {
    $candidates = [
        ADAPTER_ROOT . DIRECTORY_SEPARATOR . $class . '.php',
        ADAPTER_LIB  . DIRECTORY_SEPARATOR . $class . '.php',
        ADAPTER_SERIALIZERS . DIRECTORY_SEPARATOR . $class . '.php',
    ];
    foreach ($candidates as $path) {
        if (is_file($path)) {
            require_once $path;
            return;
        }
    }
});

require_once ADAPTER_LIB . DIRECTORY_SEPARATOR . 'RngHook.php';
require_once ADAPTER_LIB . DIRECTORY_SEPARATOR . 'GameRegistry.php';
