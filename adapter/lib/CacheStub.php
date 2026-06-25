<?php
/**
 * CacheStub
 * ------------------------------------------------------------------
 * Replaces Talishar's SHMOP + Redis cache with a disk-backed key/value
 * store (datasets/games/{game_id}/cache/{key}.txt). We must define the
 * functions Talishar expects BEFORE any Talishar file is included so
 * the engine binds to our versions, not the real shmop-backed ones.
 *
 * Functions provided
 *   ReadCache($key)
 *   WriteCache($key, $value)
 *   GetCachePiece($gameName, $idx)
 *   SetCachePiece($gameName, $idx, $value)
 *   InvalidateGamestateCache($gameName)
 *   GamestateUpdated($gameName)
 *   GamestateID($gameName)
 *   IsModeAsync($mode)
 *
 * Notes
 *   - Talishar's cache is a "!"-delimited string for the lobby/state
 *     scoreboard (visibility, status, last-update timestamps, …) plus
 *     a separate gamestate.txt for the play state. We mirror both.
 *   - This stub is intentionally NOT thread-safe across processes —
 *     run one adapter port per worker (which is the recommended layout
 *     for the 1000+ games/hr target anyway).
 */

declare(strict_types=1);

final class CacheStub
{
    /** @var array<string,string> in-process memo to amortise disk reads */
    private static array $memo = [];

    public static function register(): void
    {
        // The constants below are referenced sporadically inside Talishar.
        if (!defined('GAMESTATE_NUM_PIECES')) define('GAMESTATE_NUM_PIECES', 80);

        // If the engine has not yet defined these helpers, register ours.
        foreach (['ReadCache','WriteCache','GetCachePiece','SetCachePiece',
                  'InvalidateGamestateCache','GamestateUpdated','GamestateID',
                  'IsModeAsync','IsModeAllowedForSpectators','IsReplay'] as $fn) {
            if (!function_exists($fn)) {
                $impl = [__CLASS__, "fn_$fn"];
                // Dynamic function definition is not allowed at runtime; we instead
                // define wrappers as global functions outside the class (see bottom).
                // This array is informational for sanity checks during dev.
                unset($impl);
            }
        }
    }

    public static function dir(string $gameId): string
    {
        $d = GAMES_ROOT . DIRECTORY_SEPARATOR . $gameId . DIRECTORY_SEPARATOR . 'cache';
        if (!is_dir($d)) @mkdir($d, 0775, true);
        return $d;
    }

    public static function write(string $gameId, string $value): void
    {
        self::$memo[$gameId] = $value;
        file_put_contents(self::dir($gameId) . DIRECTORY_SEPARATOR . 'main.txt', $value);
    }

    public static function read(string $gameId): string
    {
        if (isset(self::$memo[$gameId])) return self::$memo[$gameId];
        $path = self::dir($gameId) . DIRECTORY_SEPARATOR . 'main.txt';
        $val  = is_file($path) ? (string)file_get_contents($path) : '';
        self::$memo[$gameId] = $val;
        return $val;
    }
}

// --- Globally-visible shims that Talishar will resolve --------------------
//
// The engine calls these as bare global functions; we cannot define them
// inside a class. We use function_exists() guards so that on a future PHP
// host that has the real shmop library we can short-circuit and call into
// the real implementation.

if (!function_exists('ReadCache')) {
    function ReadCache(string $key): string {
        // Engine uses GamestateID($name) to point at gamestate-specific storage.
        if (str_starts_with($key, 'GS_')) {
            $gameId = substr($key, 3);
            $path   = GAMES_ROOT . DIRECTORY_SEPARATOR . $gameId . DIRECTORY_SEPARATOR . 'cache' . DIRECTORY_SEPARATOR . 'gamestate.txt';
            return is_file($path) ? (string)file_get_contents($path) : '';
        }
        return CacheStub::read($key);
    }
}

if (!function_exists('WriteCache')) {
    function WriteCache(string $key, string $value): void {
        if (str_starts_with($key, 'GS_')) {
            $gameId = substr($key, 3);
            $dir    = GAMES_ROOT . DIRECTORY_SEPARATOR . $gameId . DIRECTORY_SEPARATOR . 'cache';
            if (!is_dir($dir)) @mkdir($dir, 0775, true);
            file_put_contents($dir . DIRECTORY_SEPARATOR . 'gamestate.txt', $value);
            return;
        }
        CacheStub::write($key, $value);
    }
}

if (!function_exists('GamestateID')) {
    function GamestateID(string $gameName): string {
        return 'GS_' . $gameName;
    }
}

if (!function_exists('WriteGamestateCache')) {
    function WriteGamestateCache($name, $data): void {
        if ($name == 0) return;
        // Match Talishar's SHMOPLibraries flow: it serializes + trims,
        // we just put the raw gamestate string into our disk-backed
        // cache. ReadCache + ReadGamestateCache below mirror this.
        $gameId = (string)$name;
        $dir = GAMES_ROOT . DIRECTORY_SEPARATOR . $gameId . DIRECTORY_SEPARATOR . 'cache';
        if (!is_dir($dir)) @mkdir($dir, 0775, true);
        file_put_contents($dir . DIRECTORY_SEPARATOR . 'gamestate.txt', (string)$data);
    }
}

if (!function_exists('ReadGamestateCache')) {
    function ReadGamestateCache($name): string {
        if ($name == 0) return '';
        $gameId = (string)$name;
        $path = GAMES_ROOT . DIRECTORY_SEPARATOR . $gameId . DIRECTORY_SEPARATOR . 'cache' . DIRECTORY_SEPARATOR . 'gamestate.txt';
        return is_file($path) ? (string)file_get_contents($path) : '';
    }
}

if (!function_exists('InvalidateGamestateCache')) {
    function InvalidateGamestateCache(string $gameName): void {}
}

if (!function_exists('GetCachePiece')) {
    function GetCachePiece(string $gameName, int $idx): string {
        $val   = CacheStub::read($gameName);
        $parts = explode('!', $val);
        return $parts[$idx] ?? '';
    }
}

if (!function_exists('SetCachePiece')) {
    function SetCachePiece(string $gameName, int $idx, $value): void {
        $val   = CacheStub::read($gameName);
        $parts = explode('!', $val);
        while (count($parts) <= $idx) $parts[] = '';
        $parts[$idx] = (string)$value;
        CacheStub::write($gameName, implode('!', $parts));
    }
}

if (!function_exists('InvalidateGamestateCache')) {
    function InvalidateGamestateCache(string $gameName): void {
        // No-op for stub: we always read from disk.
    }
}

if (!function_exists('GamestateUpdated')) {
    function GamestateUpdated(string $gameName): void {
        // No SSE pings in headless mode.
    }
}

// NOTE: IsModeAsync / IsModeAllowedForSpectators live in
// talishar/Libraries/NetworkingLibraries.php and IsReplay / SetHeaders
// live in talishar/Libraries/HTTPLibraries.php. We do NOT stub them
// here — TalisharBoot::ensureBooted() requires those libraries up
// front, so the real implementations are always available. Defining
// stubs first triggered a "Cannot redeclare" fatal in PHP-S when the
// real library was later included.
