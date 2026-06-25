<?php
/**
 * GameRegistry
 * ------------------------------------------------------------------
 * Tiny JSON-on-disk store mapping game_id -> {seed, step_counter, hero,
 * deck paths, run mode, timestamps}. Used by GameAdapter to look up
 * cross-request metadata without standing up a database.
 *
 * Files live at  datasets/games/{game_id}/registry.json
 *
 * The adapter is single-process per port, so a flock() on write is
 * sufficient for crash safety. If you later run several adapter
 * workers, partition game_id space by worker (which is the recommended
 * deployment model anyway).
 */

declare(strict_types=1);

final class GameRegistry
{
    private static function dir(string $gameId): string
    {
        // Paranoia: refuse anything that isn't [a-zA-Z0-9_-]+
        if (!preg_match('/^[A-Za-z0-9_-]+$/', $gameId)) {
            json_error('Invalid game_id', 400);
        }
        $dir = GAMES_ROOT . DIRECTORY_SEPARATOR . $gameId;
        if (!is_dir($dir)) {
            @mkdir($dir, 0775, true);
        }
        return $dir;
    }

    private static function path(string $gameId): string
    {
        return self::dir($gameId) . DIRECTORY_SEPARATOR . 'registry.json';
    }

    public static function put(string $gameId, array $record): void
    {
        $path = self::path($gameId);
        $json = json_encode($record, JSON_PRETTY_PRINT | JSON_UNESCAPED_SLASHES);
        if ($json === false) {
            throw new RuntimeException('Failed to encode registry record');
        }
        $fh = fopen($path, 'c+b');
        if ($fh === false) {
            throw new RuntimeException("Cannot open $path");
        }
        try {
            flock($fh, LOCK_EX);
            ftruncate($fh, 0);
            rewind($fh);
            fwrite($fh, $json);
            fflush($fh);
        } finally {
            flock($fh, LOCK_UN);
            fclose($fh);
        }
    }

    public static function get(string $gameId): ?array
    {
        $path = self::path($gameId);
        if (!is_file($path)) {
            return null;
        }
        $raw = file_get_contents($path);
        if ($raw === false || $raw === '') {
            return null;
        }
        $decoded = json_decode($raw, true);
        return is_array($decoded) ? $decoded : null;
    }

    public static function getOrFail(string $gameId): array
    {
        $rec = self::get($gameId);
        if ($rec === null) {
            json_error("Unknown game_id: $gameId", 404);
        }
        return $rec;
    }

    /** @return string[] */
    public static function listIds(): array
    {
        if (!is_dir(GAMES_ROOT)) return [];
        $ids = [];
        foreach (scandir(GAMES_ROOT) ?: [] as $entry) {
            if ($entry === '.' || $entry === '..') continue;
            if (is_file(GAMES_ROOT . DIRECTORY_SEPARATOR . $entry . DIRECTORY_SEPARATOR . 'registry.json')) {
                $ids[] = $entry;
            }
        }
        return $ids;
    }
}
