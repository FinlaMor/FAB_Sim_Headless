<?php
/**
 * RngHook
 * ------------------------------------------------------------------
 * Seeds PHP's MT random generator deterministically at the top of every
 * adapter request. Talishar's shuffle uses `shuffle()` / `mt_rand()`, so
 * seeding once per (game_id, step_counter) pair makes the entire game
 * reproducible from (seed, action trace) alone.
 *
 * Why include step_counter?
 *   We want the *whole game* deterministic given a seed. We achieve that
 *   by seeding to a function of (seed, step_counter) before every
 *   ProcessInput call. Any internal Talishar call that consumes random
 *   numbers does so against a known seed; the same trace replays bit-
 *   identically because each step starts from the same RNG anchor.
 */

declare(strict_types=1);

final class RngHook
{
    public static function seed(int $seed, int $stepCounter): void
    {
        // Mix seed and counter into a 32-bit space deterministically.
        // We avoid 0 (mt_srand treats 0 specially on some PHP builds).
        $mixed = self::mix($seed, $stepCounter);
        if ($mixed === 0) {
            $mixed = 1;
        }
        mt_srand($mixed);
    }

    public static function mix(int $seed, int $counter): int
    {
        // 32-bit-safe mixer. PHP's int is signed 64-bit, so any
        // multiplication of two 32-bit values can exceed 2^63 and
        // silently promote to float (losing determinism). CRC32 on the
        // composite "seed:counter" string is:
        //   - deterministic across PHP versions
        //   - returns a non-negative 32-bit integer (positive on 64-bit)
        //   - never overflows
        // It isn't crypto-grade, but mt_srand only needs a uniform seed.
        $h = crc32("$seed:$counter");
        // crc32 returns int (32-bit unsigned). Force positive and clamp.
        return ($h < 0 ? -$h : $h) & 0x7FFFFFFF;
    }
}
