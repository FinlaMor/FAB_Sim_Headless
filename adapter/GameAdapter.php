<?php
/**
 * GameAdapter
 * ------------------------------------------------------------------
 * The single place that *coordinates* a game's life cycle. Stateless
 * across requests; persistence is entirely on disk inside
 * datasets/games/{game_id}/ (or in the stub registry).
 *
 * The class deliberately keeps the public surface small:
 *
 *   - newGame(hero1, hero2, deck1Path, deck2Path, seed) -> game_id
 *   - getState(game_id)      -> array       (full deterministic JSON)
 *   - getActions(game_id)    -> array<Action>
 *   - step(game_id, action)  -> array       (done, reward, winner, next_state)
 *   - reset(game_id)         -> array       (same shape as new_game return)
 *
 * Routing logic decides between stub and real implementations at request
 * time via ADAPTER_MODE; from the outside the API is identical.
 *
 * "real" mode delegates to TalisharBoot (which manages the cwd switch,
 * stubs out CacheLibraries, and includes ParseGamestate.php /
 * ProcessInput()). "stub" mode runs StubGame which simulates a tiny
 * synthetic FAB game with the same state shape, used for smoke tests.
 */

declare(strict_types=1);

require_once ADAPTER_LIB        . '/GameRegistry.php';
require_once ADAPTER_LIB        . '/RngHook.php';
require_once ADAPTER_SERIALIZERS . '/StateSerializer.php';
require_once ADAPTER_SERIALIZERS . '/ActionSerializer.php';

if (ADAPTER_MODE === 'real') {
    require_once ADAPTER_LIB . '/TalisharBoot.php';
} else {
    require_once ADAPTER_LIB . '/StubGame.php';
}

final class GameAdapter
{
    /**
     * Create a new game and return the structured response.
     *
     * @param array{
     *   hero1: string, hero2: string,
     *   deck1: string, deck2: string,
     *   seed:  int
     * } $req
     *
     * @return array{game_id: string, state: array, legal_actions: array, info: array}
     */
    public function newGame(array $req): array
    {
        $gameId = self::makeGameId($req['seed']);
        $record = [
            'game_id'      => $gameId,
            'seed'         => (int)$req['seed'],
            'step_counter' => 0,
            'hero1'        => $req['hero1'],
            'hero2'        => $req['hero2'],
            'deck1_path'   => $req['deck1'],
            'deck2_path'   => $req['deck2'],
            'mode'         => ADAPTER_MODE,
            'created_at'   => microtime(true),
        ];

        if (ADAPTER_MODE === 'real') {
            TalisharBoot::createGame($gameId, $req);
        } else {
            StubGame::createGame($gameId, $req);
        }

        GameRegistry::put($gameId, $record);

        return $this->snapshot($gameId, /*done=*/false, /*reward=*/0.0, /*winner=*/null);
    }

    /** @return array */
    public function getState(string $gameId): array
    {
        $record = GameRegistry::getOrFail($gameId);
        RngHook::seed($record['seed'], $record['step_counter']);
        if (ADAPTER_MODE === 'real') {
            $raw = TalisharBoot::readState($gameId);
        } else {
            $raw = StubGame::readState($gameId);
        }
        return StateSerializer::serialize($raw);
    }

    /** @return array<int, array> */
    public function getActions(string $gameId): array
    {
        $record = GameRegistry::getOrFail($gameId);
        RngHook::seed($record['seed'], $record['step_counter']);
        if (ADAPTER_MODE === 'real') {
            $raw = TalisharBoot::readState($gameId);
        } else {
            $raw = StubGame::readState($gameId);
        }
        return ActionSerializer::enumerate($raw);
    }

    /**
     * Apply an action. Returns the standard {done, reward, winner, next_state, legal_actions} payload.
     */
    public function step(string $gameId, int $actionId): array
    {
        $record = GameRegistry::getOrFail($gameId);
        $record['step_counter']++;
        RngHook::seed($record['seed'], $record['step_counter']);

        // Resolve adapter action_id -> Talishar mode dispatch.
        if (ADAPTER_MODE === 'real') {
            $raw     = TalisharBoot::readState($gameId);
            $actions = ActionSerializer::enumerate($raw);
            $action  = self::resolveAction($actions, $actionId);
            TalisharBoot::applyAction($gameId, $action);
        } else {
            $raw     = StubGame::readState($gameId);
            $actions = ActionSerializer::enumerate($raw);
            $action  = self::resolveAction($actions, $actionId);
            StubGame::applyAction($gameId, $action);
        }

        GameRegistry::put($gameId, $record);

        // Re-read after mutation to compute reward / done.
        $post   = (ADAPTER_MODE === 'real')
            ? TalisharBoot::readState($gameId)
            : StubGame::readState($gameId);
        $winner = self::extractWinner($post);
        $done   = $winner !== null;
        $reward = $done ? (float)self::rewardFromWinner($winner) : 0.0;

        return $this->snapshot($gameId, $done, $reward, $winner);
    }

    public function reset(string $gameId): array
    {
        $record = GameRegistry::getOrFail($gameId);
        return $this->newGame([
            'hero1' => $record['hero1'],
            'hero2' => $record['hero2'],
            'deck1' => $record['deck1_path'],
            'deck2' => $record['deck2_path'],
            'seed'  => $record['seed'],
        ]);
    }

    // ---- helpers -------------------------------------------------------

    private function snapshot(string $gameId, bool $done, float $reward, ?int $winner): array
    {
        $raw    = (ADAPTER_MODE === 'real')
            ? TalisharBoot::readState($gameId)
            : StubGame::readState($gameId);
        $state  = StateSerializer::serialize($raw);
        $legal  = ActionSerializer::enumerate($raw);

        return [
            'game_id'       => $gameId,
            'done'          => $done,
            'reward'        => $reward,
            'winner'        => $winner,
            'state'         => $state,
            'legal_actions' => $legal,
            'info'          => [
                'mode'         => ADAPTER_MODE,
                'step_counter' => GameRegistry::get($gameId)['step_counter'] ?? 0,
            ],
        ];
    }

    private static function makeGameId(int $seed): string
    {
        // Talishar's IsGameNameValid + ParseGamestate.php:10 both insist
        // the game name is numeric (it's used as a directory name AND a
        // file-naming key under Games/). We fold the seed + a random
        // 32-bit nonce into a single 18-digit numeric string so the IDs
        // collide-resistantly distinguish concurrent games while still
        // satisfying the numeric check.
        $nonce = random_int(0, PHP_INT_MAX);
        // Use modulo to keep within 18 decimal digits even on 32-bit PHP.
        $id = ((int)$seed * 1_000_000_000) + ($nonce % 1_000_000_000);
        // Ensure positive.
        if ($id < 0) $id = -$id;
        return (string)$id;
    }

    /**
     * Look up the structured action matching the integer action_id the
     * client chose. Errors out if the id is not in the current legal set.
     */
    private static function resolveAction(array $actions, int $actionId): array
    {
        foreach ($actions as $a) {
            if ((int)$a['action_id'] === $actionId) {
                return $a;
            }
        }
        json_error("action_id $actionId is not in the current legal action set", 409, [
            'legal_action_ids' => array_map(fn($a) => $a['action_id'], $actions),
        ]);
        // unreachable
        return [];
    }

    private static function extractWinner(array $raw): ?int
    {
        $w = (int)($raw['winner'] ?? 0);
        return ($w === 1 || $w === 2) ? $w : null;
    }

    /** Zero-sum +1 / -1 from the perspective of player 1. */
    private static function rewardFromWinner(int $winner): int
    {
        return $winner === 1 ? 1 : -1;
    }
}
