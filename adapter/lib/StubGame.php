<?php
/**
 * StubGame
 * ------------------------------------------------------------------
 * Deterministic synthetic FAB-shaped game used for smoke testing the
 * Python wiring without booting Talishar. NEVER ships rules logic that
 * pretends to be real FAB; it is openly a 50-turn life-tally puppet.
 *
 *   Each "turn" the active player picks one of three actions:
 *     - ATTACK     -> deal 2-5 damage to opponent (RNG-seeded)
 *     - DEFEND     -> heal self 0-2
 *     - PASS       -> nothing, but advances priority
 *
 * The state shape returned by readState() is intentionally a *subset*
 * of the real Talishar state shape so StateSerializer / ActionSerializer
 * can be exercised against it without code-path divergence.
 *
 * Persisted to datasets/games/{id}/stub_state.json.
 */

declare(strict_types=1);

final class StubGame
{
    private const MAX_TURNS = 50;

    public static function createGame(string $gameId, array $req): void
    {
        $state = [
            'mode'           => 'stub',
            'hero1'          => $req['hero1'],
            'hero2'          => $req['hero2'],
            'deck1_path'     => $req['deck1'],
            'deck2_path'     => $req['deck2'],
            'seed'           => (int)$req['seed'],

            // Talishar-style scalars
            'turn_number'    => 1,
            'phase'          => 'M',
            'current_player' => 1,
            'main_player'    => 1,
            'priority_player' => 1,
            'action_points'  => 1,
            'winner'         => 0,

            // Player state (subset of real engine)
            'players' => [
                self::startPlayer(1, $req['hero1']),
                self::startPlayer(2, $req['hero2']),
            ],

            // Engine helpers
            'combat_chain' => [],
            'stack'        => [],
            'pending_decisions' => [],
        ];
        self::write($gameId, $state);
    }

    public static function readState(string $gameId): array
    {
        $path = self::path($gameId);
        if (!is_file($path)) {
            json_error("Stub state missing for $gameId", 500);
        }
        $raw = file_get_contents($path);
        $state = json_decode($raw, true);
        if (!is_array($state)) {
            json_error("Stub state corrupted for $gameId", 500);
        }
        return $state;
    }

    public static function applyAction(string $gameId, array $action): void
    {
        $state = self::readState($gameId);
        if ($state['winner'] !== 0) return;

        $me  = $state['current_player'] - 1;
        $opp = 1 - $me;
        $players = &$state['players'];

        switch ($action['type']) {
            case 'ATTACK':
                $damage = mt_rand(2, 5);
                $players[$opp]['health'] = max(0, $players[$opp]['health'] - $damage);
                $players[$me]['turn_stats']['attacks_made']++;
                break;
            case 'DEFEND':
                $heal = mt_rand(0, 2);
                $players[$me]['health'] = min(40, $players[$me]['health'] + $heal);
                $players[$me]['turn_stats']['defends_made']++;
                break;
            case 'PASS':
            default:
                // no-op
                break;
        }

        // Lose condition
        if ($players[$opp]['health'] <= 0) {
            $state['winner'] = $me + 1;
        }

        // Advance priority — toggle each step
        $state['current_player']  = $opp + 1;
        $state['priority_player'] = $opp + 1;
        $state['main_player']     = $opp + 1;
        $state['turn_number']++;

        if ($state['winner'] === 0 && $state['turn_number'] > self::MAX_TURNS) {
            // Sudden death: whoever has more HP wins; tie -> player 1
            $state['winner'] = $players[0]['health'] >= $players[1]['health'] ? 1 : 2;
        }

        self::write($gameId, $state);
    }

    private static function startPlayer(int $id, string $hero): array
    {
        $sampleHand = ['stub_attack_red', 'stub_attack_yellow', 'stub_defend_red', 'stub_pitch_blue'];
        return [
            'player_id'   => $id,
            'hero'        => $hero,
            'health'      => 40,
            'resources'   => 0,
            'hand'        => $sampleHand,
            'arsenal'     => [],
            'equipment'   => [],
            'graveyard'   => [],
            'banished'    => [],
            'pitch'       => [],
            'auras'       => [],
            'effects'     => [],
            'turn_stats'  => ['attacks_made' => 0, 'defends_made' => 0],
        ];
    }

    private static function path(string $gameId): string
    {
        return GAMES_ROOT . DIRECTORY_SEPARATOR . $gameId . DIRECTORY_SEPARATOR . 'stub_state.json';
    }

    private static function write(string $gameId, array $state): void
    {
        $path = self::path($gameId);
        $dir  = dirname($path);
        if (!is_dir($dir)) @mkdir($dir, 0775, true);
        file_put_contents($path, json_encode($state, JSON_PRETTY_PRINT));
    }
}
