<?php
/**
 * StateSerializer
 * ------------------------------------------------------------------
 * Turn the raw Talishar (or StubGame) state array into the canonical
 * JSON shape consumed by Python:
 *
 *   {
 *     "turn":             3,
 *     "phase":            "M",
 *     "active_player":    1,
 *     "priority_player":  1,
 *     "action_points":    1,
 *     "winner":           null | 1 | 2,
 *     "players": [{...p1...}, {...p2...}],
 *     "combat_chain":   [...],
 *     "stack":          [...],
 *     "decision_queue": [...],
 *     "current_turn_effects": [...],
 *     "events":         [...]
 *   }
 *
 * "Deterministic JSON" means: keys are emitted in a stable order, no
 * timestamps, no random IDs. The Python side hashes this for replay
 * verification.
 */

declare(strict_types=1);

final class StateSerializer
{
    public static function serialize(array $raw): array
    {
        $winner = (int)($raw['winner'] ?? 0);
        $payload = [
            'turn'             => (int)($raw['turn_number'] ?? $raw['currentTurn'] ?? 0),
            'phase'            => (string)($raw['phase'] ?? ($raw['turn'][0] ?? '')),
            'subphase'         => self::subphase($raw),
            'active_player'    => (int)($raw['main_player'] ?? $raw['mainPlayer'] ?? 1),
            'priority_player'  => (int)($raw['priority_player'] ?? $raw['currentPlayer'] ?? 1),
            'action_points'    => (int)($raw['action_points'] ?? $raw['actionPoints'] ?? 0),
            'winner'           => $winner === 0 ? null : $winner,
            'players'          => array_map(fn($p) => self::serializePlayer($p), $raw['players'] ?? []),
            'combat_chain'     => self::combatChain($raw['combatChain'] ?? $raw['combat_chain'] ?? []),
            'combat'           => self::combatInfo($raw),
            'stack'            => self::stack($raw['layers'] ?? $raw['stack'] ?? []),
            'links'            => $raw['chainLinks'] ?? [],
            'decision_queue'   => self::decisionQueue($raw),
            'current_turn_effects' => self::effects($raw['currentTurnEffects'] ?? []),
            'next_turn_effects'    => $raw['nextTurnEffects'] ?? [],
            'events'           => $raw['events'] ?? [],
            'last_played'      => $raw['lastPlayed'] ?? [],
            'landmarks'        => $raw['landmarks'] ?? [],
            'pending_decisions' => $raw['pending_decisions'] ?? [],
        ];
        return $payload;
    }

    /**
     * Surface the engine's cached combat numbers so the gameplay bot can
     * make informed block/no-block decisions. These are the CCS_* slots
     * CacheCombatResult() fills (combatChainState indices); we only read
     * them, never recompute. attack_power/total_block are the cached
     * totals; pending_damage is what would land on the target right now.
     */
    private static function combatInfo(array $raw): array
    {
        $ccs = $raw['combatChainState'] ?? [];
        $cc  = $raw['combatChain'] ?? [];
        $active = (is_array($cc) && count($cc) > 0) ? 1 : 0;
        $power = (int)($ccs[22] ?? 0);   // CCS_CachedTotalPower (DYNAMIC, buffs in)
        $block = (int)($ccs[23] ?? 0);   // CCS_CachedTotalBlock
        $link  = (int)($ccs[17] ?? 0);   // CCS_LinkTotalPower
        $eff   = $power > 0 ? $power : $link;
        // DYNAMIC attack properties — these are the engine's *current* values
        // for the incoming attack (granted or printed), not the cards' base
        // stats. A granted go again or dominate flips the right block, so the
        // defender must see them. Indices: CCS_CachedGoAgain=48 (this already
        // reflects go again granted by boost, per CR 8.3.9),
        // CCS_CachedDominateActive=27.
        $goAgain  = ((string)($ccs[48] ?? '0') === '1') ? 1 : 0;
        $dominate = ((int)($ccs[27] ?? 0) === 1) ? 1 : 0;
        return [
            'active'         => $active,
            'attacker'       => $active ? (int)($cc[1] ?? 0) : 0,   // player id of the attacker
            'attack_power'   => $eff,
            'total_block'    => $block,
            'pending_damage' => max($eff - $block, 0),
            'attack_target'  => (string)($ccs[16] ?? 'NA'),         // CCS_AttackTarget
            'damage_to_hero' => (int)($ccs[49] ?? 0),               // CCS_AttackDamageDealtToHero
            'go_again'       => $goAgain,    // attack continues -> tempo swing
            'dominate'       => $dominate,   // can only be blocked by ONE card
        ];
    }

    private static function subphase(array $raw): string
    {
        // Talishar packs subphase + responder + options into $turn[0..2].
        $turn = $raw['turn'] ?? [];
        return (string)($turn[2] ?? '');
    }

    private static function serializePlayer(array $p): array
    {
        // The character zone is [hero, weapon, head, chest, arms, legs, ...] at
        // CharacterPieces() stride. De-stride once, then split the hero out so
        // it is surfaced as `hero` and NOT double-counted as an equipment token.
        $charStride = self::stride('CharacterPieces', 15);
        $charRaw    = $p['equipment'] ?? $p['charequip'] ?? [];
        $charCards  = self::cards($charRaw, $charStride);
        $hero       = (string)($p['hero'] ?? '');
        if ($hero === '' && isset($charCards[0])) $hero = (string)$charCards[0];
        $stripHero  = ($hero !== '' && isset($charCards[0]) && (string)$charCards[0] === $hero);
        $equipment  = $stripHero ? array_slice($charCards, 1) : $charCards;
        $equipReady = self::pick($charRaw, $charStride, 1);
        if ($stripHero) $equipReady = array_slice($equipReady, 1);

        return [
            'player_id'  => (int)($p['player_id'] ?? 0),
            'hero'       => $hero,
            'intellect'  => isset($p['intellect']) ? (int)$p['intellect'] : null,
            'health'     => isset($p['health']) ? (int)$p['health'] : null,
            'resources'  => (int)($p['resources'] ?? 0),
            // Card zones are DE-STRIDED: the raw engine globals store N fields
            // per card (counters, uniqueIDs, face state, ...), e.g. CharEquip
            // is 15 fields/card and Discard is 3/card. Passing them through
            // flat meant two-thirds of the Python model's zone tokens were
            // stride junk (';;31', '0', 'UP', ...) and the card vocab exploded
            // to ~8.5k entries for a 209-card cube. Strides mirror the
            // engine's *Pieces() helpers (fallback constants for safety).
            'hand'       => self::cards($p['hand'] ?? [], 1),
            'arsenal'    => self::cards($p['arsenal'] ?? [], self::stride('ArsenalPieces', 7)),
            'equipment'  => $equipment,
            'graveyard'  => self::cards($p['graveyard'] ?? $p['discard'] ?? [], self::stride('DiscardPieces', 3)),
            'banished'   => self::cards($p['banished']  ?? $p['banish']   ?? [], self::stride('BanishPieces', 3)),
            'pitch'      => self::cards($p['pitch']     ?? [], self::stride('PitchPieces', 2)),
            'auras'      => self::cards($p['auras']     ?? [], self::stride('AuraPieces', 14)),
            'items'      => self::cards($p['items']     ?? [], self::stride('ItemPieces', 14)),
            'allies'     => self::cards($p['allies']    ?? [], self::stride('AllyPieces', 15)),
            'permanents' => self::cards($p['permanents'] ?? [], self::stride('PermanentPieces', 4)),
            'soul'       => self::cards($p['soul']      ?? [], self::stride('SoulPieces', 1)),
            // Per-card STATE (the meaningful stride fields the id extraction
            // drops): counters, life, face-up/ready flags. Index map verified
            // against BuildGameState.php's own rendering:
            //   arsenal[1]=facing(UP/DOWN) [3]=counters | auras[1]=status(2=
            //   ready) [2]=counters | items[1]=counters [2]=usable(2=ready) |
            //   allies[1]=status [2]=life | charequip[1]=status.
            'arsenal_facing' => self::pick($p['arsenal'] ?? [], self::stride('ArsenalPieces', 7), 1),
            'arsenal_counts' => self::pick($p['arsenal'] ?? [], self::stride('ArsenalPieces', 7), 3),
            'auras_counts'   => self::pick($p['auras'] ?? [], self::stride('AuraPieces', 14), 2),
            'auras_ready'    => self::pick($p['auras'] ?? [], self::stride('AuraPieces', 14), 1),
            'items_counts'   => self::pick($p['items'] ?? [], self::stride('ItemPieces', 14), 1),
            'items_ready'    => self::pick($p['items'] ?? [], self::stride('ItemPieces', 14), 2),
            'allies_counts'  => self::pick($p['allies'] ?? [], self::stride('AllyPieces', 15), 2),
            'allies_ready'   => self::pick($p['allies'] ?? [], self::stride('AllyPieces', 15), 1),
            'equipment_ready' => $equipReady,
            'effects'    => self::flat($p['effects'] ?? $p['charactereffects'] ?? []),
            'class_state' => self::flat($p['classstate'] ?? []),
            'turn_stats' => $p['turn_stats'] ?? $p['turnstats'] ?? [],
            'card_stats' => $p['card_stats'] ?? $p['cardstats'] ?? [],
            'deck_count' => isset($p['deck'])
                ? (int)(count($p['deck']) / max(1, self::stride('DeckPieces', 1)))
                : 0,
        ];
    }

    /** Engine stride for a zone; 1 in stub mode (StubGame zones are plain lists). */
    private static function stride(string $fn, int $fallback): int
    {
        if (defined('ADAPTER_MODE') && ADAPTER_MODE !== 'real') return 1;
        return function_exists($fn) ? max(1, (int)$fn()) : $fallback;
    }

    /** Extract card ids (every $stride-th element) from a raw zone array. */
    private static function cards($v, int $stride): array
    {
        $list = self::flat($v);
        if ($stride <= 1) return array_values(array_filter($list, fn($c) => $c !== '' && $c !== null));
        $out = [];
        for ($i = 0; $i < count($list); $i += $stride) {
            $c = $list[$i];
            if ($c !== '' && $c !== null) $out[] = $c;
        }
        return $out;
    }

    /** Flat-list passthrough (always returns an indexed array). */
    private static function flat($v): array
    {
        if (is_array($v)) return array_values($v);
        if ($v === null || $v === '') return [];
        return [$v];
    }

    private static function combatChain(array $cc): array
    {
        // Talishar uses 12-piece stride. Try to chunk if we got a flat list.
        if ($cc === []) return [];
        if (is_array($cc[0] ?? null)) return $cc;        // already structured

        $stride = 12;
        $out    = [];
        for ($i = 0; $i + $stride <= count($cc); $i += $stride) {
            $out[] = [
                'card_id'        => $cc[$i + 0],
                'player'         => (int)$cc[$i + 1],
                'from'           => $cc[$i + 2],
                'resources_paid' => (int)$cc[$i + 3],
                'reprise_active' => (int)$cc[$i + 4],
                'power_mod'      => (int)$cc[$i + 5],
                'defense_mod'    => (int)$cc[$i + 6],
                'unique_id'      => $cc[$i + 7],
                'origin_unique'  => $cc[$i + 8],
                'orig_card_id'   => $cc[$i + 9],
                'static_buffs'   => $cc[$i + 10],
                'num_used'       => (int)$cc[$i + 11],
            ];
        }
        return $out;
    }

    /**
     * Per-card state extraction: element at $idx within each $stride block,
     * aligned 1:1 with the card list cards() returns (slots whose card id is
     * empty are skipped, matching cards()).
     */
    private static function pick($v, int $stride, int $idx): array
    {
        $list = self::flat($v);
        if ($stride <= 1) return [];
        $out = [];
        for ($i = 0; $i < count($list); $i += $stride) {
            $c = $list[$i];
            if ($c === '' || $c === null) continue;
            $out[] = $list[$i + $idx] ?? null;
        }
        return $out;
    }

    private static function stack(array $layers): array
    {
        // Layers are stored flat at LayerPieces() stride: [kind-or-card,
        // player, params, ...]. "Special" kinds (TRIGGER/ABILITY/ATTACK/...)
        // carry the actual card at +2 (mirrors BuildGameState's own display
        // logic). Emit TOP-FIRST (the next layer to resolve is element 0).
        $stride  = (defined('ADAPTER_MODE') && ADAPTER_MODE === 'real'
                    && function_exists('LayerPieces')) ? LayerPieces() : 0;
        if ($stride < 2) return array_values($layers);   // stub passthrough
        $special = ['LAYER', 'TRIGGER', 'MELD', 'PRETRIGGER', 'ABILITY', 'ATTACK'];
        $out = [];
        for ($i = count($layers) - $stride; $i >= 0; $i -= $stride) {
            $name = in_array($layers[$i], $special, true)
                ? ($layers[$i + 2] ?? '') : $layers[$i];
            $out[] = ['card_id' => $name, 'player' => (int)($layers[$i + 1] ?? 0)];
        }
        return $out;
    }

    /** currentTurnEffects (stride 4: [id, player, ...]) -> [{card_id, player}]. */
    private static function effects(array $fx): array
    {
        $stride = (defined('ADAPTER_MODE') && ADAPTER_MODE === 'real'
                   && function_exists('CurrentTurnEffectPieces'))
            ? CurrentTurnEffectPieces() : 0;
        if ($stride < 2) return array_values($fx);       // stub passthrough
        $out = [];
        for ($i = 0; $i + 1 < count($fx); $i += $stride) {
            $id = (string)$fx[$i];
            if ($id === '' || $id === '-') continue;
            // Effect ids are "cardID" or "cardID-SUFFIX"; keep the card part.
            $card = explode('-', $id)[0];
            $out[] = ['card_id' => $card, 'player' => (int)($fx[$i + 1] ?? 0)];
        }
        return $out;
    }

    private static function decisionQueue(array $raw): array
    {
        return [
            'queue'  => $raw['decisionQueue']  ?? [],
            'vars'   => $raw['dqVars']         ?? [],
            'state'  => $raw['dqState']        ?? [],
            'turn'   => $raw['turn']           ?? [],
        ];
    }
}
