<?php
/**
 * ActionSerializer
 * ------------------------------------------------------------------
 * Enumerates every legal action available to the current priority
 * player and emits them as structured records:
 *
 *   {
 *     "action_id": 7,
 *     "type":      "PLAY_FROM_HAND",
 *     "player_id": 1,
 *     "card_id":   "wtr123",
 *     "targets":   [],
 *     "cost":      {"resources": 2},
 *     "talishar_mode":      27,
 *     "talishar_button":    "wtr123",
 *     "talishar_card_id":   "0",
 *     "talishar_chk_count": 0,
 *     "talishar_chk_input": [],
 *     "talishar_input_text": ""
 *   }
 *
 * The `talishar_*` fields are what GameAdapter::step() feeds back to
 * ProcessInput(). The leading fields are what the Python bots see.
 *
 * In REAL mode we lean on Talishar's existing helpers (PlayableCardSubset,
 * AbilityCost, CharacterPieces, ...) to determine legality so we never
 * re-implement card rules. In STUB mode we synthesise a fixed three-
 * action menu (ATTACK / DEFEND / PASS) for plumbing tests.
 */

declare(strict_types=1);

final class ActionSerializer
{
    // Affordability filter — drops unaffordable plays so the engine's pay phase
    // can't deadlock. SUPERSEDED by the engine cancel in
    // NetworkingLibraries.php::PlayCard (+ the run_match cancel/rewind), which
    // is more correct: it hides no legal actions and cancels only truly
    // unpayable plays. Verified that filter-OFF closes more games (65.6% vs
    // 53.1% lethal on drafted aggro). Left OFF; kept as a kill-switch only.
    private const AFFORD_FILTER = false;

    // Activation affordability — ALWAYS gate cost-bearing hero/equipment
    // activations on whether the player can actually pitch the cost. Unlike
    // hand plays (AFFORD_FILTER, superseded by the engine cancel + run_match
    // rewind), an unpayable ACTIVATION in an INSTANT/reaction window has NO
    // cancel backstop in the headless run_one_game loop that cc_selfplay uses,
    // so it deadlocks and re-offers forever — the Zyggy cost-2 instant wedge
    // (see project memory cc-zyggy-instant-wedge). Conservative (hand-pitch
    // count via availableResources); fail-open on any uncertainty.
    private const AFFORD_ACTIVATIONS = true;

    /** @return array<int, array> */
    public static function enumerate(array $raw): array
    {
        // Terminal state -> no legal actions.
        $winner = (int)($raw['winner'] ?? 0);
        if ($winner !== 0) return [];

        if (ADAPTER_MODE === 'stub') {
            return self::stubMenu($raw);
        }

        return self::realMenu($raw);
    }

    // -------------------------------------------------------------------
    // STUB mode menu
    // -------------------------------------------------------------------
    private static function stubMenu(array $raw): array
    {
        $pid = (int)($raw['current_player'] ?? 1);
        $idCounter = 1;
        $out = [];

        $mk = function (string $type, array $extra = []) use (&$idCounter, $pid) {
            return array_merge([
                'action_id' => $idCounter++,
                'type'      => $type,
                'player_id' => $pid,
                'card_id'   => null,
                'targets'   => [],
                'cost'      => [],
                // talishar_* values are ignored by StubGame::applyAction,
                // but we mirror the production payload shape so tests
                // exercise the same record schema.
                'talishar_mode'       => 99,
                'talishar_button'     => $type,
                'talishar_card_id'    => '0',
                'talishar_chk_count'  => 0,
                'talishar_chk_input'  => [],
                'talishar_input_text' => '',
            ], $extra);
        };

        $out[] = $mk('ATTACK');
        $out[] = $mk('DEFEND');
        $out[] = $mk('PASS');
        return $out;
    }

    // -------------------------------------------------------------------
    // REAL mode menu
    // -------------------------------------------------------------------
    private static function realMenu(array $raw): array
    {
        $turn = $raw['turn'] ?? [];
        $pid  = (int)($raw['priority_player'] ?? $raw['currentPlayer'] ?? 1);
        $phase = (string)($turn[0] ?? '');

        $out = [];
        $next = 1;

        // ---- Yes/No prompts ------------------------------------------
        // YESNO/DOCRANK are fixed two-button prompts: the engine presents
        // exactly "Yes" (mode 20, value "YES") and "No" (value "NO") — see
        // BuildPlayerInputPopup.php::"YESNO". turn[2] here is the PROMPT TEXT,
        // not the options, so deriving the button from it (as the generic block
        // below did) sends an unrecognised value, the decision queue never
        // advances, and the game WEDGES. Emit the two real buttons instead.
        // These prompts are NOPASS — the only valid answers are Yes/No.
        if ($phase === 'YESNO' || $phase === 'DOCRANK') {
            $out[] = self::row($next++, 'DECISION', $pid,
                ['option' => 'YES', 'talishar_mode' => 20, 'talishar_button' => 'YES']);
            $out[] = self::row($next++, 'DECISION', $pid,
                ['option' => 'NO', 'talishar_mode' => 20, 'talishar_button' => 'NO']);
            return $out;
        }

        // ---- Decision queue popups -----------------------------------
        if (in_array($phase, ['BUTTONINPUT','BUTTONINPUTNOPASS',
                              'CHOOSENUMBER','DYNPITCH','CHOOSEARCANE'], true)) {
            $options = self::splitOptions((string)($turn[2] ?? ''));
            $mode    = match ($phase) {
                'CHOOSENUMBER','DYNPITCH'               => 7,
                default                                 => 17,
            };
            foreach ($options as $opt) {
                $out[] = self::row($next++, 'DECISION', $pid, [
                    'option'            => $opt,
                    'talishar_mode'     => $mode,
                    'talishar_button'   => (string)$opt,
                ]);
            }
            // Allow pass-through only if the phase permits it.
            if ($phase !== 'BUTTONINPUTNOPASS') {
                $out[] = self::passRow($next++, $pid);
            }
            return $out;
        }

        // CHOOSECARD-family popups (single card selection from a zone).
        // Mode codes mirror what BuildPlayerInputPopup.php emits via
        // CreateButtonAPI; CHOOSEMULTIZONE / MAYCHOOSEMULTIZONE use mode
        // 16 with the full "ZONE-INDEX" option as button input.
        $singleCardChoosers = [
            'CHOOSETOP' => 8, 'CHOOSEBOTTOM' => 9,
            'CHOOSECARD' => 23, 'MAYCHOOSECARD' => 23,
            'CHOOSECARDID' => 16,
            'CHOOSEDECK' => 11, 'MAYCHOOSEDECK' => 11,
            'CHOOSETHEIRDECK' => 11, 'MAYCHOOSETHEIRDECK' => 11,
            'CHOOSEBANISH' => 16, 'CHOOSEDISCARD' => 16,
            'MAYCHOOSEDISCARD' => 16, 'CHOOSEDISCARDCANCEL' => 16,
            'CHOOSEMYAURA' => 16, 'CHOOSEPERMANENT' => 16,
            'MAYCHOOSEPERMANENT' => 16, 'CHOOSECHARACTER' => 16,
            'CHOOSETHEIRCHARACTER' => 16, 'CHOOSECOMBATCHAIN' => 16,
            'MAYCHOOSECOMBATCHAIN' => 16, 'CHOOSEARSENAL' => 16,
            'MAYCHOOSEARSENAL' => 16, 'CHOOSEARSENALCANCEL' => 16,
            'CHOOSETHEIRHAND' => 16, 'CHOOSETOPOPPONENT' => 29,
            'CHOOSEMULTIZONE' => 16, 'MAYCHOOSEMULTIZONE' => 16,
            // Single hand-card choice (forced discard / return etc.). Hand cards
            // get action mode 16 with the hand INDEX as the value during these
            // phases (BuildGameState.php:606). Missing here = only PASS offered
            // = a forced choice the bot can't make = wedge. turn[2] is the list
            // of choosable indices, so the generic option loop below works.
            'CHOOSEHAND' => 16, 'MAYCHOOSEHAND' => 16, 'CHOOSEHANDCANCEL' => 16,
        ];
        if (isset($singleCardChoosers[$phase])) {
            $optCount = 0;
            foreach (self::splitOptions((string)($turn[2] ?? '')) as $card) {
                $out[] = self::row($next++, 'CHOOSE_' . $phase, $pid, [
                    'card_id'        => $card,
                    'talishar_mode'  => $singleCardChoosers[$phase],
                    // Mode 16 (CHOOSEMULTIZONE, CHOOSEBANISH, ...) and
                    // its kin read the option from $cardID inside
                    // NetworkingLibraries.php::ProcessInput. Setting
                    // BOTH cardID and buttonInput keeps every dispatch
                    // path satisfied — buttonInput is harmless for the
                    // pure cardID modes.
                    'talishar_button'  => (string)$card,
                    'talishar_card_id' => (string)$card,
                ]);
                $optCount++;
            }
            // PASS is only a real engine input for the PASSABLE (MAY*) choosers —
            // these mirror PassInput()'s $passOptions in NetworkingLibraries.php.
            // For a MANDATORY choose (CHOOSEMULTIZONE / CHOOSEHAND / CHOOSECARDID
            // / ...) the engine NO-OPs a PASS, so offering it lets a deterministic
            // (argmax) policy pick PASS every step at a frozen state -> infinite
            // no_progress wedge (observed: ~17% of CC draws). Offer PASS only when
            // the phase is actually passable, or as a last-resort escape when
            // there are zero options (so we never hand the bot an empty set).
            static $passableChoosers = [
                'MAYCHOOSEMULTIZONE' => 1, 'MAYCHOOSEHAND' => 1, 'MAYCHOOSEDISCARD' => 1,
                'MAYCHOOSEARSENAL' => 1, 'MAYCHOOSEPERMANENT' => 1, 'MAYCHOOSEDECK' => 1,
                'MAYCHOOSECARD' => 1, 'MAYCHOOSECOMBATCHAIN' => 1,
            ];
            if (isset($passableChoosers[$phase]) || $optCount === 0) {
                $out[] = self::passRow($next++, $pid);
            }
            return $out;
        }

        // ---- End-of-turn pitch ordering (turn[0] == "PDECK") ---------
        // CR 4.4.3d/4.4.4: pitched cards go to the BOTTOM of the deck in an
        // order of the owner's choice ("pitch stacking"). The engine prompts
        // PDECK whenever >=2 cards are in pitch at end of turn; ProcessInput
        // case 6 takes cardID = a pitch CARD ID, bottoms it, and re-prompts
        // until the pitch is empty. This branch was missing, so the only
        // offer was PASS, and PassInput() silently bottomed everything in
        // default order — the choice never existed for the bot. PASS stays
        // legal (= accept default order for the rest).
        if ($phase === 'PDECK') {
            $pitch  = self::playerZone($raw, $pid, 'pitch');
            $stride = function_exists('PitchPieces') ? PitchPieces() : 2;
            $seen   = [];
            for ($i = 0; $i < count($pitch); $i += $stride) {
                $cardId = $pitch[$i] ?? '';
                // Duplicate ids resolve to the same SearchPitchForCard hit;
                // offering them twice would just be the same action.
                if (!$cardId || isset($seen[$cardId])) continue;
                $seen[$cardId] = true;
                $out[] = self::row($next++, 'CHOOSE_PDECK', $pid, [
                    'card_id'          => $cardId,
                    'talishar_mode'    => 6,
                    'talishar_card_id' => (string)$cardId,
                ]);
            }
            $out[] = self::passRow($next++, $pid);
            return $out;
        }

        // ---- Multi-select prompts (MULTICHOOSE* family) --------------
        // turn[2] = "max-opt1,opt2,...[-min]"; ProcessInput mode 19 takes
        // chkInput = the selected option INDICES. min>0 means "choose
        // exactly", otherwise "up to max" (empty selection legal). These
        // phases are NOT passable (CanPassPhase==0), so before this branch a
        // card reaching one would hard-wedge the game. We enumerate
        // combinations of the required size, capped to fit max_legal.
        $multiChoosers = [
            'MULTICHOOSEHAND', 'MAYMULTICHOOSEHAND',
            'MULTICHOOSEDISCARD', 'MULTICHOOSETHEIRDISCARD',
            'MULTICHOOSETEXT', 'MAYMULTICHOOSETEXT',
            'MULTICHOOSEDECK', 'MULTICHOOSETHEIRDECK',
            'MULTICHOOSEBANISH', 'MULTICHOOSEITEMS', 'MULTICHOOSESUBCARDS',
        ];
        if (in_array($phase, $multiChoosers, true)) {
            $params  = explode('-', (string)($turn[2] ?? ''));
            $max     = (int)($params[0] ?? 0);
            $options = array_values(array_filter(
                explode(',', $params[1] ?? ''), fn($v) => $v !== ''));
            $min     = isset($params[2]) ? (int)$params[2] : -1;
            $n       = count($options);
            $size    = $min > 0 ? min($min, $n) : min(max($max, 0), $n);
            foreach (self::combinations($n, max($size, 0), 18) as $combo) {
                $cards = array_map(
                    fn($ix) => self::multiOptionCard($raw, $pid, $phase, $options[$ix]),
                    $combo);
                $out[] = self::row($next++, 'DECISION', $pid, [
                    'card_id'            => $cards[0] ?? null,
                    'option'             => implode(',', array_map(
                        fn($ix) => $options[$ix], $combo)),
                    'talishar_mode'      => 19,
                    'talishar_chk_count' => count($combo),
                    'talishar_chk_input' => array_map('strval', $combo),
                ]);
            }
            if ($min <= 0) {
                // "Up to N": selecting nothing is legal — submit an EMPTY
                // mode-19 selection (PASS itself is rejected by the engine
                // for the non-MAY variants).
                $out[] = self::row($next++, 'DECISION', $pid, [
                    'option'             => 'NONE',
                    'talishar_mode'      => 19,
                    'talishar_chk_count' => 0,
                    'talishar_chk_input' => [],
                ]);
            }
            return $out;
        }

        // ---- Hand to top/bottom of deck (HANDTOPBOTTOM) ---------------
        // Also unpassable (CanPassPhase==0). One button per hand card:
        // mode 12 = put on top, mode 13 = put on bottom (cardID = card id).
        if ($phase === 'HANDTOPBOTTOM') {
            foreach (self::playerZone($raw, $pid, 'hand') as $cardId) {
                if (!$cardId) continue;
                foreach ([12 => 'TOP', 13 => 'BOTTOM'] as $mode => $where) {
                    $out[] = self::row($next++, 'CHOOSE_HANDTOPBOTTOM', $pid, [
                        'card_id'          => $cardId,
                        'option'           => $where,
                        'talishar_mode'    => $mode,
                        'talishar_card_id' => (string)$cardId,
                    ]);
                }
            }
            $out[] = self::passRow($next++, $pid);
            return $out;
        }

        // ---- Name a card (INPUTCARDNAME, mode 30 string input) ---------
        // PASS is a NO-OP here (PassInput has no branch and Pass() doesn't
        // advance non-combat phases), so without options this phase would
        // hard-wedge. Mode 30 takes inputText; CardName() resolves a card id
        // to its name when possible, and any string is accepted. Candidates:
        // cards the player can reason about — the opponent's public zones
        // (the usual targets of "name a card" effects) plus its own hand.
        if ($phase === 'INPUTCARDNAME') {
            $cands = [];
            $opp = $pid === 1 ? 2 : 1;
            foreach ([[$opp, 'discard'], [$opp, 'pitch'], [$opp, 'banish'],
                      [$pid, 'hand'], [$pid, 'discard']] as [$who, $zone]) {
                $stride = match ($zone) {
                    'discard', 'banish' => 3,
                    'pitch'             => 2,
                    default             => 1,
                };
                $cards = self::playerZone($raw, $who, $zone);
                for ($i = 0; $i < count($cards); $i += $stride) {
                    $c = (string)($cards[$i] ?? '');
                    if ($c !== '' && !isset($cands[$c])) $cands[$c] = true;
                }
            }
            foreach (array_slice(array_keys($cands), 0, 20) as $cardId) {
                $out[] = self::row($next++, 'DECISION', $pid, [
                    'card_id'             => $cardId,
                    'option'              => $cardId,
                    'talishar_mode'       => 30,
                    'talishar_input_text' => (string)$cardId,
                ]);
            }
            if (!$cands) {
                // Nothing visible to name — submit a harmless placeholder so
                // the decision queue still advances (any string is legal).
                $out[] = self::row($next++, 'DECISION', $pid, [
                    'option'              => 'NONE',
                    'talishar_mode'       => 30,
                    'talishar_input_text' => 'none',
                ]);
            }
            return $out;
        }

        // NOTE (documented gaps, deliberate): ORDERTRIGGERS falls through to
        // PASS below — this engine build's layer-reorder input (mode 33) is a
        // no-op stub, so PassInput's default trigger order is the ONLY valid
        // resolution even for the real frontend. OPT (look-at-top reorder)
        // also falls through to PASS — its popup is a drag-drop UI with no
        // button protocol; PASS keeps the looked-at cards in current order
        // (PassInput's OPT branch), which is rules-legal (CR allows any
        // order) just not strategic. COERCIVE (rearrange top N) likewise has
        // no button protocol in this build AND no PassInput/ProcessInput/DQ
        // handler at all, so it can't be resolved here. Its only user,
        // coercive_tendency_blue, was neutralized at the card level (its
        // PlayAbility no longer emits the COERCIVE DQ — see HVYCards.php), so it
        // is no longer a live wedge. If a FUTURE cube card emits COERCIVE it
        // will fall through to PASS and the game aborts via run_match's wedge
        // guard rather than hanging — fix it the same way (or implement a real
        // handler) when that happens.

        // ---- End-of-turn arsenal step (turn[0] == "ARS") -------------
        // Place ONE hand card face-down into arsenal. ProcessInput case 4
        // (NetworkingLibraries.php:58) reads cardID = the card id (HasCard) and
        // requires turn[0]=="ARS"; GameLogic.php:1847 sets that token at end of
        // turn when the hand is non-empty and arsenal isn't full. This branch
        // USED to be nested inside the M/A/D/B/P play block below, where it was
        // unreachable (phase can't be both in that set AND 'ARS') — so the
        // arsenal step only ever offered PASS and the policy never learned to
        // arsenal. Handle it as its own top-level phase.
        if ($phase === 'ARS') {
            foreach (self::playerZone($raw, $pid, 'hand') as $i => $cardId) {
                if (!$cardId) continue;
                $out[] = self::row($next++, 'ARSENAL_FROM_HAND', $pid, [
                    'card_id'          => $cardId,
                    'hand_index'       => $i,
                    'talishar_mode'    => 4,
                    'talishar_button'  => (string)$cardId,
                    'talishar_card_id' => (string)$cardId,
                ]);
            }
            $out[] = self::passRow($next++, $pid);
            return $out;
        }

        // ---- Main / action / defence / INSTANT-reaction: play cards --
        // INSTANT is the instant-speed reaction window (GameLogic.php:1200).
        // It was missing from this whitelist, so reactions during that window
        // were never enumerated and got auto-passed. IsPlayable() does the
        // per-phase legality filtering, so adding INSTANT only surfaces cards
        // that are actually playable there.
        if (in_array($phase, ['M', 'A', 'D', 'B', 'P', 'INSTANT'], true) ||
            $phase === '' /* fresh state */) {
            // Hand plays (mode 27). We CALL Talishar's IsPlayable() for
            // each hand card so the bot never receives a useless action
            // — without this filter the engine silently no-ops, the bot
            // doesn't realise its move was rejected, and the game stalls.
            $hand = self::playerZone($raw, $pid, 'hand');
            $canCheck = function_exists('IsPlayable');
            foreach ($hand as $i => $cardId) {
                if (!$cardId) continue;
                if ($canCheck) {
                    try {
                        if (!IsPlayable($cardId, $phase, 'HAND', $i)) continue;
                    } catch (\Throwable) {
                        // If IsPlayable raises (some unimplemented card
                        // metadata path), fall through and emit the
                        // action — better than dropping it.
                    }
                }
                // Affordability guard — a main-phase hand play costs CardCost,
                // paid by PITCHING other cards. The engine has no pre-pay check
                // (FAB pays interactively), so a bot that plays a card it can't
                // fund DEADLOCKS the pay phase (only PASS offered, PASS neither
                // pays nor cancels → game wedges to the cap). Drop plays the
                // player demonstrably can't pay. Only gate M (and fresh '') —
                // pitching in P and free blocks in B/D are NOT cost-bearing.
                // Fail-open: any uncertainty emits the action (run_match's
                // wedge-guard is the backstop). See availableResources().
                if (self::AFFORD_FILTER && ($phase === 'M' || $phase === '') && function_exists('CardCost')) {
                    try {
                        $cost = (int) CardCost($cardId, 'HAND');
                        if ($cost > 0 && self::availableResources($raw, $pid, $i) < $cost) continue;
                    } catch (\Throwable) {}
                }
                $out[] = self::row($next++, 'PLAY_FROM_HAND', $pid, [
                    'card_id'        => $cardId,
                    'hand_index'     => $i,
                    'talishar_mode'  => 27,
                    'talishar_card_id' => (string)$i,
                ]);
            }

            // Pitch a card from hand during P phase (mode 6 is for
            // pitch-from-deck via PDECK). The actual hand-pitch path
            // during P is mode 27 too — Talishar's IsPlayable returns
            // true for cards with non-zero pitch value when phase=='P'
            // — so the filter above already includes them.

            // Play a card FROM arsenal (mode 5). ProcessInput case 5
            // (NetworkingLibraries.php:72) reads cardID = the arsenal stride
            // INDEX. The serializer previously never iterated the arsenal zone,
            // so an arsenaled card could never be played — the arsenal mechanic
            // was entirely invisible to the policy. IsPlayable(...,"ARS",...)
            // gates per-phase legality.
            $ars       = self::playerZone($raw, $pid, 'arsenal');
            $arsStride = function_exists('ArsenalPieces') ? ArsenalPieces() : 7;
            for ($ai = 0; $ai < count($ars); $ai += $arsStride) {
                $cardId = $ars[$ai] ?? '';
                if (!$cardId) continue;
                if ($canCheck) {
                    try {
                        if (!IsPlayable($cardId, $phase, 'ARS', $ai)) continue;
                    } catch (\Throwable) {}
                }
                $out[] = self::row($next++, 'PLAY_FROM_ARSENAL', $pid, [
                    'card_id'          => $cardId,
                    'arsenal_index'    => $ai,
                    'talishar_mode'    => 5,
                    'talishar_card_id' => (string)$ai,
                ]);
            }

            // Hero / equipment ability activation (mode 3 — matches CombatDummyAI).
            // The "cardID" in mode 3 is actually the char zone stride; "chkCount"
            // is CharacterPieces() and chkInput[0] is the index. We mirror that.
            // NOTE: the raw state keys this zone 'charequip' (collectGlobals
            // lower-cases the Talishar global name $p{N}CharEquip). Reading it
            // as 'equipment' returned an empty array, so this whole loop
            // silently emitted no actions — weapon/hero/equipment activations
            // (mode 3) were never offered to the bots. Use the correct key.
            $char       = self::playerZone($raw, $pid, 'charequip');
            $stride     = function_exists('CharacterPieces') ? CharacterPieces() : 15;
            for ($idx = 0; $idx < count($char); $idx += $stride) {
                $cardId = $char[$idx] ?? '';
                // Skip empty/placeholder char slots. Empty equipment/Evo slots
                // (Mechanologist heroes carry several) are filled with a
                // "NONE00"-style sentinel — TRUTHY, so the !$cardId check misses
                // it. Offering it as an activation is a no-op the engine accepts
                // and re-offers forever: the teklovossen-vs-victor block-phase
                // wedge (ACTIVATE_HERO_OR_EQUIP 'NONE00' x57). No real card id
                // starts with "NONE".
                if (!$cardId || str_starts_with($cardId, 'NONE')) continue;
                // Only offer a piece whose activated ability is READY. The engine
                // marks readiness in the char-zone status field (stride +1): 2 =
                // ready, 1 = used-this-turn, 0 = absent/destroyed (it's exactly
                // what the FE uses to grey out spent equipment). IsPlayable does
                // NOT track per-turn usage, so without this the serializer kept
                // offering spent once-per-turn abilities (e.g. Seeker's gear) that
                // the engine then cancels. Multi-use abilities stay 2 until their
                // uses (stride +5) are exhausted, so this preserves them.
                if ((int)($char[$idx + 1] ?? 0) !== 2) continue;
                if ($canCheck) {
                    // IsPlayable enforces the card's IsPlayRestricted (e.g. Zyggy
                    // needs a lightning_flow aura + holo auras + untapped hero;
                    // CardDictionary.php:1922 runs it for INSTANT). FAIL CLOSED:
                    // unlike hand plays, an ACTIVATION we can't confirm playable
                    // is a wedge risk — the engine accepts it, PayAdditionalCosts
                    // fails, it reverts to the prior state, and the serializer
                    // re-offers it forever (the Zyggy INSTANT loop). So if
                    // IsPlayable says no OR throws, skip the activation.
                    try {
                        if (!IsPlayable($cardId, $phase, 'CHAR', $idx)) continue;
                    } catch (\Throwable) { continue; }
                }
                // Affordability guard (same pay-phase deadlock as hand plays):
                // an activated ability costs AbilityCost, paid by pitching.
                // Activations are always cost-bearing (unlike free blocks /
                // pitches), so gate them in every phase. Fail-open.
                if (self::AFFORD_ACTIVATIONS && function_exists('AbilityCost')) {
                    try {
                        $cost = (int) AbilityCost($cardId);
                        if ($cost > 0 && self::availableResources($raw, $pid) < $cost) continue;
                    } catch (\Throwable) {}
                }
                $out[] = self::row($next++, 'ACTIVATE_HERO_OR_EQUIP', $pid, [
                    'card_id'         => $cardId,
                    'equipment_index' => $idx,
                    'talishar_mode'   => 3,
                    'talishar_button' => '',
                    'talishar_card_id'=> (string)$idx,
                    'talishar_chk_count' => 0,
                    'talishar_chk_input' => [(string)$idx],
                ]);
            }

            // ---- Block with EQUIPMENT (B = block step) ---------------------
            // Armor blocks dispatch through the SAME mode 3 as ability
            // activation, but the engine treats mode 3 during turn[0]=="B" as a
            // BLOCK (NetworkingLibraries.php case 3 sets charequip[idx+6]=1) —
            // see BlockCardAttempt() in AI/EncounterPlayLogic.php:188. The
            // activation loop above gates on a READY, IsPlayable ability, so a
            // plain armor piece (block value, no ability) is never offered. Then
            // a card that FORCES an equip block (t_bone_red — "must block with
            // equipment if able") leaves PASS as the only action, the engine
            // refuses the pass (PassInput returns for "B"), and the block step
            // wedges (the dorinthea-vs-maxx no_progress freeze). Blocks are
            // free, so no IsPlayable / affordability gate — just present, unused
            // this combat, and able to block. (D-step defense reactions still
            // flow through the activation loop.)
            if ($phase === 'B') {
                $bchar   = self::playerZone($raw, $pid, 'charequip');
                $bstride = function_exists('CharacterPieces') ? CharacterPieces() : 15;
                $hasBV   = function_exists('BlockValue');
                for ($bi = 0; $bi < count($bchar); $bi += $bstride) {
                    $bCardId = $bchar[$bi] ?? '';
                    if (!$bCardId || str_starts_with($bCardId, 'NONE')) continue;
                    // status (stride +1): 2 = ready/present; +6 = has already
                    // blocked this combat (the flag case 3 sets).
                    if ((int)($bchar[$bi + 1] ?? 0) !== 2) continue;
                    if ((int)($bchar[$bi + 6] ?? 0) === 1) continue;
                    if ($hasBV) {
                        try { if (BlockValue($bCardId) < 0) continue; }
                        catch (\Throwable) { continue; }
                    }
                    $out[] = self::row($next++, 'BLOCK_WITH_EQUIP', $pid, [
                        'card_id'          => $bCardId,
                        'equipment_index'  => $bi,
                        'talishar_mode'    => 3,
                        'talishar_card_id' => (string)$bi,
                    ]);
                }
            }
        }

        // Always allow pass / OK.
        $out[] = self::passRow($next++, $pid);

        return $out;
    }

    // -------------------------------------------------------------------
    // helpers
    // -------------------------------------------------------------------
    private static function row(int $id, string $type, int $pid, array $extra): array
    {
        return array_merge([
            'action_id' => $id,
            'type'      => $type,
            'player_id' => $pid,
            'card_id'   => null,
            'targets'   => [],
            'cost'      => [],
            'talishar_mode'       => 99,
            'talishar_button'     => '',
            'talishar_card_id'    => '0',
            'talishar_chk_count'  => 0,
            'talishar_chk_input'  => [],
            'talishar_input_text' => '',
        ], $extra);
    }

    private static function passRow(int $id, int $pid): array
    {
        return self::row($id, 'PASS', $pid, [
            'talishar_mode'   => 99,
            'talishar_button' => 'OK',
        ]);
    }

    private static function playerZone(array $raw, int $pid, string $zone): array
    {
        $players = $raw['players'] ?? [];
        foreach ($players as $p) {
            if ((int)($p['player_id'] ?? 0) === $pid) {
                return array_values($p[$zone] ?? []);
            }
        }
        return [];
    }

    /**
     * Lexicographic k-combinations of [0..n-1], capped at $cap results so a
     * wide MULTICHOOSE prompt can't blow past max_legal. k=0 yields nothing
     * (the caller emits the explicit empty-selection action instead).
     * @return array<int, array<int>>
     */
    private static function combinations(int $n, int $k, int $cap): array
    {
        if ($k <= 0 || $k > $n) return [];
        $combo = range(0, $k - 1);
        $outs  = [];
        while (count($outs) < $cap) {
            $outs[] = $combo;
            // advance to next lexicographic combination
            $i = $k - 1;
            while ($i >= 0 && $combo[$i] === $n - $k + $i) $i--;
            if ($i < 0) break;
            $combo[$i]++;
            for ($j = $i + 1; $j < $k; $j++) $combo[$j] = $combo[$j - 1] + 1;
        }
        return $outs;
    }

    /**
     * Resolve a MULTICHOOSE* option value to a card id for the bot's action
     * embedding. For zone-indexed prompts the option is a raw STRIDE INDEX
     * into the zone array; for SUBCARDS/TEXT the option is the value itself.
     * Returns null when the underlying card is hidden (deck searches).
     */
    private static function multiOptionCard(array $raw, int $pid, string $phase, string $opt): ?string
    {
        $opp = $pid === 1 ? 2 : 1;
        [$zone, $who] = match ($phase) {
            'MULTICHOOSEHAND', 'MAYMULTICHOOSEHAND' => ['hand', $pid],
            'MULTICHOOSEDISCARD'                    => ['discard', $pid],
            'MULTICHOOSETHEIRDISCARD'               => ['discard', $opp],
            'MULTICHOOSEBANISH'                     => ['banish', $pid],
            'MULTICHOOSEITEMS'                      => ['items', $pid],
            default                                 => [null, $pid],
        };
        if ($zone === null) {
            // SUBCARDS/TEXT carry the value directly; deck picks stay hidden.
            return str_contains($phase, 'DECK') ? null : ($opt !== '' ? $opt : null);
        }
        $cards = self::playerZone($raw, $who, $zone);
        $card  = $cards[(int)$opt] ?? null;
        return ($card !== null && $card !== '') ? (string)$card : null;
    }

    /**
     * Resources the player could muster to pay a cost RIGHT NOW: the pitch
     * value of every card in hand (optionally excluding the card being played,
     * which can't pitch for itself). Used by the affordability guards above so
     * the bot is never offered a play it would deadlock the engine trying to
     * pay for.
     *
     * We deliberately do NOT add the serialized `resources` field: it is not a
     * usable floating pool at decision time (it reflects resources generated
     * during an in-progress payment), and counting it OVERESTIMATES what the
     * player can pitch — which is exactly what let a cost-2 activation through
     * on a one-card hand and re-wedged the pay phase. Counting only hand pitch
     * is conservative: at worst it hides a play funded by genuine floating
     * resources (rare), which the bot simply skips — far better than a wedge.
     * Fail-open: if pitch values can't be computed, returns PHP_INT_MAX so the
     * action is always emitted.
     */
    private static function availableResources(array $raw, int $pid, int $excludeHandIndex = -1): int
    {
        if (!function_exists('PitchValue')) return PHP_INT_MAX;
        $avail = 0;
        foreach (self::playerZone($raw, $pid, 'hand') as $i => $c) {
            if ($i === $excludeHandIndex || $c === '' || $c === null) continue;
            try { $avail += (int) PitchValue((string)$c); } catch (\Throwable) { return PHP_INT_MAX; }
        }
        return $avail;
    }

    private static function splitOptions(string $s): array
    {
        if ($s === '') return [];
        // Some phases pack options as "a,b,c" and some as "max-a,b,c-min" — handle both.
        if (str_contains($s, '-') && preg_match('/^\d+-/', $s)) {
            $parts = explode('-', $s);
            $list  = $parts[1] ?? '';
            return array_values(array_filter(explode(',', $list), fn($v) => $v !== ''));
        }
        return array_values(array_filter(explode(',', $s), fn($v) => $v !== ''));
    }
}
