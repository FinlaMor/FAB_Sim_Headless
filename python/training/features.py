"""Feature encoders shared by the offline-RL trainers.

Two kinds of features are produced for each domain:

* **scalars** — compact hand-engineered floats (phase, life totals, zone
  counts, action type/cost). Cheap, dense, always present.
* **card slots** — integer card ids (looked up in a :class:`CardVocab`)
  that the nets turn into *learned embeddings* (item 4: richer features).
  Gameplay uses my/opp hero + a mean-pooled hand; actions carry the card
  they play. Draft embeds each pack option + a pooled drafted pool.

Keeping the float scalars and the card ids separate lets the IQL nets own
the ``nn.Embedding`` while this module stays torch-free.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Sequence

# ---------------------------------------------------------------------------
# Card vocabulary (shared by gameplay + draft embeddings)
# ---------------------------------------------------------------------------
PAD = 0
UNK = 1


class CardVocab:
    """Maps card id -> embedding index. 0 = PAD, 1 = UNK."""

    def __init__(self, cards: Iterable[str] = ()) -> None:
        self.itos: list[str] = ["<pad>", "<unk>"]
        self.stoi: dict[str, int] = {}
        for c in sorted(set(x for x in cards if x)):
            self.stoi[c] = len(self.itos)
            self.itos.append(c)

    def index(self, card: str | None) -> int:
        if not card:
            return PAD
        return self.stoi.get(card, UNK)

    def __len__(self) -> int:
        return len(self.itos)

    def to_dict(self) -> dict[str, Any]:
        return {"itos": self.itos}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "CardVocab":
        v = cls()
        v.itos = list(d["itos"])
        v.stoi = {c: i for i, c in enumerate(v.itos) if i >= 2}
        return v


# ---------------------------------------------------------------------------
# Gameplay
# ---------------------------------------------------------------------------
PHASES: tuple[str, ...] = (
    "M", "P", "INSTANT", "ARS", "A", "D", "B", "OVER",
    "YESNO", "BUTTONINPUT", "CHOOSECARD", "PDECK",
    # Chooser/prompt phases that actually occur in recorded OMN games (the
    # 2026-06-10 audit found ~1.1k decisions per cycle whose phase one-hot
    # encoded as all-zeros because these were missing).
    "BUTTONINPUTNOPASS", "DYNPITCH", "CHOOSEARCANE",
    "CHOOSEHAND", "CHOOSEHANDCANCEL",
    "CHOOSEMULTIZONE", "MAYCHOOSEMULTIZONE",
)
_PHASE_IDX = {p: i for i, p in enumerate(PHASES)}

ACTION_TYPES: tuple[str, ...] = (
    "PLAY_FROM_HAND", "ACTIVATE_HERO_OR_EQUIP", "ARSENAL_FROM_HAND",
    "DECISION", "CHOOSE", "PASS",
)
_ATYPE_IDX = {t: i for i, t in enumerate(ACTION_TYPES)}

HAND_SLOTS = 8  # hand cards pooled into the state embedding

# 20 board/life scalars + 8 combat-awareness scalars (so the bot can make
# informed block/no-block decisions instead of blocking at random). The last 2
# are DYNAMIC attack properties (granted-or-printed go again / dominate), which
# flip the correct block even when the cards' base stats don't. (Boost is not
# tracked separately: per CR 8.3.9 its only combat effect is granting go again,
# already captured by the go-again flag — and it's a Mechanologist-only keyword
# absent from this Illusionist/Lightning cube.)
# The 3 extra (vs the original 18): my/opp FLOATING RESOURCE POINTS (CR 8.5) —
# without them the bot pitched blind during the pay phase (~21% of decisions) —
# and the number of PRIOR CHAIN LINKS this turn (CR 7.0.9, combo/tempo signal).
STATE_SCALAR_DIM = len(PHASES) + 21 + 8
ACTION_SCALAR_DIM = len(ACTION_TYPES) + 4


def _player(state: dict[str, Any], pid: int) -> dict[str, Any]:
    for p in state.get("players", []) or []:
        if int(p.get("player_id", 0)) == pid:
            return p
    return {}


def _n(x: Any) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return 0.0


def _len(p: dict[str, Any], key: str) -> int:
    v = p.get(key)
    return len(v) if isinstance(v, list) else 0


def encode_state_scalars(state: dict[str, Any], me: int) -> list[float]:
    opp = 2 if me == 1 else 1
    mp, op = _player(state, me), _player(state, opp)
    feats: list[float] = [0.0] * len(PHASES)
    pidx = _PHASE_IDX.get(str(state.get("phase", "")))
    if pidx is not None:
        feats[pidx] = 1.0
    my_hp = _n(mp.get("health")) / 20.0
    op_hp = _n(op.get("health")) / 20.0
    feats += [
        _n(state.get("action_points")),
        _n(state.get("turn")) / 20.0,
        1.0 if int(state.get("active_player", 0)) == me else 0.0,
        _n(mp.get("resources")) / 3.0, _n(op.get("resources")) / 3.0,
        my_hp, op_hp, my_hp - op_hp,
        _len(mp, "hand") / 10.0, _len(op, "hand") / 10.0,
        _len(mp, "arsenal"), _len(mp, "pitch") / 5.0,
        _n(mp.get("deck_count")) / 30.0, _n(op.get("deck_count")) / 30.0,
        _len(mp, "graveyard") / 30.0, _len(op, "graveyard") / 30.0,
        _len(mp, "auras") + _len(mp, "items") + _len(mp, "allies") + _len(mp, "permanents"),
        _len(op, "auras") + _len(op, "items") + _len(op, "allies") + _len(op, "permanents"),
        len(state.get("combat_chain") or []) / 5.0,
        len(state.get("stack") or []) / 7.0,
        len(state.get("links") or []) / 5.0,
    ]
    # --- Combat awareness (the key block/no-block signal) ---
    cb = state.get("combat") or {}
    active = float(cb.get("active", 0) or 0)
    attacker = int(cb.get("attacker", 0) or 0)
    am_defender = 1.0 if (active and attacker not in (0, me)) else 0.0
    am_attacker = 1.0 if (active and attacker == me) else 0.0
    power = _n(cb.get("attack_power")) if am_defender else 0.0
    pend = _n(cb.get("pending_damage")) if am_defender else 0.0
    my_life = _n(mp.get("health"))
    lethal = 1.0 if (am_defender and my_life > 0 and pend >= my_life) else 0.0
    # Dynamic attack properties (only meaningful while combat is live). Dominate
    # means the attack can be blocked by at most one card; go again means it
    # keeps the attacker's tempo if it lands — both change the right defence.
    go_again = _n(cb.get("go_again")) if active else 0.0
    dominate = _n(cb.get("dominate")) if active else 0.0
    feats += [
        active,
        am_defender,
        am_attacker,
        power / 10.0,
        pend / 20.0,
        lethal,
        go_again,
        dominate,
    ]
    return feats


def _hero_of(p: dict[str, Any]) -> str:
    h = p.get("hero")
    if h:
        return str(h)
    eq = p.get("equipment") or []
    return str(eq[0]) if eq else ""


def state_card_ids(state: dict[str, Any], me: int, vocab: CardVocab) -> dict[str, Any]:
    """Return embedding indices for the state's card slots."""
    opp = 2 if me == 1 else 1
    mp, op = _player(state, me), _player(state, opp)
    hand = [c for c in (mp.get("hand") or []) if c][:HAND_SLOTS]
    hand_idx = [vocab.index(c) for c in hand]
    hand_idx += [PAD] * (HAND_SLOTS - len(hand_idx))
    return {
        "hero_me": vocab.index(_hero_of(mp)),
        "hero_opp": vocab.index(_hero_of(op)),
        "hand": hand_idx,
    }


# ---------------------------------------------------------------------------
# Full public-zone token set (gameplay attention)
# ---------------------------------------------------------------------------
# The bot attends over every zone it is *allowed* to see. Per CR 3.0.4a the
# public zones are arms/equipment, banished, graveyard, hero, permanent,
# pitch, weapon (+combat chain/stack). Hand and deck are private (counts only,
# in the scalars). The arsenal is NOT a public zone, so the opponent's arsenal
# is included as presence-only (masked to UNK) — the model knows a card is
# there but not its identity.
R_CTX = 0
R_MY_HERO, R_OPP_HERO = 1, 2
R_HAND = 3
R_MY_ARS, R_OPP_ARS = 4, 5
R_MY_EQUIP, R_OPP_EQUIP = 6, 7
R_MY_GRAVE, R_OPP_GRAVE = 8, 9
R_MY_BANISH, R_OPP_BANISH = 10, 11
R_MY_BOARD, R_OPP_BOARD = 12, 13
# Pitch is a public zone (CR 3.0.4a) and its cards return to the deck bottom
# at end of turn (CR 4.4.4) — late-game draw knowledge.
R_MY_PITCH, R_OPP_PITCH = 14, 15
# Cards on the active combat chain (attack + defending cards). Without this
# the defender knew the incoming POWER but not WHICH card was attacking, so
# on-hit effects (CR 8.1.2) could never inform the block decision.
R_CHAIN = 16
# Active current-turn effects (CR 6.3 continuous effects: "+1 to next attack",
# frailty, ...) by controller, and the pending layers on the stack (CR 5.x —
# what is actually resolving during a reaction window).
R_MY_FX, R_OPP_FX = 17, 18
R_STACK = 19
N_STATE_ROLES = 20

_BOARD_KEYS = ("permanents", "items", "allies", "auras")
# (role, side, zone keys | None=hero, n_slots)
_STATE_ZONES: tuple = (
    (R_MY_HERO,    "me",  None,            1),
    (R_OPP_HERO,   "opp", None,            1),
    (R_HAND,       "me",  ("hand",),       HAND_SLOTS),
    (R_MY_ARS,     "me",  ("arsenal",),    2),
    (R_OPP_ARS,    "opp", ("arsenal",),    2),   # masked -> UNK
    (R_MY_EQUIP,   "me",  ("equipment",),  5),
    (R_OPP_EQUIP,  "opp", ("equipment",),  5),
    (R_MY_GRAVE,   "me",  ("graveyard",),  12),
    (R_OPP_GRAVE,  "opp", ("graveyard",),  12),
    (R_MY_BANISH,  "me",  ("banished",),   6),
    (R_OPP_BANISH, "opp", ("banished",),   6),
    (R_MY_BOARD,   "me",  _BOARD_KEYS,     8),
    (R_OPP_BOARD,  "opp", _BOARD_KEYS,     8),
    (R_MY_PITCH,   "me",  ("pitch",),      6),
    (R_OPP_PITCH,  "opp", ("pitch",),      6),
)
_MASKED_ROLES = frozenset({R_OPP_ARS})
CHAIN_SLOTS = 4  # attack card + up to 3 more cards on the active chain link
FX_SLOTS = 4     # current-turn effects per side
STACK_SLOTS = 3  # topmost pending layers (next-to-resolve first)
STATE_TOKENS = (sum(n for _, _, _, n in _STATE_ZONES)
                + CHAIN_SLOTS + 2 * FX_SLOTS + STACK_SLOTS)
STATE_ROLES: list[int] = ([r for r, _, _, n in _STATE_ZONES for _ in range(n)]
                          + [R_CHAIN] * CHAIN_SLOTS
                          + [R_MY_FX] * FX_SLOTS + [R_OPP_FX] * FX_SLOTS
                          + [R_STACK] * STACK_SLOTS)

# Per-token STATE vector fed alongside the card id: (counter-ish value / 3,
# ready-or-face-up flag). The de-stride fix (2026-06-10) cleaned the junk out
# of the zone tokens but also dropped real per-card state — counters (CR 9.x),
# ally life, equipment readiness, arsenal facing. This restores it.
TOKEN_STATE_DIM = 2
# (counts_key, ready_key, ready_value) per zone key; ready compares equal.
_ZONE_STATE: dict = {
    "arsenal":   ("arsenal_counts", "arsenal_facing", "UP"),
    "items":     ("items_counts",   "items_ready",    "2"),
    "allies":    ("allies_counts",  "allies_ready",   "2"),
    "auras":     ("auras_counts",   "auras_ready",    "2"),
    "equipment": (None,             "equipment_ready", "2"),
}


def _zone_cards(p: dict[str, Any], keys) -> list[str]:
    if keys is None:
        h = _hero_of(p)
        return [h] if h else []
    out: list[str] = []
    for k in keys:
        out += [c for c in (p.get(k) or []) if c]
    return out


def state_tokens(state: dict[str, Any], me: int, vocab: CardVocab) -> list[int]:
    """Embedding indices for every zone the bot may see (length STATE_TOKENS).

    Opponent arsenal cards are masked to UNK (presence, not identity); empty
    slots are PAD. Roles are positional and fixed (see STATE_ROLES).
    """
    opp = 2 if me == 1 else 1
    mp, op = _player(state, me), _player(state, opp)
    ids: list[int] = []
    for role, side, keys, n in _STATE_ZONES:
        p = mp if side == "me" else op
        cards = _zone_cards(p, keys)[:n]
        if role in _MASKED_ROLES:
            # Opponent arsenal: face-DOWN cards are hidden (UNK = presence
            # only), but a face-UP arsenal card is public per CR — show it.
            facing = p.get("arsenal_facing") or []
            slot = [vocab.index(c) if str(facing[i] if i < len(facing) else "") == "UP"
                    else UNK
                    for i, c in enumerate(cards)]
        else:
            slot = [vocab.index(c) for c in cards]
        slot += [PAD] * (n - len(slot))
        ids += slot
    ids += _chain_card_ids(state, vocab)
    ids += _fx_card_ids(state, me, vocab) + _fx_card_ids(state, opp, vocab)
    ids += _stack_card_ids(state, vocab)
    return ids


def state_token_state(state: dict[str, Any], me: int) -> list[list[float]]:
    """Per-token (count/3, ready-flag) vector aligned with state_tokens()."""
    opp = 2 if me == 1 else 1
    mp, op = _player(state, me), _player(state, opp)
    out: list[list[float]] = []
    for role, side, keys, n in _STATE_ZONES:
        p = mp if side == "me" else op
        pairs = _zone_state_pairs(p, keys)[:n]
        pairs += [[0.0, 0.0]] * (n - len(pairs))
        out += pairs
    out += [[0.0, 0.0]] * (CHAIN_SLOTS + 2 * FX_SLOTS + STACK_SLOTS)
    return out


def _zone_state_pairs(p: dict[str, Any], keys) -> list[list[float]]:
    if keys is None:                      # hero slot
        h = _hero_of(p)
        return [[0.0, 1.0]] if h else []
    out: list[list[float]] = []
    for k in keys:
        cards = [c for c in (p.get(k) or []) if c]
        counts_key, ready_key, ready_val = _ZONE_STATE.get(k, (None, None, None))
        counts = p.get(counts_key) or [] if counts_key else []
        ready = p.get(ready_key) or [] if ready_key else []
        for i, _ in enumerate(cards):
            c = _n(counts[i]) if i < len(counts) else 0.0
            r = 1.0
            if ready_key and i < len(ready):
                r = 1.0 if str(ready[i]) == ready_val else 0.0
            out.append([c / 3.0, r])
    return out


def _fx_card_ids(state: dict[str, Any], pid: int, vocab: CardVocab) -> list[int]:
    ids = []
    for e in state.get("current_turn_effects") or []:
        if isinstance(e, dict) and int(e.get("player", 0) or 0) == pid and e.get("card_id"):
            ids.append(vocab.index(str(e["card_id"])))
        if len(ids) >= FX_SLOTS:
            break
    return ids + [PAD] * (FX_SLOTS - len(ids))


def _stack_card_ids(state: dict[str, Any], vocab: CardVocab) -> list[int]:
    ids = []
    for e in state.get("stack") or []:
        if isinstance(e, dict) and e.get("card_id"):
            ids.append(vocab.index(str(e["card_id"])))
        if len(ids) >= STACK_SLOTS:
            break
    return ids + [PAD] * (STACK_SLOTS - len(ids))


def _chain_cards(state: dict[str, Any]) -> list[str]:
    out: list[str] = []
    for e in state.get("combat_chain") or []:
        c = e.get("card_id") if isinstance(e, dict) else e
        if c:
            out.append(str(c))
    return out


def _chain_card_ids(state: dict[str, Any], vocab: CardVocab) -> list[int]:
    ids = [vocab.index(c) for c in _chain_cards(state)[:CHAIN_SLOTS]]
    return ids + [PAD] * (CHAIN_SLOTS - len(ids))


def state_token_card_slugs(state: dict[str, Any]):
    """Yield identity-bearing slugs across embedded zones (vocab building)."""
    for p in state.get("players", []) or []:
        h = _hero_of(p)
        if h:
            yield h
        for k in ("hand", "arsenal", "equipment", "graveyard", "banished",
                  "pitch", *_BOARD_KEYS):
            for c in (p.get(k) or []):
                if c:
                    yield str(c)
    yield from _chain_cards(state)
    for e in state.get("current_turn_effects") or []:
        if isinstance(e, dict) and e.get("card_id"):
            yield str(e["card_id"])
    for e in state.get("stack") or []:
        if isinstance(e, dict) and e.get("card_id"):
            yield str(e["card_id"])


def encode_action_scalars(action: dict[str, Any]) -> list[float]:
    feats = [0.0] * len(ACTION_TYPES)
    atype = str(action.get("type", ""))
    idx = _ATYPE_IDX.get(atype)
    if idx is None and atype.startswith("CHOOSE"):
        idx = _ATYPE_IDX["CHOOSE"]
    if idx is not None:
        feats[idx] = 1.0
    cost = action.get("cost") or {}
    res = _n(cost.get("resources")) if isinstance(cost, dict) else 0.0
    feats += [
        res / 3.0,
        1.0 if action.get("card_id") else 0.0,
        _n(action.get("hand_index")) / 10.0 if action.get("hand_index") is not None else 0.0,
        1.0,
    ]
    return feats


def action_card_id(action: dict[str, Any], vocab: CardVocab) -> int:
    return vocab.index(action.get("card_id"))


def next_mover(next_state: dict[str, Any]) -> int:
    return int(next_state.get("priority_player", 0) or 0)


def player_health(state: dict[str, Any], pid: int) -> float:
    return _n(_player(state, pid).get("health"))


# ---------------------------------------------------------------------------
# Draft
# ---------------------------------------------------------------------------
# Scalars: seat/8, pack#/3, pick#/15, pool/45, then POOL COMPOSITION — the
# constraints the HeuristicDeckBuilder will grade the pool against (12/9/9
# pitch curve) and the class-plurality counts that decide the seat's hero
# (the projected-hero signal; heroes are assigned after the draft).
DRAFT_SCALAR_DIM = 10
POOL_SLOTS = 45       # drafted-pool cards (role POOL) attended over
SEEN_SLOTS = 64       # cards seen wheel by but not taken (role SEEN = signal)
MAX_PACK = 15         # candidate cards in the current pack (role PACK)
WHEEL_SLOTS = 13      # incoming queue: cards from this round's prior packs

# Token roles for the attention draft net.
ROLE_CTX = 0
ROLE_PACK = 1
ROLE_POOL = 2
ROLE_SEEN = 3
# The known incoming queue: packs seen earlier in the CURRENT round arrive
# back over the next few picks (and wheel). SEEN is a flat all-history set;
# WHEEL is the structured "what's about to reach me" subset.
ROLE_WHEEL = 4
N_DRAFT_ROLES = 5

_DRAFT_CLASSES = ("illusionist", "wizard", "runeblade")
_class_map_cache: dict | None = None


def _draft_class_map() -> dict:
    """slug -> class name, lazily parsed from the cube's CustomCards.
    Self-contained (regex over the cube file) so this module stays
    torch-free and dependency-light; {} on any failure."""
    global _class_map_cache
    if _class_map_cache is not None:
        return _class_map_cache
    out: dict = {}
    try:
        import re as _re
        cube = Path(__file__).resolve().parents[2] / "OMN_Draft_3.5.txt"
        m = _re.search(r"\[CustomCards\]\s*(\[.*?\])\s*\[", cube.read_text(encoding="utf-8"), _re.S)
        for c in json.loads(m.group(1)):
            slug = _re.sub(r"[^a-z0-9]+", "_", str(c.get("name", "")).lower()).strip("_")
            ty = str(c.get("type") or "").lower()
            for cls in _DRAFT_CLASSES:
                if cls in ty:
                    out[slug] = cls
                    break
    except Exception:  # noqa: BLE001 — composition scalars degrade to zeros
        out = {}
    _class_map_cache = out
    return out


def encode_draft_scalars(seat: int, pack_number: int, pick_number: int,
                         pool: Sequence[str]) -> list[float]:
    pool = list(pool)
    red = sum(1 for c in pool if str(c).endswith("_red"))
    yellow = sum(1 for c in pool if str(c).endswith("_yellow"))
    blue = sum(1 for c in pool if str(c).endswith("_blue"))
    cmap = _draft_class_map()
    cls_counts = {k: 0 for k in _DRAFT_CLASSES}
    for c in pool:
        k = cmap.get(str(c))
        if k:
            cls_counts[k] += 1
    return [seat / 8.0, pack_number / 3.0, pick_number / 15.0, len(pool) / 45.0,
            red / 12.0, yellow / 9.0, blue / 9.0,        # builder pitch targets
            cls_counts["illusionist"] / 15.0,
            cls_counts["wizard"] / 15.0,
            cls_counts["runeblade"] / 15.0]


def wheel_cards(round_packs: Sequence[Sequence[str]], pool: Sequence[str],
                current_pack: Sequence[str]) -> list[str]:
    """The known incoming queue: cards from packs seen earlier in the
    CURRENT round (most recent first) that aren't in my pool or the pack
    in front of me. Capped at WHEEL_SLOTS."""
    ps, pk = set(pool), set(current_pack)
    out, taken = [], set()
    for pack in reversed(list(round_packs)):
        for c in pack:
            if c in ps or c in pk or c in taken:
                continue
            taken.add(c)
            out.append(c)
            if len(out) >= WHEEL_SLOTS:
                return out
    return out


def wheel_card_ids(round_packs, pool, current_pack, vocab: CardVocab) -> list[int]:
    return _slot_ids(wheel_cards(round_packs, pool, current_pack), vocab, WHEEL_SLOTS)


def _slot_ids(cards: Sequence[str], vocab: CardVocab, n_slots: int) -> list[int]:
    idx = [vocab.index(c) for c in list(cards)[:n_slots]]
    idx += [PAD] * (n_slots - len(idx))
    return idx


def pool_card_ids(pool: Sequence[str], vocab: CardVocab) -> list[int]:
    return _slot_ids(pool, vocab, POOL_SLOTS)


def seen_card_ids(seen: Sequence[str], vocab: CardVocab) -> list[int]:
    return _slot_ids(seen, vocab, SEEN_SLOTS)


def pack_slot_ids(pack: Sequence[str], vocab: CardVocab) -> list[int]:
    return _slot_ids(pack, vocab, MAX_PACK)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def loads(s: Any) -> Any:
    if isinstance(s, (bytes, bytearray)):
        s = s.decode()
    if isinstance(s, str):
        return json.loads(s)
    return s
