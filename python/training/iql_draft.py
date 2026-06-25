"""IQL trainer for the draft policy — signal-aware, with cross-card attention.

Each pick is one transition. The state is a *set of card tokens*, each
tagged with a role:

    PACK  — the cards you can pick from right now
    POOL  — the cards you've already drafted (your archetype so far)
    SEEN  — cards you've seen wheel past but didn't take (the SIGNAL:
            what's been passing = what's open)

A small Transformer encoder lets every candidate attend to your pool and
the seen/signal cards before it's scored, so the policy can read signals
and pivot into open lanes — not just greedily evaluate cards in isolation.

The SEEN set is reconstructed from the dataset's per-pick ``pack_contents``
history (cumulative union of packs a seat has looked at, minus its pool
and the current pack), so no dataset schema change is needed.

Reward is the placement-derived terminal reward at the final pick
(champion -> +1, last -> -1, linear). Draft is single-agent, so no
negamax flip. One forward of the net yields the policy logits, per-card
Q values, and V(state); the three IQL updates use those directly.
"""

from __future__ import annotations

import re
import time
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path

from . import features as F
from .iql_gameplay import load_parquet_rows


@dataclass
class DraftIQLHyperparams:
    advantage_beta: float = 3.0
    expectile_tau: float = 0.7
    gamma: float = 0.99
    adv_clip: float = 100.0
    lr: float = 3e-4
    batch_size: int = 128
    # 3000 -> 1500 on 2026-06-12: 3000 steps x batch 128 over ~2k selfplay
    # picks was ~190 epochs — pure memorization (loss 0.03, human pick
    # agreement at chance). The human-BC corpus below also ~4.7x's the data.
    n_steps: int = 1500
    emb_dim: int = 48
    n_heads: int = 4
    n_layers: int = 2
    target_tau: float = 0.005
    window: int = 0
    # --- Human behavior cloning (2026-06-12, RETIRED 2026-06-17) ---
    # ~180 real Draftmancer drafts (~7.4k human picks) carried the card-quality
    # signal selfplay lacked, and lifted human pick-agreement from chance (~25%)
    # to ~58% — but that PLATEAUED at the imitation ceiling (humans don't agree
    # with each other much higher). Default is now 0: we've moved OFF imitation
    # onto a real performance signal — the drafted deck's actual decisive
    # WINRATE in the round-robin (see `winrate_reward_weight`). Set >0 to blend
    # BC back in. Vocab still spans both corpora when loaded.
    human_bc_weight: float = 0.0
    human_refs: str = "real_draft_references"
    weight_decay: float = 1e-4
    # --- Winrate reward (2026-06-17) ---
    # The draft terminal reward is the seat's decisive-win RATE across the
    # round-robin (from the matches parquet), mapped 2*wr-1 in [-1,+1]. This is
    # the true objective (draft decks that WIN), is dense (each seat plays ~70
    # games), and replaces the placement-rank reward — which was recording all
    # zeros, so the draft IQL terminal had been contributing nothing. Seats with
    # no decisive games fall back to the deck-quality prior only.
    winrate_reward_weight: float = 1.0
    # The FIRST `human_holdout` reference drafts are excluded from BC
    # training and used as the validation set — the same first-40 subset the
    # continuous loop's agreement metric evaluates, so that metric stays
    # honest (no train/eval leakage).
    human_holdout: int = 40
    # Early stopping on held-out human top-1 agreement: stop after this many
    # evals (n_steps//10 apart) without improvement; best weights are saved.
    val_patience: int = 3
    # Deck-quality auxiliary terminal reward: placement is one noisy scalar
    # per 42 picks (and zero for draft-only data pods), so the terminal also
    # earns `deck_quality_weight * (2*quality-1)`, where quality in [0,1]
    # blends pitch-curve fit (vs the deck builder's 12/9/9 target) and
    # on-class fraction. Dense-ish, stationary, computable offline.
    deck_quality_weight: float = 0.3
    # Learned hero choice: weighted CE on (final pool -> assigned hero),
    # weighted by placement so winning seats' hero choices are imitated
    # hardest. Requires the decks parquet for the (pod, seat) -> hero join;
    # 0 (or no decks data) disables and pick_hero stays heuristic.
    hero_head_weight: float = 0.5


def _placement_reward(placement: int, max_place: int) -> float:
    if placement <= 0 or max_place <= 1:
        return 0.0
    return 1.0 - 2.0 * (placement - 1) / (max_place - 1)


_SEATNAME = re.compile(r"^(.*_pod\d+)_.+_seat(\d+)$")


def _winrate_by_seat(matches_dir, window: int = 0) -> dict:
    """{(pod_id, seat): decisive winrate in [0,1]} from the matches parquet.
    Only engine-decisive games count (draws excluded from both wins and the
    denominator), so this is pure deck-vs-deck performance. Player names are
    `{pod_id}_{hero}_seat{N}` (see _persist_matches)."""
    import glob as _glob
    try:
        import pyarrow.parquet as pq
    except ImportError:
        return {}
    if not matches_dir:
        return {}
    files = sorted(_glob.glob(str(Path(matches_dir) / "*.parquet")),
                   key=lambda p: Path(p).stat().st_mtime)
    if window and window > 0:
        files = files[-window:]
    wins: dict = {}
    games: dict = {}
    for fp in files:
        try:
            rows = pq.read_table(fp, columns=["p1_name", "p2_name",
                                              "winner_seat", "term_reason"]).to_pylist()
        except Exception:  # noqa: BLE001
            continue
        for r in rows:
            if r.get("term_reason") != "engine_winner":
                continue                       # only decisive games
            ws = r.get("winner_seat")
            if ws not in (0, 1):
                continue
            keys = []
            for nm in (r.get("p1_name"), r.get("p2_name")):
                m = _SEATNAME.match(str(nm or ""))
                keys.append((m.group(1), int(m.group(2))) if m else None)
            if None in keys:
                continue
            win_k, lose_k = keys[ws], keys[1 - ws]
            for k in (win_k, lose_k):
                games[k] = games.get(k, 0) + 1
            wins[win_k] = wins.get(win_k, 0) + 1
    return {k: wins.get(k, 0) / g for k, g in games.items() if g > 0}


def _deck_quality(pool: list[str]) -> float:
    """[0,1] offline pool quality: pitch-curve fit to the deck builder's
    12/9/9 target + on-class (plurality class or generic) fraction."""
    if not pool:
        return 0.0
    red = sum(1 for c in pool if str(c).endswith("_red"))
    yellow = sum(1 for c in pool if str(c).endswith("_yellow"))
    blue = sum(1 for c in pool if str(c).endswith("_blue"))
    curve = max(0.0, 1.0 - (abs(red - 12) + abs(yellow - 9) + abs(blue - 9)) / 30.0)
    cmap = F._draft_class_map()
    counts: dict = {}
    classed = 0
    for c in pool:
        k = cmap.get(str(c))
        if k:
            counts[k] = counts.get(k, 0) + 1
            classed += 1
    if counts:
        leader = max(counts.values())
        # generic (unclassed) cards are always on-class
        onclass = (leader + (len(pool) - classed)) / len(pool)
    else:
        onclass = 1.0
    return 0.5 * curve + 0.5 * onclass


# ---------------------------------------------------------------------------
# Attention net
# ---------------------------------------------------------------------------
# The format's hero choices (assigned post-draft; the hero head learns
# which hero a finished pool wants). Order defines the head's logit index.
DRAFT_HEROES = ("zyggy", "aurora_emissary_of_lightning",
                "oscilio_scion_of_the_third_age")


def build_draft_net(n_cards: int, dims, attr_matrix=None):
    """Cross-card self-attention draft net (with explicit card attributes).

    Each card token = id-embedding + role-embedding + projection of the
    card's attribute vector (cost/colour/type/class), so the model
    generalises across cards that share attributes instead of memorising
    ids. Tokens = [CTX] + PACK + POOL + SEEN. One forward returns:
      pi_logits [B, MAX_PACK]  — policy over pack cards
      q_pack    [B, MAX_PACK]  — Q for choosing each pack card
      v         [B]            — state value (from the CTX token)
      pack_pad  [B, MAX_PACK]  — True where a pack slot is padding

    ``attr_matrix`` [n_cards, ATTR_DIM] is registered as a buffer (saved in
    the checkpoint), so inference needs no cube file. Pass None to create a
    zero buffer that ``load_state_dict`` will overwrite.
    """
    import torch
    import torch.nn as nn
    from .card_attrs import ATTR_DIM

    emb_dim = dims["emb_dim"] if isinstance(dims, dict) else dims.emb_dim
    n_heads = dims["n_heads"] if isinstance(dims, dict) else dims.n_heads
    n_layers = dims["n_layers"] if isinstance(dims, dict) else dims.n_layers
    MP, POOL, SEEN = F.MAX_PACK, F.POOL_SLOTS, F.SEEN_SLOTS

    class AttnDraftNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.card_emb = nn.Embedding(n_cards, emb_dim, padding_idx=F.PAD)
            self.role_emb = nn.Embedding(F.N_DRAFT_ROLES, emb_dim)  # CTX/PACK/POOL/SEEN/WHEEL
            self.scalar_proj = nn.Linear(F.DRAFT_SCALAR_DIM, emb_dim)
            self.attr_proj = nn.Linear(ATTR_DIM, emb_dim)
            if attr_matrix is not None:
                buf = torch.as_tensor(attr_matrix, dtype=torch.float32)
            else:
                buf = torch.zeros(n_cards, ATTR_DIM)
            self.register_buffer("attr_table", buf)
            layer = nn.TransformerEncoderLayer(
                d_model=emb_dim, nhead=n_heads, dim_feedforward=4 * emb_dim,
                dropout=0.0, batch_first=True, activation="gelu")
            self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
            self.pi_head = nn.Linear(emb_dim, 1)
            self.q_head = nn.Linear(emb_dim, 1)
            self.v_head = nn.Linear(emb_dim, 1)
            # Learned hero choice: logits over DRAFT_HEROES from the CTX
            # encoding of a (usually final) pool state.
            self.hero_head = nn.Linear(emb_dim, len(DRAFT_HEROES))
            self.max_pack = MP

        def _encode(self, scalars, pack_ids, pool_ids, seen_ids, wheel_ids=None):
            B = pack_ids.shape[0]
            dev = pack_ids.device
            if wheel_ids is None:
                wheel_ids = torch.full((B, F.WHEEL_SLOTS), F.PAD,
                                       device=dev, dtype=torch.long)
            ctx = (self.scalar_proj(scalars).unsqueeze(1)
                   + self.role_emb(torch.full((B, 1), F.ROLE_CTX, device=dev, dtype=torch.long)))
            def toks(ids, role):
                return (self.card_emb(ids)
                        + self.role_emb(torch.full(ids.shape, role, device=dev, dtype=torch.long))
                        + self.attr_proj(self.attr_table[ids]))
            tokens = torch.cat([
                ctx,
                toks(pack_ids, F.ROLE_PACK),
                toks(pool_ids, F.ROLE_POOL),
                toks(seen_ids, F.ROLE_SEEN),
                toks(wheel_ids, F.ROLE_WHEEL),
            ], dim=1)
            pad = torch.cat([
                torch.zeros(B, 1, dtype=torch.bool, device=dev),
                pack_ids == F.PAD, pool_ids == F.PAD, seen_ids == F.PAD,
                wheel_ids == F.PAD,
            ], dim=1)
            enc = self.encoder(tokens, src_key_padding_mask=pad)
            return enc[:, 0], enc[:, 1:1 + MP]

        def forward(self, scalars, pack_ids, pool_ids, seen_ids, wheel_ids=None):
            ctx_out, pack_out = self._encode(scalars, pack_ids, pool_ids,
                                             seen_ids, wheel_ids)
            pi_logits = self.pi_head(pack_out).squeeze(-1)
            q_pack = self.q_head(pack_out).squeeze(-1)
            v = self.v_head(ctx_out).squeeze(-1)
            return pi_logits, q_pack, v, (pack_ids == F.PAD)

        def hero_logits(self, scalars, pool_ids):
            B = pool_ids.shape[0]
            dev = pool_ids.device
            pack = torch.full((B, MP), F.PAD, device=dev, dtype=torch.long)
            seen = torch.full((B, SEEN), F.PAD, device=dev, dtype=torch.long)
            ctx_out, _ = self._encode(scalars, pack, pool_ids, seen)
            return self.hero_head(ctx_out)

    return AttnDraftNet()


# ---------------------------------------------------------------------------
# Human reference drafts (Draftmancer logs -> BC picks)
# ---------------------------------------------------------------------------
def _load_human_drafts(refs_dir: str | Path) -> list[list[tuple]]:
    """Each draft = ordered [(pack_no, pick_no, booster_slugs, pick_idx)]
    for one HUMAN player. Card names slug like the cube loader:
    "Nebula Duality (red)_custom_OMN121" -> nebula_duality_red."""
    import json
    import re

    def slug(name: str) -> str:
        base = name.split("_custom_")[0]
        return re.sub(r"[^a-z0-9]+", "_", base.lower()).strip("_")

    out: list[list[tuple]] = []
    refs = Path(refs_dir)
    if not refs.is_dir():
        return out
    for fp in sorted(refs.glob("DraftLog_*.txt")):
        try:
            users = (json.loads(fp.read_text(encoding="utf-8")).get("users") or {})
        except Exception:  # noqa: BLE001
            continue
        for u in users.values():
            if u.get("isBot"):
                continue
            picks = []
            for p in u.get("picks") or []:
                booster = [slug(c) for c in (p.get("booster") or [])]
                sel = p.get("pick") or []
                if booster and sel and 0 <= int(sel[0]) < len(booster):
                    picks.append((int(p.get("packNum", 0)), int(p.get("pickNum", 0)),
                                  booster, int(sel[0])))
            picks.sort(key=lambda t: (t[0], t[1]))
            if picks:
                out.append(picks)
    return out


def _build_human(drafts: list[list[tuple]], vocab):
    """Encode human picks for the BC auxiliary loss (policy head only)."""
    import numpy as np

    sc, pack_ids, pool_ids, seen_ids, wheel_ids, chosen_idx = [], [], [], [], [], []
    for picks in drafts:
        pool: list[str] = []
        seen_seq: list[str] = []
        round_no = -1
        round_packs: list[list[str]] = []
        for pack_no, pick_no, booster, idx in picks:
            if pack_no != round_no:
                round_no = pack_no
                round_packs = []
            if idx >= F.MAX_PACK or not booster:
                round_packs.append(list(booster))
                seen_seq += booster
                pool.append(booster[idx])
                continue
            ps, pk = set(pool), set(booster)
            sl, taken = [], set()
            for c in reversed(seen_seq):
                if c in ps or c in pk or c in taken:
                    continue
                taken.add(c); sl.append(c)
                if len(sl) >= F.SEEN_SLOTS:
                    break
            sc.append(F.encode_draft_scalars(0, pack_no, pick_no, pool))
            pack_ids.append(F.pack_slot_ids(booster, vocab))
            pool_ids.append(F.pool_card_ids(pool, vocab))
            seen_ids.append(F.seen_card_ids(sl, vocab))
            wheel_ids.append(F.wheel_card_ids(round_packs, pool, booster, vocab))
            chosen_idx.append(idx)
            round_packs.append(list(booster))
            seen_seq += booster
            pool.append(booster[idx])
    if not sc:
        return None
    A = lambda x, dt: np.asarray(x, dtype=dt)
    return {
        "sc": A(sc, "float32"), "pack": A(pack_ids, "int64"),
        "pool": A(pool_ids, "int64"), "seen": A(seen_ids, "int64"),
        "wheel": A(wheel_ids, "int64"),
        "chosen_idx": A(chosen_idx, "int64"),
    }


def _build_hero(rows: list[dict], vocab, decks_dir, winrate_map=None) -> dict | None:
    """(final pool -> assigned hero, WINRATE-weighted) training set for the
    hero head. Joins drafts to the decks parquet on (pod_id, seat); draft-only
    pods have no deck row and are skipped. Winning seats' hero choices are
    imitated hardest (weight = winrate, fall back to placement)."""
    import glob as _glob
    import numpy as np
    try:
        import pyarrow.parquet as pq
    except ImportError:
        return None
    if not decks_dir:
        return None
    winrate_map = winrate_map or {}
    hero_by: dict = {}
    for fp in _glob.glob(str(Path(decks_dir) / "*.parquet")):
        try:
            for r in pq.read_table(fp, columns=["pod_id", "seat", "hero"]).to_pylist():
                hero_by[(r["pod_id"], int(r["seat"]))] = str(r["hero"])
        except Exception:  # noqa: BLE001
            continue
    if not hero_by:
        return None
    hero_idx = {h: i for i, h in enumerate(DRAFT_HEROES)}
    by_seat: dict = defaultdict(list)
    max_place: dict = defaultdict(int)
    for r in rows:
        by_seat[(r["pod_id"], r["seat"])].append(r)
        max_place[r["pod_id"]] = max(max_place[r["pod_id"]],
                                     int(r.get("placement", 0) or 0))
    sc, pool_ids, target, weight = [], [], [], []
    for (pod, seat), picks in by_seat.items():
        hero = hero_by.get((pod, int(seat)))
        hi = hero_idx.get(hero or "")
        if hi is None:
            continue
        picks.sort(key=lambda r: (r["pack_number"], r["pick_number"]))
        pool = [r["chosen_card"] for r in picks]
        wr = winrate_map.get((pod, int(seat)))
        if wr is not None:
            w = max(0.1, wr)                              # winrate directly
        else:
            place_r = _placement_reward(int(picks[-1].get("placement", 0) or 0),
                                        max_place[pod])
            w = max(0.1, (place_r + 1.0) / 2.0)
        sc.append(F.encode_draft_scalars(int(seat), 3, 15, pool))
        pool_ids.append(F.pool_card_ids(pool, vocab))
        target.append(hi)
        weight.append(w)
    if not sc:
        return None
    A = lambda x, dt: np.asarray(x, dtype=dt)
    return {"sc": A(sc, "float32"), "pool": A(pool_ids, "int64"),
            "target": A(target, "int64"), "weight": A(weight, "float32")}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def _build(rows: list[dict], hyper: DraftIQLHyperparams,
           extra_universe: set | None = None, winrate_map: dict | None = None):
    import numpy as np

    winrate_map = winrate_map or {}
    universe: set[str] = set(extra_universe or ())
    for r in rows:
        universe.update(F.loads(r["pack_contents_json"]) or [])
        universe.add(r["chosen_card"])
    vocab = F.CardVocab(sorted(universe))

    max_place_by_pod: dict[str, int] = defaultdict(int)
    for r in rows:
        max_place_by_pod[r["pod_id"]] = max(max_place_by_pod[r["pod_id"]], int(r.get("placement", 0) or 0))

    by_seat: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        by_seat[(r["pod_id"], r["seat"])].append(r)
    for k in by_seat:
        by_seat[k].sort(key=lambda r: (r["pack_number"], r["pick_number"]))

    sc, pack_ids, pool_ids, seen_ids, wheel_ids = [], [], [], [], []
    sc2, pack_ids2, pool_ids2, seen_ids2, wheel_ids2 = [], [], [], [], []
    rew, done, chosen_idx = [], [], []

    def seen_list(seen_seq: list[str], pool: list[str], pack: list[str]) -> list[str]:
        ps, pk = set(pool), set(pack)
        # most-recent-first, dedup, drop cards now in pool or in the current pack
        out, taken = [], set()
        for c in reversed(seen_seq):
            if c in ps or c in pk or c in taken:
                continue
            taken.add(c); out.append(c)
            if len(out) >= F.SEEN_SLOTS:
                break
        return out

    for (pod_id, seat), picks in by_seat.items():
        pool: list[str] = []
        seen_seq: list[str] = []   # ordered cards seen in prior packs
        max_place = max_place_by_pod.get(pod_id, 0)

        # Precompute per-pick state so we can also build s'.
        states = []
        tmp_pool: list[str] = []
        tmp_seen: list[str] = []
        round_no = -1
        round_packs: list[list[str]] = []  # packs seen earlier in current round
        for r in picks:
            pack = F.loads(r["pack_contents_json"]) or []
            chosen = r["chosen_card"]
            if int(r["pack_number"]) != round_no:
                round_no = int(r["pack_number"])
                round_packs = []
            scal = F.encode_draft_scalars(int(r["seat"]), int(r["pack_number"]),
                                          int(r["pick_number"]), tmp_pool)
            sl = seen_list(tmp_seen, tmp_pool, pack)
            ci = next((j for j, c in enumerate(pack[:F.MAX_PACK]) if c == chosen), 0)
            states.append({
                "scal": scal,
                "pack": F.pack_slot_ids(pack, vocab),
                "pool": F.pool_card_ids(tmp_pool, vocab),
                "seen": F.seen_card_ids(sl, vocab),
                "wheel": F.wheel_card_ids(round_packs, tmp_pool, pack, vocab),
                "chosen_idx": ci,
                "valid_pack": len(pack[:F.MAX_PACK]) > 0,
            })
            round_packs = round_packs + [list(pack)]
            tmp_seen = tmp_seen + list(pack)     # we've now seen this pack
            tmp_pool = tmp_pool + [chosen]

        n = len(states)
        final_pool = tmp_pool
        for i, st in enumerate(states):
            if not st["valid_pack"]:
                continue
            sc.append(st["scal"]); pack_ids.append(st["pack"])
            pool_ids.append(st["pool"]); seen_ids.append(st["seen"])
            wheel_ids.append(st["wheel"])
            chosen_idx.append(st["chosen_idx"])
            if i == n - 1:
                sc2.append([0.0] * F.DRAFT_SCALAR_DIM)
                pack_ids2.append([F.PAD] * F.MAX_PACK)
                pool_ids2.append([F.PAD] * F.POOL_SLOTS)
                seen_ids2.append([F.PAD] * F.SEEN_SLOTS)
                wheel_ids2.append([F.PAD] * F.WHEEL_SLOTS)
                # Primary signal: the deck's decisive WINRATE this round-robin
                # (2*wr-1). Fall back to placement-rank when winrate is missing
                # (e.g. draft-only data pods with no matches), which is itself
                # 0 today. Plus the offline deck-quality prior.
                wr = winrate_map.get((pod_id, int(seat)))
                if wr is not None:
                    term = hyper.winrate_reward_weight * (2.0 * wr - 1.0)
                else:
                    term = _placement_reward(int(picks[i].get("placement", 0) or 0), max_place)
                if hyper.deck_quality_weight:
                    term += hyper.deck_quality_weight * (2.0 * _deck_quality(final_pool) - 1.0)
                rew.append(term)
                done.append(1.0)
            else:
                nx = states[i + 1]
                sc2.append(nx["scal"]); pack_ids2.append(nx["pack"])
                pool_ids2.append(nx["pool"]); seen_ids2.append(nx["seen"])
                wheel_ids2.append(nx["wheel"])
                rew.append(0.0); done.append(0.0)

    if not rew:
        raise RuntimeError("no usable draft transitions found")

    A = lambda x, dt: np.asarray(x, dtype=dt)
    data = {
        "sc": A(sc, "float32"), "pack": A(pack_ids, "int64"),
        "pool": A(pool_ids, "int64"), "seen": A(seen_ids, "int64"),
        "wheel": A(wheel_ids, "int64"),
        "sc2": A(sc2, "float32"), "pack2": A(pack_ids2, "int64"),
        "pool2": A(pool_ids2, "int64"), "seen2": A(seen_ids2, "int64"),
        "wheel2": A(wheel_ids2, "int64"),
        "rew": A(rew, "float32"), "done": A(done, "float32"),
        "chosen_idx": A(chosen_idx, "int64"),
    }
    return data, vocab


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------
def train(
    *,
    parquet_dir: str | Path,
    out_dir: str | Path,
    hyper: DraftIQLHyperparams | None = None,
    device: str = "cpu",
    decks_dir: str | Path | None = None,
    matches_dir: str | Path | None = None,
) -> Path:
    try:
        import numpy as np
        import torch
        import torch.nn.functional as Fnn
    except ImportError as e:  # pragma: no cover
        raise RuntimeError("Install torch+numpy to train IQL: pip install torch numpy") from e

    hyper = hyper or DraftIQLHyperparams()
    out_path = Path(out_dir) / "iql_draft.pt"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = load_parquet_rows(parquet_dir, hyper.window)
    # Performance signal: decisive winrate per (pod, seat) from the matches
    # parquet (window matched to the drafts window).
    winrate_map = _winrate_by_seat(matches_dir, hyper.window) if matches_dir else {}
    # Human drafts: always loaded for the VALIDATION agreement metric (and to
    # widen the vocab); only used for the BC LOSS when human_bc_weight>0.
    human_drafts = _load_human_drafts(hyper.human_refs)
    human_universe: set = set()
    for picks in human_drafts:
        for _, _, booster, _ in picks:
            human_universe.update(booster)
    data, vocab = _build(rows, hyper, extra_universe=human_universe,
                         winrate_map=winrate_map)
    # Honest split: the FIRST `human_holdout` drafts are the validation set
    # (identical to the continuous loop's agreement subset) and are NEVER
    # trained on; BC (if enabled) uses the remainder.
    hold_n = min(hyper.human_holdout, max(0, len(human_drafts) - 1))
    val = _build_human(human_drafts[:hold_n], vocab) if hold_n else None
    human = (_build_human(human_drafts[hold_n:], vocab)
             if hyper.human_bc_weight and len(human_drafts) > hold_n else None)
    hero = (_build_hero(rows, vocab, decks_dir, winrate_map)
            if hyper.hero_head_weight and decks_dir else None)
    n = len(data["rew"])
    n_h = len(human["chosen_idx"]) if human else 0
    n_v = len(val["chosen_idx"]) if val else 0
    n_hero = len(hero["target"]) if hero else 0
    nz = sum(1 for v in data["rew"] if abs(float(v)) > 1e-6)
    print(f"[iql-draft] {n} pick transitions | |vocab|={len(vocab)} | "
          f"attn(emb={hyper.emb_dim},heads={hyper.n_heads},layers={hyper.n_layers}) | "
          f"window={hyper.window or 'all'} | winrate-reward seats={len(winrate_map)} "
          f"(nonzero terminals {nz}) | human_bc={n_h} (w={hyper.human_bc_weight}) | "
          f"val={n_v} | hero={n_hero} pools")

    # Explicit card attributes (cost/colour/type/class), baked into the net.
    attr_matrix = None
    try:
        from .card_attrs import CardAttributes, build_attr_matrix
        cube_path = Path(__file__).resolve().parents[2] / "OMN_Draft_3.5.txt"
        if cube_path.is_file():
            attr_matrix = build_attr_matrix(vocab, CardAttributes.from_cube(cube_path))
            print(f"[iql-draft] card attributes ON (dim={attr_matrix.shape[1]})")
    except Exception as e:  # noqa: BLE001 — attrs are a bonus, never block training
        print(f"[iql-draft] card attributes unavailable: {e!r}")

    dev = torch.device(device)
    t = {k: torch.as_tensor(v).to(dev) for k, v in data.items()}
    th = ({k: torch.as_tensor(v).to(dev) for k, v in human.items()}
          if human else None)
    tv = ({k: torch.as_tensor(v).to(dev) for k, v in val.items()}
          if val else None)
    thero = ({k: torch.as_tensor(v).to(dev) for k, v in hero.items()}
             if hero else None)
    net = build_draft_net(len(vocab), hyper, attr_matrix=attr_matrix).to(dev)
    targ = build_draft_net(len(vocab), hyper).to(dev); targ.load_state_dict(net.state_dict())
    opt = torch.optim.AdamW(net.parameters(), lr=hyper.lr,
                            weight_decay=hyper.weight_decay)
    tau, beta, gamma = hyper.expectile_tau, hyper.advantage_beta, hyper.gamma

    def expectile(diff):
        w = torch.where(diff > 0, tau, 1.0 - tau)
        return (w * diff.pow(2)).mean()

    def _val_top1() -> float:
        """Held-out human top-1 agreement (batched argmax)."""
        if tv is None:
            return -1.0
        hits = tot = 0
        with torch.no_grad():
            for lo in range(0, len(tv["chosen_idx"]), 512):
                sl = slice(lo, lo + 512)
                lg, _, _, pad = net(tv["sc"][sl], tv["pack"][sl],
                                    tv["pool"][sl], tv["seen"][sl], tv["wheel"][sl])
                lg = lg.masked_fill(pad, float("-inf"))
                hits += int((lg.argmax(dim=1) == tv["chosen_idx"][sl]).sum())
                tot += len(tv["chosen_idx"][sl])
        return hits / max(1, tot)

    bs = min(hyper.batch_size, n)
    last = 0.0
    best_val, best_state, evals_since_best = -1.0, None, 0
    eval_every = max(1, hyper.n_steps // 10)
    for step in range(hyper.n_steps):
        idx = torch.randint(0, n, (bs,), device=dev)
        ci = t["chosen_idx"][idx]
        pi_logits, q_pack, v, pack_pad = net(t["sc"][idx], t["pack"][idx],
                                             t["pool"][idx], t["seen"][idx],
                                             t["wheel"][idx])
        q_chosen = q_pack.gather(1, ci.unsqueeze(1)).squeeze(1)

        v_loss = expectile(q_chosen.detach() - v)

        with torch.no_grad():
            _, _, v2, _ = targ(t["sc2"][idx], t["pack2"][idx], t["pool2"][idx],
                               t["seen2"][idx], t["wheel2"][idx])
            q_target = t["rew"][idx] + gamma * (1.0 - t["done"][idx]) * v2
        q_loss = Fnn.mse_loss(q_chosen, q_target)

        with torch.no_grad():
            adv = (q_chosen - v).clamp(max=hyper.adv_clip)
            weight = torch.exp(beta * adv).clamp(max=hyper.adv_clip)
        logits = pi_logits.masked_fill(pack_pad, float("-inf"))
        logp = Fnn.log_softmax(logits, dim=1)
        chosen_logp = logp.gather(1, ci.unsqueeze(1)).squeeze(1)
        pi_loss = -(weight * chosen_logp).mean()

        # Human BC auxiliary: plain cross-entropy on a batch of real human
        # picks (policy head only — humans carry no Q/V targets).
        bc_loss = torch.tensor(0.0, device=dev)
        if th is not None:
            n_hh = len(th["chosen_idx"])
            hidx = torch.randint(0, n_hh, (min(bs, n_hh),), device=dev)
            h_logits, _, _, h_pad = net(th["sc"][hidx], th["pack"][hidx],
                                        th["pool"][hidx], th["seen"][hidx],
                                        th["wheel"][hidx])
            h_logits = h_logits.masked_fill(h_pad, float("-inf"))
            h_logp = Fnn.log_softmax(h_logits, dim=1)
            bc_loss = -(h_logp.gather(
                1, th["chosen_idx"][hidx].unsqueeze(1)).squeeze(1)).mean()

        # Hero head: placement-weighted CE over (final pool -> hero). The
        # set is tiny (one row per played seat), so use it whole each step.
        hero_loss = torch.tensor(0.0, device=dev)
        if thero is not None:
            hlg = net.hero_logits(thero["sc"], thero["pool"])
            hlp = Fnn.log_softmax(hlg, dim=1)
            nll = -hlp.gather(1, thero["target"].unsqueeze(1)).squeeze(1)
            hero_loss = (thero["weight"] * nll).mean()

        loss = (v_loss + q_loss + pi_loss + hyper.human_bc_weight * bc_loss
                + hyper.hero_head_weight * hero_loss)
        opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            for tp, p in zip(targ.parameters(), net.parameters()):
                tp.mul_(1 - hyper.target_tau).add_(hyper.target_tau * p)

        last = float(loss.item())
        if step % eval_every == 0 or step == hyper.n_steps - 1:
            vacc = _val_top1()   # human agreement — a MONITORING metric now
            print(f"  step {step:>6} loss={last:.4f} "
                  f"(v={float(v_loss):.4f} q={float(q_loss):.4f} pi={float(pi_loss):.4f} "
                  f"bc={float(bc_loss):.4f}) val_top1={vacc:.3f}")
            # Early-stop / best-weights on human agreement ONLY when BC is the
            # objective. With BC off we optimise WINRATE, so selecting weights by
            # human agreement would defeat the purpose — keep the final model.
            if hyper.human_bc_weight > 0:
                if vacc > best_val:
                    best_val = vacc
                    best_state = {k: p.detach().cpu().clone()
                                  for k, p in net.state_dict().items()}
                    evals_since_best = 0
                else:
                    evals_since_best += 1
                    if tv is not None and evals_since_best >= hyper.val_patience:
                        print(f"  early stop at step {step} "
                              f"(no val improvement in {hyper.val_patience} evals; "
                              f"best={best_val:.3f})")
                        break

    if best_state is not None:
        net.load_state_dict(best_state)

    ckpt = {
        "net_state": net.state_dict(), "vocab": vocab.to_dict(),
        "arch": "attn", "emb_dim": hyper.emb_dim, "n_heads": hyper.n_heads,
        "n_layers": hyper.n_layers, "n_cards": len(vocab),
        "heroes": list(DRAFT_HEROES),
        "hyper": asdict(hyper), "n_transitions": n, "trained_at": time.time(),
    }
    torch.save(ckpt, out_path)
    print(f"[iql-draft] saved -> {out_path} (final loss {last:.4f})")
    return out_path
