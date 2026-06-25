"""Draft bot backed by a trained, signal-aware attention IQL policy.

Loads a checkpoint from ``python.training.iql_draft.train`` (the attention
architecture) and scores every pack card with the policy head. The bot
tracks the cards it has *seen wheel past* across picks (reset each draft),
so the model attends to the draft signal — what's open — not just the
current pack and its own pool.

Falls back to the heuristic draft bot if torch/checkpoint are missing (or
an old non-attention checkpoint is supplied) so a pod never crashes.
"""

from __future__ import annotations

import random
from pathlib import Path

from .base import DraftBot, DraftDecision, DraftPodView
from .heuristic_bot import HeuristicDraftBot
from ...training import features as F


class IQLDraftBot(DraftBot):
    name = "iql-draft"

    def __init__(self, *, checkpoint: str | Path, seed: int = 0,
                 temperature: float = 0.0) -> None:
        self.checkpoint = str(checkpoint)
        # temperature > 0: softmax-sample the pick instead of argmax. Used
        # during DATA COLLECTION — with all 8 seats running the same
        # checkpoint at argmax, every cycle's drafts were near-clones and
        # the trainer never saw counterfactual picks. Deploy/gates use 0.
        self.temperature = temperature
        self._rng = random.Random(seed)
        self._fallback = HeuristicDraftBot(seed=seed)
        self._net = None
        self._torch = None
        self._vocab: F.CardVocab | None = None
        self._seen_seq: list[str] = []   # cards seen in packs so far this draft
        self._round_no: int = -1         # current pack round
        self._round_packs: list[list[str]] = []  # packs seen this round (wheel)
        self._load()

    def _load(self) -> None:
        try:
            import torch
            from ...training.iql_draft import build_draft_net
        except ImportError:
            return
        try:
            ckpt = torch.load(self.checkpoint, map_location="cpu", weights_only=False)
            if ckpt.get("arch") != "attn":
                raise ValueError("checkpoint is not the attention draft architecture")
            self._vocab = F.CardVocab.from_dict(ckpt["vocab"])
            self._heroes = list(ckpt.get("heroes") or [])
            net = build_draft_net(ckpt["n_cards"], {
                "emb_dim": ckpt["emb_dim"], "n_heads": ckpt["n_heads"], "n_layers": ckpt["n_layers"]})
            net.load_state_dict(ckpt["net_state"])
            net.eval()
            self._net = net
            self._torch = torch
        except Exception as e:  # noqa: BLE001 — stale/incompatible/missing -> fall back
            print(f"[IQLDraftBot] {type(e).__name__} loading {self.checkpoint} "
                  f"(likely architecture mismatch); using heuristic")
            self._net = None

    def reset(self, *, seed: int | None = None) -> None:
        self._seen_seq = []
        self._round_no = -1
        self._round_packs = []
        if seed is not None:
            self._rng = random.Random(seed)
            self._fallback.reset(seed=seed)

    # ------------------------------------------------------------------
    def _seen_for(self, pool, pack) -> list[str]:
        ps, pk = set(pool), set(pack)
        out, taken = [], set()
        for c in reversed(self._seen_seq):
            if c in ps or c in pk or c in taken:
                continue
            taken.add(c); out.append(c)
            if len(out) >= F.SEEN_SLOTS:
                break
        return out

    def _forward_logits(self, pack, drafted_cards, seat, pick_number, pack_number):
        torch = self._torch
        net = self._net
        v = self._vocab
        seen = self._seen_for(drafted_cards, pack)
        if pack_number != self._round_no:   # new pack round -> wheel resets
            self._round_no = pack_number
            self._round_packs = []
        with torch.no_grad():
            scal = torch.tensor([F.encode_draft_scalars(seat, pack_number, pick_number,
                                                        list(drafted_cards))],
                                dtype=torch.float32)
            pack_ids = torch.tensor([F.pack_slot_ids(pack, v)], dtype=torch.long)
            pool_ids = torch.tensor([F.pool_card_ids(list(drafted_cards), v)], dtype=torch.long)
            seen_ids = torch.tensor([F.seen_card_ids(seen, v)], dtype=torch.long)
            wheel_ids = torch.tensor([F.wheel_card_ids(self._round_packs,
                                                       list(drafted_cards), pack, v)],
                                     dtype=torch.long)
            pi_logits, _q, _v, _pad = net(scal, pack_ids, pool_ids, seen_ids, wheel_ids)
            return pi_logits[0].tolist()  # length MAX_PACK

    def _observe(self, pack) -> None:
        self._seen_seq.extend(pack)
        self._round_packs.append(list(pack))

    # ------------------------------------------------------------------
    def choose_card(self, pack, drafted_cards, seat_position, pick_number, pack_number, pod_state):
        if self._net is None or self._torch is None or self._vocab is None or not pack:
            self._observe(pack)
            return self._fallback.choose_card(
                pack, drafted_cards, seat_position, pick_number, pack_number, pod_state)
        logits = self._forward_logits(pack, drafted_cards, seat_position, pick_number, pack_number)
        # argmax (or temperature-sample) over the real pack cards only.
        npk = min(len(pack), F.MAX_PACK)
        if self.temperature and self.temperature > 0 and npk > 1:
            import math
            mx = max(logits[:npk])
            ws = [math.exp((logits[j] - mx) / self.temperature) for j in range(npk)]
            best = self._rng.choices(range(npk), weights=ws, k=1)[0]
        else:
            best = max(range(npk), key=lambda j: logits[j])
        self._observe(pack)
        return DraftDecision(card_id=pack[best], info={"policy": "iql-draft-attn"})

    def score_cards(self, pack, drafted_cards, seat_position, pick_number, pack_number, pod_state):
        """Advisor API: the trained policy's logit for each card (signal-aware)."""
        if self._net is None or self._torch is None or self._vocab is None or not pack:
            if hasattr(self._fallback, "score_cards"):
                out = self._fallback.score_cards(
                    pack, drafted_cards, seat_position, pick_number, pack_number, pod_state)
            else:
                out = {c: 0.0 for c in pack}
            self._observe(pack)
            return out
        logits = self._forward_logits(pack, drafted_cards, seat_position, pick_number, pack_number)
        out = {c: float(logits[j]) for j, c in enumerate(pack[:F.MAX_PACK])}
        self._observe(pack)
        return out

    def pick_hero(self, drafted_cards, available_heroes, card_classes):
        """Learned hero choice: the checkpoint's hero head scores the final
        pool; argmax over the heroes actually available. Falls back to the
        heuristic for old checkpoints or unknown heroes."""
        heroes = getattr(self, "_heroes", None) or []
        if (self._net is None or self._torch is None or self._vocab is None
                or not heroes or not available_heroes
                or not hasattr(self._net, "hero_logits")):
            return self._fallback.pick_hero(drafted_cards, available_heroes, card_classes)
        candidates = [(i, h) for i, h in enumerate(heroes) if h in set(available_heroes)]
        if not candidates:
            return self._fallback.pick_hero(drafted_cards, available_heroes, card_classes)
        torch = self._torch
        try:
            with torch.no_grad():
                pool = list(drafted_cards)
                scal = torch.tensor([F.encode_draft_scalars(0, 3, 15, pool)],
                                    dtype=torch.float32)
                pool_ids = torch.tensor([F.pool_card_ids(pool, self._vocab)],
                                        dtype=torch.long)
                logits = self._net.hero_logits(scal, pool_ids)[0].tolist()
            best = max(candidates, key=lambda c: logits[c[0]])
            return best[1]
        except Exception:  # noqa: BLE001 — hero choice must never crash a pod
            return self._fallback.pick_hero(drafted_cards, available_heroes, card_classes)
