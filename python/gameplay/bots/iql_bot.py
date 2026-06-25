"""Gameplay bot backed by a trained IQL policy (with card embeddings).

Loads a checkpoint from ``python.training.iql_gameplay.train`` and scores
every legal action with the policy head, picking the argmax (or sampling
at ``temperature>0``). For data collection it supports epsilon-soft
exploration: with probability ``epsilon`` it defers to a ``BalancedBot``
(which sometimes blocks), guaranteeing defensive lines in the dataset even
while the learned policy is still weak.

Falls back entirely to the BalancedBot if torch or the checkpoint is
missing, so a tournament never crashes on a missing model.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

from ..env import Action
from .base import Bot, BotDecision
from .balanced_bot import BalancedBot
from ...training import features as F


class IQLGameplayBot(Bot):
    name = "iql"

    def __init__(self, *, checkpoint: str | Path, seed: int = 0,
                 temperature: float = 0.0, epsilon: float = 0.0,
                 block_prob: float = 0.5, explorer: str = "balanced") -> None:
        self.checkpoint = str(checkpoint)
        self.temperature = temperature
        self.epsilon = epsilon
        self._rng = random.Random(seed)
        # Exploration / fallback policy. "aggressive" almost never passes when
        # it can act (and blocks lethal), so the collected data is rich in
        # proactive lines instead of defensive passing.
        if explorer == "aggressive":
            from .aggressive_bot import AggressiveBot
            self._explore = AggressiveBot(seed=seed)
        else:
            self._explore = BalancedBot(seed=seed, block_prob=block_prob)
        self._net = None
        self._torch = None
        self._vocab: F.CardVocab | None = None
        self._load()

    def _load(self) -> None:
        try:
            import torch
            from ...training.iql_gameplay import build_net
        except ImportError:
            return
        try:
            ckpt = torch.load(self.checkpoint, map_location="cpu", weights_only=False)
            self._vocab = F.CardVocab.from_dict(ckpt["vocab"])
            net = build_net(ckpt["n_cards"], {
                "emb_dim": ckpt["emb_dim"], "hidden": ckpt["hidden"],
                "n_heads": ckpt.get("n_heads", 4), "n_layers": ckpt.get("n_layers", 2)},
                attr_dim=ckpt.get("attr_dim"))  # CC=79; None on legacy OMN -> OMN default
            net.load_state_dict(ckpt["net_state"])
            net.eval()
            self._net = net
            self._torch = torch
        except Exception as e:  # noqa: BLE001 — stale/incompatible/missing ckpt -> fall back
            print(f"[IQLGameplayBot] {type(e).__name__} loading {self.checkpoint} "
                  f"(likely architecture mismatch); using BalancedBot")
            self._net = None

    def reset(self, *, seed: int | None = None) -> None:
        if seed is not None:
            self._rng = random.Random(seed)
            self._explore.reset(seed=seed)

    def choose(self, state: dict[str, Any], legal_actions: list[Action], *, player_id: int) -> BotDecision:
        if self._net is None or self._torch is None or self._vocab is None or not legal_actions:
            return self._explore.choose(state, legal_actions, player_id=player_id)
        if self.epsilon > 0 and self._rng.random() < self.epsilon:
            return self._explore.choose(state, legal_actions, player_id=player_id)

        torch = self._torch
        net = self._net
        vocab = self._vocab
        ids = F.state_tokens(state, player_id, vocab)
        tst = F.state_token_state(state, player_id)
        with torch.no_grad():
            s_sc = torch.tensor([F.encode_state_scalars(state, player_id)], dtype=torch.float32)
            sv = net.state_vec(s_sc, torch.tensor([ids]),
                               torch.tensor([tst], dtype=torch.float32))
            a_sc = torch.tensor([F.encode_action_scalars(a.raw) for a in legal_actions], dtype=torch.float32)
            a_card = torch.tensor([F.action_card_id(a.raw, vocab) for a in legal_actions], dtype=torch.long)
            av = net.action_vec(a_sc, a_card)
            sv_rep = sv.expand(len(legal_actions), net.sdim)
            logits = net.Pi(sv_rep, av)
            if self.temperature and self.temperature > 0:
                probs = torch.softmax(logits / self.temperature, dim=0).tolist()
                pick = self._rng.choices(range(len(legal_actions)), weights=probs, k=1)[0]
            else:
                pick = int(torch.argmax(logits).item())
        chosen = legal_actions[pick]
        return BotDecision(action_id=chosen.action_id, info={"policy": "iql"})
