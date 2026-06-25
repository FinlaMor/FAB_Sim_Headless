"""Transformer policy scaffold (torch optional).

This module is **import-safe without torch**: ``import .transformer_bot``
will succeed even on a worker that has no PyTorch installed. The class
constructor performs the lazy import — instantiate ``TransformerBot``
only when you actually want to load a policy network.

The scaffold gives you:

* a tokeniser that flattens the JSON state into a fixed-length sequence
  of cardID + zone tokens,
* an action-masking head that aligns 1-indexed adapter action IDs with
  a fixed-width softmax,
* a model factory you can subclass for your own architecture (default:
  6-layer transformer encoder),
* a ``choose`` method that takes the masked argmax (or sample) for play.

For IQL specifically the ``forward()`` should be wired through your IQL
critic; the policy weights would come from the policy network produced
by IQL's advantage-weighted regression.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from ..env import Action
from .base import Bot, BotDecision


@dataclass
class TransformerConfig:
    vocab_size: int = 4096
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 6
    max_seq_len: int = 256
    max_actions: int = 256
    dropout: float = 0.1


class TransformerBot(Bot):
    name = "transformer"

    def __init__(
        self,
        config: TransformerConfig | None = None,
        *,
        weights_path: str | None = None,
        device: str = "cpu",
        sample: bool = False,
        seed: int | None = None,
    ) -> None:
        try:
            import torch  # noqa: F401 — imported for the side effects (CUDA init etc.)
        except ImportError as e:  # pragma: no cover
            raise RuntimeError(
                "TransformerBot requires PyTorch. Install with "
                "`pip install torch` (CPU-only) or follow the official "
                "GPU install guide."
            ) from e
        self.config = config or TransformerConfig()
        self.device = device
        self.sample = sample
        self._seed = seed
        self.model = self._build_model()
        if weights_path is not None:
            self._load_weights(weights_path)

    def reset(self, *, seed: int | None = None) -> None:
        if seed is not None:
            import torch
            torch.manual_seed(seed)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def choose(
        self,
        state: dict[str, Any],
        legal_actions: list[Action],
        *,
        player_id: int,
    ) -> BotDecision:
        import torch

        if not legal_actions:
            raise RuntimeError("TransformerBot received zero legal actions.")
        tokens = self._tokenise(state, player_id)
        with torch.no_grad():
            logits = self._forward(tokens)
        mask = torch.full((self.config.max_actions,), float("-inf"))
        for a in legal_actions:
            i = a.action_id - 1
            if 0 <= i < self.config.max_actions:
                mask[i] = 0.0
        masked = logits + mask
        if self.sample:
            probs = torch.softmax(masked, dim=-1)
            action_idx = int(torch.multinomial(probs, 1).item())
        else:
            action_idx = int(torch.argmax(masked).item())
        chosen_action_id = action_idx + 1

        # If the network produced an action outside the legal set (mask
        # makes that impossible at inference, but defensive coding never
        # hurts), fall back to the highest-logit legal.
        legal_ids = {a.action_id for a in legal_actions}
        if chosen_action_id not in legal_ids:
            chosen_action_id = sorted(legal_ids)[0]

        return BotDecision(
            action_id=chosen_action_id,
            info={
                "policy": "transformer",
                "device": self.device,
                "sample": self.sample,
                "top_logit": float(masked.max().item()),
            },
        )

    # ------------------------------------------------------------------
    # Model construction
    # ------------------------------------------------------------------
    def _build_model(self):
        import torch
        import torch.nn as nn

        cfg = self.config

        class PolicyNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.tok = nn.Embedding(cfg.vocab_size, cfg.d_model)
                self.pos = nn.Embedding(cfg.max_seq_len, cfg.d_model)
                layer = nn.TransformerEncoderLayer(
                    d_model=cfg.d_model, nhead=cfg.n_heads,
                    dim_feedforward=cfg.d_model * 4,
                    dropout=cfg.dropout, batch_first=True,
                )
                self.enc = nn.TransformerEncoder(layer, num_layers=cfg.n_layers)
                self.head = nn.Linear(cfg.d_model, cfg.max_actions)

            def forward(self, ids: "torch.Tensor") -> "torch.Tensor":
                pos = torch.arange(ids.size(-1), device=ids.device)
                x   = self.tok(ids) + self.pos(pos)[None]
                h   = self.enc(x)
                pooled = h.mean(dim=1)
                return self.head(pooled).squeeze(0)

        return PolicyNet().to(self.device).eval()

    def _load_weights(self, path: str) -> None:
        import torch
        state = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.eval()

    def _forward(self, tokens: list[int]):
        import torch
        ids = torch.tensor([tokens], dtype=torch.long, device=self.device)
        return self.model(ids)

    # ------------------------------------------------------------------
    # State -> token sequence (placeholder)
    # ------------------------------------------------------------------
    def _tokenise(self, state: dict[str, Any], player_id: int) -> list[int]:
        """Flatten the JSON state into a fixed-length integer sequence.

        Production replacement should use a trained tokenizer keyed on
        canonical card IDs. The placeholder hashes string fragments into
        the configured vocab range.
        """
        cfg = self.config
        tokens: list[int] = []
        # Game-level scalars first.
        tokens.append(self._h(state.get("phase", "")) % cfg.vocab_size)
        tokens.append(int(state.get("turn", 0)) % cfg.vocab_size)
        tokens.append(int(state.get("priority_player", 0)) % cfg.vocab_size)
        for p in state.get("players", []):
            pid = int(p.get("player_id", 0))
            tokens.append((self._h(p.get("hero", "")) ^ pid) % cfg.vocab_size)
            tokens.append((int(p.get("health") or 0) | 0x8000) % cfg.vocab_size)
            for zone in ("hand", "arsenal", "equipment", "pitch", "graveyard"):
                for card in p.get(zone, []) or []:
                    tokens.append(self._h(f"{zone}:{card}") % cfg.vocab_size)
                    if len(tokens) >= cfg.max_seq_len:
                        return tokens[: cfg.max_seq_len]
        # Pad
        while len(tokens) < cfg.max_seq_len:
            tokens.append(0)
        return tokens[: cfg.max_seq_len]

    @staticmethod
    def _h(s: str) -> int:
        h = 1469598103934665603
        for ch in s.encode("utf-8"):
            h ^= ch
            h = (h * 1099511628211) & 0xFFFFFFFFFFFFFFFF
        return h
