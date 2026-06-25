"""Transformer-policy draft bot scaffold (torch-optional).

Mirrors the gameplay :class:`TransformerBot` scaffold pattern: lazy torch
import inside ``__init__``, action-masked softmax head, deterministic
seeding.

For training:
    * inputs   = (pack tokens, drafted pool tokens, neighbour-signal tokens,
                  pack-number, pick-number, seat embedding)
    * outputs  = logits over the pack (mask out non-pack cards)
    * loss     = supervised imitation (winning trophies) OR IQL critic-
                 advantage weighted regression — wire this in
                 :mod:`python.training.iql_draft`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .base import DraftBot, DraftDecision, DraftPodView


@dataclass
class TransformerDraftConfig:
    vocab_size: int = 4096
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 4
    max_seq_len: int = 128
    max_pack_size: int = 16
    dropout: float = 0.1


class TransformerDraftBot(DraftBot):
    name = "draft-transformer"

    def __init__(
        self,
        config: TransformerDraftConfig | None = None,
        *,
        weights_path: str | None = None,
        device: str = "cpu",
        sample: bool = False,
        seed: int | None = None,
    ) -> None:
        try:
            import torch  # noqa: F401
        except ImportError as e:  # pragma: no cover
            raise RuntimeError(
                "TransformerDraftBot requires PyTorch. `pip install torch`."
            ) from e
        self.config = config or TransformerDraftConfig()
        self.device = device
        self.sample = sample
        self.model = self._build_model()
        if weights_path is not None:
            self._load_weights(weights_path)

    def reset(self, *, seed: int | None = None) -> None:
        if seed is not None:
            import torch
            torch.manual_seed(seed)

    # ------------------------------------------------------------------
    # Pick logic
    # ------------------------------------------------------------------
    def choose_card(
        self,
        pack: tuple[str, ...],
        drafted_cards: tuple[str, ...],
        seat_position: int,
        pick_number: int,
        pack_number: int,
        pod_state: DraftPodView,
    ) -> str | DraftDecision:
        import torch
        if not pack:
            raise RuntimeError("TransformerDraftBot received empty pack")
        tokens = self._tokenise(pack, drafted_cards, pod_state)
        with torch.no_grad():
            logits = self._forward(tokens)
        cfg = self.config
        mask = torch.full((cfg.max_pack_size,), float("-inf"))
        for i in range(min(len(pack), cfg.max_pack_size)):
            mask[i] = 0.0
        masked = logits + mask
        if self.sample:
            probs = torch.softmax(masked, dim=-1)
            idx = int(torch.multinomial(probs, 1).item())
        else:
            idx = int(torch.argmax(masked).item())
        idx = min(idx, len(pack) - 1)
        return DraftDecision(
            card_id=pack[idx],
            info={"policy": "transformer", "argmax_idx": idx, "device": self.device},
        )

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    def _build_model(self):
        import torch
        import torch.nn as nn
        cfg = self.config

        class DraftPolicyNet(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.tok = nn.Embedding(cfg.vocab_size, cfg.d_model)
                self.pos = nn.Embedding(cfg.max_seq_len, cfg.d_model)
                layer = nn.TransformerEncoderLayer(
                    d_model=cfg.d_model, nhead=cfg.n_heads,
                    dim_feedforward=cfg.d_model * 4,
                    dropout=cfg.dropout, batch_first=True,
                )
                self.enc = nn.TransformerEncoder(layer, num_layers=cfg.n_layers)
                self.head = nn.Linear(cfg.d_model, cfg.max_pack_size)

            def forward(self, ids: "torch.Tensor") -> "torch.Tensor":
                pos = torch.arange(ids.size(-1), device=ids.device)
                x = self.tok(ids) + self.pos(pos)[None]
                h = self.enc(x).mean(dim=1)
                return self.head(h).squeeze(0)

        return DraftPolicyNet().to(self.device).eval()

    def _load_weights(self, path: str) -> None:
        import torch
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.eval()

    def _forward(self, tokens: list[int]):
        import torch
        ids = torch.tensor([tokens], dtype=torch.long, device=self.device)
        return self.model(ids)

    # ------------------------------------------------------------------
    # Token construction (placeholder hash; train a real tokeniser later)
    # ------------------------------------------------------------------
    def _tokenise(
        self,
        pack: tuple[str, ...],
        drafted: tuple[str, ...],
        view: DraftPodView,
    ) -> list[int]:
        cfg = self.config
        toks: list[int] = []
        toks.append(view.pack_number)
        toks.append(view.pick_number)
        toks.append(view.seat)
        for c in pack[: cfg.max_pack_size]:
            toks.append(self._h(f"PACK:{c}") % cfg.vocab_size)
        for c in drafted:
            toks.append(self._h(f"OWN:{c}") % cfg.vocab_size)
        for c in view.left_neighbour_drafted:
            toks.append(self._h(f"LEFT:{c}") % cfg.vocab_size)
        for c in view.right_neighbour_drafted:
            toks.append(self._h(f"RIGHT:{c}") % cfg.vocab_size)
        toks = toks[: cfg.max_seq_len]
        while len(toks) < cfg.max_seq_len:
            toks.append(0)
        return toks

    @staticmethod
    def _h(s: str) -> int:
        h = 0xCBF29CE484222325
        for ch in s.encode("utf-8"):
            h ^= ch
            h = (h * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
        return h
