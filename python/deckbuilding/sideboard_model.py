"""Behavior-cloning sideboard model (v2 Stage 1).

Learns to imitate the authors' per-matchup card choices captured in the scraped
pools (`matchup_quantities`), conditioned on (my_hero, opp_hero, card_features),
so a NOVEL opponent gets a learned sideboard plan instead of only the
class-similarity fallback. Card features come from slug_index.json, so the model
generalises across cards that share cost/class/keywords rather than memorising
slugs — the same idea as the draft bot's card-attribute projection.

The model predicts a maindeck count (0..3) per card; `predict_overrides()` runs
it over a pool for a given opponent and returns {slug: count}, which feeds
straight into `sideboard.resolve(pool, opp_hero, overrides=...)` like the author
matchup data. This is a pure imitation prior (Stage 1); the winrate/RL refinement
(Stage 2) plugs the same predictions into the continuous-train loop later.

Pure-numpy feature extraction (torch imported lazily only for train/predict), so
importing this module never requires torch.
"""
from __future__ import annotations

import glob
import json
import time
from collections import Counter
from dataclasses import dataclass, asdict
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_SLUG_INDEX = _REPO / "slug_index.json"
MAX_COUNT = 3  # counts are clamped 0..3 (Unlimited handled separately at resolve)

# Fixed FaB feature vocabularies (stable -> fixed feature dim, baked in ckpt).
_CLASSES = ("Generic", "Brute", "Guardian", "Warrior", "Ninja", "Assassin",
            "Mechanologist", "Ranger", "Runeblade", "Wizard", "Illusionist",
            "Necromancer", "Pirate", "Merchant", "Shapeshifter", "Bard")
_TYPES = ("Action", "Instant", "Attack", "Defense Reaction", "Attack Reaction",
          "Equipment", "Weapon", "Item", "Aura", "Ally", "Landmark", "Block")
_KEYWORDS = ("Go again", "Dominate", "Intimidate", "Ward", "Arcane Barrier",
             "Reprise", "Combo", "Crush", "Boost", "Battleworn", "Phantasm",
             "Channel", "Charge", "Heave", "Specialization")
FEAT_DIM = 2 + 3 + 2 + len(_CLASSES) + len(_TYPES) + len(_KEYWORDS)


@dataclass
class SideboardBCHyper:
    emb_dim: int = 32
    hidden: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-4
    n_steps: int = 1200
    batch_size: int = 256
    # Examples where the matchup count DIFFERS from base are the actual
    # sideboard decisions; upweight them so the signal isn't washed out by the
    # many cards that stay at their base count regardless of opponent.
    delta_weight: float = 4.0


# ---------------------------------------------------------------------------
# Features (torch-free)
# ---------------------------------------------------------------------------
_idx_cache: dict | None = None


def _by_slug() -> dict:
    global _idx_cache
    if _idx_cache is None:
        _idx_cache = (json.loads(_SLUG_INDEX.read_text(encoding="utf-8"))["by_slug"]
                      if _SLUG_INDEX.is_file() else {})
    return _idx_cache


def _meta(slug: str) -> dict:
    idx = _by_slug()
    return idx.get(slug.replace("_", "-")) or idx.get(slug) or {}


def _num(x) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return 0.0


def card_features(slug: str) -> list[float]:
    m = _meta(slug)
    cost = m.get("cost")
    pitch = m.get("pitch")
    f = [_num(cost) / 10.0, 1.0 if cost is not None else 0.0]
    f += [1.0 if pitch == i else 0.0 for i in (1, 2, 3)]
    f += [_num(m.get("power")) / 8.0, _num(m.get("defense")) / 4.0]
    cl = set(m.get("classes") or [])
    f += [1.0 if c in cl else 0.0 for c in _CLASSES]
    ty = set(m.get("types") or [])
    f += [1.0 if t in ty else 0.0 for t in _TYPES]
    kw = set(m.get("keywords") or [])
    f += [1.0 if k in kw else 0.0 for k in _KEYWORDS]
    return f


# ---------------------------------------------------------------------------
# Dataset: (my_hero, opp_hero, card_features) -> author maindeck count
# ---------------------------------------------------------------------------
class HeroVocab:
    """hero slug -> index. 0 = UNK/pad."""

    def __init__(self, heroes=()):
        self.itos = ["<unk>"] + sorted({h for h in heroes if h})
        self.stoi = {h: i for i, h in enumerate(self.itos)}

    def index(self, h):
        return self.stoi.get(h or "", 0)

    def __len__(self):
        return len(self.itos)

    def to_dict(self):
        return {"itos": self.itos}

    @classmethod
    def from_dict(cls, d):
        v = cls()
        v.itos = list(d["itos"])
        v.stoi = {h: i for i, h in enumerate(v.itos)}
        return v


_name_map_cache: dict | None = None


def _hero_name_map() -> dict:
    """{hero short-name lower -> hero slug (underscore)} from slug_index, used
    to recover the opponent from a free-text matchup name (e.g. 'Boltyn',
    'Gravy Bones 1st')."""
    global _name_map_cache
    if _name_map_cache is None:
        out: dict = {}
        for slug, c in _by_slug().items():
            if "Hero" in (c.get("types") or []):
                short = str(c.get("hero") or "").lower().strip()
                if short:
                    out.setdefault(short, slug.replace("-", "_"))
        _name_map_cache = out
    return _name_map_cache


def _opp_from_matchup(m: dict) -> str | None:
    """Opponent hero slug for a matchup: linked heroIdentifiers first, else the
    longest hero short-name appearing as a whole word in the free-text name."""
    import re
    heroes = m.get("heroes") or []
    if heroes:
        return heroes[0]
    name = str(m.get("name") or "").lower()
    best = None
    for short, slug in _hero_name_map().items():
        if re.search(r"\b" + re.escape(short) + r"\b", name):
            if best is None or len(short) > len(best[0]):
                best = (short, slug)
    return best[1] if best else None


def build_dataset(decks_dir: str | Path):
    """Returns (examples, hero_vocab). Each example = (my_idx, opp_idx, feats,
    count, weight). Only matchups with an identifiable opponent hero and actual
    overrides are used."""
    import numpy as np

    pools = [json.loads(Path(p).read_text(encoding="utf-8"))
             for p in glob.glob(str(Path(decks_dir) / "cc_*.json"))]
    heroes = set()
    for pool in pools:
        heroes.add(pool.get("hero"))
        for m in pool.get("matchups") or []:
            heroes.update(m.get("heroes") or [])
    vocab = HeroVocab(heroes)

    my, opp, feats, label, weight = [], [], [], [], []
    for pool in pools:
        mh = pool.get("hero")
        base = Counter(pool.get("deck") or []) + Counter(pool.get("equipment") or [])
        universe = (set(base) | set(pool.get("sideboard") or [])
                    | set(pool.get("sideboard_equipment") or []))
        mq = pool.get("matchup_quantities") or {}
        for m in pool.get("matchups") or []:
            overrides = mq.get(m.get("matchupId")) or {}
            oh = _opp_from_matchup(m)
            if not oh or not overrides:
                continue
            cards = universe | set(overrides)
            for slug in cards:
                base_c = base.get(slug, 0)
                cnt = overrides.get(slug, base_c)
                cnt = min(MAX_COUNT, max(0, int(cnt)))
                my.append(vocab.index(mh))
                opp.append(vocab.index(oh))
                feats.append(card_features(slug))
                label.append(cnt)
                weight.append(1.0 if cnt == min(MAX_COUNT, max(0, base_c)) else 0.0)
    if not label:
        raise RuntimeError("no usable (matchup, card) examples; need scraped "
                           "pools with matchups + matchup_quantities")
    A = lambda x, dt: np.asarray(x, dt)
    data = {"my": A(my, "int64"), "opp": A(opp, "int64"),
            "feat": A(feats, "float32"), "label": A(label, "int64"),
            "is_base": A(weight, "float32")}
    return data, vocab


# ---------------------------------------------------------------------------
# Net
# ---------------------------------------------------------------------------
def build_net(n_heroes: int, hyper: SideboardBCHyper):
    import torch.nn as nn

    class SideboardBCNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.hero_emb = nn.Embedding(n_heroes, hyper.emb_dim, padding_idx=0)
            self.card_proj = nn.Linear(FEAT_DIM, hyper.hidden)
            self.mlp = nn.Sequential(
                nn.Linear(2 * hyper.emb_dim + hyper.hidden, hyper.hidden), nn.ReLU(),
                nn.Linear(hyper.hidden, hyper.hidden), nn.ReLU(),
                nn.Linear(hyper.hidden, MAX_COUNT + 1),
            )

        def forward(self, my, opp, feat):
            import torch
            h = torch.cat([self.hero_emb(my), self.hero_emb(opp),
                           self.card_proj(feat).relu()], dim=-1)
            return self.mlp(h)

    return SideboardBCNet()


def train(*, decks_dir: str | Path, out_dir: str | Path,
          hyper: SideboardBCHyper | None = None, device: str = "cpu") -> Path:
    import numpy as np
    import torch
    import torch.nn.functional as Fnn

    hyper = hyper or SideboardBCHyper()
    out_path = Path(out_dir) / "sideboard_bc.pt"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    data, vocab = build_dataset(decks_dir)
    n = len(data["label"])
    # Per-example weight: upweight the actual sideboard decisions (count != base).
    w = np.where(data["is_base"] > 0.5, 1.0, hyper.delta_weight).astype("float32")
    n_delta = int((data["is_base"] < 0.5).sum())
    print(f"[sideboard-bc] {n} (matchup,card) examples | {n_delta} are deltas | "
          f"|heroes|={len(vocab)} feat_dim={FEAT_DIM}")

    dev = torch.device(device)
    t = {k: torch.as_tensor(v).to(dev) for k, v in data.items()}
    tw = torch.as_tensor(w).to(dev)
    net = build_net(len(vocab), hyper).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=hyper.lr, weight_decay=hyper.weight_decay)

    # Honest held-out split (20%): report accuracy on unseen examples, and
    # separately on the DELTA examples (count != base) — the actual sideboard
    # decisions, where memorization vs real generalization actually shows.
    perm = np.random.default_rng(0).permutation(n)
    n_val = max(1, n // 5)
    val_i = torch.as_tensor(perm[:n_val]).to(dev)
    tr_i = torch.as_tensor(perm[n_val:]).to(dev)
    ntr = len(tr_i)

    def _val():
        with torch.no_grad():
            pv = net(t["my"][val_i], t["opp"][val_i], t["feat"][val_i]).argmax(-1)
            yv = t["label"][val_i]
            acc = float((pv == yv).float().mean())
            dm = t["is_base"][val_i] < 0.5
            dacc = float((pv[dm] == yv[dm]).float().mean()) if bool(dm.any()) else -1.0
        return acc, dacc, int(dm.sum())

    bs = min(hyper.batch_size, ntr)
    last = 0.0
    for step in range(hyper.n_steps):
        sel = tr_i[torch.randint(0, ntr, (bs,), device=dev)]
        logits = net(t["my"][sel], t["opp"][sel], t["feat"][sel])
        ce = Fnn.cross_entropy(logits, t["label"][sel], reduction="none")
        loss = (ce * tw[sel]).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        last = float(loss.item())
        if step % max(1, hyper.n_steps // 8) == 0 or step == hyper.n_steps - 1:
            acc, dacc, nd = _val()
            print(f"  step {step:>5} loss={last:.4f} val_acc={acc:.3f} "
                  f"val_delta_acc={dacc:.3f} (n_delta={nd})")

    ckpt = {"net_state": net.state_dict(), "hero_vocab": vocab.to_dict(),
            "hyper": asdict(hyper), "feat_dim": FEAT_DIM,
            "classes": _CLASSES, "types": _TYPES, "keywords": _KEYWORDS,
            "n_examples": n, "trained_at": time.time()}
    torch.save(ckpt, out_path)
    print(f"[sideboard-bc] saved -> {out_path} (final loss {last:.4f})")
    return out_path


# ---------------------------------------------------------------------------
# Inference: predict per-card overrides for a pool vs an opponent
# ---------------------------------------------------------------------------
class SideboardModel:
    def __init__(self, net, vocab: HeroVocab):
        self.net = net
        self.vocab = vocab

    @classmethod
    def load(cls, ckpt_path: str | Path, device: str = "cpu"):
        import torch
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        vocab = HeroVocab.from_dict(ck["hero_vocab"])
        hyper = SideboardBCHyper(**ck["hyper"])
        net = build_net(len(vocab), hyper)
        net.load_state_dict(ck["net_state"])
        net.eval()
        return cls(net, vocab)

    def _universe_logits(self, pool: dict, opp_hero: str):
        """(universe, logits[U, MAX_COUNT+1]) for the pool vs opponent."""
        import numpy as np
        import torch
        universe = sorted(set(pool.get("deck") or []) | set(pool.get("equipment") or [])
                          | set(pool.get("sideboard") or [])
                          | set(pool.get("sideboard_equipment") or []))
        if not universe:
            return [], None
        feat = torch.as_tensor(np.asarray([card_features(s) for s in universe], "float32"))
        my = torch.full((len(universe),), self.vocab.index(pool.get("hero")), dtype=torch.long)
        opp = torch.full((len(universe),), self.vocab.index(opp_hero), dtype=torch.long)
        with torch.no_grad():
            logits = self.net(my, opp, feat)
        return universe, logits

    def predict_overrides(self, pool: dict, opp_hero: str) -> dict[str, int]:
        """{slug: predicted maindeck count} over the pool's whole card universe
        for this opponent — drop into sideboard.resolve(..., overrides=...)."""
        universe, logits = self._universe_logits(pool, opp_hero)
        if not universe:
            return {}
        return {s: int(c) for s, c in zip(universe, logits.argmax(-1).tolist())}

    def sample_overrides(self, pool: dict, opp_hero: str, temperature: float = 1.0,
                         rng=None) -> dict[str, int]:
        """Like predict_overrides but SAMPLES each card's count from
        softmax(logits/temperature) — the exploration needed for Stage-2 winrate
        RL (deterministic argmax gives no choice variation to learn from). temp<=0
        falls back to argmax."""
        import numpy as np
        import torch
        universe, logits = self._universe_logits(pool, opp_hero)
        if not universe:
            return {}
        if temperature is None or temperature <= 0:
            return {s: int(c) for s, c in zip(universe, logits.argmax(-1).tolist())}
        rng = rng or np.random.default_rng()
        probs = torch.softmax(logits / float(temperature), dim=-1).numpy()
        counts = [int(rng.choice(probs.shape[1], p=p)) for p in probs]
        return {s: int(c) for s, c in zip(universe, counts)}


if __name__ == "__main__":
    import sys
    out = train(decks_dir=_REPO / "decks", out_dir=_REPO / "outputs" / "models" / "sideboard")
    print("done:", out)
