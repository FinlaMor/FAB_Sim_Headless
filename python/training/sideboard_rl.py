"""Stage-2 sideboard winrate RL: AWR fine-tune of the Stage-1 BC model.

The BC model (`deckbuilding/sideboard_model.py`) imitates the authors' per-matchup
card choices. Stage 2 nudges it toward choices that actually WIN, using a
contextual-bandit advantage-weighted regression (AWR) over the explored self-play
matches in `outputs/cc_sideboard_matches.jsonl` (collected with
`cc_selfplay --model <bc> --explore-sideboard <temp>` — the sampler gives the
choice VARIATION that argmax BC lacks).

Sideboarding is a ONE-SHOT decision (pick the per-card counts for a matchup), so
unlike the sequential draft IQL there's no Q/V bootstrap — the reward is just the
game's decisive outcome (+1 win / -1 loss; draws carry no signal and are dropped).
AWR pushes up the probability of counts seen in WINNING decks:

    pi_loss = mean( exp(beta * adv) * CE(logits, chosen_count) )      # adv = reward

Warm-started from the BC checkpoint, with a KL-to-FROZEN-BC anchor so it can't
drift far from the imitation prior where the (sparse) winrate signal is weak.

    python -m python.training.sideboard_rl --matches outputs/cc_sideboard_matches.jsonl
        --bc outputs/models/sideboard/sideboard_bc.pt --out-dir outputs/models/sideboard
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path

from ..deckbuilding.sideboard_model import (
    HeroVocab, SideboardBCHyper, build_net, card_features, MAX_COUNT)

_REPO = Path(__file__).resolve().parents[2]


@dataclass
class SideboardRLHyper:
    beta: float = 3.0          # AWR advantage temperature
    bc_anchor: float = 0.5     # weight on KL-to-frozen-BC (grounding)
    lr: float = 5e-4
    weight_decay: float = 1e-4
    n_steps: int = 1500
    batch_size: int = 256
    adv_clip: float = 10.0
    min_decisive: int = 20     # refuse to train on too little signal


def load_matches(path: str | Path, window: int = 0) -> list[dict]:
    rows: list[dict] = []
    p = Path(path)
    if not p.is_file():
        return rows
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return rows[-window:] if window else rows


def build_dataset(matches: list[dict], vocab: HeroVocab):
    """Each decisive match yields two sides; each (side, card) is one example
    (my_idx, opp_idx, card_features, chosen_count, reward). Draws are dropped."""
    import numpy as np

    my, opp, feats, label, reward = [], [], [], [], []
    n_dec = n_draw = 0
    for r in matches:
        wa, wb = int(r.get("winA", 0)), int(r.get("winB", 0))
        if wa == wb:                       # 0-0 / tie => no decisive signal
            n_draw += 1
            continue
        n_dec += 1
        rA = 1.0 if wa > wb else -1.0
        for mh, oh, ov, rew in (
            (r.get("heroA"), r.get("heroB"), r.get("overrideA") or {}, rA),
            (r.get("heroB"), r.get("heroA"), r.get("overrideB") or {}, -rA),
        ):
            for slug, cnt in ov.items():
                my.append(vocab.index(mh))
                opp.append(vocab.index(oh))
                feats.append(card_features(slug))
                label.append(min(MAX_COUNT, max(0, int(cnt))))
                reward.append(rew)
    A = lambda x, dt: np.asarray(x, dt)
    data = {"my": A(my, "int64"), "opp": A(opp, "int64"),
            "feat": A(feats, "float32"), "label": A(label, "int64"),
            "reward": A(reward, "float32")}
    return data, n_dec, n_draw


def train(*, matches_path: str | Path, bc_ckpt: str | Path, out_dir: str | Path,
          hyper: SideboardRLHyper | None = None, window: int = 0,
          device: str = "cpu") -> Path:
    import numpy as np
    import torch
    import torch.nn.functional as Fnn

    hyper = hyper or SideboardRLHyper()
    out_path = Path(out_dir) / "sideboard_rl.pt"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ck = torch.load(bc_ckpt, map_location=device, weights_only=False)
    vocab = HeroVocab.from_dict(ck["hero_vocab"])
    bc_hyper = SideboardBCHyper(**ck["hyper"])

    matches = load_matches(matches_path, window)
    data, n_dec, n_draw = build_dataset(matches, vocab)
    n = len(data["label"])
    if n_dec < hyper.min_decisive:
        raise RuntimeError(f"only {n_dec} decisive matches (< min_decisive="
                           f"{hyper.min_decisive}); collect more with "
                           f"`cc_selfplay --explore-sideboard`")
    nwin = int((data["reward"] > 0).sum())
    print(f"[sideboard-rl] {len(matches)} matches ({n_dec} decisive, {n_draw} draws) "
          f"-> {n} (side,card) examples | win-examples={nwin} | |heroes|={len(vocab)}")

    dev = torch.device(device)
    t = {k: torch.as_tensor(v).to(dev) for k, v in data.items()}

    net = build_net(len(vocab), bc_hyper).to(dev)
    net.load_state_dict(ck["net_state"])             # warm-start from BC
    bc = build_net(len(vocab), bc_hyper).to(dev)
    bc.load_state_dict(ck["net_state"])              # frozen anchor
    bc.eval()
    for p in bc.parameters():
        p.requires_grad_(False)

    opt = torch.optim.AdamW(net.parameters(), lr=hyper.lr,
                            weight_decay=hyper.weight_decay)
    bs = min(hyper.batch_size, n)
    last = 0.0
    for step in range(hyper.n_steps):
        idx = torch.randint(0, n, (bs,), device=dev)
        logits = net(t["my"][idx], t["opp"][idx], t["feat"][idx])
        ce = Fnn.cross_entropy(logits, t["label"][idx], reduction="none")
        with torch.no_grad():
            adv = t["reward"][idx].clamp(-hyper.adv_clip, hyper.adv_clip)
            w = torch.exp(hyper.beta * adv)
        pi_loss = (w * ce).mean()

        # KL(net || frozen BC) anchor — stay near the imitation prior.
        with torch.no_grad():
            bc_logp = Fnn.log_softmax(bc(t["my"][idx], t["opp"][idx], t["feat"][idx]), dim=-1)
        net_logp = Fnn.log_softmax(logits, dim=-1)
        kl = Fnn.kl_div(net_logp, bc_logp, log_target=True, reduction="batchmean")

        loss = pi_loss + hyper.bc_anchor * kl
        opt.zero_grad(); loss.backward(); opt.step()
        last = float(loss.item())
        if step % max(1, hyper.n_steps // 8) == 0 or step == hyper.n_steps - 1:
            print(f"  step {step:>5} loss={last:.4f} (pi={float(pi_loss):.4f} "
                  f"kl={float(kl):.4f})")

    ckpt = {"net_state": net.state_dict(), "hero_vocab": vocab.to_dict(),
            "hyper": ck["hyper"], "feat_dim": ck.get("feat_dim"),
            "classes": ck.get("classes"), "types": ck.get("types"),
            "keywords": ck.get("keywords"),
            "rl_hyper": asdict(hyper), "n_decisive": n_dec,
            "from_bc": str(bc_ckpt), "trained_at": time.time()}
    torch.save(ckpt, out_path)
    print(f"[sideboard-rl] saved -> {out_path} (final loss {last:.4f})")
    return out_path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Stage-2 sideboard winrate AWR.")
    ap.add_argument("--matches", default=str(_REPO / "outputs" / "cc_sideboard_matches.jsonl"))
    ap.add_argument("--bc", default=str(_REPO / "outputs" / "models" / "sideboard" / "sideboard_bc.pt"))
    ap.add_argument("--out-dir", default=str(_REPO / "outputs" / "models" / "sideboard"))
    ap.add_argument("--window", type=int, default=0, help="use only the last N matches (0=all)")
    ap.add_argument("--beta", type=float, default=3.0)
    ap.add_argument("--bc-anchor", type=float, default=0.5)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args(argv)
    hyper = SideboardRLHyper(beta=args.beta, bc_anchor=args.bc_anchor, n_steps=args.steps)
    train(matches_path=args.matches, bc_ckpt=args.bc, out_dir=args.out_dir,
          hyper=hyper, window=args.window, device=args.device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
