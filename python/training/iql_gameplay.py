"""Implicit Q-Learning (IQL) trainer for the gameplay policy.

Full offline IQL over the per-transition parquet ``DatasetWriter`` emits.
Implements the three IQL updates for a 2-player zero-sum game (negamax
value bootstrapping):

    V(s)      <- expectile_tau( Q_target(s,a) - V(s) )        (a from data)
    Q(s,a)    <- r + gamma * (1-done) * sign * V_target(s')   (Bellman)
    pi(a|s)   <- advantage-weighted regression over legal actions

Improvements wired in here:

* **Shaped reward (item 1).** Terminal reward is margin-aware:
  ``sign * (0.5 + 0.5 * clip(|my_life - opp_life|/20, 0, 1))`` so a lethal
  blowout is worth ~1.0 and a narrow life-tiebreak win ~0.5. Falls back to
  flat +/-1 when ``use_shaped_reward=False``.
* **Recency window (item 3).** ``window`` keeps only the most recent N
  parquet files (= N cycles); 0 means all accumulated data.
* **Card embeddings (item 4).** ``IQLNet`` owns an ``nn.Embedding``; the
  state vector includes my/opp hero + mean-pooled hand embeddings and the
  action vector includes the played card's embedding.

Deployment scores every legal action with the policy head (see
``python.gameplay.bots.iql_bot``); the checkpoint bundles the vocab + dims
so the bot rebuilds the net exactly.
"""

from __future__ import annotations

import argparse
import glob
import time
from dataclasses import dataclass, asdict
from pathlib import Path

from . import features as F


@dataclass
class IQLHyperparams:
    gamma: float = 0.99
    expectile_tau: float = 0.7
    advantage_beta: float = 3.0
    adv_clip: float = 100.0
    lr: float = 3e-4
    batch_size: int = 256
    n_steps: int = 4000
    # Scaled 128/32 -> 256/64 on 2026-06-11: the critic plateaued (corr(V,G)
    # ~0.14-0.17, V std 0.075 vs return std 0.47) with ~250k transitions in
    # the window — capacity, not data, was the binding constraint. Candidates
    # train from scratch each iteration, so a size change coexists with an
    # older champion checkpoint (each ckpt carries its own dims).
    hidden: int = 256
    emb_dim: int = 64
    n_heads: int = 4
    n_layers: int = 2
    max_legal: int = 24
    target_tau: float = 0.005
    use_shaped_reward: bool = True
    # Z-score the advantages (per batch) before the exp() weighting. The 2026-06-10
    # Q/V audit found raw advantages living in [-0.16, +0.09] (std 0.067): at
    # beta=3 that puts every AWR weight in ~[0.62, 1.33] — effectively UNIFORM,
    # so the "advantage-weighted" regression silently degraded to plain
    # behavior cloning of the exploration mixture. Normalizing restores a real
    # good-action/bad-action contrast at the same beta regardless of the
    # critic's (collapsed or not) value scale.
    adv_norm: bool = True
    # Terminal transitions are ~1/85 of the data, so the +-1 reward anchors are
    # rarely sampled and the value function drifts toward predicting ~0
    # everywhere (audit: V std 0.075 vs MC-return std 0.63). Sample terminals
    # this many times more often than non-terminals.
    terminal_upsample: float = 8.0
    # Auxiliary V loss against the negamax MONTE-CARLO return of each
    # transition. Bootstrapped TD signal decays through ~85-decision chains;
    # regressing V on the realized return injects the win/loss outcome
    # directly at every state. 0 = off (pure expectile IQL).
    # 0.5 -> 1.0 on 2026-06-11 to fight persistent V under-dispersion
    # (V std 0.075 vs MC-return std 0.47 at the iter-6 audit).
    mc_aux_weight: float = 1.0
    # Dense aggression/tempo shaping (item: stop the policy stalling). Each
    # transition earns `aggression_weight * d`, where d is the change in the
    # mover's life lead, (my_life - opp_life), over that step, normalised by
    # 20. This is potential-based (Phi = lead/20), so it densifies the win
    # signal toward dealing damage and gaining tempo without changing the
    # optimal policy. 0 disables it (terminal reward only).
    #
    # NOTE: precisely because it is potential-based it is policy-INVARIANT, so
    # on its own it does NOT make a stalling policy start closing games. The
    # `draw_penalty` below is the non-invariant lever that does.
    aggression_weight: float = 0.0
    # Terminal penalty for a STEP-CAP DRAW. Without it a draw is reward 0
    # (neutral) while chasing a kill risks a -1 loss, so a risk-averse policy
    # correctly learns to coast to the clock (~92% of games draw). Marking each
    # drawn game's last recorded decision as a terminal with reward
    # `-draw_penalty` makes the value ordering win (+0.5..1) > draw (-p) > loss
    # (-1), so the policy is pushed to actually finish. 0 = legacy behaviour
    # (draw is a non-terminal reward-0 bootstrap).
    draw_penalty: float = 0.0
    # Dense anti-stall living cost. Every NON-terminal recorded transition of a
    # DRAWN game (winner==0) is charged `-draw_step_cost`. Unlike draw_penalty
    # (one terminal, ~1.7% of transitions) this hits ~all drawn-game
    # transitions (the bulk of the data, since ~92% of games draw), so stalling
    # lines are devalued densely with full coverage and the contrast vs lethal
    # lines (reward 0 / +1) is large. This is outcome-conditioned relabeling,
    # NOT potential-based, so it genuinely changes the policy. Keep small so a
    # draw stays better than a loss (total drawn-game cost should stay < 1).
    draw_step_cost: float = 0.0
    # Global per-decision LIVING COST on every non-terminal transition (ALL
    # games). With gamma<1 a sooner win is already worth more; this AMPLIFIES it
    # so the policy prefers a fast lethal over a slow one (fewer recorded steps =
    # less accrued cost). Keep small vs the +-1 terminal so it never inverts the
    # win > draw > loss ordering (a fast loss must stay worse than a slow draw).
    # 0 = off.
    time_penalty: float = 0.0
    window: int = 0  # 0 = all parquet files; N = most recent N


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_parquet_rows(parquet_dir: str | Path, window: int = 0) -> list[dict]:
    import pyarrow.parquet as pq
    files = sorted(glob.glob(str(Path(parquet_dir) / "*.parquet")),
                   key=lambda p: Path(p).stat().st_mtime)
    if window and window > 0:
        files = files[-window:]
    rows: list[dict] = []
    for fp in files:
        rows.extend(pq.read_table(fp).to_pylist())
    return rows


def _terminal_reward(row: dict, shaped: bool) -> float:
    if not row.get("done"):
        return 0.0
    winner = int(row.get("winner", 0) or 0)
    mover = int(row.get("player_to_move", 0) or 0)
    if winner == 0 or mover not in (1, 2):
        return 0.0
    sign = 1.0 if winner == mover else -1.0
    if not shaped:
        return sign
    opp = 2 if mover == 1 else 1
    ns = F.loads(row["next_state_json"])
    margin = min(abs(F.player_health(ns, mover) - F.player_health(ns, opp)) / 20.0, 1.0)
    return sign * (0.5 + 0.5 * margin)


def _build(rows: list[dict], hyper: IQLHyperparams, vocab: "F.CardVocab | None" = None):
    import numpy as np

    # OMN path: build the vocab from the training rows. CC path: a prebuilt vocab
    # (cards U state sentinels, from build_cc_card_table) is passed in so it
    # stays aligned to the CC attr_matrix; tokens absent from it map to UNK.
    if vocab is None:
        vocab = F.CardVocab(_iter_cards(rows))
    ML, AS, HS = hyper.max_legal, F.ACTION_SCALAR_DIM, F.HAND_SLOTS

    # For draw_penalty: find each game's LAST recorded decision (max step_index)
    # and its winner, so a step-cap draw's final decision can be turned into a
    # terminal carrying -draw_penalty (see IQLHyperparams.draw_penalty).
    last_step: dict = {}
    winner_by: dict = {}
    if hyper.draw_penalty or hyper.draw_step_cost:
        for r in rows:
            gid = r.get("game_id")
            si = int(r.get("step_index", 0) or 0)
            if gid not in last_step or si > last_step[gid]:
                last_step[gid] = si
            winner_by[gid] = int(r.get("winner", 0) or 0)

    s_sc, a_sc, s2_sc = [], [], []
    s_ids, s_ids2 = [], []          # [N, STATE_TOKENS] zone token ids (state, next)
    s_tst, s_tst2 = [], []          # [N, STATE_TOKENS, TOKEN_STATE_DIM] per-token state
    a_card = []
    rew, done, sign = [], [], []
    lf_sc, lf_card, lf_mask, chosen_idx = [], [], [], []
    gids, sidx = [], []             # game ordering for the MC-return aux target
    _n_draw_term = [0]              # count of drawn-game terminals penalised
    _n_draw_cost = [0]             # count of drawn-game transitions charged the living cost

    for r in rows:
        mover = int(r.get("player_to_move", 0) or 0)
        if mover not in (1, 2):
            continue
        state = F.loads(r["state_json"]); nstate = F.loads(r["next_state_json"])
        chosen = F.loads(r["chosen_action_json"]); legals = F.loads(r["legal_actions_json"]) or []
        # Forced windows (a single legal action — almost always PASS) are not
        # decisions: there is nothing for the policy to learn and they make up
        # ~85% of raw transitions, swamping the gradient toward PASS. Drop them
        # here (a safety net for datasets recorded before match.py started
        # skipping them) but keep any terminal row so its reward survives.
        if len(legals) <= 1 and not bool(r.get("done")):
            continue

        nm = F.next_mover(nstate)

        lfs = [[0.0] * AS for _ in range(ML)]
        lfc = [F.PAD] * ML
        lm = [0.0] * ML
        ci = 0
        caid = chosen.get("action_id")
        for j, la in enumerate(legals[:ML]):
            lfs[j] = F.encode_action_scalars(la)
            lfc[j] = F.action_card_id(la, vocab)
            lm[j] = 1.0
            if la.get("action_id") == caid:
                ci = j
        if not any(lm):
            continue

        s_sc.append(F.encode_state_scalars(state, mover))
        a_sc.append(F.encode_action_scalars(chosen))
        s2_sc.append(F.encode_state_scalars(nstate, mover))
        s_ids.append(F.state_tokens(state, mover, vocab))
        s_ids2.append(F.state_tokens(nstate, mover, vocab))
        s_tst.append(F.state_token_state(state, mover))
        s_tst2.append(F.state_token_state(nstate, mover))
        a_card.append(F.action_card_id(chosen, vocab))
        is_done = bool(r.get("done"))
        gid = r.get("game_id")
        is_draw_game = bool(winner_by) and winner_by.get(gid, 1) == 0
        # A drawn game's final recorded decision becomes a terminal carrying
        # -draw_penalty (overrides shaping; it IS the end of the line).
        is_draw_term = (
            not is_done and bool(hyper.draw_penalty) and is_draw_game
            and int(r.get("step_index", 0) or 0) == last_step.get(gid)
        )
        if is_draw_term:
            step_r = -float(hyper.draw_penalty)
            _n_draw_term[0] += 1
        else:
            step_r = _terminal_reward(r, hyper.use_shaped_reward)
            if hyper.aggression_weight:
                opp = 2 if mover == 1 else 1
                lead_s = F.player_health(state, mover) - F.player_health(state, opp)
                lead_n = F.player_health(nstate, mover) - F.player_health(nstate, opp)
                step_r += hyper.aggression_weight * (lead_n - lead_s) / 20.0
            # Dense anti-stall living cost on every other transition of a draw.
            if hyper.draw_step_cost and is_draw_game and not is_done:
                step_r -= float(hyper.draw_step_cost)
                _n_draw_cost[0] += 1
            # Global per-decision time cost (all games) -> reward faster wins.
            if hyper.time_penalty and not is_done:
                step_r -= float(hyper.time_penalty)
        rew.append(step_r)
        done.append(1.0 if (is_done or is_draw_term) else 0.0)
        sign.append(1.0 if nm == mover else -1.0)
        lf_sc.append(lfs); lf_card.append(lfc); lf_mask.append(lm); chosen_idx.append(ci)
        gids.append(gid); sidx.append(int(r.get("step_index", 0) or 0))

    if not rew:
        raise RuntimeError("no usable gameplay transitions found")

    # Negamax Monte-Carlo return per transition (aux target for V):
    # G_t = r_t + gamma * sign_t * G_{t+1} over RECORDED decisions, G = r at a
    # terminal. Same perspective convention as the Bellman backup.
    mc_g = [0.0] * len(rew)
    order: dict = {}
    for i, g in enumerate(gids):
        order.setdefault(g, []).append(i)
    for g, idxs in order.items():
        idxs.sort(key=lambda i: sidx[i])
        g_next = 0.0
        for i in reversed(idxs):
            if done[i] >= 0.5:
                g_next = rew[i]
            else:
                g_next = rew[i] + hyper.gamma * sign[i] * g_next
            mc_g[i] = g_next
    if hyper.draw_penalty or hyper.draw_step_cost or hyper.time_penalty:
        print(f"[iql-gameplay] draw_penalty={hyper.draw_penalty} on {_n_draw_term[0]} "
              f"terminals | draw_step_cost={hyper.draw_step_cost} on {_n_draw_cost[0]} "
              f"| time_penalty={hyper.time_penalty} (of {len(rew)} transitions)")

    A = lambda x, dt: np.asarray(x, dtype=dt)
    data = {
        "s_sc": A(s_sc, "float32"), "a_sc": A(a_sc, "float32"), "s2_sc": A(s2_sc, "float32"),
        "s_ids": A(s_ids, "int64"), "s_ids2": A(s_ids2, "int64"),
        "s_tst": A(s_tst, "float32"), "s_tst2": A(s_tst2, "float32"),
        "a_card": A(a_card, "int64"),
        "rew": A(rew, "float32"), "done": A(done, "float32"), "sign": A(sign, "float32"),
        "mc_g": A(mc_g, "float32"),
        "lf_sc": A(lf_sc, "float32"), "lf_card": A(lf_card, "int64"),
        "lf_mask": A(lf_mask, "float32"), "chosen_idx": A(chosen_idx, "int64"),
    }
    return data, vocab


def _iter_cards(rows: list[dict]):
    for r in rows:
        st = F.loads(r["state_json"])
        yield from F.state_token_card_slugs(st)
        ca = F.loads(r["chosen_action_json"]).get("card_id")
        if ca:
            yield str(ca)
        for la in F.loads(r["legal_actions_json"]) or []:
            if la.get("card_id"):
                yield str(la["card_id"])


# ---------------------------------------------------------------------------
# Net
# ---------------------------------------------------------------------------
def _mlp(in_dim: int, hidden: int, out_dim: int):
    import torch.nn as nn
    return nn.Sequential(
        nn.Linear(in_dim, hidden), nn.ReLU(),
        nn.Linear(hidden, hidden), nn.ReLU(),
        nn.Linear(hidden, out_dim),
    )


def build_net(n_cards: int, hyper_or_dims, attr_matrix=None, attr_dim=None):
    """Construct an IQLNet with cross-card attention + explicit attributes.

    The state is encoded by self-attention over [CTX, my-hero, opp-hero,
    hand cards] tokens (each = id-embedding + role + attribute projection),
    so the policy can reason about hand/board synergies instead of mean-
    pooling the hand. The CTX token carries the state scalars (phase, life,
    and the combat-awareness signals) and its attended output is the state
    vector. Actions are then scored against that vector — which keeps
    variable-length and non-card actions (PASS, decisions) easy to handle.

    ``attr_matrix`` [n_cards, attr_dim] is a saved buffer (pass None for a
    zero buffer that ``load_state_dict`` overwrites). ``attr_dim`` sizes the
    attribute projection; if None it's inferred from ``attr_matrix`` when given,
    else defaults to the OMN cube schema (``card_attrs.ATTR_DIM``). The CC
    schema (``cc_card_attrs.CC_ATTR_DIM``) is wider, so the CC retrain passes
    its dim explicitly (and target nets must pass the SAME dim)."""
    import torch
    import torch.nn as nn
    from .card_attrs import ATTR_DIM as _OMN_ATTR_DIM

    if attr_dim is None:
        attr_dim = (int(attr_matrix.shape[1]) if attr_matrix is not None
                    else _OMN_ATTR_DIM)
    ATTR_DIM = attr_dim

    if isinstance(hyper_or_dims, dict):
        emb_dim = hyper_or_dims["emb_dim"]; hidden = hyper_or_dims["hidden"]
        n_heads = hyper_or_dims.get("n_heads", 4); n_layers = hyper_or_dims.get("n_layers", 2)
    else:
        emb_dim = hyper_or_dims.emb_dim; hidden = hyper_or_dims.hidden
        n_heads = hyper_or_dims.n_heads; n_layers = hyper_or_dims.n_layers
    SS, ASd = F.STATE_SCALAR_DIM, F.ACTION_SCALAR_DIM
    R_CTX = F.R_CTX  # the rest of the zone roles live in F.STATE_ROLES

    class IQLNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(n_cards, emb_dim, padding_idx=F.PAD)
            self.role_emb = nn.Embedding(F.N_STATE_ROLES, emb_dim)
            self.attr_proj = nn.Linear(ATTR_DIM, emb_dim)
            self.scalar_proj = nn.Linear(SS, emb_dim)
            # Per-token state (counters, ready/face-up) — see
            # features.TOKEN_STATE_DIM. Restores the card state the
            # de-stride id extraction drops.
            self.tok_state_proj = nn.Linear(F.TOKEN_STATE_DIM, emb_dim)
            buf = (torch.as_tensor(attr_matrix, dtype=torch.float32)
                   if attr_matrix is not None else torch.zeros(n_cards, ATTR_DIM))
            self.register_buffer("attr_table", buf)
            # Fixed positional roles for the zone tokens (length STATE_TOKENS).
            self.register_buffer("state_roles",
                                 torch.as_tensor(F.STATE_ROLES, dtype=torch.long))
            layer = nn.TransformerEncoderLayer(
                d_model=emb_dim, nhead=n_heads, dim_feedforward=4 * emb_dim,
                dropout=0.0, batch_first=True, activation="gelu")
            self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
            self.sdim = emb_dim            # state vector = CTX token output
            self.adim = ASd + emb_dim
            self.v = _mlp(self.sdim, hidden, 1)
            self.q = _mlp(self.sdim + self.adim, hidden, 1)
            self.pi = _mlp(self.sdim + self.adim, hidden, 1)

        def _card(self, ids):
            return self.emb(ids) + self.attr_proj(self.attr_table[ids])

        def _role(self, shape, role, dev):
            return self.role_emb(torch.full(shape, role, device=dev, dtype=torch.long))

        def state_vec(self, s_sc, ids, tok_state=None):
            # ids: [B, STATE_TOKENS] embedding indices for every visible zone
            # (PAD = empty slot, UNK = masked opponent-arsenal card).
            # tok_state: [B, STATE_TOKENS, TOKEN_STATE_DIM] per-token
            # counters/ready (None = zeros, for older callers).
            B = ids.shape[0]; dev = ids.device
            ctx = self.scalar_proj(s_sc).unsqueeze(1) + self._role((B, 1), R_CTX, dev)
            roles = self.state_roles.unsqueeze(0).expand(B, -1)         # [B,T]
            cards = self._card(ids) + self.role_emb(roles)             # [B,T,E]
            if tok_state is not None:
                cards = cards + self.tok_state_proj(tok_state)
            tokens = torch.cat([ctx, cards], dim=1)                    # [B,1+T,E]
            pad = torch.cat([
                torch.zeros(B, 1, dtype=torch.bool, device=dev),
                ids == F.PAD,
            ], dim=1)
            enc = self.encoder(tokens, src_key_padding_mask=pad)
            return enc[:, 0]              # CTX token output [B, E]

        def action_vec(self, a_sc, a_card):
            return torch.cat([a_sc, self._card(a_card)], dim=-1)

        def V(self, sv):
            return self.v(sv).squeeze(-1)

        def Q(self, sv, av):
            return self.q(torch.cat([sv, av], dim=-1)).squeeze(-1)

        def Pi(self, sv, av):
            return self.pi(torch.cat([sv, av], dim=-1)).squeeze(-1)

    return IQLNet()


def train(
    *,
    parquet_dir: str | Path,
    out_dir: str | Path,
    hyper: IQLHyperparams | None = None,
    device: str = "cpu",
    card_table: str | Path | None = None,
    warm_start: str | Path | None = None,
) -> Path:
    try:
        import numpy as np
        import torch
        import torch.nn.functional as Fnn
    except ImportError as e:  # pragma: no cover
        raise RuntimeError("Install torch+numpy to train IQL: pip install torch numpy") from e

    hyper = hyper or IQLHyperparams()
    out_path = Path(out_dir) / "iql_gameplay.pt"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # CC path: a prebuilt vocab + attribute table (python.training.build_cc_card_table)
    # gives the net "eyes" on the full CC card pool. OMN path: vocab from rows +
    # cube attributes (back-compat default when no card_table is supplied).
    pre_vocab = None
    attr_matrix = None
    attr_dim = None
    if card_table:
        tbl = torch.load(card_table, map_location="cpu", weights_only=False)
        pre_vocab = F.CardVocab.from_dict(tbl["vocab"])
        attr_matrix = tbl["attr_matrix"]
        attr_dim = int(tbl["attr_dim"])
        print(f"[iql-gameplay] CC card table: {card_table} "
              f"(|vocab|={len(pre_vocab)}, attr_dim={attr_dim})")

    rows = load_parquet_rows(parquet_dir, hyper.window)
    data, vocab = _build(rows, hyper, vocab=pre_vocab)
    n = len(data["rew"])
    print(f"[iql-gameplay] {n} transitions | |vocab|={len(vocab)} | "
          f"shaped={hyper.use_shaped_reward} window={hyper.window or 'all'} | "
          f"adv_norm={hyper.adv_norm} term_upsample={hyper.terminal_upsample} "
          f"mc_aux={hyper.mc_aux_weight}")

    if card_table:
        print(f"[iql-gameplay] card attributes ON (CC table, dim={attr_dim})")
    else:
        try:
            from .card_attrs import CardAttributes, build_attr_matrix
            cube_path = Path(__file__).resolve().parents[2] / "OMN_Draft_3.5.txt"
            if cube_path.is_file():
                attr_matrix = build_attr_matrix(vocab, CardAttributes.from_cube(cube_path))
                attr_dim = int(attr_matrix.shape[1])
                print(f"[iql-gameplay] card attributes ON (OMN cube, dim={attr_dim})")
        except Exception as e:  # noqa: BLE001
            print(f"[iql-gameplay] card attributes unavailable: {e!r}")

    dev = torch.device(device)
    t = {k: torch.as_tensor(v).to(dev) for k, v in data.items()}
    net = build_net(len(vocab), hyper, attr_matrix=attr_matrix, attr_dim=attr_dim).to(dev)
    # WARM-START: seed the non-card pathways (scalar/tok-state projections,
    # role embedding, transformer encoder, V/Q/Pi heads) from a source policy so
    # the model INHERITS a mature scalar tempo policy. We ALSO inherit `emb` and
    # `attr_proj` (the card knowledge) WHENEVER they're shape-compatible — i.e. a
    # same-vocab/same-attr_dim CC->CC warm-start, so card embeddings COMPOUND across
    # iterations instead of being relearned from scratch each time. The shape check
    # below auto-skips them on a genuine change (OMN cube -> CC table: different
    # vocab/attr_dim), which is the only case where they legitimately differ. Only
    # the fixed buffers are force-skipped (attr_table is the destination's own card
    # table; state_roles is deterministic).
    if warm_start:
        src = torch.load(warm_start, map_location="cpu", weights_only=False)["net_state"]
        dst = net.state_dict()
        skip = ("attr_table", "state_roles")
        copied = mismatch = 0
        copied_emb = False
        for k, vv in src.items():
            if any(k.startswith(s) for s in skip):
                continue
            if k in dst and dst[k].shape == vv.shape:
                dst[k] = vv; copied += 1
                if k.startswith("emb.") or k.startswith("attr_proj."):
                    copied_emb = True
            else:
                mismatch += 1
        net.load_state_dict(dst)
        print(f"[iql-gameplay] warm-start from {warm_start}: copied {copied} tensors"
              f"{f', SKIPPED {mismatch} shape-mismatched' if mismatch else ''} "
              f"({'INHERITED card emb+attr_proj' if copied_emb else 'emb+attr_proj left fresh (vocab/attr_dim changed)'})")
    targ = build_net(len(vocab), hyper, attr_dim=attr_dim).to(dev)
    targ.load_state_dict(net.state_dict())
    opt = torch.optim.Adam(net.parameters(), lr=hyper.lr)
    tau, beta, gamma = hyper.expectile_tau, hyper.advantage_beta, hyper.gamma

    def expectile(diff):
        w = torch.where(diff > 0, tau, 1.0 - tau)
        return (w * diff.pow(2)).mean()

    bs = min(hyper.batch_size, n)
    last = 0.0
    # Terminal upsampling: the +-1 reward anchors are ~1/85 of transitions;
    # without oversampling them the critic collapses toward predicting 0.
    if hyper.terminal_upsample and hyper.terminal_upsample > 1.0:
        samp_w = 1.0 + (hyper.terminal_upsample - 1.0) * t["done"]
    else:
        samp_w = torch.ones(n, device=dev)
    for step in range(hyper.n_steps):
        idx = torch.multinomial(samp_w, bs, replacement=True)
        sv = net.state_vec(t["s_sc"][idx], t["s_ids"][idx], t["s_tst"][idx])
        av = net.action_vec(t["a_sc"][idx], t["a_card"][idx])
        with torch.no_grad():
            q_for_v = net.Q(sv.detach(), av.detach())
        v = net.V(sv)
        v_loss = expectile(q_for_v - v)
        if hyper.mc_aux_weight:
            v_loss = v_loss + hyper.mc_aux_weight * Fnn.mse_loss(v, t["mc_g"][idx])

        with torch.no_grad():
            sv2 = targ.state_vec(t["s2_sc"][idx], t["s_ids2"][idx], t["s_tst2"][idx])
            v_s2 = targ.V(sv2)
            q_target = t["rew"][idx] + gamma * (1.0 - t["done"][idx]) * t["sign"][idx] * v_s2
        q = net.Q(sv, av)
        q_loss = Fnn.mse_loss(q, q_target)

        with torch.no_grad():
            adv = (q_for_v - v).clamp(max=hyper.adv_clip)
            if hyper.adv_norm:
                # Batch z-score so beta operates on a unit scale even when the
                # critic's raw value spread is tiny (see IQLHyperparams.adv_norm).
                adv = (adv - adv.mean()) / (adv.std() + 1e-6)
            weight = torch.exp(beta * adv).clamp(max=hyper.adv_clip)
        B = bs
        sv_rep = sv.unsqueeze(1).expand(B, hyper.max_legal, net.sdim)
        lav = net.action_vec(t["lf_sc"][idx], t["lf_card"][idx])      # [B,ML,adim]
        logits = net.Pi(sv_rep, lav)                                  # [B,ML]
        logits = logits.masked_fill(t["lf_mask"][idx] < 0.5, float("-inf"))
        logp = Fnn.log_softmax(logits, dim=1)
        chosen_logp = logp.gather(1, t["chosen_idx"][idx].unsqueeze(1)).squeeze(1)
        pi_loss = -(weight * chosen_logp).mean()

        loss = v_loss + q_loss + pi_loss
        opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            for tp, p in zip(targ.parameters(), net.parameters()):
                tp.mul_(1 - hyper.target_tau).add_(hyper.target_tau * p)

        last = float(loss.item())
        if step % max(1, hyper.n_steps // 10) == 0 or step == hyper.n_steps - 1:
            print(f"  step {step:>6} loss={last:.4f} "
                  f"(v={float(v_loss):.4f} q={float(q_loss):.4f} pi={float(pi_loss):.4f})")

    ckpt = {
        "net_state": net.state_dict(),
        "vocab": vocab.to_dict(),
        "arch": "attn", "hidden": hyper.hidden, "emb_dim": hyper.emb_dim,
        "n_heads": hyper.n_heads, "n_layers": hyper.n_layers,
        "n_cards": len(vocab),
        # attr_dim lets inference size attr_proj correctly (CC=79 vs OMN cube
        # dim). Absent on legacy OMN checkpoints -> build_net defaults to the
        # OMN dim, so old checkpoints still load.
        "attr_dim": attr_dim,
        "hyper": asdict(hyper), "n_transitions": n, "trained_at": time.time(),
    }
    torch.save(ckpt, out_path)
    print(f"[iql-gameplay] saved -> {out_path} (final loss {last:.4f})")
    return out_path


def main(argv: list[str] | None = None) -> int:
    """Standalone IQL gameplay training. For the CC retrain, point --parquet-dir
    at the CC games and pass --card-table so the net gets CC card attributes:

        python -m python.training.iql_gameplay \\
            --parquet-dir datasets/cc/parquet/games \\
            --out-dir outputs/models/cc \\
            --card-table outputs/models/cc/cc_card_table.pt \\
            --steps 4000 --draw-penalty 0.3 --time-penalty 0.002
    """
    ap = argparse.ArgumentParser(description="Train the IQL gameplay policy.")
    ap.add_argument("--parquet-dir", required=True, help="dir of per-transition parquet")
    ap.add_argument("--out-dir", required=True, help="checkpoint output dir")
    ap.add_argument("--card-table", default="",
                    help="prebuilt CC vocab+attr table (build_cc_card_table); "
                         "omit for the OMN cube path")
    ap.add_argument("--warm-start", default="",
                    help="seed non-card weights (encoder/heads/scalar proj) from "
                         "this checkpoint, e.g. outputs/models/gameplay/latest.pt; "
                         "emb + attr_proj stay fresh. Inherits a mature tempo policy.")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--steps", type=int, default=IQLHyperparams.n_steps)
    ap.add_argument("--window", type=int, default=IQLHyperparams.window,
                    help="train on the most recent N parquet files (0=all)")
    ap.add_argument("--draw-penalty", type=float, default=IQLHyperparams.draw_penalty)
    ap.add_argument("--time-penalty", type=float, default=IQLHyperparams.time_penalty)
    ap.add_argument("--aggression-weight", type=float,
                    default=IQLHyperparams.aggression_weight)
    args = ap.parse_args(argv)

    hyper = IQLHyperparams(
        n_steps=args.steps, window=args.window,
        draw_penalty=args.draw_penalty, time_penalty=args.time_penalty,
        aggression_weight=args.aggression_weight,
    )
    train(parquet_dir=args.parquet_dir, out_dir=args.out_dir, hyper=hyper,
          device=args.device, card_table=args.card_table or None,
          warm_start=args.warm_start or None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
