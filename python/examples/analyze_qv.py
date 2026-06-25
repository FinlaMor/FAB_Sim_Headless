"""Diagnostic: inspect the trained IQL gameplay model's Q / V landscape.

Loads ``outputs/models/gameplay/latest.pt`` (the current champion), encodes
the most recent N games parquets with the CHECKPOINT's vocab (exactly like
deployment), and reports:

  * V(s) / Q(s,a) / advantage distributions
  * AWR weight saturation (how much of the policy gradient is clipped)
  * Bellman residual against the target the trainer regresses on
  * Monte-Carlo negamax return per transition vs V (calibration + correlation)
  * V separation: eventual winners vs losers vs draws
  * policy/Q agreement and policy entropy over legal actions

Run:  python -m python.examples.analyze_qv [--window 6] [--max-rows 40000]
"""

from __future__ import annotations

import argparse
import math
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve()
PROJECT_ROOT = HERE.parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from python.training import features as F  # noqa: E402
from python.training.iql_gameplay import (  # noqa: E402
    build_net, load_parquet_rows, IQLHyperparams, _terminal_reward,
)


def encode_with_vocab(rows, vocab, hyper: IQLHyperparams):
    """Replicates iql_gameplay._build but with a FIXED (checkpoint) vocab and
    keeps game_id / step_index / winner / mover for the MC-return analysis."""
    ML, AS = hyper.max_legal, F.ACTION_SCALAR_DIM

    last_step, winner_by = {}, {}
    for r in rows:
        gid = r.get("game_id")
        si = int(r.get("step_index", 0) or 0)
        if gid not in last_step or si > last_step[gid]:
            last_step[gid] = si
        winner_by[gid] = int(r.get("winner", 0) or 0)

    out = defaultdict(list)
    for r in rows:
        mover = int(r.get("player_to_move", 0) or 0)
        if mover not in (1, 2):
            continue
        state = F.loads(r["state_json"]); nstate = F.loads(r["next_state_json"])
        chosen = F.loads(r["chosen_action_json"])
        legals = F.loads(r["legal_actions_json"]) or []
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

        is_done = bool(r.get("done"))
        gid = r.get("game_id")
        is_draw_game = winner_by.get(gid, 1) == 0
        is_draw_term = (not is_done and bool(hyper.draw_penalty) and is_draw_game
                        and int(r.get("step_index", 0) or 0) == last_step.get(gid))
        if is_draw_term:
            step_r = -float(hyper.draw_penalty)
        else:
            step_r = _terminal_reward(r, hyper.use_shaped_reward)
            if hyper.aggression_weight:
                opp = 2 if mover == 1 else 1
                lead_s = F.player_health(state, mover) - F.player_health(state, opp)
                lead_n = F.player_health(nstate, mover) - F.player_health(nstate, opp)
                step_r += hyper.aggression_weight * (lead_n - lead_s) / 20.0
            if hyper.draw_step_cost and is_draw_game and not is_done:
                step_r -= float(hyper.draw_step_cost)
            if hyper.time_penalty and not is_done:
                step_r -= float(hyper.time_penalty)

        out["s_sc"].append(F.encode_state_scalars(state, mover))
        out["a_sc"].append(F.encode_action_scalars(chosen))
        out["s2_sc"].append(F.encode_state_scalars(nstate, mover))
        out["s_ids"].append(F.state_tokens(state, mover, vocab))
        out["s_ids2"].append(F.state_tokens(nstate, mover, vocab))
        out["s_tst"].append(F.state_token_state(state, mover))
        out["s_tst2"].append(F.state_token_state(nstate, mover))
        out["a_card"].append(F.action_card_id(chosen, vocab))
        out["rew"].append(step_r)
        out["done"].append(1.0 if (is_done or is_draw_term) else 0.0)
        out["sign"].append(1.0 if nm == mover else -1.0)
        out["lf_sc"].append(lfs); out["lf_card"].append(lfc)
        out["lf_mask"].append(lm); out["chosen_idx"].append(ci)
        out["n_legal"].append(int(sum(lm)))
        out["game_id"].append(gid)
        out["step_index"].append(int(r.get("step_index", 0) or 0))
        out["mover"].append(mover)
        out["winner"].append(winner_by.get(gid, 0))

    data = {}
    for k, v in out.items():
        if k == "game_id":
            data[k] = v
        elif k in ("s_ids", "s_ids2", "a_card", "lf_card", "chosen_idx",
                   "n_legal", "step_index", "mover", "winner"):
            data[k] = np.asarray(v, dtype="int64")
        else:
            data[k] = np.asarray(v, dtype="float32")
    return data


def mc_returns(data, gamma: float) -> np.ndarray:
    """Negamax MC return over RECORDED decisions: G_t = r_t + gamma*sign_t*G_{t+1}."""
    n = len(data["rew"])
    order = defaultdict(list)
    for i in range(n):
        order[data["game_id"][i]].append(i)
    G = np.zeros(n, dtype="float64")
    for gid, idxs in order.items():
        idxs.sort(key=lambda i: data["step_index"][i])
        g_next = 0.0
        for i in reversed(idxs):
            if data["done"][i] >= 0.5:
                g = float(data["rew"][i])
            else:
                g = float(data["rew"][i]) + gamma * float(data["sign"][i]) * g_next
            G[i] = g
            g_next = g
    return G


def pct(x, ps=(1, 5, 25, 50, 75, 95, 99)):
    return " ".join(f"p{p}={np.percentile(x, p):+.3f}" for p in ps)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="outputs/models/gameplay/latest.pt")
    ap.add_argument("--parquet", default="outputs/parquet/games")
    ap.add_argument("--window", type=int, default=6)
    ap.add_argument("--max-rows", type=int, default=40000)
    ap.add_argument("--batch", type=int, default=512)
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    vocab = F.CardVocab.from_dict(ckpt["vocab"])
    hyper = IQLHyperparams(**{k: v for k, v in ckpt.get("hyper", {}).items()
                              if k in IQLHyperparams.__dataclass_fields__})
    net = build_net(ckpt["n_cards"], {
        "emb_dim": ckpt["emb_dim"], "hidden": ckpt["hidden"],
        "n_heads": ckpt.get("n_heads", 4), "n_layers": ckpt.get("n_layers", 2)})
    net.load_state_dict(ckpt["net_state"])
    net.eval()
    print(f"[ckpt] {args.ckpt} | vocab={len(vocab)} cards | "
          f"trained on n={ckpt.get('n_transitions')} | "
          f"gamma={hyper.gamma} tau={hyper.expectile_tau} beta={hyper.advantage_beta} "
          f"adv_clip={hyper.adv_clip}")
    print(f"[ckpt] reward cfg: shaped={hyper.use_shaped_reward} "
          f"aggr_w={hyper.aggression_weight} draw_pen={hyper.draw_penalty} "
          f"time_pen={hyper.time_penalty}")

    rows = load_parquet_rows(args.parquet, args.window)
    print(f"[data] {len(rows)} raw rows from last {args.window} parquets")
    data = encode_with_vocab(rows, vocab, hyper)
    n = len(data["rew"])
    if n > args.max_rows:
        # Keep whole games: sample game_ids until budget.
        keep_gids = []
        seen = set()
        for g in data["game_id"]:
            if g not in seen:
                seen.add(g); keep_gids.append(g)
        rng = np.random.default_rng(0)
        rng.shuffle(keep_gids)
        budget, chosen_g = 0, set()
        counts = defaultdict(int)
        for g in data["game_id"]:
            counts[g] += 1
        for g in keep_gids:
            if budget + counts[g] > args.max_rows:
                continue
            chosen_g.add(g); budget += counts[g]
        mask = np.array([g in chosen_g for g in data["game_id"]])
        for k in list(data):
            if k == "game_id":
                data[k] = [g for g, m in zip(data[k], mask) if m]
            else:
                data[k] = data[k][mask]
        n = int(mask.sum())
    print(f"[data] analyzing {n} decision transitions "
          f"({len(set(data['game_id']))} games)")

    # UNK rate: how much of the current data the checkpoint vocab can't name.
    ids = data["s_ids"]
    nonpad = ids != F.PAD
    unk_rate = float(((ids == F.UNK) & nonpad).sum()) / max(1, int(nonpad.sum()))
    print(f"[data] state-token UNK rate vs ckpt vocab: {100*unk_rate:.2f}% "
          f"(includes masked opp arsenal)")

    t = {k: torch.as_tensor(v) for k, v in data.items() if k != "game_id"}
    V = np.zeros(n); Q = np.zeros(n); V2 = np.zeros(n)
    Qmax = np.zeros(n); Qchosen_rank = np.zeros(n)
    ent = np.zeros(n); agree = np.zeros(n); pi_top = np.zeros(n)
    ML = hyper.max_legal
    with torch.no_grad():
        for lo in range(0, n, args.batch):
            hi = min(n, lo + args.batch)
            sl = slice(lo, hi)
            sv = net.state_vec(t["s_sc"][sl], t["s_ids"][sl], t["s_tst"][sl])
            av = net.action_vec(t["a_sc"][sl], t["a_card"][sl])
            V[sl] = net.V(sv).numpy()
            Q[sl] = net.Q(sv, av).numpy()
            sv2 = net.state_vec(t["s2_sc"][sl], t["s_ids2"][sl], t["s_tst2"][sl])
            V2[sl] = net.V(sv2).numpy()
            B = hi - lo
            sv_rep = sv.unsqueeze(1).expand(B, ML, net.sdim)
            lav = net.action_vec(t["lf_sc"][sl], t["lf_card"][sl])
            qs = net.Q(sv_rep.reshape(B * ML, -1).contiguous(),
                       lav.reshape(B * ML, -1)).reshape(B, ML)
            logits = net.Pi(sv_rep, lav)
            m = t["lf_mask"][sl] < 0.5
            qs = qs.masked_fill(m, float("-inf"))
            logits = logits.masked_fill(m, float("-inf"))
            Qmax[sl] = qs.max(dim=1).values.numpy()
            ci = t["chosen_idx"][sl]
            q_ci = qs.gather(1, ci.unsqueeze(1)).squeeze(1)
            Qchosen_rank[sl] = (qs > q_ci.unsqueeze(1)).sum(dim=1).numpy()
            p = torch.softmax(logits, dim=1)
            lp = torch.log(p.clamp(min=1e-12))
            ent[sl] = (-(p * lp).sum(dim=1)).numpy()
            agree[sl] = (logits.argmax(dim=1) == qs.argmax(dim=1)).float().numpy()
            pi_top[sl] = (logits.argmax(dim=1) == ci).float().numpy()

    adv = Q - V
    beta = hyper.advantage_beta
    sat_thresh = math.log(hyper.adv_clip) / beta
    # Mirror the trainer: with adv_norm the exp() operates on z-scored
    # advantages (batch-normalized there; dataset-normalized here, close
    # enough for reporting). Without this the report showed the OLD raw-adv
    # weights and wrongly implied uniform weighting.
    adv_w = adv
    if getattr(hyper, "adv_norm", False):
        adv_w = (adv - adv.mean()) / (adv.std() + 1e-6)
    w = np.exp(np.clip(beta * np.minimum(adv_w, hyper.adv_clip), None,
                       math.log(hyper.adv_clip)))
    G = mc_returns(data, hyper.gamma)
    bell_target = data["rew"] + hyper.gamma * (1 - data["done"]) * data["sign"] * V2
    bell_res = Q - bell_target

    print("\n================ Q / V REPORT ================")
    print(f"V(s)        : mean={V.mean():+.3f} std={V.std():.3f} | {pct(V)}")
    print(f"Q(s,chosen) : mean={Q.mean():+.3f} std={Q.std():.3f} | {pct(Q)}")
    print(f"Q(s,best)   : mean={Qmax.mean():+.3f} std={Qmax.std():.3f}")
    print(f"adv=Q-V     : mean={adv.mean():+.3f} std={adv.std():.3f} | {pct(adv)}")
    print(f"AWR weight  : mean={w.mean():.2f} "
          f"(adv_norm={getattr(hyper, 'adv_norm', False)}) | "
          f"saturated(z>{sat_thresh:.3f}): {100*float((adv_w > sat_thresh).mean()):.1f}% | "
          f"weight<0.05 (ignored): {100*float((w < 0.05).mean()):.1f}%")
    print(f"Bellman res : mean={bell_res.mean():+.4f} rmse={np.sqrt((bell_res**2).mean()):.4f}")
    print(f"MC return G : mean={G.mean():+.3f} std={G.std():.3f} | {pct(G)}")
    for nm, x in (("V", V), ("Q", Q)):
        c = np.corrcoef(x, G)[0, 1]
        print(f"corr({nm}, G) = {c:+.3f}")

    # Calibration: V deciles vs realized G.
    print("\nV-decile calibration (mean V vs mean realized G):")
    qs_ = np.quantile(V, np.linspace(0, 1, 11))
    for d in range(10):
        m = (V >= qs_[d]) & (V <= qs_[d + 1] if d == 9 else V < qs_[d + 1])
        if m.sum() == 0:
            continue
        print(f"  d{d}: n={int(m.sum()):>6}  V={V[m].mean():+.3f}  G={G[m].mean():+.3f}")

    # Outcome separation.
    win = data["winner"] == data["mover"]
    loss = (data["winner"] > 0) & ~win
    draw = data["winner"] == 0
    print("\nOutcome separation (all decisions of eventual ...):")
    for nm, m in (("winner", win), ("loser", loss), ("draw", draw)):
        if m.sum():
            print(f"  {nm:>6}: n={int(m.sum()):>6}  V={V[m].mean():+.3f}  "
                  f"Q={Q[m].mean():+.3f}  G={G[m].mean():+.3f}")

    # Early/late split for winners vs losers (does V know early?).
    si = data["step_index"].astype("float64")
    half = np.zeros(n, dtype=bool)
    for gid in set(data["game_id"]):
        m = np.array([g == gid for g in data["game_id"]])
        med = np.median(si[m])
        half |= m & (si <= med)
    print("\nV by game half (winner vs loser decisions):")
    for half_name, hm in (("1st half", half), ("2nd half", ~half)):
        for nm, m in (("winner", win), ("loser", loss)):
            mm = m & hm
            if mm.sum():
                print(f"  {half_name} {nm:>6}: V={V[mm].mean():+.3f} (n={int(mm.sum())})")

    print("\nPolicy head:")
    print(f"  entropy: mean={ent.mean():.3f} nats (uniform over k legals would be ln k; "
          f"mean ln(n_legal)={np.log(data['n_legal'].clip(1)).mean():.3f})")
    print(f"  argmax(pi)==argmax(Q): {100*agree.mean():.1f}%")
    print(f"  argmax(pi)==logged action: {100*pi_top.mean():.1f}%")
    print(f"  chosen action's Q-rank: mean={Qchosen_rank.mean():.2f} "
          f"(0 = chosen had highest Q) | top-1 {100*float((Qchosen_rank==0).mean()):.1f}%")

    # Terminal sanity: Q on terminal transitions should ~equal the terminal reward.
    term = data["done"] >= 0.5
    if term.sum():
        tr = data["rew"][term]
        tq = Q[term]
        print(f"\nTerminals: n={int(term.sum())} | reward mean={tr.mean():+.3f} | "
              f"Q mean={tq.mean():+.3f} | mae={np.abs(tq-tr).mean():.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
