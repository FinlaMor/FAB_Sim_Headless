"""Live CC self-play dashboard — leave it running to watch data collection.

Reads the per-matchup status the orchestrator writes and refreshes in place:
  * outputs/cc_selfplay.jsonl  — one row per completed matchup (run_id + ts)
  * outputs/cc_draws.log       — wedge/stall headlines

Headlines the CURRENT run (segmented by the per-process run_id stamp, with an
inter-row time-gap fallback for legacy rows) and keeps an all-time line for
context. No dependencies beyond the stdlib, and it only READS — always safe to
run or kill alongside self-play.

Run (from project root):
    python -m python.examples.pipeline_monitor              # refresh every 30s
    python -m python.examples.pipeline_monitor --interval 60
    python -m python.examples.pipeline_monitor --once       # print once, exit
    python -m python.examples.pipeline_monitor --run-gap-min 20

NOTE: the OMN / gameplay-training section (iteration gate/lethal/draw table) is
omitted for now; parse_current / gate_str / fmt_dur remain in the module so it
can be restored.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve()
PROJECT_ROOT = HERE.parents[2]

_BANNER = re.compile(r"CONTINUOUS ITER (\d+)")
_RR = re.compile(r"\] .+ vs .+: \d+-\d+")
_STEP = re.compile(r"^\s*step\s+(\d+)\s+loss=")
_OK = re.compile(r"\[continuous\] iter (\d+) OK")


def tail_text(path: Path, nbytes: int = 300_000) -> str:
    try:
        size = path.stat().st_size
        with open(path, "rb") as fh:
            if size > nbytes:
                fh.seek(size - nbytes)
            return fh.read().decode("utf-8", errors="replace")
    except OSError:
        return ""


def read_metrics(path: Path) -> list[dict]:
    rows = []
    try:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    except OSError:
        pass
    return rows


def fmt_dur(s) -> str:
    try:
        s = int(s)
    except (TypeError, ValueError):
        return "?"
    h, rem = divmod(s, 3600)
    m, _ = divmod(rem, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m"


def parse_current(log: str) -> dict:
    """Infer the in-progress iteration + phase from the log tail."""
    lines = log.splitlines()
    last_banner = -1
    cur_iter = None
    for i, ln in enumerate(lines):
        m = _BANNER.search(ln)
        if m:
            last_banner = i
            cur_iter = int(m.group(1))
    if cur_iter is None:
        return {"iter": None, "phase": "no banner found"}
    sl = lines[last_banner:]
    text = "\n".join(sl)

    # If this iteration already printed its OK line, the loop is between iters.
    if _OK.search(text):
        return {"iter": cur_iter, "phase": "between iterations", "done": True}

    pairs = sum(1 for ln in sl if _RR.search(ln))
    last_step = None
    for ln in sl:
        m = _STEP.search(ln)
        if m:
            last_step = int(m.group(1))

    has_gp = "[iql-gameplay]" in text
    gp_saved = "[iql-gameplay] saved" in text
    # The loop still trains a draft model after the gameplay gate, but we don't
    # report on it — its log markers only tell us the iteration is wrapping up.
    post_gp = "[iql-draft]" in text

    if post_gp:
        phase, detail = "finalizing", "writing metrics"
    elif gp_saved:
        phase, detail = "gate-gp", "candidate vs champion (200 games)"
    elif has_gp:
        phase, detail = "train-gp", f"step {last_step}/12000" if last_step is not None else "starting"
    elif pairs:
        phase, detail = "generating", f"{pairs} round-robin pairs done"
    else:
        phase, detail = "starting", ""
    return {"iter": cur_iter, "phase": phase, "detail": detail, "pairs": pairs}


def gate_str(g: dict | None) -> str:
    if not g:
        return "-"
    if g.get("cold_start"):
        return "cold-start"
    cw, hw, dr = g.get("cand_wins", 0), g.get("champ_wins", 0), g.get("draws", 0)
    tag = "PROMOTED" if g.get("promoted") else "kept"
    return f"{tag} {cw}-{hw} (d{dr})"


def _short(h: str) -> str:
    """Readable hero name: first word, e.g. gravy_bones_... -> 'gravy'."""
    return (h or "?").split("_")[0][:13]


def _read_draws(path: Path) -> list[dict]:
    """Parse the DRAW headlines in outputs/cc_draws.log into structured rows."""
    rows: list[dict] = []
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if not line.startswith("DRAW "):
                    continue
                m = re.match(r"DRAW (\S+) vs (\S+) .*?\| "
                             r"(STUCK[^|]*|life moving) \| (abort=\S+|hit step_cap)", line)
                if not m:
                    continue
                rows.append({
                    "a": m.group(1), "b": m.group(2),
                    "wedge": m.group(3).startswith("STUCK"),
                    "why": m.group(4).replace("abort=", "").replace("_", " ").strip(),
                    "gravy": "GRAVY" in line,
                })
    except OSError:
        pass
    return rows


def _agg(recs: list[dict]) -> dict:
    """Aggregate a list of cc_selfplay status rows into totals."""
    games = matchups = decisive = draws = errors = 0
    heroes: set = set()
    model = False
    for r in recs:
        if r.get("error"):
            errors += 1
            continue
        games += int(r.get("completed", 0))
        matchups += 1
        model = model or bool(r.get("model"))
        decisive += int(r.get("winA", 0)) + int(r.get("winB", 0))
        draws += int(r.get("draws", 0))
        heroes.update([r.get("heroA"), r.get("heroB")])
    return {"games": games, "matchups": matchups, "decisive": decisive,
            "draws": draws, "errors": errors, "heroes": heroes, "model": model}


def _pct(part: int, whole: int) -> str:
    return f"{round(100 * part / whole)}%" if whole else "-"


def _split_runs(recs: list[dict], gap_s: float) -> list[list[dict]]:
    """Segment append-ordered status rows into runs. The last group is the
    current (active or most-recent) run.

    Prefers the `run_id` stamp (one per cc_selfplay process) so overlapping
    runs separate cleanly; falls back to an inter-row time gap for legacy rows
    written before the stamp existed.
    """
    # Stamped rows: collapse by run_id (concurrent runs interleave their rows in
    # the file, so grouping — not contiguity — is what keeps each run whole).
    stamped: dict[str, list[dict]] = {}
    legacy: list[dict] = []
    for r in recs:
        rid = r.get("run_id")
        if rid is None:
            legacy.append(r)
        else:
            stamped.setdefault(rid, []).append(r)

    groups: list[list[dict]] = list(stamped.values())

    # Legacy rows (pre-stamp): fall back to inter-row time-gap segmentation.
    cur: list[dict] = []
    prev_ts = None
    for r in legacy:
        ts = r.get("ts")
        if cur and prev_ts is not None and ts is not None and ts - prev_ts > gap_s:
            groups.append(cur)
            cur = []
        cur.append(r)
        prev_ts = ts
    if cur:
        groups.append(cur)

    # Order by most-recent activity so groups[-1] is the current run.
    groups.sort(key=lambda g: max((r.get("ts") or 0) for r in g))
    return groups


def cc_block(cc_path: Path, draws_path: Path, now: float, n_heroes: int = 38,
             run_gap_s: float = 1800.0) -> list[str]:
    """Clean, plain-language CC self-play dashboard. Headlines the CURRENT run
    (segmented from cc_selfplay.jsonl by time gap) and keeps an all-time line
    for context. Wedge detail comes from the cc_draws.log tail."""
    title = " " + "=" * 26 + " CC SELF-PLAY " + "=" * 26
    recs = read_metrics(cc_path)
    out = ["", title]
    if not recs:
        out += ["   No CC self-play games recorded yet.",
                "   Start one:  python -m python.gameplay.cc_selfplay --adapters 8000-8007 --pairs 48",
                " " + "=" * (len(title) - 1)]
        return out

    try:
        age = now - cc_path.stat().st_mtime
        status = (f"RUNNING   (last game finished {int(age)}s ago)" if age < 150
                  else f"idle      ({int(age // 60)} min since the last game)")
    except OSError:
        status = "unknown"

    runs = _split_runs(recs, run_gap_s)
    cur = runs[-1]
    c = _agg(cur)
    a = _agg(recs)
    start_ts = next((r.get("ts") for r in cur if r.get("ts")), None)
    started = f"{int((now - start_ts) // 60)}m ago" if start_ts else "?"

    out.append(f"   status:           {status}")
    out.append(f"   sideboard model:  {'ON  (decks tuned to each opponent)' if c['model'] else 'off (default decks)'}")
    out.append("")
    out.append(f"   THIS RUN  (started {started},  {c['matchups']} matchups"
               + (f",  {c['errors']} setup errors" if c["errors"] else "") + ")")
    out.append(f"      games ...................  {c['games']}")
    out.append(f"      decisive (someone won) ..  {c['decisive']:<4} ({_pct(c['decisive'], c['games'])})")
    out.append(f"      draws ...................  {c['draws']}")
    out.append(f"      heroes seen ............  {len(c['heroes'])} of {n_heroes}")
    out.append("")
    out.append(f"   ALL TIME ({len(runs)} runs):  {a['games']} games   "
               f"decisive {a['decisive']} ({_pct(a['decisive'], a['games'])})   "
               f"draws {a['draws']}   {len(a['heroes'])} of {n_heroes} heroes")

    dr = _read_draws(draws_path)
    wedges = [d for d in dr if d["wedge"]]
    if wedges:
        culprit: Counter = Counter()
        for d in wedges:
            culprit[d["a"]] += 1
            culprit[d["b"]] += 1
        out.append("")
        out.append("   recent wedges / stalls (latest in the log):")
        for d in wedges[-6:]:
            flag = "[GRAVY] " if d["gravy"] else ""
            out.append(f"      {flag}{_short(d['a']):<13} vs {_short(d['b']):<13}  -> {d['why']}")
        out.append("   heroes wedging most (all time):  "
                   + ", ".join(f"{_short(h)} x{n}" for h, n in culprit.most_common(4)))
    out.append(" " + "=" * (len(title) - 1))
    return out


def render(metrics: list[dict], cur: dict, log_path: Path, tail_n: int,
           cc_path: Path | None = None, draws_path: Path | None = None,
           run_gap_s: float = 1800.0) -> str:
    now = time.time()
    out = []
    out.append("=" * 78)
    out.append(f" FAB MONITOR    {datetime.now():%Y-%m-%d %H:%M:%S}")
    out.append("=" * 78)

    # CC self-play only. The OMN / gameplay-training section (heartbeat, current
    # iteration, iteration table, promotions) is intentionally omitted for now —
    # parse_current / gate_str / fmt_dur are kept available to restore it later.
    if cc_path is not None:
        out += cc_block(cc_path, draws_path or cc_path.parent / "cc_draws.log", now,
                        run_gap_s=run_gap_s)

    out.append("=" * 78)
    out.append(" (read-only; refreshes automatically; Ctrl-C to quit)")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(PROJECT_ROOT / "outputs"))
    ap.add_argument("--log", default=str(PROJECT_ROOT / "continuous.log"))
    ap.add_argument("--interval", type=float, default=30.0)
    ap.add_argument("--tail", type=int, default=12, help="recent iterations to show")
    ap.add_argument("--cc-status", default=str(PROJECT_ROOT / "outputs" / "cc_selfplay.jsonl"),
                    help="CC self-play status jsonl (cc_selfplay writes this)")
    ap.add_argument("--cc-draws", default=str(PROJECT_ROOT / "outputs" / "cc_draws.log"),
                    help="CC wedge log (cc_selfplay writes this)")
    ap.add_argument("--run-gap-min", type=float, default=30.0,
                    help="gap (minutes) between status rows that marks a new self-play run "
                         "for the THIS RUN block")
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--no-clear", action="store_true")
    args = ap.parse_args()

    metrics_path = Path(args.out) / "continuous_metrics.jsonl"
    log_path = Path(args.log)
    cc_path = Path(args.cc_status)
    draws_path = Path(args.cc_draws)

    while True:
        metrics = read_metrics(metrics_path)
        cur = parse_current(tail_text(log_path))
        screen = render(metrics, cur, log_path, args.tail, cc_path, draws_path,
                        run_gap_s=args.run_gap_min * 60.0)
        if not args.no_clear and not args.once:
            print("\033[2J\033[H", end="")
        print(screen, flush=True)
        if args.once:
            return 0
        try:
            time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\nbye.")
            return 0


if __name__ == "__main__":
    raise SystemExit(main())
