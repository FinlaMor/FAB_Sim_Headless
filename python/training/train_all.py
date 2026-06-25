"""Train both offline-IQL policies from a pipeline cycle's parquet output.

    python -m python.training.train_all --out outputs

Reads:
    <out>/parquet/games/*.parquet   -> gameplay IQL  -> models/gameplay/<v>/
    <out>/parquet/drafts/*.parquet  -> draft IQL     -> models/draft/<v>/

Both checkpoints are also copied to ``<out>/models/<role>/latest.pt`` for
easy pickup by the bots.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve()
PROJECT_ROOT = HERE.parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from python.training import iql_gameplay, iql_draft  # noqa: E402
from python.models import registry  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--gameplay-steps", type=int, default=4000)
    ap.add_argument("--draft-steps", type=int, default=3000)
    ap.add_argument("--window", type=int, default=0,
                    help="Train on the most recent N parquet files (cycles); 0 = all.")
    ap.add_argument("--no-shaped-reward", action="store_true",
                    help="Use flat +/-1 terminal reward instead of the margin-aware shaped reward.")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    out = Path(args.out)
    games_dir = out / "parquet" / "games"
    drafts_dir = out / "parquet" / "drafts"
    staging = out / "models" / "_staging"
    staging.mkdir(parents=True, exist_ok=True)

    results: dict[str, str] = {}

    # ---- gameplay ----
    if list(games_dir.glob("*.parquet")):
        print("\n=== Training gameplay IQL ===")
        hp = iql_gameplay.IQLHyperparams(
            n_steps=args.gameplay_steps, window=args.window,
            use_shaped_reward=not args.no_shaped_reward)
        ckpt = iql_gameplay.train(parquet_dir=games_dir, out_dir=staging, hyper=hp, device=args.device)
        dest = registry.save_checkpoint(ckpt, root=out, role="gameplay")
        shutil.copy2(ckpt, out / "models" / "gameplay" / "latest.pt")
        results["gameplay"] = str(dest)
    else:
        print(f"[train_all] no gameplay parquet under {games_dir}; skipping")

    # ---- draft ----
    if list(drafts_dir.glob("*.parquet")):
        print("\n=== Training draft IQL ===")
        hp = iql_draft.DraftIQLHyperparams(n_steps=args.draft_steps, window=args.window)
        ckpt = iql_draft.train(parquet_dir=drafts_dir, out_dir=staging, hyper=hp, device=args.device)
        dest = registry.save_checkpoint(ckpt, root=out, role="draft")
        shutil.copy2(ckpt, out / "models" / "draft" / "latest.pt")
        results["draft"] = str(dest)
    else:
        print(f"[train_all] no draft parquet under {drafts_dir}; skipping")

    print("\n=== Training complete ===")
    for role, path in results.items():
        print(f"  {role}: {path}")
    print(f"  gameplay latest: {out/'models'/'gameplay'/'latest.pt'}")
    print(f"  draft latest:    {out/'models'/'draft'/'latest.pt'}")
    return 0 if results else 1


if __name__ == "__main__":
    raise SystemExit(main())
