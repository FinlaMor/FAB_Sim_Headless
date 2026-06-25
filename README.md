# FAB_Sim_Headless — Limited Testing Laboratory

A **complete offline Flesh and Blood limited-testing pipeline** built on
top of the real [Talishar](https://github.com/Talishar/Talishar) PHP
engine.

```
Draft Pod -> Build Decks -> Tournament Bracket -> Record Replays
                                                       -> Train Bots
                                                       -> Repeat
```

The Python side **never reimplements game rules**. Talishar is the only
component that mutates in-game state — Python observes legal actions
and chooses; PHP applies the choice via the same `ProcessInput()` the
production frontend uses. Drafting and deck construction are pure
Python (Talishar isn't a deckbuilder), but card metadata is loaded from
Talishar's own `CardDictionaries/` and the resulting deck files are fed
straight back to `TalisharBoot::createGame` so legality is enforced at
game start.

## Project layout

```
FAB_Sim_Headless/
├── talishar/                       upstream clone (rules engine)
├── adapter/                        PHP 8.1 HTTP adapter (Docker)
│   ├── api.php / GameAdapter.php / bootstrap.php
│   ├── routes/ {new_game, state, actions, step, reset, health}.php
│   ├── serializers/ {State,Action}Serializer.php
│   ├── lib/ {TalisharBoot, StubGame, CacheStub, RngHook, GameRegistry}.php
│   ├── Dockerfile (PHP 8.1 + shmop + redis + pdo_mysql + opcache JIT)
│   └── TALISHAR_NOTES.md           recon: entry points, mode codes, state shape
├── python/
│   ├── draft/                      8-player pod simulator + draft bots
│   │   ├── format.py               LEGAL_HEROES / LEGAL_WEAPONS / HERO_WEAPONS
│   │   ├── pack_loader.py          packs.json reader, deterministic seat assignment
│   │   ├── simulator.py            full pod with LRL pass rotation
│   │   ├── dataset.py              DraftDatasetWriter (parquet/msgpack/npz)
│   │   └── bots/ {random, heuristic, transformer}.py
│   ├── deckbuilding/               pool -> Deck builder + legality
│   │   ├── card_catalog.py         parses talishar/CardDictionaries (or shim)
│   │   ├── legality.py             min-size + hero+weapon match check
│   │   ├── deck.py                 Deck dataclass + to_talishar_dict()
│   │   └── builder.py              Random + Heuristic + Transformer scaffolds
│   ├── tournament/                 single-elim 8-player bracket
│   │   ├── bracket.py              QF A-E / C-G / B-F / D-H, SF, Final
│   │   ├── player.py / match.py    run_match drives TalisharEnv per game
│   │   └── runner.py               full bracket + placement logic + Bo1/BoN
│   ├── gameplay/                   (formerly top-level) Gym env + replay
│   │   ├── env.py                  TalisharEnv HTTP client
│   │   ├── selfplay.py / replay_buffer.py / dataset_writer.py
│   │   └── bots/ {random, heuristic, transformer}.py
│   ├── training/                   IQL + BC scaffolds (torch-optional)
│   │   └── iql_draft.py, iql_gameplay.py, supervised_imitation.py
│   ├── datasets/                   parquet reader (drafts/decks/games/tournaments)
│   ├── models/                     versioned weights registry
│   ├── analytics/                  hero/archetype WR, seat EV, pick EV, matchup matrix, signals
│   ├── pipeline.py                 LimitedPipeline orchestrator (Draft -> Tournament -> Record)
│   └── examples/ {smoke_test, limited_pipeline_smoke, add_a_bot.md}
├── decks/sample_packs/oma_sample.json   40-pack OMA synthetic for smoke tests
├── outputs/                        parquet/{drafts,decks,games,tournaments}/
└── docker-compose.yml
```

---

## Quick start (3 commands, no PHP required)

```powershell
cd C:\Users\Joseph\Desktop\FAB_Sim_Headless
python -m pip install -r python\requirements.txt

# 1. Gameplay-only smoke (RandomBot vs HeuristicBot, ~50 turns)
python -m python.examples.smoke_test

# 2. Full limited pipeline against synthetic OMA packs
python -m python.examples.limited_pipeline_smoke

# 3. Full limited pipeline against the REAL OMN cube (Draftmancer format)
python -m python.examples.omn_pipeline_smoke
```

Both tests spin up an in-process Python HTTP stub matching the adapter
wire protocol — no Docker, no PHP. The "real Talishar" path swaps the
stub for the bundled adapter container.

## Switch to real Talishar (Docker)

```powershell
$env:ADAPTER_MODE = "real"
docker compose up --build adapter            # health at http://localhost:8000/health
```

Then point the Python pipeline at it:

```powershell
python - <<PY
from python.pipeline import LimitedPipeline, PipelineConfig
res = LimitedPipeline(PipelineConfig(
    adapter_url="http://localhost:8000",
    packs_path="decks/sample_packs/oma_sample.json",
    out_dir="outputs",
    seed=42, n_pods=4, best_of=3,
)).run_cycle()
print(res.cycle_id, "tournaments:", len(res.tournaments))
PY
```

---

## The format: Omens of the Third Age

Heroes / weapons are declared in `python/draft/format.py`:

| Hero                                | Signature weapon       |
| ----------------------------------- | ---------------------- |
| `zyggy`                             | `aphrodias`            |
| `aurora_emissary_of_lightning`      | `scorpio_comet_tails`  |
| `oscilio_scion_of_the_third_age`    | `volzar_meteor_storm`  |

Patching in a new legal hero/weapon only requires editing `format.py` —
the simulator, deck builder, legality module, and analytics all source
from those constants.

## Bracket (8 players, single elimination)

```
QF: A vs E   C vs G   B vs F   D vs H
SF: W(QF1) vs W(QF2)   W(QF3) vs W(QF4)
F : W(SF1) vs W(SF2)
```

Matches drive through the existing `TalisharEnv`. Bo1 by default;
`TournamentRunner(..., best_of=3)` plays Bo3 instead.

## Parquet schemas

| Artefact      | Path                                     | Row granularity                                 |
| ------------- | ---------------------------------------- | ----------------------------------------------- |
| drafts        | `outputs/parquet/drafts/*.parquet`       | 1 per pick (with placement column once known)   |
| decks         | `outputs/parquet/decks/*.parquet`        | 1 per (pod, seat) deck                          |
| games         | `outputs/parquet/games/*.parquet`        | 1 per transition (full state JSON + mask)       |
| tournaments   | `outputs/parquet/tournaments/*.parquet`  | 1 per (tournament, player) with placement       |

`python.datasets.DatasetReader(out, artefact="drafts").load_pandas()`
returns a ready-to-train DataFrame.

## Bots

Three layers, each with the same RandomBot / HeuristicBot / Transformer
scaffold pattern:

* **Draft bots** — `python/draft/bots/`. Receive a `DraftPodView`
  snapshot per pick with the seat's drafted pool + neighbour signals.
* **Deck builders** — `python/deckbuilding/builder.py`.
  `HeuristicDeckBuilder` targets a 12/9/9 red/yellow/blue pitch curve
  and locks the signature weapon to the hero.
* **Gameplay bots** — `python/gameplay/bots/`. Drive matches via
  `TalisharEnv.step`.

Each transformer scaffold lazy-imports torch so a worker that only uses
random/heuristic baselines stays small.

## Determinism

Reproducible from `(seed, packs_path)` alone:

* `sample_player_packs(seed=…)` for pack-to-seat assignment.
* Per-pick RNG seeded inside each `DraftBot.reset(seed=…)`.
* Per-match game seed = stable mix of `(tournament_seed, match_id, game_idx)`.
* Adapter PHP `RngHook::seed($seed, $step_counter)` before every step.

Both smoke tests assert determinism (same seed → same winner & step count).

## Analytics

Each analytic returns a pandas DataFrame the caller can plot or pivot:

* `python.analytics.hero_winrate.compute(out)`
* `python.analytics.archetype_winrate.compute(out)`
* `python.analytics.seat_ev.compute(out)`
* `python.analytics.pick_order_ev.compute(out)`
* `python.analytics.matchup_matrix.compute(out)`
* `python.analytics.signal_detection.compute(out)`

The `limited_pipeline_smoke` test exercises three of them so you can
see the expected output shape immediately.

## Iterative training loop

`python/training/` ships scaffolds for:

* **IQL gameplay** — implicit Q-learning over the games parquet schema
  (terminal-only reward by default, shaped reward column supported).
* **IQL draft** — applies IQL with sparse placement-derived reward
  back-propagated to each pick.
* **Supervised imitation (BC)** — warm-start for both heads.

After training, register weights via `python.models.save_checkpoint(...)`
and the corresponding `TransformerXxxBot` will load them at next run.

The autopilot loop is:

```
1. Run cycle (pipeline.run_cycle)        -> parquet
2. Train IQL                             -> weights
3. Replace weak bots                     -> next cycle uses learned policy
4. Repeat
```

⚠️ The training step is currently a **typed scaffold** — calling
`train()` raises `NotImplementedError` to make sure nobody assumes
training has happened when it hasn't. Fill in the IQL update loop in
`python/training/iql_*.py` once you have enough trajectories.

## Parallelism

For high-throughput cycles, run N adapter containers (one per port) and
N pipeline processes (one per adapter). Per project memory: **cap at 32
workers on this host** — anything higher crashes MySQL via the existing
FAB_Sim Docker stack.

```powershell
for ($i = 0; $i -lt 8; $i++) {
    $port = 8000 + $i
    docker run --rm -d --name adapter$i -p ${port}:8000 `
        -e ADAPTER_MODE=real fab-sim-headless-adapter:dev
}
1..8 | ForEach-Object -Parallel {
    python -m python.pipeline `
        --adapter "http://localhost:$((7999 + $_))" `
        --packs decks/sample_packs/oma_sample.json `
        --seed (1000 * $_) --pods 5 --out "outputs/worker$_"
} -ThrottleLimit 8
```

## Cube formats

The pipeline accepts two pack-file shapes:

| Extension                           | Loader                                | Notes                                                            |
| ----------------------------------- | ------------------------------------- | ---------------------------------------------------------------- |
| `.json`                             | `python.draft.pack_loader`            | Simple `[{pack_id, cards: [...]}]` list. Used by the synthetic OMA sample. |
| `.txt` / `.draft` / `.draftmancer`  | `python.draft.draftmancer`            | Real Draftmancer cube spec ([Settings] + [CustomCards] + [Layouts] + named sections). Used by `OMN_Draft_3.5.txt`.    |

The OMN cube ships **no heroes / weapons in the packs**, so the
pipeline auto-assigns one per seat via
`pipeline.default_hero_assignment` — a three-layer cascade:

1. **Draft-bot preference** (`bot.pick_hero(...)` if defined).
   `HeuristicDraftBot` returns the hero matching the dominant class
   in its drafted pool; `RandomDraftBot` returns a uniform choice;
   the base default is `None` (defer to the next layer).
2. **Drafted-class plurality.** Count the seat's drafted cards by
   decisive class (Illusionist / Wizard / Runeblade) using the
   `card_classes` map parsed from the cube's `CustomCards`. If
   exactly one class leads, the corresponding hero is chosen.
3. **Deterministic random** seeded from `(pod.seed, seat)`.

`pipeline.round_robin_hero_assignment` is kept as a named alternative
for ablation runs. Set `PipelineConfig.hero_assignment = None` to
require heroes/weapons in the booster instead (the legacy
synthetic-OMA flow).

## Constraints (what is not implemented)

* **Synthetic OMA pool is for tests only** —
  `decks/sample_packs/oma_sample.json` exists purely to exercise the
  hero-in-pack code path. Real drafts should use the OMN cube
  (`OMN_Draft_3.5.txt`).
* **`TransformerBot` weights are random** — the scaffold compiles a
  6-layer transformer but ships no trained weights. The deck builder
  variant currently delegates to `HeuristicDeckBuilder` so end-to-end
  tests still produce legal decks.
* **Training scaffolds raise NotImplementedError** — see above.
* **Talishar real-mode** is wired but not yet exercised on this host
  (PHP isn't installed). Run `docker compose up --build adapter` with
  `ADAPTER_MODE=real` to verify.

See `adapter/TALISHAR_NOTES.md` for the engine-side recon details.

## Smoke tests

```powershell
python -m python.examples.smoke_test                 # gameplay env
python -m python.examples.limited_pipeline_smoke     # full draft -> bracket (synthetic OMA)
python -m python.examples.omn_pipeline_smoke         # full draft -> bracket (real OMN cube)
```

All three must finish with `ALL CHECKS PASSED` before any change ships.
