# FAB_Sim_Headless — How To Run Everything (Operator's Guide)

*Last updated: 2026-06-24. This reflects the **current Classic Constructed (CC) pipeline**.
The old top-level `README.md` / `HANDOFF_NEXT_SESSION.md` describe the 2026-05-30 OMN-draft era
and are out of date — use this document.*

This is the practical, copy-paste guide to operating the project by yourself: start the engine,
collect data, train each model, monitor, and recover from problems. Everything runs from the
project root in **PowerShell**:

```powershell
cd C:\Users\Joseph\Desktop\FAB_Sim_Headless
$env:PYTHONPATH = "C:\Users\Joseph\Desktop\FAB_Sim_Headless"   # so `python -m python.*` resolves
```

For *why* each piece exists and how the ML fits together, read the companion doc
`docs/ML_ARCHITECTURE.md`.

---

## 0. The mental model (30 seconds)

```
Docker adapters (PHP Talishar engine)  ──HTTP──►  Python driver (bots choose actions)
        ports 8000–8007                              │
                                                     ├─ writes game transitions  → datasets/cc/parquet/games/*.parquet
                                                     └─ writes sideboard choices → outputs/cc_sideboard_matches.jsonl
                                                                  │
                                            train models from those files (offline, CPU)
                                                                  │
                              outputs/models/cc_warmN/   (gameplay IQL)
                              outputs/models/sideboard/  (sideboard BC + RL)
```

- **The engine never lives in Python.** Talishar (PHP, in Docker) owns all game rules. Python only
  observes legal actions and picks one.
- **Two things are produced by self-play:** per-move game transitions (to train the *gameplay* bot)
  and per-matchup sideboard choices + outcomes (to train the *sideboard* bot).
- **Training is offline and CPU-only.** It reads the files above; it does not need the engine running.

---

## 1. Prerequisites (one-time)

1. **Docker Desktop** running (Linux containers).
2. **Python 3.12** at `C:\Users\Joseph\AppData\Local\Programs\Python\Python312\python.exe`.
3. Python deps: `python -m pip install -r python\requirements.txt`
   (torch, numpy, pandas, pyarrow, requests).
4. The adapter image built once: it builds automatically the first time you `docker compose up --build`.

---

## 2. Start the engine (Docker adapters)

The collection driver expects **8 adapter workers on ports 8000–8007**.

```powershell
$env:ADAPTER_MODE = "real"
docker compose -f docker-compose.yml -f docker-compose.parallel.yml up -d `
  adapter adapter2 adapter3 adapter4 adapter5 adapter6 adapter7 adapter8
```

- `adapter` = port **8000**, `adapter2…8` = ports **8001–8007** (each container's internal port is 8000).
- The compose files also define `adapter9–16` (ports 8008–8015) if you ever want more workers, but
  **8 is the sweet spot on this 6-core box** — more doesn't help (see ML doc, "Throughput").

**Verify they're healthy:**

```powershell
foreach ($p in 8000..8007) {
  try { (Invoke-WebRequest "http://localhost:$p/health" -TimeoutSec 5 -UseBasicParsing).StatusCode }
  catch { "DOWN: $p" }
}
```

Each should print `200`. (A healthy response body looks like `{"ok":true,...,"mode":"real",...}`.)

**Stop the engine:**

```powershell
docker compose -f docker-compose.yml -f docker-compose.parallel.yml down
```

> ⚠️ **opcache rule.** The adapters cache compiled PHP with `opcache.validate_timestamps=0`, so they
> do **not** notice edits to anything under `talishar/` or `adapter/`. After any engine/card edit you
> must `docker restart` the adapters for it to take effect. A CLI `opcache_reset` won't work (it
> resets a different process's cache). See §8.

---

## 3. Collect self-play data

This is the core ongoing activity — it grows both the gameplay corpus and the Stage-2 sideboard
corpus at the same time.

### 3a. One batch (foreground or background)

```powershell
python -m python.gameplay.cc_selfplay `
  --adapters 8000-8007 `
  --pairs 200 --games 2 `
  --model outputs/models/sideboard/sideboard_bc.pt `
  --gameplay-model outputs/models/cc_warm4/iql_gameplay.pt `
  --explore-sideboard 0.7 `
  --step-cap 800 `
  --base-seed 300000
```

What the flags mean:

| Flag | Meaning |
|---|---|
| `--adapters 8000-8007` | which adapter ports to shard games across |
| `--pairs N` | number of hero matchups this batch (0 = all) |
| `--games K` | games per matchup |
| `--gameplay-model PATH` | the IQL gameplay bot that actually plays (current champion = `cc_warm4`, promoted 2026-06-25) |
| `--model PATH` | the **sideboard** BC model that picks each side's deck (omit → use authors' matchup data) |
| `--explore-sideboard T` | sample sideboard choices at softmax temp T (0 = argmax). **Needed for Stage-2** — it's what creates choice variation to learn from. |
| `--step-cap 800` | hard cap on game length before it's called a draw |
| `--base-seed N` | deterministic seed; **advance it between batches** so you don't replay identical games |

Outputs:
- `datasets/cc/parquet/games/*.parquet` — one row per in-game decision (trains the gameplay bot).
- `outputs/cc_sideboard_matches.jsonl` — one row per matchup: the sideboard choices + win/loss
  (trains the Stage-2 sideboard bot).

### 3b. Run it durably (recommended — "keep collecting")

A single batch finishes and leaves the adapters idle. To keep collecting indefinitely (and survive
crashes), use the wrapper script:

```powershell
Start-Process powershell -ArgumentList `
  "-NoProfile","-ExecutionPolicy","Bypass","-File","outputs\keep_collecting.ps1" `
  -RedirectStandardOutput "outputs\keep_collecting.log" -WindowStyle Hidden
```

`outputs/keep_collecting.ps1` loops `cc_selfplay` batches forever with advancing seeds, using
`cc_warm4` + sideboard BC + `--explore-sideboard 0.7`. Logs:
- `outputs/keep_collecting.log` — which batch is running.
- `outputs/collect_<timestamp>_seed<N>.err` — per-batch game-by-game output + draw diagnostics.

**To stop it:** kill the wrapper PowerShell process *and* its current python child:

```powershell
Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
  Where-Object { $_.CommandLine -match "cc_selfplay" } |
  ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
# then close the wrapper window/process (Get-Process powershell | … or Stop-Process the wrapper PID)
```

### 3c. Useful collection variants

```powershell
# Only one hero's matchups (e.g. to reproduce a wedge):
… cc_selfplay … --focus-hero arakni_huntsman

# Heuristic bot instead of the trained model (no --gameplay-model):
python -m python.gameplay.cc_selfplay --adapters 8000-8007 --pairs 8 --games 2

# Round-robin instead of random pairings:
… cc_selfplay … --pairing roundrobin
```

---

## 4. Train the models

All training is **offline and CPU-only** — it reads files, not the engine. You can train while
collection runs (throttle threads so you don't starve collection):

```powershell
$env:OMP_NUM_THREADS = "4"; $env:MKL_NUM_THREADS = "4"
```

### 4a. Gameplay model (IQL) — the bot that plays games

Warm-start from the current champion and train on the accumulated game corpus:

```powershell
python -u -m python.training.iql_gameplay `
  --parquet-dir datasets/cc/parquet/games `
  --out-dir outputs/models/cc_warm5 `
  --card-table outputs/models/cc/cc_card_table.pt `
  --warm-start outputs/models/cc_warm4/iql_gameplay.pt `
  --steps 8000 --draw-penalty 0.3 --time-penalty 0.002 `
  --device cpu `
  2> outputs\cc_warm5_train.err
```

- `--card-table` gives the model card features ("eyes" on CC cards) — **always pass it for CC.**
- `--warm-start` continues from the current champion. **It now correctly inherits the card
  embedding + attr_proj on a same-table CC→CC warm-start** (fixed 2026-06-24 — they used to be
  re-initialized every iteration, throwing away learned card knowledge). Card knowledge now compounds.
- `--steps 8000` for the ~400K-row corpus (scale steps with corpus size).
- Output: `outputs/models/cc_warm5/iql_gameplay.pt`.
- Naming convention: `cc → cc_warm → cc_warm2 → cc_warm3 → cc_warm4 (current champion) → cc_warm5 → …`
  (each a warm-start iteration from the previous champion).
- A new model is **unproven until gated** (see §5). The collection driver keeps using the current
  champion (`cc_warm4`) until a new model beats it.

### 4b. Sideboard BC model (Stage 1) — picks each deck's 60 + sideboard per matchup

Trains from the scraped human decks in `decks/cc_*.json` (their per-matchup card choices):

```powershell
python -m python.deckbuilding.sideboard_model
# → outputs/models/sideboard/sideboard_bc.pt
```

This rarely needs re-running — only when the scraped deck corpus changes.

### 4c. Sideboard RL model (Stage 2) — tunes sideboarding by winrate

Once `outputs/cc_sideboard_matches.jsonl` has enough **decisive** games (default needs ≥20),
refine the BC model toward choices that actually win:

```powershell
python -m python.training.sideboard_rl `
  --matches outputs/cc_sideboard_matches.jsonl `
  --bc outputs/models/sideboard/sideboard_bc.pt `
  --out-dir outputs/models/sideboard `
  --beta 3.0 --bc-anchor 0.5 --steps 1500
# → outputs/models/sideboard/sideboard_rl.pt
```

- `--beta` = how hard to favor winning choices (AWR temperature).
- `--bc-anchor` = how tightly to stay near the BC prior (prevents drifting into nonsense on thin data).
- `--window N` = use only the last N matches (0 = all).

### 4d. Card attribute table (prerequisite, already built)

The gameplay model's card features come from `outputs/models/cc/cc_card_table.pt`, built by
`python -m python.training.build_cc_card_table`. It only needs rebuilding if the CC card pool or
`slug_index.json` changes. It's already present, so normally you don't touch this.

---

## 5. Evaluate / gate a new gameplay model

A new `cc_warmN` should only replace the champion if it **wins more decisive games head-to-head**
(draws don't count, and life totals never break ties).

**Gameplay gate** (`python/examples/cc_gameplay_gate.py`): builds each matchup's two decks once (via
sideboard BC), then plays candidate vs champion on the SAME decks, swapping seats both orientations so
only the gameplay model differs. Decisive wins only. Games go to `datasets/cc_gp_gate/` (kept OUT of
the training corpus).

```powershell
python -m python.examples.cc_gameplay_gate `
  --cand outputs/models/cc_warm5/iql_gameplay.pt `
  --champ outputs/models/cc_warm4/iql_gameplay.pt `
  --sideboard outputs/models/sideboard/sideboard_bc.pt `
  --adapters 8000-8007 --matchups 60 --games 1
```

It prints `CAND X - Y CHAMP` progress and a final verdict + winrate ± SE. **Read the verdict
critically:** the script says `PROMOTE cand` on any `X > Y`, but if the winrate is within ±SE of 50%
it's a statistical tie — promotion is then a judgment call, not a clear win. (cc_warm4 was promoted on
exactly such a tie: 51.7% ± 5.4%.) At ~30 games/hr a 120-game gate is ~3.5–4 h and **needs the
adapters, so pause collection first** (§3b). Promote = repoint `outputs/keep_collecting.ps1`'s
`--gameplay-model` at the winner.

**Sideboard gate** (`python/examples/sideboard_gate.py`): the analogous tool for Stage-2 — RL-built vs
BC-built decks, same gameplay model both seats. Same flags shape (`--rl --bc --gameplay-model`).

**Inspect model internals** (sanity-check a checkpoint):

```powershell
python -c "import torch; ck=torch.load('outputs/models/cc_warm4/iql_gameplay.pt',map_location='cpu',weights_only=False); print('cards',ck['n_cards'],'attr_dim',ck['attr_dim'],'transitions',ck['n_transitions'])"
```

**Critic health audit** (is the value function actually learning?):

```powershell
python -m python.examples.analyze_qv
```

---

## 6. Monitor what's happening

```powershell
# Live pipeline status (CC self-play results, draws, wedges):
python -m python.examples.pipeline_monitor
python -m python.examples.pipeline_monitor --cc-status     # just the CC block

# Are the background jobs alive?
Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
  Where-Object { $_.CommandLine -match "cc_selfplay|iql_gameplay|sideboard_rl" } |
  Select-Object ProcessId, CreationDate

# Tail the newest collection log:
Get-ChildItem outputs\collect_*.err | Sort-Object LastWriteTime | Select-Object -Last 1 |
  ForEach-Object { Get-Content $_.FullName -Tail 30 }

# How big is the gameplay corpus right now?
python -c "import glob,pandas as pd; print(sum(len(pd.read_parquet(f)) for f in glob.glob('datasets/cc/parquet/games/*.parquet')), 'transitions')"

# How many sideboard matches collected (Stage-2 fuel)?
(Get-Content outputs\cc_sideboard_matches.jsonl | Measure-Object -Line).Lines
```

Reading a draw diagnostic line: a `DRAW … STUCK no-life-change … abort=no_progress_60_steps` means a
game wedged (engine or bot got stuck); a `DRAW … hit step_cap` with life moving means the bot just
couldn't close in time (normal, a model-quality problem, not a bug).

---

## 7. Everyday recipes

**"I just sat down, get everything running":**
```powershell
cd C:\Users\Joseph\Desktop\FAB_Sim_Headless; $env:PYTHONPATH = $PWD
$env:ADAPTER_MODE = "real"
docker compose -f docker-compose.yml -f docker-compose.parallel.yml up -d adapter adapter2 adapter3 adapter4 adapter5 adapter6 adapter7 adapter8
foreach ($p in 8000..8007) { try { (Invoke-WebRequest "http://localhost:$p/health" -TimeoutSec 5 -UseBasicParsing).StatusCode } catch { "DOWN $p" } }
Start-Process powershell -ArgumentList "-NoProfile","-ExecutionPolicy","Bypass","-File","outputs\keep_collecting.ps1" -RedirectStandardOutput "outputs\keep_collecting.log" -WindowStyle Hidden
```

**"Train the next gameplay iteration on what I've collected":** §4a (bump `cc_warm4`→`cc_warm5`,
point `--warm-start` at the current champion).

**"Run Stage-2 sideboard training":** §4c.

**"Shut everything down":** stop the wrapper + python children (§3b), then `docker compose … down`.

---

## 8. Troubleshooting

| Symptom | Cause / Fix |
|---|---|
| Edited a card/engine PHP, nothing changed | opcache. `docker restart fab_sim_headless-adapter-1 fab_sim_headless-adapter2-1 …` for all 8. New games then use the change. |
| Games all draw / one hero always stalls | A **wedge** (engine can't resolve some decision). Find the matchup in the `collect_*.err` draw diagnostics, look at the most-repeated `(phase,type,card)`. Past examples + fixes are in the memory files (`cc-wedge-hunt-findings`, `serializer-action-space-gaps`). |
| Power outage / Docker died | Restart Docker Desktop, re-run §2 (engine), then §3b (collection). Data on disk is safe (parquet/jsonl are append-only and flushed per game). |
| Collection driver exited | It may have just finished its batch (look for `DONE: N games`). The §3b wrapper auto-relaunches; a single batch does not. |
| Adapter restart interrupts in-flight games | Expected and harmless — those games are dropped, not corrupted; the driver retries and moves on. |
| Training is slow / starving collection | Set `OMP_NUM_THREADS`/`MKL_NUM_THREADS` to 3–4 before launching training (§4). |
| `python -m python.*` import errors | `$env:PYTHONPATH` not set to the project root. |

---

## 9. Where things live

| Path | What |
|---|---|
| `talishar/` | The PHP rules engine (upstream + headless edits). Card logic in `Classes/CardObjects/*Cards.php`. |
| `adapter/` | PHP HTTP wrapper around the engine. `serializers/ActionSerializer.php` = what actions the bot is offered. |
| `docker-compose*.yml` | Adapter container definitions (8000–8015). |
| `python/gameplay/cc_selfplay.py` | The self-play collection driver. |
| `python/training/iql_gameplay.py` | Gameplay IQL trainer (`--warm-start`, `--card-table`). |
| `python/deckbuilding/sideboard_model.py` | Sideboard BC (Stage 1). |
| `python/training/sideboard_rl.py` | Sideboard winrate RL (Stage 2). |
| `python/deckbuilding/sideboard.py` | `resolve()` — turns a pool + overrides into a legal 60+equipment deck. |
| `datasets/cc/parquet/games/` | Game transition corpus (gameplay training input). |
| `outputs/cc_sideboard_matches.jsonl` | Sideboard choices + outcomes (Stage-2 input). |
| `outputs/models/cc_warm*/` | Gameplay model checkpoints (`cc_warm4` = current champion, promoted 2026-06-25). |
| `outputs/models/sideboard/` | `sideboard_bc.pt`, `sideboard_rl.pt`. |
| `outputs/models/cc/cc_card_table.pt` | Card attribute table (gameplay model features). |
| `decks/cc_*.json` | Scraped human decks (pools + per-matchup choices) — Stage-1 training data. |
| `outputs/keep_collecting.ps1` | The durable collection wrapper. |
```
