# Running CC collection on a RunPod CPU pod

A proof-of-concept runbook for collecting self-play games on rented Linux cores. The
tooling (`cloud_collect.sh`) runs N bare `php -S` adapters + the driver inside **one**
pod — no docker-compose (RunPod pods can't nest Docker). Reusable on any Linux box.

## Honest budget note
$8.50 is a **validation** budget: prove the pipeline runs on cloud cores and measure the
real throughput multiplier on native Linux (where the Windows-Docker VM wall doesn't
exist). It is NOT a "scale for weeks" budget. Goal = decide whether a bigger box is worth
real money. **Use a CPU pod, never a GPU pod** — we don't use the GPU; it'd waste credit.

## Step 1 — create the pod (RunPod console; only you can do this)
1. RunPod → **Pods → Deploy → CPU** (not GPU).
2. Pick a CPU instance with as many vCPUs as your budget allows (more cores ≈ proportionally
   more games/hr). Note the **$/hr** so you know your runway (`$8.50 / $perHr` = hours).
3. Base image: a plain Ubuntu or Python image (e.g. `runpod/base` or `python:3.12`).
4. **Attach a Volume** (e.g. mount at `/workspace`) — this is how data survives if the pod
   restarts. Put the repo + outputs on the volume.
5. Deploy, then **Connect → SSH** (or the web terminal).

## Step 2 — get the repo onto the pod
Option A (push from your Windows box — keeps your local edits, e.g. the Coercive fix &
cc_warm4):
```bash
# from your LOCAL machine (Git Bash), replace HOST/PORT with the pod's SSH details:
rsync -avz --exclude datasets/cc/parquet --exclude '.git' \
  /c/Users/Joseph/Desktop/FAB_Sim_Headless/ root@HOST:/workspace/FAB_Sim_Headless/ -e 'ssh -p PORT'
```
(You DO need `talishar/ adapter/ decks/ python/ cloud/ slug_index.json outputs/models/`.
You do NOT need to ship the local game corpus — the pod generates its own.)

## Step 3 — install PHP + Python deps (on the pod)
```bash
apt-get update
apt-get install -y php8.1-cli php8.1-mbstring php8.1-mysql php8.1-gd php8.1-zip curl rsync
php -m | grep -i shmop   # MUST print "shmop" — the engine needs it (it's bundled in php8.1-cli)
cd /workspace/FAB_Sim_Headless
pip install -r python/requirements.txt
```
If `shmop` is missing, install `php8.1-common` or your distro's shmop package and re-check.

## Step 4 — collect
```bash
cd /workspace/FAB_Sim_Headless
# NUM_ADAPTERS defaults to all cores; BASE_SEED kept high so cloud games don't
# collide with local seeds (local uses 5xxxxx).
NUM_ADAPTERS=$(nproc) BASE_SEED=20000000 bash cloud/cloud_collect.sh
```
You'll see `all N adapters healthy`, then per-batch progress. Games land in
`datasets/cc/parquet/games/` and sideboard matches in `outputs/cc_sideboard_matches.jsonl`
— exactly the same paths/format as local, so the corpora merge by just copying files.

Run it under `tmux`/`nohup` so it survives your SSH disconnecting:
```bash
tmux new -s collect      # then run the command above; detach with Ctrl-b d
```

## Step 5 — pull the data back (from your LOCAL machine)
```bash
rsync -avz root@HOST:/workspace/FAB_Sim_Headless/datasets/cc/parquet/games/ \
  /c/Users/Joseph/Desktop/FAB_Sim_Headless/datasets/cc/parquet/games/ -e 'ssh -p PORT'
rsync -avz root@HOST:/workspace/FAB_Sim_Headless/outputs/cc_sideboard_matches.jsonl \
  /c/Users/Joseph/Desktop/FAB_Sim_Headless/outputs/ -e 'ssh -p PORT'
```
**Sync often** (every hour or so) — a community-cloud pod can be interrupted; unsynced
parquet on an ephemeral disk is lost. (On a Volume mount it survives a restart.)

Then train + gate locally exactly as in `docs/HOWTO_RUN.md` — the extra games just make the
next `cc_warm5` corpus bigger.

## What to look for (the validation payoff)
- **games/hr** in the batch logs vs our local ~21–35. If a 16–32-core pod gives a clean
  multiple AND each game runs faster than local (native Linux, no VM proxy), the cloud path
  is proven and a bigger box is worth real money.
- Watch your spend: `hours_used × $perHr`. Stop the pod from the console when done — billing
  continues while it's running, even idle.

## Notes / gotchas
- **No GPU.** Inference is CPU (torch num_threads is pinned to 1 inside the driver — the key
  perf win — carries over automatically).
- **No MySQL/Redis.** The headless engine stubs the DB; state is file-based. Nothing to
  provision beyond CPU + disk.
- **opcache.** `cloud/php-adapter.ini` sets `validate_timestamps=0`, so if you edit engine
  PHP on the pod you must restart `cloud_collect.sh` for it to load.
