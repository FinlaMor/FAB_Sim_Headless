#!/usr/bin/env bash
# Run N bare `php -S` adapters + durable CC self-play collection, ALL inside one
# Linux box (no docker-compose). Built for a single RunPod CPU pod, but works on
# any Linux host. Each adapter is one PHP process on port 8000+i; the driver
# shards games across them. Stop with Ctrl-C (kills the adapters via the trap).
#
# Tunables (env vars, all optional):
#   NUM_ADAPTERS   how many php -S adapters / driver workers   (default: nproc)
#   BASE_SEED      starting seed; use a HIGH range so cloud games don't collide
#                  with local ones (default: 20000000)
#   PAIRS / GAMES  matchups per batch / games per matchup      (default 400 / 2)
#   GAMEPLAY_MODEL / SB_MODEL   checkpoints to pilot/sideboard  (default cc_warm4 / sideboard_bc)
#
# Usage:  NUM_ADAPTERS=32 BASE_SEED=20000000 bash cloud/cloud_collect.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export ADAPTER_MODE=real
export PYTHONPATH="$ROOT"

N="${NUM_ADAPTERS:-$(nproc)}"
BASE_SEED="${BASE_SEED:-20000000}"
PAIRS="${PAIRS:-400}"
GAMES="${GAMES:-2}"
GAMEPLAY_MODEL="${GAMEPLAY_MODEL:-outputs/models/cc_warm4/iql_gameplay.pt}"
SB_MODEL="${SB_MODEL:-outputs/models/sideboard/sideboard_bc.pt}"
PHPINI="$ROOT/cloud/php-adapter.ini"
HI=$((8000 + N - 1))

command -v php   >/dev/null || { echo "FATAL: php not installed (see cloud/README_RUNPOD.md)"; exit 1; }
php -m | grep -qi shmop || { echo "FATAL: php 'shmop' extension missing (the engine needs it)"; exit 1; }

echo "[cloud] launching $N adapters on ports 8000..$HI"
pids=()
for i in $(seq 0 $((N - 1))); do
  php -c "$PHPINI" -S "0.0.0.0:$((8000 + i))" -t adapter adapter/api.php >/dev/null 2>&1 &
  pids+=($!)
done
trap 'echo "[cloud] stopping adapters"; kill "${pids[@]}" 2>/dev/null || true' EXIT

echo "[cloud] waiting for adapters to come up..."
for i in $(seq 0 $((N - 1))); do
  until curl -sf "http://localhost:$((8000 + i))/health" >/dev/null 2>&1; do sleep 0.5; done
done
echo "[cloud] all $N adapters healthy"

base="$BASE_SEED"
batch=0
while true; do
  echo "[cloud] batch=$batch base-seed=$base start=$(date -u +%FT%TZ)"
  python -u -m python.gameplay.cc_selfplay \
    --adapters "8000-$HI" --pairs "$PAIRS" --games "$GAMES" \
    --model "$SB_MODEL" --gameplay-model "$GAMEPLAY_MODEL" \
    --explore-sideboard 0.7 --step-cap 800 --base-seed "$base" \
    2>> "outputs/cloud_collect_seed${base}.err" || true
  echo "[cloud] batch=$batch done=$(date -u +%FT%TZ)"
  base=$((base + 1000))
  batch=$((batch + 1))
  sleep 3
done
