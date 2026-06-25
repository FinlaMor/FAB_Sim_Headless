# Updating the two external data sources

Two pieces of this repo are vendored from upstream projects and need a
repeatable refresh procedure.

---

## 1. `slug_index.json` ← fabrary/cards

`slug_index.json` (repo root) is the authoritative card/format index used to
pick CC-legal heroes (`python/datasets/fabrary/heroes.py::cc_legal_hero_ids`)
and, later, equipment slots for the sideboard bot.

**Source:** the [`fabrary/cards`](https://github.com/fabrary/cards) project,
distributed as the npm package **`@flesh-and-blood/cards`** (re-exports
`the-fab-cube/flesh-and-blood-cards`).

**Format:** kept in the package's **native** form — no transform, so it tracks
upstream exactly and never drifts:
- key   = `cardIdentifier` (dashed, e.g. `kano-dracai-of-aether`) — this is
  *exactly* fabrary's `heroIdentifier`, so scrape queries use the keys directly.
- value = the card object verbatim (`legalFormats` keep spaces, e.g.
  `"Classic Constructed"`).

**To update:**
```bash
cd tools/data_update
npm install            # or: npm update @flesh-and-blood/cards @flesh-and-blood/types
node gen_slug_index.mjs   # rewrites ../../slug_index.json
```
Sanity check (should print the current CC-legal hero count, ~38):
```bash
cd ../.. && python -c "from python.datasets.fabrary.heroes import cc_legal_hero_ids as f; print(len(f()))"
```

> Note: consumers read native format — `'Classic Constructed'` (with space) and
> dashed keys. If you ever re-introduce a transform, update `heroes.py` too.

---

## 2. `talishar/` ← Talishar/Talishar (the PHP backend)

`talishar/` is a clone of <https://github.com/Talishar/Talishar.git> carrying
**~52 surgical headless adaptations** (uncommitted working-tree edits, e.g.
`include` → `include_once` for the persistent php-server). A plain `git pull`
would clobber/conflict with them, so updates must carry them across.

**No branches, no commits** — the clone stays exactly as it is now (uncommitted
edits on upstream `main`). Just run:
```bash
tools/data_update/update_talishar.sh
```
which: saves a `.patch` backup of the edits → `git stash` → `git fetch` +
`git merge --ff-only origin/main` → `git stash pop` to reapply. On conflict it
stops and points you at the backup (`git apply --3way`).

Equivalent by hand:
```bash
cd talishar
git diff > ../tools/data_update/headless.patch   # backup
git stash push -u
git fetch origin && git merge --ff-only origin/main
git stash pop                                     # reapply; resolve any conflicts
```

> A local stash/patch is entirely local — nothing is ever pushed to the Talishar
> repo. We only ever *read* from upstream (`fetch`), never write to it.

**Deploy:** `talishar/` is a *volume mount* (`./talishar:/srv/talishar:rw`), so
no image rebuild is needed — just restart the adapters to clear PHP OPcache:
```bash
docker compose -f docker-compose.yml -f docker-compose.parallel.yml restart
```

> The headless edits are tiny and mechanical; conflicts (when they happen) are in
> the same `APIs/*.php` files — reapply the `include_once` / idempotency change.
