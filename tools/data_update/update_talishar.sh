#!/usr/bin/env bash
# Refresh the bundled Talishar engine from upstream while preserving the
# ~52 surgical headless edits (e.g. include -> include_once for the persistent
# php-server). No branches, no commits — your clone stays exactly as it is now
# (uncommitted edits on upstream main); the edits are just carried across the
# update via stash, with a .patch backup as a safety net.
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
talishar="$here/../../talishar"
cd "$talishar"

ts="$(date +%Y%m%d-%H%M%S)"
patch="$here/headless_${ts}.patch"
git diff > "$patch"
echo "Saved current headless edits -> $patch"

# Stash the local edits (incl. any new untracked files), pull upstream, reapply.
git stash push -u -m "headless-${ts}"
git fetch origin
echo "Upstream commits being applied:"
git log --oneline "HEAD..origin/main" | head -20 || true
git merge --ff-only origin/main

if git stash pop; then
  echo
  echo "Reapplied headless edits cleanly. Restart adapters to clear PHP OPcache"
  echo "(talishar is a volume mount, no image rebuild needed):"
  echo "  docker compose -f docker-compose.yml -f docker-compose.parallel.yml restart"
else
  echo
  echo "Some edits conflicted with upstream changes. Resolve the conflicts, or"
  echo "discard and re-apply selectively from the backup:"
  echo "  git checkout -- . && git apply --3way \"$patch\""
  exit 1
fi
