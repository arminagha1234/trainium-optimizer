# bank-snapshots

Off-box durable backups of the trainium-optimizer **knowledge bank** (the
framework's accumulated learning). This is a DATA-ONLY orphan branch — it shares
no history with `main` and contains no framework code, so it never interferes
with the code on `main` or the optimizer loop's `git pull --ff-only` on main.

## Contents
- `knowledge-bank/` — a periodic snapshot of the pooled bank, mirrored from the
  live bank on the publisher box (`.211`). Layout:
  `<tier>/<family>/<type>/<lesson_id>.yaml` (one lesson per file).
  Tiers: `provisional`, `verified`.
- `MANIFEST.txt` — lesson counts per tier/family at snapshot time.

Snapshots are pushed by `bank_publish.sh` on the publisher box (`.211` only)
using a repo-scoped, write-enabled **deploy key** — never the user's broad PAT.

## Restore onto a fresh box
Populate a fresh framework clone's `knowledge-bank/` from the latest snapshot:

```bash
git clone https://github.com/arminagha1234/trainium-optimizer.git
cd trainium-optimizer
git fetch origin bank-snapshots
git restore --source=origin/bank-snapshots -- knowledge-bank
# knowledge-bank/ is now populated; the loop seeds its beam from it.
```

One-liner (into an existing clone, leaves your current branch untouched):

```bash
git fetch origin bank-snapshots && git restore --source=origin/bank-snapshots -- knowledge-bank
```
