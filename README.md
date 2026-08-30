# lunar-tie-r0 - LUNAR-TIE R0 Spine (SIH26166, milestone 1)

Classical-track lunar image registration vertical slice, per
sih-2026/docs/BRIEF-MILESTONE-1-R0-SPINE.md (the plan of record).
Repo type: fresh implementation, window-1/lunar-tie is reference only.

## layout (per brief)
```
  src/lunar_tie/        the package (one module per DECOMPOSITION-FINE unit)
  tests/                gate tests, one file per unit id
  fixtures/             3-sample PDS3 labels, synthetic mask fixtures
  manifests/            pair manifests (CNSF / MoonAnything / Kaguya M/E)
  scripts/fetch_pairs.py  zenodo/MDPI fetcher (pre-step, Hermes-owned)
  zoo/schema.json       the monster ledger schema (monster #0 = this R0 track)
  PROGRESS-LOG.md       one row per unit: id, gate result, commit hash
  ARCHITECTURE.md       live architecture doc, updated per unit
```

## rule set (from the brief, enforced)
- one unit per commit, gate test exists BEFORE a unit merges
- no learned models, no crater-graph, no zoo dispatch in R0
- every metric carries a truth-tier label (tier-3 SPICE+DEM for this milestone)
- zero Ch-2 inputs: must run with zero Ch-2 products on disk (N9 guarantee)
- no em-dash, no AI-tell words in any file

## status
| unit | id | title | gate | commit |
|---|---|---|---|---|
| (none yet) | | | | |

queue: UNIT-1 = PDS3 label parser (D1 1.1 / P1.m1)
       UNIT-2 = shadow/nodata triple mask (D1 1.3 / P1.m3)
MINI-GATE after every 3 units: end-to-end sanity on one pair.