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

## status (updated 2026-08-30 after units 1-9)

| # | unit id | title | gate result | commit |
|---|---|---|---|---|
| 1 | D1 1.1 / P1.m1 | PDS3 label parser | 10/10 + 5/5 neg | ea1965e |
| 2 | D1 1.3 / P1.m3 | triple mask | 9/9 + 7/7 | ed69fdd |
| 3 | D1 1.5 / P1.m5 | memmap tiler | 10/10 + 8/8 | 1f03527 |
| MG1 | MINI-GATE-1 | 12000x12000 end-to-end | PASS | d18ffb8 |
| 4 | P3.m1/D4 | photometric ratio-norm | 12/12 + 7/7 | 1059ccc |
| 5 | D4 5.1-5.2/P4.m2 | SIFT + RD-SIFT | 12/12 + 8/8 | 261d5e6 |
| 6 | D4 7.1-7.3/P5.m1 | MAGSAC + DLT homography | 15/15 + 8/8 | 98f3721 |
| 7 | D4 8.1-8.3/P6.m1 | phase-corr subpixel + ECC-ready | 12/12 + 8/8 | 09f1fb6 |
| 8 | D9 10.1-10.4/P7 | coverage gates + ANMS | 11/11 + 8/8 | 8b4a1d0 |
| 9 | D10 12.x/P8.m3 | conformal ABSTAIN + panel | 17/17 + 8/8 | 4a035a2 |

full suite: 108/108 (9 modules). lie-hunt battery after every merge: 6 catches,
all fixed + re-audited (protocol = sih-2026/LIE-CHECK-PROTOCOL.md; per-unit notes in PROGRESS-LOG.md).

remaining M1: MINI-GATE-2 (full-chain synthetic pair; fixture fix in flight),
UNIT-10 pipeline CLI driver, close-out row. then milestone-2 (learned track).
