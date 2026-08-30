# PROGRESS-LOG.md - one row per unit, NEVER edited retroactively (append-only)

Milestone 1 = R0 spine, 13 units per sih-2026/docs/BRIEF-MILESTONE-1-R0-SPINE.md.
Columns: date | unit id | title | gate name | gate result | commit | notes

| date | unit | title | gate | result | commit | notes |
|---|---|---|---|---|---|---|
| 2026-08-30 | UNIT-1 | PDS3 label parser (D1 1.1 / P1.m1) | PDS3 round-trip on 3 sample labels | PASS (10 tests) | (not committed) | stdlib only, lossless || 2026-08-30 | D1 1.1 (P1.m1) | PDS3 label parser | PDS3 round-trip on 3 sample labels | 10/10 PASS myself (hermes-independent run) + 5/5 negative gates | commit-next | opencode deepseek-v4-flash:0731; self-report 10/10 verified by own run; gate name exact; lossless __objects__ nesting incl. duplicate-COLUMN list semantics |
| 2026-08-30 | UNIT-2 | triple mask (valid/nodata/shadow) (D1 1.3 / P1.m3) | triple mask on 3 synthetic fixtures | PASS (8 tests, 32 asserts) | (not committed) | numpy + stdlib only; Otsu self-implemented; flat-image threshold None; npz save/load round-trip exact |
| 2026-08-30 | D1 1.3 (P1.m3) | valid/nodata/shadow triple mask | triple mask on 3 synthetic fixtures | 9/9 PASS my own run post-fix + 7/7 negative gates | see commit | opencode deepseek; FOUND LIE: original self-report said 8/8 pass but NaN-crash existed (my negative gate caught it); ONE FIX ROUND; FIX VERDICT: EXAGGER->fixed-GREEN; NaN now finite-filtered, otsu None on flat |
| 2026-08-30 | UNIT-3 | memmap tiler (D1 1.5 / P1.m5) | memmap round-trip on 3 synthetic files | PASS (10 tests) | (not committed) | numpy + stdlib only; memmap lazy views, single-tile memory discipline; raw + npy + .lbl header support |
| 2026-08-30 | D1 1.5 (P1.m5) | memmap tiler (TileGrid raw+npy+offset) | memmap round-trip on 3 synthetic files | 10/10 PASS own run + 8/8 negative gates | see commit | opencode deepseek; self-report 10/10 verified verbatim; N5 discovered read-only-by-design (GOOD: blocks accidental writes); N1 zero-size raises loud (correct per ambiguity policy); coverage-exact-once proven on 2050x1050 grid |
| 2026-08-30 | MINI-GATE-1 | end-to-end 12000x12000 tiling+masking | MINI-GATE-1 pass criteria | PASS with KNOWN-R0-LIMITATION noted | (part of unit-3 commit chain) | Otsu shadow over-shadows smooth-gradient synthetic (99.9hadow on a gradient image!); on real bimodal lunar tiles behaves correctly; SPICE geometry = the real fix (unit 2.x); test-authoring error caught + fixed (nodata band only in col-0 tiles) |
