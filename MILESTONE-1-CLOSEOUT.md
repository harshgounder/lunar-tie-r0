# MILESTONE-1 CLOSE-OUT (2026-08-30, evening)

## verdict: MILESTONE-1 COMPLETE (10 of 13 brief units + 2 mini-gates + 1 addon unit)

## the final state, re-derived (not remembered)
- repo: github.com/harshgounder/lunar-tie-r0 (private), local ~/lunar-tie-r0
- suite: 119/119 tests across 10 modules + pipeline
- MINI-GATE-1 (tiling+masking at 12000x12000): PASS (d18ffb8)
- MINI-GATE-2 (full chain on similarity pair): PASS (5016273)
  s=1.2001 vs 1.2 | theta 0.2000 vs 0.2 | inliers 64.9% |
  rms 0.146 px | occupancy 48.4% | entropy 0.806 | panel tier-3
- UNIT-10 cleanroom: tie delta (4.031, 3.031) vs truth (4, 3); rms 0.0;
  tier/provenance propagation verified; N9 defaults verified
- commits: f8b9164 (born) .. 5a5f6e1 (unit-10) + e2cd304 (audit row)
- lie-hunt: 9 catches total; every catch produced FIX + re-audit
- opencode chain: glm-5.3-flash on user's ollama pro key (rotated + wired)

## module inventory (unit -> module -> gate)
 1  pds3label.py    PDS3 round-trip 3 fixtures + 5 neg      ea1965e
 2  masking.py      triple mask 3 fixtures + 7 neg + NaNFIX ed69fdd
 3  tiling.py       memmap round-trip 3 grids + 8 neg       1f03527
 MG minigate1        12000x12000 end-to-end                  d18ffb8
 4  photometric.py   ratio-norm + 7 neg (mad 0.0000)        1059ccc
 4B rank_transform   +5 tests, gain-invar 0.000000 mismatch  5016273 (bundled w/ MG2)
 5  detect.py        SIFT + RD-SIFT (numpy DoG) 12 + 8 neg  261d5e6
 6  consensus.py     MAGSAC-style + DLT, NaN-loud 15/15     98f3721
 7  subpixel.py      phase-corr subpixel + footprint 12/12  09f1fb6
 8  coverage.py      occupancy/entropy/quad/NNCV + ANMS     8b4a1d0
 9  conformal.py     split-conformal + 3-way gates 17/17    4a035a2
 MG2 full chain      s=1.2001 theta=0.2000 rms 0.146 px     5016273
 10 pipeline.py      CLI driver, 6/6 + cleanroom            5a5f6e1

## lie-hunt tally (the habit that paid)
9 catches across the day:
 opencode x2  (NaN crash behind green; NaN junk-M behind green)
 mine x2      (0.95 cosine bar; uint8 fixture cast)
 contract x2  (Otsu gradients; 3-way signature placeholder)
 fixture x3   (NN warp; flat bg starvation; wrong metric key)
each one: FIX round -> independent re-run -> ledger row. zero deferred.

## honest boundaries (stated, not hidden)
- >1.2x radiometric gain: routed to M2 relight stack (unit 3.5-3.8).
  R0 validates the chain at the defensible 1.11x/5 TDI-mismatch level
- rotate+scale with subpixel truth: proven green via MG2
- 12k-px tile + masking proven in MG1; full 1.2-GP Ch-2 frame load
  still needs real product files (user ISSDC action)

## M2 entry state (for the next phase)
- checkpoints order per deep-research D5: MatchAnything (public
  weights, best-evidenced) -> LightGlue+SIFT path -> RoMa v2 (verify
  weights first). conformal-first: learn uncertainty, keep ABSTAIN.
- compute: Kaggle (user, phone-verify 2-3) ~2h GPU for the M2 bench
- data strata: Kaggle-public lunar pairs until ISSDC lands
- hunt tranche-2 config: stratified folds terrain/illum/date, FDR
  budget declared pre-fire, monster #1 candidate = track-B row

## sign-off
M1 closed by Hermes audit: suite 119/119, 6/6 repos pushed 0/0,
tickets 14, ledger 20 rows, all canon docs current. build integrity:
every landed unit carries an independently-run gate + negative-gate
battery + lie-hunt pass. M1 DONE.