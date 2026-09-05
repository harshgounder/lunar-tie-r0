# AUDIT-MUSE-20260905.md

Read-only hostile audit of lunar-tie-r0. Suite run at start: `.venv/bin/pytest tests/ -q` gives 119 passed. No files changed, nothing committed. No em dashes used below.

## P0 blockers

### F1. Panel honesty numbers are vacuous in the live chain
File: `src/lunar_tie/pipeline.py`, lines 221-222.
What the code does: step 10 calls `metrics_panel(resid_f, np.ones(resid_f.size, dtype=bool), q_hat, ...)` with an all-True inlier mask, so `inlier_ratio` is always 1.0, `rms_subpx` always equals `rms_px`, and `ece_proxy` is always `|1.0 - 0.95| = 0.05` no matter what consensus found.
What it should do: pass the real post-refit inlier mask (for example from `cons2["inliers"]` intersected with the coverage selection) so the tier-stamped panel reports measured values.
Severity: P0 blocker. The panel is sold as the honesty contract and its three headline numbers cannot vary.
Evidence (ran):
```
p = metrics_panel(resid, np.ones(10, dtype=bool), q, tier_label=1)
# inlier_ratio= 1.0 rms_subpx == rms_px: True ece_proxy= 0.05
```

### F2. Live-chain ABSTAIN ladder is calibrated on the same array it gates
File: `src/lunar_tie/pipeline.py`, lines 207-209.
What the code does: `three_way_gate(resid_f, resid_f, np.zeros(resid_f.size, dtype=bool), alpha=...)` passes the test residuals as their own calibration set, so the accept fraction sits near 95 percent by construction and the gate cannot fail on real data.
What it should do: calibrate on held-out residuals (the module docstring itself warns that placeholder calibration input voids the ladder) or drop the in-chain three-way label and report only the split-conformal q_hat.
Severity: P0 blocker. Same known class as stated in the brief: a comparison satisfiable by construction.
Evidence (ran):
```
q = conformal_calibrate(resid, alpha=0.05)
lab = three_way_gate(resid, resid, np.zeros(10, dtype=bool), alpha=0.05)
# self-cal q= 2.0 accept frac= 1.0 on data containing a 2.0 px outlier
```

## P1 real bugs

### F3. Tier validation in metrics_panel is bypassed by the pipeline
File: `src/lunar_tie/pipeline.py`, lines 222 and 233.
What the code does: the call passes `tier_label=min(tier_label, 1) or 1`, which is 1 for every integer input, then writes the raw manifest value back with `panel["tier_label"] = tier_label`. An invalid tier such as 9 passes validation silently and is stamped on the panel.
What it should do: validate the manifest tier first and raise `PipelineError` on values outside {0} union {1..5} (0 allowed only via the N9 unlabeled path), instead of laundering every value through 1.
Severity: P1 real bug. The loud-vs-silent error policy for tiers is defeated.
Evidence (ran):
```
for t in (0, 3, 9): min(t, 1) or 1
# 0 -> 1, 3 -> 1, 9 -> 1 (all pass; 9 stamped afterwards)
metrics_panel(resid, ones, q, tier_label=9)  # direct call raises ValueError
```

### F4. match_descriptors crashes when side B has exactly one descriptor
File: `src/lunar_tie/detect.py`, lines 309-315.
What the code does: with `dist.shape == (N, 1)`, `k` is set to 1 but line 314 still indexes `best_idx[:, 1]` and `dist[..., best_idx[:, 1]]`, raising `IndexError`. Empty input is guarded, single-row input is not.
What it should do: return `[]` (or run a distance-threshold match) when either side has fewer than 2 descriptors, matching the documented empty-safe behavior.
Severity: P1 real bug. A sparse but legal detection outcome kills the chain with an unhandled exception instead of a loud `PipelineError`.
Evidence (ran):
```
match_descriptors(rand(5,128), rand(1,128))
# IndexError index 1 is out of bounds for axis 1 with size 1
```

## P2 smells and test holes

### F5. fit_homography divides by H[2,2] with no guard and no finite check
File: `src/lunar_tie/consensus.py`, line 177.
What the code does: `return H / H[2, 2]`. When the null vector has a zero last component (degenerate geometry), this returns inf/nan with only a numpy warning. The sibling `fit_similarity` has an explicit non-finite guard (lines 65-66); this function has none.
What it should do: check `H[2, 2]` for zero and check finiteness of the result, raising `ValueError` loudly per the module ambiguity policy.
Severity: P2 smell. Not triggered in 250 random trials, but the junk path is one degenerate call away and the fix is two lines.
Evidence (ran):
```
H0 = array([[1,0,0],[0,1,0],[0,0,0]]); H0 / H0[2,2]
# array([[inf, nan, nan], [nan, inf, nan], [nan, nan, nan]])
```

### F6. anms_select docstring promises all points when n_target exceeds pool size
File: `src/lunar_tie/coverage.py`, lines 154-157.
What the code does: both selection rounds enforce the `min_distance` guard, so a tight cluster of 4 with `n_target=10` returns 1 index, not all 4.
What it should do: either return all pool points in that case or correct the docstring to state the distance guard takes precedence.
Severity: P2 smell (contract lie). The existing test only covers far-apart points, so the lie is untested.
Evidence (ran):
```
anms_select([1,.9,.8,.7], clustered4, n_target=10, min_distance=40, W=512, H=512)
# pool=4 n_target=10 selected= 1
```

### F7. conformal small-n quantile clipping breaks the exactness claim
File: `src/lunar_tie/conformal.py`, lines 51-52.
What the code does: `k = min(ceil((n+1)(1-alpha)), n)` clips to the sample max. Exact split-conformal requires q_hat = +inf when k > n (abstain on everything); clipping returns a finite max instead and overstates coverage for tiny calibration sets.
What it should do: return `inf` (or raise) when `k > n`, and state the minimum-n behavior in the docstring.
Severity: P2 smell. The live chain uses large n, so impact is limited to small-stratum callers relying on the stated finite-sample exactness.
Evidence (ran):
```
n= 1 q_hat= 1.0 (exact k= 2 -> clipped to n)
n= 5 q_hat= 5.0 (exact k= 6 -> clipped to n)
```

### F8. Memory-discipline test cannot fail by construction
File: `tests/test_tiling.py`, lines 116-125.
What the code does: `assert sys.getsizeof(tile) < 1024*1024` on a numpy memmap view. `getsizeof` measures the view object header (160 bytes for a 4 MB view, verified by run), never the mapped data, so the assertion passes regardless of memory behavior.
What it should do: assert on `.nbytes` accounting plus a resident-memory bound, or drop the claim that streaming discipline is tested.
Severity: P2 smell (test hole, same by-construction class as the brief hunt list).
Evidence (ran):
```
mm = memmap(shape=(2048,2048), dtype=u1)  # 4 MB
sys.getsizeof(mm)  # 160, always < 1 MB; mm.nbytes 4194304
```

### F9. PDS3 round-trip test compares the parse output to itself
File: `tests/test_pds3label.py`, lines 91-100 (same pattern lines 127-137, 154-163).
What the code does: a test-local serializer formats the already-parsed dict, then asserts the parsed values appear in that text. Any lossy parse (dropped duplicate keys, coerced `007` to `7`) passes as long as formatting is self-consistent. It never re-parses output or diffs against source bytes.
What it should do: re-parse the serialized text and compare dicts, or byte-compare normalized source lines, to earn the word round-trip.
Severity: P2 smell (test hole). Related parser losses verified live: duplicate same-level keys overwrite silently (`LINES=100` then `LINES=200` keeps 200) and `007` coerces to int 7.
Evidence (ran):
```
parse t3.au with LINES=100, LINES=200, ZERO=007
# dup LINES kept: 200 | ZERO 007 -> 7
```

### F10. pds3label mishandles single-quoted multi-line values
File: `src/lunar_tie/pds3label.py`, lines 93-106.
What the code does: the continuation join only triggers for values starting with `"`. A single-quoted value split across lines parses as a truncated literal (`'123` in the probe).
What it should do: handle both quote styles, since the module docstring claims multi-line quoted values generally.
Severity: P2 smell. Fixtures only exercise double quotes, so the suite stays green.
Evidence (ran):
```
parse 'A = '"'"'123<newline>  continued'"'"''  ->  {'A': "'123", 'B': 5}
```

### F11. README contradicts itself on MINI-GATE-2; neither mini-gate is in the pytest suite
Files: `README.md` lines 58 vs 105-108; `tests/test_minigate1.py`, `tests/test_minigate2.py` (zero collected tests).
What the code/docs do: the module table says MINI-GATE-2 PASS with exact numbers while the honesty layer says it is not yet green. Both scripts define `main()` with bare asserts and no `test_` functions, so `pytest tests/ -q` collects 0 tests from them; the 119 green count excludes both gates.
What is true (verified by run): the MG2 script passes when executed directly (111 s; s=1.2001 vs 1.2, theta=0.2000 vs 0.2, inliers 64.9 percent, rms 0.146 px, occupancy 0.484, tier-3 panel), matching PROGRESS-LOG and MILESTONE-1-CLOSEOUT. The table is right and the honesty-layer paragraph is stale.
What it should do: update the honesty-layer paragraph and either convert the mini-gates to collected pytest tests or stop implying the 119 count covers them.
Severity: P2 smell (doc-claim drift, the exact item the brief asked to reconcile). Fixture truth for the record: the header print claims s=1.15, t=(9,-5) (line 24) but the code builds s=1.2, t=(9,10.5) (lines 48, 57); theta=0.2 both places. Assertions checked: matches >= 8, inlier_ratio >= 0.6, |s - 1.2| < 0.03, |theta - 0.2| < 0.02, occupancy >= 0.35.
Evidence (ran):
```
pytest --collect-only: 119 tests, 0 from test_minigate1.py / test_minigate2.py
python tests/test_minigate2.py  # PASS, numbers as above
```

### F12. MAGSAC docstring describes sigma marginalization the code does not do
File: `src/lunar_tie/consensus.py`, lines 83-89 and 137.
What the code does: scores with a hard `res < sigma_thr` gate and returns `sigma_max = float(sigma_thr)`. The docstring says sigma_max equals sigma_thr times the median residual with soft-threshold tolerance.
What it should do: align the docstring with the hard gate or implement the described marginalization.
Severity: P2 smell (contract lie). Behavior itself is deterministic and tested; only the description misleads.
Evidence (read): lines 109 and 137 show the hard mask and constant; no median appears anywhere in the function.

## P3 docs

### F13. Photometric memory invariant understates the live allocation
File: `src/lunar_tie/photometric.py`, lines 13-17 and 64-76.
What the code does: `normalize_ratio` simultaneously holds L, mu, L2, mu2, num, var, den, out (about eight full-size float64 arrays at peak) before returning float32. The docstring says peak memory has a constant factor of two full-size temporaries.
What it should do: state the true constant for `normalize_ratio` (the factor-of-two claim holds for `gaussian_blur` alone: padded plus work buffer).
Severity: P3 doc.

### F14. refine_matches edge rule is stricter than its contract
File: `src/lunar_tie/subpixel.py`, line 156 vs docstring lines 23-25.
What the code does: any point within `half` (default 16 px) of the border is marked invalid, although `extract_patch` uses reflection and could serve such points. The docstring promises only fully-outside windows are rejected.
What it should do: document the half-margin requirement as the real rule.
Severity: P3 doc. Behavior is safe (loud invalid flag, zero delta, input position kept; verified: point (5,5) invalid with delta (0,0)).
Evidence (ran):
```
refine_matches(img, img, [[5,5],[32,32]], same, half=16)
# valid: [False True] deltas: [[0.0, 0.0], ...]
```

## VERIFIED-OK (attacked, held up)

* NaN consensus path is loud: `fit_similarity` rejects non-finite input (lines 39-40), `magsac_consensus` skips poisoned hypotheses and raises when none are valid; `test_nan_input_raises_loud` and `test_fit_similarity_nan_raises` pass in the suite.
* Masking NaN and boundary values: NaN yields combined False without crash (suite); value 254 is valid and combined True (ran), so no valid/nodata limbo band exists.
* Tiling grid math: divisible, non-divisible (2050x1050 edge tiles (2048,0,2050,1024) and (2048,1024,2050,1050)), and smaller-than-tile cases all asserted in the suite and correct by code read.
* Phase-correlation identity and known-shift recovery: suite asserts hold; no silent junk observed.
* Pipeline translation-only chain: 6/6 gate tests pass (artifacts written, median tie within 0.5 px of (4,3), tier/provenance propagation, loud nonzero exit on missing input, tier-0 UNLABELED default).
* Coverage bounds policy: out-of-bounds and non-finite points raise ValueError (suite `test_out_of_bound_point_raises_loud`); quadrant convention (`>=` midpoint) is consistent between metric and selection paths.
* Collinear and too-few-points consensus inputs raise loudly (suite).
* Flat-image Otsu returns None with zero shadow (suite); the smooth-gradient over-shadowing limitation is already disclosed in the PROGRESS-LOG MINI-GATE-1 row, so it is a logged limitation, not a new finding.
* `summary.txt` provenance: `panel.get("provenance_str", ...)` resolves because `extra` carries `provenance_str` through `metrics_panel`; no KeyError path found.
