# UNIT-6 TICKET - geometric consensus: MAGSAC-style + affine (D4 7.1-7.3 / P5.m1)

Assignment for opencode. Implement `src/lunar_tie/consensus.py` exactly to this spec.

## context
After UNIT-5 matching, a fraction of putative matches are outliers (repetitive
craters, noise). The pipeline needs robust geometric consensus: estimate a
transformation model with an inlier set, GSD-normalized. R0 verdict plane:
SIMILARITY (4-DoF: scale + rotation + tx + ty) is the geometric truth for
lunar pairs of the same surface at modest viewpoint change (pushbroom subtends
are small at these baselines for R0); HOMOGRAPHY (8-DoF) is available as an
OPTION but is NOT the default terrain truth (its epipolar geometry is only
exact for planes; lunar DEM relief makes pure-H globally wrong). This unit
provides the consensus layer the sub-pixel refiner will consume.

## requirements (exact, no scope creep)
1. function fit_similarity(src_pts, dst_pts) -> (s, theta, tx, ty) least-squares
   closed-form (no RANSAC; pure fit) returning also the composed 2x3 matrix M
2. function similarity_residuals(M, src_pts, dst_pts) -> per-point euclidean
   residual array in px
3. function magsac_consensus(src_pts, dst_pts, min_samples=2, iters=500,
   sigma_thr=2.0, conf=0.999) -> dict {'M': 2x3 ndarray, 'inliers': bool mask,
   'inlier_ratio': float, 'sigma_max': float, 'residuals': array}
   - RANSAC-style loop with similarity model; randomized minimal sets of 2
     matched pairs; score = count of residuals < sigma_max (MAGSAC-flavored
     sigma-marginalized spirit: tolerate a soft sigma rather than hard 0)
   - final M refit on ALL inliers via fit_similarity (progressive polishing:
     2 polish rounds or until inlier set stops changing)
   - rng: use numpy default_rng(seed=None) unless seed param given
4. function gsd_normalize(pts, gsd) -> pts scaled by 1/gsd (so thresholds are
   GSD-normalized, per canon D8 9.2: sub-pixel targets in SOURCE pixels)
5. function rms_px(residuals[inliers]) helper: sqrt(mean(res^2))
6. unit gate (tests/test_consensus.py): on 3 in-test synthetic fixtures:
   (a) 200 random pts on similarity-transformed grid + 10% gross outliers:
       recovered (s, theta) within 1% of truth, tx/ty within 0.5 px,
       inlier_ratio >= 0.88, RMS(inlier residual) < 0.15 px
   (b) pure-translation pair (s=1, theta=0): fit recovers identity exactly
       (|s-1| < 1e-9); magsac inliers = 100%
   (c) 60-degree rotation + 1.3 scale: recovery within same tolerances
7. ALSO expose fit_homography(src, dst) (via DLT, numpy only; used later by
   tier-4 eval; document that similarity remains the default verdict model)

## out of scope
- NFA / a-contrario validation (unit 9.x later; keep interface TODO)
- conformal ABSTAIN (unit 12)
- learned match filtering (milestone 2)
- dense optical flow (no)

## constraints
- numpy + stdlib only; no cv2 (no cv2.estimateAffinePartial2D!)
- reuse no other module internals beyond numpy; self-contained
- < 300 lines, docstring with unit id + gate name
- no em-dash, no AI-tell vocabulary
- ambiguity: if len(src_pts) < 2 -> raise ValueError loudly
- degenerate all-collinear points: raise ValueError (loud), do not return junk M

## definition of done
- tests/test_consensus.py: >= 12 asserts, 3 fixtures, all pass
- gate name: 'consensus recovery on 3 synthetic fixtures'
- one PROGRESS-LOG row appended
- no git commit (Hermes audits + commits)