# UNIT-8 TICKET - coverage gates + ANMS uniform selection (D9 10.1-10.4 / P7.m1-m2)

Assignment for opencode. Implement `src/lunar_tie/coverage.py` exactly to this spec.

## context
A tie set can have low RMSE but be clustered in one crater = globally wrong (wall
U). The pipeline needs (a) coverage METRICS per the canon gates, (b) uniform
SELECTION (ANMS + grid quotas) that keeps the best-scoring points while forcing
spatial spread. Canon gates (from PROBLEM-STATEMENT-CANONICAL + brief): 8x8 grid
occupancy >= 60% (proposed; PS wants uniform density), plus quadrant checks +
coverage entropy. R0 works on points only (no raster needed).

## requirements (exact, no scope creep)
1. function grid_occupancy(pts, W, H, grid_n=8) -> (occupancy_ratio,
   occupied_cells) : pts Nx2 (x, y); divide image into grid_n x grid_n cells;
   return occupied cell count / total
2. function coverage_entropy(pts, W, H, grid_n=8) -> float in [0,1]: Shannon
   entropy of the count distribution over cells, normalized by log(total_cells)
3. function quadrant_check(pts, W, H) -> bool: every quadrant has >= 1 point
4. function coverage_metrics(pts, W, H, grid_n=8) -> dict combining all three +
   nearest_neighbor_cv(pts) (coefficient of variation of 1-NN distances; the
   canon "spread evenness" proxy; implement with numpy only, O(N^2) fine for
   R0 sizes, but note it)
5. function anms_select(scores, pts, n_target, min_distance) -> indices of
   selected points: greedy ANMS variant - sort by score desc; greedily accept a
   point if it is >= min_distance (px) from every already-accepted point; stop
   at n_target or exhaustion; return also the acceptance-order guarantee that
   high-score points inside locked cells still lose to spread (i.e. selection
   is NOT purely score-greedy: alternate between best-score-in-an-uncovered-
   cell and best-global-score so that every cell gets a chance first)
   - document the exact policy: ROUND 1 = one best point per occupied cell
     (by score), ROUND 2 = fill remaining slots by pure score among remaining
     points subject to min_distance
6. function select_uniform_ties(src_pts, dst_pts, match_scores, M, W, H,
   n_target=120, min_distance=32.0, grid_n=8) -> dict {'indices', 'metrics'}
   : warp src via M, keep in-image points, run the two-round ANMS on WARped
   source positions, return indices + final coverage_metrics dict on the
   selected set
7. unit gate (tests/test_coverage.py): 3 in-test fixtures:
   (a) 500 pts all inside one cell: occupancy <= 1/64 + eps, entropy < 0.2,
       quadrant check False
   (b) uniform grid of 64 pts across full image: occupancy == 1.0,
       entropy > 0.95, quadrants True
   (c) selection task: 400 clustered pts + 100 spread pts, scores random:
       anms_select(n_target=60, min_distance=40) selected set must have
       occupancy >= 0.55 on 8x8 (proves the two-round policy forces spread)
8. expose apply_similarity(M, pts) helper (reuse formula from consensus M).

## out of scope
- hull metrics (unit 12 later)
- coverage-entropy LOSS for training (milestone 2, R2 gold)
- weighted entropy variants (later)

## constraints
- numpy + stdlib only
- do NOT modify existing modules (7 exist)
- < 300 lines, docstring unit id + gate
- no em-dash, no AI-tell vocabulary
- ambiguity: pts empty -> all metrics 0.0, selection returns empty indices;
  n_target > n_pts -> return all, no crash

## definition of done
- tests/test_coverage.py: >= 12 asserts, 3 fixtures, pass
- gate name: 'coverage gates + uniform selection on 3 synthetic fixtures'
- one PROGRESS-LOG row
- no git commit (Hermes audits + commits)