# UNIT-8 FIX TICKET - negative-coordinate point must raise loudly (Hermes lie-hunt gate N3)

fix exactly one thing:

1. src/lunar_tie/coverage.py _cell_indices(): currently np.clip silently folds
   out-of-range points (negative coords, x >= W, y >= H) into edge cells. A
   point OUTSIDE the image is a caller bug: it must raise loudly, not lie.
   Change: before clipping, check finite + range:
     if pts.size == 0: return empty int array
     if not np.isfinite(pts).all(): raise ValueError('non-finite point in coverage metrics')
     if (pts[:, 0] < 0).any() or (pts[:, 1] < 0).any() or (pts[:, 0] >= W).any() or (pts[:, 1] >= H).any():
         raise ValueError('point outside image bounds in coverage metrics')
   (callers in select_uniform_ties already filter in-image before calling,
   so this only makes contract violations loud - verify that path still green.)
2. ADD one test to tests/test_coverage.py: test_out_of_bound_point_raises_loud
   (a point at x=-5 raises ValueError).
3. Rerun FULL suite (all test files). Report pass counts.

No commit. No em dashes. numpy+stdlib only.