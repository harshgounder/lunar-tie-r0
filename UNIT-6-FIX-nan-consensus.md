# UNIT-6 FIX TICKET - consensus.py NaN silent-junk path (found by Hermes lie-hunt)

fix exactly these, nothing else:

1. In src/lunar_tie/consensus.py, fit_similarity (and any lstsq call) must
   GUARD against non-finite or degenerate input and raise ValueError LOUDLY
   instead of silently returning a junk matrix:
   - before solving: if not np.isfinite(src).all() or not np.isfinite(dst).all()
     raise ValueError('non-finite coordinates in consensus fit')
   - after solving: if not np.isfinite(M).all() raise ValueError('degenerate
     fit produced non-finite matrix')
2. In magsac_consensus loop, skip hypotheses whose residual array contains NaN:
   res = similarity_residuals(...); if not np.isfinite(res).all(): continue
3. ADD 2 tests to tests/test_consensus.py:
   - test_nan_input_raises_loud: magsac_consensus on 2 pts incl NaN raises
     ValueError (not junk)
   - test_fit_similarity_nan_raises: fit_similarity with a NaN coordinate
     raises ValueError
4. Rerun the full test suite (all test files) and report pass counts.

No git commit. No em dashes. numpy+stdlib only.