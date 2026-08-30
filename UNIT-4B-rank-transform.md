# UNIT-4B TICKET - rank-transform normalizer (RIFT-style), the real radiometric fix

## why (evidence trail, from MINI-GATE-2 lie-hunt)
bilinear warp fix proved the chain: no-gain pair recovers s=1.2000 exact,
inliers 60%, rms 0.21 px. with 1.25x gain + 8 offset, matches collapse to
31% inliers -> ratio-norm (local mean/variance) insufficient for SIFT under
multiplier gain. corpus answer: RIFT2's rank/phase transform exists for
exactly this. implement the rank-transform path.

## spec (add to src/lunar_tie/photometric.py; do NOT touch other modules)
def rank_transform(img, bins=6):
    img float64. min-max scale to [0, N-1], N = 2**quantize_bits where
    quantize = number of STABLE intensity ranks. simplest correct version:
    uniform quantization of [0,255] into `quant_levels` bins (default 6),
    per-pixel bin index, returned SAME dtype float64, same shape.
    plus apply_rank(img, quantize_bins=6) -> returns rank_transform(img).

## gate (tests/test_photometric.py add 5 tests)
1. rank_transform shape/dtype/range: output in [0, quantize-1]
2. rank_transform gain invariance: rank(rt(img * 1.25 + 8)) == rank(rt(img))
   EXACTLY (quantization boundaries may differ only at exact bin edges
   with float noise -> allow 0.1% pixel-count tolerance, assert < 0.5%
   mismatched pixels)
3. rank_transform preserves edges: 2-rock synthetic, gradient location
   preserved (argmax of |grad| within 2 px)
4. existing ratio-norm tests still pass (non-breaking addition)
5. roundtrip determinism: rt applied twice == rt once

## scope
- photometric.py: +1 function (rank_transform) + quantize helper
- tests/test_photometric.py: +4-5 tests
- nothing else. no cv2. no em-dash. no commit.

## report
- files changed, test names, pass counts, and the exact mismatch rate for
  gate 2.