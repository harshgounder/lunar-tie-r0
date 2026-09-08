TICKET-RD05 (COMMON-GSD RESAMPLE): the cross-resolution shape gate fix (audit L2)
AUTHOR: hermes (glm planning) | IMPL: opencode glm-5.3-flash | VERIFY: hermes

## THE BUG
crop_pair produces different-size crops for different-resolution pair sides
(OHRC 0.25m vs TMC-2 5m). run_pair requires exact same shape. no resampling
exists. THIS blocked the real pair-A registration (first-light).

## FIX: new file src/lunar_tie/resample.py (numpy + stdlib only)

1. area_average(img, factor) -> float32 ndarray
   block-mean decimation by integer factor. crop img to a multiple of
   factor FIRST (documented: the crop is the price of integer decimation),
   then reshape to (-1, factor, -1, factor) and mean over the block axes.
   for factor=1 return img unchanged.

2. gauss_pyr_down(img, factor) -> float32 ndarray
   gaussian_blur(img, sigma=factor*0.5) then area_average by factor.
   the energy-preserving variant (DR1: ablation).

3. lanczos3_decimate(img, factor) -> float32 ndarray
   Lanczos-3 kernel (sinc windowed by sinc(x/3)) applied as separable
   1D convolution + decimate by factor. edge handling: reflect pad.
   NOTE: this is the expensive variant; used only in the bake-off.

4. match_shapes_to_min(a, b, gsd_ratio) -> (a_rs, b_rs, factor)
   given the two crops + the GSD ratio (e.g. 20 for OHRC:TMC-2), compute
   the integer decimation for the finer side to match the coarser side,
   resample the finer side, then center-crop BOTH to the smaller common
   shape. returns both arrays at identical (H, W) float32.

## TESTS (tests/test_resample.py, failing first)
- test_area_average_exact: 4x4 known -> 2x2 block-mean matches manually
- test_area_average_nondivisible: 7x7 by factor 2 crops to 6x6 first
- test_gauss_pyr_differs: gauss_pyr_down != area_average on the same img
- test_match_shapes_identical: two different-size inputs -> same shape out
- test_match_shapes_values: the center-crop picks the central overlap
- test_memory_bounded: peak RSS < 3x input size on a 2000x2000 input
  (reuse the F8 ru_maxrss pattern from test_tiling.py)

## DEFINITION OF DONE
pytest full suite green (187 baseline + 6 new). no em dashes. numpy+stdlib
only. commit message: "TICKET-RD05: common-GSD resample module (audit L2)"