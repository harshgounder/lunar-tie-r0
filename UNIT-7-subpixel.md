# UNIT-7 TICKET - sub-pixel refinement: parabola + phase correlation (D4 8.1-8.3 / P6.m1)

Assignment for opencode. Implement `src/lunar_tie/subpixel.py` exactly to this spec.

## context
UNIT-6 consensus gives a similarity M with inliers at integer-px accuracy. The
acceptance contract needs sub-pixel. R0 provides a two-method refiner:
(1) parabola refinement of each keypoint from the DoG neighborhood scores
    (dx, dy from the 3-point quadratic around the extremum: opencode's
    detect.py already computes offsets internally; HERE we refit at matched-
    pair level using local patch correlation)
(2) phase correlation on a small patch around each match (upsampled FFT peak
    -> sub-pixel dx, dy); robust for translation-like local displacement after
    the global similarity warp

## requirements (exact, no scope creep)
1. function extract_patch(img, x, y, half=16) -> np.ndarray float32 patch
   centered at (x, y), with 'symmetric' pad if the window exits the image;
   (x, y) may be fractional
2. function bilinear_sample(img, x, y) -> float (single value at fractional
   coordinate; boundary clamp)
3. function phase_shift(patchA, patchB, upsample=16) -> (dx, dy, peak_val):
   zero-mean both patches, Hann window them, cross-power spectrum
   F_A * conj(F_B) / |.|, IFFT, locate peak integer + quadratic parabola fit
   around the peak IN THE UPSAMPLED grid to get sub-pixel dx, dy; peak_val =
   normalized peak height [0,1] (confidence proxy)
4. function refine_matches(imgA, imgB, src_pts, dst_pts, half=16,
   upsample=16, max_phase=8.0) -> refined dict:
   {'src': fractional pts after refinement, 'dst': ..., 'deltas': (N,2)
   applied shifts, 'peak_vals': (N,) confidences, 'valid': bool mask
   where |delta| <= max_phase and peak_val >= 0.2}
5. navigation contract: deltas are measured on imgB relative to the
   similarity-warped position of src (i.e., first apply M to src_pts, then
   phase-shift). Return final refined dst positions as warpped_src + delta.
   Document this composition EXACTLY in the docstring (test asserts it)
6. unit gate (tests/test_subpixel.py) on 3 in-test synthetic fixtures:
   (a) imgB = imgA shifted by (+3.5, -2.25) px: for >= 80% of sampled
       patches, |measured - true| <= 0.25 px and peak_val >= 0.6
   (b) sub-pixel translation tolerance ladder: true shifts 0.25/0.5/1.0 px
       all recovered within 0.2 px error
   (c) noise sanity: + Gaussian noise sigma=5 on uint8-scale; error grows
       but stays <= 0.35 px and peak_val >= 0.4 for >= 70% of pairs
7. expose rms_subpx(errors) helper: sqrt(mean(e^2)) over the valid mask

## out of scope
- ECC / Lucas-Kanade intensity refinement (later ablation; phase suffices R0)
- optical flow fields (no)
- SPICE geometry residuals (2.x)

## constraints
- numpy + stdlib only
- do NOT modify any existing module (5 exist: pds3label, masking, tiling,
  photometric, detect, consensus)
- < 300 lines, docstring with unit id + gate
- no em-dash, no AI-tell vocabulary
- ambiguity: patch fully outside image -> valid=False for that pair, no crash

## definition of done
- tests/test_subpixel.py: >= 12 asserts, 3 fixtures, pass
- gate name: 'subpixel recovery on 3 synthetic fixtures'
- one PROGRESS-LOG row
- no git commit (Hermes audits + commits)