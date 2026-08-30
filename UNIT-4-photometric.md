# UNIT-4 TICKET - photometric ratio normalization (P3.m1/D3 / P4.m1)

Assignment for opencode. Implement `src/lunar_tie/photometric.py` exactly to this spec.

## context
LUNAR-TIE must register imagery taken at DIFFERENT sun angles. OHRC (0.25 m, low
sun) vs TMC-2 (5 m, different hour) vs IIRS have wildly different radiometry.
In R0 the classical matcher lives or dies on normalization. The R0 method is
the "ratio-of-convolved-pairs" ratio-architecture family (from wave redo-ab-wb +
P3.m1) implemented on a per-tile basis: normalize luminance + local contrast so
two sun-angle views of the same terrain produce similar normalized images.

## requirements (exact, no scope creep)
1. function normalize_ratio(img, sigma=8.0, eps=1e-6) -> float32 ndarray,
   output SAME shape as input, values roughly in [-1, 1]:
   L = img cast float; mu = gaussian blur(L, sigma); num = L - mu;
   den = sqrt(gaussian blur(L^2, sigma) - mu^2) + eps; out = num / den
   (this is the locally-normalized image, classic "ratio norm": zero mean +
   unit variance under a local Gaussian window)
2. expose gaussian_blur(img, sigma) -> ndarray; implement separable convolution
   yourself with numpy (NO scipy, NO cv2). kernel radius = ceil(3*sigma);
   handle odd kernel size; rows then cols (separable) for speed
3. memory discipline: do NOT allocate more than 2 temporaries of img.size;
   constant-space separable conv (docstring must state the invariant)
4. function normalize_pair(imgA, imgB, sigma=8.0) -> (nA, nB) convenience:
   normalize both with the SAME sigma (cross-view pairs need a shared scale)
5. expose correlation_preview(nA, nB) -> float: normalized cross-correlation
   (zero-mean) on the two full images for a quick sanity gate in tests (this
   is NOT the full matcher; just verifies normalization improves alignment)
6. edge handling: reflect padding at image borders (np.pad mode 'symmetric');
   document this choice in the docstring
7. unit gate (tests/test_photometric.py): on 3 in-test synthetic fixtures
   (a) same terrain, different global gain + offset (imgB = 1.8*imgA + 25):
       normalized outputs must match closely (mean abs diff < 0.05 after
       normalization; correlation_preview > 0.95)
   (b) flat-constant image: variance 0 -> den ~ eps, output must NOT blow up
       (all finite, |out| bounded by say < 10)
   (c) realistic two-peak histogram (sunlit/shadow bimodal): output preserves
       the structures (normalized cross-correlation with itself > 0.9 after
       second normalization; no NaN, no Inf anywhere)
8. ALSO add shift-invariance smoke: normalized image shifted by 1 px must have
   corr_preview with itself > 0.90 (This validates the normalization doesn't
   amplify single-pixel aliasing).

## out of scope
- radiance factor / Hapke physics (unit 3.x, later)
- cGAN learned normalization (E8, milestone 2+)
- color/multispectral (R0 is PAN)
- CLAHE (a different normalization family, later ablation)

## constraints
- numpy + stdlib only (explicitly NO scipy.ndimage, NO cv2)
- do NOT modify pds3label.py, masking.py, tiling.py
- < 250 lines, docstring states unit id + gate
- no em-dash, no AI-tell vocabulary
- ambiguity: if sigma <= 0 raise ValueError (loud)

## definition of done
- tests/test_photometric.py: >= 10 asserts, all pass
- gate name: 'photometric ratio on 3 synthetic fixtures'
- one PROGRESS-LOG appended row
- no git commit (Hermes audits + commits)