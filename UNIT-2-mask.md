# UNIT-2 TICKET - valid/nodata/shadow triple mask (D1 1.3 / P1.m3)

Assignment for opencode. Implement `src/lunar_tie/masking.py` exactly to this spec.

## context
LUNAR-TIE R0 needs a triple mask applied to every image before matching:
(1) VALID mask   = pixels inside the actual image footprint (nonzero data area)
(2) NODATA mask  = pixels whose value == the declared nodata constant (0 for
    many lunar products) OR saturated (>= declared max)
(3) SHADOW mask  = pixels likely in cast shadow or permanent shadow. In R0 we
    DO NOT have SPICE: approximate geometric shadow via a luminance threshold
    Otsu-derived per-tile, i.e. pixels below (Otsu threshold) in the LOW class.
    SPICE-based shadow geometry is a LATER rigor unit (2.x). The mask module
    must expose the interface so Otsu can be swapped for a SPICE mask later.

## requirements (exact, no scope creep)
1. function compute_masks(img, nodata_value=0, sat_value=None) ->
   returns dict of boolean numpy arrays: {'valid': ..., 'nodata': ...,
   'shadow': ..., 'combined': ...} where combined = valid AND NOT nodata AND
   NOT shadow
2. img is a 2D numpy array (any uint8/uint16/float32); nodata_value optional
   (default 0); if None, skip nodata masking
3. valid mask: True where img is inside [nodata+1, sat_limit-1]; sat_limit
   optional param (default = dtype max for the img dtype)
4. shadow via Otsu threshold (implement Otsu yourself with numpy histogram,
   256 bins for uint8; for uint16 downsample bincount adaptively); shadow
   mask = img <= otsu_low_threshold (the darker class lower edge)
5. expose otsu_threshold(img) -> float as a public function (later units +
   tests need it directly)
6. Return also a summary dict: summary_masks(img, ...) -> {'valid_pct': ...,
   'nodata_pct': ..., 'shadow_pct': ..., 'usable_pct': ...} floats 0-100
7. unit gate (tests/test_masking.py): on 3 synthetic fixtures you author
   (a) uniform bright image (no shadows, no nodata): usable_pct ~ 100
   (b) half dark half bright: shadow mask covers the dark half, usable ~ 50
   (c) image with a nodata band of exactly 0 rows AND a saturated band of 255s:
       nodata and valid masks behave per spec, usable_pct excludes both
8. Also expose save/load mask functions (npz write/read) so the mask can be
   persisted alongside tiles in later units; round-trip must be exact.

## out of scope
- SPICE kernel geometry (later unit 2.7)
- crater detection (D6)
- learned shadow segmentation (would be milestone 2)
- color images (R0 is single-band PAN only)

## constraints
- numpy + stdlib only (no cv2, no skimage, no pandas)
- pds3label.py from UNIT-1 exists; do NOT modify it
- < 300 lines module, docstring states unit id + gate name
- no em-dash, no AI-tell vocabulary
- ambiguity policy: if the Otsu threshold on flat images is undefined
  (single-valued image), return threshold = None and shadow mask = all False
  (nothing provably shadow), do not crash

## definition of done
- tests/test_masking.py: >= 9 asserts, 3 fixtures, all pass
- gate name: 'triple mask on 3 synthetic fixtures'
- PROGRESS-LOG.md: exactly one appended row
- no git commit (Hermes audits and commits)