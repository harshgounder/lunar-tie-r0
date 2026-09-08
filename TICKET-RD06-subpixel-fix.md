TICKET-RD06 (SUBPIXEL PEAK-WINDOW + SNR GATE): the accuracy ceiling fix (audit A3+A4)
AUTHOR: hermes | IMPL: opencode glm-5.3-flash | VERIFY: hermes

## THE BUGS (two, both in subpixel.py phase_shift)
A3: peak_val normalization collapses with shift: pv(0,0)=1.0, pv(4,4)=0.087.
    the >= 0.2 validity gate REJECTS correct refinements at shifts >= 3px.
    THIS caused first-light valid=0 (the root cause).
A4: integer-shift lobes: at true shifts of 4-8px the upsampled corr surface
    has side-lobes that beat the main peak (measured: +8,+8 returned
    (-5.06,+4.50) pv=0.14). the global argmax jumps to an alias.

## FIX: edit src/lunar_tie/subpixel.py phase_shift signature
    phase_shift(patch_a, patch_b, upsample=16, max_shift=None):

    1. move the peak search INSIDE phase_shift: compute the full upsampled
       correlation surface as now, then find the argmax RESTRICTED to
       [-max_shift, +max_shift] around the surface center (index
       center = surface_size // 2). if max_shift is None use the full
       surface (backward compat).
    2. add snr to the return: snr = peak_val / median(corr_surface).
       a correct phase correlation has snr >> 5; an alias or noise
       floor has snr ~ 1-2. the snr is shift-invariant (unlike
       peak_val which normalizes by the zero-shift response).
    3. keep peak_val in the return for provenance.

    return dict gains 'snr': float (and the tuple form stays for compat).
    callers: pipeline.py reads the valid mask from phase_shift via
    refine_matches; update refine_matches to pass max_shift through and
    use snr instead of peak_val for the validity gate.

## TESTS (tests/test_subpixel.py additions, failing first)
- test_integer_shift_inside_window: patch shifted by (4,4): phase_shift
  returns (4.0+-0.1, 4.0+-0.1) with snr > 5 (currently FAILS: pv 0.087)
- test_fractional_shift_accuracy: shift (0.5,0.5): err < 0.05 (was 0.011 OK)
- test_snr_correct_match: snr > 5 on a correct shift (any magnitude
  within the search window)
- test_snr_uncorrelated: snr < 3 on noise-vs-noise (must reject)
- test_max_shift_enforced: max_shift=5 on a true 8px shift: the result
  is the best peak inside [-5,5], NOT the alias at 8px
- test_peak_val_still_recorded: peak_val is in the return dict

## PIPELINE WIRING (pipeline.py step 7)
refine_matches passes max_shift=SUBPIXEL_MAX_SHIFT (new constant, default
8.0) and uses the snr field: valid = snr >= SNR_GATE (new constant, default
5.0). the peak_val >= 0.2 gate is REMOVED. panel.json gains "snr_gate"
and the per-tie snr values.

## TESTS (tests/test_fix_round_a.py additions)
- test_real_shift_refinement: consensus inliers at shifts 3-8px now
  produce valid >= 3 (was valid=0: the first-light failure)
- test_panel_has_snr_gate

## DEFINITION OF DONE
pytest full suite green (187 baseline + 8 new). no em dashes.
commit: "TICKET-RD06: subpixel peak window + SNR gate (audit A3+A4)"