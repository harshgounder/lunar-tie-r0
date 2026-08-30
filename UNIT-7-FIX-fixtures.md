# UNIT-7 FIX TICKET - subpixel fixtures quantized + thresholds (found by Hermes lie-hunt)

fix exactly these, nothing else in tests/test_subpixel.py:

1. _shift() helper: remove the final .astype(np.uint8) (it QUANTIZES sub-pixel
   FFT shifts: 1.5 px becomes 1 or 2). Return float64 sub-pixel-shifted image.
   Keep phase-ramp math identical otherwise. Update _synthetic() consumers so
   fixtures pass float64 images through (phase_shift itself accepts float).
   Where a uint8 image is genuinely needed for a test (e.g. dtype variants),
   cast at the LAST moment before the assertion, never before the shift.
2. Adjust ONLY the peak_val thresholds to the calibrated honest bars (the
   clean-room measurement: pv=0.42 clean-windowed, 0.37 without window, on a
   true 1.5 px translation with clean data):
   - test_phase_shift_known_shift: pv >= 0.30 (was 0.5)
   - test_fixture_a_shift_recovery: pv >= 0.30 (was 0.6) AND keep the
     0.25 px error tolerance (the module MEETS it; dx error was 0.026)
   - test_fixture_c_noise_sanity: pv >= 0.25 (was 0.4), error tolerance 0.35
   - test_fixture_b_translation_ladder: paper tolerances stand (0.2 px)
   Add a one-line comment above each adjusted threshold: the measured value
   that calibrated it (honesty trail).
3. Do NOT touch src/lunar_tie/subpixel.py (its core is verified correct:
   integer-shift repro got (+4.009, +2.994) on a true (+4, +3)).
4. Rerun the FULL test suite (every test_*.py) and report pass counts.

No git commit. No em dashes. numpy+stdlib only.