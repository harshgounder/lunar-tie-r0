# demo/ - runnable proof (2026-08-31)

## what's in here
- side_by_side.png: the two synthetic "sensor" frames (left = OHRC-like
  reference; right = same terrain, rotated 8 deg, scaled 1.15x, shifted
  (14.5, 9.5) px, gain-boosted 1.28x + offset 9, plus noise; emulating a
  different sensor pass).
- comparison_ours_vs_naive.png: THE money shot. left panel = naive SIFT +
  plain least-squares fit (what most teams ship first): NCC -0.021 =
  the registered image does not line up AT ALL. right panel = LUNAR-TIE R0
  chain (normalize -> SIFT -> ratio match -> MAGSAC): NCC 0.9725 over
  99.8% of the frame. bright yellow in the overlay = pixel agreement.
- registered_overlay.png: full-resolution red/green alignment check
  (red = registered A, green = B, yellow = agreement).
- comparison_stats.json: the numbers above, machine-readable.

## reproduce it yourself (the whole point)
git clone https://github.com/harshgounder/lunar-tie-r0.git
cd lunar-tie-r0
python -m venv .venv
.venv\Scripts\pip install pytest pillow numpy     # Windows
.venv/Scripts/pip install pytest pillow numpy     # Linux/Mac
.venv/Scripts/python -m pytest tests/ -q          # Windows: expect "119 passed"
.venv/bin/python -m pytest tests/ -q              # Linux/Mac

## honesty notes
- the pair is SYNTHETIC (generated lunar-like terrain: craters, boulders,
  regolith noise; real Chandrayaan-2 frames enter at M2 via ISSDC).
- the naive baseline here is re-run LIVE on the same pair in this run
  (not a strawman from another repo; it is plain SIFT + ratio 0.8 +
  all-point least-squares similarity, the common first implementation).
- the PUBLISHED external numbers (OHRC->LRO NAC: SIFT RMSE 0.69-2.01 px,
  SuperGlue 0.51-0.93 px) are from the literature (see sih-2026
  research/blind-spot-waves/G8) and were NOT re-run here; only our own
  chain and the in-repo naive baseline are re-runnable today.
- our headline sub-pixel numbers on the gate pair live in the README
  (s 1.2001 vs 1.2, theta 0.2000 rad, rms 0.146 px).