# UNIT-10 TICKET - pipeline CLI driver (D12 13.x / the M1 capstone)

Assignment for opencode. Implement `src/lunar_tie/pipeline.py` exactly to
this spec. This is the last build-gated unit of milestone 1.

## context
Units 1-9 are landed and green (113 tests + 4B's 5 = 117... see
PROGRESS-LOG for the exact current counts). They exist as isolated modules.
M1's remaining job: compose them behind ONE command-line entry point so a
pair manifest becomes a metrics panel + artifacts without any human
gluing. This is the "product" face of M1.

## contract (exact, do not deviate)

CLI (argparse, module `src/lunar_tie/pipeline.py`, guard `if __name__ == "__main__"`):
```
python -m lunar_tie.pipeline run_pair <manifest.json> --outdir <dir> [--verbose]
```
- `run_pair` is the only subcommand in M1.
- manifest.json (JSON): {"a": "<path>", "b": "<path>", "tier_label": int,
  "provenance": "<str>"}; paths are .npy float32 arrays (the tiler's own
  format proves memmap integration).
- steps in order, printing a one-line progress line per step (step number
  + name + key count / metric), exactly this order:
   1 load (np.load, mmap_mode='r'), float64 cast
   2 mask: compute_masks per image; combined usable boolean
   3 normalize: normalize_ratio float64
   4 detect: detect_keypoints_rdsift on each normalized image
   5 match: match_descriptors(dA, dB, ratio=0.8)
   6 consensus: magsac_consensus(src, dst, min_samples=3, iters=2000, sigma_thr=2.5, seed=13)
   7 subpixel: refine_matches on inlier matches (half=16), drop invalid,
     re-fit via magsac ONE more time on refined coords
   8 coverage: select_uniform_ties on the refined set
   9 conformal: conformal_calibrate on final inlier residuals (alpha=0.05)
     + conformal_gate + three_way_gate
  10 panel: metrics_panel(tier_label from manifest, provenance) + write
     panel.json, ties.json (list of {x_a,y_a,x_b,y_b,weight}), summary.txt
     (human-readable one-page)

## hard rules
- import ONLY existing module functions; if a name differs from this
  ticket, READ the module and use its real name; do not modify other
  modules to fit the ticket; if a needed function truly does not exist,
  STOP and report instead of inventing.
- outputs go ONLY under --outdir (create it).
- panel.json MUST carry tier_label + provenance from the manifest (N9:
  works with zero Ch-2 products).
- numpy + stdlib only. no cv2/scipy. no network. no em-dashes.
- exit code 0 on success; nonzero with a loud stderr message on any
  gate failure (never exit 0 silently on a red chain).

## gate (tests/test_pipeline.py, 6 tests)
1. run_pair on a 256x256 synthetic pair (translation dx=4 dy=3, no scale)
   produces panel.json + ties.json + summary.txt on disk, exit code 0
2. panel.json parses; keys superset of {inlier_ratio, rms_px, q_hat,
   occupancy_ratio, tier_label, provenance}
3. recovered translation within 0.5 px of truth (median of ties after
   subpixel), i.e. quality bar, not just "runs"
4. manifest tier_label/provenance PROPAGATE into panel.json unchanged
5. missing file path in manifest -> stderr message + nonzero exit (no
   traceback dump needed, loud is enough)
6. run_pair with tier_label omitted -> defaults tier_label=0 AND marks
   provenance "UNLABELED" in panel (honesty default, never silently tier 0)

## scope
- create: src/lunar_tie/pipeline.py, tests/test_pipeline.py
- everything else: READ ONLY.

## report
files created, test names, pass counts, the panel.json from the gate
pair, and chain time.