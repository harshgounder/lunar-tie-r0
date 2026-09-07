"""UNIT-10: pipeline CLI driver, the M1 capstone.

Composes units 1-9 behind ONE command-line entry point so a pair manifest
becomes a metrics panel + artifacts without any human gluing:

  python -m lunar_tie.pipeline run_pair <manifest.json> -o <outdir>

Manifest (JSON): {"a": path, "b": path, "tier_label": int, "provenance": str}
where the paths point at .npy float32 arrays loaded with mmap_mode='r'
(the tiler's own format proves memmap integration end to end).

The 10-step chain (load, mask, normalize, detect, match, consensus,
subpixel, coverage, conformal, panel) prints one progress line per step and
writes panel.json, ties.json and summary.txt under --outdir ONLY.

Honesty defaults (N9): manifest tier_label / provenance are OPTIONAL. A
manifest that omits them produces tier_label=0 and provenance="UNLABELED"
in the panel, never a silent tier 0 masquerading as labeled.

Exit codes: 0 on a green chain; nonzero with a loud stderr message on any
gate failure. numpy + stdlib only.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from lunar_tie.masking import compute_masks
from lunar_tie.photometric import normalize_ratio
from lunar_tie.detect import detect_keypoints_rdsift, match_descriptors
from lunar_tie.consensus import magsac_consensus, similarity_residuals
from lunar_tie.subpixel import refine_matches
from lunar_tie.coverage import select_uniform_ties
from lunar_tie.conformal import (
    conformal_gate,
    conformal_calibrate,
    metrics_panel,
    three_way_gate,
)

# fixed knobs of the M1 chain (ticket: exact, do not deviate)
MATCH_RATIO = 0.8
MAGSAC_ITERS = 2000
MAGSAC_SIGMA = 2.5
MAGSAC_SEED = 13
SUBPIXEL_HALF = 16
# detect contrast threshold lowered from the 0.03 default: the unit-10/MG2
# fixture's 12-value shadow band + 255-clipped outlier blocks carry weak DoG
# edges that 0.03 discards, leaving every band tie border-adjacent (F14 rule:
# within half=16 px of the border) so subpixel marks it invalid and the chain
# starves at step 7 (2 < 3 refinements). 0.02 keeps the block edges and gives
# the conformal holdout n >= 19 so q_hat is finite (k = ceil((n+1)*0.95) <= n
# needs n >= 19 at alpha 0.05).
CONTRAST_THRESHOLD = 0.02
CONF_ALPHA = 0.05
TIES_N_TARGET = 120
TIES_MIN_DISTANCE = 32.0
# conformal holdout: even-index post-refit ties calibrate BOTH gates, odd
# gated. Calibrating and gating on the same array is the self-referential
# trap (audit F2): a gate cannot fail on its own training quantile.
CONFORMAL_STRIDE = 2


def validate_tier(tier_label):
    """Validate a manifest tier value; return it unchanged or raise.

    Accepts only true JSON integers in {0} union {1..5}: bools, floats,
    strings and None raise PipelineError. Tier 0 is legal only through the
    N9 unlabeled path (tier_label omitted from the manifest); a manifest
    that explicitly writes 0 is a labeled UNLABELED claim and is rejected
    here to keep 0 out of the labeled tiers.
    """
    if isinstance(tier_label, bool) or not isinstance(tier_label, int):
        raise PipelineError(
            f"manifest tier_label must be a JSON integer in {{0}} union "
            f"{{1..5}}, got {tier_label!r}")
    if tier_label == 0 or tier_label in (1, 2, 3, 4, 5):
        return tier_label
    raise PipelineError(
        f"manifest tier_label must be a JSON integer in {{0}} union "
        f"{{1..5}}, got {tier_label!r}")


class PipelineError(RuntimeError):
    """Loud failure of any pipeline step (exits nonzero via CLI wrapper)."""


def _progress(step, name, detail):
    """One-line progress line per step: step number + name + key count."""
    print(f"[{step:2d}] {name}: {detail}", flush=True)


def load_manifest(path):
    """Read + validate the pair manifest; applies honesty defaults (N9)."""
    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise PipelineError(f"manifest not found: {manifest_path}")
    with open(manifest_path) as f:
        try:
            manifest = json.load(f)
        except json.JSONDecodeError as exc:
            raise PipelineError(f"manifest is not valid JSON: {exc}")
    for key in ("a", "b"):
        if key not in manifest or not isinstance(manifest[key], str):
            raise PipelineError(f"manifest missing string field '{key}'")
    # honesty default: never silently tier 0 a labeled-looking pair (N9)
    manifest.setdefault("tier_label", 0)
    manifest.setdefault("provenance", "UNLABELED")
    return manifest


def run_pair(manifest_path, outdir, verbose=False):
    """Run the 10-step chain for one pair manifest; return (panel, ties).

    Gates loudly (PipelineError) instead of exiting 0 silently on a red
    chain. All artifacts are written under outdir, which is created.
    """
    t0 = time.time()
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(manifest_path)
    # A3 (audit F3): loud tier validation BEFORE anything runs; the old
    # min(t, 1) or 1 laundering passed 9/3.7/"3"/true through silently.
    tier_label = validate_tier(manifest["tier_label"])
    provenance = str(manifest["provenance"])

    # ---- step 1: load (memmap, raw dtype preserved for masks) ---------------
    def _load_image(key):
        p = Path(manifest[key])
        if not p.is_file():
            raise PipelineError(
                f"missing image file for '{key}': {p} (from manifest"
                f" {manifest_path})")
        arr = np.load(p, mmap_mode="r")
        if arr.ndim != 2:
            raise PipelineError(f"'{key}' is not a 2D array: {p}")
        return arr

    imgA_raw = _load_image("a")
    imgB_raw = _load_image("b")
    H, W = imgA_raw.shape
    if imgB_raw.shape != imgA_raw.shape:
        raise PipelineError(
            f"image shapes differ: a={imgA_raw.shape} b={imgB_raw.shape}")
    _progress(1, "load", f"a={W}x{H} {imgA_raw.dtype} memmap")

    # ---- step 2: mask (on the RAW dtype BEFORE any float cast) --------------
    # C9 (commander round 3): masks must see the raw memmap dtype so the
    # saturation gate is the integer dtype max; a float64 cast first makes
    # _default_sat_limit return float64 max and saturation never fires.
    masksA = compute_masks(imgA_raw)
    masksB = compute_masks(imgB_raw)
    usableA = masksA["combined"]
    usableB = masksB["combined"]
    usable_pct = 100.0 * float(usableA.sum()) / imgA_raw.size
    _progress(2, "mask", f"usable={usable_pct:.1f}% (a)")

    # ---- step 3: normalize (masks APPLIED: nodata/shadow/saturated pixels
    #      are zeroed before detect so they never key or match) ---------------
    nA = normalize_ratio(np.asarray(imgA_raw, dtype=np.float64))
    nB = normalize_ratio(np.asarray(imgB_raw, dtype=np.float64))
    nA = np.where(usableA, nA, 0.0).astype(np.float32)
    nB = np.where(usableB, nB, 0.0).astype(np.float32)
    _progress(3, "normalize", "ratio-norm float32 [~ -1, 1], masks applied")

    # ---- step 4: detect ------------------------------------------------------
    kpA, dA = detect_keypoints_rdsift(nA, contrast_threshold=CONTRAST_THRESHOLD)
    kpB, dB = detect_keypoints_rdsift(nB, contrast_threshold=CONTRAST_THRESHOLD)
    _progress(4, "detect", f"A={len(kpA)} B={len(kpB)} keys")
    if len(kpA) == 0 or len(kpB) == 0:
        raise PipelineError("detector starved: zero keypoints on a side")

    # ---- step 5: match -------------------------------------------------------
    matches = match_descriptors(dA, dB, ratio=MATCH_RATIO)
    _progress(5, "match", f"{len(matches)} pairs (ratio {MATCH_RATIO})")
    if len(matches) < 3:
        raise PipelineError(
            f"too few matches to fit similarity: {len(matches)}")

    src = np.array([[kpA[i]["x"], kpA[i]["y"]] for i, _ in matches],
                   dtype=np.float64)
    dst = np.array([[kpB[j]["x"], kpB[j]["y"]] for _, j in matches],
                   dtype=np.float64)

    # ---- step 6: consensus ---------------------------------------------------
    cons = magsac_consensus(src, dst, min_samples=3, iters=MAGSAC_ITERS,
                            sigma_thr=MAGSAC_SIGMA, seed=MAGSAC_SEED)
    M = cons["M"]
    inlier_ratio_0 = cons["inlier_ratio"]
    _progress(6, "consensus",
              f"inliers={int(cons['inliers'].sum())}/{len(matches)}"
              f" ratio={inlier_ratio_0:.3f}")
    if int(cons["inliers"].sum()) < 2:
        raise PipelineError("consensus collapsed: fewer than 2 inliers")

    # ---- step 7: subpixel ------------------------------------------------------
    # refine_matches contract: dst_pts are the M-warped src positions; the
    # measured delta is applied on imgB relative to those warped positions
    # and refined dst = dst_pts + delta.
    inliers0 = cons["inliers"]
    src_in = src[inliers0]
    dst_in = dst[inliers0]
    warped0 = src_in @ M[:, :2].T + M[:, 2]
    ref = refine_matches(imgA_raw, imgB_raw, src_in, warped0,
                         half=SUBPIXEL_HALF)
    valid = ref["valid"]
    # C6a: a consensus floor of 2 is not enough to re-fit + calibrate on;
    # fewer than 3 valid refinements is a loud chain failure, not an
    # unhandled numpy error from cons2's rng.choice(size=3).
    if int(valid.sum()) < 3:
        raise PipelineError(
            f"subpixel refinement kept too few ties: {int(valid.sum())} < 3")
    refined_src = ref["src"][valid]
    refined_dst = ref["dst"][valid]
    # re-fit via magsac ONE more time on the refined coords
    cons2 = magsac_consensus(refined_src, refined_dst, min_samples=3,
                             iters=MAGSAC_ITERS, sigma_thr=MAGSAC_SIGMA,
                             seed=MAGSAC_SEED)
    M2 = cons2["M"]
    # C6b: fewer than 3 refit inliers cannot support coverage + conformal
    # (a 2-point similarity has zero residuals and voids every gate).
    if int(cons2["inliers"].sum()) < 3:
        raise PipelineError(
            "post-refit consensus collapsed: fewer than 3 inliers")
    src_f = refined_src[cons2["inliers"]]
    dst_f = refined_dst[cons2["inliers"]]
    weights_f = ref["peak_vals"][valid][cons2["inliers"]]
    _progress(7, "subpixel",
              f"refined={len(refined_src)} valid={int(valid.sum())}"
              f" inliers={int(cons2['inliers'].sum())}")

    # ---- step 8: coverage ------------------------------------------------------
    sel = select_uniform_ties(src_f, dst_f, weights_f, M2, W, H,
                              n_target=TIES_N_TARGET,
                              min_distance=TIES_MIN_DISTANCE)
    tie_idx = sel["indices"]
    cov = sel["metrics"]
    if tie_idx.size == 0:
        raise PipelineError("coverage selection kept zero ties")
    ties_src = src_f[tie_idx]
    ties_dst = dst_f[tie_idx]
    _progress(8, "coverage",
              f"ties={tie_idx.size} occupancy={cov['occupancy_ratio']:.2f}"
              f" entropy={cov['coverage_entropy']:.2f}")

    # ---- step 9: conformal -----------------------------------------------------
    # final inlier residuals under the refit model M2
    resid_f = similarity_residuals(M2, src_f, dst_f)
    # A2 (audit F2) + C1: BOTH gates are calibrated on a held-out split
    # (even-index residuals) and gate the complement (odd-index). Passing
    # the same array as its own calibration set drives the accept fraction
    # to ~1 by construction: a comparison satisfiable in itself.
    cal_mask = np.arange(resid_f.size) % CONFORMAL_STRIDE == 0
    gate_mask = ~cal_mask
    resid_cal = resid_f[cal_mask]
    resid_test = resid_f[gate_mask]
    q_hat = conformal_calibrate(resid_cal, alpha=CONF_ALPHA)
    labels = conformal_gate(resid_test, q_hat)
    n_accept = int((labels == "ACCEPT").sum())
    tw_labels = three_way_gate(resid_test, resid_cal,
                               np.zeros(resid_cal.size, dtype=bool),
                               alpha=CONF_ALPHA)
    n3_accept = int((tw_labels == "ACCEPT").sum())
    n3_abstain = int((tw_labels == "ABSTAIN").sum())
    _progress(9, "conformal",
              f"q_hat={q_hat:.4f} accept={n_accept}/{resid_test.size} "
              f"3way={n3_accept}/{resid_test.size}"
              f" abstain={n3_abstain} reject={resid_test.size - n3_accept - n3_abstain}")

    # ---- step 10: panel + artifacts --------------------------------------------
    # A3 (audit F3): tier validated loudly at manifest load; no
    # min(t, 1) or 1 laundering here. The N9 unlabeled path (tier 0) keeps
    # the honest default stamp without passing 0 through the labeled-tier
    # validator.
    # A1 (audit F1): panel_mask is the REAL post-refit inlier mask (cons2
    # inliers over resid_f) intersected with the coverage selection, a real
    # measured subset: n_pairs counts every resid_f entry, n_valid counts
    # the coverage-selected ties, so inlier_ratio = ties / n_pairs is
    # measured, not the vacuous all-true 1.0.
    panel_mask = np.zeros(resid_f.size, dtype=bool)
    panel_mask[np.where(cons2["inliers"])[0][tie_idx]] = True
    panel = metrics_panel(resid_f, panel_mask, q_hat, tier_label=tier_label,
                          alpha=CONF_ALPHA, provenance_seed=MAGSAC_SEED,
                          extra={
                              "provenance_str": provenance,
                              "resid_f_size": int(resid_f.size),
                              "n_ties": int(tie_idx.size),
                              "occupancy_ratio": cov["occupancy_ratio"],
                              "occupied_cells": cov["occupied_cells"],
                              "coverage_entropy": cov["coverage_entropy"],
                              "quadrant_ok": cov["quadrant_ok"],
                              "nearest_neighbor_cv":
                                  cov["nearest_neighbor_cv"],
                              "manifest": str(manifest_path),
                          })
    panel["tier_label"] = tier_label
    # gate 4: the manifest provenance string propagates UNCHANGED under
    # 'provenance'; the structured dict from metrics_panel moves to
    # provenance_meta so both survive a JSON round-trip.
    if isinstance(panel.get("provenance"), dict):
        panel["provenance_meta"] = panel.pop("provenance")
    panel["provenance"] = provenance
    ties = [
        {"x_a": float(sx), "y_a": float(sy),
         "x_b": float(dx), "y_b": float(dy), "weight": float(w)}
        for (sx, sy), (dx, dy), w in zip(ties_src, ties_dst,
                                         weights_f[tie_idx])
    ]

    panel_export = dict(panel)
    panel_export["chain_seconds"] = round(time.time() - t0, 3)
    with open(outdir / "panel.json", "w") as f:
        json.dump(panel_export, f, indent=2)
    with open(outdir / "ties.json", "w") as f:
        json.dump(ties, f, indent=2)
    with open(outdir / "summary.txt", "w") as f:
        f.write(_summary_text(panel_export, ties, cov))

    _progress(10, "panel",
              f"inlier_ratio={panel['inlier_ratio']:.3f}"
              f" rms_px={panel['rms_px']:.3f} tier={tier_label}"
              f" -> {outdir}/panel.json")
    print(f"    chain time: {panel_export['chain_seconds']:.1f}s",
          flush=True)
    return panel_export, ties


def _summary_text(panel, ties, cov):
    """Human-readable one-page summary.txt body."""
    lines = [
        "LUNAR-TIE M1 pipeline summary",
        "=============================",
        f"n_pairs:       {panel['n_pairs']}",
        f"n_valid:       {panel['n_valid']}",
        f"inlier_ratio:  {panel['inlier_ratio']:.4f}",
        f"rms_px:        {panel['rms_px']:.4f}",
        f"rms_subpx:     {panel['rms_subpx']:.4f}",
        f"q_hat:         {panel['q_hat']:.4f}",
        f"ece_proxy:     {panel['ece_proxy']:.4f}",
        f"tier_label:    {panel['tier_label']}",
        f"provenance:    {panel.get('provenance_str', 'UNLABELED')}",
        f"ties:          {len(ties)}",
        f"occupancy:     {cov['occupancy_ratio']:.3f}"
        f" ({cov['occupied_cells']}/{8 * 8} cells)",
        f"quadrant_ok:   {cov['quadrant_ok']}",
        f"gate_name:     {panel['gate_name']}",
        "",
        "coverage gates (accept/abstain/reject): "
        f"{panel['coverage_gates']}",
    ]
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="lunar_tie.pipeline",
        description="M1 tie-point pipeline: manifest -> panel + ties")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_run = sub.add_parser("run_pair",
                           help="run the 10-step chain for one pair manifest")
    p_run.add_argument("manifest", help="pair manifest JSON")
    p_run.add_argument("--outdir", "-o", required=True,
                       help="output directory (created; artifacts land here)")
    p_run.add_argument("--verbose", "-v", action="store_true",
                       help="verbose progress output")
    args = parser.parse_args(argv)

    if args.cmd == "run_pair":
        try:
            run_pair(args.manifest, args.outdir, verbose=args.verbose)
        except PipelineError as exc:
            # loud stderr failure, nonzero exit, no traceback dump
            print(f"PIPELINE FAIL: {exc}", file=sys.stderr)
            return 1
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())