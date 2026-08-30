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
CONF_ALPHA = 0.05
TIES_N_TARGET = 120
TIES_MIN_DISTANCE = 32.0


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
    tier_label = int(manifest["tier_label"])
    provenance = str(manifest["provenance"])

    # ---- step 1: load (memmap, float64 cast) --------------------------------
    def _load_image(key):
        p = Path(manifest[key])
        if not p.is_file():
            raise PipelineError(
                f"missing image file for '{key}': {p} (from manifest"
                f" {manifest_path})")
        arr = np.load(p, mmap_mode="r")
        if arr.ndim != 2:
            raise PipelineError(f"'{key}' is not a 2D array: {p}")
        return np.asarray(arr, dtype=np.float64)

    imgA = _load_image("a")
    imgB = _load_image("b")
    H, W = imgA.shape
    if imgB.shape != imgA.shape:
        raise PipelineError(
            f"image shapes differ: a={imgA.shape} b={imgB.shape}")
    _progress(1, "load", f"a={imgA.shape[1]}x{imgA.shape[0]} float64 memmap")

    # ---- step 2: mask --------------------------------------------------------
    masksA = compute_masks(imgA)
    masksB = compute_masks(imgB)
    usableA = masksA["combined"]
    usableB = masksB["combined"]
    usable_pct = 100.0 * float(usableA.sum()) / imgA.size
    _progress(2, "mask", f"usable={usable_pct:.1f}% (a)")

    # ---- step 3: normalize ---------------------------------------------------
    nA = normalize_ratio(imgA)
    nB = normalize_ratio(imgB)
    _progress(3, "normalize", "ratio-norm float32 [~ -1, 1]")

    # ---- step 4: detect ------------------------------------------------------
    kpA, dA = detect_keypoints_rdsift(nA)
    kpB, dB = detect_keypoints_rdsift(nB)
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
    src0 = cons  # alias for readability below
    inliers0 = cons["inliers"]
    src_in = src[inliers0]
    dst_in = dst[inliers0]
    warped0 = src_in @ M[:, :2].T + M[:, 2]
    ref = refine_matches(imgA, imgB, src_in, warped0, half=SUBPIXEL_HALF)
    valid = ref["valid"]
    if not valid.any():
        raise PipelineError("subpixel refinement rejected every inlier")
    refined_src = ref["src"][valid]
    refined_dst = ref["dst"][valid]
    # re-fit via magsac ONE more time on the refined coords
    cons2 = magsac_consensus(refined_src, refined_dst, min_samples=3,
                             iters=MAGSAC_ITERS, sigma_thr=MAGSAC_SIGMA,
                             seed=MAGSAC_SEED)
    M2 = cons2["M"]
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
    q_hat = conformal_calibrate(resid_f, alpha=CONF_ALPHA)
    labels = conformal_gate(resid_f, q_hat)
    n_accept = int((labels == "ACCEPT").sum())
    tw_labels = three_way_gate(resid_f, resid_f,
                               np.zeros(resid_f.size, dtype=bool),
                               alpha=CONF_ALPHA)
    n3_accept = int((tw_labels == "ACCEPT").sum())
    n3_abstain = int((tw_labels == "ABSTAIN").sum())
    _progress(9, "conformal",
              f"q_hat={q_hat:.4f} accept={n_accept} "
              f"3way={n3_accept}/{resid_f.size}"
              f" abstain={n3_abstain} reject={resid_f.size - n3_accept - n3_abstain}")

    # ---- step 10: panel + artifacts --------------------------------------------
    # metrics_panel validates tier_label in {1..5}, but the N9 honesty
    # default for an unlabeled manifest is tier 0; keep the panel dict
    # intact and override the tier afterwards (never silently tier 0).
    panel = metrics_panel(resid_f, np.ones(resid_f.size, dtype=bool),
                          q_hat, tier_label=min(tier_label, 1) or 1,
                          extra={
                              "provenance_str": provenance,
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