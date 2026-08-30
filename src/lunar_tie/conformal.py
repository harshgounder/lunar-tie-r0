"""UNIT-9 conformal ABSTAIN + metrics panel (D10 12.x / P8.m3 + P9.m4).
Gate: 'conformal gate on 3 synthetic fixtures'.

The pipeline must say "I do not know" loudly when it cannot prove a tie. R0
unit: split-conformal calibration on per-tie residuals produces a per-tie
ACCEPT/REJECT/ABSTAIN ladder, plus a metrics panel exporter that stamps every
number with a truth-tier label (the honesty contract IS the product).

Pure numpy + stdlib. No cv2, no scipy, no skimage.

Ambiguity policy: an empty scores_test returns an empty labels array without
crashing; conformal_calibrate raises ValueError on empty or non-finite input;
metrics_panel raises ValueError when tier_label is not one of {1,2,3,4,5}.
"""

import datetime
import json
import subprocess

import numpy as np

GATE_NAME = "conformal gate on 3 synthetic fixtures"
ALPHA_DEFAULT = 0.05
ABSTAIN_PENALTY_DEFAULT = 1.5
SEED_DEFAULT = 0


def _git_sha():
    """Current HEAD sha via git, or 'unknown' when unavailable."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
        return out.decode().strip()
    except Exception:
        return "unknown"


def conformal_calibrate(residuals_cal, alpha=0.05):
    """Split-conformal quantile of the calibration residuals.

    q_hat is the k-th order statistic with k = ceil((n+1)(1-alpha)), the
    finite-sample exact split-conformal quantile. residuals_cal are error px
    of held-out ties. Raises ValueError if empty or non-finite.
    """
    residuals_cal = np.asarray(residuals_cal, dtype=np.float64)
    if residuals_cal.size == 0:
        raise ValueError("residuals_cal is empty")
    if not np.isfinite(residuals_cal).all():
        raise ValueError("residuals_cal contains non-finite values")
    n = residuals_cal.size
    k = int(np.ceil((n + 1) * (1.0 - alpha)))
    k = min(k, n)
    sorted_r = np.sort(residuals_cal)
    return float(sorted_r[k - 1])


def conformal_gate(scores_test, q_hat, sigma_thr=None):
    """Binary ACCEPT/REJECT gate on test scores.

    ACCEPT iff score <= q_hat (and <= sigma_thr when provided); else REJECT.
    Empty scores_test returns an empty labels array.
    """
    scores_test = np.asarray(scores_test, dtype=np.float64)
    if scores_test.size == 0:
        return np.array([], dtype=object)
    if sigma_thr is not None:
        accept = (scores_test <= q_hat) & (scores_test <= sigma_thr)
    else:
        accept = scores_test <= q_hat
    return np.where(accept, "ACCEPT", "REJECT")


def three_way_gate(scores_test, scores_cal, outliers_flag_cal, alpha=0.05,
                   abstain_penalty=1.5):
    """Three-way ACCEPT/ABSTAIN/REJECT ladder on test scores.

    q_hat is calibrated on the inlier calibration residuals (outliers flagged
    by outliers_flag_cal are excluded; if none remain, all are used). The
    ladder is exact:

      score <= q_hat                       -> ACCEPT
      q_hat < score <= q_hat * penalty     -> ABSTAIN (flagged for review)
      score >  q_hat * penalty             -> REJECT

    ABSTAIN is neither trusted nor discarded. Empty scores_test returns an
    empty labels array.

    CONTRACT: scores_cal MUST be real calibration residuals; passing
    zeros/placeholder makes q_hat=0 and every pair REJECTs (no ABSTAIN band).
    """
    scores_cal = np.asarray(scores_cal, dtype=np.float64)
    outliers_flag_cal = np.asarray(outliers_flag_cal, dtype=bool)
    inliers = scores_cal[~outliers_flag_cal]
    if inliers.size == 0:
        inliers = scores_cal
    q_hat = conformal_calibrate(inliers, alpha)
    hi = q_hat * abstain_penalty

    scores_test = np.asarray(scores_test, dtype=np.float64)
    if scores_test.size == 0:
        return np.array([], dtype=object)
    labels = np.empty(scores_test.size, dtype=object)
    labels[scores_test <= q_hat] = "ACCEPT"
    labels[(scores_test > q_hat) & (scores_test <= hi)] = "ABSTAIN"
    labels[scores_test > hi] = "REJECT"
    return labels


def metrics_panel(residuals, inliers, q_hat, tier_label, extra=None):
    """JSON-serializable metrics panel stamped with a truth-tier label.

    ece_proxy is |inlier_ratio - (1 - alpha)| with alpha = ALPHA_DEFAULT. It
    is a PROXY only: it does not bin by predicted probability, so it is an
    honest stand-in, not a full ECE. tier_label must be one of {1,2,3,4,5};
    any other value raises ValueError.
    """
    if tier_label not in (1, 2, 3, 4, 5):
        raise ValueError("tier_label must be one of {1,2,3,4,5}")
    residuals = np.asarray(residuals, dtype=np.float64)
    inliers = np.asarray(inliers, dtype=bool)
    n_pairs = int(residuals.size)
    n_valid = int(inliers.sum())
    inlier_ratio = float(n_valid / n_pairs) if n_pairs else 0.0
    rms_px = float(np.sqrt(np.mean(residuals ** 2))) if n_pairs else 0.0
    sub = residuals[inliers]
    rms_subpx = float(np.sqrt(np.mean(sub ** 2))) if sub.size else 0.0
    ece_proxy = float(abs(inlier_ratio - (1.0 - ALPHA_DEFAULT)))

    hi = q_hat * ABSTAIN_PENALTY_DEFAULT
    if n_pairs:
        coverage_gates = {
            "accept": float(np.mean(residuals <= q_hat)),
            "abstain": float(np.mean(
                (residuals > q_hat) & (residuals <= hi))),
            "reject": float(np.mean(residuals > hi)),
        }
    else:
        coverage_gates = {"accept": 0.0, "abstain": 0.0, "reject": 0.0}

    panel = {
        "n_pairs": n_pairs,
        "n_valid": n_valid,
        "inlier_ratio": inlier_ratio,
        "rms_px": rms_px,
        "rms_subpx": rms_subpx,
        "q_hat": float(q_hat),
        "ece_proxy": ece_proxy,
        "coverage_gates": coverage_gates,
        "tier_label": int(tier_label),
        "gate_name": GATE_NAME,
        "provenance": {
            "date": datetime.date.today().isoformat(),
            "git_sha": _git_sha(),
            "seed": SEED_DEFAULT,
        },
    }
    if extra:
        panel.update(extra)
    return panel


def export_panel(panel, path):
    """Write the panel dict to path as JSON with indent=2."""
    with open(path, "w") as f:
        json.dump(panel, f, indent=2)


def load_panel(path):
    """Read a panel dict back from a JSON file; round-trip exact."""
    with open(path) as f:
        return json.load(f)


def summarize_zoo_candidate(panel):
    """Exact subset of fields a monster-zoo ledger row needs.

    stratum_key and config are honest PLACEHOLDERS (the zoo router owns the
    real values); gate_evidence is lifted from the panel.
    """
    return {
        "stratum_key": "PLACEHOLDER-STRATUM",
        "config": "PLACEHOLDER-CONFIG",
        "gate_evidence": {
            "gate_name": panel["gate_name"],
            "tier_label": panel["tier_label"],
            "inlier_ratio": panel["inlier_ratio"],
            "rms_px": panel["rms_px"],
            "q_hat": panel["q_hat"],
            "ece_proxy": panel["ece_proxy"],
        },
    }
