"""UNIT-6 geometric consensus: MAGSAC-style + affine (D4 7.1-7.3 / P5.m1).
Gate: 'consensus recovery on 3 synthetic fixtures'.

After UNIT-5 matching a fraction of putative matches are outliers (repetitive
craters, noise). This module estimates a robust geometric model with an inlier
set, GSD-normalized. For R0 the SIMILARITY model (4-DoF: scale + rotation +
tx + ty) is the geometric truth for lunar pairs of the same surface at modest
viewpoint change. HOMOGRAPHY (8-DoF) is exposed as an OPTION but is NOT the
default terrain truth (its epipolar geometry is only exact for planes; lunar
DEM relief makes pure-H globally wrong). This unit provides the consensus
layer the sub-pixel refiner will consume.

Pure numpy + stdlib. No cv2, no scipy, no skimage. Self-contained; reuses no
other module internals beyond numpy.

Ambiguity policy: fewer than 2 point pairs raises ValueError loudly; a
degenerate all-collinear source set raises ValueError loudly rather than
returning a junk matrix.
"""

import numpy as np


def fit_similarity(src_pts, dst_pts):
    """Least-squares closed-form similarity fit (no RANSAC).

    Model: dst = s * R(theta) * src + t, with
        R = [[cos, -sin], [sin, cos]].
    Let a = s*cos(theta), b = s*sin(theta); the map is linear in (a, b, tx, ty):
        dst_x = a*src_x - b*src_y + tx
        dst_y = b*src_x + a*src_y + ty
    solved by normal equations. Returns (s, theta, tx, ty, M) where M is the
    composed 2x3 matrix [[a, -b, tx], [b, a, ty]].
    """
    src = np.asarray(src_pts, dtype=np.float64)
    dst = np.asarray(dst_pts, dtype=np.float64)
    if src.ndim != 2 or src.shape[1] != 2 or dst.shape != src.shape:
        raise ValueError("src_pts and dst_pts must be Nx2 arrays")
    if not np.isfinite(src).all() or not np.isfinite(dst).all():
        raise ValueError("non-finite coordinates in consensus fit")
    n = src.shape[0]
    if n < 2:
        raise ValueError("need at least 2 point pairs for a similarity fit")
    c = src - src.mean(axis=0)
    cov = c.T @ c
    ev = np.linalg.eigvalsh(cov)
    if n > 2 and ev[0] < 1e-12 * max(ev[1], 1e-12):
        raise ValueError("degenerate all-collinear source points")

    A = np.zeros((2 * n, 4), dtype=np.float64)
    b = np.zeros(2 * n, dtype=np.float64)
    A[0::2, 0] = src[:, 0]
    A[0::2, 1] = -src[:, 1]
    A[0::2, 2] = 1.0
    A[1::2, 0] = src[:, 1]
    A[1::2, 1] = src[:, 0]
    A[1::2, 3] = 1.0
    b[0::2] = dst[:, 0]
    b[1::2] = dst[:, 1]
    x, *_ = np.linalg.lstsq(A, b, rcond=None)
    a, bb, tx, ty = x
    s = float(np.hypot(a, bb))
    theta = float(np.arctan2(bb, a))
    M = np.array([[a, -bb, tx], [bb, a, ty]], dtype=np.float64)
    if not np.isfinite(M).all():
        raise ValueError("degenerate fit produced non-finite matrix")
    return s, theta, tx, ty, M


def similarity_residuals(M, src_pts, dst_pts):
    """Per-point euclidean residual (px) of the similarity model M."""
    src = np.asarray(src_pts, dtype=np.float64)
    dst = np.asarray(dst_pts, dtype=np.float64)
    ones = np.ones((src.shape[0], 1), dtype=np.float64)
    proj = np.hstack([src, ones]) @ M.T
    return np.sqrt(((proj - dst) ** 2).sum(axis=1))


def magsac_consensus(src_pts, dst_pts, min_samples=2, iters=500,
                     sigma_thr=2.0, conf=0.999, seed=None):
    """RANSAC-style consensus with a similarity model.

    Randomized minimal sets of 2 matched pairs; each hypothesis is scored by
    a HARD residual gate: the inlier count of res < sigma_thr (px). The
    reported sigma_max is the constant sigma_thr itself, NOT a sigma
    marginalization over the median residual: no soft threshold or
    probabilistic weighting is applied (the MAGSAC flavor is the
    hypothesis-then-refit loop, not the sigma integral). The best inlier set
    is then polished by refitting on all inliers (2 rounds or until the set
    stops changing). Returns a dict with M, inliers, inlier_ratio,
    sigma_max and residuals.
    """
    src = np.asarray(src_pts, dtype=np.float64)
    dst = np.asarray(dst_pts, dtype=np.float64)
    n = src.shape[0]
    if n < 2:
        raise ValueError("need at least 2 point pairs for consensus")
    rng = np.random.default_rng(seed)

    best_score = -1
    best_mask = None
    for _ in range(iters):
        idx = rng.choice(n, size=min_samples, replace=False)
        try:
            _, _, _, _, M = fit_similarity(src[idx], dst[idx])
        except ValueError:
            continue
        res = similarity_residuals(M, src, dst)
        if not np.isfinite(res).all():
            continue
        mask = res < sigma_thr
        score = int(mask.sum())
        if score > best_score:
            best_score = score
            best_mask = mask

    if best_mask is None:
        raise ValueError("no valid similarity model found")

    mask = best_mask.copy()
    for _ in range(2):
        if mask.sum() < 2:
            break
        _, _, _, _, M = fit_similarity(src[mask], dst[mask])
        res = similarity_residuals(M, src, dst)
        new_mask = res < sigma_thr
        if np.array_equal(new_mask, mask):
            break
        mask = new_mask

    if mask.sum() < 2:
        raise ValueError("no valid similarity model found")
    _, _, _, _, M = fit_similarity(src[mask], dst[mask])
    res = similarity_residuals(M, src, dst)
    return {
        "M": M,
        "inliers": mask,
        "inlier_ratio": float(mask.sum()) / n,
        "sigma_max": float(sigma_thr),
        "residuals": res,
    }


def gsd_normalize(pts, gsd):
    """Scale points by 1/gsd so thresholds are GSD-normalized (source px)."""
    pts = np.asarray(pts, dtype=np.float64)
    if gsd <= 0:
        raise ValueError("gsd must be positive")
    return pts / gsd


def rms_px(residuals):
    """RMS of a residual array: sqrt(mean(res^2))."""
    res = np.asarray(residuals, dtype=np.float64)
    if res.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(res ** 2)))


def fit_homography(src_pts, dst_pts):
    """8-DoF homography via DLT (numpy only).

    Exposed as an OPTION for tier-4 eval; similarity remains the default
    verdict model for R0 (lunar DEM relief makes pure-H globally wrong).

    Degenerate geometry (null vector with a ~zero last component, or a
    non-finite result) raises ValueError loudly instead of returning
    inf/nan junk from the H[2,2] normalization, mirroring fit_similarity's
    policy.
    """
    src = np.asarray(src_pts, dtype=np.float64)
    dst = np.asarray(dst_pts, dtype=np.float64)
    n = src.shape[0]
    if n < 4:
        raise ValueError("need at least 4 point pairs for a homography")
    A = np.zeros((2 * n, 9), dtype=np.float64)
    for i in range(n):
        x, y = src[i]
        u, v = dst[i]
        A[2 * i] = [x, y, 1.0, 0.0, 0.0, 0.0, -u * x, -u * y, -u]
        A[2 * i + 1] = [0.0, 0.0, 0.0, x, y, 1.0, -v * x, -v * y, -v]
    _, _, Vt = np.linalg.svd(A)
    H = Vt[-1].reshape(3, 3)
    if abs(H[2, 2]) < 1e-12:
        raise ValueError(
            "degenerate homography: null vector has ~zero H[2,2]")
    H = H / H[2, 2]
    if not np.isfinite(H).all():
        raise ValueError("degenerate fit produced non-finite homography")
    return H
