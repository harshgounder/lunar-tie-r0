"""UNIT-8 coverage gates + ANMS uniform selection (D9 10.1-10.4 / P7.m1-m2).
Gate: 'coverage gates + uniform selection on 3 synthetic fixtures'.

A tie set can have low RMSE yet be clustered inside one crater, which is
globally wrong (wall U). This module provides (a) coverage METRICS following
the canon gates and (b) uniform SELECTION (ANMS + grid quotas) that keeps the
best-scoring points while forcing spatial spread. Canon gates: 8x8 grid
occupancy >= 60%, quadrant checks, and coverage entropy. R0 works on points
only (no raster needed).

Pure numpy + stdlib. No cv2, no scipy, no skimage. nearest_neighbor_cv is the
canon "spread evenness" proxy and is O(N^2); that is fine for R0 tie-set sizes
but noted so a future unit can bound it.

Ambiguity policy: an empty point set returns all-zero metrics (occupancy 0.0,
entropy 0.0, quadrant False, NN-CV 0.0) and an empty selection; a target count
larger than the pool returns all points without crashing.
"""

import numpy as np


def apply_similarity(M, pts):
    """Apply the 2x3 similarity matrix M to Nx2 pts (reuse consensus formula)."""
    pts = np.asarray(pts, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("pts must be an Nx2 array")
    ones = np.ones((pts.shape[0], 1), dtype=np.float64)
    return np.hstack([pts, ones]) @ M.T


def _cell_indices(pts, W, H, grid_n):
    """Linear cell index [0, grid_n^2) for each point."""
    if pts.size == 0:
        return np.array([], dtype=np.int64)
    if not np.isfinite(pts).all():
        raise ValueError('non-finite point in coverage metrics')
    if (pts[:, 0] < 0).any() or (pts[:, 1] < 0).any() \
            or (pts[:, 0] >= W).any() or (pts[:, 1] >= H).any():
        raise ValueError('point outside image bounds in coverage metrics')
    x = np.clip(pts[:, 0], 0.0, W - 1e-9)
    y = np.clip(pts[:, 1], 0.0, H - 1e-9)
    col = np.minimum((x / W * grid_n).astype(np.int64), grid_n - 1)
    row = np.minimum((y / H * grid_n).astype(np.int64), grid_n - 1)
    return row * grid_n + col


def grid_occupancy(pts, W, H, grid_n=8):
    """Fraction of the grid_n x grid_n cells that contain at least one point.

    Returns (occupancy_ratio, occupied_cells). Empty input yields (0.0, 0).
    """
    pts = np.asarray(pts, dtype=np.float64)
    if pts.shape[0] == 0 or W <= 0 or H <= 0 or grid_n < 1:
        return 0.0, 0
    cells = _cell_indices(pts, float(W), float(H), grid_n)
    total = grid_n * grid_n
    occupied = int(np.unique(cells).size)
    return occupied / total, occupied


def coverage_entropy(pts, W, H, grid_n=8):
    """Shannon entropy of the cell count distribution in [0, 1].

    Normalized by log(total_cells) so a perfectly uniform spread scores 1.0.
    Empty input yields 0.0.
    """
    pts = np.asarray(pts, dtype=np.float64)
    if pts.shape[0] == 0 or W <= 0 or H <= 0 or grid_n < 1:
        return 0.0
    cells = _cell_indices(pts, float(W), float(H), grid_n)
    total = grid_n * grid_n
    counts = np.bincount(cells, minlength=total).astype(np.float64)
    probs = counts / counts.sum()
    nonzero = probs[probs > 0.0]
    ent = float(-(nonzero * np.log(nonzero)).sum())
    hmax = float(np.log(total))
    if hmax <= 0.0:
        return 0.0
    return ent / hmax


def quadrant_check(pts, W, H):
    """True when every image quadrant holds at least one point."""
    pts = np.asarray(pts, dtype=np.float64)
    if pts.shape[0] == 0:
        return False
    mx = float(W) / 2.0
    my = float(H) / 2.0
    quads = set(zip(pts[:, 0] >= mx, pts[:, 1] >= my))
    return len(quads) == 4


def nearest_neighbor_cv(pts):
    """Coefficient of variation of 1-NN distances (spread evenness proxy).

    O(N^2) pairwise via numpy; acceptable for R0 tie-set sizes. Empty or
    single-point input yields 0.0.
    """
    pts = np.asarray(pts, dtype=np.float64)
    n = pts.shape[0]
    if n < 2:
        return 0.0
    diff = pts[:, None, :] - pts[None, :, :]
    d = np.sqrt((diff ** 2).sum(axis=-1))
    np.fill_diagonal(d, np.inf)
    nn = d.min(axis=1)
    mean = float(nn.mean())
    if mean == 0.0:
        return 0.0
    return float(nn.std()) / mean


def coverage_metrics(pts, W, H, grid_n=8):
    """Combine all coverage gates plus the NN-CV spread proxy into a dict."""
    pts = np.asarray(pts, dtype=np.float64)
    if pts.shape[0] == 0:
        return {
            "occupancy_ratio": 0.0,
            "occupied_cells": 0,
            "coverage_entropy": 0.0,
            "quadrant_ok": False,
            "nearest_neighbor_cv": 0.0,
        }
    occ, occ_cells = grid_occupancy(pts, W, H, grid_n)
    return {
        "occupancy_ratio": float(occ),
        "occupied_cells": int(occ_cells),
        "coverage_entropy": float(coverage_entropy(pts, W, H, grid_n)),
        "quadrant_ok": bool(quadrant_check(pts, W, H)),
        "nearest_neighbor_cv": float(nearest_neighbor_cv(pts)),
    }


def _far_enough(pt, accepted, min_distance):
    """True if pt is >= min_distance from every already-accepted point."""
    if not accepted:
        return True
    a = np.array(accepted, dtype=np.float64)
    d = np.sqrt(((a - pt) ** 2).sum(axis=1))
    return bool(d.min() >= min_distance)


def anms_select(scores, pts, n_target, min_distance, W=None, H=None,
                grid_n=None):
    """Select points with a two-round ANMS that forces spatial spread.

    Policy (documented): ROUND 1 takes one best-scoring point per occupied
    grid cell, so every cell gets a chance first; ROUND 2 fills remaining
    slots by pure score among the leftover points. Both rounds accept a point
    only if it is >= min_distance (px) from every already-accepted point, so a
    high-score point inside a locked cluster still loses to spread.

    When W/H are omitted the grid is derived from the point bounding box.
    Returns the selected point indices in acceptance order. Empty input or
    n_target <= 0 yields an empty array.

    min_distance PRECEDENCE: the distance guard applies in BOTH rounds even
    when it leaves slots unfilled, so n_target > pool size does NOT
    necessarily return every pool point: a tight cluster (pairwise closer
    than min_distance) contributes only points that clear the guard from the
    already-accepted set (a 4-point cluster 1 px apart with n_target=10,
    min_distance=40 selects 1, not 4). Spread wins over the target count by
    design; call with min_distance <= 0 when the full pool is wanted.
    """
    scores = np.asarray(scores, dtype=np.float64)
    pts = np.asarray(pts, dtype=np.float64)
    n = pts.shape[0]
    if n == 0 or n_target <= 0:
        return np.array([], dtype=np.int64)

    offset = np.zeros(2, dtype=np.float64)
    if W is None or H is None:
        lo = pts.min(axis=0)
        hi = pts.max(axis=0)
        ext = hi - lo
        W = float(ext[0]) if ext[0] > 0 else 1.0
        H = float(ext[1]) if ext[1] > 0 else 1.0
        offset = lo
        pts = pts - offset
    grid_n = grid_n or 8

    order = np.argsort(-scores)
    cells = _cell_indices(pts, float(W), float(H), int(grid_n))
    best_of_cell = {}
    for i in order:
        c = int(cells[i])
        if c not in best_of_cell:
            best_of_cell[c] = i

    accepted = []
    accepted_pts = []

    # ROUND 1: one best per occupied cell, highest-scoring cell first.
    cell_order = sorted(best_of_cell.items(), key=lambda kv: -scores[kv[1]])
    for _c, i in cell_order:
        if len(accepted) >= n_target:
            break
        pt = pts[i]
        if _far_enough(pt, accepted_pts, min_distance):
            accepted.append(i)
            accepted_pts.append(pt)

    # ROUND 2: fill remaining slots by pure score, still min_distance-guarded.
    for i in order:
        if len(accepted) >= n_target:
            break
        if i in accepted:
            continue
        pt = pts[i]
        if _far_enough(pt, accepted_pts, min_distance):
            accepted.append(i)
            accepted_pts.append(pt)

    return np.array(accepted, dtype=np.int64)


def select_uniform_ties(src_pts, dst_pts, match_scores, M, W, H,
                        n_target=120, min_distance=32.0, grid_n=8):
    """Warp src via M, keep in-image points, and run two-round ANMS on the
    warped source positions. Returns {'indices', 'metrics'} where indices
    refer to rows of the original (src,dst,score) arrays and metrics is the
    final coverage_metrics dict on the selected warped positions.
    """
    src = np.asarray(src_pts, dtype=np.float64)
    scores = np.asarray(match_scores, dtype=np.float64)
    warped = apply_similarity(M, src)
    mask = ((warped[:, 0] >= 0) & (warped[:, 0] < W)
            & (warped[:, 1] >= 0) & (warped[:, 1] < H))
    empty = coverage_metrics(np.empty((0, 2)), W, H, grid_n)
    if not mask.any():
        return {"indices": np.array([], dtype=np.int64), "metrics": empty}
    w = warped[mask]
    s = scores[mask]
    local_idx = anms_select(s, w, n_target, min_distance, W=W, H=H,
                            grid_n=grid_n)
    full_idx = np.where(mask)[0][local_idx]
    selected_w = w[local_idx]
    metrics = coverage_metrics(selected_w, W, H, grid_n)
    return {"indices": full_idx, "metrics": metrics}
