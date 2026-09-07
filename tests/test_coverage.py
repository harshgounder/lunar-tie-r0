"""UNIT-8 gate test. Gate: 'coverage gates + uniform selection on 3 synthetic fixtures'."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from lunar_tie.coverage import (
    anms_select,
    apply_similarity,
    coverage_entropy,
    coverage_metrics,
    grid_occupancy,
    nearest_neighbor_cv,
    quadrant_check,
    select_uniform_ties,
)

W, H = 512, 512


def _fixture_clustered(n_clust=400, n_spread=100, seed=0):
    rng = np.random.default_rng(seed)
    clust = np.array([W / 2, H / 2]) + rng.normal(0.0, 8.0, size=(n_clust, 2))
    spread = rng.uniform(0.0, W, size=(n_spread, 2))
    return np.vstack([clust, spread])


def test_gate_a_clustered_metrics():
    pts = np.array([287.5, 287.5]) + np.random.default_rng(1).normal(0.0, 1.0,
                                                                    size=(500, 2))
    occ, cells = grid_occupancy(pts, W, H)
    assert occ <= 1.0 / 64.0 + 1e-9
    assert coverage_entropy(pts, W, H) < 0.2
    assert quadrant_check(pts, W, H) is False


def test_gate_b_uniform_grid_metrics():
    xs = np.linspace(W / 16.0, W - W / 16.0, 8)
    ys = np.linspace(H / 16.0, H - H / 16.0, 8)
    gx, gy = np.meshgrid(xs, ys)
    pts = np.stack([gx.ravel(), gy.ravel()], axis=1)
    occ, cells = grid_occupancy(pts, W, H)
    assert occ == 1.0
    assert coverage_entropy(pts, W, H) > 0.95
    assert quadrant_check(pts, W, H) is True


def test_gate_c_anms_forces_spread():
    pts = _fixture_clustered(seed=3)
    rng = np.random.default_rng(5)
    scores = rng.uniform(0.0, 1.0, size=pts.shape[0])
    idx = anms_select(scores, pts, n_target=60, min_distance=40.0, W=W, H=H)
    assert idx.size == 60
    selected = pts[idx]
    occ, _ = grid_occupancy(selected, W, H)
    assert occ >= 0.55


def test_empty_pts_all_metrics_zero():
    pts = np.empty((0, 2))
    assert grid_occupancy(pts, W, H) == (0.0, 0)
    assert coverage_entropy(pts, W, H) == 0.0
    assert quadrant_check(pts, W, H) is False
    assert nearest_neighbor_cv(pts) == 0.0
    m = coverage_metrics(pts, W, H)
    assert m["occupancy_ratio"] == 0.0
    assert m["coverage_entropy"] == 0.0
    assert m["quadrant_ok"] is False


def test_anms_target_larger_than_pool_returns_all():
    pts = np.array([[10.0, 10.0], [200.0, 300.0]])
    idx = anms_select(np.array([1.0, 0.5]), pts, n_target=10, min_distance=1.0,
                      W=W, H=H)
    assert idx.size == 2


def test_anms_min_distance_guard_precedes_target_when_clustered():
    """F6 (audit): the docstring used to promise 'n_target > pool yields
    all', but the min_distance guard takes precedence in both rounds: a
    tight cluster with n_target above the pool size selects only the points
    that clear the guard, not the whole pool. Spread wins over the count."""
    cluster = np.array([[256.0, 256.0], [256.5, 256.0],
                        [257.0, 256.0], [257.5, 256.0]])
    idx = anms_select(np.array([1.0, 0.9, 0.8, 0.7]), cluster, n_target=10,
                      min_distance=40.0, W=W, H=H)
    assert idx.size == 1, idx
    # the guard lifted: the same pool fills past the target
    idx_all = anms_select(np.array([1.0, 0.9, 0.8, 0.7]), cluster, n_target=10,
                          min_distance=0.0, W=W, H=H)
    assert idx_all.size == 4
    assert sorted(idx_all.tolist()) == [0, 1, 2, 3]


def test_anms_high_score_cluster_loses_to_spread():
    rng = np.random.default_rng(7)
    base = np.array([W / 2, H / 2]) + rng.normal(0.0, 2.0, size=(30, 2))
    far = np.array([[20.0, 20.0], [W - 20.0, H - 20.0], [20.0, H - 20.0]])
    pts = np.vstack([base, far])
    scores = np.r_[np.ones(30) * 0.99, np.array([0.1, 0.1, 0.1])]
    idx = anms_select(scores, pts, n_target=3, min_distance=40.0, W=W, H=H)
    occ, _ = grid_occupancy(pts[idx], W, H)
    assert occ == 3.0 / 64.0


def test_coverage_metrics_contains_all_keys():
    pts = _fixture_clustered(seed=11)
    m = coverage_metrics(pts, W, H)
    for k in ("occupancy_ratio", "occupied_cells", "coverage_entropy",
              "quadrant_ok", "nearest_neighbor_cv"):
        assert k in m


def test_out_of_bound_point_raises_loud():
    pts = np.array([[-5.0, 10.0], [10.0, 10.0]])
    with pytest.raises(ValueError):
        grid_occupancy(pts, W, H)


def test_single_point_metrics():
    pts = np.array([[10.0, 10.0]])
    assert nearest_neighbor_cv(pts) == 0.0
    occ, cells = grid_occupancy(pts, W, H)
    assert cells == 1
    assert quadrant_check(pts, W, H) is False


def test_apply_similarity_formula():
    s, theta = 1.3, 0.4
    R = np.array([[s * np.cos(theta), -s * np.sin(theta)],
                  [s * np.sin(theta), s * np.cos(theta)]])
    M = np.hstack([R, np.array([[7.0], [-3.0]])])
    src = np.random.default_rng(2).uniform(-50.0, 50.0, size=(20, 2))
    out = apply_similarity(M, src)
    assert out.shape == src.shape
    assert np.allclose(out, src @ R.T + np.array([7.0, -3.0]))


def test_select_uniform_ties_struct():
    rng = np.random.default_rng(13)
    src = rng.uniform(-50.0, 50.0, size=(200, 2))
    dst = src + np.array([5.0, 2.0])
    M = np.array([[1.0, 0.0, 5.0], [0.0, 1.0, 2.0]])
    scores = rng.uniform(0.0, 1.0, size=200)
    res = select_uniform_ties(src, dst, scores, M, W, H, n_target=80,
                              min_distance=32.0)
    assert set(res.keys()) == {"indices", "metrics"}
    assert res["indices"].dtype.kind == "i"
    assert 0 < res["indices"].size <= 80
    for k in ("occupancy_ratio", "occupied_cells", "coverage_entropy",
              "quadrant_ok", "nearest_neighbor_cv"):
        assert k in res["metrics"]
