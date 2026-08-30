"""UNIT-6 gate test. Gate: 'consensus recovery on 3 synthetic fixtures'."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from lunar_tie.consensus import (
    fit_homography,
    fit_similarity,
    gsd_normalize,
    magsac_consensus,
    rms_px,
    similarity_residuals,
)


def _apply_similarity(src, s, theta, tx, ty):
    c, sn = np.cos(theta), np.sin(theta)
    R = np.array([[c, -sn], [sn, c]])
    return src @ (s * R).T + np.array([tx, ty])


def _fixture_similarity(n=200, s=1.0, theta=0.0, tx=0.0, ty=0.0,
                        outlier_frac=0.0, seed=3):
    rng = np.random.default_rng(seed)
    src = rng.uniform(-50.0, 50.0, size=(n, 2))
    dst = _apply_similarity(src, s, theta, tx, ty)
    if outlier_frac > 0.0:
        n_out = int(round(n * outlier_frac))
        idx = rng.choice(n, size=n_out, replace=False)
        dst[idx] += rng.uniform(-30.0, 30.0, size=(n_out, 2))
    return src, dst


def test_fixture_a_similarity_recovery():
    src, dst = _fixture_similarity(n=200, s=1.3, theta=0.4, tx=7.0, ty=-3.0,
                                   outlier_frac=0.10, seed=3)
    out = magsac_consensus(src, dst, iters=500, sigma_thr=2.0, seed=1)
    s, theta, tx, ty, M = fit_similarity(src[out["inliers"]],
                                         dst[out["inliers"]])
    assert abs(s - 1.3) / 1.3 < 0.01
    assert abs(theta - 0.4) < 0.01
    assert abs(tx - 7.0) < 0.5
    assert abs(ty - (-3.0)) < 0.5
    assert out["inlier_ratio"] >= 0.88
    assert rms_px(out["residuals"][out["inliers"]]) < 0.15


def test_fixture_b_pure_translation_identity():
    src, dst = _fixture_similarity(n=200, s=1.0, theta=0.0, tx=5.0, ty=2.0,
                                   seed=5)
    s, theta, tx, ty, M = fit_similarity(src, dst)
    assert abs(s - 1.0) < 1e-9
    assert abs(theta) < 1e-9
    out = magsac_consensus(src, dst, iters=200, seed=2)
    assert out["inlier_ratio"] == 1.0
    assert out["inliers"].all()


def test_fixture_c_rotation_scale_recovery():
    src, dst = _fixture_similarity(n=200, s=1.3, theta=np.pi / 3.0, tx=4.0,
                                   ty=-6.0, seed=7)
    out = magsac_consensus(src, dst, iters=500, sigma_thr=2.0, seed=4)
    s, theta, tx, ty, M = fit_similarity(src[out["inliers"]],
                                         dst[out["inliers"]])
    assert abs(s - 1.3) / 1.3 < 0.01
    assert abs(theta - np.pi / 3.0) < 0.01
    assert abs(tx - 4.0) < 0.5
    assert abs(ty - (-6.0)) < 0.5
    assert out["inlier_ratio"] >= 0.95


def test_fit_similarity_returns_matrix():
    src, dst = _fixture_similarity(n=50, s=1.2, theta=0.3, tx=1.0, ty=2.0,
                                   seed=9)
    s, theta, tx, ty, M = fit_similarity(src, dst)
    assert M.shape == (2, 3)
    assert np.all(np.isfinite(M))


def test_similarity_residuals_zero_for_exact():
    src, dst = _fixture_similarity(n=30, s=1.0, theta=0.0, tx=0.0, ty=0.0,
                                   seed=11)
    _, _, _, _, M = fit_similarity(src, dst)
    res = similarity_residuals(M, src, dst)
    assert np.all(res < 1e-9)


def test_gsd_normalize_scales():
    pts = np.array([[10.0, 20.0], [30.0, 40.0]])
    out = gsd_normalize(pts, 2.0)
    assert np.allclose(out, pts / 2.0)


def test_gsd_normalize_nonpositive_raises():
    pts = np.array([[1.0, 2.0]])
    for bad in (0.0, -1.0):
        try:
            gsd_normalize(pts, bad)
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError for gsd=%s" % bad)


def test_rms_px():
    assert abs(rms_px(np.array([3.0, 4.0])) - np.sqrt(12.5)) < 1e-9
    assert rms_px(np.array([])) == 0.0


def test_fit_similarity_too_few_raises():
    src = np.array([[0.0, 0.0]])
    dst = np.array([[1.0, 1.0]])
    try:
        fit_similarity(src, dst)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for <2 pairs")


def test_fit_similarity_collinear_raises():
    src = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    dst = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    try:
        fit_similarity(src, dst)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for collinear points")


def test_magsac_too_few_raises():
    src = np.array([[0.0, 0.0]])
    dst = np.array([[1.0, 1.0]])
    try:
        magsac_consensus(src, dst)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for <2 pairs")


def test_nan_input_raises_loud():
    src = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    dst = np.array([[0.0, 0.0], [1.0, 1.0], [np.nan, 2.0]])
    try:
        magsac_consensus(src, dst, iters=50, seed=1)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for NaN input")


def test_fit_similarity_nan_raises():
    src = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    dst = np.array([[0.0, 0.0], [1.0, 1.0], [np.nan, 2.0]])
    try:
        fit_similarity(src, dst)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for NaN coordinate")


def test_fit_homography_identity():
    src = np.random.default_rng(0).uniform(-10.0, 10.0, size=(8, 2))
    dst = src.copy()
    H = fit_homography(src, dst)
    assert H.shape == (3, 3)
    assert abs(H[2, 2] - 1.0) < 1e-9
    ones = np.ones((8, 1))
    proj = np.hstack([src, ones]) @ H.T
    proj = proj[:, :2] / proj[:, 2:3]
    assert np.allclose(proj, dst, atol=1e-6)


def test_fit_homography_too_few_raises():
    src = np.random.default_rng(1).uniform(-5.0, 5.0, size=(3, 2))
    dst = src.copy()
    try:
        fit_homography(src, dst)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for <4 pairs")
