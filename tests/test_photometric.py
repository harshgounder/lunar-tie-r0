"""UNIT-4 gate test. Gate: 'photometric ratio on 3 synthetic fixtures'."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from lunar_tie.photometric import (
    correlation_preview,
    gaussian_blur,
    normalize_pair,
    normalize_ratio,
)


def _fixture_terrain(shape=(96, 96), seed=7):
    """(a) synthetic terrain with smooth structure and some texture."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    base = 60.0 + 40.0 * np.sin(xx / 9.0) * np.cos(yy / 11.0)
    base += 20.0 * np.sin((xx + yy) / 5.0)
    base += rng.normal(0.0, 3.0, size=shape)
    return base.astype(np.float32)


def _fixture_flat(shape=(64, 64)):
    """(b) flat-constant image: zero variance everywhere."""
    return np.full(shape, 128.0, dtype=np.float32)


def _fixture_bimodal(shape=(96, 96), seed=11):
    """(c) realistic two-peak histogram: sunlit/shadow bimodal."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    shadow = (xx + yy) % 24 < 12
    img = np.where(shadow, 30.0, 200.0).astype(np.float32)
    img += rng.normal(0.0, 4.0, size=shape).astype(np.float32)
    return img


def test_gaussian_blur_shape_and_dtype():
    img = _fixture_terrain()
    out = gaussian_blur(img, sigma=8.0)
    assert out.shape == img.shape
    assert out.dtype == img.dtype


def test_gaussian_blur_flat_preserves_value():
    img = _fixture_flat()
    out = gaussian_blur(img, sigma=8.0)
    assert np.allclose(out, 128.0, atol=1e-6)


def test_gaussian_blur_sigma_nonpositive_raises():
    img = _fixture_terrain()
    for bad in (0.0, -1.0):
        try:
            gaussian_blur(img, sigma=bad)
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError for sigma=%s" % bad)


def test_normalize_ratio_shape_dtype():
    img = _fixture_terrain()
    out = normalize_ratio(img, sigma=8.0)
    assert out.shape == img.shape
    assert out.dtype == np.float32


def test_normalize_ratio_sigma_nonpositive_raises():
    img = _fixture_terrain()
    try:
        normalize_ratio(img, sigma=0.0)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for sigma=0")


def test_fixture_a_gain_offset_invariant():
    imgA = _fixture_terrain()
    imgB = 1.8 * imgA + 25.0
    nA, nB = normalize_pair(imgA, imgB, sigma=8.0)
    assert np.mean(np.abs(nA - nB)) < 0.05
    assert correlation_preview(nA, nB) > 0.95


def test_fixture_b_flat_does_not_blow_up():
    img = _fixture_flat()
    out = normalize_ratio(img, sigma=8.0)
    assert np.all(np.isfinite(out))
    assert np.all(np.abs(out) < 10.0)


def test_fixture_c_bimodal_preserves_structure():
    img = _fixture_bimodal()
    n1 = normalize_ratio(img, sigma=8.0)
    n2 = normalize_ratio(n1, sigma=8.0)
    assert np.all(np.isfinite(n1))
    assert np.all(np.isfinite(n2))
    assert correlation_preview(n1, n2) > 0.9


def test_shift_invariance_smoke():
    img = _fixture_terrain()
    n = normalize_ratio(img, sigma=8.0)
    shifted = np.roll(n, 1, axis=1)
    assert correlation_preview(n, shifted) > 0.90


def test_normalize_pair_shared_sigma():
    imgA = _fixture_terrain()
    imgB = _fixture_bimodal()
    nA, nB = normalize_pair(imgA, imgB, sigma=8.0)
    assert nA.shape == imgA.shape
    assert nB.shape == imgB.shape
    assert nA.dtype == np.float32
    assert nB.dtype == np.float32


def test_correlation_preview_self_is_one():
    img = _fixture_terrain()
    n = normalize_ratio(img, sigma=8.0)
    assert abs(correlation_preview(n, n) - 1.0) < 1e-9


def test_correlation_preview_flat_is_zero():
    img = _fixture_flat()
    n = normalize_ratio(img, sigma=8.0)
    assert correlation_preview(n, n) == 0.0
