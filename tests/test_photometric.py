"""UNIT-4 gate test. Gate: 'photometric ratio on 3 synthetic fixtures'."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from lunar_tie.photometric import (
    apply_rank,
    correlation_preview,
    gaussian_blur,
    normalize_pair,
    normalize_ratio,
    quantize,
    rank_transform,
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


def test_rank_transform_shape_dtype_range():
    # Gate 1: shape/dtype/range, output in [0, quantize-1].
    img = _fixture_terrain()
    rt = rank_transform(img, bins=6)
    assert rt.shape == img.shape
    assert rt.dtype == np.float64
    assert float(rt.min()) >= 0.0
    assert float(rt.max()) <= 5.0
    # min-max scaling guarantees both extremes are hit.
    assert float(rt.min()) == 0.0
    assert float(rt.max()) == 5.0
    # quantize helper agrees; apply_rank is the ticket spelling.
    assert np.array_equal(rt, quantize(img, 6))
    assert np.array_equal(rt, apply_rank(img, quantize_bins=6))


def test_rank_transform_gain_offset_invariance():
    # Gate 2: rank(rt(img * 1.25 + 8)) == rank(rt(img)), with the ticket's
    # 0.5% pixel-mismatch allowance for exact bin-edge float noise.
    img = _fixture_terrain()
    rtA = rank_transform(img, bins=6)
    rtB = rank_transform(img * 1.25 + 8.0, bins=6)
    mismatch = float(np.mean(rtA != rtB))
    assert mismatch < 0.005


def test_rank_transform_preserves_edges():
    # Gate 3: 2-rock synthetic, gradient location preserved (argmax of
    # |grad| within 2 px). Deterministic (noise-free) synthetic: the
    # bright rock's top-left corner borders the dark rock, so that
    # single corner is the strictly unique global max of |gx| + |gy|
    # in BOTH the raw image (280 vs next best 200) and the rank image
    # (|5-0| + |5-2| = 9 vs next best 6). No tie-breaking involved.
    shape = (64, 64)
    img = np.full(shape, 60.0, dtype=np.float64)
    img[16:40, 12:28] = 20.0   # dark rock
    img[20:44, 28:52] = 180.0  # bright rock, adjacent to dark rock

    def _edge_max(a):
        a = np.asarray(a, dtype=np.float64)
        gx = np.zeros_like(a)
        gy = np.zeros_like(a)
        gx[:, 1:-1] = a[:, 2:] - a[:, :-2]
        gy[1:-1, :] = a[2:, :] - a[:-2, :]
        mag = np.abs(gx) + np.abs(gy)
        peak = float(mag.max())
        # strictly unique peak: quantization must not blur it away
        assert np.sum(mag == peak) == 1
        return np.unravel_index(np.argmax(mag), a.shape)

    r1, c1 = _edge_max(img)
    r2, c2 = _edge_max(rank_transform(img, bins=6))
    # brightest corner sits at (20, 28) in both
    assert (int(r1), int(c1)) == (20, 28)
    assert (int(r2), int(c2)) == (20, 28)
    assert abs(int(r1) - int(r2)) <= 2
    assert abs(int(c1) - int(c2)) <= 2


def test_existing_ratio_norm_tests_unaffected():
    # Gate 4: non-breaking addition; the 3 original fixtures still behave.
    imgA = _fixture_terrain()
    imgB = 1.8 * imgA + 25.0
    nA, nB = normalize_pair(imgA, imgB, sigma=8.0)
    assert np.mean(np.abs(nA - nB)) < 0.05
    assert correlation_preview(nA, nB) > 0.95
    flat = normalize_ratio(_fixture_flat(), sigma=8.0)
    assert np.all(np.isfinite(flat)) and np.all(np.abs(flat) < 10.0)
    bim = normalize_ratio(_fixture_bimodal(), sigma=8.0)
    assert np.all(np.isfinite(bim))
    assert correlation_preview(bim, normalize_ratio(bim, sigma=8.0)) > 0.9


def test_rank_transform_roundtrip_determinism():
    # Gate 5: applying the transform twice equals applying it once.
    img = _fixture_terrain()
    once = rank_transform(img, bins=6)
    twice = rank_transform(once, bins=6)
    assert np.array_equal(once, twice)
