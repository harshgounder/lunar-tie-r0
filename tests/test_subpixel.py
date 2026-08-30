"""UNIT-7 gate test. Gate: 'subpixel recovery on 3 synthetic fixtures'."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from lunar_tie.subpixel import (
    bilinear_sample,
    extract_patch,
    phase_shift,
    refine_matches,
    rms_subpx,
)


def _synthetic(size=128, seed=0):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size]
    base = rng.uniform(0.0, 255.0, size=(size, size))
    for _ in range(1):
        base = (base[1:, :] + base[:-1, :]) / 2.0
        base = (base[:, 1:] + base[:, :-1]) / 2.0
    base = np.pad(base, 1, mode="edge")[:size, :size]
    for _ in range(12):
        cx = rng.uniform(20.0, size - 20.0)
        cy = rng.uniform(20.0, size - 20.0)
        r = rng.uniform(4.0, 14.0)
        base = base + 90.0 * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2)
                                    / (2.0 * r * r))
    base = np.clip(base, 0.0, 255.0)
    return base.astype(np.uint8)


def _shift(img, dx, dy):
    """Exact sub-pixel translation via FFT phase ramp (true translation)."""
    img = np.asarray(img, dtype=np.float64)
    h, w = img.shape
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.fftfreq(w)[None, :]
    F = np.fft.fft2(img)
    F = F * np.exp(-2j * np.pi * (fx * dx + fy * dy))
    return np.fft.ifft2(F).real


def _sample_pts(img, n=40, seed=1, margin=24):
    rng = np.random.default_rng(seed)
    h, w = img.shape
    xs = rng.uniform(margin, w - margin, size=n)
    ys = rng.uniform(margin, h - margin, size=n)
    return np.stack([xs, ys], axis=1)


def test_fixture_a_shift_recovery():
    imgA = _synthetic(seed=0)
    dx, dy = 3.5, -2.25
    imgB = _shift(imgA, dx, dy)
    src = _sample_pts(imgA, n=40, seed=1)
    dst = src.copy()
    out = refine_matches(imgA, imgB, src, dst, half=16, upsample=16)
    err = np.linalg.norm(out["deltas"] - np.array([dx, dy]), axis=1)
    frac = float((err <= 0.25).sum()) / len(err)
    assert frac >= 0.80
    # calibrated bar: pv=0.42 clean-windowed on a true 1.5 px translation
    assert float((out["peak_vals"] >= 0.30).sum()) / len(err) >= 0.80
    assert out["valid"].sum() >= 0.8 * len(err)
    assert rms_subpx(err[out["valid"]]) <= 0.25


def test_fixture_b_translation_ladder():
    imgA = _synthetic(seed=2)
    src = _sample_pts(imgA, n=30, seed=3)
    for shift in (0.25, 0.5, 1.0):
        imgB = _shift(imgA, shift, shift)
        dst = src.copy()
        out = refine_matches(imgA, imgB, src, dst, half=16, upsample=16)
        err = np.linalg.norm(out["deltas"] - np.array([shift, shift]), axis=1)
        assert float(np.max(err)) <= 0.2
        assert float(np.mean(err)) <= 0.1


def test_fixture_c_noise_sanity():
    imgA = _synthetic(seed=4)
    dx, dy = 2.0, -1.5
    imgB = _shift(imgA, dx, dy)
    rng = np.random.default_rng(5)
    imgB = np.clip(imgB.astype(np.float64) + rng.normal(0.0, 5.0, imgB.shape),
                   0.0, 255.0).astype(np.uint8)
    src = _sample_pts(imgA, n=40, seed=6)
    dst = src.copy()
    out = refine_matches(imgA, imgB, src, dst, half=16, upsample=16)
    err = np.linalg.norm(out["deltas"] - np.array([dx, dy]), axis=1)
    frac = float((err <= 0.35).sum()) / len(err)
    assert frac >= 0.70
    # calibrated bar: pv=0.37 without window on a true 1.5 px translation
    assert float((out["peak_vals"] >= 0.25).sum()) / len(err) >= 0.70


def test_extract_patch_shape_and_dtype():
    img = _synthetic(seed=7)
    p = extract_patch(img, 50.0, 50.0, half=16)
    assert p.shape == (33, 33)
    assert p.dtype == np.float32


def test_extract_patch_fractional_center():
    img = _synthetic(seed=8)
    p = extract_patch(img, 50.5, 50.5, half=2)
    assert p.shape == (5, 5)
    assert np.all(np.isfinite(p))


def test_bilinear_sample_identity():
    img = _synthetic(seed=9)
    assert abs(bilinear_sample(img, 10.0, 10.0) - float(img[10, 10])) < 1e-6


def test_bilinear_sample_clamp():
    img = _synthetic(seed=10)
    h, w = img.shape
    assert abs(bilinear_sample(img, -5.0, -5.0) - float(img[0, 0])) < 1e-6
    assert abs(bilinear_sample(img, w + 5.0, h + 5.0) -
               float(img[h - 1, w - 1])) < 1e-6


def test_phase_shift_identity_zero():
    img = _synthetic(seed=11)
    p = extract_patch(img, 60.0, 60.0, half=16)
    dx, dy, pv = phase_shift(p, p, upsample=16)
    assert abs(dx) < 0.1
    assert abs(dy) < 0.1
    assert pv >= 0.9


def test_phase_shift_known_shift():
    img = _synthetic(seed=12)
    pA = extract_patch(img, 60.0, 60.0, half=16)
    pB = extract_patch(_shift(img, 1.5, -0.5), 60.0, 60.0, half=16)
    dx, dy, pv = phase_shift(pA, pB, upsample=16)
    assert abs(dx - 1.5) < 0.2
    assert abs(dy - (-0.5)) < 0.2
    assert pv >= 0.5


def test_refine_matches_outside_patch_invalid():
    imgA = _synthetic(seed=13)
    imgB = imgA.copy()
    src = np.array([[5.0, 5.0], [60.0, 60.0]])
    dst = src.copy()
    out = refine_matches(imgA, imgB, src, dst, half=16)
    assert out["valid"][0] == False
    assert out["valid"][1] == True


def test_refine_matches_returns_contract():
    imgA = _synthetic(seed=14)
    imgB = imgA.copy()
    src = np.array([[60.0, 60.0], [70.0, 70.0]])
    dst = src.copy()
    out = refine_matches(imgA, imgB, src, dst, half=16)
    assert set(out.keys()) == {"src", "dst", "deltas", "peak_vals", "valid"}
    assert out["deltas"].shape == (2, 2)
    assert out["peak_vals"].shape == (2,)
    assert out["valid"].shape == (2,)
    assert np.allclose(out["dst"], dst + out["deltas"])


def test_rms_subpx():
    assert abs(rms_subpx(np.array([3.0, 4.0])) - np.sqrt(12.5)) < 1e-9
    assert rms_subpx(np.array([])) == 0.0
