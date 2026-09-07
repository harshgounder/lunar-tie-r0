"""UNIT-2 gate test. Gate: 'triple mask on 3 synthetic fixtures'."""

import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from lunar_tie.masking import (
    compute_masks,
    load_masks,
    otsu_threshold,
    save_masks,
    summary_masks,
)


def _fixture_uniform_bright():
    """(a) uniform bright image: no shadows, no nodata, usable ~ 100."""
    return np.full((64, 64), 200, dtype=np.uint8)


def _fixture_half_dark_half_bright():
    """(b) left half dark, right half bright: shadow covers dark half."""
    img = np.full((64, 64), 200, dtype=np.uint8)
    img[:, :32] = 10
    return img


def _fixture_nodata_and_saturated_bands():
    """(c) top nodata band of 0s and bottom saturated band of 255s."""
    img = np.full((64, 64), 100, dtype=np.uint8)
    img[:16, :] = 0
    img[48:, :] = 255
    return img


def test_fixture_a_uniform_bright_usable_100():
    img = _fixture_uniform_bright()
    masks = compute_masks(img)
    assert masks["valid"].all()
    assert not masks["nodata"].any()
    assert not masks["shadow"].any()
    assert masks["combined"].all()
    s = summary_masks(img)
    assert s["usable_pct"] == 100.0
    assert s["valid_pct"] == 100.0


def test_fixture_a_flat_otsu_is_none():
    img = _fixture_uniform_bright()
    assert otsu_threshold(img) is None


def test_fixture_b_shadow_covers_dark_half():
    img = _fixture_half_dark_half_bright()
    masks = compute_masks(img)
    assert masks["shadow"][:, :32].all()
    assert not masks["shadow"][:, 32:].any()
    assert masks["valid"].all()
    assert not masks["nodata"].any()
    s = summary_masks(img)
    assert s["shadow_pct"] == 50.0
    assert s["usable_pct"] == 50.0


def test_fixture_b_otsu_separates_classes():
    img = _fixture_half_dark_half_bright()
    thr = otsu_threshold(img)
    assert thr is not None
    assert 10 <= thr < 200


def test_fixture_c_nodata_and_saturated_excluded():
    """(c) top nodata band of 0s and bottom saturated band of 255s.

    The middle 100-band is neither nodata nor shadow: otsu splits at 100
    making the dark side (0s + 100s) the 75% majority, which the shadow
    minority rule correctly refuses as a saturation-edge latch verdict, so
    usable == valid == 50% (the old code shadowed the whole middle band and
    usable was 0.0, a vacuous pass of the old < 50 assertion).
    """
    img = _fixture_nodata_and_saturated_bands()
    masks = compute_masks(img)
    assert masks["nodata"][:16, :].all()
    assert masks["nodata"][48:, :].all()
    assert not masks["nodata"][16:48, :].any()
    assert not masks["valid"][:16, :].any()
    assert not masks["valid"][48:, :].any()
    assert masks["valid"][16:48, :].all()
    assert not masks["combined"][:16, :].any()
    assert not masks["combined"][48:, :].any()
    s = summary_masks(img)
    assert s["nodata_pct"] == 50.0
    assert s["valid_pct"] == 50.0
    assert s["usable_pct"] == 50.0


def test_nodata_none_skips_nodata_masking():
    img = _fixture_nodata_and_saturated_bands()
    masks = compute_masks(img, nodata_value=None)
    assert not masks["nodata"].any()
    assert masks["valid"][:16, :].all()


def test_float32_input_supported():
    img = np.full((32, 32), 2.0, dtype=np.float32)
    masks = compute_masks(img, nodata_value=0.0)
    assert masks["valid"].all()
    assert not masks["nodata"].any()


def test_float32_nan_does_not_crash():
    img = np.full((32, 32), 2.0, dtype=np.float32)
    img[16, 16] = np.nan
    masks = compute_masks(img, nodata_value=0.0)
    assert masks["combined"].shape == img.shape
    assert not masks["combined"][16, 16]


def test_save_load_roundtrip_exact():
    img = _fixture_half_dark_half_bright()
    masks = compute_masks(img)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "mask.npz"
        save_masks(path, masks)
        loaded = load_masks(path)
        assert set(loaded.keys()) == set(masks.keys())
        for key in masks:
            assert np.array_equal(loaded[key], masks[key])


def test_gate_fixture_bright_tail_no_shadow_latch():
    """Shadow minority rule on the exact unit-10/MG2-style gate fixture.

    Recipe: rng 10, 256x256, bg uniform(60, 190), 150 gaussian rocks clipped
    at 255, uint8. The long bright tail makes between-class variance rise
    monotonically to the saturation edge, so otsu latches at the last bin
    (thr=254) and the naive (img <= thr) verdict marks 98.7% of the image as
    'shadow', starving the pipeline (combined usable = 0.0%). Shadow is by
    definition the dark MINORITY class: a majority-dark otsu verdict is that
    latch, not shadow, so the mask must fall back to empty and combined
    usable must stay well above 0.
    """
    rng = np.random.default_rng(10)
    H = W = 256
    img = rng.uniform(60, 190, (H, W))
    for _ in range(150):
        cx = rng.uniform(0, W - 1)
        cy = rng.uniform(0, H - 1)
        amp = rng.uniform(30, 110)
        rad = rng.uniform(3, 12)
        x0, x1 = int(max(0, cx - rad)), int(min(W, cx + rad))
        y0, y1 = int(max(0, cy - rad)), int(min(H, cy + rad))
        yy, xx = np.mgrid[y0:y1, x0:x1]
        d2 = (xx - cx) ** 2 + (yy - cy) ** 2
        img[y0:y1, x0:x1] = np.minimum(
            255.0, img[y0:y1, x0:x1]
            + amp * np.exp(-d2 / (2 * rad * rad / 4)))
    img = img.astype(np.uint8)

    thr = otsu_threshold(img)
    assert thr == 254.0, thr  # the saturation-edge latch itself
    assert (img <= thr).mean() > 0.5  # latch verdict: dark side is majority

    masks = compute_masks(img)
    assert not masks["shadow"].any()
    assert masks["combined"].mean() > 0.3, masks["combined"].mean()
    s = summary_masks(img)
    assert s["usable_pct"] > 30.0
