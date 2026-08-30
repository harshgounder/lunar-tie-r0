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
    assert s["usable_pct"] < 50.0


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
