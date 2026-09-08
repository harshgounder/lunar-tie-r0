"""TICKET-RD05 gate tests: common-GSD resample module (audit L2)."""

import resource
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from lunar_tie.resample import (
    area_average,
    gauss_pyr_down,
    lanczos3_decimate,
    match_shapes_to_min,
)


def test_area_average_exact():
    img = np.arange(16, dtype=np.float64).reshape(4, 4)
    out = area_average(img, 2)
    assert out.dtype == np.float32
    assert out.shape == (2, 2)
    expected = np.array(
        [[(0 + 1 + 4 + 5) / 4, (2 + 3 + 6 + 7) / 4],
         [(8 + 9 + 12 + 13) / 4, (10 + 11 + 14 + 15) / 4]],
        dtype=np.float32,
    )
    np.testing.assert_allclose(out, expected)


def test_area_average_nondivisible():
    img = np.arange(49, dtype=np.float64).reshape(7, 7)
    out = area_average(img, 2)
    assert out.shape == (3, 3)
    assert out.dtype == np.float32
    # crop to 6x6 first, then block-mean
    np.testing.assert_allclose(
        out, area_average(img[:6, :6], 2)
    )
    np.testing.assert_allclose(out, area_average(np.ascontiguousarray(img[:6, :6]), 2))


def test_gauss_pyr_differs():
    rng = np.random.default_rng(7)
    img = rng.uniform(0, 255, size=(32, 32)).astype(np.float32)
    a = gauss_pyr_down(img, 2)
    b = area_average(img, 2)
    assert a.shape == b.shape
    assert a.dtype == np.float32
    assert not np.allclose(a, b)


def test_match_shapes_identical():
    rng = np.random.default_rng(3)
    a = rng.uniform(0, 255, size=(100, 120)).astype(np.float32)
    b = rng.uniform(0, 255, size=(120, 140)).astype(np.float32)
    a_rs, b_rs, factor = match_shapes_to_min(a, b, gsd_ratio=2)
    assert a_rs.shape == b_rs.shape
    assert a_rs.dtype == np.float32 and b_rs.dtype == np.float32
    assert factor == 2


def test_match_shapes_values():
    rng = np.random.default_rng(11)
    a = rng.uniform(0, 255, size=(20, 20)).astype(np.float32)
    b = rng.uniform(0, 255, size=(40, 40)).astype(np.float32)
    a_rs, b_rs, factor = match_shapes_to_min(a, b, gsd_ratio=2)
    assert a_rs.shape == (20, 20)
    assert b_rs.shape == (20, 20)
    # b (40x40) should be resampled to 20x20 exactly matching a in shape
    assert b_rs.shape == a_rs.shape
    # center-crop of the resampled b side must equal area_average of b itself
    np.testing.assert_allclose(b_rs, area_average(b, 2))


def test_memory_bounded():
    # F8 ru_maxrss pattern: peak RSS must stay below 3x the input size
    # for a 2000x2000 f4 input (~16 MB), so the slack budget is ~32 MB.
    img = np.random.default_rng(0).uniform(
        0, 255, size=(2000, 2000)
    ).astype(np.float32)
    n_bytes = img.nbytes
    before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    out = gauss_pyr_down(img, 4)
    after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    assert out.shape == (500, 500)
    assert out.dtype == np.float32
    # ru_maxrss is in KB on linux
    slack_kb = after - before
    assert slack_kb < 3 * n_bytes / 1024, (before, after)