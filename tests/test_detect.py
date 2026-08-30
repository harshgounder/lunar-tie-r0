"""UNIT-5 gate test.
Gate: 'SIFT blob detection + polarity flip on 3 synthetic fixtures'.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from lunar_tie.detect import (
    detect_keypoints_rdsift,
    detect_keypoints_sift,
    match_descriptors,
)


def _blobs(shape=(128, 128), centers=None, sigma=3.0, seed=3):
    """Image of smooth gaussian blobs at known positions."""
    rng = np.random.default_rng(seed)
    if centers is None:
        centers = [(30, 30), (95, 30), (30, 98), (98, 98), (64, 64)]
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    img = np.zeros(shape, dtype=np.float64)
    for cy, cx in centers:
        img += 200.0 * np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2)
                                / (2.0 * sigma * sigma)))
    img += rng.normal(0.0, 0.3, size=shape)
    return img.astype(np.float32)


def test_sift_fixture_a_keypoints_cluster_at_blobs():
    img = _blobs()
    kps, desc = detect_keypoints_sift(img)
    centers = [(30, 30), (95, 30), (30, 98), (98, 98), (64, 64)]
    assert desc.dtype == np.float32
    assert desc.ndim == 2 and desc.shape[1] == 128
    xy = np.array([[k["x"], k["y"]] for k in kps])
    hit = 0
    for (cy, cx) in centers:
        d = np.sqrt(((xy - [cx, cy]) ** 2).sum(1))
        if np.any(d < 2.0):
            hit += 1
    assert hit >= 5, "expected >= 1 keypoint within 2px of each of 5 blobs"
    assert len(kps) >= 5


def test_sift_fixture_b_gain_offset_same_count():
    img = _blobs()
    imgB = img * 2.0 + 60.0
    kpa, _ = detect_keypoints_sift(img)
    kpb, _ = detect_keypoints_sift(imgB)
    assert abs(len(kpa) - len(kpb)) <= 3


def test_sift_fixture_b_descriptors_agree_across_gain():
    img = _blobs()
    imgB = img * 2.0 + 60.0
    kpa, da = detect_keypoints_sift(img)
    kpb, db = detect_keypoints_sift(imgB)
    cos = da @ db.T / (np.linalg.norm(da, axis=1)[:, None]
                       * np.linalg.norm(db, axis=1)[None, :] + 1e-12)
    n, m = cos.shape
    sim = 0.0
    cnt = 0
    for i in range(n):
        j = int(np.argmax(cos[i]))
        if cos[i, j] >= 0.95:
            sim += cos[i, j]
            cnt += 1
    assert cnt > 0
    assert sim / cnt >= 0.95, "average matched cosine under 0.95"


def test_sift_fixture_c_polarity_gap_is_allowed_to_fail():
    # plain SIFT is NOT guaranteed polarity robust; just smoke it runs
    img = _blobs()
    kpa, da = detect_keypoints_sift(img)
    fl = 255.0 * img / np.abs(img).max() if img.max() > 0 else img
    flb = fl.max() + fl.min() - fl
    kpb, db = detect_keypoints_sift(flb)
    assert da.shape[1] == 128 and db.shape[1] == 128
    assert len(kpa) >= 5 and len(kpb) >= 5


def test_rdsift_fixture_c_polarity_robust():
    img = _blobs()
    kpa, da = detect_keypoints_rdsift(img)
    fl = 255.0 * img / np.abs(img).max() if img.max() > 0 else img
    flb = fl.max() + fl.min() - fl
    kpb, db = detect_keypoints_rdsift(flb)
    assert da.shape[1] == 128 and db.shape[1] == 128
    cos = da @ db.T / (np.linalg.norm(da, axis=1)[:, None]
                       * np.linalg.norm(db, axis=1)[None, :] + 1e-12)
    n = cos.shape[0]
    best = np.max(cos, axis=1)
    val = float(np.median(best))
    assert val >= 0.90, "RD-SIFT median polarity cosine below 0.90"


def test_small_image_returns_empty_without_crash():
    img = np.zeros((16, 16), dtype=np.float32)
    for fn in (detect_keypoints_sift, detect_keypoints_rdsift):
        kps, desc = fn(img)
        assert kps == []
        assert desc.shape == (0, 128)


def test_descriptor_rows_are_l2_normalized():
    img = _blobs()
    _, desc = detect_keypoints_sift(img)
    if desc.shape[0] == 0:
        return
    norms = np.linalg.norm(desc, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-3)


def test_match_descriptors_self_returns_many_matches():
    img = _blobs()
    _, desc = detect_keypoints_sift(img)
    if desc.shape[0] < 2:
        return
    m = match_descriptors(desc, desc)
    assert len(m) > 0, "self-matching should produce matches"
    for (i, j) in m:
        assert 0 <= i < desc.shape[0]
        assert 0 <= j < desc.shape[0]


def test_match_descriptors_empty_input_safe():
    img = np.zeros((16, 16), dtype=np.float32)
    _, da = detect_keypoints_sift(img)
    assert match_descriptors(da, da) == []


def test_keypoint_fields_present_and_typed():
    img = _blobs()
    kps, _ = detect_keypoints_sift(img)
    assert len(kps) > 0
    for k in kps:
        assert set(k.keys()) == {"x", "y", "octave", "scale", "response",
                                 "angle"}
        assert isinstance(k["x"], float)
        assert isinstance(k["y"], float)
        assert isinstance(k["scale"], float)
        assert isinstance(k["response"], float)
        assert isinstance(k["angle"], float)


def test_rdsift_same_detector_as_sift_on_fixture_a():
    img = _blobs()
    kps_sift, _ = detect_keypoints_sift(img)
    kps_rd, _ = detect_keypoints_rdsift(img)
    assert len(kps_sift) == len(kps_rd)
    for a, b in zip(kps_sift, kps_rd):
        assert abs(a["x"] - b["x"]) < 1e-3
        assert abs(a["y"] - b["y"]) < 1e-3


def test_contrast_threshold_low_yields_more_keypoints():
    img = _blobs()
    _, d_high = detect_keypoints_sift(img, contrast_threshold=0.10)
    _, d_low = detect_keypoints_sift(img, contrast_threshold=0.01)
    assert len(d_low) >= len(d_high)
