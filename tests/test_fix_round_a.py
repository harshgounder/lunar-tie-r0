"""Fix-round A/C gate tests: audit F1-F4 + commander round-3 findings.

Covers: A1 (F1) real inlier mask in panel, A2 (F2) + C1 held-out conformal
split for both gates, A3 (F3) loud tier validation, C3 provenance seed,
C6 step-7 inlier floors. Each test pins a live-repro'd bug from
AUDIT-MUSE-20260905.md / commander round 3.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from lunar_tie.pipeline import PipelineError, run_pair

REPO = Path(__file__).resolve().parent.parent
W = H = 256
SEED = 20


def _write_npy(path, arr):
    np.save(str(path), arr.astype(np.float32))
    return path


def _make_pair_with_outliers(tmpdir, outlier_frac):
    """Pair like the unit-10 gate fixture, plus large block mismatches.

    Planted 'outliers' are big bright rectangles placed at DIFFERENT
    positions in A and B so matches inside them disagree with the global
    similarity, imitating repetitive-terrain gross mismatches.

    Fixture recipe (MG2-style): a real shadow band imgA[:120, :] = 12 painted
    AFTER the rocks gives the pair a genuine dark class (belt + suspenders
    alongside the masking shadow minority rule). Rocks are sampled from rows
    130+ so the band stays intact, and planted outlier blocks all sit BELOW
    row 130 so they live inside the usable (non-shadow) region and actually
    contaminate the matches. Positions keep the original scheme: the same
    count, block size 6, distinct spots per side, and the n_out scaling; the
    four original rows above 130 were shifted to the nearest row below it
    ((40,40)->(140,40), (60,210)->(150,210), (40,200)->(140,200),
    (210,40)->(170,40) in A; (100,100)->(160,100), (30,150)->(170,150) in B),
    the rest are unchanged.
    """
    rng = np.random.default_rng(SEED)
    imgA = rng.uniform(60, 190, (H, W))
    n_rock = 150
    for _ in range(n_rock):
        cx = rng.uniform(0, W - 1)
        cy = rng.uniform(130, H - 1)
        amp = rng.uniform(30, 110)
        rad = rng.uniform(3, 12)
        x0, x1 = int(max(0, cx - rad)), int(min(W, cx + rad))
        y0, y1 = int(max(0, cy - rad)), int(min(H, cy + rad))
        yy, xx = np.mgrid[y0:y1, x0:x1]
        d2 = (xx - cx) ** 2 + (yy - cy) ** 2
        imgA[y0:y1, x0:x1] = np.minimum(
            255.0, imgA[y0:y1, x0:x1]
            + amp * np.exp(-d2 / (2 * rad * rad / 4)))
    imgA = imgA.astype(np.uint8)
    # real shadow band at the top, painted after the rocks (MG2 recipe)
    imgA[:120, :] = 12
    # planted outliers: distinct bright blocks, different spots per side,
    # all BELOW row 130 (inside the usable region, outside the shadow band)
    for (cy, cx) in [(140, 40), (200, 60), (150, 210)]:
        r = 6
        imgA[cy:cy + r, cx:cx + r] = 255
    for (cy, cx) in [(140, 200), (200, 200), (170, 40)]:
        r = 6
        imgA[cy:cy + r, cx:cx + r] = 255

    yy, xx = np.mgrid[0:H, 0:W]
    coords = np.stack([xx.ravel(), yy.ravel()], axis=1).astype(np.float64)
    a_coords = coords - np.array([4.0, 3.0])
    ax = np.clip(a_coords[:, 0], 0, W - 1.001)
    ay = np.clip(a_coords[:, 1], 0, H - 1.001)
    x0 = np.floor(ax).astype(int)
    y0 = np.floor(ay).astype(int)
    x1 = np.minimum(x0 + 1, W - 1)
    y1 = np.minimum(y0 + 1, H - 1)
    wx = ax - x0
    wy = ay - y0
    imgB = (imgA.ravel()[y0 * W + x0] * (1 - wx) * (1 - wy)
            + imgA.ravel()[y0 * W + x1] * wx * (1 - wy)
            + imgA.ravel()[y1 * W + x0] * (1 - wx) * wy
            + imgA.ravel()[y1 * W + x1] * wx * wy).reshape(H, W)
    imgB = (1.1 * imgB + 5).clip(0, 255).astype(np.uint8)
    # outlier blocks in B at different positions, all below row 130
    # (outlier_frac scales the count)
    n_out = int(round(3 * outlier_frac / 0.05))
    for (cy, cx) in [(160, 100), (200, 30), (170, 150)][:max(1, n_out)]:
        imgB[cy:cy + 6, cx:cx + 6] = 255
    a_path = _write_npy(Path(tmpdir) / "a.npy", imgA)
    b_path = _write_npy(Path(tmpdir) / "b.npy", imgB)
    return a_path, b_path


@pytest.fixture(scope="module")
def contaminated_pair(tmp_path_factory):
    base = tmp_path_factory.mktemp("contaminated")
    a_path, b_path = _make_pair_with_outliers(base, outlier_frac=0.05)
    manifest = {
        "a": str(a_path),
        "b": str(b_path),
        "tier_label": 3,
        "provenance": "audit-fix-round",
    }
    m_path = base / "manifest.json"
    m_path.write_text(json.dumps(manifest))
    return base, m_path


@pytest.fixture(scope="module")
def clean_pair(tmp_path_factory):
    base = tmp_path_factory.mktemp("clean")
    a_path, b_path = _make_pair_with_outliers(base, outlier_frac=0.0)
    manifest = {
        "a": str(a_path),
        "b": str(b_path),
        "tier_label": 3,
        "provenance": "audit-fix-round-clean",
    }
    m_path = base / "manifest.json"
    m_path.write_text(json.dumps(manifest))
    return base, m_path


def test_a1_panel_inlier_ratio_measured_and_moves(contaminated_pair, clean_pair):
    """A1 (audit F1): panel numbers measured on the REAL post-refit inlier
    mask (cons2 inliers), intersected with the coverage selection as a real
    measured subset.

    Exact semantics of what the panel counts:
      n_pairs       == size of resid_f, the full post-refit inlier set
                       (src_f/dst_f = refined matches under cons2 inliers)
      n_valid       == number of coverage-selected ties (the subset of
                       resid_f surviving the ANMS spread selection)
      inlier_ratio  == n_valid / n_pairs = ties / n_pairs, so it is < 1.0
                       whenever coverage selects a strict subset, and it
                       moves when contamination changes the inlier pool.
    """
    _, m_path = contaminated_pair
    _, clean_m = clean_pair
    panel_c = run_pair(m_path, str(Path(m_path).parent / "out_a1c"),
                       )[0]
    panel_x = run_pair(clean_m, str(Path(clean_m).parent / "out_a1x"),
                       )[0]
    for panel in (panel_c, panel_x):
        assert panel["n_pairs"] == panel["resid_f_size"], (
            panel["n_pairs"], panel["resid_f_size"])
        assert panel["n_valid"] == panel["n_ties"], (
            panel["n_valid"], panel["n_ties"])
        assert abs(panel["inlier_ratio"]
                   - panel["n_valid"] / panel["n_pairs"]) < 1e-12, panel
    # measured: not the vacuous constant 1.0 in the contaminated run
    assert panel_c["inlier_ratio"] < 1.0, panel_c["inlier_ratio"]
    # and it moves when the outlier fraction changes
    assert panel_x["inlier_ratio"] > panel_c["inlier_ratio"], (
        panel_x["inlier_ratio"], panel_c["inlier_ratio"])


def test_a2_c1_gates_calibrated_on_holdout_not_self(contaminated_pair):
    """A2 (audit F2) + C1: both gates get a real held-out split. The
    three-way ladder must be able to output non-ACCEPT labels and the
    binary accept fraction must be below 1.0 on contaminated data."""
    _, m_path = contaminated_pair
    panel = run_pair(m_path, str(Path(m_path).parent / "out_a2"),
                     )[0]
    gates = panel["coverage_gates"]
    assert gates["accept"] < 1.0, gates
    assert (gates["accept"] + gates["abstain"] + gates["reject"]) == 1.0


def test_a3_invalid_tier_exits_nonzero(contaminated_pair):
    """A3 (audit F3): manifest tier 9 / 3.7 / '3' / true all fail LOUD
    (nonzero exit + stderr), never silently stamped as tier 1."""
    base, m_path = contaminated_pair
    base = Path(m_path).parent
    manifest = json.loads(Path(m_path).read_text())
    for bad in (9, 3.7, "3", True):
        m = dict(manifest)
        m["tier_label"] = bad
        bad_path = base / f"manifest_tier_{repr(bad)}.json"
        bad_path.write_text(json.dumps(m))
        out = base / f"out_tier_{repr(bad)}"
        with pytest.raises(PipelineError):
            run_pair(bad_path, str(out))
    # and a legal tier still runs green
    ok = base / "manifest_tier_ok.json"
    m = dict(manifest)
    m["tier_label"] = 3
    ok.write_text(json.dumps(m))
    panel = run_pair(ok, str(base / "out_tier_ok"))[0]
    assert panel["tier_label"] == 3


def test_a3_cli_invalid_tier_nonzero_exit(contaminated_pair):
    """A3 CLI level: subprocess exits nonzero with a loud stderr mark."""
    base, m_path = contaminated_pair
    base = Path(m_path).parent
    manifest = json.loads(Path(m_path).read_text())
    m = dict(manifest)
    m["tier_label"] = 9
    bad_path = base / "manifest_tier_cli.json"
    bad_path.write_text(json.dumps(m))
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO / "src") + os.pathsep + env.get(
        "PYTHONPATH", "")
    proc = subprocess.run(
        [sys.executable, "-m", "lunar_tie.pipeline", "run_pair",
         str(bad_path), "--outdir", str(base / "out_tier_cli")],
        capture_output=True, text=True, env=env, timeout=900)
    assert proc.returncode != 0
    assert proc.stderr.strip()


def test_c3_panel_provenance_seed_is_pipeline_seed(contaminated_pair):
    """C3: the panel stamps the pipeline's real MAGSAC_SEED (13), not the
    SEED_DEFAULT=0 placeholder."""
    _, m_path = contaminated_pair
    panel = run_pair(m_path, str(Path(m_path).parent / "out_c3"))[0]
    meta = panel.get("provenance_meta", panel.get("provenance"))
    assert meta["seed"] == 13, meta


def test_c6_too_few_valid_refinements_loud(contaminated_pair, monkeypatch):
    """C6a: fewer than 3 valid subpixel refinements is a loud
    PipelineError, not an unhandled numpy error from cons2."""
    _, m_path = contaminated_pair
    import lunar_tie.pipeline as pl

    real_refine = pl.refine_matches

    def fake_refine(*args, **kwargs):
        out = real_refine(*args, **kwargs)
        out["valid"] = np.zeros_like(out["valid"])
        return out

    monkeypatch.setattr(pl, "refine_matches", fake_refine)
    with pytest.raises(PipelineError):
        run_pair(m_path, str(Path(m_path).parent / "out_c6a"))


def test_c6_too_few_cons2_inliers_loud(contaminated_pair, monkeypatch):
    """C6b: a refit consensus with fewer than 3 inliers is a loud
    PipelineError, not a 2-point coverage chain."""
    _, m_path = contaminated_pair
    import lunar_tie.pipeline as pl

    real_magsac = pl.magsac_consensus
    state = {"calls": 0}

    def fake_magsac(*args, **kwargs):
        out = real_magsac(*args, **kwargs)
        state["calls"] += 1
        if state["calls"] == 2:  # the post-refit cons2 call
            inl = out["inliers"].copy()
            keep = np.where(inl)[0][:2]
            slim = np.zeros_like(inl)
            slim[keep] = True
            out["inliers"] = slim
        return out

    monkeypatch.setattr(pl, "magsac_consensus", fake_magsac)
    with pytest.raises(PipelineError):
        run_pair(m_path, str(Path(m_path).parent / "out_c6b"))

# ---------------------------------------------------------------------------
# unit-level reds: A4 / A5 / A6 / C8
# ---------------------------------------------------------------------------

def test_a4_match_descriptors_single_descriptor_side():
    """A4 (audit F4): a legal single-descriptor side returns [] instead of
    an unhandled IndexError (empty-safe contract)."""
    from lunar_tie.detect import match_descriptors
    dA = np.random.default_rng(0).normal(size=(5, 128)).astype(np.float32)
    dB = np.random.default_rng(1).normal(size=(1, 128)).astype(np.float32)
    assert match_descriptors(dA, dB) == []
    assert match_descriptors(dB, dA) == []


def test_a5_binary_gate_nan_raises():
    """A5: conformal_gate raises ValueError on non-finite test scores
    (one loud policy, matching conformal_calibrate)."""
    from lunar_tie.conformal import conformal_gate
    with pytest.raises(ValueError):
        conformal_gate(np.array([0.1, np.nan, 0.3]), 0.2)


def test_a5_three_way_gate_nan_raises():
    """A5: three_way_gate raises ValueError on non-finite test scores
    instead of returning an UNINITIALIZED None label slot."""
    from lunar_tie.conformal import three_way_gate
    with pytest.raises(ValueError):
        three_way_gate([0.1, np.nan, 0.3], [0.1, 0.2, 0.15],
                       np.zeros(3, dtype=bool))


def test_a6_homography_degenerate_raises():
    """A6 (audit F5): a degenerate null vector raises ValueError loudly
    instead of returning inf/nan junk from H / H[2,2]."""
    from lunar_tie.consensus import fit_homography
    src = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    dst = src * 2.0
    with pytest.raises(ValueError):
        fit_homography(src, dst)


def test_c8_panel_alpha_changes_ece_basis():
    """C8: metrics_panel accepts an alpha parameter and the ece_proxy is
    measured against 1 - alpha."""
    from lunar_tie.conformal import metrics_panel
    resid = np.array([0.02, 0.04, 0.06, 0.08, 0.10, 0.30])
    inl = np.array([True, True, True, True, True, False])
    p05 = metrics_panel(resid, inl, q_hat=0.08, tier_label=3)
    p10 = metrics_panel(resid, inl, q_hat=0.08, tier_label=3, alpha=0.10)
    assert p05["ece_proxy"] != p10["ece_proxy"]
    assert abs(p10["ece_proxy"] - abs(p10["inlier_ratio"] - 0.90)) < 1e-12
    # default keeps ALPHA_DEFAULT = 0.05 behavior
    assert abs(p05["ece_proxy"] - abs(p05["inlier_ratio"] - 0.95)) < 1e-12
