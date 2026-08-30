"""UNIT-10 gate test: the pipeline CLI (M1 capstone).

Gates:
1. run_pair on a 256x256 synthetic pair (translation dx=4 dy=3, no scale)
   produces panel.json + ties.json + summary.txt on disk, exit code 0.
2. panel.json parses; keys superset of {inlier_ratio, rms_px, q_hat,
   occupancy_ratio, tier_label, provenance}.
3. recovered translation within 0.5 px of truth (median of ties after
   subpixel) - a quality bar, not just "runs".
4. manifest tier_label/provenance PROPAGATE into panel.json unchanged.
5. missing file path in manifest -> stderr message + nonzero exit.
6. manifest with tier_label omitted -> tier_label=0 AND provenance
   "UNLABELED" in panel (honesty default, never silently tier 0).
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

REPO = Path(__file__).resolve().parent.parent
W = H = 256
DX, DY = 4.0, 3.0
SEED = 10


def _write_npy(path, arr):
    np.save(str(path), arr.astype(np.float32))
    return path


def _make_synthetic_pair(tmpdir):
    """256x256 texture pair, B = A shifted by (dx, dy), gain 1.1 offset 5.

    Same recipe as MINI-GATE 2 (uniform background + small bright rocks so
    the DoG detector has gray-level variation to key on), bilinear inverse
    sampling to preserve subpixel structure.
    """
    rng = np.random.default_rng(SEED)
    imgA = rng.uniform(60, 190, (H, W))
    n_rock = 150
    for _ in range(n_rock):
        cx = rng.uniform(0, W - 1)
        cy = rng.uniform(0, H - 1)
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

    yy, xx = np.mgrid[0:H, 0:W]
    coords = np.stack([xx.ravel(), yy.ravel()], axis=1).astype(np.float64)
    # B(x') = A(x' - t) via bilinear inverse sampling
    a_coords = coords - np.array([DX, DY])
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
    a_path = _write_npy(Path(tmpdir) / "a.npy", imgA)
    b_path = _write_npy(Path(tmpdir) / "b.npy", imgB)
    return a_path, b_path


def _run_cli(manifest, outdir):
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO / "src") + os.pathsep + env.get(
        "PYTHONPATH", "")
    return subprocess.run(
        [sys.executable, "-m", "lunar_tie.pipeline", "run_pair",
         str(manifest), "--outdir", str(outdir)],
        capture_output=True, text=True, env=env, timeout=900)


@pytest.fixture(scope="module")
def gate_pair(tmp_path_factory):
    """Synthetic pair + manifest + ONE full chain run; shared by gates 1-4."""
    base = tmp_path_factory.mktemp("gate_pair")
    a_path, b_path = _make_synthetic_pair(base)
    manifest = {
        "a": str(a_path),
        "b": str(b_path),
        "tier_label": 3,
        "provenance": "synthetic-translation-4x3",
    }
    m_path = base / "manifest.json"
    m_path.write_text(json.dumps(manifest))
    outdir = base / "out"
    proc = _run_cli(m_path, outdir)
    return base, m_path, outdir, proc


def test_gate1_run_pair_writes_artifacts_exit0(gate_pair):
    base, m_path, outdir, proc = gate_pair
    assert proc.returncode == 0, (
        f"exit={proc.returncode}\nstdout={proc.stdout}\nstderr={proc.stderr}")
    assert (outdir / "panel.json").is_file()
    assert (outdir / "ties.json").is_file()
    assert (outdir / "summary.txt").is_file()


def test_gate2_panel_keys_superset(gate_pair):
    base, m_path, outdir, proc = gate_pair
    panel = json.loads((outdir / "panel.json").read_text())
    for key in ("inlier_ratio", "rms_px", "q_hat", "occupancy_ratio",
                "tier_label", "provenance"):
        assert key in panel, f"panel.json missing '{key}'"


def test_gate3_recovered_translation_within_half_px(gate_pair):
    base, m_path, outdir, proc = gate_pair
    ties = json.loads((outdir / "ties.json").read_text())
    assert len(ties) >= 3, f"too few ties for a median: {len(ties)}"
    dx = np.median([t["x_b"] - t["x_a"] for t in ties])
    dy = np.median([t["y_b"] - t["y_a"] for t in ties])
    assert abs(dx - DX) <= 0.5, f"dx median {dx:.3f} vs truth {DX}"
    assert abs(dy - DY) <= 0.5, f"dy median {dy:.3f} vs truth {DY}"


def test_gate4_tier_and_provenance_propagate(gate_pair):
    base, m_path, outdir, proc = gate_pair
    manifest = json.loads(m_path.read_text())
    panel = json.loads((outdir / "panel.json").read_text())
    assert panel["tier_label"] == manifest["tier_label"]
    assert panel["provenance"] == manifest["provenance"]


def test_gate5_missing_file_loud_nonzero(gate_pair):
    base, m_path, outdir, proc = gate_pair
    bad_manifest = base / "manifest_bad.json"
    bad_manifest.write_text(json.dumps({
        "a": str(base / "does_not_exist.npy"),
        "b": str(base / "b.npy"),
        "tier_label": 2,
        "provenance": "gate5",
    }))
    proc_bad = _run_cli(bad_manifest, base / "out_bad")
    assert proc_bad.returncode != 0, "missing input must exit nonzero"
    assert proc_bad.stderr.strip(), "missing input must leave a stderr mark"


def test_gate6_unlabeled_manifest_defaults(gate_pair):
    base, m_path, outdir, proc = gate_pair
    unlabeled = base / "manifest_unlabeled.json"
    unlabeled.write_text(json.dumps({
        "a": str(base / "a.npy"),
        "b": str(base / "b.npy"),
    }))
    outdir6 = base / "out_unlabeled"
    proc6 = _run_cli(unlabeled, outdir6)
    assert proc6.returncode == 0, (
        f"exit={proc6.returncode}\nstderr={proc6.stderr}")
    panel = json.loads((outdir6 / "panel.json").read_text())
    assert panel["tier_label"] == 0
    assert panel["provenance"] == "UNLABELED"