"""UNIT-9 gate test. Gate: 'conformal gate on 3 synthetic fixtures'."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from lunar_tie.conformal import (
    conformal_calibrate,
    conformal_gate,
    export_panel,
    load_panel,
    metrics_panel,
    summarize_zoo_candidate,
    three_way_gate,
)


@pytest.fixture
def cal_residuals():
    rng = np.random.default_rng(0)
    return rng.normal(0.1, 0.05, size=400)


@pytest.fixture
def three_way_inputs():
    scores_cal = np.linspace(0.01, 0.08, 20)
    outliers = np.zeros(20, dtype=bool)
    scores_test = np.array([0.01, 0.05, 0.09, 0.11, 0.5])
    return scores_cal, outliers, scores_test


@pytest.fixture
def sample_panel():
    residuals = np.array([0.02, 0.04, 0.06, 0.08, 0.10, 0.30])
    inliers = np.array([True, True, True, True, True, False])
    return metrics_panel(residuals, inliers, q_hat=0.08, tier_label=3)


def test_gate_a_qhat_near_q95(cal_residuals):
    q_hat = conformal_calibrate(cal_residuals, alpha=0.05)
    q95 = np.quantile(cal_residuals, 0.95)
    assert abs(q_hat - q95) <= 0.02


def test_gate_a_coverage_on_fresh(cal_residuals):
    q_hat = conformal_calibrate(cal_residuals, alpha=0.05)
    rng = np.random.default_rng(1)
    fresh = rng.normal(0.1, 0.05, size=2000)
    coverage = float(np.mean(fresh <= q_hat))
    assert coverage >= 0.93


def test_gate_a_qhat_finite_and_positive(cal_residuals):
    q_hat = conformal_calibrate(cal_residuals, alpha=0.05)
    assert np.isfinite(q_hat)
    assert q_hat > 0.0


def test_calibrate_empty_raises():
    with pytest.raises(ValueError):
        conformal_calibrate(np.array([]))


def test_calibrate_nonfinite_raises():
    with pytest.raises(ValueError):
        conformal_calibrate(np.array([0.1, np.nan, 0.2]))


def test_gate_b_three_way_ladder(three_way_inputs):
    scores_cal, outliers, scores_test = three_way_inputs
    labels = three_way_gate(scores_test, scores_cal, outliers, alpha=0.05,
                            abstain_penalty=1.5)
    expected = ["ACCEPT", "ACCEPT", "ABSTAIN", "ABSTAIN", "REJECT"]
    assert list(labels) == expected


def test_gate_b_band_edges(three_way_inputs):
    scores_cal, outliers, _ = three_way_inputs
    q_hat = conformal_calibrate(scores_cal, alpha=0.05)
    hi = q_hat * 1.5
    edge = np.array([q_hat, q_hat + 1e-9, hi, hi + 1e-9])
    labels = three_way_gate(edge, scores_cal, outliers, alpha=0.05,
                            abstain_penalty=1.5)
    assert list(labels) == ["ACCEPT", "ABSTAIN", "ABSTAIN", "REJECT"]


def test_gate_b_empty_test_no_crash(three_way_inputs):
    scores_cal, outliers, _ = three_way_inputs
    labels = three_way_gate(np.array([]), scores_cal, outliers)
    assert labels.size == 0


def test_accept_when_score_below_qhat():
    scores_cal = np.linspace(0.01, 0.05, 20)
    outliers = np.zeros(20, dtype=bool)
    labels = three_way_gate(np.array([0.005, 0.03]), scores_cal, outliers)
    assert labels[0] == "ACCEPT"


def test_conformal_gate_binary():
    labels = conformal_gate(np.array([0.05, 0.20]), q_hat=0.10)
    assert list(labels) == ["ACCEPT", "REJECT"]


def test_conformal_gate_sigma_thr():
    labels = conformal_gate(np.array([0.05, 0.20]), q_hat=0.30, sigma_thr=0.10)
    assert list(labels) == ["ACCEPT", "REJECT"]


def test_conformal_gate_empty():
    labels = conformal_gate(np.array([]), q_hat=0.10)
    assert labels.size == 0


def test_gate_c_round_trip(tmp_path, sample_panel):
    path = tmp_path / "panel.json"
    export_panel(sample_panel, path)
    loaded = load_panel(path)
    assert loaded == sample_panel


def test_gate_c_round_trip_nested_provenance(tmp_path, sample_panel):
    path = tmp_path / "panel2.json"
    export_panel(sample_panel, path)
    loaded = load_panel(path)
    assert loaded["provenance"] == sample_panel["provenance"]
    assert loaded["coverage_gates"] == sample_panel["coverage_gates"]


def test_gate_c_missing_tier_raises():
    with pytest.raises(ValueError):
        metrics_panel(np.array([0.1]), np.array([True]), q_hat=0.1,
                      tier_label=9)


def test_panel_fields(sample_panel):
    for k in ("n_pairs", "n_valid", "inlier_ratio", "rms_px", "rms_subpx",
              "q_hat", "ece_proxy", "coverage_gates", "tier_label",
              "gate_name", "provenance"):
        assert k in sample_panel


def test_summarize_zoo_candidate(sample_panel):
    row = summarize_zoo_candidate(sample_panel)
    assert row["stratum_key"] == "PLACEHOLDER-STRATUM"
    assert row["config"] == "PLACEHOLDER-CONFIG"
    assert row["gate_evidence"]["gate_name"] == sample_panel["gate_name"]
    assert row["gate_evidence"]["tier_label"] == sample_panel["tier_label"]
