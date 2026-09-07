"""TICKET-RD02 tests: geometry.py, TMC-2 geometry_calibrated CSV loader.

Split from the crop-first ingest tests so each module has its own gate file.
All synthetic: a year-bundle zip with 2 CSVs matched by timestamp prefix,
a streaming RAM-bound parse, and a nearest-lookup window. No em dashes.
"""

import resource
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from lunar_tie.geometry import (
    find_geometry_csv,
    load_geometry_grid,
    window_from_geometry,
)

GEO_HEADER = "time,scan,pix,lon,lat\n"
GEO_TIMESTAMP = "20260901T0000"


def _geo_csv_text(n_scan=8, n_pix=4):
    rows = [GEO_HEADER]
    for scan in range(n_scan):
        for pix in range(n_pix):
            lon = 20.0 + 6.0 * pix / (n_pix - 1)
            lat = 10.0 + 4.0 * scan / (n_scan - 1)
            rows.append(
                f"{GEO_TIMESTAMP}T{scan:02d}{pix:02d},"
                f"{scan},{pix},{lon:.6f},{lat:.6f}\n")
    return "".join(rows)


def _geo_bundle(tmp_path, name="geometry_2026.zip"):
    zpath = str(tmp_path / name)
    with zipfile.ZipFile(zpath, "w") as zf:
        zf.writestr(f"tmc2_geom_{GEO_TIMESTAMP}_a.csv", _geo_csv_text())
        zf.writestr("tmc2_geom_20270101T0000_b.csv", _geo_csv_text(n_scan=5))
    return zpath


def test_find_geometry_csv_in_zip(tmp_path):
    zpath = _geo_bundle(tmp_path)
    p = find_geometry_csv(zpath, GEO_TIMESTAMP)
    assert p is not None
    assert Path(p).is_file()
    assert Path(p).read_text().startswith(GEO_HEADER)


def test_find_geometry_csv_in_extracted_dir(tmp_path):
    d = tmp_path / "geom_dir"
    d.mkdir()
    (d / f"tmc2_geom_{GEO_TIMESTAMP}_a.csv").write_text(_geo_csv_text())
    (d / "tmc2_geom_20270101T0000_b.csv").write_text(_geo_csv_text(n_scan=5))
    p = find_geometry_csv(str(d), GEO_TIMESTAMP)
    assert p is not None
    assert Path(p).is_file()


def test_find_geometry_csv_no_match_returns_none(tmp_path):
    zpath = _geo_bundle(tmp_path)
    assert find_geometry_csv(zpath, "19990101T0000") is None


def test_load_geometry_grid_full(tmp_path):
    d = tmp_path / "geom_dir"
    d.mkdir()
    csv_path = d / f"tmc2_geom_{GEO_TIMESTAMP}_a.csv"
    csv_path.write_text(_geo_csv_text())
    grid = load_geometry_grid(str(csv_path))
    assert grid["n"] == 32
    assert grid["lon"].shape == (32,)
    assert abs(grid["lon"][0] - 20.0) < 1e-9
    assert abs(grid["lat"][0] - 10.0) < 1e-9
    assert grid["scan"][0] == 0 and grid["pix"][0] == 0
    assert grid["scan"][-1] == 7 and grid["pix"][-1] == 3


def test_load_geometry_grid_bbox_subset(tmp_path):
    d = tmp_path / "geom_dir"
    d.mkdir()
    csv_path = d / f"tmc2_geom_{GEO_TIMESTAMP}_a.csv"
    csv_path.write_text(_geo_csv_text())
    grid = load_geometry_grid(str(csv_path), bbox=(11.0, 14.0, 21.0, 26.0))
    assert grid["n"] == 18
    assert grid["lat"].min() >= 11.0
    assert grid["lat"].max() <= 14.0
    assert grid["lon"].min() >= 21.0
    assert grid["lon"].max() <= 26.0


def test_load_geometry_grid_ram_bounded_streaming(tmp_path):
    """Streaming read: a >12 MB CSV parsed without full-file RAM (F8 pattern)."""
    d = tmp_path / "geom_dir"
    d.mkdir()
    csv_path = d / f"tmc2_geom_{GEO_TIMESTAMP}_big.csv"
    n_scan, n_pix = 1200, 600
    with open(csv_path, "w") as f:
        f.write(GEO_HEADER)
        for scan in range(n_scan):
            for pix in range(n_pix):
                f.write(f"t{scan:05d}{pix:04d},{scan},{pix},"
                        f"{20.0 + pix / 1000.0:.6f},{10.0 + scan / 1000.0:.6f}\n")
    before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    grid = load_geometry_grid(str(csv_path))
    after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    assert grid["n"] == n_scan * n_pix
    file_kb = csv_path.stat().st_size / 1024
    assert file_kb > 12288
    delta_kb = after - before
    # output arrays are 4 x 720000 float64 (~23 MB); a readlines() of the
    # 13 MB text file would add far more on top (row tuple lists blow up).
    assert delta_kb < 64 * 1024, (
        f"load_geometry_grid grew RSS by {delta_kb} KB; streaming is broken")


def test_window_from_geometry_known_mapping():
    n_scan, n_pix = 8, 4
    grid = {
        "lon": np.array([20.0 + 6.0 * p / (n_pix - 1)
                         for _s in range(n_scan) for p in range(n_pix)]),
        "lat": np.array([10.0 + 4.0 * s / (n_scan - 1)
                         for s in range(n_scan) for _p in range(n_pix)]),
        "scan": np.array([s for s in range(n_scan) for _p in range(n_pix)]),
        "pix": np.array([p for _s in range(n_scan) for p in range(n_pix)]),
    }
    line, sample = window_from_geometry(grid, 12.0, 12.0, 23.5, 23.5)
    # nearest row: lat 12 -> scan 4, lon 23.5 -> pix 2 (lon 24, d=0.5)
    assert line == 4
    assert sample == 2


def test_window_from_geometry_outside_raises():
    grid = {
        "lon": np.array([20.0, 21.0]),
        "lat": np.array([10.0, 10.0]),
        "scan": np.array([0, 0]),
        "pix": np.array([0, 1]),
    }
    with pytest.raises(ValueError):
        window_from_geometry(grid, 40.0, 41.0, 20.0, 21.0)