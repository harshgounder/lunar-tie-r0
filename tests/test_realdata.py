"""TICKET-RD02 tests: realdata.py crop-first ingest (all synthetic, real grammar).

Fixtures mimic the ISDA product grammar with ZERO real bytes:
- synthetic product zip: fake .img (u2), PDS4 .xml label copied from
  tests/fixtures/pds4_minimal.xml structure (dims 64x48), .png stub,
  .spm stub.
- a big-member zip for the RAM-bound streaming check (F8 pattern:
  ru_maxrss delta must stay far below the full member size).
- geometry year-bundle zips with 2 CSVs, matched by timestamp prefix.

No em dashes. numpy + stdlib only.
"""

import json
import os
import resource
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from lunar_tie.realdata import (
    crop_pair,
    crop_strip,
    extract_product,
    region_window,
)

REPO = Path(__file__).resolve().parent.parent
LINES, SAMPLES = 128, 160
SEED = 7


def _label_xml(lines=LINES, samples=SAMPLES,
               stem="ch2_tmc_f_20260901T0000000000_1_c"):
    """PDS4 label with the REAL structure of tests/fixtures/pds4_minimal.xml
    plus a corner-coordinate window (the crop-first georeference input)."""
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<Product_Observational xmlns="http://pds.nasa.gov/pds4/pds/v1">
  <Identification_Area>
    <logical_identifier>urn:isro:ch2:tmc2:data:rd02</logical_identifier>
  </Identification_Area>
  <File_Area_Observational>
    <File>
      <file_name>{stem}.img</file_name>
    </File>
    <Array_2D_Image>
      <data_type>UnsignedLSB2</data_type>
      <Axis_Array>
        <axis_name>Line</axis_name>
        <elements>{lines}</elements>
      </Axis_Array>
      <Axis_Array>
        <axis_name>Sample</axis_name>
        <elements>{samples}</elements>
      </Axis_Array>
    </Array_2D_Image>
  </File_Area_Observational>
  <Spice_Window>
    <Latitude>
      <corner_1_latitude>14.0</corner_1_latitude>
      <corner_2_latitude>14.0</corner_2_latitude>
      <corner_3_latitude>10.0</corner_3_latitude>
      <corner_4_latitude>10.0</corner_4_latitude>
    </Latitude>
    <Longitude>
      <corner_1_longitude>20.0</corner_1_longitude>
      <corner_2_longitude>26.0</corner_2_longitude>
      <corner_3_longitude>26.0</corner_3_longitude>
      <corner_4_longitude>20.0</corner_4_longitude>
    </Longitude>
  </Spice_Window>
  <File_Area_Misc>
    <File>
      <file_name>ch2_tmc_spm_20260901T0000000000_1_c.000</file_name>
    </File>
  </File_Area_Misc>
</Product_Observational>"""


def _make_img_arr(lines=LINES, samples=SAMPLES, seed=SEED, rocks=True):
    """Textured uint16 strip: noise floor + bright rocks (DoG needs variation).

    The rock field is SHARED across seeds (same rng stream for positions)
    so the two sides of a synthetic pair keep common texture to match on;
    the seed varies only the fine detail. No geometric shift: the M1 chain
    fits a near-identity similarity, which is the synthetic-crop smoke run.
    """
    rng_rocks = np.random.default_rng(SEED)
    positions = []
    for _ in range(200):
        positions.append((rng_rocks.uniform(0, samples - 1),
                          rng_rocks.uniform(0, lines - 1),
                          rng_rocks.uniform(1000, 20000),
                          rng_rocks.uniform(2, 6)))
    rng = np.random.default_rng(seed)
    img = rng.uniform(60, 190, (lines, samples))
    if rocks:
        for cx, cy, amp, rad in positions:
            x0 = int(max(0, cx - rad))
            x1 = int(min(samples, cx + rad))
            y0 = int(max(0, cy - rad))
            y1 = int(min(lines, cy + rad))
            yy, xx = np.mgrid[y0:y1, x0:x1]
            d2 = (xx - cx) ** 2 + (yy - cy) ** 2
            img[y0:y1, x0:x1] += amp * np.exp(-d2 / (2 * rad * rad / 4))
    return np.clip(img, 0, 65535).astype("<u2")


def _product_zip(tmp_path, name="ch2_tmc_f_20260901T0000000000_1_c.zip",
                 img=None, lines=LINES, samples=SAMPLES, seed=SEED,
                 geometry_csv=None):
    """Synthetic product zip. The default zip NAME carries the real Table 9
    product-ID grammar (RD07 parses the strip timestamp from the basename):
    ch2_<inst>_<mtc>_<timestamp>_<p>_<prd>_<stn>.zip."""
    zpath = str(tmp_path / name)
    if img is None:
        img = _make_img_arr(lines=lines, samples=samples, seed=seed)
    with zipfile.ZipFile(zpath, "w") as zf:
        zf.writestr("ch2_tmc_f_20260901T0000000000_1_c.img", img.tobytes())
        zf.writestr("ch2_tmc_f_20260901T0000000000_1_c.xml", _label_xml(lines, samples))
        zf.writestr("ch2_tmc_f_20260901T0000000000_1_c.png", b"\x89PNG fake")
        zf.writestr("ch2_tmc_spm_20260901T0000000000_1_c.000", b"1 2 3\n")
        if geometry_csv is not None:
            zf.writestr(
                "geometry/calibrated/20260901/"
                "ch2_tmc_f_20260901T0000000000_1_c_g_grd_d18.csv",
                geometry_csv)
    return zpath


# ---------------------------------------------------------------------------
# extract_product
# ---------------------------------------------------------------------------

def test_extract_product_streams_img_to_dest(tmp_path):
    zpath = _product_zip(tmp_path)
    dest = tmp_path / "extracted"
    out = extract_product(zpath, dest, member_kind="img")
    p = Path(out)
    assert p.is_file()
    assert p.stat().st_size == LINES * SAMPLES * 2
    arr = np.fromfile(str(p), dtype="<u2").reshape(LINES, SAMPLES)
    assert int(arr[0, 0]) == int(_make_img_arr()[0, 0])
    assert int(arr[47, 63]) == int(_make_img_arr()[47, 63])


def test_extract_product_member_kind_xml(tmp_path):
    zpath = _product_zip(tmp_path)
    out = extract_product(zpath, tmp_path / "x", member_kind="xml")
    assert Path(out).suffix == ".xml"
    assert Path(out).read_bytes() == _label_xml().encode()


def test_extract_product_unknown_kind_raises(tmp_path):
    zpath = _product_zip(tmp_path)
    with pytest.raises(ValueError):
        extract_product(zpath, tmp_path / "y", member_kind="nonexistent")


def test_extract_product_ram_bounded_big_member(tmp_path):
    """F8 pattern: ru_maxrss delta must stay far below the streamed member.

    A big synthetic .img (24 MB) is streamed in 4 MB chunks: copying it
    must not add the full member size to peak RSS.
    """
    big_lines = 1024
    big_samples = 6000
    big_bytes = big_lines * big_samples * 2
    assert big_bytes == 12_288_000
    img = np.zeros((big_lines, big_samples), dtype="<u2")
    img[::100] = 12345
    zpath = _product_zip(tmp_path, name="big.zip", img=img,
                         lines=big_lines, samples=big_samples)
    before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    out = extract_product(zpath, tmp_path / "big", member_kind="img")
    after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    assert Path(out).stat().st_size == big_bytes
    delta_kb = after - before
    # 4 MB chunk bound + interpreter slack; a whole-member read would add
    # ~12 MB (12000 KB) on top of the chunked path.
    assert delta_kb < 8192, (
        f"extract_product grew RSS by {delta_kb} KB; streaming is broken")


# ---------------------------------------------------------------------------
# crop_strip
# ---------------------------------------------------------------------------

def test_crop_strip_memmap_slice_float32_roundtrip(tmp_path):
    zpath = _product_zip(tmp_path)
    dest = tmp_path / "x"
    img_path = extract_product(zpath, dest)
    from lunar_tie.ch2_ingest import ingest_product, read_strip
    info = ingest_product(zpath)
    mm = read_strip(img_path, info["label"])
    manifest = crop_strip(mm, info["label"], 10, 30, 20, 50,
                          tmp_path / "crop_a.npy")
    del mm
    arr = np.load(str(manifest["a"]), mmap_mode="r")
    assert arr.dtype == np.float32
    assert arr.shape == (20, 30)
    assert int(manifest["tier_label"]) == 0
    assert manifest["provenance"] == "ISDA-REAL"
    assert np.array_equal(arr, _make_img_arr()[10:30, 20:50].astype(np.float32))
    del arr


def test_crop_strip_accepts_path(tmp_path):
    """memmap_or_path: a str/Path input opens its own memmap via the label."""
    zpath = _product_zip(tmp_path)
    img_path = extract_product(zpath, tmp_path / "x")
    from lunar_tie.ch2_ingest import ingest_product
    info = ingest_product(zpath)
    manifest = crop_strip(str(img_path), info["label"], 0, 8, 0, 8,
                          tmp_path / "crop_b.npy")
    arr = np.load(str(manifest["a"]), mmap_mode="r")
    assert arr.shape == (8, 8)
    assert np.array_equal(arr, _make_img_arr()[:8, :8].astype(np.float32))
    del arr


def test_crop_strip_bounds_raise(tmp_path):
    zpath = _product_zip(tmp_path)
    from lunar_tie.ch2_ingest import ingest_product, read_strip
    info = ingest_product(zpath)
    img_path = extract_product(zpath, tmp_path / "x")
    mm = read_strip(img_path, info["label"])
    with pytest.raises(ValueError):
        crop_strip(mm, info["label"], LINES - 10, LINES + 10, 0, 10,
                   tmp_path / "c.npy")
    with pytest.raises(ValueError):
        crop_strip(mm, info["label"], -5, 10, 0, 10, tmp_path / "c.npy")
    with pytest.raises(ValueError):
        crop_strip(mm, info["label"], 0, 10, SAMPLES + 1, SAMPLES + 5,
                   tmp_path / "c.npy")
    del mm


# ---------------------------------------------------------------------------
# region_window
# ---------------------------------------------------------------------------

def _corner_label(lat0=10.0, lat1=14.0, lon0=20.0, lon1=26.0):
    """Synthetic PDS4 label with 4 corner coordinates (real corner grammar).

    Corners: UL = (lat1, lon0), UR = (lat1, lon1), LL = (lat0, lon0),
    LR = (lat0, lon1) mapping linearly onto the 48x64 strip.
    """
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<Product_Observational xmlns="http://pds.nasa.gov/pds4/pds/v1">
  <Identification_Area>
    <logical_identifier>urn:isro:ch2:tmc2:data:rd02</logical_identifier>
  </Identification_Area>
  <File_Area_Observational>
    <Array_2D_Image>
      <data_type>UnsignedLSB2</data_type>
      <Axis_Array>
        <axis_name>Line</axis_name>
        <elements>{LINES}</elements>
      </Axis_Array>
      <Axis_Array>
        <axis_name>Sample</axis_name>
        <elements>{SAMPLES}</elements>
      </Axis_Array>
    </Array_2D_Image>
  </File_Area_Observational>
  <Spice_Window>
    <Latitude>
      <corner_1_latitude>{lat1}</corner_1_latitude>
      <corner_2_latitude>{lat1}</corner_2_latitude>
      <corner_3_latitude>{lat0}</corner_3_latitude>
      <corner_4_latitude>{lat0}</corner_4_latitude>
    </Latitude>
    <Longitude>
      <corner_1_longitude>{lon0}</corner_1_longitude>
      <corner_2_longitude>{lon1}</corner_2_longitude>
      <corner_3_longitude>{lon1}</corner_3_longitude>
      <corner_4_longitude>{lon0}</corner_4_longitude>
    </Longitude>
  </Spice_Window>
</Product_Observational>"""


def _write_corner_label(tmp_path, **corners):
    p = tmp_path / "corner_label.xml"
    p.write_text(_corner_label(**corners))
    from lunar_tie.pds4label import parse_label
    return parse_label(str(p))


def test_region_window_full_strip(tmp_path):
    label = _write_corner_label(tmp_path)
    w = region_window(label, None)
    assert w == (0, LINES, 0, SAMPLES)


def test_region_window_corner_interp_center(tmp_path):
    label = _write_corner_label(tmp_path)
    # the exact center of the bbox maps to the exact center of the strip
    w = region_window(label, 12.0, 12.0, 23.0, 23.0)
    l0, l1, s0, s1 = w
    assert abs(l0 - LINES / 2) <= 1.0
    assert abs(l1 - LINES / 2) <= 1.0
    assert abs(s0 - SAMPLES / 2) <= 1.0
    assert abs(s1 - SAMPLES / 2) <= 1.0


def test_region_window_corner_interp_quarter(tmp_path):
    label = _write_corner_label(tmp_path)
    # bbox covering the top-left quarter of the footprint
    w = region_window(label, 12.0, 14.0, 20.0, 23.0)
    l0, l1, s0, s1 = w
    assert l0 == 0
    assert abs(l1 - LINES / 2) <= 1.0
    assert s0 == 0
    assert abs(s1 - SAMPLES / 2) <= 1.0


def test_region_window_clamps_to_bounds(tmp_path):
    label = _write_corner_label(tmp_path)
    # a bbox extending past every edge must clamp, never go negative
    w = region_window(label, 0.0, 30.0, 10.0, 40.0)
    l0, l1, s0, s1 = w
    assert l0 == 0
    assert l1 <= LINES
    assert s0 == 0
    assert s1 <= SAMPLES


def test_region_window_beyond_footprint_raises(tmp_path):
    label = _write_corner_label(tmp_path)
    with pytest.raises(ValueError):
        region_window(label, 40.0, 50.0, 20.0, 26.0)


def test_region_window_accepts_corners_dict():
    """label_or_corners: a plain corners dict works without a full label."""
    corners = {
        "lines": LINES, "samples": SAMPLES,
        "corner_1_latitude": "14.0", "corner_2_latitude": "14.0",
        "corner_3_latitude": "10.0", "corner_4_latitude": "10.0",
        "corner_1_longitude": "20.0", "corner_2_longitude": "26.0",
        "corner_3_longitude": "26.0", "corner_4_longitude": "20.0",
    }
    w = region_window(corners, 12.0, 12.0, 23.0, 23.0)
    l0, l1, s0, s1 = w
    assert abs(l0 - LINES / 2) <= 1.0
    assert abs(s1 - SAMPLES / 2) <= 1.0


# ---------------------------------------------------------------------------
# crop_pair
# ---------------------------------------------------------------------------

def test_crop_pair_same_window_manifest(tmp_path):
    za = _product_zip(tmp_path, name="ch2_tmc_f_20260901T0000000000_1_a.zip")
    zb = _product_zip(tmp_path, name="ch2_tmc_f_20260901T0000000000_1_b.zip", seed=SEED + 1)
    manifest = crop_pair(za, zb, (12.0, 14.0, 20.0, 23.0),
                         out_dir=tmp_path / "crops")
    arrA = np.load(manifest["a"], mmap_mode="r")
    arrB = np.load(manifest["b"], mmap_mode="r")
    assert arrA.shape == arrB.shape
    assert arrA.dtype == np.float32
    assert arrB.dtype == np.float32
    assert manifest["tier_label"] == 0
    assert manifest["provenance"] == "ISDA-REAL"
    del arrA
    del arrB


def test_crop_pair_manifest_round_trips_run_pair(tmp_path):
    """The manifest crop_pair writes must satisfy run_pair's loader."""
    za = _product_zip(tmp_path, name="ch2_tmc_f_20260901T0000000000_1_a.zip")
    zb = _product_zip(tmp_path, name="ch2_tmc_f_20260901T0000000000_1_b.zip", seed=SEED + 1)
    manifest = crop_pair(za, zb, (12.0, 14.0, 20.0, 23.0),
                         out_dir=tmp_path / "crops")
    m_path = tmp_path / "manifest.json"
    m_path.write_text(json.dumps(manifest))
    from lunar_tie.pipeline import load_manifest
    loaded = load_manifest(str(m_path))
    assert loaded["a"] == manifest["a"]
    assert loaded["b"] == manifest["b"]


# ---------------------------------------------------------------------------
# end-to-end gate (definition of done): crop_pair -> run_pair -> panel.json
# ---------------------------------------------------------------------------

def test_crop_pair_to_run_pair_end_to_end(tmp_path):
    """THE gate: crop both synthetic sides, run the 10-step chain, panel.json."""
    from lunar_tie.pipeline import run_pair

    za = _product_zip(tmp_path, name="ch2_tmc_f_20260901T0000000000_1_a.zip")
    zb = _product_zip(tmp_path, name="ch2_tmc_f_20260901T0000000000_1_b.zip", seed=SEED + 1)
    manifest = crop_pair(za, zb, (12.0, 14.0, 20.0, 23.0),
                         out_dir=tmp_path / "crops")
    m_path = tmp_path / "manifest.json"
    m_path.write_text(json.dumps(manifest))
    outdir = tmp_path / "out"
    panel, ties = run_pair(str(m_path), str(outdir))
    assert (outdir / "panel.json").is_file()
    assert (outdir / "ties.json").is_file()
    assert panel["provenance"] == "ISDA-REAL"
    assert panel["tier_label"] == 0


# ---------------------------------------------------------------------------
# TICKET-RD07: geometry-grid georef preference (audit L3)
# ---------------------------------------------------------------------------

def _geometry_csv_text(rows):
    """A geometry_calibrated CSV with the REAL header grammar."""
    out = ["lon,lat,scan,pix"]
    for lon, lat, scan, pix in rows:
        out.append("%s,%s,%s,%s" % (lon, lat, scan, pix))
    return "\n".join(out) + "\n"


def _grid_rows_rotated(lines=LINES, samples=SAMPLES, lat_top=14.0, lat_bot=10.0,
                       lon_left=20.0, lon_right=26.0):
    """NON-axis-aligned footprint rows: latitude twists linearly along the
    top edge (limb shear), so corner min/max and true grid rows disagree.

    scan = line, pix = sample. Row r line l runs
    lon = lon_left + lon_span * (x + 0.10 * y) and
    lat = lat_top - lat_span * y, with x = pix/(samples-1),
    y = scan/(lines-1), both in [0, 1]. The +0.10 skew tilts the footprint
    so a bbox corner falls OUTSIDE the true quad while still inside the
    corner min/max box.
    """
    rows = []
    lon_span = lon_right - lon_left
    lat_span = lat_top - lat_bot
    for scan in range(lines):
        y = scan / (lines - 1)
        lat = lat_top - lat_span * y
        for pix in range(samples):
            x = pix / (samples - 1)
            lon = lon_left + lon_span * (x + 0.10 * y)
            rows.append((lon, lat, scan, pix))
    return rows


def test_crop_pair_grid_path(tmp_path):
    """RD07 gate 1: zip WITH a geometry CSV stamps georef=geometry-grid and
    the window comes from the grid, not the corners."""
    za = _product_zip(tmp_path, name="ch2_tmc_f_20260901T0000000000_1_a.zip",
                      geometry_csv=_geometry_csv_text(_grid_rows_rotated()))
    zb = _product_zip(tmp_path, name="ch2_tmc_f_20260901T0000000000_1_b.zip", seed=SEED + 1,
                      geometry_csv=_geometry_csv_text(_grid_rows_rotated()))
    manifest = crop_pair(za, zb, (12.0, 14.0, 20.0, 23.0),
                         out_dir=tmp_path / "crops")
    assert manifest["georef"] == "geometry-grid"
    w = manifest["window"]
    assert w["georef"] == "geometry-grid"
    strip_a = w["strip"]["a"]
    strip_b = w["strip"]["b"]
    for line0, line1, sample0, sample1 in (strip_a, strip_b):
        assert 0 <= line0 < line1 <= LINES
        assert 0 <= sample0 < sample1 <= SAMPLES
    arrA = np.load(manifest["a"], mmap_mode="r")
    arrB = np.load(manifest["b"], mmap_mode="r")
    assert arrA.shape == (strip_a[1] - strip_a[0], strip_a[3] - strip_a[2])
    assert arrB.shape == (strip_b[1] - strip_b[0], strip_b[3] - strip_b[2])
    del arrA
    del arrB


def test_crop_pair_corner_fallback(tmp_path):
    """RD07 gate 2: zip WITHOUT a geometry CSV stamps georef=corner-interp-v0."""
    za = _product_zip(tmp_path, name="ch2_tmc_f_20260901T0000000000_2_a.zip")
    zb = _product_zip(tmp_path, name="ch2_tmc_f_20260901T0000000000_2_b.zip", seed=SEED + 1)
    manifest = crop_pair(za, zb, (12.0, 14.0, 20.0, 23.0),
                         out_dir=tmp_path / "crops")
    assert manifest["georef"] == "corner-interp-v0"
    w = manifest["window"]
    assert w["georef"] == "corner-interp-v0"
    arrA = np.load(manifest["a"], mmap_mode="r")
    arrB = np.load(manifest["b"], mmap_mode="r")
    assert arrA.shape == arrB.shape
    del arrA
    del arrB


def test_grid_vs_corner_differ(tmp_path):
    """RD07 gate 3: on a non-axis-aligned footprint the two georef methods
    produce DIFFERENT windows (the fix matters)."""
    geometry_csv = _geometry_csv_text(_grid_rows_rotated())
    za = _product_zip(tmp_path, name="ch2_tmc_f_20260901T0000000000_3_a.zip", geometry_csv=geometry_csv)
    zb = _product_zip(tmp_path, name="ch2_tmc_f_20260901T0000000000_3_b.zip", seed=SEED + 1,
                      geometry_csv=geometry_csv)
    m_grid = crop_pair(za, zb, (12.0, 14.0, 20.0, 23.0),
                       out_dir=tmp_path / "grid")
    assert m_grid["georef"] == "geometry-grid"
    # the corner path: same strip, NO geometry CSV next to it
    za_corner = _product_zip(tmp_path, name="ch2_tmc_f_20260901T0000000000_4_a.zip")
    zb_corner = _product_zip(tmp_path, name="ch2_tmc_f_20260901T0000000000_4_b.zip", seed=SEED + 1)
    m_corner = crop_pair(za_corner, zb_corner, (12.0, 14.0, 20.0, 23.0),
                         out_dir=tmp_path / "corner")
    assert m_corner["georef"] == "corner-interp-v0"
    g = m_grid["window"]["strip"]["a"]
    c = m_corner["window"]["strip"]["a"]
    assert list(g) != list(c)
    # the twist means the grid window is strictly different, not a rounding
    # artifact: at least one bound moves by many pixels
    assert max(abs(g[i] - c[i]) for i in range(4)) >= 4

