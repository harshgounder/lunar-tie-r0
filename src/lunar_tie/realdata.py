"""TICKET-RD02: crop-first real-data ingest. numpy + stdlib only.

WHY CROP-FIRST (ordering, per ticket): pipeline.py:164 casts the FULL image
to float64. A synthetic MG1 strip (12k x 12k uint16) survives that cast; a
REAL OHRC cal strip (101075 x 12000 uint16 = 1.21 GB) would balloon to
~9.7 GB float64 and OOM the 7 GB box. So the real-data path CROPS FIRST:
each side of a pair is cut to the shared lat/lon window (< 100 MB both
sides), saved as float32 .npy (the pipeline's expected input, loaded with
mmap_mode='r'), and only the crop runs the 10-step chain. Full-strip
conversion (float32 .npy of the whole 1.21 GB strip = 4.8 GB file, still
OOMing inside run_pair's float64 cast) is M2 scope behind a tiled
pipeline refactor; NOT here.

memmap discipline: every function below materializes at most ONE slice
(memmap indexing is lazy: arr[l0:l1, s0:s1] touches only the slice pages).

All errors are loud (ValueError); no silent fallbacks. No em dashes.
"""

import os
import shutil
import zipfile
from pathlib import Path

import numpy as np

from . import ch2_ingest
from . import geometry

# streaming chunk for zf.open + copyfileobj (4 MB per the ticket)
_EXTRACT_CHUNK = 4 * 1024 * 1024

# TICKET-RD07 geometry-grid preference: +/- line margin around the grid
# center (ticket default 1000 lines); samples span the full strip width,
# so the default window is 2001 lines x full samples around the anchor.
_GRID_LINE_MARGIN = 1000


def extract_product(zip_path, dest_dir, member_kind="img"):
    """Stream ONE member (by kind) out of a product zip into dest_dir.

    Uses zf.open + shutil.copyfileobj in 4 MB chunks: the member is never
    loaded whole into RAM (the 1.21 GB OHRC .img must stream). Discovery
    and label parsing are reused from ch2_ingest.list_contents /
    ingest_product. Returns the extracted path.
    """
    zip_path = Path(zip_path)
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    contents = ch2_ingest.list_contents(str(zip_path))
    matches = [name for name, kind in contents if kind == member_kind]
    if not matches:
        raise ValueError(
            "no member of kind %r in product zip: %s" % (member_kind, zip_path))
    member = matches[0]
    out_path = dest_dir / os.path.basename(member)
    with zipfile.ZipFile(str(zip_path)) as zf:
        with zf.open(member) as src, open(out_path, "wb") as dst:
            shutil.copyfileobj(src, dst, _EXTRACT_CHUNK)
    return out_path


def _open_memmap(memmap_or_path, label):
    """Accept a memmap OR a str/Path; return the memmap for slicing."""
    if isinstance(memmap_or_path, np.memmap):
        return memmap_or_path
    return ch2_ingest.read_strip(str(memmap_or_path), label)


def _check_window(arr, line0, line1, sample0, sample1):
    """Loud bound checks; no silent clipping of a wrong window."""
    lines, samples = arr.shape
    if not (0 <= line0 < line1 <= lines):
        raise ValueError(
            "line window [%s, %s) invalid for strip lines=%s"
            % (line0, line1, lines))
    if not (0 <= sample0 < sample1 <= samples):
        raise ValueError(
            "sample window [%s, %s) invalid for strip samples=%s"
            % (sample0, sample1, samples))


def crop_strip(memmap_or_path, label, line0, line1, sample0, sample1,
               out_path):
    """Slice a strip region and save it as float32 .npy; return a manifest.

    memmap indexing IS lazy: arr[l0:l1, s0:s1] materializes ONLY the
    slice. np.save writes float32, the pipeline's expected input
    (np.load mmap float32). Returns a run_pair-ready manifest dict:
      {'a': str(out_path), 'tier_label': 0, 'provenance': 'ISDA-REAL'}
    """
    mm = _open_memmap(memmap_or_path, label)
    try:
        _check_window(mm, line0, line1, sample0, sample1)
        slice2d = np.asarray(mm[line0:line1, sample0:sample1],
                             dtype=np.float32)
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(out_path), slice2d)
    finally:
        if isinstance(mm, np.memmap) and mm is not memmap_or_path:
            del mm
    return {
        "a": str(out_path),
        "tier_label": 0,
        "provenance": "ISDA-REAL",
    }


def _corner_values(label_or_corners):
    """Extract the 8 corner scalars from a label dict or corners dict."""
    corners = label_or_corners.get("corners", label_or_corners)
    out = {}
    for key, value in corners.items():
        try:
            out[key] = float(value)
        except (TypeError, ValueError):
            continue
    return out


def region_window(label_or_corners, lat0=None, lat1=None, lon0=None, lon1=None):
    """Map a lat/lon bbox onto strip (line, sample) via corner linear interp.

    v0 georeferencing, documented as such: the label's 4 corner
    coordinates (UL, UR, LL, LR as parsed by pds4label into
    label['corners']) are assumed to map LINEARLY onto the strip grid:
    lines run along-track (top edge = UL/UR row = high-lat edge),
    samples run cross-track (left edge = UL/LL column = low-lon edge).
    A bbox of (lat0, lat1, lon0, lon1) maps to
    (line0, line1, sample0, sample1); values clamp to the strip bounds.
    This is v0 linear georeferencing; geometry.window_from_geometry
    (real per-pixel lookup) supersedes it when a geometry grid exists.

    Passing no bbox (all four coordinates None) returns the full strip
    window. A bbox entirely outside the corner footprint raises
    ValueError (loud, no silent empty crop).
    """
    vals = _corner_values(label_or_corners)

    if lat0 is None and lat1 is None and lon0 is None and lon1 is None:
        lines, samples = _strip_shape_from_corners_source(label_or_corners)
        return (0, lines, 0, samples)

    # corner footprint: high-lat top edge, low-lon left edge
    lats = [vals[k] for k in
            ("corner_1_latitude", "corner_2_latitude",
             "corner_3_latitude", "corner_4_latitude") if k in vals]
    lons = [vals[k] for k in
            ("corner_1_longitude", "corner_2_longitude",
             "corner_3_longitude", "corner_4_longitude") if k in vals]
    if len(lats) != 4 or len(lons) != 4:
        raise ValueError(
            "label lacks 4 corner latitudes/longitudes; cannot georeference")

    lat_top, lat_bot = max(lats), min(lats)
    lon_left, lon_right = min(lons), max(lons)
    lines, samples = _strip_shape_from_corners_source(label_or_corners)

    # loud: a window that does not touch the footprint at all is an error
    if lat1 <= lat_bot or lat0 >= lat_top or lon1 <= lon_left or lon0 >= lon_right:
        raise ValueError(
            "bbox %r lies entirely outside strip footprint lat=[%s, %s] "
            "lon=[%s, %s]" % ((lat0, lat1, lon0, lon1),
                              lat_bot, lat_top, lon_left, lon_right))

    def frac_lat(lat):
        if lat_top == lat_bot:
            return 0.0
        return (lat_top - lat) / (lat_top - lat_bot)

    def frac_lon(lon):
        if lon_right == lon_left:
            return 0.0
        return (lon - lon_left) / (lon_right - lon_left)

    # linear interp along-track (lines from top) / cross-track (samples
    # from left), then clamp to strip bounds
    fl0 = frac_lat(lat1)
    fl1 = frac_lat(lat0)
    fs0 = frac_lon(lon0)
    fs1 = frac_lon(lon1)
    line0 = int(max(0, min(lines, round(fl0 * (lines - 1)))))
    line1 = int(max(0, min(lines, round(fl1 * (lines - 1)))))
    sample0 = int(max(0, min(samples, round(fs0 * (samples - 1)))))
    sample1 = int(max(0, min(samples, round(fs1 * (samples - 1)))))
    if line1 < line0 or sample1 < sample0:
        raise ValueError(
            "bbox %r maps to an inverted window after clamping "
            "(line %s..%s sample %s..%s)"
            % ((lat0, lat1, lon0, lon1),
               line0, line1, sample0, sample1))
    # a point (or clamped-flat) bbox still yields a 1-pixel half-open window
    line1 = max(line1, min(line0 + 1, lines))
    sample1 = max(sample1, min(sample0 + 1, samples))
    return (line0, line1, sample0, sample1)


def _strip_shape_from_corners_source(label_or_corners):
    """Strip (lines, samples) from a full label dict; corners-only dicts
    must carry lines/samples keys explicitly."""
    lines = label_or_corners.get("lines")
    samples = label_or_corners.get("samples")
    if not lines or not samples:
        raise ValueError(
            "no strip dims (lines/samples) available for region_window")
    return int(lines), int(samples)


def _label_corners_from_zip(zip_path):
    """ingest_product(zip) -> label dict (reuses ch2_ingest discovery)."""
    return ch2_ingest.ingest_product(str(zip_path))


def _product_timestamp(zip_path, label):
    """Strip timestamp from the product ID (zip basename stem).

    Real grammar (ch2_ingest docstring, user guide Table 9):
      ch2_<inst>_<mtc>_<YYYYMMDDTHHMMSSssss>_<p>_<prd>_<stn>.<ext>
    The timestamp is token 4 of the '_' split. Falls back to the label's
    logical_identifier when the zip name is missing or malformed; raises
    loudly when neither carries a parseable timestamp token.
    """
    candidates = []
    zip_name = os.path.basename(str(zip_path))
    stem = zip_name.rsplit(".", 1)[0]
    candidates.append(stem)
    lid = label.get("logical_identifier")
    if lid:
        candidates.append(str(lid))
    for candidate in candidates:
        tokens = candidate.replace(".", "_").split("_")
        for token in tokens:
            if len(token) >= 15 and token[:8].isdigit() and token[8] == "T" \
                    and token[9:15].isdigit():
                return token
    raise ValueError(
        "no product timestamp token in zip name or logical_identifier: %s"
        % zip_path)


def _window_from_grid(zip_path, label, lat0, lat1, lon0, lon1):
    """RD07 geometry-grid path: (line0, line1, sample0, sample1) from the
    per-product geometry CSV inside the zip.

    1. find_geometry_csv(zip, timestamp) using the product-ID timestamp.
    2. load_geometry_grid(csv, bbox=window) then
       window_from_geometry(grid, ...) -> (scan, pix) at the window center.
    3. scan a +/- margin (1000 lines x full samples) around that center
       to get the window.

    Returns None when no CSV ships in the zip (caller falls back to the
    corner path); raises loudly when a CSV exists but cannot georeference
    the window (NEVER silently falls back to corners when a grid exists).
    """
    timestamp = _product_timestamp(zip_path, label)
    csv_path = geometry.find_geometry_csv(zip_path, timestamp)
    if csv_path is None:
        return None
    lines = label.get("lines")
    samples = label.get("samples")
    if not lines or not samples:
        raise ValueError(
            "label lacks Line/Sample dims; cannot build a geometry window")
    bbox = (lat0, lat1, lon0, lon1)
    grid = geometry.load_geometry_grid(str(csv_path), bbox=bbox)
    line0, sample0 = geometry.window_from_geometry(
        grid, lat0, lat1, lon0, lon1)
    line1 = min(int(lines), line0 + 1 + _GRID_LINE_MARGIN)
    line0 = max(0, line0 - _GRID_LINE_MARGIN)
    return (int(line0), int(line1), 0, int(samples))


def crop_pair(zip_a, zip_b, window, out_dir):
    """Crop BOTH sides of a pair to the SAME lat/lon window.

    window = (lat0, lat1, lon0, lon1). Each side's (line, sample) window
    comes from its own per-product geometry CSV when the zip ships one
    (RD07: the grid path is strictly better; the CSVs are always there in
    real products): the grid row nearest the window center anchors a
    +/- 1000-line x full-samples scan, stamped manifest["georef"] =
    "geometry-grid". When NO CSV exists the v0 corner linear interp
    (region_window) is the fallback, stamped manifest["georef"] =
    "corner-interp-v0"; the corner path is NEVER used silently when a
    grid exists. Slices are cut off each memmap and written as float32
    .npy. Returns a run_pair-ready manifest dict:
      {'a': path_a_npy, 'b': path_b_npy,
       'tier_label': 0, 'provenance': 'ISDA-REAL', 'georef': stamp,
       'window': {..., 'georef': stamp}}
    """
    lat0, lat1, lon0, lon1 = window
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = {}
    georef_by_side = {}
    for key, zip_path in (("a", zip_a), ("b", zip_b)):
        info = _label_corners_from_zip(zip_path)
        label = info["label"]
        window_bounds = _window_from_grid(zip_path, label,
                                          lat0, lat1, lon0, lon1)
        if window_bounds is not None:
            georef = "geometry-grid"
        else:
            georef = "corner-interp-v0"
            window_bounds = region_window(
                label, lat0, lat1, lon0, lon1)
        georef_by_side[key] = georef
        line0, line1, sample0, sample1 = window_bounds
        img_member = info["offsets"]["img"]
        if img_member is None:
            raise ValueError("product zip lacks an .img member: %s" % zip_path)
        img_path = out_dir / os.path.basename(img_member)
        with zipfile.ZipFile(str(zip_path)) as zf:
            with zf.open(img_member) as src, open(img_path, "wb") as dst:
                shutil.copyfileobj(src, dst, _EXTRACT_CHUNK)
        mm = ch2_ingest.read_strip(str(img_path), label)
        try:
            crop = np.asarray(mm[line0:line1, sample0:sample1],
                              dtype=np.float32)
        finally:
            del mm
        side_path = out_dir / ("crop_%s.npy" % key)
        np.save(str(side_path), crop)
        manifest[key] = str(side_path)
        manifest.setdefault("tier_label", 0)
        manifest.setdefault("provenance", "ISDA-REAL")
        manifest.setdefault("window", {
            "lat0": lat0, "lat1": lat1, "lon0": lon0, "lon1": lon1,
            "zip": {"a": str(zip_a), "b": str(zip_b)},
            "strip": {},
        })
        manifest["window"]["strip"][key] = [line0, line1, sample0, sample1]
    if not manifest.get("a") or not manifest.get("b"):
        raise ValueError("crop_pair produced an incomplete manifest")
    # one georef stamp for the pair; a sides-disagree pair is stamped
    # "mixed" (honest, never silently uniform)
    stamps = set(georef_by_side.values())
    stamp = stamps.pop() if len(stamps) == 1 else "mixed"
    manifest["georef"] = stamp
    manifest["window"]["georef"] = stamp
    return manifest