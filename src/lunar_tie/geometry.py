"""TICKET-RD02: TMC-2 geometry_calibrated CSV loader. numpy + stdlib only.

The geometry products are per-year bundles (the 898 MB zip holds per-strip
lon/lat/scan/pix grids as CSV). Three functions:

1. find_geometry_csv(zip_path_or_dir, product_id_or_timestamp) -> Path|None
   locate the CSV matching a strip timestamp prefix, from a zip bundle OR
   an extracted directory. The matched member is streamed to a temp file
   (never whole-member RAM) and its path returned.
2. load_geometry_grid(csv_path, bbox=None) -> dict
   csv-module streaming parse of the lon/lat/scan/pix columns; a bbox
   (lat0, lat1, lon0, lon1) keeps only the rows covering it. Never
   full-file RAM: row-by-row csv.reader over an open file handle.
   Returns {'lon': arr, 'lat': arr, 'scan': arr, 'pix': arr, 'n': int}.
3. window_from_geometry(grid, lat0, lat1, lon0, lon1) -> (line, sample)
   nearest-row lookup: the (scan, pix) of the single grid row nearest the
   query lat/lon. This is the REAL georeferencing, finer than the corner
   linear interp in realdata.region_window (documented v0 there).

All errors loud (ValueError); no silent fallbacks. No em dashes.
"""

import csv
import io
import os
import tempfile
import zipfile
from pathlib import Path

import numpy as np


def _csv_members(source):
    """Yield (name, extract_fn_or_None) for CSV members of a zip or dir."""
    p = Path(source)
    if p.is_dir():
        for name in sorted(os.listdir(p)):
            if name.lower().endswith(".csv"):
                full = p / name
                yield name, (lambda full=full: str(full))
    elif zipfile.is_zipfile(str(p)):
        with zipfile.ZipFile(str(p)) as zf:
            for info in zf.infolist():
                if info.is_dir():
                    continue
                if info.filename.lower().endswith(".csv"):
                    yield info.filename, info.filename
    else:
        raise ValueError(
            "geometry source is neither an extracted dir nor a zip: %s" % p)


def _extract_member(zip_path, member):
    """Stream one zip member to a NamedTemporaryFile; return its path."""
    with zipfile.ZipFile(str(zip_path)) as zf:
        with zf.open(member) as src:
            with tempfile.NamedTemporaryFile(
                    suffix=".csv", delete=False) as tmp:
                chunk = 4 * 1024 * 1024
                while True:
                    data = src.read(chunk)
                    if not data:
                        break
                    tmp.write(data)
                return tmp.name


def find_geometry_csv(zip_path_or_dir, product_id_or_timestamp):
    """Find the geometry CSV whose name matches a strip timestamp prefix.

    Accepts either a per-year bundle zip or an extracted directory.
    Matching is a plain prefix match on the CSV file name (the real
    grammar embeds the strip timestamp in the CSV name). Returns the
    extracted-file path, or None when nothing matches (the caller decides
    what absence means: this is a lookup, not a failure).
    """
    prefix = str(product_id_or_timestamp)
    if not prefix:
        raise ValueError("empty product_id_or_timestamp for find_geometry_csv")
    zip_path = Path(zip_path_or_dir)
    matches = []
    if zip_path.is_dir():
        for name, resolve in _csv_members(zip_path):
            base = os.path.basename(name)
            if prefix in base:
                matches.append(resolve())
    elif zipfile.is_zipfile(str(zip_path)):
        with zipfile.ZipFile(str(zip_path)) as zf:
            for name in zf.namelist():
                if name.lower().endswith(".csv") and prefix in os.path.basename(name):
                    matches.append(_extract_member(zip_path, name))
    else:
        raise ValueError(
            "geometry source is neither an extracted dir nor a zip: %s"
            % zip_path)
    if not matches:
        return None
    return Path(sorted(matches)[0])


def _bbox_keep(lat, lon, bbox):
    lat0, lat1, lon0, lon1 = bbox
    return lat0 <= lat <= lat1 and lon0 <= lon <= lon1


def load_geometry_grid(csv_path, bbox=None):
    """Stream-parse a geometry_calibrated CSV into lon/lat/scan/pix arrays.

    csv module row-by-row over an open handle: never full-file RAM (the
    real per-strip grids are large text tables). Columns are matched by
    header name (lon/lat/scan/pix; case-insensitive); rows outside a
    given bbox=(lat0, lat1, lon0, lon1) are dropped during the stream.
    Returns {'lon': arr, 'lat': arr, 'scan': arr, 'pix': arr, 'n': int}.

    RAM discipline: rows land in preallocated numpy buffers (grow-by-
    doubling via np.resize-style chunks), never in per-row Python lists
    (a list of 720k float boxes costs ~200 MB; the buffer path stays at
    the raw array size, ~23 MB).
    """
    if bbox is not None and len(bbox) != 4:
        raise ValueError("bbox must be (lat0, lat1, lon0, lon1), got %r" % (bbox,))
    # preallocated growing buffers: chunked, bounded, numpy-native
    cap = 4096
    lons = np.empty(cap, dtype=np.float64)
    lats = np.empty(cap, dtype=np.float64)
    scans = np.empty(cap, dtype=np.int64)
    pixs = np.empty(cap, dtype=np.int64)
    count = 0

    def _ensure(new_count):
        nonlocal cap, lons, lats, scans, pixs
        if new_count <= cap:
            return
        new_cap = cap
        while new_cap < new_count:
            new_cap *= 2
        lons = np.resize(lons, new_cap)
        lats = np.resize(lats, new_cap)
        scans = np.resize(scans, new_cap)
        pixs = np.resize(pixs, new_cap)
        cap = new_cap

    with open(str(csv_path), newline="") as f:
        reader = csv.reader(f)
        header = None
        idx = None
        for row in reader:
            if not row or all(not cell.strip() for cell in row):
                continue
            if header is None:
                names = [cell.strip().lower() for cell in row]
                wanted = ("lon", "lat", "scan", "pix")
                if all(w in names for w in wanted):
                    idx = [names.index(w) for w in wanted]
                    header = names
                    continue
                else:
                    # headerless table: positional lon/lat/scan/pix is NOT
                    # guessable, loud error instead of silent semantics
                    raise ValueError(
                        "geometry CSV header lacks lon/lat/scan/pix columns: "
                        "%r" % (row,))
            try:
                lon = float(row[idx[0]])
                lat = float(row[idx[1]])
                scan = int(row[idx[2]])
                pix = int(row[idx[3]])
            except (ValueError, IndexError):
                # a malformed data row is loud, never silently skipped
                raise ValueError(
                    "malformed geometry CSV row: %r" % (row,))
            if bbox is not None and not _bbox_keep(lat, lon, bbox):
                continue
            _ensure(count + 1)
            lons[count] = lon
            lats[count] = lat
            scans[count] = scan
            pixs[count] = pix
            count += 1
    if header is None:
        raise ValueError("geometry CSV is empty: %s" % csv_path)
    return {
        "lon": lons[:count].copy(),
        "lat": lats[:count].copy(),
        "scan": scans[:count].copy(),
        "pix": pixs[:count].copy(),
        "n": count,
    }


def window_from_geometry(grid, lat0, lat1, lon0, lon1):
    """Nearest-lookup: map a lat/lon bbox to the (scan, pix) grid row
    nearest its center; return (line, sample).

    The grid's scan/pix ARE the image (line, sample) indices of the
    geometry product, so the nearest row's (scan, pix) is the real
    georeferenced window anchor. A query outside the grid's lat/lon
    extent raises ValueError (loud, no silent nearest-edge snap).
    """
    lon = grid["lon"]
    lat = grid["lat"]
    if lon.size == 0:
        raise ValueError("empty geometry grid; cannot map a window")
    if (lat.min() > max(lat0, lat1) or lat.max() < min(lat0, lat1)
            or lon.min() > max(lon0, lon1) or lon.max() < min(lon0, lon1)):
        raise ValueError(
            "query lat/lon %r outside geometry grid extent lat=[%s, %s] "
            "lon=[%s, %s]"
            % ((lat0, lat1, lon0, lon1),
               lat.min(), lat.max(), lon.min(), lon.max()))
    plat = (lat0 + lat1) / 2.0
    plon = (lon0 + lon1) / 2.0
    d2 = (lat - plat) ** 2 + (lon - plon) ** 2
    row = int(np.argmin(d2))
    return (int(grid["scan"][row]), int(grid["pix"][row]))