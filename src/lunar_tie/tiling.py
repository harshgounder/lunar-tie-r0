"""UNIT-3 memmap tiler. Gate: 'memmap round-trip on 3 synthetic files'.

Provides random access to any region of a large single-band image without
loading it whole. Backed by numpy memmap. Supports .npy files and raw binary
PDS-style files with an explicit (width, height, dtype, offset) header given
by a companion .lbl path or explicit constructor args.

Memory discipline invariant: no call materializes more than ONE tile as a
real ndarray at a time. get_tile returns a memmap view (lazy, no copy);
iter_tiles streams row-major and yields views one at a time. The caller owns
the returned view and may copy it if it must outlive the next call.

numpy + stdlib only.
"""

from pathlib import Path

import numpy as np


def make_grid_index(H, W, tile_h, tile_w):
    """Return list of (row0, col0, row1, col1) tile bounds covering the rect.

    Edge tiles are clipped to the image boundary. If the image is smaller than
    one tile, a single tile covering the whole image is returned.
    """
    if H <= 0 or W <= 0:
        raise ValueError("image must have positive dimensions")
    if tile_h <= 0 or tile_w <= 0:
        raise ValueError("tile dimensions must be positive")
    rows = max(1, int(np.ceil(H / tile_h)))
    cols = max(1, int(np.ceil(W / tile_w)))
    index = []
    for r in range(rows):
        r0 = r * tile_h
        r1 = min(H, r0 + tile_h)
        for c in range(cols):
            c0 = c * tile_w
            c1 = min(W, c0 + tile_w)
            index.append((r0, c0, r1, c1))
    return index


class TileGrid:
    """Lazy memmap-backed tiler over a single-band image file.

    path may be a .npy file (loaded with np.load mmap_mode='r') or a raw
    binary file. For raw binary files the (width, height, dtype, offset)
    header must be supplied via explicit args or a companion .lbl path.
    """

    _SAMPLE_TYPE_TO_DTYPE = {
        "MSB_UNSIGNED_INTEGER": ">u2",
        "LSB_UNSIGNED_INTEGER": "<u2",
        "UNSIGNED_INTEGER": "u2",
        "UNSIGNED_BYTE": "u1",
        "MSB_INTEGER": ">i2",
        "LSB_INTEGER": "<i2",
        "INTEGER": "i2",
        "IEEE_REAL": "f4",
    }

    def __init__(self, path, tile_h=1024, tile_w=1024, width=None, height=None,
                 dtype=None, offset=0, lbl=None):
        path = Path(path)
        self.path = path
        self.tile_shape = (int(tile_h), int(tile_w))
        self._mm = None

        if path.suffix.lower() == ".npy":
            self._mm = np.load(str(path), mmap_mode="r")
        else:
            if lbl is not None:
                width, height, dtype = self._read_lbl(lbl, width, height, dtype)
            if width is None or height is None or dtype is None:
                raise ValueError(
                    "raw binary file requires width, height, dtype "
                    "(via args or a companion .lbl)"
                )
            self._mm = np.memmap(
                str(path), dtype=dtype, mode="r", offset=int(offset),
                shape=(int(height), int(width)), order="C",
            )

        self.shape = self._mm.shape
        self.dtype = self._mm.dtype
        self.grid_rows = max(1, int(np.ceil(self.shape[0] / tile_h)))
        self.grid_cols = max(1, int(np.ceil(self.shape[1] / tile_w)))
        self._index = make_grid_index(
            self.shape[0], self.shape[1], tile_h, tile_w
        )
        self.n_tiles = len(self._index)

    @staticmethod
    def _read_lbl(lbl, width, height, dtype):
        from lunar_tie.pds3label import parse_label

        data = parse_label(lbl)
        if width is None:
            width = int(data.get("LINE_SAMPLES", data.get("WIDTH")))
        if height is None:
            height = int(data.get("LINES", data.get("HEIGHT")))
        if dtype is None:
            st = str(data.get("SAMPLE_TYPE", "")).upper()
            dtype = TileGrid._SAMPLE_TYPE_TO_DTYPE.get(st)
        return width, height, dtype

    def tile_bounds(self, r, c):
        """Return (row0, col0, row1, col1) for grid position (r, c)."""
        if not (0 <= r < self.grid_rows and 0 <= c < self.grid_cols):
            raise IndexError("tile grid position out of range")
        return self._index[r * self.grid_cols + c]

    def get_tile(self, r, c):
        """Return a memmap view of the tile at grid position (r, c)."""
        r0, c0, r1, c1 = self.tile_bounds(r, c)
        return self._mm[r0:r1, c0:c1]

    def iter_tiles(self):
        """Yield (r, c, tile_array, (row0, col0)) in row-major order."""
        for r in range(self.grid_rows):
            for c in range(self.grid_cols):
                r0, c0, r1, c1 = self.tile_bounds(r, c)
                yield r, c, self._mm[r0:r1, c0:c1], (r0, c0)
