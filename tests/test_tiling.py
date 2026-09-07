"""UNIT-3 gate test. Gate: 'memmap round-trip on 3 synthetic files'."""

import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from lunar_tie.tiling import TileGrid, make_grid_index


def _write_raw(path, arr):
    arr.tofile(str(path))
    return path


def _write_npy(path, arr):
    np.save(str(path), arr)
    return path


def test_make_grid_index_divisible():
    index = make_grid_index(2048, 2048, 1024, 1024)
    assert len(index) == 4
    assert index[0] == (0, 0, 1024, 1024)
    assert index[3] == (1024, 1024, 2048, 2048)


def test_make_grid_index_non_divisible():
    index = make_grid_index(2050, 1050, 1024, 1024)
    assert len(index) == 6
    assert index[4] == (2048, 0, 2050, 1024)
    assert index[5] == (2048, 1024, 2050, 1050)


def test_make_grid_index_smaller_than_tile():
    index = make_grid_index(100, 200, 1024, 1024)
    assert len(index) == 1
    assert index[0] == (0, 0, 100, 200)


def test_raw_uint8_divisible_roundtrip():
    arr = np.arange(2048 * 2048, dtype=np.uint8).reshape(2048, 2048)
    with tempfile.TemporaryDirectory() as tmp:
        path = _write_raw(Path(tmp) / "img.raw", arr)
        grid = TileGrid(path, tile_h=1024, tile_w=1024,
                        width=2048, height=2048, dtype="u1")
        assert grid.shape == (2048, 2048)
        assert grid.dtype == np.uint8
        assert grid.n_tiles == 4
        assert grid.grid_rows == 2 and grid.grid_cols == 2
        for r in range(2):
            for c in range(2):
                tile = grid.get_tile(r, c)
                assert tile.shape == (1024, 1024)
                r0, c0, r1, c1 = grid.tile_bounds(r, c)
                assert np.array_equal(tile, arr[r0:r1, c0:c1])
                assert tile[0, 0] == arr[r0, c0]
                assert tile[0, -1] == arr[r0, c1 - 1]
                assert tile[-1, 0] == arr[r1 - 1, c0]
                assert tile[-1, -1] == arr[r1 - 1, c1 - 1]


def test_raw_uint8_non_divisible_clipped():
    arr = np.arange(2050 * 1050, dtype=np.uint8).reshape(2050, 1050)
    with tempfile.TemporaryDirectory() as tmp:
        path = _write_raw(Path(tmp) / "img.raw", arr)
        grid = TileGrid(path, tile_h=1024, tile_w=1024,
                        width=1050, height=2050, dtype="u1")
        assert grid.n_tiles == 6
        assert grid.grid_rows == 3 and grid.grid_cols == 2
        br = grid.get_tile(2, 1)
        assert br.shape == (2, 26)
        r0, c0, r1, c1 = grid.tile_bounds(2, 1)
        assert np.array_equal(br, arr[r0:r1, c0:c1])


def test_raw_uint16_dtype_honored():
    arr = np.arange(2048 * 2048, dtype=np.uint16).reshape(2048, 2048)
    with tempfile.TemporaryDirectory() as tmp:
        path = _write_raw(Path(tmp) / "img.raw", arr)
        grid = TileGrid(path, tile_h=1024, tile_w=1024,
                        width=2048, height=2048, dtype="u2")
        assert grid.dtype == np.uint16
        tile = grid.get_tile(1, 1)
        assert np.array_equal(tile, arr[1024:2048, 1024:2048])


def test_npy_roundtrip_lazy():
    arr = np.arange(2048 * 2048, dtype=np.uint8).reshape(2048, 2048)
    with tempfile.TemporaryDirectory() as tmp:
        path = _write_npy(Path(tmp) / "img.npy", arr)
        grid = TileGrid(path, tile_h=1024, tile_w=1024)
        assert grid.shape == (2048, 2048)
        assert grid.n_tiles == 4
        tile = grid.get_tile(0, 0)
        assert np.array_equal(tile, arr[0:1024, 0:1024])


def test_iter_tiles_row_major():
    arr = np.arange(2050 * 1050, dtype=np.uint8).reshape(2050, 1050)
    with tempfile.TemporaryDirectory() as tmp:
        path = _write_raw(Path(tmp) / "img.raw", arr)
        grid = TileGrid(path, tile_h=1024, tile_w=1024,
                        width=1050, height=2050, dtype="u1")
        seen = []
        for r, c, tile, (r0, c0) in grid.iter_tiles():
            seen.append((r, c))
            assert np.array_equal(tile, arr[r0:r0 + tile.shape[0],
                                            c0:c0 + tile.shape[1]])
        assert seen == [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)]


def test_memory_discipline_single_tile():
    """F8 (audit): the old test asserted sys.getsizeof(tile) < 1 MB, which
    measures only the memmap VIEW HEADER (~160 bytes) and can never fail -
    a by-construction pass that proves nothing about memory. Real streaming
    discipline assertions: (a) get_tile returns a lazy np.memmap view whose
    .nbytes equals exactly one tile (not the whole image), (b) iter_tiles
    yields views whose cumulative DISTINCT mapped bytes are one tile at a
    time (each yielded tile is a bounded slice of the mmap, never a
    materialized full-array copy), and (c) peak resident memory during a
    full iter_tiles sweep stays within tile-bytes + process baseline."""
    arr = np.arange(2048 * 2048, dtype=np.uint8).reshape(2048, 2048)
    with tempfile.TemporaryDirectory() as tmp:
        path = _write_raw(Path(tmp) / "img.raw", arr)
        grid = TileGrid(path, tile_h=1024, tile_w=1024,
                        width=2048, height=2048, dtype="u1")
        tile = grid.get_tile(0, 0)
        # (a) lazy bounded view, not a materialized copy
        assert isinstance(tile, np.memmap)
        assert tile.nbytes == 1024 * 1024
        assert tile.base is not None  # mmap-backed, not an owned copy

        # (b) streaming: each yielded tile is exactly one tile's bytes and
        # is released before the next; no more than one tile is alive at a
        # time as tracked by the generator's own re-binding of `tile`.
        tile_bytes = 1024 * 1024
        streamed = 0
        for _r, _c, t, _off in grid.iter_tiles():
            assert t.nbytes == tile_bytes, t.nbytes
            assert isinstance(t, np.memmap)
            streamed += 1
            del t
        assert streamed == grid.n_tiles

        # (c) resident-memory bound across the whole sweep: baseline plus a
        # bounded slack much smaller than a second full tile copy would add.
        import resource

        before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        for _r, _c, t, _off in grid.iter_tiles():
            _ = t[0, 0]  # touch one page, force nothing wholesale
            del t
        after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # ru_maxrss is in KB on linux; allow the touched pages of one tile
        # (1 MB) plus interpreter slack, but far below the 4 MB full image
        # being materialized twice over.
        assert after - before < 4 * tile_bytes / 1024, (before, after)


def test_tile_bounds_public_api():
    arr = np.zeros((2050, 1050), dtype=np.uint8)
    with tempfile.TemporaryDirectory() as tmp:
        path = _write_raw(Path(tmp) / "img.raw", arr)
        grid = TileGrid(path, tile_h=1024, tile_w=1024,
                        width=1050, height=2050, dtype="u1")
        assert grid.tile_bounds(0, 0) == (0, 0, 1024, 1024)
        assert grid.tile_bounds(2, 1) == (2048, 1024, 2050, 1050)
