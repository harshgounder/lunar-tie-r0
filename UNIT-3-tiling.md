# UNIT-3 TICKET - memmap tiler (D1 1.5 / P1.m5)

Assignment for opencode. Implement `src/lunar_tie/tiling.py` exactly to this spec.

## context
LUNAR-TIE R0 must handle lunar images up to 12,000 x 101,075 pixels (OHRC, ~1.21
billion pixels, ~1.13 GiB uint8, ~2.3 GiB uint16). Loading these into RAM with
imread is impossible at scale. The tiler gives the pipeline random access to
any region of any large image WITHOUT loading it whole.

## requirements (exact, no scope creep)
1. class TileGrid(path, tile_h=1024, tile_w=1024) with numpy memmap-backed
   lazy reading:
   - build index of tiles: [(row0, col0, row1, col1), ...] covering the image
     rect completely; edge tiles may be smaller (clip to image boundary)
   - attributes: shape (H, W), dtype, tile_shape (tile_h, tile_w),
     grid_rows, grid_cols, n_tiles
   - method get_tile(r, c) -> np.ndarray slice for tile at grid position
     (clipped at edges)
   - method iter_tiles() -> yields (r, c, tile_array, (row0, col0)) in
     row-major order
2. class works for .npy files (np.load mmap_mode='r') AND for raw binary
   PDS-style files with an explicit (width, height, dtype, offset) header
   provided by a companion .lbl path or explicit args
3. function make_grid_index(H, W, tile_h, tile_w) -> list of tile bounds
   (pure function; unit-testable without any file)
4. memory discipline: no call may materialize more than ONE tile as a real
   ndarray at a time; iter_tiles streams; document this invariant in the
   docstring; TESTS must verify memory discipline by checking we never
   materialize more than tile_shape at once (e.g. by patching np.array or
   by peak-memory assertion using sys.getsizeof of returned tile)
5. unit gate (tests/test_tiling.py): on synthetic fixtures you author in-test
   (np/tmp raw files)
   (a) 2048x2048 uint8 raw file, tiles 1024x1024: n_tiles == 4, get_tile
       round-trips exact pixel values at corners + center of each tile
   (b) non-divisible size 2050x1050, tiles 1024x1024: edge tiles clipped,
       n_tiles == 6 (3 cols x 2 rows... verify your own math: ceil(2050/1024)=3,
       ceil(1050/1024)=2 -> 6), get_tile on the bottom-right tile returns
       shape (26, 26)? no: 2050-2048=2 rows, 1050-1024=26 cols -> shape (2, 26)
   (c) uint16 raw file: dtype honored, values exact after round trip
   (d) npy file: same behavior as raw (round-trip + lazy)
6. expose also tile_bounds(r, c) -> (row0, col0, row1, col1) public API.

## out of scope
- any resizing/rescaling (later units)
- pyramids (unit 1.6 later)
- writing tiles (read-only R0)
- SPICE georeferencing (2.7)
- multi-band stacks (single-band PAN only in R0)

## constraints
- numpy + stdlib only
- do NOT modify pds3label.py or masking.py
- < 300 lines, docstring states unit id + gate
- no em-dash, no AI-tell vocabulary
- ambiguity: if file smaller than one tile, n_tiles == 1, tile = whole image

## definition of done
- tests/test_tiling.py: >= 10 asserts, passes
- gate name: 'memmap round-trip on 3 synthetic files'
- PROGRESS-LOG.md: exactly one appended row
- no git commit (Hermes audits and commits)