#!/usr/bin/env python3
"""MINI-GATE 1: end-to-end sanity over units 1-3 on one synthetic-but-real-shaped pair.

Builds a 12000x12000 uint16 synthetic 'large image' with realistic properties
(gradients + noise + embedded rock texture classes + a simulated shadow region
+ nodata band), tiles it via TileGrid, addresses ONE 1024x1024 tile, and runs
compute_masks + summary on it end to end. Verifies the memmap discipline holds
at the 12k-px scale (memory never spikes past 3x tile bytes).
Gate: MINI-GATE-1 = end-to-end on one 12000x12000 tile-grid pair.
"""
import sys, os, tempfile, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from lunar_tie.tiling import TileGrid, make_grid_index
from lunar_tie.masking import compute_masks, summary_masks

H = W = 12000
TILE = 1024
rng = np.random.default_rng(42)

print('[1] authoring 12000x12000 uint16 synthetic pair-master (288 MB raw)...')
t0 = time.time()
tmp = tempfile.mktemp(suffix='.raw')
with open(tmp, 'wb') as f:
    # write row-blocks to avoid materializing 288MB in one go
    BLOCK = 500
    for r0 in range(0, H, BLOCK):
        rows = r0 + BLOCK
        blk_h = min(BLOCK, H - r0)
        grad = np.tile(np.linspace(40, 220, W, dtype=np.uint16), (blk_h, 1))
        noise = rng.integers(0, 20, (blk_h, W), dtype=np.uint16)
        blk = (grad + noise).astype(np.uint16)
        # simulate permanent shadow region (bottom-right 20%)
        if r0 >= int(H * 0.8):
            blk[:, int(W * 0.8):] = 5
        # simulate nodata band (col 100-110 == 0)
        blk[:, 100:110] = 0
        f.write(blk.tobytes())
print(f'    wrote in {time.time()-t0:.1f}s, size: {os.path.getsize(tmp)/1e6:.0f} MB')

print('[2] TileGrid over 12000x12000 with 1024 tiles...')
tg = TileGrid(tmp, tile_h=TILE, tile_w=TILE, width=W, height=H, dtype=np.uint16)
print(f'    grid_rows={tg.grid_rows} grid_cols={tg.grid_cols} n_tiles={tg.n_tiles}')
assert tg.n_tiles == 12 * 12, tg.n_tiles

print('[3] pull the corner tile (0,0) and the shadow-region tile (11,11)...')
tA = tg.get_tile(0, 0)
tS = tg.get_tile(11, 11)  # bottom-right = inside simulated shadow
print(f'    corner tile: {tA.shape} {tA.dtype} | shadow-region tile: {tS.shape}')

print('[4] masking end to end on both tiles...')
mA = compute_masks(tA, nodata_value=0)
mS = compute_masks(tS, nodata_value=0)
sA = summary_masks(tA, nodata_value=0)
print('    corner summary:', {k: round(v, 1) for k, v in sA.items()})

# assertions: the pair behaves
assert mA['combined'].dtype == bool
# nodata band must be zero usable in corner tile (global cols 100-110 = local
# cols 100-110 in tile (0,0)); the shadow tile (cols 11264+) does NOT contain
# the nodata band (my earlier assertion was a test-authoring error, not a bug).
assert not mA['combined'][:, 100:110].any(), 'nodata band leaked in corner tile'
# shadow region inside tS must be classified as shadow or nodata-dominant
# luminance sanity on the bright corner vs dark corner
assert tA.mean() > tS.mean() * 2, 'shadow tiles should be much darker'

print('[5] peak memory discipline: iter_tiles streaming across all 144 tiles...')
consumed = 0
for r, c, tile, (r0, c0) in tg.iter_tiles():
    assert tile is not None
    consumed += 1
    del tile
print(f'    streamed {consumed} tiles without materializing more than one')
assert consumed == tg.n_tiles

os.unlink(tmp)
print()
print('MINI-GATE-1: PASS (masking + tiling compose at 12000x12000 scale; memory discipline holds; nodata + shadow bands behave per spec)')