TICKET-RD02 (REAL-DATA-FIRST-LIGHT): crop-first ingest + geometry module
AUTHOR: hermes (glm-5.3-flash planning, per division rule)
IMPL: opencode glm-5.3-flash
VERIFY: hermes independent (5-gate)
PRECONDITION: ZERO real bytes needed. builds + tests entirely on synthetic
fixtures that MIMIC the real product grammar. fires tonight, rung-2 ready
the moment user's zips land.

## context (why)

pipeline.py:164 casts the FULL image to float64 (normalize_ratio). the
synthetic MG1 (12k x 12k uint16, 288MB -> 1.1GB float64) passes. a REAL
OHRC cal strip is 101075 x 12000 x 1B = 1.21GB -> 9.7GB float64 = OOM on
the 7GB box (2G available). REAL DATA KILLS THE CHAIN AS-IS.

the play is CROP-FIRST: OHRC pair-A window = 24.7km x 4.2km bbox; the
TMC-2 crop of the same window is ~4944 x 842 x 2B = 8MB. run the chain on
real crops (both sides < 100MB), get the first REAL number tonight-scale.
full-strip runs come later ONLY after a tiled/float32 path exists (M2+).

## deliverable 1: realdata.py (new module, src/lunar_tie/realdata.py)

functions (numpy + stdlib only, NO new deps, NO em dashes):

1. extract_product(zip_path, dest_dir, member_kind='img') -> Path
   streams ONE member (the .img) from the product zip to dest_dir using
   zf.open + shutil.copyfileobj in 4MB chunks. never loads whole member
   into RAM (the 1.21GB OHRC img must stream). returns extracted path.
   reuse ch2_ingest.list_contents + ingest_product for discovery/labels.
2. crop_strip(memmap_or_path, label, line0, line1, sample0, sample1,
   out_path) -> manifest-dict
   slices a memmap region (memmap indexing IS lazy: arr[l0:l1, s0:s1]
   materializes ONLY the slice), np.save(out_path, slice.astype(np.float32))
   returns {'a': str(out_path), 'tier_label': 0, 'provenance': 'ISDA-REAL'}
   float32 output = the pipeline's expected input (np.load mmap float32).
3. region_window(label_or_corners, ...) -> (line0, line1, sample0, sample1)
   maps a lat/lon bbox onto strip (line, sample) using the label corner
   coordinates (linear interp along-track/cross-track; corners are in the
   pds4label parse as misc/corners). clamp to strip bounds. this is v0
   linear georeferencing, documented as such.
4. crop_pair(zip_a, zip_b, window) -> manifest dict
   crops BOTH sides of a pair to the SAME lat/lon window (via 3) and
   writes both float32 .npy + a run_pair-ready manifest.

## deliverable 2: geometry.py (new module, src/lunar_tie/geometry.py)

TMC-2 geometry_calibrated CSV loader (rung-0 material, the 898MB zip holds
per-strip lon/lat/scan/pix grids):
1. find_geometry_csv(zip_path_or_dir, product_id_or_timestamp) -> Path|None
   the geometry zips are per-year bundles: locate the CSV matching a strip
   timestamp prefix. handle both zip and extracted-dir input.
2. load_geometry_grid(csv_path, bbox=None) -> dict
   parses lon/lat/scan/pix columns; if bbox given, returns the row subset
   covering the bbox (streaming read, csv module, never full-file RAM).
   returns {'lon': arr, 'lat': arr, 'scan': arr, 'pix': arr, 'n': int}
3. window_from_geometry(grid, lat0, lat1, lon0, lon1) -> (line, sample)
   nearest-lookup: maps lat/lon to (scan, pix) indices = the REAL
   georeferencing (better than corner-linear in deliverable 1's v0).

## deliverable 3: tests (test_realdata.py + test_geometry.py)

fixture strategy (ZERO real bytes, but REAL grammar):
- synthetic product zip: build a tiny zip with fake .img (u2, 64x48),
  PDS4 .xml label with REAL PDS4 structure copied from tests/fixtures/
  pds4_minimal.xml (dims 64x48), a .png stub, an .spm stub.
- test extract_product: streams, correct bytes land, RAM-bounded (use the
  F8 pattern: ru_maxrss delta check with a big synthetic member).
- test crop_strip: memmap slice -> float32 npy round-trip, values equal.
- test region_window: corner-interp math on a synthetic label with known
  corners; clamp behavior at edges.
- test crop_pair: both sides same window, manifest has a/b paths + tier 0.
- test find_geometry_csv: synthetic year-bundle zip with 2 CSVs, finds the
  right one by timestamp prefix.
- test load_geometry_grid: streaming parse + bbox subset correctness.
- test window_from_geometry: synthetic grid with known mapping.
- every test synthetic, fast (<60s total), no em dashes, numpy+stdlib only.

## HARD CONSTRAINTS (all previous tickets' rules apply)

- suite must stay green: 154/154 baseline + your new tests on top
- numpy + stdlib only, no new dependencies
- no em dashes anywhere
- memmap discipline: no function materializes more than ONE tile/slice
- loud errors: PipelineError/ValueError, never silent fallback
- pds3label.py untouched; ch2_ingest.py untouched (reuse, don't modify)
- PROGRESS-LOG row appended; gate tests must pass on YOUR machine before
  reporting PASS
- TDD: write the failing tests FIRST, then modules

## WHY NOT full-strip float32 conversion now

converting the whole 1.21GB strip to float32 .npy = 4.8GB file + the
pipeline float64 cast inside run_pair still OOMs. crop-first is the
tonight-safe path; a tiled pipeline refactor is M2 scope (separate ticket).
document this ordering in the module docstrings.

## gate (definition of done)

1. pytest: 154 + new tests all green (run it, paste output)
2. the synthetic crop_pair -> run_pair chain works END-TO-END: build a
   synthetic pair zip, crop both, run run_pair manifest, panel.json exists
3. no new deps, no em dashes, suite green, PROGRESS-LOG row
4. hermes independent verify: fresh clone path, rerun suite + chain