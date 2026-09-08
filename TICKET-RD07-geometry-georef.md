TICKET-RD07 (GEOMETRY-GRID GEOREF PREFERENCE): the corner-box collapse fix (audit L3)
AUTHOR: hermes | IMPL: opencode glm-5.3-flash | VERIFY: hermes

## THE BUG
crop_pair uses realdata.region_window (corner linear interp), which:
1. collapses the 4-corner quadrilateral to an axis-aligned box (min/max
   of the corners: a rotated/limb strip georeferences against a box
   that doesn't match the actual footprint)
2. uses linear lat/lon -> grid interpolation (wrong near the limb)
3. is used EVEN WHEN the per-product geometry CSV is inside the zip
   (confirmed: every product zip ships its own geometry/calibrated/
   YYYYMMDD/<product_id>_g_grd_d18.csv)

## FIX: edit src/lunar_tie/realdata.py crop_pair
for each side (a, b):
1. try geometry.find_geometry_csv(zip_path, product_timestamp) using the
   strip timestamp from the label's logical_identifier or the zip name.
   (the timestamp is in the product ID: parse from the zip basename.)
2. if the CSV exists: geometry.load_geometry_grid(csv_path, bbox=window)
   then geometry.window_from_geometry(grid, lat0, lat1, lon0, lon1)
   -> returns (scan, pix) at the window center. THEN scan a +/- margin
   (1000 lines x full samples as a default) around that center to get
   the (line0, line1, sample0, sample1) window. stamp manifest["georef"]
   = "geometry-grid".
3. if NO CSV: fall back to region_window (corner interp). stamp
   manifest["georef"] = "corner-interp-v0".
4. NEVER silently use the corner path when a grid exists (the grid path
   is strictly better and the CSVs are always there).

## TESTS (tests/test_realdata.py additions, failing first)
- test_crop_pair_grid_path: a synthetic product zip WITH a geometry CSV:
  crop_pair stamps georef="geometry-grid" and the window comes from
  the grid (not the corners)
- test_crop_pair_corner_fallback: a synthetic zip WITHOUT a CSV:
  crop_pair stamps georef="corner-interp-v0"
- test_grid_vs_corner_differ: on a synthetic strip with a non-axis-
  aligned footprint, the two georef methods produce DIFFERENT windows
  (proving the fix matters)

## DEFINITION OF DONE
pytest full suite green (187 baseline + 3 new). no em dashes.
commit: "TICKET-RD07: geometry-grid georef preference (audit L3)"