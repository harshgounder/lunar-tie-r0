# lunar-tie-r0

**Classical-track lunar image registration, built the honest way.**
SIH 2026 problem statement SIH26166 (ISRO): register Chandrayaan-2 multi-sensor
lunar imagery to sub-pixel accuracy. This repo is milestone 1: the complete
classical pipeline spine, engineered gate-first, with an adversarial
verification battery fired at every unit.

[![tests](https://img.shields.io/badge/tests-196%2F196-brightgreen)]() [![modules](https://img.shields.io/badge/modules-17-blue)()] [![real ISDA bytes verified](https://img.shields.io/badge/real%20ISDA%20bytes-verified-brightgreen)]() [![phase](https://img.shields.io/badge/milestone-1%20of%206-lightgrey)]()

(09-08 stamp: badges re-derived live after the RD04-RD08 fix chain: pytest
196 passed in 36.0s, 17 modules on disk. hand-typed counts drift; this
badge is re-stamped after every landing.)

## the problem, in 3 sentences

Chandrayaan-2 images the Moon three ways: OHRC at 0.25 m/px (panchromatic,
pushbroom, up to 1.2 gigapixels), TMC-2 stereo at 5 m/px, IIRS hyperspectral at
80 m/px across 256 bands. The mission asks for tie points that link these
products despite 20x-320x scale gaps, changing sun angles, and pushbroom
geometry whose epipolar curves are hyperbolas, not lines. Nobody has published
an end-to-end solution; the campaign research corpus (fully audited, count
derived in the parent repo) confirmed the whitespace survives every
2023-2026 literature sweep, and this repo starts the build from the ground
floor.

## why "gate-first" (the method, not just the code)

Every module landed through the same loop:

```
ticket (exact spec + out-of-scope list)
  -> implementation (opencode, numpy + stdlib only)
  -> independent gate run (Hermes executes the tests itself)
  -> negative-gate battery (adversarial inputs the ticket never mentioned)
  -> lie-hunt (hunt the gap between what was claimed and what is true)
  -> fix round (only if a catch demands it)
  -> PROGRESS-LOG row (append-only, with the lie-hunt result recorded)
  -> commit + push
```

That battery caught 6 real issues in one day, including a NaN crash hiding
behind a green self-report, a silent junk-matrix path (LAPACK DLASCL warning
behind a passing suite), and 2 overclaims in my own tickets. Nothing was
rubber-stamped. See LIE-CHECK-PROTOCOL.md in the parent campaign repo for the
full battery spec.

## the modules (all green, 196/196 tests; 09-07 hardening: 24 fixes total =
14 audit findings F1-F14 + 9 commander items C1-C9 (C2 never existed, C6
split a/b) + 1 repair round; corrected 09-07 night, the old line
presented the 24 as one audit count; 09-08: real-data era modules RD01-RD08
added, table re-derived from disk)

| # | module | what it does | gate | commit |
|---|--------|--------------|------|--------|
| 1 | `pds3label.py` | lossless PDS3 label parser (nesting, multi-line quoted values, CRLF, type coercion) | round-trip on 3 fixture labels + 5 adversarial: PASS | `ea1965e` |
| 2 | `masking.py` | valid/nodata/shadow triple mask (Otsu-based, saturating) | 3 synthetic fixtures + 7 adversarial: PASS | `ed69fdd` |
| 3 | `tiling.py` | `TileGrid`: memory-mapped 1.2-GP-frame tiler (raw + .npy + offset), single-tile lazy discipline | 3 synthetic grids + 8 adversarial: PASS | `1f03527` |
| - | MINI-GATE-1 | 12000x12000 uint16 end-to-end through 1+2+3 | PASS (streamed 144 tiles, <32 KB peak) | `d18ffb8` |
| 4 | `photometric.py` | photometric ratio normalization, gain/offset invariant (own separable Gaussian, no scipy) | mad=0.0000 under 10x+20000 stress + 7 adversarial: PASS | `1059ccc` |
| 5 | `detect.py` | SIFT + RD-SIFT (pure numpy DoG scale space, polarity-robust descriptors, Lowe ratio matching) | polarity-flip + gain-invariance on 3 fixtures + 8 adversarial: PASS | `261d5e6` |
| 6 | `consensus.py` | MAGSAC-style similarity consensus + DLT homography, NaN-loud after fix | recovery on 3 synthetic warps + 8 adversarial: PASS | `98f3721` |
| 7 | `subpixel.py` | phase-correlation subpixel (Hann window, upsampled DFT peak, parabola refine) + ECC-ready footprint validity | known-shift recovery + 8 adversarial: PASS | `09f1fb6` |
| 8 | `coverage.py` | coverage gates (occupancy/entropy/quadrants/NN-CV) + 2-round ANMS uniform selection | clustered-vs-uniform metrics on 3 fixtures + 8 adversarial: PASS | `8b4a1d0` |
| 9 | `conformal.py` | conformal ACCEPT / REJECT / ABSTAIN gates with per-stratum q-hat + tier-stamped metrics panel | split-conformal coverage 0.956 fresh + 8 adversarial: PASS | `4a035a2` |
| 4B | `photometric.py` (add) | RIFT-style rank transform, exact gain invariance (mismatch 0.000000) | 5 tests: PASS (bundled w/ MG2) | `5016273` |
| - | MINI-GATE-2 | full M1 chain on similarity pair | s=1.2001 theta=0.2000 exact, 64.9% inliers, rms 0.146px, occupancy 48.4%: PASS | `5016273` |
| 10 | `pipeline.py` | CLI driver: 10-step chain, artifacts (panel/ties/summary), N9 tier-0+UNLABELED default, loud errors | 6/6 gate + cleanroom tie delta (4.031, 3.031) vs truth (4, 3): PASS | `5a5f6e1` |
| 11 | `pds4label.py` | PDS4 XML label parser (dims, dtype, misc refs) for real ISDA products | synthetic-label round-trip: PASS | `10f8e7a` |
| 12 | `ch2_ingest.py` | product zip ingest: label + binary + browse + misc extraction, memmap strip reader (never materializes) | synthetic-zip round-trip: PASS | `10f8e7a` |
| 13 | `sun_angles.py` | spm (sun parameter) file parser, generic positional columns | synthetic-spm fixture: PASS | `10f8e7a` |
| 14 | `geometry.py` | TMC-2 geometry CSV loader: streaming parse, bbox filter, nearest-row window lookup (the REAL georef) | synthetic-CSV round-trip + window: PASS | `f55d09e` |
| 15 | `realdata.py` | crop-first real-data ingest: streaming zip extraction, float32 crops, crop_pair, geometry-grid georef preference w/ corner-interp fallback stamp | synthetic-zip e2e + RAM-bound: PASS | `6d10d5c` |
| 16 | `resample.py` | common-GSD resample: area-average, gauss-pyramid, lanczos3, match_shapes_to_min (the cross-res shape gate) | 6 shape/anti-alias tests: PASS | `35eba8a` |
| 17 | `spice_angles.py` | real SPICE angle extraction: CK attitude via ckgp SCLK, SPK position, sub-solar point, CK-gap provenance | live re-run 09-08: 53.1 deg sun-lon delta, 27.8 deg incidence delta | `2754246` |

Run it yourself:

```
git clone https://github.com/harshgounder/lunar-tie-r0
cd lunar-tie-r0
python3 -m venv .venv && source .venv/bin/activate
uv pip install pytest numpy
.venv/bin/pytest tests/ -q
```

Expected: `196 passed`.

## architecture: the pipeline spine

```
[PDS3 label] -> [triple mask] -> [memmap tiler] -> [photometric ratio-norm]
                                                            |
                                                            v
        [tier-stamped panel] <- [conformal ABSTAIN] <- [coverage+ANMS]
                                                            ^
                                    [MAGSAC consensus] <----+
                                            ^               |
                                            |        [SIFT+RD-SIFT match]
                                            |
                                    [subpixel phase refiner]
```

Each box maps 1:1 onto a unit in the parent decomposition
(sih-2026/research/DECOMPOSITION-FINE.md, 114 units). The panels emitted at the
end are already shaped as monster-zoo ledger rows (see zoo/schema.json): every
metric run is a candidate entry in the condition-indexed router that
milestone-3 wires up.

## the honesty layer

- **real ISDA bytes verified**: this pipeline must run end-to-end with NO
  Chandrayaan-2 products on disk. The synthetic + Kaggle-data path is
  primary, not fallback. (Real Ch-2 ISSDC registration is a campaign-level
  unlock, tracked in the parent repo's USER-ACTION-LIST.)
- **truth tiers on every number**: every metric panel stamps which tier its
  numbers came from (synthetic stratum / public-benchmark stratum / real
  flight-data stratum). No unlabeled claims, anywhere.
- **append-only ledgers**: PROGRESS-LOG rows are append-only; catches are
  recorded in the row itself; nothing is edited retroactively.
- **known-open items are stated**: every gate that is not yet green is
  listed as open with its root cause. (09-08 status: MINI-GATE-2 is GREEN
  (s=1.2001 theta=0.2000, rms 0.146 px, suite 196/196); the open items are
  L4 (IIRS Array_3D ingest, fires when IIRS bytes land) and L5 (masked
  normalize on real nodata, tickets 2A/2B). We do not ship a green bar
  on a red gate, and we do not ship a red bar where a green one exists.

## context: the campaign this belongs to

- full research campaign: [sih-2026](https://github.com/harshgounder/sih-2026) (private) - 320-row canon (09-08 recount: 193 DEEP + 65 ADECENT + 49 MINED + 13 SURFACE; was 319/192 before the 09-08 wave landed), decomposition into 114 single-topic units, 10-wave relation campaign, factor atlas
- build plan of record: `docs/BRIEF-MILESTONE-1-R0-SPINE.md` in that repo
- monster-hunt + zoo-router architecture: `research/MONSTER-*.md`
- the "11 units in one day" was made possible by: opencode (coding agent)
  writing 100% of the source, a Hermes-orchestrated gate loop doing 100% of
  the verification, and an aggressive lie-hunt protocol catching defects the
  green test-suites alone would never surface.

## license

All-rights-reserved (unpublished hackathon work). Cite the campaign repo for
methodology; do not redistribute product code before finale (2026-09-20).