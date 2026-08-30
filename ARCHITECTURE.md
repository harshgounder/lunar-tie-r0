# ARCHITECTURE.md - live document, updated per unit merge

R0 vertical slice = ONE pipeline:

```
[pair manifest JSON]
        |
        v
 +--------------------+     +--------------------+
 | 1 PDS3/PDS4 loader | --> | 2 mask (valid/     |
 |    (unit 1.1)      |     |    nodata/shadow)  |
 +--------------------+     +--------------------+
                                     |
        +----------------------------+----------------------------+
        v                            v                            v
 +-------------+             +------------------+          +--------------+
 | 3 memmap    |             | 4 mapping        |          | 6 photometric|
 | tiles+pyr   |             | contract + DEM   |          | ratio norm   |
 | (unit 1.5)  |             | warp (3.1-3.3)   |          | (4.1)        |
 +-------------+             +------------------+          +--------------+
        \                            |                            /
         \                           v                           /
          +---------------[ normalized shared grid ]------------/
                                     |
                                     v
                            +------------------+
                            | 7 detect (SIFT + | 
                            |    RD-SIFT)      |
                            +------------------+
                                     |
                                     v
                            +------------------+
                            | 8 match (NNDR +  |
                            |    ratio test)   |
                            +------------------+
                                     |
                                     v
                            +------------------+
                            | 9 consensus      |
                            |    MAGSAC++ + NFA|
                            +------------------+
                                     |
                                     v
                            +------------------+
                            | 10 subpixel      |
                            |    phase + ECC   |
                            +------------------+
                                     |
            +------------------------+------------------+
            v                        v                  v
   +----------------+      +----------------+      +----------------+
   | 11 coverage    |      | 12 conformal   |      | 13 exports:    |
   |    gates+ANMS  |      |    ABSTAIN     |      | tif+geojson+   |
   +----------------+      +----------------+      | metrics JSON   |
                                                      +----------------+
```

- units map 1:1 to DECOMPOSITION-FINE ids (see PROGRESS-LOG).
- each box = own module in src/lunar_tie/, each has its own gate test.
- the zoo schema (zoo/schema.json) is ALREADY defined so that every metric output
  from day 1 is a candidate row for the monster ledger.

## what this repo is NOT (scope discipline, per brief)
- no learned models (no D5) - milestone 2
- no crater graph (no D6) - milestone 3
- no Ch-2 products required anywhere - N9 zero-input guarantee
- no geometric refinement beyond 2-px sanity; sub-pixel claims need milestones 4-5