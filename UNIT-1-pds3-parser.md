# UNIT-1 TICKET - PDS3 label parser (D1 1.1 / P1.m1)

Assignment for opencode. Implement `src/lunar_tie/pds3label.py` exactly to this spec.

## context (why this exists)
LUNAR-TIE R0 needs to read PDS3-labeled lunar imagery (CNSF MDPI supplement files,
Kaguya TC from USGS STAC, MoonAnything manifests). PDS3 is the OLD (pre-2018) label
syntax: ASCII key = value pairs, plus multi-line OBJECT/FIELD groups. PDS4 is XML
and is a LATER unit (1.2). Do NOT conflate them.

## requirements (exact, no scope creep)
1. parse PDS3 text labels from a file path or open file object into a python dict
2. handle: KEY = VALUE (string, int, float, quoted strings with embedded spaces,
   datetimes), OBJECT = name ... END_OBJECT = name nesting, GROUP/END_GROUP,
   multi-line text values (continuation lines starting with whitespace + quote),
   and stray CR/LF differences from windows-made PDS3
3. return dict with structure: {key: value} plus '__objects__': nested dicts for
   OBJECT blocks keyed by their OBJECT name (keep EVERY field, even ones unused
   later; lossless parse)
4. expose: parse_label(path_or_file) -> dict
5. unit gate (test_pds3label.py): round-trip + fidelity test on 3 fixture labels
   (a) my own hand-author minimal PDS3 (b) copy-paste of a real Kaguya TC label
   snippet (c) a pathological label with the multi-line value + CRLF mix.
   Gate passes iff: no crash, all top-level keys present, OBJECT nested dict
   structure accessible, and re-serializing the dict to key=value text round-trips
   values identically for all scalar fields.

## out of scope (do NOT add)
- PDS4 XML parsing (unit 1.2)
- image data interpretation (unit 1.4)
- OS filesystem navigation, network fetch, path traversal assumptions

## constraints
- stdlib only, no external deps
- file must be < 300 lines, with docstring on top stating unit id + gate name
- no em-dash, no AI-tell vocabulary anywhere
- if a fixture edge case is ambiguous, prefer LOSSLESS fidelity: store raw value
  under '__raw__' subdict rather than dropping it
- stop and note in commit message if any spec above is unclear  (do not invent)

## definition of done
- tests/test_pds3label.py passes: 9+ asserts, 3 fixtures, all lossless
- gate name: 'PDS3 round-trip on 3 sample labels'
- PROGRESS-LOG.md gets EXACTLY one appended row (do not touch prior rows)