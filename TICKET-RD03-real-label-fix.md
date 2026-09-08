TICKET-RD03 (REAL-LABEL FIX): pds4label._child_text misses nested data_type
AUTHOR: hermes (glm-5.3-flash planning) | IMPL: opencode glm-5.3-flash | VERIFY: hermes

## THE BUG (found on REAL ISDA bytes, rung-1 verify 09-08 ~07:40)
real label: <Array_2D_Image><Element_Array><data_type>UnsignedByte</data_type></Element_Array><Axis_Array>...
parse_label returns dtype=None because _child_text(ac, "data_type") checks only
DIRECT children of Array_2D_Image; data_type lives under Element_Array
(grandchild). our synthetic fixtures had data_type as a direct child, so the
154-suite + 24 new tests all passed while real bytes fail. EXACTLY the
"fixtures too kind" class.

## impact
read_strip raises ValueError("label lacks...dtype") -> every REAL ISDA
OHRC (UnsignedByte) + TMC (UnsignedShort, check!) product unparseable.

## FIX (minimal, in src/lunar_tie/pds4label.py parse_label only)
when _child_text(ac, "data_type") is None, search one level deeper:
Element_Array/data_type. keep direct-child check first (fixture compat).
PSEUDO:
    dt = _child_text(ac, "data_type")
    if dt is None:
        ea = _find_all_local(ac, "Element_Array")
        if ea: dt = _child_text(ea[0], "data_type")
    out["data_type"] = dt or out["data_type"]

## NEW TESTS (failing first, TDD)
1. test_real_label_element_array_nesting: label with
   Array_2D_Image > Element_Array > data_type = UnsignedByte (real
   ISDA nesting, EXACT from tonight's OHRC bytes: L121-137) -> dtype == "u1"
2. test_direct_child_still_works (existing fixtures stay green)
3. test_unsignedshort_maps_u2 (TMC path: real TMC label has
   UnsignedLSB2? verify from the nca zip in data/real-scratch or fixtures)

## DEFINITION OF DONE
- pytest full suite green (178 baseline + new)
- ingest_product on the REAL OHRC zip at
  /home/liebert511/Downloads/ABDM/Compressed/ch2_ohr_ncp_20240330T0035085365_d_img_d18.zip
  (workdir repo root, read-only on that zip) returns
  dims 79796x12000, dtype u1
- no em dashes. numpy+stdlib only.

## FILES
- src/lunar_tie/pds4label.py (fix)
- tests/test_pds4label.py (3 new tests)
commit message: "TICKET-RD03: real-ISDA label fix (Element_Array nesting)"