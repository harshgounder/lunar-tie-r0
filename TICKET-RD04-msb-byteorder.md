TICKET-RD04 (MSB BYTE-ORDER FIX): the most dangerous silent-wrong-answer in the stack

## THE BUG (audit L1, pds4label.py)
PDS4_DATA_TYPES maps UnsignedMSB2 -> "u2" (native little-endian). On this
LE box, a big-endian ISDA product reads byte-swapped garbage through every
downstream stage with NO error. _PDS4_ORDER is defined but never used.
Our pair-A products happened to be LSB: we dodged it by luck. Any MSB
product would produce silently wrong registration.

## FIX (minimal, pds4label.py only)
Use _PDS4_ORDER to prefix the numpy dtype: MSB* -> ">" + base, LSB* ->
"<" + base. e.g. UnsignedMSB2 -> ">u2", SignedMSB4 -> ">i4". Keep the
_UnsignedByte/SignedByte (1-byte: endianness irrelevant) as u1/i1.

## NEW TESTS (failing first, TDD)
1. test_msb2_maps_big_endian: data_type UnsignedMSB2 -> dtype ">u2"
2. test_msb4_maps_big_endian: SignedMSB4 -> ">i4"
3. test_lsb_maps_little_endian: UnsignedLSB2 -> "<u2" (or keep native u2:
   both work on LE, but be EXPLICIT: "<u2" is the spec-correct form)
4. test_byte_order_actually_matters: build a 4-pixel buffer [1, 258, 3, 4]
   as big-endian u2 bytes; parse a label saying UnsignedMSB2; read_strip
   must return [1, 258, 3, 4] NOT [256, 770, ...] (the byte-swapped read)

## DEFINITION OF DONE
- pytest full suite green (182 baseline + 4 new)
- no em dashes. numpy+stdlib only.
- commit message: "TICKET-RD04: MSB byte-order fix (audit L1)"
