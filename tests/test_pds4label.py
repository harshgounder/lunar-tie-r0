import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from lunar_tie.pds4label import (parse_label, label_misc_paths,
                                 _parse_data_type)
from lunar_tie.ch2_ingest import read_strip

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "pds4_minimal.xml")


@pytest.fixture(scope="module")
def label():
    return parse_label(FIXTURE)


def test_dimensions_from_axis_arrays(label):
    assert label["lines"] == 7
    assert label["samples"] == 5


def test_data_type_mapping(label):
    assert label["data_type"] == "UnsignedLSB2"
    assert label["dtype"] == "<u2"
    np.dtype(label["dtype"])  # must be a real numpy dtype string


def test_misc_paths_collected(label):
    oat = label_misc_paths(label, "oat")
    spm = label_misc_paths(label, "spm")
    assert any("oat" in p for p in oat)
    assert any("spm" in p for p in spm)
    # 'spm' must not bleed into the 'oat' bucket or vice versa
    assert all("spm" not in p for p in oat)
    assert all("oat" not in p for p in spm)


def test_logical_identifier(label):
    assert label["logical_identifier"] == "urn:isro:ch2:tmc2:data:demo"


def test_no_corners_present_is_empty_dict(label):
    assert label["corners"] == {}


# TICKET-RD03: real ISDA labels nest data_type under Element_Array
# (Array_2D_Image > Element_Array > data_type), the parser must find it.

_REAL_OHRC_LABEL = """<?xml version="1.0" encoding="UTF-8"?>
<Product_Observational xmlns="http://pds.nasa.gov/pds4/pds/v1">
  <File_Area_Observational>
    <Array_2D_Image>
      <offset unit="byte">0</offset>
      <axes>2</axes>
      <axis_index_order>Last Index Fastest</axis_index_order>
      <Element_Array>
        <data_type>UnsignedByte</data_type>
      </Element_Array>
      <Axis_Array>
        <axis_name>Line</axis_name>
        <elements>79796</elements>
      </Axis_Array>
      <Axis_Array>
        <axis_name>Sample</axis_name>
        <elements>12000</elements>
      </Axis_Array>
    </Array_2D_Image>
  </File_Area_Observational>
</Product_Observational>
"""

_REAL_TMC_LABEL = """<?xml version="1.0" encoding="UTF-8"?>
<Product_Observational xmlns="http://pds.nasa.gov/pds4/pds/v1">
  <File_Area_Observational>
    <Array_2D_Image>
      <Element_Array>
        <data_type>UnsignedLSB2</data_type>
      </Element_Array>
      <Axis_Array>
        <axis_name>Line</axis_name>
        <elements>100</elements>
      </Axis_Array>
      <Axis_Array>
        <axis_name>Sample</axis_name>
        <elements>50</elements>
      </Axis_Array>
    </Array_2D_Image>
  </File_Area_Observational>
</Product_Observational>
"""


def _parse_xml_str(xml_text):
    with tempfile.NamedTemporaryFile(
            mode="w", suffix=".xml", delete=False, encoding="utf-8") as tmp:
        tmp.write(xml_text)
        path = tmp.name
    try:
        return parse_label(path)
    finally:
        os.unlink(path)


def test_real_label_element_array_nesting():
    out = _parse_xml_str(_REAL_OHRC_LABEL)
    assert out["data_type"] == "UnsignedByte"
    assert out["dtype"] == "u1"
    assert out["lines"] == 79796
    assert out["samples"] == 12000


def test_direct_child_still_works():
    out = _parse_xml_str(
        '<?xml version="1.0"?>\n<Product_Observational'
        ' xmlns="http://pds.nasa.gov/pds4/pds/v1">\n'
        "  <Array_2D_Image><data_type>UnsignedByte</data_type>"
        "</Array_2D_Image>\n</Product_Observational>\n")
    assert out["data_type"] == "UnsignedByte"
    assert out["dtype"] == "u1"


def test_unsignedshort_maps_u2():
    out = _parse_xml_str(_REAL_TMC_LABEL)
    assert out["data_type"] == "UnsignedLSB2"
    assert out["dtype"] == "<u2"

def test_real_isda_nesting_fixture():
    """TICKET-RD03 follow-up: the shared real-nesting fixture parses to u1
    (real ISDA labels nest data_type under Element_Array)."""
    fx = os.path.join(os.path.dirname(__file__), "fixtures", "pds4_real_nesting.xml")
    lbl = parse_label(fx)
    assert lbl["lines"] == 120
    assert lbl["samples"] == 80
    assert lbl["dtype"] == "u1"


# TICKET-RD04 (MSB byte-order fix, audit L1): endianness must be explicit.
# MSB* types map to big-endian numpy dtypes, LSB* to little-endian. Native
# ('=') dtypes silently misread big-endian products as byte-swapped garbage.

def test_msb2_maps_big_endian():
    assert _parse_data_type("UnsignedMSB2") == ">u2"
    assert _parse_data_type("SignedMSB2") == ">i2"


def test_msb4_maps_big_endian():
    assert _parse_data_type("SignedMSB4") == ">i4"
    assert _parse_data_type("UnsignedMSB4") == ">u4"


def test_lsb_maps_little_endian():
    assert _parse_data_type("UnsignedLSB2") == "<u2"
    assert _parse_data_type("SignedLSB2") == "<i2"
    assert _parse_data_type("SignedLSB4") == "<i4"
    assert _parse_data_type("UnsignedLSB8") == "<u8"


def test_byte_order_actually_matters(tmp_path):
    """A big-endian u2 buffer parsed via a label saying UnsignedMSB2 must
    round-trip [1, 258, 3, 4], not the byte-swapped [256, 770, ...]."""
    values = [1, 258, 3, 4]
    img_path = tmp_path / "msb.img"
    img_path.write_bytes(np.array(values, dtype=">u2").tobytes())
    label = {"lines": 1, "samples": 4, "dtype": _parse_data_type("UnsignedMSB2")}
    mm = read_strip(str(img_path), label)
    got = [int(v) for v in mm.ravel()]
    del mm
    assert got == values


def test_parse_data_type_passthrough_unknown():
    """Unknown type names keep passing through raw (documented behavior)."""
    assert _parse_data_type("SomeFutureType9") == "SomeFutureType9"
    assert _parse_data_type(None) is None
