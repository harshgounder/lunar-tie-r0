import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from lunar_tie.pds4label import parse_label, label_misc_paths

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "pds4_minimal.xml")


@pytest.fixture(scope="module")
def label():
    return parse_label(FIXTURE)


def test_dimensions_from_axis_arrays(label):
    assert label["lines"] == 7
    assert label["samples"] == 5


def test_data_type_mapping(label):
    assert label["data_type"] == "UnsignedLSB2"
    assert label["dtype"] == "u2"
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