import os
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from lunar_tie.ch2_ingest import ingest_product, read_strip, list_contents

LABEL_XML = b"""<?xml version="1.0" encoding="UTF-8"?>
<Product_Observational xmlns="http://pds.nasa.gov/pds4/pds/v1">
  <File_Area_Observational>
    <File>
      <file_name>ch2_tmc_f_20260901T0000000000_1_c.img</file_name>
    </File>
    <Array_2D_Image>
      <data_type>UnsignedLSB2</data_type>
      <Axis_Array>
        <axis_name>Line</axis_name>
        <elements>6</elements>
      </Axis_Array>
      <Axis_Array>
        <axis_name>Sample</axis_name>
        <elements>4</elements>
      </Axis_Array>
    </Array_2D_Image>
  </File_Area_Observational>
  <File_Area_Misc>
    <File>
      <file_name>ch2_tmc_spm_20260901T0000000000_1_c.000</file_name>
    </File>
  </File_Area_Misc>
</Product_Observational>"""


@pytest.fixture
def product_zip(tmp_path):
    img = (np.arange(24, dtype="<u2").reshape(6, 4) + 100).tobytes()
    zpath = str(tmp_path / "demo_product.zip")
    with zipfile.ZipFile(zpath, "w") as zf:
        zf.writestr("ch2_tmc_f_20260901T0000000000_1_c.img", img)
        zf.writestr("ch2_tmc_f_20260901T0000000000_1_c.xml", LABEL_XML)
        zf.writestr("ch2_tmc_f_20260901T0000000000_1_c.png", b"\x89PNG fake")
        zf.writestr("ch2_tmc_spm_20260901T0000000000_1_c.000", b"1 2 3\n")
    return zpath


def test_list_contents_kinds(product_zip):
    kinds = dict(list_contents(product_zip))
    assert kinds["ch2_tmc_f_20260901T0000000000_1_c.img"] == "img"
    assert kinds["ch2_tmc_f_20260901T0000000000_1_c.xml"] == "xml"
    assert kinds["ch2_tmc_f_20260901T0000000000_1_c.png"] == "browse"
    assert kinds["ch2_tmc_spm_20260901T0000000000_1_c.000"] == "misc"


def test_ingest_product_structure(product_zip):
    info = ingest_product(product_zip)
    assert info["dims"] == {"lines": 6, "samples": 4}
    assert info["dtype"] == "<u2"
    assert info["offsets"]["img"].endswith(".img")
    assert info["offsets"]["xml"].endswith(".xml")
    assert info["offsets"]["browse"].endswith(".png")
    assert any("spm" in m for m in info["offsets"]["misc"])


def test_ingest_product_missing_img_raises(tmp_path):
    zpath = str(tmp_path / "bad.zip")
    with zipfile.ZipFile(zpath, "w") as zf:
        zf.writestr("only.xml", LABEL_XML)
    with pytest.raises(ValueError):
        ingest_product(zpath)


def test_read_strip_memmap(product_zip, tmp_path):
    info = ingest_product(product_zip)
    # extract the img member to a plain file for memmap access
    import zipfile as zf_mod
    img_member = info["offsets"]["img"]
    out_path = str(tmp_path / "strip.img")
    with zf_mod.ZipFile(product_zip) as zf:
        with zf.open(img_member) as src, open(out_path, "wb") as dst:
            dst.write(src.read())
    mm = read_strip(out_path, info["label"])
    assert isinstance(mm, np.memmap)
    assert mm.shape == (6, 4)
    assert mm.dtype == np.uint16
    assert int(mm[0, 0]) == 100
    assert int(mm[5, 3]) == 123
    del mm  # close the memmap