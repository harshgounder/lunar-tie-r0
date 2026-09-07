"""ISDA CH-2 product zip ingest. numpy + stdlib only.

Product grammar (user guide Table 9):
  ch2_<inst>_<mtc>_<YYYYMMDDTHHMMSSssss>_<p>_<prd>_<stn>.fff
A calibrated image product zip carries:
  *.img  generic binary UInt16 image strip
  *.xml  PDS4 label
  *.png  browse image
  oat / lbr / spm text misc files
Nothing here materializes the image; read_strip returns a np.memmap.
"""

import os
import zipfile

import numpy as np

from . import pds4label

# extensions we recognize inside a product zip
_IMG_EXT = ".img"
_XML_EXT = ".xml"
_PNG_EXT = ".png"
_MISC_EXTS = (".oat", ".oath", ".lbr", ".spm", ".000", ".xml_", ".png_")


def _classify(name_lower):
    ext = os.path.splitext(name_lower)[1]
    base = name_lower.split(".")[0]
    tokens = set(base.replace("-", "_").split("_"))
    if ext == _IMG_EXT:
        return "img"
    if ext == _XML_EXT:
        return "xml"
    if ext == _PNG_EXT:
        return "browse"
    if ext in _MISC_EXTS or bool(tokens & set(pds4label.MISC_KINDS)):
        return "misc"
    return None


def list_contents(zip_path):
    """List (path, kind) for every member of the product zip."""
    out = []
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            kind = _classify(info.filename.lower())
            if kind is not None:
                out.append((info.filename, kind))
    return out


def ingest_product(zip_path):
    """Open a CH-2 product zip, read its PDS4 label, return a dict:
      {
        'zip_path': ...,
        'contents': [(name, kind), ...],
        'dims': {'lines': int|None, 'samples': int|None},
        'dtype': numpy dtype string or raw PDS4 name,
        'offsets': {'img': name, 'xml': name, 'browse': name,
                    'misc': [name, ...]},
        'paths': {'img': extracted temp path, 'xml': extracted temp path,
                  'browse': ..., 'misc': [...]},
        'label': full parse_label dict (misc/corners inside),
      }
    Raises ValueError when the zip lacks the img/xml pair.
    """
    contents = list_contents(zip_path)
    offsets = {"img": None, "xml": None, "browse": None, "misc": []}
    for name, kind in contents:
        if kind == "img" and offsets["img"] is None:
            offsets["img"] = name
        elif kind == "xml" and offsets["xml"] is None:
            offsets["xml"] = name
        elif kind == "browse" and offsets["browse"] is None:
            offsets["browse"] = name
        elif kind == "misc":
            offsets["misc"].append(name)

    if offsets["img"] is None or offsets["xml"] is None:
        raise ValueError(
            "product zip missing required *.img or *.xml: %s" % zip_path)

    with zipfile.ZipFile(zip_path) as zf:
        with zf.open(offsets["xml"]) as fh:
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".xml", delete=False) as tmp:
                tmp.write(fh.read())
                xml_tmp_path = tmp.name
    label = pds4label.parse_label(xml_tmp_path)
    os.unlink(xml_tmp_path)

    dtype = label.get("dtype")
    dims = {"lines": label.get("lines"), "samples": label.get("samples")}
    return {
        "zip_path": zip_path,
        "contents": contents,
        "dims": dims,
        "dtype": dtype,
        "offsets": offsets,
        "paths": {},
        "label": label,
    }


def read_strip(img_path, label):
    """Open the binary image strip as a np.memmap (never materialized).

    Shape comes from the label dims; dtype from the label dtype.
    Caller is responsible for closing (mmap._close / del).
    """
    dtype_str = label.get("dtype") or "u2"
    lines = label.get("lines")
    samples = label.get("samples")
    if not lines or not samples:
        raise ValueError("label lacks Line/Sample dims; cannot shape memmap")
    dtype = np.dtype(dtype_str)
    return np.memmap(img_path, dtype=dtype, mode="r",
                     shape=(lines, samples))