"""PDS4 XML product label parser (ISDA ch2). Stdlib only.

Reads a PDS4 label (the *.xml that ships next to a CH-2 *.img),
extracts the fields the ingest path needs:
  - image dimensions (Line/Sample axes of the Array_2D_Image)
  - data type (PDS4 data_type name, mapped to a numpy dtype string)
  - misc file paths (oat / oath / lbr / spm references, if present)
  - corner coordinates (if the label carries them)
pds3label.py is intentionally untouched; PDS4 is a separate format.
"""

import os
import xml.etree.ElementTree as ET

# PDS4 data_type -> numpy dtype string (numpy+stdlib only; no planetary db)
PDS4_DATA_TYPES = {
    "UnsignedByte": "u1",
    "SignedByte": "i1",
    "UnsignedLSB2": "u2",
    "SignedLSB2": "i2",
    "UnsignedMSB2": "u2",
    "SignedMSB2": "i2",
    "UnsignedLSB4": "u4",
    "SignedLSB4": "i4",
    "UnsignedMSB4": "u4",
    "SignedMSB4": "i4",
    "UnsignedLSB8": "u8",
    "SignedLSB8": "i8",
    "UnsignedMSB8": "u8",
    "SignedMSB8": "i8",
    "IEEE754LSBSingle": "f4",
    "IEEE754MSBSingle": "f4",
    "IEEE754LSBDouble": "f8",
    "IEEE754MSBDouble": "f8",
}

# PDS4 byte order -> numpy endianness prefix
_PDS4_ORDER = {
    "LSB": "<",
    "MSB": ">",
}

# misc product kinds we look for in File_Area_* label references
MISC_KINDS = ("oat", "oath", "lbr", "spm")


def _local(tag):
    """Strip XML namespace: '{http://pds.nasa.gov/pds4/pds/v1}Element' -> 'Element'."""
    return tag.rsplit("}", 1)[-1]


def _find_all_local(root, name):
    return [el for el in root.iter() if _local(el.tag) == name]


def _find_first_local(root, name):
    found = _find_all_local(root, name)
    return found[0] if found else None


def _child_text(element, name):
    if element is None:
        return None
    for child in element:
        if _local(child.tag) == name:
            return child.text
    return None


def _parse_axis_array(axis_array):
    """Axis_Array element -> {'axis_name': ..., 'elements': int} or None."""
    if axis_array is None:
        return None
    name = _child_text(axis_array, "axis_name")
    elements = _child_text(axis_array, "elements")
    return {
        "axis_name": name,
        "elements": int(elements) if elements is not None else None,
    }


def _parse_data_type(data_type_name):
    """Map a PDS4 data_type name to a numpy dtype string (or None)."""
    if data_type_name is None:
        return None
    if data_type_name in PDS4_DATA_TYPES:
        return PDS4_DATA_TYPES[data_type_name]
    # unknown generic: keep the name, let the caller decide
    return data_type_name


def _iter_file_areas(root):
    for el in root.iter():
        if _local(el.tag).startswith("File_Area"):
            yield el


def parse_label(xml_path):
    """Parse a PDS4 label file at xml_path -> dict of extracted fields.

    Returns a dict with keys:
      xml_path          absolute-ish path as given
      logical_identifier Product_LID if present
      lines             int, from Axis_Array axis_name == 'Line'
      samples           int, from Axis_Array axis_name == 'Sample'
      data_type         raw PDS4 data_type string (e.g. 'UnsignedLSB2')
      dtype             numpy dtype string (e.g. 'uint16') or the raw
                        name if not in the known map
      misc              {'<kind>': [abs paths]} for oat/oath/lbr/spm
                        references found in label File entries
      corners           {'corner_*_lat/lon': value} if present, else {}
    Unknown/absent fields stay absent (no guessed semantics).
    """
    root = ET.parse(xml_path).getroot()

    out = {"xml_path": xml_path, "lines": None, "samples": None,
           "data_type": None, "dtype": None, "misc": {}, "corners": {}}

    lid = _find_first_local(root, "logical_identifier")
    if lid is not None and lid.text:
        out["logical_identifier"] = lid.text.strip()

    # array characteristics: lines/samples + data type
    for ac in _find_all_local(root, "Array_2D_Image") + _find_all_local(root, "Array_2D"):
        out["data_type"] = _child_text(ac, "data_type") or out["data_type"]
        for aa in _find_all_local(ac, "Axis_Array"):
            parsed = _parse_axis_array(aa)
            if parsed["axis_name"] == "Line":
                out["lines"] = parsed["elements"]
            elif parsed["axis_name"] == "Sample":
                out["samples"] = parsed["elements"]
        if out["data_type"] is not None:
            break

    if out["data_type"] is not None:
        out["dtype"] = _parse_data_type(out["data_type"])

    # misc file references: per the Table 9 grammar the product kind is a
    # delimiter-bounded token (ch2_<inst>_<mtc>_..._<prd>_<stn>.fff), so
    # match on tokens split at '_'/'.' rather than substring/extension.
    for area in _iter_file_areas(root):
        for file_el in _find_all_local(area, "File"):
            fname = _child_text(file_el, "file_name")
            if not fname:
                continue
            base = fname.strip().lower()
            tokens = set(base.replace(".", "_").split("_"))
            for kind in MISC_KINDS:
                if kind in tokens:
                    out["misc"].setdefault(kind, []).append(fname.strip())

    # corner coordinates, if the label carries them (documented unknowns:
    # the exact tag naming is only in real labels, so scan generically)
    for tag in ("corner_", "Latitude_", "Longitude_"):
        for el in root.iter():
            local = _local(el.tag)
            if local.startswith(tag.strip("_")) and el.text and el.text.strip():
                out["corners"][local] = el.text.strip()

    return out


def label_misc_paths(label, kind):
    """Convenience: misc paths of one kind from a parse_label dict."""
    return label.get("misc", {}).get(kind, [])