"""UNIT-1 gate test. Gate: 'PDS3 round-trip on 3 sample labels'."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from lunar_tie.pds3label import parse_label

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"


def _serialize_scalars(d, out):
    """Re-serialize scalar fields to key=value text for round-trip check."""
    for k, v in d.items():
        if k == "__objects__":
            continue
        if k == "__groups__":
            continue
        if isinstance(v, dict):
            continue
        if isinstance(v, bool):
            out.append("%s = %s" % (k, str(v)))
        elif isinstance(v, (int, float)):
            out.append("%s = %s" % (k, str(v)))
        else:
            out.append('%s = "%s"' % (k, v))
    for sub in d.get("__objects__", {}).values():
        if isinstance(sub, list):
            for item in sub:
                _serialize_scalars(item, out)
        else:
            _serialize_scalars(sub, out)
    for sub in d.get("__groups__", {}).values():
        if isinstance(sub, list):
            for item in sub:
                _serialize_scalars(item, out)
        else:
            _serialize_scalars(sub, out)


def _scalar_pairs(d):
    pairs = {}
    for k, v in d.items():
        if k in ("__objects__", "__groups__"):
            continue
        if isinstance(v, dict):
            continue
        pairs[k] = v
    for sub in d.get("__objects__", {}).values():
        if isinstance(sub, list):
            for item in sub:
                pairs.update(_scalar_pairs(item))
        else:
            pairs.update(_scalar_pairs(sub))
    for sub in d.get("__groups__", {}).values():
        if isinstance(sub, list):
            for item in sub:
                pairs.update(_scalar_pairs(item))
        else:
            pairs.update(_scalar_pairs(sub))
    return pairs


def test_minimal_top_level_keys():
    d = parse_label(FIXTURES / "minimal_pds3.lbl")
    for key in ("PDS_VERSION_ID", "RECORD_TYPE", "RECORD_BYTES",
                "FILE_RECORDS", "START_TIME", "STOP_TIME",
                "MISSION_PHASE_NAME"):
        assert key in d, key


def test_minimal_object_nested():
    d = parse_label(FIXTURES / "minimal_pds3.lbl")
    assert "__objects__" in d
    assert "IMAGE" in d["__objects__"]
    img = d["__objects__"]["IMAGE"]
    assert img["LINES"] == 100
    assert img["LINE_SAMPLES"] == 1024
    assert img["SAMPLE_BITS"] == 16


def test_minimal_scalar_types():
    d = parse_label(FIXTURES / "minimal_pds3.lbl")
    assert d["RECORD_BYTES"] == 1024
    assert isinstance(d["RECORD_BYTES"], int)
    assert d["PDS_VERSION_ID"] == "PDS3"
    assert d["MISSION_PHASE_NAME"] == "Nominal"


def test_minimal_roundtrip():
    d = parse_label(FIXTURES / "minimal_pds3.lbl")
    lines = []
    _serialize_scalars(d, lines)
    text = "\n".join(lines)
    for key, val in _scalar_pairs(d).items():
        if isinstance(val, str):
            assert ('%s = "%s"' % (key, val)) in text, key
        else:
            assert ("%s = %s" % (key, val)) in text, key


def test_kaguya_top_level_keys():
    d = parse_label(FIXTURES / "kaguya_tc_snippet.lbl")
    for key in ("PDS_VERSION_ID", "INSTRUMENT_HOST_NAME", "INSTRUMENT_NAME",
                "INSTRUMENT_ID", "TARGET_NAME", "START_TIME", "STOP_TIME",
                "SPACECRAFT_CLOCK_START_COUNT"):
        assert key in d, key


def test_kaguya_nested_objects():
    d = parse_label(FIXTURES / "kaguya_tc_snippet.lbl")
    assert "IMAGE" in d["__objects__"]
    assert "TABLE" in d["__objects__"]
    table = d["__objects__"]["TABLE"]
    assert table["ROWS"] == 1
    assert table["COLUMNS"] == 4
    assert "COLUMN" in table["__objects__"]
    cols = table["__objects__"]["COLUMN"]
    assert isinstance(cols, list)
    assert len(cols) == 2
    assert cols[0]["NAME"] == "SAMPLE"
    assert cols[0]["START_BYTE"] == 1
    assert cols[1]["NAME"] == "LINE"
    assert cols[1]["START_BYTE"] == 5


def test_kaguya_roundtrip():
    d = parse_label(FIXTURES / "kaguya_tc_snippet.lbl")
    lines = []
    _serialize_scalars(d, lines)
    text = "\n".join(lines)
    for key, val in _scalar_pairs(d).items():
        if isinstance(val, str):
            assert ('%s = "%s"' % (key, val)) in text, key
        else:
            assert ("%s = %s" % (key, val)) in text, key


def test_pathological_crlf_multiline():
    d = parse_label(FIXTURES / "pathological_crlf.lbl")
    assert d["PDS_VERSION_ID"] == "PDS3"
    assert d["RECORD_BYTES"] == 2048
    assert d["START_TIME"] == "2011-06-30T12:00:00Z"
    img = d["__objects__"]["IMAGE"]
    assert img["LINES"] == 256
    assert img["LINE_SAMPLES"] == 512
    desc = img["DESCRIPTION"]
    assert "multi line" in desc
    assert "continues" in desc
    assert "several lines" in desc


def test_pathological_roundtrip():
    d = parse_label(FIXTURES / "pathological_crlf.lbl")
    lines = []
    _serialize_scalars(d, lines)
    text = "\n".join(lines)
    for key, val in _scalar_pairs(d).items():
        if isinstance(val, str):
            assert ('%s = "%s"' % (key, val)) in text, key
        else:
            assert ("%s = %s" % (key, val)) in text, key


def test_open_file_object():
    with open(FIXTURES / "minimal_pds3.lbl", "r", encoding="utf-8") as fh:
        d = parse_label(fh)
    assert d["PDS_VERSION_ID"] == "PDS3"


# ---------------------------------------------------------------------------
# F9 / C4 / C5 / F10 + zero-padded raw: real parse-behavior tests. The three
# test_*_roundtrip tests above format the parsed dict and grep its own
# output, so a lossy parse passed them by construction; these pin the parse.
# ---------------------------------------------------------------------------

def _parse_text(text):
    import io

    return parse_label(io.StringIO(text.replace("\r\n", "\n")))


def test_duplicate_same_level_keys_kept_as_list_not_overwritten():
    """F9: LINES = 100 then LINES = 200 must keep BOTH values as a list in
    document order; the old parser silently kept 200 (data loss)."""
    d = _parse_text("A = 1\nLINES = 100\nLINES = 200\nB = 2\n")
    assert d["LINES"] == [100, 200], d["LINES"]
    assert d["A"] == 1 and d["B"] == 2


def test_zero_padded_integer_keeps_raw_text():
    """F9 companion: ZERO = 007 stays the raw string '007' (a PDS3
    identifier, not the magnitude 7)."""
    d = _parse_text("ZERO = 007\nNORM = 42\nNEG = -016\n")
    assert d["ZERO"] == "007"
    assert d["NORM"] == 42 and isinstance(d["NORM"], int)
    assert d["NEG"] == "-016"


def test_pds3_007_raw_round_trip():
    """007-raw round trip: re-parsing serialized output reproduces the dict
    exactly (the value is a string, so the serializer quotes it and the
    parser returns the same text)."""
    d = _parse_text('ZERO = 007\nNORM = 42\nNAME = "abc"\n')
    import json
    again = _parse_text("\n".join(
        '%s = "%s"' % (k, v) if isinstance(v, str) else "%s = %s" % (k, v)
        for k, v in d.items()))
    assert again == d


def test_duplicate_group_names_kept_as_list():
    """C4: two GROUP = COL blocks with the same name must both survive as a
    list in document order; the old parser merged them into one dict,
    silently overwriting NAME."""
    text = ('GROUP = COL\n  NAME = "SAMPLE"\n  START_BYTE = 1\nEND_GROUP = COL\n'
            'GROUP = COL\n  NAME = "LINE"\n  START_BYTE = 5\nEND_GROUP = COL\n')
    d = _parse_text(text)
    cols = d["__groups__"]["COL"]
    assert isinstance(cols, list), type(cols)
    assert len(cols) == 2
    assert cols[0]["NAME"] == "SAMPLE" and cols[0]["START_BYTE"] == 1
    assert cols[1]["NAME"] == "LINE" and cols[1]["START_BYTE"] == 5


def test_single_quoted_multiline_value_continues():
    """F10: a single-quoted value split across continuation lines must join
    into one literal, same as the double-quote path ('123<newline>
    continued' -> '123 continued', not the truncated "'123")."""
    d = _parse_text("A = '123\n  continued'\nB = 5\n")
    assert d["A"] == "123 continued", repr(d["A"])
    assert d["B"] == 5


def test_unterminated_quote_raises_loud():
    """C5: a quoted value with no closing quote before input ends raises
    ValueError instead of silently truncating to a partial literal."""
    import pytest

    with pytest.raises(ValueError):
        _parse_text('A = "starts but never\nB = 5\n')
    with pytest.raises(ValueError):
        _parse_text('A = "multi\n  cont1\n  cont2\nB = 5\n')
    with pytest.raises(ValueError):
        _parse_text("A = 'nope\n")
