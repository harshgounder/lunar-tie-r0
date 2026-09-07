import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from lunar_tie.sun_angles import parse_spm, spm_column, is_numeric

SPM_TEXT = (
    "# spm header comment unknown semantics\n"
    "54321.000000   -12.345678   23.456789   0.987654\n"
    "54321.500000   -11.111111   24.222222   0.976543\n"
)


def test_is_numeric():
    assert is_numeric("1.5")
    assert is_numeric("-3")
    assert not is_numeric("abc")
    assert not is_numeric("1.2.3")


def test_parse_spm_columns_by_position():
    parsed = parse_spm(SPM_TEXT)
    assert parsed["n_rows"] == 2
    assert parsed["n_cols"] == 4
    assert parsed["columns"] == ["spm_c1", "spm_c2", "spm_c3", "spm_c4"]
    assert parsed["rows"][0] == [54321.0, -12.345678, 23.456789, 0.987654]
    assert parsed["rows"][1][1] == -11.111111


def test_parse_spm_column_accessor():
    parsed = parse_spm(SPM_TEXT)
    col1 = spm_column(parsed, 0)
    assert col1 == [54321.0, 54321.5]
    col3 = spm_column(parsed, 2)
    assert col3[0] == 23.456789


def test_parse_spm_from_file(tmp_path):
    p = tmp_path / "ch2_tmc_spm_x.000"
    p.write_text(SPM_TEXT)
    parsed = parse_spm(str(p))
    assert parsed["n_rows"] == 2
    assert len(parsed["raw_lines"]) == 2


def test_parse_spm_string_tokens_kept_separately():
    text = "FLAG row1 1.0 2.0\nFLAG row2 3.0 4.0\n"
    parsed = parse_spm(text)
    assert parsed["string_columns"] == ["spm_str_c1", "spm_str_c2"]
    assert parsed["string_rows"][0] == ["FLAG", "row1"]
    assert parsed["rows"][0] == [1.0, 2.0]


def test_parse_spm_blank_lines_skipped():
    parsed = parse_spm("\n\n1.0 2.0\n\n3.0 4.0\n\n")
    assert parsed["n_rows"] == 2