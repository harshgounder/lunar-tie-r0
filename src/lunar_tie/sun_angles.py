"""SPM (sun parameter) misc-file parser. numpy + stdlib only.

The ISDA spm text file's exact column semantics are NOT documented in
the user guide we have, so this parser is deliberately generic:
  - whitespace-split each line
  - numeric fields kept by COLUMN POSITION as spm_c1..spm_cN
  - non-numeric tokens kept as spm_str_cK
  - no physics guessed, no angle renamed, unknowns documented here
Axiom A6 (shipped angles == SPICE-computed angles?) fires downstream
once real products land; this module only hands over the numbers.
"""


def is_numeric(token):
    try:
        float(token)
        return True
    except (TypeError, ValueError):
        return False


def parse_spm(path_or_text):
    """Parse an spm file (path or raw text) -> dict.

    Returns:
      {
        'n_rows': int,
        'n_cols': int,          # max columns seen across rows
        'columns': ['spm_c1', ...],   # positional names, numeric cols only
        'rows': [ [float, ...], ... ],  # numeric fields per row, positional
        'string_columns': ['spm_str_cK', ...],
        'string_rows': [ [str, ...], ... ],
        'raw_lines': [...],     # verbatim non-blank lines (comments included)
      }
    """
    if "\n" in path_or_text or ("=" in path_or_text and not path_or_text.endswith(".000")):
        text = path_or_text
    else:
        with open(path_or_text, "r") as fh:
            text = fh.read()

    raw_lines = []
    rows = []
    string_rows = []
    n_cols = 0
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            continue
        raw_lines.append(line)
        tokens = stripped.split()
        numeric = [float(t) for t in tokens if is_numeric(t)]
        strings = [t for t in tokens if not is_numeric(t)]
        rows.append(numeric)
        string_rows.append(strings)
        n_cols = max(n_cols, len(numeric))

    columns = ["spm_c%d" % (i + 1) for i in range(n_cols)]
    string_columns = ["spm_str_c%d" % (i + 1) for i in range(
        max((len(r) for r in string_rows), default=0))]
    return {
        "n_rows": len(rows),
        "n_cols": n_cols,
        "columns": columns,
        "rows": rows,
        "string_columns": string_columns,
        "string_rows": string_rows,
        "raw_lines": raw_lines,
    }


def spm_column(parsed, index):
    """Extract one numeric column by position (0-based) as a list."""
    return [row[index] for row in parsed["rows"] if len(row) > index]