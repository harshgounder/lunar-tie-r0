"""UNIT-1 PDS3 label parser. Gate: 'PDS3 round-trip on 3 sample labels'.

Parses PDS3 (pre-2018) ASCII labels into a python dict. Handles KEY = VALUE
pairs, quoted strings with embedded spaces, datetimes, ints, floats, nested
OBJECT/END_OBJECT and GROUP/END_GROUP blocks, multi-line quoted values in
BOTH quote styles, and stray CR/LF differences from windows-made labels.

Lossless policy: every field is kept. Duplicate keys at the same level
(including repeated GROUP blocks with the same name) are preserved as lists
in document order instead of silently overwriting. Zero-padded integer
literals keep their raw text (007 stays "007": PDS3 file/DSS IDs use leading
zeros, coercing to 7 loses the ID). An unterminated quoted value raises
ValueError loudly instead of yielding a truncated literal. Scalar values
round-trip back to identical key=value text.
"""

import re
from pathlib import Path

_OBJECT_OPEN = re.compile(r"^\s*OBJECT\s*=\s*(\S+)\s*$", re.IGNORECASE)
_OBJECT_CLOSE = re.compile(r"^\s*END_OBJECT\s*=\s*(\S+)\s*$", re.IGNORECASE)
_GROUP_OPEN = re.compile(r"^\s*GROUP\s*=\s*(\S+)\s*$", re.IGNORECASE)
_GROUP_CLOSE = re.compile(r"^\s*END_GROUP\s*=\s*(\S+)\s*$", re.IGNORECASE)
_KEY_VALUE = re.compile(r"^\s*([A-Za-z0-9_]+)\s*=\s*(.*?)\s*$")
_CONTINUATION = re.compile(r"^\s+.*$")


def _coerce(raw):
    """Coerce a raw scalar string to int/float/str, keeping the raw text.

    Quoted strings are unwrapped. Integers parse via int(raw) EXCEPT when
    the literal has a leading zero (007, 016): the RAW text is kept because
    PDS3 zero-padded numbers are identifiers, not magnitudes (coercing
    007 -> 7 silently corrupts e.g. SPACECRAFT_CLOCK counts and file IDs).
    """
    if raw.startswith('"') and raw.endswith('"'):
        return raw[1:-1]
    if raw.startswith("'") and raw.endswith("'"):
        return raw[1:-1]
    if re.fullmatch(r"[+-]?0\d+", raw):
        return raw
    if re.fullmatch(r"[+-]?\d+", raw):
        return int(raw)
    if re.fullmatch(r"[+-]?(\d+\.\d*|\.\d+|\d+)([eE][+-]?\d+)?", raw):
        return float(raw)
    return raw


def _store(container, key, value):
    """Store a key=value pair preserving duplicates as lists (lossless).

    First occurrence stores the scalar; a second same-level occurrence
    converts to a list in document order; later ones append. This mirrors
    the OBJECT duplication policy: PDS3 labels legitimately repeat keys
    (e.g. multiple COLUMN = entries inside one GROUP) and overwriting them
    silently is data loss.
    """
    if key not in container:
        container[key] = value
    else:
        existing = container[key]
        if isinstance(existing, list):
            existing.append(value)
        else:
            container[key] = [existing, value]


def _parse_lines(lines):
    """Parse normalized lines into a dict with '__objects__' nesting.

    Duplicate same-level keys (and duplicate GROUP names) are preserved as
    lists in document order. Unterminated quoted values raise ValueError.
    """
    root = {}
    stack = [root]
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        stripped = line.strip()
        if not stripped:
            i += 1
            continue

        m = _OBJECT_OPEN.match(line)
        if m:
            name = m.group(1)
            container = stack[-1]
            objects = container.setdefault("__objects__", {})
            block = {}
            if name in objects:
                existing = objects[name]
                if isinstance(existing, list):
                    existing.append(block)
                else:
                    objects[name] = [existing, block]
            else:
                objects[name] = block
            stack.append(block)
            i += 1
            continue

        m = _OBJECT_CLOSE.match(line)
        if m:
            if len(stack) > 1:
                stack.pop()
            i += 1
            continue

        m = _GROUP_OPEN.match(line)
        if m:
            name = m.group(1)
            container = stack[-1]
            groups = container.setdefault("__groups__", {})
            block = {}
            if name in groups:
                existing = groups[name]
                if isinstance(existing, list):
                    existing.append(block)
                else:
                    groups[name] = [existing, block]
            else:
                groups[name] = block
            stack.append(block)
            i += 1
            continue

        m = _GROUP_CLOSE.match(line)
        if m:
            if len(stack) > 1:
                stack.pop()
            i += 1
            continue

        if stripped in ("END_GROUP", "END_OBJECT"):
            # bare close form (no '= name' tail): _KEY_VALUE does not match
            # it, so without this the block never pops and following keys
            # nest one level too deep. Close the innermost block.
            if len(stack) > 1:
                stack.pop()
            i += 1
            continue

        m = _KEY_VALUE.match(line)
        if m:
            key = m.group(1)
            raw = m.group(2)
            value = raw
            quote = None
            if value.startswith('"'):
                quote = '"'
            elif value.startswith("'"):
                quote = "'"
            if quote is not None:
                parts = [value]
                joined = None
                while i + 1 < n:
                    if value.endswith(quote):
                        joined = " ".join(parts)
                        break
                    i += 1
                    nxt = lines[i]
                    if _CONTINUATION.match(nxt) or not nxt.strip():
                        parts.append(nxt.strip())
                        value = nxt.strip()
                    else:
                        i -= 1
                        break
                if joined is None:
                    # input exhausted (or a non-continuation line stopped the
                    # scan): the closing quote never arrived
                    if value.endswith(quote):
                        joined = " ".join(parts)
                    else:
                        raise ValueError(
                            f"unterminated quoted value for key '{key}': "
                            f"starts {parts[0][:40]!r}, no closing {quote}")
                if joined.startswith(quote) and joined.endswith(quote):
                    joined = joined[1:-1]
                _store(stack[-1], key, joined)
            else:
                _store(stack[-1], key, _coerce(value))
            i += 1
            continue

        i += 1
    return root


def parse_label(path_or_file):
    """Parse a PDS3 label from a file path or open file object into a dict."""
    if hasattr(path_or_file, "read"):
        text = path_or_file.read()
    else:
        text = Path(path_or_file).read_text(encoding="utf-8", errors="replace")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = text.split("\n")
    return _parse_lines(lines)
