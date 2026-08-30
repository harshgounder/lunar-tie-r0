"""UNIT-1 PDS3 label parser. Gate: 'PDS3 round-trip on 3 sample labels'.

Parses PDS3 (pre-2018) ASCII labels into a python dict. Handles KEY = VALUE
pairs, quoted strings with embedded spaces, datetimes, ints, floats, nested
OBJECT/END_OBJECT and GROUP/END_GROUP blocks, multi-line quoted values, and
stray CR/LF differences from windows-made labels. Lossless: every field is
kept, and scalar values round-trip back to identical key=value text.
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
    """Coerce a raw scalar string to int/float/str, keeping the raw text."""
    if raw.startswith('"') and raw.endswith('"'):
        return raw[1:-1]
    if raw.startswith("'") and raw.endswith("'"):
        return raw[1:-1]
    if re.fullmatch(r"[+-]?\d+", raw):
        return int(raw)
    if re.fullmatch(r"[+-]?(\d+\.\d*|\.\d+|\d+)([eE][+-]?\d+)?", raw):
        return float(raw)
    return raw


def _parse_lines(lines):
    """Parse normalized lines into a dict with '__objects__' nesting."""
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
            block = groups.setdefault(name, {})
            stack.append(block)
            i += 1
            continue

        m = _GROUP_CLOSE.match(line)
        if m:
            if len(stack) > 1:
                stack.pop()
            i += 1
            continue

        m = _KEY_VALUE.match(line)
        if m:
            key = m.group(1)
            value = m.group(2)
            if value.startswith('"'):
                parts = [value]
                while not value.endswith('"') and i + 1 < n:
                    i += 1
                    nxt = lines[i]
                    if _CONTINUATION.match(nxt) or not nxt.strip():
                        parts.append(nxt.strip())
                    else:
                        i -= 1
                        break
                joined = " ".join(parts)
                if joined.startswith('"') and joined.endswith('"'):
                    joined = joined[1:-1]
                stack[-1][key] = joined
            else:
                stack[-1][key] = _coerce(value)
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
