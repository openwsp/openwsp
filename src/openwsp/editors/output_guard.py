"""Bounded structural diagnostics for runaway trailing Markdown formatting."""

MAX_SCAN_CHARS = 16_384
MIN_BACKTICK_LINES = 128
MIN_SUFFIX_CHARS = 1_024
_LINE_BREAKS = frozenset("\n\r\v\f\x1c\x1d\x1e\x85\u2028\u2029")


def detect_runaway_formatting(text: str) -> dict | None:
    """Describe a long backtick/blank-line suffix, without returning its content.

    Pass accumulated output, or retain at least ``MAX_SCAN_CHARS + 1`` trailing
    characters between streaming chunks. The extra character identifies whether
    the scan starts at a complete line; a clipped first line is never counted.
    This function has no cancellation, billing, or provider behavior.

    Only the final 16,384 characters are scanned. Nonempty lines must consist
    entirely of backticks after stripping whitespace. Blank lines contribute
    suffix length but not the backtick-line count. A final unterminated backtick
    line is accepted, allowing detection during incremental generation.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    start = max(0, len(text) - MAX_SCAN_CHARS)
    tail = text[start:]
    lines = tail.splitlines(keepends=True)
    if start and text[start - 1] not in _LINE_BREAKS:
        lines = lines[1:]
    backtick_lines = suffix_chars = 0
    for line in reversed(lines):
        content = line.strip()
        if content and content.strip("`"):
            break
        suffix_chars += len(line)
        backtick_lines += bool(content)
    if backtick_lines < MIN_BACKTICK_LINES or suffix_chars < MIN_SUFFIX_CHARS:
        return None
    return {
        "kind": "runaway_backtick_suffix",
        "backtick_lines": backtick_lines,
        "suffix_chars": suffix_chars,
        "scanned_chars": len(tail),
        "scan_truncated": bool(start),
    }
