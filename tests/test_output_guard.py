"""The guard inspects a bounded suffix and never returns generated content."""

import json

import pytest

from openwsp.editors.output_guard import MAX_SCAN_CHARS, detect_runaway_formatting

LINE = "`" * 7 + "\n"


def test_thresholds_are_inclusive_and_independent():
    result = detect_runaway_formatting(LINE * 128)
    assert result == {
        "kind": "runaway_backtick_suffix",
        "backtick_lines": 128,
        "suffix_chars": 1024,
        "scanned_chars": 1024,
        "scan_truncated": False,
    }
    assert detect_runaway_formatting(("`" * 20 + "\n") * 127) is None
    assert detect_runaway_formatting("`\n" * 128) is None
    assert detect_runaway_formatting("`\n" * 127 + " \n" * 1000) is None


@pytest.mark.parametrize("last_line", ["`", "``", "```", " `` \t"])
def test_final_partial_backtick_line_and_crlf_whitespace(last_line):
    result = detect_runaway_formatting(("  `` \t\r\n\r\n" * 127) + last_line)
    assert result is not None
    assert result["backtick_lines"] == 128
    assert result["suffix_chars"] >= 1024


@pytest.mark.parametrize("ending", ["Explanation.", "```python", "x = 1", "`value`", "``\u200b", "DONE"])
def test_real_content_ends_the_suspicious_suffix(ending):
    assert detect_runaway_formatting(LINE * 256 + ending + "\n \t\n") is None


@pytest.mark.parametrize(
    "text",
    [
        "",
        " \t\r\n" * 4000,
        "```python\nx = 1\nprint(x)\n```\n",
        "Text with `inline code`.\n\n```\nvalue\n```\n" * 300,
        "```python\n" + "# normal code\n" * 300 + "```\n",
        "A code example:\n" + "``\n" * 5 + "\n" * 3000,
    ],
)
def test_ordinary_markdown_code_blank_output_and_few_repeats_are_not_flagged(text):
    assert detect_runaway_formatting(text) is None


@pytest.mark.parametrize("chunk_size", [1, 2, 7, 53, 1023])
def test_incremental_chunks_agree_with_accumulated_output(chunk_size):
    output = "Private prose before formatting.\n" + " `` \n \t\n" * 180 + "`"
    accumulated, retained = "", ""
    detected = False
    for offset in range(0, len(output), chunk_size):
        chunk = output[offset : offset + chunk_size]
        accumulated += chunk
        retained = (retained + chunk)[-(MAX_SCAN_CHARS + 1) :]
        expected = detect_runaway_formatting(accumulated)
        assert detect_runaway_formatting(retained) == expected
        detected |= expected is not None
    assert detected
    assert detect_runaway_formatting(retained + "actual content") is None


def test_clipped_mixed_content_line_is_not_counted_as_backtick_only():
    text = "real code " + "`" * MAX_SCAN_CHARS + "\n" + LINE * 127
    assert detect_runaway_formatting(text) is None
    assert detect_runaway_formatting(text[-(MAX_SCAN_CHARS + 1) :]) is None


def test_complete_line_at_window_boundary_is_counted():
    suffix = ("`" * 127 + "\n") * 128
    assert len(suffix) == MAX_SCAN_CHARS
    result = detect_runaway_formatting("ordinary prose\n" + suffix)
    assert result["backtick_lines"] == 128
    assert result["suffix_chars"] == result["scanned_chars"] == MAX_SCAN_CHARS
    assert result["scan_truncated"]


def test_long_stream_retains_bounded_structural_diagnostics_without_content():
    prefix = "PRIVATE CONTENT MUST NOT BE RETURNED\n"
    retained = ""
    output = prefix + " `` \n \t\n" * 40_000 + "`"
    for offset in range(0, len(output), 997):
        retained = (retained + output[offset : offset + 997])[-(MAX_SCAN_CHARS + 1) :]
        result = detect_runaway_formatting(retained)
        if result:
            assert result["scanned_chars"] <= MAX_SCAN_CHARS
            assert result["suffix_chars"] <= MAX_SCAN_CHARS
            assert "PRIVATE" not in json.dumps(result) and "`" not in json.dumps(result)
    assert result is not None
    assert result == detect_runaway_formatting(output)


def test_non_text_is_rejected():
    with pytest.raises(TypeError, match="string"):
        detect_runaway_formatting(None)
