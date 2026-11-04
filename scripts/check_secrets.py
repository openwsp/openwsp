"""Reject credentials and raw live-transcript payloads without printing values."""
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path, PurePosixPath

PREFIXES = ("sk-", "hf_", "ak-", "as-", "ghp_", "github_pat_")
PATTERN = re.compile(r"(?:" + "|".join(map(re.escape, PREFIXES)) + r")[A-Za-z0-9_-]{16,}")
_OMITTED_LITERAL = re.compile(
    r'''(["'])\[Raw (?:content|generations) omitted; '''
    r"(?:\d+ UTF-8 bytes; sha256=[a-f0-9]{64}; (?:truncated value; )?)?"
    r"see run volume\.\]\1"
)
_OUTPUT_PREFIXES = ("OUTPUT of Python run ", "FORWARD PASS ", "TOOL RESULT:")


def _live_transcript(name):
    parts = PurePosixPath(name).parts
    return (len(parts) in (5, 6) and parts[0] == "replications" and parts[2] == "results"
            and parts[-1] == "transcript.md"
            and (len(parts) == 5 and parts[3] == "live-run"
                 or len(parts) == 6 and parts[3] == "live-runs"))


def _raw_output(text):
    # Publication may run in a fresh clone with an older driver's PYTHONPATH.
    # Always use the detector shipped alongside this check, without installing it.
    source = str(Path(__file__).resolve().parents[1] / "src")
    if sys.path[0] != source:
        sys.path.insert(0, source)
    from openwsp.reporting import _compact_output_text

    def inspect(value):
        if isinstance(value, dict):
            return any(inspect(item) for item in value.values())
        if isinstance(value, list):
            return any(inspect(item) for item in value)
        if not isinstance(value, str):
            return False
        # Existing omission receipts are already safe; the renderer deliberately
        # does not make its compaction of every raw-field string idempotent.
        candidate = _OMITTED_LITERAL.sub("null", value)
        if _compact_output_text(candidate) != candidate:
            return True
        try:
            decoded = json.loads(value)
        except (ValueError, RecursionError):
            return False
        # Old tool results can hide printed dictionaries inside JSON stdout.
        return decoded != value and inspect(decoded)

    return inspect(text)


def _raw_transcript(text):
    """Check generated output spans, preserving editor code and task examples."""
    section, parent, fence, language, body = "", "", "", "", []

    def unsafe_block():
        value = "".join(body)
        return (language == "text"
                and (section == "tool" or section == "input" and value.startswith(_OUTPUT_PREFIXES))
                and _raw_output(value))

    for line in text.splitlines(keepends=True):
        if fence:
            if re.fullmatch(r"`{" + str(len(fence)) + r",}\s*", line):
                if unsafe_block():
                    return True
                fence, body = "", []
            else:
                body.append(line)
            continue
        opening = re.fullmatch(r"(`{3,})([^`\r\n]*)\s*", line)
        if opening:
            fence, language = opening.group(1), opening.group(2).strip()
        elif line.startswith("## "):
            parent, section = line.strip(), ""
        elif line.startswith("### "):
            heading = line.strip()
            section = ("input" if heading == "### Input" else "tool"
                       if heading == "### Tool result" or heading.startswith("### Tool record ")
                       or parent == "## Forward-pass summaries" else "")
    return bool(fence and unsafe_block())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    scope = parser.add_mutually_exclusive_group()
    scope.add_argument("--staged", action="store_true", help="Check added or changed staged files (default)")
    scope.add_argument("--all", action="store_true", help="Check every tracked file in the index")
    args = parser.parse_args()
    command = ["git", "ls-files", "--cached", "-z"] if args.all else [
        "git", "diff", "--cached", "--name-only", "--diff-filter=ACMR", "-z"
    ]
    paths = subprocess.check_output(command).split(b"\0")
    bad = []
    raw = []
    for path in filter(None, paths):
        name = path.decode("utf-8", errors="surrogateescape")
        data = subprocess.check_output(["git", "show", ":" + name])
        text = data.decode("utf-8", errors="ignore")
        if PATTERN.search(text):
            bad.append(name)
        if _live_transcript(name) and _raw_transcript(text):
            raw.append(name)
    if bad:
        print("Credential-shaped value detected in: " + ", ".join(bad), file=sys.stderr)
    if raw:
        print("Raw sample payload detected in: " + ", ".join(raw), file=sys.stderr)
    return int(bool(bad or raw))


if __name__ == "__main__":
    raise SystemExit(main())
