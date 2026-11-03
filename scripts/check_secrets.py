"""Reject credential-shaped values in the index without printing the value."""
import argparse
import re
import subprocess
import sys

PREFIXES = ("sk-", "hf_", "ak-", "as-", "ghp_", "github_pat_")
PATTERN = re.compile(r"(?:" + "|".join(map(re.escape, PREFIXES)) + r")[A-Za-z0-9_-]{16,}")


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
    for path in filter(None, paths):
        name = path.decode("utf-8", errors="surrogateescape")
        data = subprocess.check_output(["git", "show", ":" + name])
        if PATTERN.search(data.decode("utf-8", errors="ignore")):
            bad.append(name)
    if bad:
        print("Credential-shaped value detected in: " + ", ".join(bad), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
