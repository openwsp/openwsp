"""Native metadata/token-count preflight only: this module never creates a Message."""

import argparse
import math
from datetime import UTC, datetime
from pathlib import Path

from openwsp.budget import atomic_json

from .runner import inventory, verified_source

MODEL = "claude-fable-5-1"


def prepare(source_root, output, *, client):
    sources = [(item, verified_source(source_root, item)) for item in inventory()["items"]]
    metadata = client.models.retrieve(MODEL)
    if metadata.id != MODEL or metadata.lifecycle != "active":
        raise ValueError("Native model metadata does not confirm the pinned active Fable identity")
    if type(metadata.max_input_tokens) is not int or type(metadata.max_tokens) is not int:
        raise ValueError("Native model metadata omitted required context/output limits")
    limits = {}
    for item, source in sources:
        counted = client.messages.count_tokens(
            model=MODEL, system=source["instructions"], messages=[{"role": "user", "content": source["prompt"]}],
            thinking={"type": "adaptive"}, output_config={"effort": item["authors_config"]["effort"]}, tools=[])
        if type(counted.input_tokens) is not int or counted.input_tokens <= 0:
            raise ValueError("Native token count is invalid")
        # Token counting is an estimate; actual usage still settles from the
        # provider receipt. The existing adapter halts on any ceiling overrun.
        input_ceiling = math.ceil(counted.input_tokens * 1.05) + 1024
        output_ceiling = 8192 if item["experiment"] in ("exp1", "exp2") else 32768
        if (output_ceiling > metadata.max_tokens or input_ceiling > metadata.max_input_tokens
                or input_ceiling + output_ceiling > 1_000_000):
            raise ValueError("A released prompt exceeds the verified native Fable limits")
        limits[item["experiment"] + "/" + item["item_id"]] = {
            "prompt_sha256": item["prompt"]["sha256"], "instructions_sha256": item["instructions"]["sha256"],
            "max_input_tokens": input_ceiling, "max_output_tokens": output_ceiling,
            "count_evidence": {"source": "anthropic.messages.count_tokens", "model": MODEL,
                "observed_at": datetime.now(UTC).isoformat(), "input_tokens": counted.input_tokens,
                "margin": "ceil(native_count * 1.05) + 1024", "model_max_input_tokens": metadata.max_input_tokens,
                "model_max_output_tokens": metadata.max_tokens, "tools": [], "thinking": "adaptive",
                "output_effort": item["authors_config"]["effort"]},
        }
        atomic_json(Path(output), limits)
    return limits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count-native-tokens", action="store_true", required=True,
                        help="Call only native model metadata and token-count endpoints, not generation")
    args = parser.parse_args()
    from anthropic import Anthropic
    prepare(args.source_root, args.output, client=Anthropic(max_retries=0, timeout=120))


if __name__ == "__main__":
    main()
