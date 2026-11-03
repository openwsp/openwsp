"""Source-described label scoring, explicitly reconstructed rather than upstream code."""

import re

ALIASES = {
    "exp1": {"breakout": ("breakout",), "freeway": ("freeway", "frogger"),
             "pacman": ("pacman", "pac-man", "pac man"), "pong": ("pong",)},
    "exp2": {"circle": ("circle", "disc", "disk", "round", "oval"),
             "square": ("square", "rectangle", "box"), "triangle": ("triangle", "pyramid", "wedge")},
}


def score_label(experiment, expected, answer):
    """Score only the requested answer section; ambiguous or absent labels fail."""
    if experiment not in ALIASES or expected not in ALIASES[experiment]:
        raise ValueError("Unknown label task")
    label = "Game" if experiment == "exp1" else "Images"
    section = re.search(rf"(?im)^\s*(?:#+\s*)?(?:\*\*)?{label}(?:\*\*)?\s*:?[ \t]*(.*)$", answer)
    content = ""
    if section:
        content = section[1].strip(" *")
        if not content:
            following = answer[section.end():].strip().splitlines()
            content = following[0] if following else ""
    matches = sorted(name for name, aliases in ALIASES[experiment].items()
                     if any(re.search(r"\b" + re.escape(alias) + r"(?:s)?\b", content, re.IGNORECASE) for alias in aliases))
    return {"correct": matches == [expected], "expected": expected, "predicted": matches,
            "answer_section_found": bool(section), "scorer": "openwsp-reconstructed-label-v1"}
