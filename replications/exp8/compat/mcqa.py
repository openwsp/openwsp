"""Import compatibility for an unreleased, grading-independent upstream helper.

The released mathqa.py imports this function for its optional loop_signals()
diagnostic. Neither extract() nor is_correct() uses it. Do not fabricate that
unreleased heuristic: fail explicitly if a caller attempts to use it.
"""


def loop_signals(text):
    raise NotImplementedError("Upstream mcqa.loop_signals was not released; it is not used for grading")
