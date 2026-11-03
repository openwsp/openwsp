"""Reconstructed parser for the authors' small numeric edit language.

This is not an upstream scorer: the released prompts specify the grammar but
the authors did not release its implementation. Nothing in a reply is executed.
"""

from __future__ import annotations

import ast
import re

import numpy as np
import torch

from openwsp.core import WeightStore


def _literal(node):
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        value = node.value
    elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        value = _literal(node.operand)
        if not isinstance(value, (int, float)):
            raise ValueError("Unary sign requires a number")
        value = -value if isinstance(node.op, ast.USub) else value
    elif isinstance(node, (ast.List, ast.Tuple)):
        return [_literal(part) for part in node.elts]
    else:
        raise ValueError("Only numeric literals and lists are accepted")
    if not np.isfinite(value) or abs(value) > 1e6:
        raise ValueError("Numeric literal exceeds the prompt's limit")
    return value


def _index(node):
    if isinstance(node, ast.Tuple):
        return tuple(_index(part) for part in node.elts)
    if isinstance(node, ast.Slice):
        return slice(*(_index(part) if part is not None else None for part in (node.lower, node.upper, node.step)))
    value = _literal(node)
    if type(value) is not int:
        raise ValueError("An index must be an integer")
    return value


def _statements(block):
    pending = []
    depth = 0
    for line in block.splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        if pending and re.match(r"(?:ZERO\s+[A-Za-z_]|[A-Za-z_][\w.]*\s*\[)", line):
            # An unfinished numeric list cannot contain another tensor
            # statement. Skip that malformed statement and preserve the next.
            yield " ".join(pending)
            pending, depth = [], 0
        pending.append(line)
        depth += line.count("[") - line.count("]")
        if depth <= 0:
            yield " ".join(pending)
            pending, depth = [], 0
    if pending:
        yield " ".join(pending)


def apply_numeric_edits(answer: str, store: WeightStore) -> dict:
    """Apply the last complete edits fence, skipping invalid statements in order."""
    blocks = re.findall(r"(?m)^```edits\s*\n(.*?)^```\s*$", answer, re.DOTALL)
    if not blocks:
        return {"block_found": False, "statements": 0, "applied": 0, "errors": []}
    report = {"block_found": True, "statements": 0, "applied": 0, "errors": []}
    for number, statement in enumerate(_statements(blocks[-1]), 1):
        report["statements"] += 1
        try:
            zero = re.fullmatch(r"ZERO\s+([A-Za-z_][\w.]*)", statement)
            if zero:
                name = zero[1]
                value = store.load(name)
                value.zero_()
            else:
                match = re.fullmatch(r"([A-Za-z_][\w.]*)(\[.*?\])\s*(=|\+=|-=|\*=)\s*(.+)", statement)
                if not match:
                    raise ValueError("Invalid edit statement")
                name, indices, operation, raw = match.groups()
                value = store.load(name)
                target = ast.parse("x" + indices, mode="eval").body
                if not isinstance(target, ast.Subscript) or not isinstance(target.value, ast.Name):
                    raise ValueError("Invalid tensor selection")
                index = _index(target.slice)
                parsed = _literal(ast.parse(raw, mode="eval").body)
                array = value.numpy()
                rhs = np.asarray(parsed, dtype=array.dtype)
                selected = array[index]
                if rhs.ndim and rhs.shape != selected.shape:
                    raise ValueError("List shape must exactly match the selection")
                if operation == "=":
                    replacement = rhs
                elif operation == "+=":
                    replacement = selected + rhs
                elif operation == "-=":
                    replacement = selected - rhs
                else:
                    replacement = selected * rhs
                if not np.isfinite(replacement).all():
                    raise ValueError("Nonfinite result")
                array[index] = replacement
            store.save_edit(name, value)
            report["applied"] += 1
        except (ValueError, TypeError, KeyError, IndexError, RuntimeError, SyntaxError, OverflowError) as exc:
            report["errors"].append({"statement": number, "error_type": type(exc).__name__})
    return report


def zero_minatar_store() -> WeightStore:
    shapes = {"W1": (600, 32), "b1": (32,), "W2": (32, 32), "b2": (32,), "W3": (32, 6), "b3": (6,)}
    return WeightStore({name: torch.zeros(shape) for name, shape in shapes.items()}, refuse_rescale=False)


def shown_car_store(prompt: str) -> WeightStore:
    """Recover the rounded numbers actually shown, never a higher precision checkpoint."""
    header = re.compile(r"^([\w.]+) \((\d+)(?: x (\d+))?\), scale (1e-\d+):", re.MULTILINE)
    matches = list(header.finditer(prompt))
    arrays = {}
    for i, match in enumerate(matches):
        name, rows, columns, scale = match.groups()
        block = prompt[match.end():matches[i + 1].start() if i + 1 < len(matches) else None]
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if columns:
            numbered = [re.fullmatch(r"(\d+):\s*([-\d\s]+)", line) for line in lines]
            numbered = [row for row in numbered if row]
            if [int(row[1]) for row in numbered] != list(range(int(rows))):
                raise ValueError("Missing or reordered shown weight rows")
            values = [[int(x) for x in row[2].split()] for row in numbered]
            expected = (int(rows), int(columns))
        else:
            numeric = [line for line in lines if re.fullmatch(r"[-\d\s]+", line)]
            values = [int(x) for line in numeric for x in line.split()]
            expected = (int(rows),)
        array = np.asarray(values, dtype=np.float64) * float(scale)
        if array.shape != expected:
            raise ValueError("Shown weight dimensions disagree")
        arrays[name] = torch.from_numpy(array.astype(np.float32))
    expected_names = {"embed.W", "embed.b", "actor_head.W", "actor_head.b"}
    expected_names.update(f"actor{i}.{suffix}" for i in range(5) for suffix in ("W", "b"))
    if set(arrays) != expected_names:
        raise ValueError("Prompt does not contain the complete shown actor")
    return WeightStore(arrays, refuse_rescale=False)
