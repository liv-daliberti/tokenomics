"""Docstring lint: fail if any module, class, function, or method in the
production code lacks a docstring. Pure standard library — no dependencies, runs
anywhere the repo does.

Usage:
    python scripts/lint_docstrings.py            # checks agora, analysis, web, scripts
    python scripts/lint_docstrings.py agora      # check specific dirs
Exit status is non-zero if anything is undocumented, so it works as a CI gate.
"""
from __future__ import annotations

import ast
import os
import sys
from typing import List, Tuple

DEFAULT_DIRS = ["agora", "analysis", "web", "scripts"]
Finding = Tuple[str, int, str]


def _has_docstring(node: ast.AST) -> bool:
    """True if a def/class/module node's first statement is a string literal."""
    body = getattr(node, "body", None)
    return bool(body and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str))


def check_file(path: str) -> List[Finding]:
    """Return (path, line, name) for every undocumented definition in one file."""
    tree = ast.parse(open(path).read(), path)
    missing: List[Finding] = []
    if not _has_docstring(tree):
        missing.append((path, 1, "<module>"))
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if not _has_docstring(node):
                missing.append((path, node.lineno, node.name))
    return missing


def collect(dirs: List[str]) -> List[Finding]:
    """Walk the given directories and gather every undocumented definition."""
    missing: List[Finding] = []
    for d in dirs:
        for root, _, files in os.walk(d):
            if "__pycache__" in root:
                continue
            for f in sorted(files):
                if f.endswith(".py"):
                    missing.extend(check_file(os.path.join(root, f)))
    return missing


def main(argv: List[str] = None) -> int:
    """CLI: print undocumented definitions and exit non-zero if any exist."""
    argv = argv if argv is not None else sys.argv[1:]
    dirs = argv or DEFAULT_DIRS
    missing = collect(dirs)
    for path, line, name in missing:
        print(f"{path}:{line}: missing docstring: {name}")
    if missing:
        print(f"\n✗ {len(missing)} undocumented definition(s)")
        return 1
    print(f"✓ docstring lint clean — every definition in {', '.join(dirs)} is documented")
    return 0


if __name__ == "__main__":
    sys.exit(main())
