"""Enforce the docstring lint: every production definition must be documented.

This runs scripts/lint_docstrings.py over the library code so a missing docstring
fails the test suite (a lint that lives with the tests).
"""
from __future__ import annotations

import importlib.util
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_lint():
    """Import the standalone docstring linter by path (scripts/ is not a package)."""
    path = os.path.join(REPO, "scripts", "lint_docstrings.py")
    spec = importlib.util.spec_from_file_location("lint_docstrings", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_all_production_code_has_docstrings():
    lint = _load_lint()
    cwd = os.getcwd()
    os.chdir(REPO)
    try:
        missing = lint.collect(lint.DEFAULT_DIRS)
    finally:
        os.chdir(cwd)
    assert not missing, "undocumented definitions:\n" + "\n".join(
        f"  {p}:{ln} {name}" for p, ln, name in missing)


if __name__ == "__main__":
    test_all_production_code_has_docstrings()
    print("ok  docstring lint")
