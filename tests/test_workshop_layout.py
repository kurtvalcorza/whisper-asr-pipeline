"""Source-layout checks for the speech-recognition workshop notebook (no overlong cell lines)."""

from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKSHOP = ROOT / "tutorials" / "DIMER_Whisper_Speech_Recognition_Workshop.ipynb"


def _load_splitter():
    path = ROOT / "tools" / "split_workshop_carrier.py"
    spec = importlib.util.spec_from_file_location("split_workshop_carrier", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_no_workshop_cell_line_exceeds_2000_characters() -> None:
    notebook = json.loads(WORKSHOP.read_text(encoding="utf-8"))
    for cell in notebook["cells"]:
        longest = max(len(line) for line in "".join(cell["source"]).split("\n"))
        assert longest <= 2000, f"{cell.get('id')}: line of {longest} characters"
    assert _load_splitter().long_lines(notebook) == []


def test_carried_literal_round_trips() -> None:
    splitter = _load_splitter()
    files = {
        "empty.txt": "",
        "long.txt": "x" * 2500 + "\n",
        "multi.py": "a = 1\n\nb = 'two'\nno newline at end",
    }
    literal = splitter.carried_literal(files)
    assert ast.literal_eval(literal) == files
    assert max(len(line) for line in literal.split("\n")) <= splitter.CARRIER_PIECE + 20
    source = f"CARRIED_FILES = {files!r}\nCARRIED_HASHES = {{}}\n"
    rewritten = splitter.split_carrier(source)
    assert rewritten.endswith("\nCARRIED_HASHES = {}\n")
    assert splitter.split_carrier(rewritten) == rewritten
