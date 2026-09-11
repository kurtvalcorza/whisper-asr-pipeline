from __future__ import annotations

import ast
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REQUIRED_CARD_HEADINGS = [
    (6, "Description"),
    (4, "Intended Use and Limitations"),
    (6, "Primary Intended Uses"),
    (6, "Primary Intended Users"),
    (6, "Out-of-scope use cases"),
    (4, "Factors"),
    (6, "Groups"),
    (6, "Instrumentation"),
    (6, "Environment"),
    (4, "Metrics"),
    (6, "Performance Measures"),
    (6, "Decision thresholds"),
    (6, "Approaches to uncertainty and variability"),
    (4, "Ethical considerations and biases"),
    (6, "Data"),
    (6, "Human Life"),
    (6, "Mitigations"),
    (6, "Risks and harms"),
    (6, "Use cases"),
]
ALLOWED_PROFILES = {"E2E", "ARTIFACT-INFERENCE", "TASK-INFERENCE", "MULTI-CAPABILITY", "SMOKE"}
FORBIDDEN = re.compile(r"\b(TODO|TBD|FIXME)\b|Insert text here|Tooltip:", re.I)


def validate_model_card() -> None:
    text = (ROOT / "MODEL_CARD.md").read_text(encoding="utf-8")
    assert text.startswith("---\n"), "MODEL_CARD.md must start with YAML front matter"
    front = text.split("---", 2)[1]
    for key in ("license:", "model_card_spec:", "base_model:"):
        assert key in front, f"missing front-matter field: {key}"
    assert 'model_card_spec: "1.0"' in front, "model_card_spec must be 1.0"
    assert not FORBIDDEN.search(text), "MODEL_CARD.md contains placeholder/scaffolding text"
    h1 = re.findall(r"(?m)^# (?!#)(.+)$", text)
    assert len(h1) == 1, f"MODEL_CARD.md must contain exactly one H1, got {len(h1)}"
    found = []
    for line in text.splitlines():
        match = re.match(r"^(#{1,6})\s+(.+?)\s*$", line)
        if match:
            found.append((len(match.group(1)), match.group(2).strip()))
    positions = []
    for heading in REQUIRED_CARD_HEADINGS:
        matches = [
            index
            for index, item in enumerate(found)
            if item[0] == heading[0] and item[1].casefold() == heading[1].casefold()
        ]
        assert len(matches) == 1, f"required model-card heading missing/duplicated: {heading}"
        positions.append(matches[0])
    assert positions == sorted(positions), "required model-card headings are out of order"


def _compile_notebook(path: Path) -> None:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    profile = notebook.get("metadata", {}).get("dimer", {}).get("notebook_profile")
    assert profile in ALLOWED_PROFILES, f"{path}: missing/invalid metadata.dimer.notebook_profile"
    assert not FORBIDDEN.search(path.read_text(encoding="utf-8")), f"{path}: placeholder text found"
    for index, cell in enumerate(notebook.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        stripped = "\n".join(
            line for line in source.splitlines() if not line.lstrip().startswith(("!", "%"))
        )
        if stripped.strip():
            try:
                ast.parse(stripped)
            except SyntaxError as exc:
                raise AssertionError(f"{path}: code cell {index} does not compile: {exc}") from exc


def validate_notebooks() -> None:
    tutorials = ROOT / "tutorials"
    notebooks = sorted(tutorials.glob("*.ipynb"))
    assert notebooks, "at least one tutorial notebook is required"
    registry = (tutorials / "README.md").read_text(encoding="utf-8")
    for path in notebooks:
        _compile_notebook(path)
        assert path.name in registry, f"{path.name} missing from tutorials/README.md"


if __name__ == "__main__":
    validate_model_card()
    validate_notebooks()
    print("release asset validation: PASS")
