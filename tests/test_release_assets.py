"""Run the static release-asset validator and prove it discriminates.

A validator that passes on the committed tree proves little unless a mutated tree fails,
so each negative control below re-runs the validator against a copy carrying one defect
that has actually shipped in, or been found to evade the checks of, DIMER tutorials:
editable self-install in either spelling, hard-coded or rebound revision, persisted
outputs, drifting identity, conflicting release status, an enabled or non-form BYOD gate,
a required call surviving only in a comment, and a stale-import guard that no longer raises.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
VALIDATOR = ROOT / "tools" / "validate_release_assets.py"
COPIED = ("MODEL_CARD.md", "README.md", "STATUS.md", "docs", "tutorials", "src")


def _load_validator(root: Path):
    spec = importlib.util.spec_from_file_location("validate_release_assets", VALIDATOR)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.ROOT = root
    return module


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    for name in COPIED:
        source = ROOT / name
        if source.is_dir():
            shutil.copytree(source, tmp_path / name, ignore=shutil.ignore_patterns("__pycache__"))
        else:
            shutil.copy2(source, tmp_path / name)
    return tmp_path


def _notebook_path(module) -> Path:
    return module.ROOT / "tutorials" / module.NOTEBOOK_NAME


def _edit_notebook(module, mutate) -> None:
    path = _notebook_path(module)
    notebook = json.loads(path.read_text(encoding="utf-8"))
    mutate(notebook)
    path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


def _replace_in_code(module, old: str, new: str) -> None:
    def mutate(notebook: dict) -> None:
        hits = 0
        for cell in notebook["cells"]:
            if cell["cell_type"] != "code":
                continue
            text = "".join(cell["source"])
            if old in text:
                hits += 1
                cell["source"] = [text.replace(old, new)]
        assert hits, f"control marker not found in notebook: {old!r}"

    _edit_notebook(module, mutate)


def _first_marker_line(module) -> str:
    """A required profile-specific call that appears as a whole source line."""
    notebook = json.loads(_notebook_path(module).read_text(encoding="utf-8"))
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        for line in "".join(cell["source"]).splitlines():
            if any(line.strip() == marker for marker in module.CODE_MARKERS):
                return line
    raise AssertionError("no whole-line CODE_MARKER found to use as a control")


def test_committed_tree_passes() -> None:
    module = _load_validator(ROOT)
    assert module.validate_all() == ["model-card", "identity-consistency", "release-status", "notebook"]


@pytest.mark.parametrize("flag", ["'-e', ", "'--editable', "])
def test_control_editable_self_install_is_rejected(tree: Path, flag: str) -> None:
    module = _load_validator(tree)
    _replace_in_code(module, "'pip', 'install', '-q', ", f"'pip', 'install', '-q', {flag}")
    with pytest.raises(module.ValidationError, match="editable self-install"):
        module.validate_notebooks()


def test_control_missing_profile_is_rejected(tree: Path) -> None:
    module = _load_validator(tree)
    _edit_notebook(module, lambda notebook: notebook["metadata"]["dimer"].pop("notebook_profile"))
    with pytest.raises(module.ValidationError, match="notebook_profile"):
        module.validate_notebooks()


def test_control_persisted_output_is_rejected(tree: Path) -> None:
    module = _load_validator(tree)

    def mutate(notebook: dict) -> None:
        cell = next(cell for cell in notebook["cells"] if cell["cell_type"] == "code")
        cell["execution_count"] = 1

    _edit_notebook(module, mutate)
    with pytest.raises(module.ValidationError, match="execution_count"):
        module.validate_notebooks()


def test_control_hard_coded_revision_is_rejected(tree: Path) -> None:
    module = _load_validator(tree)
    _, revision = module._package_identity()
    _replace_in_code(module, "'revision': MODEL_REVISION", f"'revision': '{revision}'")
    with pytest.raises(module.ValidationError, match="revision"):
        module.validate_notebooks()


def test_control_rebound_identity_is_rejected(tree: Path) -> None:
    module = _load_validator(tree)
    _replace_in_code(
        module,
        "print({'model_id': MODEL_ID, 'revision': MODEL_REVISION",
        "MODEL_REVISION = 'abc1234'\nprint({'model_id': MODEL_ID, 'revision': MODEL_REVISION",
    )
    with pytest.raises(module.ValidationError, match="must not be rebound"):
        module.validate_notebooks()


@pytest.mark.parametrize("spelling", ["{gate} = True", "{gate}=True"])
def test_control_enabled_byod_gate_is_rejected(tree: Path, spelling: str) -> None:
    module = _load_validator(tree)
    gate = module.BYOD_GATES[0]
    _replace_in_code(module, f"{gate} = False", f"{gate} = False\n{spelling.format(gate=gate)}")
    with pytest.raises(module.ValidationError, match="assigned exactly once"):
        module.validate_notebooks()


def test_control_byod_gate_without_form_annotation_is_rejected(tree: Path) -> None:
    module = _load_validator(tree)
    gate = module.BYOD_GATES[0]
    _replace_in_code(module, f'{gate} = False  # @param {{type:"boolean"}}', f"{gate} = False")
    with pytest.raises(module.ValidationError, match="Colab form parameter"):
        module.validate_notebooks()


def test_control_required_call_only_in_comment_is_rejected(tree: Path) -> None:
    module = _load_validator(tree)
    line = _first_marker_line(module)
    _replace_in_code(module, line, f"# {line}")
    with pytest.raises(module.ValidationError, match="missing required source markers"):
        module.validate_notebooks()


def test_control_guard_that_no_longer_raises_is_rejected(tree: Path) -> None:
    module = _load_validator(tree)
    _replace_in_code(
        module,
        "    if stale:\n        raise RuntimeError(",
        "    if stale:\n        print(  # Restart the runtime, then rerun from the top.\n            ",
    )
    with pytest.raises(module.ValidationError, match="must raise RuntimeError"):
        module.validate_notebooks()


def test_control_identity_drift_is_rejected(tree: Path) -> None:
    module = _load_validator(tree)
    _, revision = module._package_identity()
    readme = tree / "README.md"
    readme.write_text(readme.read_text(encoding="utf-8").replace(revision, "0" * 40), encoding="utf-8")
    with pytest.raises(module.ValidationError, match="README.md"):
        module.validate_identity_consistency()


def test_control_conflicting_release_status_is_rejected(tree: Path) -> None:
    module = _load_validator(tree)
    status = tree / "STATUS.md"
    promoted = status.read_text(encoding="utf-8").replace("**Candidate", "**Release-grade")
    status.write_text(promoted, encoding="utf-8")
    with pytest.raises(module.ValidationError, match="status"):
        module.validate_release_status()


def test_control_placeholder_in_model_card_is_rejected(tree: Path) -> None:
    module = _load_validator(tree)
    card = tree / "MODEL_CARD.md"
    card.write_text(card.read_text(encoding="utf-8") + "\nTODO: fill in.\n", encoding="utf-8")
    with pytest.raises(module.ValidationError, match="placeholder"):
        module.validate_model_card()
