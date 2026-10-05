"""Run the static release-asset validator and prove it discriminates.

A validator that passes on the committed tree proves little unless a mutated tree fails,
so each negative control below re-runs the validator against a copy carrying one defect
that has actually shipped in, or been found to evade the checks of, DIMER tutorials:
editable self-install in either spelling, hard-coded or rebound revision, persisted
outputs, drifting identity, conflicting release status, an enabled or non-form BYOD gate,
a required call surviving only in a comment, an isolated install without its hash check, and a restart
instruction.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
VALIDATOR = ROOT / "tools" / "validate_release_assets.py"
COPIED = (
    "MODEL_CARD.md",
    "README.md",
    "STATUS.md",
    "pyproject.toml",
    "docs",
    "tutorials",
    "src",
    "tools",
    "weights",
)
# weights/ is copied for its committed manifest only; the git-ignored checkpoints never enter the tmp tree.
IGNORED = shutil.ignore_patterns("__pycache__", ".cache", "*.ckpt", "*.safetensors", "*.bin", "*.pt", "*.pth")


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
            shutil.copytree(source, tmp_path / name, ignore=IGNORED)
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
    expected = [
        "model-card",
        "identity-consistency",
        "weight-facts",
        "release-status",
        "notebooks+parity",
        "workshop-notebooks",
    ]
    assert module.validate_all() == expected


@pytest.mark.parametrize("flag", ["'-e', ", "'--editable', "])
def test_control_editable_self_install_is_rejected(tree: Path, flag: str) -> None:
    module = _load_validator(tree)
    # A pip call that installs the tree editably, added to the isolated install cell; its markers stay intact.
    _replace_in_code(
        module,
        'SKIP_INSTALL = os.environ.get("DIMER_NOTEBOOK_CI_PREINSTALLED") == "1"\n',
        'SKIP_INSTALL = os.environ.get("DIMER_NOTEBOOK_CI_PREINSTALLED") == "1"\n'
        f"subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', {flag}'.'], check=True)\n",
    )
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


def test_control_isolated_install_without_hash_check_is_rejected(tree: Path) -> None:
    # WSP-M1: the stale-import restart guard is gone; the isolated install must keep --require-hashes.
    module = _load_validator(tree)
    _replace_in_code(module, '"--require-hashes", ', "")
    with pytest.raises(module.ValidationError, match="--require-hashes"):
        module.validate_notebooks()


def test_control_restart_instruction_is_rejected(tree: Path) -> None:
    module = _load_validator(tree)
    anchor = "os.makedirs('outputs', exist_ok=True)\n"
    _replace_in_code(module, anchor, anchor + "print('Restart the runtime, then rerun from the top.')\n")
    with pytest.raises(module.ValidationError, match="restart"):
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


WORKSHOP = "DIMER_Whisper_Speech_Recognition_Workshop.ipynb"


def _edit_workshop(module, mutate) -> None:
    path = module.ROOT / "tutorials" / WORKSHOP
    notebook = json.loads(path.read_text(encoding="utf-8"))
    mutate(notebook)
    path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


def _carried_cell(notebook: dict) -> dict:
    return next(c for c in notebook["cells"] if "".join(c["source"]).startswith("CARRIED_FILES = "))


def test_workshop_carried_files_verify_and_match_the_package() -> None:
    import ast
    import hashlib

    notebook = json.loads((ROOT / "tutorials" / WORKSHOP).read_text(encoding="utf-8"))
    body = ast.parse("".join(_carried_cell(notebook)["source"])).body
    files, hashes = (ast.literal_eval(node.value) for node in body[:2])
    assert {name: hashlib.sha256(text.encode()).hexdigest() for name, text in files.items()} == hashes
    pipeline = (ROOT / "src" / "whisper_asr_pipeline" / "pipeline.py").read_text(encoding="utf-8")
    assert files["whisper_reference.py"] == pipeline
    # The prepared 16 kHz waveform is reloaded as stored; re-validating it as learner audio rejected
    # a resampled 8 kHz clip whose peak overshoots 1.01.
    assert 'row["waveform"], rate = sf.read(row["audio_path"], dtype="float32")' in files["workshop.py"]


def test_control_workshop_tampered_carried_file_is_rejected(tree: Path) -> None:
    module = _load_validator(tree)

    def mutate(notebook: dict) -> None:
        cell = _carried_cell(notebook)
        text = "".join(cell["source"])
        cell["source"] = [text.replace("MIN_CHUNK_LENGTH_S = 1", "MIN_CHUNK_LENGTH_S = 2", 1)]

    _edit_workshop(module, mutate)
    with pytest.raises(module.ValidationError, match="CARRIED_HASHES digest"):
        module.validate_workshop_notebooks()


def test_control_workshop_pipeline_drift_is_rejected(tree: Path) -> None:
    module = _load_validator(tree)
    pipeline = tree / "src" / "whisper_asr_pipeline" / "pipeline.py"
    pipeline.write_text(pipeline.read_text(encoding="utf-8") + "\n# drift\n", encoding="utf-8")
    with pytest.raises(module.ValidationError, match="differs from src/whisper_asr_pipeline/pipeline.py"):
        module.validate_workshop_notebooks()


def test_control_workshop_persisted_output_is_rejected(tree: Path) -> None:
    module = _load_validator(tree)

    def mutate(notebook: dict) -> None:
        _carried_cell(notebook)["execution_count"] = 1

    _edit_workshop(module, mutate)
    with pytest.raises(module.ValidationError, match="persists outputs"):
        module.validate_workshop_notebooks()


@pytest.mark.parametrize("key", ["notebook_profile", "notebook_mode"])
def test_control_workshop_metadata_key_is_required(tree: Path, key: str) -> None:
    module = _load_validator(tree)
    _edit_workshop(module, lambda notebook: notebook["metadata"]["dimer"].pop(key))
    with pytest.raises(module.ValidationError, match=key):
        module.validate_workshop_notebooks()
