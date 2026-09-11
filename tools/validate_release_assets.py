"""Static release-asset validation for the Whisper ASR DIMER pipeline.

Checks the tutorial notebook, tutorial registry, model card, README, STATUS.md and
weight documentation for DIMER Notebook Specification 1.0 / Model Card Specification 1.0
source conformance and cross-document identity consistency.

This is source validation only. A PASS here is NOT clean-runtime execution evidence;
the release gate is defined in docs/release-verification.md.
"""
from __future__ import annotations

import ast
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "whisper_asr_pipeline"
REPO_NAME = "whisper-asr-pipeline"
NOTEBOOK_NAME = "whisper_asr_colab.ipynb"
EXPECTED_PROFILE = "TASK-INFERENCE"
EXPECTED_MODEL_ID = "openai/whisper-large-v3-turbo"
# Additional 40-hex revisions a document may legitimately cite (none by default).
KNOWN_SHAS: frozenset[str] = frozenset(())
# Colab form gates that must default to the non-interactive sample path.
BYOD_GATES = ('USE_BYOD',)
# Machine-readable artifacts the notebook must write.
EXPECTED_OUTPUTS = ('outputs/whisper_asr_result.json',)
# Profile-specific code the notebook must exercise through the repository public API.
CODE_MARKERS = (
    'from whisper_asr_pipeline import MODEL_ID, MODEL_REVISION, WhisperASRPipeline, word_error_rate',
    'pipe = WhisperASRPipeline.from_pretrained()',
    "result = pipe.transcribe(audio_input, language='en', task='transcribe')",
    "metrics['word_error_rate'] = word_error_rate(reference, result['text'])",
    "load_dataset('hf-internal-testing/librispeech_asr_dummy', 'clean', split='validation')",
    "ds.cast_column('audio', Audio(decode=False))",
    'sf.read(io.BytesIO(',
    "'model_revision'",
    'transformers.__version__',
    "'device': pipe.device",
)
# Profile-specific learner-facing statements.
MARKDOWN_MARKERS = (
    '**Capability:** multilingual automatic speech recognition',
    'WER is computed only when a reference transcript is available',
    'no diarization, speaker identity, biometric inference',
)
FORBIDDEN_CODE_EXTRA = (
    'from transformers import',
    'AutoModelForSpeechSeq2Seq',
)

# ---------------------------------------------------------------------------
# Shared checks. Everything below is source/structure validation only. Passing
# these checks is NOT clean-runtime execution evidence under DIMER Notebook
# Specification 1.0; see docs/release-verification.md for the release gate.
# ---------------------------------------------------------------------------

ALLOWED_PROFILES = {"E2E", "ARTIFACT-INFERENCE", "TASK-INFERENCE", "MULTI-CAPABILITY", "SMOKE"}
STATUS_TOKENS = ("Candidate", "Release-grade")
PLACEHOLDER = re.compile(r"\b(TODO|TBD|FIXME)\b|Insert text here|Tooltip:", re.I)
SHA40 = re.compile(r"^[0-9a-f]{40}$")
# Affirmative claims that tutorial execution cannot support (Notebook Spec: no unsupported
# release-grade, benchmark, or deployment claims). Negated/comparative phrasing is allowed.
UNSUPPORTED_CLAIMS = re.compile(
    r"\b(production[- ]ready|battle[- ]tested|state[- ]of[- ]the[- ]art results (were|are) reproduced"
    r"|benchmark superiority (is|was) (shown|established)|is release-grade|now release-grade)\b",
    re.I,
)
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
# Markers every DIMER tutorial in this fleet must carry, independent of profile.
COMMON_CODE_MARKERS = (
    "REPO_REF = os.environ.get('DIMER_TUTORIAL_REF', 'main')",
    "if not (ROOT / 'pyproject.toml').exists():",
    "'git', 'clone'",
    "'checkout', '--detach', 'FETCH_HEAD'",
    "REPO_SHA = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()",
    "Restart the runtime, then rerun from the top.",
    "platform.python_version()",
    "torch.__version__",
    "'repository_revision': REPO_SHA",
    "os.makedirs('outputs', exist_ok=True)",
    "from google.colab import files",
    "files.upload()",
)
COMMON_MARKDOWN_MARKERS = (
    "**Notebook specification:** DIMER Notebook Specification 1.0",
    "**Learning objectives:**",
    "## Prerequisites",
    "Do not upload confidential or restricted",
    "## 1. Bootstrap the repository and pinned runtime",
    "`DIMER_TUTORIAL_REF`",
    "## Interpretation and limits",
    "Successful execution proves that the recorded repository revision",
    "It does **not** establish benchmark superiority",
    "## References",
    "- Repository model card: `../MODEL_CARD.md`",
)
# Patterns that must never appear in tutorial code: credential-in-URL, unpinned trust,
# unsafe deserialization, notebook magics (the executable harness runs plain Python), and an
# editable self-install, which is not importable in the same interpreter until it restarts.
FORBIDDEN_CODE = (
    "https://x-access-token:",
    "@github.com/",
    "trust_remote_code=True",
    "pickle.load",
    "torch.load(",
    "extractall(",
    "pip install -e ",
    "'install', '-e'",
    "'install', '-q', '-e'",
    "%pip",
    "!pip",
)


class ValidationError(AssertionError):
    """Raised for any release-asset defect; the message names the file and rule."""


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _cell_source(cell: dict) -> str:
    value = cell.get("source", "")
    return "".join(value) if isinstance(value, list) else value


def _package_identity() -> tuple[str, str]:
    """Read MODEL_ID / MODEL_REVISION from the package source without importing torch."""
    text = _read(ROOT / "src" / PACKAGE / "pipeline.py")
    model_id = re.search(r'^MODEL_ID = "([^"]+)"$', text, re.M)
    revision = re.search(r'^MODEL_REVISION = "([^"]+)"$', text, re.M)
    _check(
        model_id is not None and revision is not None,
        "pipeline.py must define MODEL_ID and MODEL_REVISION",
    )
    _check(SHA40.match(revision.group(1)) is not None, "MODEL_REVISION must be a 40-hex immutable commit")
    _check(model_id.group(1) == EXPECTED_MODEL_ID, f"MODEL_ID drifted from {EXPECTED_MODEL_ID}")
    return model_id.group(1), revision.group(1)


def validate_model_card() -> None:
    path = ROOT / "MODEL_CARD.md"
    text = _read(path)
    _check(text.startswith("---\n"), "MODEL_CARD.md must start with YAML front matter")
    front = text.split("---", 2)[1]
    for key in ("license:", "model_card_spec:", "base_model:"):
        _check(key in front, f"MODEL_CARD.md missing front-matter field: {key}")
    _check('model_card_spec: "1.0"' in front, "MODEL_CARD.md model_card_spec must be 1.0")
    _check(f"base_model: {EXPECTED_MODEL_ID}" in front, "MODEL_CARD.md base_model must equal MODEL_ID")
    _check(not PLACEHOLDER.search(text), "MODEL_CARD.md contains placeholder/scaffolding text")
    _check(not UNSUPPORTED_CLAIMS.search(text), "MODEL_CARD.md makes an unsupported release/benchmark claim")
    h1 = re.findall(r"(?m)^# (?!#)(.+)$", text)
    _check(len(h1) == 1, f"MODEL_CARD.md must contain exactly one H1, got {len(h1)}")
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
        _check(len(matches) == 1, f"required model-card heading missing/duplicated: {heading}")
        positions.append(matches[0])
    _check(positions == sorted(positions), "required model-card headings are out of order")
    _check("## Immutable provenance" in text, "MODEL_CARD.md must carry an '## Immutable provenance' section")


def validate_identity_consistency() -> None:
    """The immutable upstream identity must be the same string in every document."""
    model_id, revision = _package_identity()
    for name in ("README.md", "MODEL_CARD.md", "docs/WEIGHTS.md"):
        text = _read(ROOT / name)
        _check(model_id in text, f"{name} must name the upstream model `{model_id}`")
        _check(revision in text, f"{name} must cite the immutable revision {revision}")
        other = re.findall(r"\b[0-9a-f]{40}\b", text)
        stray = sorted({sha for sha in other if sha != revision and sha not in KNOWN_SHAS})
        _check(not stray, f"{name} cites an unexpected 40-hex revision: {stray}")


def validate_release_status() -> None:
    """STATUS.md, README.md and tutorials/README.md must agree on one status token."""
    status = _read(ROOT / "STATUS.md")
    match = re.search(r"Current status: \*\*(Candidate|Release-grade)\b", status)
    _check(match is not None, "STATUS.md must declare 'Current status: **Candidate**' or '**Release-grade**'")
    token = match.group(1)
    readme = _read(ROOT / "README.md")
    _check("## Release status" in readme, "README.md must have a '## Release status' section")
    section = readme.split("## Release status", 1)[1]
    _check(section.lstrip().startswith(f"**{token}"), f"README.md release status must open with **{token}**")
    registry = _read(ROOT / "tutorials" / "README.md").replace("**", "")
    _check(f"| {token}" in registry, f"tutorials/README.md must record the {token} status")
    other = [t for t in STATUS_TOKENS if t != token]
    for name, text in (("README.md", section.replace("**", "")), ("tutorials/README.md", registry)):
        for stale in other:
            _check(f"| {stale}" not in text, f"{name} carries a conflicting status token")
    if token == "Candidate":
        _check(
            "docs/release-verification.md" in registry or "release-verification" in registry,
            "tutorials/README.md must point Candidate notebooks at docs/release-verification.md",
        )
    for name in ("README.md", "STATUS.md", "tutorials/README.md", "docs/release-verification.md"):
        text = _read(ROOT / name)
        _check(not PLACEHOLDER.search(text), f"{name} contains placeholder text")
        _check(not UNSUPPORTED_CLAIMS.search(text), f"{name} makes an unsupported release/benchmark claim")
    verification = _read(ROOT / "docs" / "release-verification.md")
    _check(
        "## Recorded executions" in verification,
        "docs/release-verification.md must have '## Recorded executions'",
    )


def _validate_notebook_structure(path: Path, notebook: dict) -> tuple[str, str]:
    _check(notebook.get("nbformat") == 4, f"{path.name}: nbformat must be 4")
    dimer = notebook.get("metadata", {}).get("dimer")
    _check(isinstance(dimer, dict), f"{path.name}: metadata.dimer block is required")
    profile = dimer.get("notebook_profile")
    _check(profile in ALLOWED_PROFILES, f"{path.name}: invalid metadata.dimer.notebook_profile {profile!r}")
    _check(profile == EXPECTED_PROFILE, f"{path.name}: profile {profile!r} != declared {EXPECTED_PROFILE!r}")
    # NOTEBOOK_SPEC prescribes the profile declaration but not the spec-version key; accept the
    # `notebook_spec` spelling used here and the `_version` spelling used by sibling repositories.
    spec = dimer.get("notebook_spec", dimer.get("notebook_spec_version"))
    _check(spec == "1.0", f"{path.name}: metadata.dimer must declare notebook spec version '1.0'")
    cells = notebook.get("cells", [])
    _check(
        bool(cells) and cells[0].get("cell_type") == "markdown",
        f"{path.name}: first cell must be markdown",
    )
    code_parts: list[str] = []
    markdown_parts: list[str] = []
    for index, cell in enumerate(cells):
        source = _cell_source(cell)
        if cell.get("cell_type") == "markdown":
            markdown_parts.append(source)
            continue
        _check(cell.get("cell_type") == "code", f"{path.name}: unexpected cell type at {index}")
        _check(cell.get("execution_count") is None, f"{path.name}: code cell {index} has execution_count")
        _check(not cell.get("outputs"), f"{path.name}: code cell {index} persists outputs")
        _check(
            index > 0 and cells[index - 1].get("cell_type") == "markdown",
            f"{path.name}: code cell {index} lacks a preceding explanatory markdown cell",
        )
        for line in source.splitlines():
            _check(not line.lstrip().startswith(("%", "!")), f"{path.name}: cell {index} uses a magic")
        try:
            ast.parse(source)
        except SyntaxError as exc:
            raise ValidationError(f"{path.name}: code cell {index} does not compile: {exc}") from exc
        code_parts.append(source)
    code = "\n".join(code_parts)
    markdown = "\n".join(markdown_parts)
    _check(not PLACEHOLDER.search(code + markdown), f"{path.name}: placeholder text found")
    _check(not UNSUPPORTED_CLAIMS.search(markdown), f"{path.name}: unsupported release/benchmark claim")
    return code, markdown


def _validate_notebook_content(path: Path, code: str, markdown: str) -> None:
    model_id, revision = _package_identity()
    _check(
        f"REPO_URL = 'https://github.com/kurtvalcorza/{ROOT.name}.git'" in code
        or f"REPO_URL = 'https://github.com/kurtvalcorza/{REPO_NAME}.git'" in code,
        f"{path.name}: bootstrap must clone this repository by its canonical URL",
    )
    _check(f"REPO_NAME = '{REPO_NAME}'" in code, f"{path.name}: REPO_NAME must be {REPO_NAME}")
    missing = [marker for marker in COMMON_CODE_MARKERS + CODE_MARKERS if marker not in code]
    _check(not missing, f"{path.name}: missing required source markers: {missing}")
    present = [marker for marker in FORBIDDEN_CODE + FORBIDDEN_CODE_EXTRA if marker in code]
    _check(not present, f"{path.name}: forbidden/insecure source markers: {present}")
    for gate in BYOD_GATES:
        _check(f"{gate} = False" in code, f"{path.name}: {gate} must default to False")
        _check(f"{gate} = True" not in code, f"{path.name}: {gate} must not be enabled in committed source")
    _check("from google.colab import files" in code, f"{path.name}: BYOD path must use google.colab.files")
    _check("import google.colab" not in code, f"{path.name}: google.colab import must stay inside the gate")
    _check(
        "MODEL_REVISION" in code and "MODEL_ID" in code,
        f"{path.name}: must use MODEL_ID and MODEL_REVISION",
    )
    _check(revision not in code, f"{path.name}: model revision must be imported, not hard-coded")
    for filename in EXPECTED_OUTPUTS:
        _check(filename in code, f"{path.name}: must export {filename}")
    missing_md = [marker for marker in COMMON_MARKDOWN_MARKERS + MARKDOWN_MARKERS if marker not in markdown]
    _check(not missing_md, f"{path.name}: missing learner-facing markers: {missing_md}")
    _check(f"**Profile:** `{EXPECTED_PROFILE}`" in markdown, f"{path.name}: markdown must state the profile")
    _check(f"https://huggingface.co/{model_id}" in markdown, f"{path.name}: references must link {model_id}")


def validate_notebooks() -> None:
    tutorials = ROOT / "tutorials"
    notebooks = sorted(tutorials.glob("*.ipynb"))
    _check(len(notebooks) == 1, f"exactly one tutorial notebook is expected, found {len(notebooks)}")
    path = notebooks[0]
    _check(path.name == NOTEBOOK_NAME, f"tutorial notebook must be named {NOTEBOOK_NAME}, found {path.name}")
    notebook = json.loads(_read(path))
    code, markdown = _validate_notebook_structure(path, notebook)
    _validate_notebook_content(path, code, markdown)
    registry = _read(tutorials / "README.md")
    _check(f"`{path.name}`" in registry, f"{path.name} missing from tutorials/README.md")
    _check(f"`{EXPECTED_PROFILE}`" in registry, f"tutorials/README.md must record `{EXPECTED_PROFILE}`")
    _check("DIMER Notebook Specification 1.0" in registry, "tutorials/README.md must name the notebook spec")


def validate_all() -> list[str]:
    validate_model_card()
    validate_identity_consistency()
    validate_release_status()
    validate_notebooks()
    return ["model-card", "identity-consistency", "release-status", "notebook"]


def main() -> int:
    passed = validate_all()
    print(f"release asset validation: PASS ({', '.join(passed)})")
    print("NOTE: static source validation only; not clean-runtime execution evidence.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
