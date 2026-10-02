"""Regression tests for the 2026-10-02 speech-recognition notebook review fixes (ASR-M1..M3, ASR-m1..m5).

The workshop notebook carries its runner (`workshop.py`) inside `CARRIED_FILES`; these tests extract it
from the committed notebook and exercise it with synthetic 16 kHz audio and a stand-in transcriber, so
they need only CI's lightweight dependencies (numpy, soundfile). No Whisper model, torch or GPU is used.
See docs/reviews/2026-10-02-notebook-review/.
"""

from __future__ import annotations

import ast
import importlib.util
import io
import json
import shutil
import sys
import types
import zipfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "tutorials" / "DIMER_Whisper_Speech_Recognition_Workshop.ipynb"


def _notebook() -> dict:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def _cells() -> dict[str, str]:
    return {cell["id"]: "".join(cell["source"]) for cell in _notebook()["cells"]}


def _carried() -> dict[str, str]:
    body = ast.parse(_cells()["code-03"]).body
    return ast.literal_eval(body[0].value)


@pytest.fixture
def workshop(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Import the notebook's carried workshop.py; stub scipy.signal when CI does not install SciPy."""
    for name, text in _carried().items():
        target = tmp_path / "carried" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    try:
        import scipy.signal  # noqa: F401
    except ModuleNotFoundError:
        scipy = types.ModuleType("scipy")
        signal = types.ModuleType("scipy.signal")

        def resample_poly(*_args, **_kwargs):
            raise AssertionError("tests use 16 kHz audio; no resampling expected")

        signal.resample_poly = resample_poly
        scipy.signal = signal
        monkeypatch.setitem(sys.modules, "scipy", scipy)
        monkeypatch.setitem(sys.modules, "scipy.signal", signal)
    monkeypatch.syspath_prepend(str(tmp_path / "carried"))
    spec = importlib.util.spec_from_file_location(
        "asr_workshop_carried", tmp_path / "carried" / "workshop.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _wav(seconds: float, hz: float) -> bytes:
    t = np.arange(int(seconds * 16000)) / 16000
    buffer = io.BytesIO()
    sf.write(
        buffer, (0.3 * np.sin(2 * np.pi * hz * t)).astype("float32"), 16000, format="WAV", subtype="FLOAT"
    )
    return buffer.getvalue()


def _zip(path: Path, members: dict[str, bytes]) -> Path:
    with zipfile.ZipFile(path, "w") as archive:
        for name, data in members.items():
            archive.writestr(name, data)
    return path


# ASR-M1 ---------------------------------------------------------------------------------------------


def test_reference_provenance_is_disclosed_where_scores_are_read() -> None:
    cells = _cells()
    assert "Google ASR" in cells["md-05"] and "agreement between two automatic transcripts" in cells["md-05"]
    assert "either one can be wrong" in cells["md-05"]
    assert "another recognizer's output" in cells["md-11"] and "reference's error" in cells["md-11"]
    assert "rather than human transcriptions" in cells["md-19"]
    assert "arXiv:2104.08524" in cells["md-19"]
    assert "another speech recognizer" in cells["md-00"]


def test_exported_summary_names_the_reference_source(workshop, tmp_path: Path) -> None:
    rows = [
        {
            "id": "a",
            "role": "evaluation",
            "condition": "clean",
            "seconds": 2.0,
            **workshop.score("pay my bill", "pay my bill"),
            "inference_seconds": 0.1,
            "truncated": False,
            "sequence_tokens": 4,
            "failure": None,
        },
    ]
    metrics = {"clean": workshop.aggregate(rows)}
    workshop.export_report(tmp_path / "canonical", rows, metrics, {"dataset": workshop.DATASET})
    canonical = (tmp_path / "canonical" / "summary.md").read_text(encoding="utf-8")
    assert "Google ASR" in canonical and "not human transcripts" in canonical
    workshop.export_report(tmp_path / "byod", rows, metrics, {"dataset": "BYOD"})
    byod = (tmp_path / "byod" / "summary.md").read_text(encoding="utf-8")
    assert "your supplied reference_text" in byod and "Google" not in byod


# ASR-M2 ---------------------------------------------------------------------------------------------


def test_metric_demo_worked_answer_matches_the_printed_alignment(workshop) -> None:
    result = workshop.score("pay my bill", "pay the bills now")
    assert (result["S"], result["D"], result["I"], result["wer"]) == (2, 0, 1, 1.0)
    worked = _cells()["md-09"]
    for step in result["alignment"]:
        if step["op"] == "S":
            assert f"{step['reference']}→{step['hypothesis']}" in worked
        elif step["op"] == "I":
            assert f"insertion of **{step['hypothesis']}**" in worked
    assert "S = 2, D = 0, I = 1" in worked
    assert "more than one minimal alignment" in worked.lower()
    assert "ties" in worked


# ASR-M3 ---------------------------------------------------------------------------------------------


def test_unlabelled_byod_records_keep_their_file_names(workshop, tmp_path: Path) -> None:
    archive = _zip(tmp_path / "clips.zip", {"zeta_call.wav": _wav(2, 300), "alpha_call.wav": _wav(3, 400)})
    records = workshop.load_byod(archive, tmp_path / "validation")
    assert [(r["id"], r["file"]) for r in records] == [("0", "alpha_call.wav"), ("1", "zeta_call.wav")]


def test_labelled_byod_records_keep_id_and_file(workshop, tmp_path: Path) -> None:
    csv_text = "id,file,reference_text\nq1,a.wav,pay my bill\nq2,b.wav,check my balance\n"
    archive = _zip(
        tmp_path / "labelled.zip",
        {"a.wav": _wav(2, 300), "b.wav": _wav(2, 500), "transcripts.csv": csv_text.encode()},
    )
    records = workshop.load_byod(archive, tmp_path / "validation")
    assert {(r["id"], r["file"]) for r in records} == {("q1", "a.wav"), ("q2", "b.wav")}


def _stage_stubs(workshop, monkeypatch: pytest.MonkeyPatch) -> None:
    def transcriber(_weights):
        def infer(audio):
            return {
                "text": "pay my bill",
                "inference_seconds": 0.01,
                "truncated": False,
                "sequence_tokens": 4,
            }

        return infer, {"device": "stand-in", "dtype": "float32"}

    monkeypatch.setattr(workshop, "transcriber", transcriber)
    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(max_memory_allocated=lambda: 0)
    monkeypatch.setitem(sys.modules, "torch", torch)
    resource = types.ModuleType("resource")
    resource.RUSAGE_SELF = 0
    resource.getrusage = lambda _who: types.SimpleNamespace(ru_maxrss=0)
    monkeypatch.setitem(sys.modules, "resource", resource)


def _stage_root(tmp_path: Path) -> Path:
    root = tmp_path / "root"
    root.mkdir()
    for name, text in _carried().items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text(text, encoding="utf-8")
    return root


def test_byod_stage_exports_the_file_column(
    workshop, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stage_stubs(workshop, monkeypatch)
    root = _stage_root(tmp_path)
    archive = _zip(tmp_path / "clips.zip", {"zeta_call.wav": _wav(2, 300), "alpha_call.wav": _wav(3, 400)})
    workshop.run_stage(root, "byod", archive)
    output = Path(json.loads((root / "byod_output.json").read_text())["path"])
    rows = json.loads((output / "transcripts.json").read_text(encoding="utf-8"))
    assert [(r["id"], r["file"]) for r in rows] == [("0", "alpha_call.wav"), ("1", "zeta_call.wav")]
    header = (output / "transcripts.csv").read_text(encoding="utf-8").splitlines()[0].split(",")
    assert "file" in header
    assert all("file" in r for r in json.loads((output / "input_manifest.json").read_text(encoding="utf-8")))


def test_canonical_stage_rows_gain_no_file_column(
    workshop, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stage_stubs(workshop, monkeypatch)
    root = _stage_root(tmp_path)
    prepared = []
    for index, hz in enumerate((300, 500)):
        audio = (0.3 * np.sin(2 * np.pi * hz * np.arange(32000) / 16000)).astype("float32")
        path = root / f"clip{index}.wav"
        sf.write(path, audio, 16000, subtype="FLOAT")
        prepared.append(
            {
                "id": f"clip{index}",
                "role": "evaluation",
                "reference": "pay my bill",
                "seconds": 2.0,
                "audio_path": str(path),
                "waveform_sha256": workshop.digest(audio.astype("<f4").tobytes()),
            }
        )
    (root / "prepared.json").write_text(json.dumps(prepared), encoding="utf-8")
    workshop.run_stage(root, "evaluate")
    output = Path(json.loads((root / "evaluate_output.json").read_text())["path"])
    rows = json.loads((output / "transcripts.json").read_text(encoding="utf-8"))
    assert len(rows) == 2 and all("file" not in r for r in rows)
    assert "Google ASR" in (output / "summary.md").read_text(encoding="utf-8")


# ASR-m1 ---------------------------------------------------------------------------------------------


def test_byod_refusal_names_the_failing_clip(workshop, tmp_path: Path) -> None:
    archive = _zip(
        tmp_path / "bad.zip", {"a.wav": _wav(2, 300), "b.wav": _wav(2, 500), "short.wav": _wav(0.4, 700)}
    )
    with pytest.raises(ValueError, match=r"short\.wav: Expected 1.30 seconds.*run the BYOD cell again"):
        workshop.load_byod(archive, tmp_path / "v")


def test_byod_refusal_names_undeclared_members(workshop, tmp_path: Path) -> None:
    archive = _zip(
        tmp_path / "extra.zip", {"a.wav": _wav(2, 300), "notes.txt": b"hi", "docs/readme.md": b"x"}
    )
    with pytest.raises(ValueError, match=r"undeclared files: docs/readme\.md, notes\.txt\. Keep only"):
        workshop.load_byod(archive, tmp_path / "v")


def test_byod_refusal_names_both_duplicates(workshop, tmp_path: Path) -> None:
    clip = _wav(2, 300)
    archive = _zip(tmp_path / "dup.zip", {"a.wav": clip, "copy.wav": clip})
    with pytest.raises(ValueError, match=r"Duplicate audio recordings: a\.wav and copy\.wav"):
        workshop.load_byod(archive, tmp_path / "v")


# ASR-m2 ---------------------------------------------------------------------------------------------


def test_snr_prose_says_whole_recording_and_construction_is_unchanged(workshop) -> None:
    prose = _cells()["md-11"]
    assert "whole recording" in prose and "pauses included" in prose
    assert "SNR compares speech power" not in prose
    speech = (0.3 * np.sin(2 * np.pi * 220 * np.arange(32000) / 16000)).astype("float32")
    speech[:16000] = 0
    gains = set()
    for snr in (5, 10, 20):
        _clean, _noisy, meta = workshop.noise_pair(speech, "probe", snr)
        assert abs(meta["actual_snr"] - snr) < 1e-3
        gains.add(meta["gain"])
    assert len(gains) == 1


# ASR-m3 ---------------------------------------------------------------------------------------------


def test_carrier_cell_metadata_matches_generated_from() -> None:
    notebook = _notebook()
    carrier = next(c for c in notebook["cells"] if c["id"] == "code-03")
    assert (
        carrier["metadata"]["dimer"]["sources"] == notebook["metadata"]["dimer"]["generated_from"]["sources"]
    )


def test_validator_rejects_a_stale_carrier_cell_digest(tmp_path: Path) -> None:
    for relative in (
        "tutorials",
        "docs/speech-recognition-workshop-spec.md",
        "src/whisper_asr_pipeline/pipeline.py",
        "weights/whisper-large-v3-turbo/dimer-base-manifest.json",
    ):
        source, target = ROOT / relative, tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.is_dir():
            shutil.copytree(source, target)
        else:
            shutil.copy2(source, target)
    spec = importlib.util.spec_from_file_location(
        "validate_release_assets_asr", ROOT / "tools" / "validate_release_assets.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.ROOT = tmp_path
    module.validate_workshop_notebooks()
    path = tmp_path / "tutorials" / NOTEBOOK.name
    notebook = json.loads(path.read_text(encoding="utf-8"))
    carrier = next(c for c in notebook["cells"] if c["id"] == "code-03")
    carrier["metadata"]["dimer"]["sources"]["workshop.py"] = "0" * 64
    path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    with pytest.raises(module.ValidationError, match="carrier cell metadata"):
        module.validate_workshop_notebooks()


# ASR-m4 ---------------------------------------------------------------------------------------------


def test_glossary_defines_the_recurring_terms() -> None:
    text = _cells()["md-19"]
    glossary = text.split("## Glossary", 1)[1].split("## AI Assistance Disclosure", 1)[0]
    for term in (
        "Reference transcript",
        "WER / CER",
        "Alignment",
        "Normalization",
        "Sample rate",
        "SNR",
        "Real-time factor",
        "Token ceiling",
        "Greedy decoding",
        "BYOD",
    ):
        assert f"**{term}" in glossary, term


# ASR-m5 ---------------------------------------------------------------------------------------------


def _show_results_namespace(root: Path, recorded: list) -> dict:
    display_module = types.ModuleType("IPython.display")
    display_module.Audio = lambda filename, normalize: ("audio", filename)
    display_module.Markdown = lambda text: ("markdown", text)
    display_module.display = recorded.append
    ipython = types.ModuleType("IPython")
    ipython.display = display_module
    sys.modules["IPython"], sys.modules["IPython.display"] = ipython, display_module
    tree = ast.parse(_cells()["code-04"])
    functions = ast.Module([n for n in tree.body if isinstance(n, ast.FunctionDef)], type_ignores=[])
    namespace = {"json": json, "Path": Path, "ROOT": root}
    exec(compile(functions, "code-04", "exec"), namespace)
    return namespace


def _write_stage(root: Path, workshop, stage: str, rows: list[dict]) -> None:
    output = root / f"{stage}-out"
    output.mkdir()
    (root / f"{stage}_output.json").write_text(json.dumps({"path": str(output)}))
    (output / "transcripts.json").write_text(json.dumps(rows))
    (output / "metrics.json").write_text(json.dumps({"clean": workshop.aggregate(rows)}))


def test_show_results_plays_the_selected_examples(
    workshop, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "IPython", sys.modules.get("IPython"))
    monkeypatch.setitem(sys.modules, "IPython.display", sys.modules.get("IPython.display"))
    root = tmp_path
    rows = []
    prepared = []
    for ident, hyp in (("good", "pay my bill"), ("bad", "pay the bills now"), ("zfirst", "pay my bill")):
        rows.append(
            {
                "id": ident,
                "role": "evaluation",
                "condition": "clean",
                "seconds": 2.0,
                **workshop.score("pay my bill", hyp),
                "truncated": False,
            }
        )
        prepared.append({"id": ident, "audio_path": str(root / f"{ident}.wav")})
    rows.insert(0, rows.pop())
    (root / "prepared.json").write_text(json.dumps(prepared))
    _write_stage(root, workshop, "evaluate", rows)
    recorded: list = []
    _show_results_namespace(root, recorded)["show_results"]("evaluate")
    played = [item[1] for item in recorded if isinstance(item, tuple) and item[0] == "audio"]
    assert str(root / "bad.wav") in played and str(root / "good.wav") in played

    byod_rows = [{**rows[1], "id": "0", "file": "mine.wav", "role": "byod"}]
    _write_stage(root, workshop, "byod", byod_rows)
    recorded.clear()
    _show_results_namespace(root, recorded)["show_results"]("byod")
    assert not [item for item in recorded if isinstance(item, tuple) and item[0] == "audio"]
