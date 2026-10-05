"""Regression tests for the 2026-10-05 notebook reviews (WSP-M1..M3, WSP-m1..m2; WSF-M1..M3, WSF-m1..m3).

They need only CI's lightweight dependencies (numpy, soundfile, pytest): the generated notebooks' own learner cells
are executed with stand-ins for the model, `datasets` and `torchaudio`, and the carried helpers run on real audio
written with `soundfile`. No torch, no weights, no GPU; none of this is model evidence.
"""
# ruff: noqa: E501

from __future__ import annotations

import csv
import importlib.util
import io
import json
import os
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

import whisper_asr_pipeline.pipeline as pipeline_module
import whisper_asr_pipeline.tutorial_support as support_module
from whisper_asr_pipeline import (
    decode_audio,
    evaluation_report,
    with_empty_transcript_baseline,
    word_error_breakdown,
)

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"wsp_fix_{name}", ROOT / "tools" / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


build = _load("build_notebook")
INFERENCE = _load("notebook_template").TEMPLATE
FINETUNE = _load("notebook_template_finetune").TEMPLATE


def _notebook(template: dict) -> dict:
    return json.loads((ROOT / "tutorials" / template["notebook_name"]).read_text(encoding="utf-8"))


def _code(template: dict) -> list[str]:
    return [c["source"] for c in _notebook(template)["cells"] if c["cell_type"] == "code"]


def _markdown(template: dict) -> str:
    return "\n".join(c["source"] for c in _notebook(template)["cells"] if c["cell_type"] == "markdown")


def _cell(template: dict, needle: str) -> str:
    (source,) = [s for s in _code(template) if needle in s]
    return source


def _wav(seconds: float, rate: int = 16_000, freq: float = 220.0) -> bytes:
    t = np.arange(int(seconds * rate)) / rate
    buffer = io.BytesIO()
    sf.write(buffer, (0.1 * np.sin(2 * np.pi * freq * t)).astype(np.float32), rate, format="WAV")
    return buffer.getvalue()


class _StubPipe:
    device = "cpu"
    source = "stand-in"
    adapter = None

    def __init__(self, text: str = "Mr. Quilter is the apostle.") -> None:
        self.text = text
        self.calls: list[dict] = []

    def transcribe(self, audio, language=None, task="transcribe", return_timestamps=False, chunk_length_s=30):
        assert isinstance(audio, dict) and set(audio) == {"array", "sampling_rate"}, "the model must receive a waveform, never a file name"
        self.calls.append({"language": language, "task": task, "return_timestamps": return_timestamps})
        return {"text": self.text, "chunks": [{"timestamp": (0.0, 1.0), "text": self.text}] if return_timestamps else None, "task": task, "language": language, "device": "cpu", "source": "stand-in", "adapter": None}


def _namespace(monkeypatch, tmp_path: Path) -> dict:
    monkeypatch.chdir(tmp_path)
    datasets = types.ModuleType("datasets")
    datasets.Audio = lambda decode=True: None
    datasets.load_dataset = lambda *a, **k: pytest.fail("the default sample must not load in a BYOD test")
    monkeypatch.setitem(sys.modules, "datasets", datasets)
    torchaudio = types.ModuleType("torchaudio")
    monkeypatch.setitem(sys.modules, "torchaudio", torchaudio)
    ns = {k: v for k, v in vars(pipeline_module).items() if not k.startswith("__")}
    ns.update({k: v for k, v in vars(support_module).items() if not k.startswith("__")})
    ns.update(os=os, json=json, pipe=_StubPipe())
    return ns


def _colab_upload(monkeypatch, uploads: dict) -> None:
    files = types.ModuleType("google.colab.files")
    files.upload = lambda: uploads
    colab = types.ModuleType("google.colab")
    colab.files = files
    google = types.ModuleType("google")
    google.colab = colab
    for name, module in (("google", google), ("google.colab", colab), ("google.colab.files", files)):
        monkeypatch.setitem(sys.modules, name, module)


# ---- WSP-M1 / WSF-M1: isolated runtime, no restart; the record names what each run covered ---------------------------


@pytest.mark.parametrize("template", [INFERENCE, FINETUNE], ids=["WSP", "WSF"])
def test_wsp_M1_wsf_M1_no_kernel_install_no_restart_and_a_hash_lock(template: dict) -> None:
    assert "Restart the runtime" not in (ROOT / "tutorials" / template["notebook_name"]).read_text(encoding="utf-8")
    sources = _code(template)
    assert not any("[sys.executable, '-m', 'pip'" in s for s in sources)
    kernel = [s for s in sources if "# dimer: kernel cell" in s]
    assert len(kernel) == 2
    install = next(s for s in kernel if "LOCK_TEXT = r'''" in s)
    for needed in ('"--managed-python"', '"--require-hashes"', '"--only-binary"', "UV_SHA256", "LOCK_SHA256"):
        assert needed in install
    lock = (ROOT / template["lock"]).read_text(encoding="utf-8")
    build.check_lock(build._pins(ROOT, template), lock)
    assert "--only-binary :all:" in lock.splitlines()[1] and "x86_64-manylinux" in lock.splitlines()[1]


def test_wsp_M1_wsf_M1_registry_no_longer_claims_runs_of_these_blobs() -> None:
    registry = (ROOT / "tutorials" / "README.md").read_text(encoding="utf-8")
    assert "verified — clean-runtime `Run all` execution recorded" not in registry
    assert "REL1/REL8 satisfied" not in registry
    record = (ROOT / "docs" / "release-verification.md").read_text(encoding="utf-8")
    assert "Both the task-inference and fine-tuning standalone notebooks have now completed clean-room execution" not in record
    assert "no run of the current blobs is recorded" in record


# ---- WSP-M2: datasets is pinned and locked ------------------------------------------------------------------------


def test_wsp_M2_inference_pins_are_pyproject_dependencies_plus_datasets() -> None:
    import tomllib

    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    datasets_pin = next(p for p in project["optional-dependencies"]["tutorial"] if p.startswith("datasets=="))
    assert build._pins(ROOT, INFERENCE) == [*project["dependencies"], datasets_pin]
    assert "    'datasets==4.4.0',\n" in _cell(INFERENCE, "LOCK_TEXT = r'''")


# ---- WSP-M3 / WSF-M2: the guided layer ----------------------------------------------------------------------------

GUIDED = ["## How to use this notebook", "**Who this notebook is for.**", "## The task: Input → Model/System → Output", "## Roadmap", "<strong>Glossary</strong>", "**Predict before running:**", "**What to notice", "<summary>Check your reasoning</summary>", "## Troubleshooting", "## Conclusion (your notes)"]


@pytest.mark.parametrize("marker", [*GUIDED, "## 9. Activity: what does Whisper write for silence?"])
def test_wsp_M3_guided_layer_marker_is_present(marker: str) -> None:
    assert marker in _markdown(INFERENCE)


@pytest.mark.parametrize("marker", [*GUIDED, "## 11. Activity: change one thing — the number of epochs"])
def test_wsf_M2_guided_layer_marker_is_present(marker: str) -> None:
    assert marker in _markdown(FINETUNE)


@pytest.mark.parametrize("template", [INFERENCE, FINETUNE], ids=["WSP", "WSF"])
def test_wsp_M3_wsf_M2_infrastructure_cells_are_collapsed(template: dict) -> None:
    code = [c for c in _notebook(template)["cells"] if c["cell_type"] == "code"]
    first_learner = next(i for i, c in enumerate(code) if "USE_BYOD = False" in c["source"])
    assert first_learner >= 5
    for cell in code[:first_learner]:
        assert cell["metadata"].get("cellView") == "form", cell["source"][:60]


def test_wsp_M3_experiments_are_form_fields_and_silence_runs_without_upload(monkeypatch, tmp_path: Path) -> None:
    section4 = _cell(INFERENCE, "USE_BYOD = False")
    for field in ("TASK = 'transcribe'  # @param", "LANGUAGE = 'en'  # @param", "RETURN_TIMESTAMPS = False  # @param"):
        assert field in section4
    ns = _namespace(monkeypatch, tmp_path)
    ns.update(np=np, language="en", TASK="transcribe")
    ns["pipe"] = _StubPipe(text="Thank you.")
    exec(_cell(INFERENCE, "ACTIVITY_AUDIO = 'silence'"), ns)  # noqa: S102 - the notebook's own cell
    assert ns["activity"]["text"] == "Thank you." and ns["activity_waveform"].max() == 0.0
    exec(_cell(INFERENCE, "ACTIVITY_AUDIO = 'silence'").replace("ACTIVITY_AUDIO = 'silence'", "ACTIVITY_AUDIO = 'noise'"), ns)  # noqa: S102
    assert ns["activity_waveform"].std() > 0


# ---- WSP-m1: BYOD by path, as a waveform, named errors, exactly one upload ------------------------------------------


def test_wsp_m1_path_byod_reaches_the_model_as_a_waveform_with_seconds_and_rate(monkeypatch, tmp_path: Path) -> None:
    ns = _namespace(monkeypatch, tmp_path)
    (tmp_path / "mine.wav").write_bytes(_wav(1.5, rate=22_050))
    source = _cell(INFERENCE, "USE_BYOD = False").replace("USE_BYOD = False", "USE_BYOD = True").replace("BYOD_PATH = ''", "BYOD_PATH = 'mine.wav'").replace("REFERENCE_TEXT = ''", "REFERENCE_TEXT = 'mister quilter is the apostle'")
    exec(source, ns)  # noqa: S102
    assert ns["audio_input"]["sampling_rate"] == 22_050 and ns["sample_kind"] == "BYOD"
    exec(_cell(INFERENCE, "input_manifest = validate_inputs("), ns)  # noqa: S102
    entry = ns["input_manifest"]["inputs"][0]
    assert entry["kind"] == "waveform" and entry["seconds"] == 1.5 and entry["sampling_rate"] == 22_050
    exec(_cell(INFERENCE, "result = pipe.transcribe("), ns)  # noqa: S102
    exec(_cell(INFERENCE, "report = with_empty_transcript_baseline("), ns)  # noqa: S102
    assert ns["report"]["verdict"] == "sample-sanity" and ns["report"]["baselines"][0]["id"] == "empty_transcript"


def test_wsp_m1_corrupt_or_empty_audio_is_refused_by_name() -> None:
    with pytest.raises(ValueError, match=r"^broken\.wav: not an audio file soundfile can read"):
        decode_audio("broken.wav", b"not audio at all")
    with pytest.raises(ValueError, match=r"^empty\.wav: the file is empty"):
        decode_audio("empty.wav", b"")
    waveform, rate = decode_audio("stereo.wav", _stereo_wav())
    assert waveform.ndim == 1 and rate == 8_000


def _stereo_wav() -> bytes:
    buffer = io.BytesIO()
    sf.write(buffer, np.zeros((800, 2), dtype=np.float32), 8_000, format="WAV")
    return buffer.getvalue()


@pytest.mark.parametrize("uploads", [{}, {"a.wav": b"x", "b.wav": b"y"}])
def test_wsp_m1_cancelled_or_multi_file_upload_stops_with_the_rule(monkeypatch, tmp_path: Path, uploads: dict) -> None:
    ns = _namespace(monkeypatch, tmp_path)
    _colab_upload(monkeypatch, uploads)
    source = _cell(INFERENCE, "USE_BYOD = False").replace("USE_BYOD = False", "USE_BYOD = True")
    with pytest.raises(RuntimeError, match=rf"Upload exactly one audio file \(received {len(uploads)}\)"):
        exec(source, ns)  # noqa: S102


# ---- WSP-m2: the empty-transcript baseline --------------------------------------------------------------------------


def test_wsp_m2_report_lists_the_empty_transcript_baseline_only_with_a_reference() -> None:
    result = {"text": "Mr. Quilter is the apostle of the middle classes", "task": "transcribe", "language": "en"}
    reference = "MISTER QUILTER IS THE APOSTLE OF THE MIDDLE CLASSES"
    report = with_empty_transcript_baseline(evaluation_report(result, reference), reference)
    assert report["baselines"] == [{"id": "empty_transcript", "word_error_rate": 1.0, "meaning": "a system that outputs nothing; every reference word counts as a deletion"}]
    assert report["metrics"][0]["value"] == pytest.approx(1 / 9)
    assert with_empty_transcript_baseline(evaluation_report(result), None)["baselines"] == []
    assert "**empty-transcript baseline**" in _markdown(INFERENCE)


# ---- WSF-M3: a re-run from Section 4 rebuilds the baseline pipeline -------------------------------------------------


def test_wsf_M3_rerun_after_pipe_was_released_rebuilds_it(monkeypatch, tmp_path: Path) -> None:
    ns = _namespace(monkeypatch, tmp_path)
    built: list[dict] = []

    class _StubClass:
        @staticmethod
        def from_pretrained(**kwargs):
            built.append(kwargs)
            return _StubPipe(text="hello world")

    clips = [{"id": f"c{i}", "audio": np.zeros(16_000, dtype=np.float32), "text": "hello world"} for i in range(3)]
    ns.update(WhisperASRPipeline=_StubClass, WEIGHTS_DIR=tmp_path / "w", eval_clips=clips, TARGET_RATE=16_000, LANGUAGE="en", torch=None)
    source = _cell(FINETUNE, "baseline_transcripts = transcribe_all(pipe, eval_clips)")
    exec(source, ns)  # noqa: S102 - first run: Section 3's pipe exists
    assert "pipe" not in ns and built == []
    exec(source, ns)  # noqa: S102 - the documented re-run from Section 4: pipe was released
    assert built == [{"weights_dir": tmp_path / "w", "allow_download": False}]
    assert ns["baseline_wer"] == 0.0 and ns["baseline_edits"]["reference_words"] == 6


# ---- WSF-m1 / WSF-m2: long clips recorded, BYOD by path with a derived split and named errors -----------------------


def _byod_folder(tmp_path: Path, n: int, long_clip: bool = True, omit: str | None = None) -> Path:
    folder = tmp_path / "byod"
    folder.mkdir()
    rows = []
    for i in range(n):
        name = f"clip{i:02d}.wav"
        rows.append({"file": name, "text": f"call number {i}"})
        if name != omit:
            (folder / name).write_bytes(_wav(35.0 if long_clip and i == 0 else 1.0))
    with open(folder / "transcripts.csv", "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["file", "text"])
        writer.writeheader()
        writer.writerows(rows)
    return folder


def _run_finetune_section4(ns: dict, folder: Path) -> None:
    source = _cell(FINETUNE, "USE_BYOD = False").replace("USE_BYOD = False", "USE_BYOD = True").replace("BYOD_PATH = ''", f"BYOD_PATH = {str(folder)!r}")
    exec(source, ns)  # noqa: S102


def test_wsf_m1_m2_long_clip_is_dropped_and_recorded_and_the_byod_split_is_derived(monkeypatch, tmp_path: Path) -> None:
    ns = _namespace(monkeypatch, tmp_path)
    _run_finetune_section4(ns, _byod_folder(tmp_path, 13))
    assert ns["DROPPED_OVER_30S"] == ["clip00.wav"]
    assert (len(ns["train_clips"]), len(ns["eval_clips"])) == (10, 2)  # 12 usable clips, 80 / 20
    assert ns["EVAL_TRANSCRIPTS_IN_TRAIN"] == 0
    exec(_cell(FINETUNE, "input_manifest = {'schema'"), ns)  # noqa: S102 - Section 5
    finding = next(f for f in ns["input_manifest"]["findings"] if f["input"] == "clips-over-30s")
    assert finding["ids"] == ["clip00.wav"] and finding["count"] == 1
    assert "nothing is dropped silently" in _markdown(FINETUNE)


def test_wsf_m2_missing_audio_and_too_few_clips_are_refused_by_name(monkeypatch, tmp_path: Path) -> None:
    ns = _namespace(monkeypatch, tmp_path)
    with pytest.raises(ValueError, match=r"names 1 audio file\(s\) that were not supplied: \['clip03\.wav'\]"):
        _run_finetune_section4(ns, _byod_folder(tmp_path, 12, omit="clip03.wav"))
    other = tmp_path / "small"
    other.mkdir()
    ns = _namespace(monkeypatch, other)
    with pytest.raises(ValueError, match=r"BYOD needs at least 10 usable clips"):
        _run_finetune_section4(ns, _byod_folder(other, 6, long_clip=False))


# ---- WSF-m3: what the WER gain is made of; current timing ------------------------------------------------------------


def test_wsf_m3_edit_breakdown_adds_up_to_the_corpus_wer_and_is_reported() -> None:
    references = ["the cat sat on the mat", "one hundred dollars", "I'd like to pay"]
    hypotheses = ["the cat sat mat today", "100 dollars", "I like to pay my bill"]
    edits = word_error_breakdown(references, hypotheses)
    total = edits["substitutions"] + edits["insertions"] + edits["deletions"]
    assert total / edits["reference_words"] == pytest.approx(pipeline_module.corpus_word_error_rate(references, hypotheses))
    code = "\n".join(_code(FINETUNE))
    assert "adapted_edits = word_error_breakdown(eval_references, adapted_transcripts)" in code
    assert "'eval_transcripts_also_in_train': EVAL_TRANSCRIPTS_IN_TRAIN" in code
    md = _markdown(FINETUNE)
    assert "1002 s end to end on a Kaggle T4" in md and "**Read the gain carefully.**" in md
    assert "about 27 minutes" not in md
