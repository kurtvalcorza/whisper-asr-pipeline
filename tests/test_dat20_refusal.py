"""DAT19/DAT20: a refused BYOD input shows one `Refused: <file>: <reason>` line and no traceback.

Two halves, both read from the generated notebook:
- the router (Section 1, template `dat20-router-refusal/1` from ml-worker
  `integrations/dimer/fleet-specs/notebook-router/`) relays an exception carrying `dimer_refusal` as one
  line, while every other exception keeps its traceback;
- the BYOD cells turn every user-input problem into `BYODRefusal` and remove the earlier run's results and
  `outputs/` files before they stop.
"""

# ruff: noqa: E501  -- refusal messages and parametrized cases read best on one line
from __future__ import annotations

import csv
import io
import json
import os
import signal
import subprocess
import sys
import types
import zipfile
from multiprocessing.connection import Connection
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

import whisper_asr_pipeline.pipeline as pipeline_module
import whisper_asr_pipeline.tutorial_support as support_module

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "tutorials/whisper_asr_colab.ipynb"
BYOD_CELL = "whisper_asr-18"
CARRIED = (
    "class BYODRefusal(ValueError):\n"
    '    """A refused BYOD input: ``BYODRefusal(file_name, reason_and_fix)``."""\n\n'
    "    dimer_refusal = True\n\n"
    "    def __init__(self, file_name, reason):\n"
    '        super().__init__(f"{file_name}: {reason}")\n'
    "        self.file_name = file_name\n"
    "        self.reason = reason\n\n"
    "    def _render_traceback_(self):\n"
    '        return [f"Refused: {self}"]\n'
)


def _code_cells(path: Path = NOTEBOOK) -> list[dict]:
    return [c for c in json.loads(path.read_text(encoding="utf-8"))["cells"] if c["cell_type"] == "code"]


def _source(cell: dict) -> str:
    return "".join(cell["source"])


def _cell(cell_id: str, path: Path = NOTEBOOK) -> str:
    return next(_source(c) for c in _code_cells(path) if c.get("id") == cell_id)


# --- router, kernel side (portable) -------------------------------------------------------------------------------


def _router_namespace(start: str) -> dict:
    source = next(_source(c) for c in _code_cells() if '_WORKER_SOURCE = r"""' in _source(c))
    chunk = source[source.index(start) : source.index("def _is_user_cell():")]
    namespace = {"os": os, "sys": sys, "signal": signal, "subprocess": subprocess, "Connection": Connection, "__name__": "router"}
    exec(compile(chunk, "<router>", "exec"), namespace)
    return namespace


class _Pipe:
    def __init__(self, messages=()):
        self.messages, self.sent = list(messages), []

    def recv(self):
        return self.messages.pop(0)

    def send(self, message):
        self.sent.append(message)


def _kernel_runtime(messages):
    namespace = _router_namespace("class IsolatedCellError(RuntimeError):")
    runtime = object.__new__(namespace["IsolatedRuntime"])
    runtime._send, runtime._recv, runtime._display = _Pipe(), _Pipe(messages), lambda *a, **k: None
    return namespace, runtime


def test_router_relays_a_refusal_as_one_line_without_a_traceback(capsys):
    """Acceptance: fails on main, whose router has no refusal channel."""
    namespace, runtime = _kernel_runtime([("refusal", "hostname: not readable")])
    with pytest.raises(namespace["IsolatedCellRefusal"]) as caught:
        runtime.run("x = 1\n")
    assert caught.value._render_traceback_() == ["Refused: hostname: not readable"]
    assert caught.value.__cause__ is None and caught.value.__suppress_context__
    assert capsys.readouterr().err == ""
    assert runtime._send.sent == [("run", "x = 1\n")]


def test_router_keeps_the_traceback_of_a_genuine_error(capsys):
    namespace, runtime = _kernel_runtime([("error", "Traceback (most recent call last):\nRuntimeError: bug\n", "RuntimeError: bug")])
    with pytest.raises(namespace["IsolatedCellError"], match="RuntimeError: bug"):
        runtime.run("x = 1\n")
    assert "Traceback (most recent call last)" in capsys.readouterr().err


def test_router_normal_cell_output_is_unchanged(capsys):
    _, runtime = _kernel_runtime([("stream", "stdout", "hello\n"), ("done",)])
    assert runtime.run("print('hello')\n") is None
    assert capsys.readouterr() == ("hello\n", "")


def test_router_template_is_applied():
    source = next(_source(c) for c in _code_cells() if '_WORKER_SOURCE = r"""' in _source(c))
    assert source.index('elif kind == "refusal":') < source.index('elif kind == "error":')
    assert 'if getattr(exc, "dimer_refusal", False):' in source
    header = "Its router refusal channel (DAT20) is applied from ml-worker integrations/dimer/fleet-specs/notebook-router/."
    assert header in (ROOT / "tools" / "build_notebook.py").read_text(encoding="utf-8")


@pytest.mark.skipif(os.name == "nt", reason="the isolated-cell router is POSIX-only (pass_fds, start_new_session)")
def test_router_worker_end_to_end(capsys):
    namespace = _router_namespace('_WORKER_SOURCE = r"""')
    runtime = namespace["IsolatedRuntime"](sys.executable, display=lambda *a, **k: None)
    try:
        runtime.run("class R(ValueError):\n    dimer_refusal = True\nkept = 41\n")
        with pytest.raises(namespace["IsolatedCellRefusal"]) as caught:
            runtime.run("raise R('hostname: not readable')\n")
        assert caught.value._render_traceback_() == ["Refused: hostname: not readable"]
        assert "Traceback" not in capsys.readouterr().err
        runtime.run("print(kept + 1)\n")  # the worker namespace survives a refusal
        assert capsys.readouterr().out == "42\n"
        with pytest.raises(namespace["IsolatedCellError"], match="RuntimeError: bug"):
            runtime.run("raise RuntimeError('bug')\n")
        assert "Traceback (most recent call last)" in capsys.readouterr().err
    finally:
        runtime.close()


def test_byod_refusal_is_carried_verbatim():
    assert CARRIED in _cell(BYOD_CELL)


def _refusal_line(exc: BaseException) -> str:
    assert getattr(exc, "dimer_refusal", False), f"not a refusal: {exc!r}"
    (line,) = exc._render_traceback_()
    assert "\n" not in line and "Traceback" not in line
    return line


FINETUNE_NOTEBOOK = ROOT / "tutorials" / "whisper_asr_finetune_colab.ipynb"
FINETUNE_CELL = "whisper_asr_finetune-18"


def test_finetune_notebook_carries_the_router_channel_and_the_refusal_class():
    source = next(_source(c) for c in _code_cells(FINETUNE_NOTEBOOK) if '_WORKER_SOURCE = r"""' in _source(c))
    assert source.index('elif kind == "refusal":') < source.index('elif kind == "error":')
    assert CARRIED in _cell(FINETUNE_CELL, FINETUNE_NOTEBOOK)


# --- Section 4 BYOD cells, executed with stand-ins for `datasets` and `torchaudio` -------------------------------------


def _wav(seconds: float = 1.0, rate: int = 16_000) -> bytes:
    t = np.arange(int(seconds * rate)) / rate
    buffer = io.BytesIO()
    sf.write(buffer, (0.1 * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32), rate, format="WAV")
    return buffer.getvalue()


def _namespace(monkeypatch, tmp_path: Path, load_dataset=None) -> dict:
    monkeypatch.chdir(tmp_path)
    datasets = types.ModuleType("datasets")
    datasets.Audio = lambda decode=True: None
    datasets.load_dataset = load_dataset or (lambda *a, **k: pytest.fail("the default sample must not load in a BYOD test"))
    monkeypatch.setitem(sys.modules, "datasets", datasets)
    monkeypatch.setitem(sys.modules, "torchaudio", types.ModuleType("torchaudio"))
    ns = {k: v for k, v in vars(pipeline_module).items() if not k.startswith("__")}
    ns.update({k: v for k, v in vars(support_module).items() if not k.startswith("__")})
    ns.update(os=os, json=json)
    return ns


def _colab(monkeypatch, uploads):
    files = types.ModuleType("google.colab.files")
    files.upload = lambda: uploads
    colab = types.ModuleType("google.colab")
    colab.files = files
    google = types.ModuleType("google")
    google.colab = colab
    for name, module in (("google", google), ("google.colab", colab), ("google.colab.files", files)):
        monkeypatch.setitem(sys.modules, name, module)


def _no_colab(monkeypatch):
    monkeypatch.setitem(sys.modules, "google.colab", None)  # importing google.colab raises ImportError


def _source_with(cell: str, notebook: Path, use_byod: bool, path: str) -> str:
    source = _cell(cell, notebook).replace("USE_BYOD = False", f"USE_BYOD = {use_byod}", 1)
    return source.replace("BYOD_PATH = ''", f"BYOD_PATH = {path!r}", 1)


def _earlier_run(ns: dict, outputs: list[str]) -> None:
    for name in ("result", "report", "payload", "input_manifest"):
        ns[name] = "from an earlier run"
    Path("outputs").mkdir(exist_ok=True)
    for name in outputs:
        Path(name).write_text("{}", encoding="utf-8")


def _refuse(ns: dict, source: str) -> str:
    with pytest.raises(ValueError) as caught:
        exec(compile(source, "<section 4>", "exec"), ns)  # noqa: S102
    return _refusal_line(caught.value)


INFERENCE_OUTPUTS = ["outputs/whisper_asr_result.json", "outputs/whisper_asr_transcript.txt", "outputs/whisper_asr_evaluation_report.json", "outputs/whisper_asr_input_manifest.json"]


def test_inference_scripted_refusal_input_is_one_line(monkeypatch, tmp_path):
    """The hosted refusal run sets BYOD_PATH='/etc/hostname': a text file that soundfile cannot read."""
    ns = _namespace(monkeypatch, tmp_path)
    hostname = tmp_path / "hostname"
    hostname.write_text("colab-vm\n", encoding="utf-8")
    _earlier_run(ns, INFERENCE_OUTPUTS)
    line = _refuse(ns, _source_with(BYOD_CELL, NOTEBOOK, True, str(hostname)))
    assert line.startswith("Refused: hostname: not an audio file soundfile can read")
    assert "hostname: hostname:" not in line and "again" in line
    assert not any(name in ns for name in ("result", "report", "payload", "input_manifest"))
    assert not any(Path(name).exists() for name in INFERENCE_OUTPUTS)


@pytest.mark.skipif(not Path("/etc/hostname").is_file(), reason="needs /etc/hostname (POSIX, as on Colab)")
def test_inference_scripted_refusal_input_as_set_by_the_suite(monkeypatch, tmp_path):
    ns = _namespace(monkeypatch, tmp_path)
    line = _refuse(ns, _source_with(BYOD_CELL, NOTEBOOK, True, "/etc/hostname"))
    assert line.startswith("Refused: hostname: ")


@pytest.mark.parametrize(
    ("path", "uploads", "expected"),
    [
        ("folder", None, "Refused: folder: BYOD_PATH "),
        ("empty.wav", None, "Refused: empty.wav: the file is empty"),
        ("", {}, "Refused: (no file): the upload was cancelled or empty"),
        ("", {"a.wav": b"x", "b.wav": b"y"}, "Refused: a.wav, b.wav: upload exactly one audio file (received 2)"),
        ("", "no colab", "Refused: (no file): USE_BYOD = True but BYOD_PATH is empty"),
    ],
    ids=["directory", "empty-file", "cancelled-upload", "two-uploads", "no-upload-dialog"],
)
def test_inference_byod_refusals_are_one_line(monkeypatch, tmp_path, path, uploads, expected):
    ns = _namespace(monkeypatch, tmp_path)
    (tmp_path / "folder").mkdir()
    (tmp_path / "empty.wav").write_bytes(b"")
    if uploads == "no colab":
        _no_colab(monkeypatch)
    elif uploads is not None:
        _colab(monkeypatch, uploads)
    _earlier_run(ns, INFERENCE_OUTPUTS)
    line = _refuse(ns, _source_with(BYOD_CELL, NOTEBOOK, True, str(tmp_path / path) if path else ""))
    assert line.startswith(expected), line
    assert "result" not in ns and not Path(INFERENCE_OUTPUTS[0]).exists()


def test_inference_valid_byod_audio_still_loads(monkeypatch, tmp_path):
    ns = _namespace(monkeypatch, tmp_path)
    (tmp_path / "mine.wav").write_bytes(_wav())
    exec(compile(_source_with(BYOD_CELL, NOTEBOOK, True, str(tmp_path / "mine.wav")), "<section 4>", "exec"), ns)  # noqa: S102
    assert ns["sample_kind"] == "BYOD" and ns["sample_name"] == "mine.wav" and ns["sampling_rate"] == 16_000


def test_inference_genuine_error_inside_the_wrap_keeps_its_traceback(monkeypatch, tmp_path):
    ns = _namespace(monkeypatch, tmp_path)
    (tmp_path / "mine.wav").write_bytes(_wav())
    source = _source_with(BYOD_CELL, NOTEBOOK, True, str(tmp_path / "mine.wav")).replace(
        "        waveform, sampling_rate = decode_audio(sample_name, audio_bytes)\n", "        raise RuntimeError('bug')\n", 1)
    with pytest.raises(RuntimeError, match="bug") as caught:
        exec(compile(source, "<section 4>", "exec"), ns)  # noqa: S102
    assert not getattr(caught.value, "dimer_refusal", False)


def test_inference_public_sample_failure_is_not_a_refusal(monkeypatch, tmp_path):
    def offline(*args, **kwargs):
        raise OSError("the Hub is unreachable")

    ns = _namespace(monkeypatch, tmp_path, load_dataset=offline)
    with pytest.raises(OSError, match="unreachable") as caught:
        exec(compile(_cell(BYOD_CELL), "<section 4>", "exec"), ns)  # noqa: S102
    assert not getattr(caught.value, "dimer_refusal", False)


# --- row 112: the fine-tuning notebook's folder / zip / upload BYOD -----------------------------------------------------

FINETUNE_OUTPUTS = ["outputs/whisper_asr_finetune_result.json", "outputs/whisper_asr_finetune_evaluation_report.json", "outputs/whisper-asr-lora-adapter.zip"]


def _finetune_earlier_run(ns: dict) -> None:
    ns.update(PER_CLIP={"clips": "from an earlier run"}, adapted_wer=0.1, train_clips=["old"])
    Path("outputs/whisper-asr-lora-adapter").mkdir(parents=True, exist_ok=True)
    (Path("outputs/whisper-asr-lora-adapter") / "adapter.safetensors").write_bytes(b"x")
    for name in FINETUNE_OUTPUTS:
        Path(name).write_text("{}", encoding="utf-8")


def _assert_finetune_results_removed(ns: dict) -> None:
    assert not any(name in ns for name in ("PER_CLIP", "adapted_wer", "train_clips"))
    assert not any(Path(name).exists() for name in FINETUNE_OUTPUTS)
    assert not Path("outputs/whisper-asr-lora-adapter").exists()


def _clip_folder(root: Path, count: int = 12, header=("file", "text"), names=None) -> Path:
    folder = root / "mine"
    folder.mkdir()
    rows = []
    for i in range(count):
        name = f"clip{i:02d}.wav"
        (folder / name).write_bytes(_wav())
        rows.append({"file": name, "text": f"clip number {i}"})
    with (folder / "clips.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows([[r["file"], r["text"]] for r in rows])
    return folder


def test_finetune_scripted_refusal_input_is_one_line(monkeypatch, tmp_path):
    """The hosted refusal run sets BYOD_PATH='/etc': a folder that holds no .csv manifest."""
    ns = _namespace(monkeypatch, tmp_path)
    etc = tmp_path / "etc"
    etc.mkdir()
    (etc / "hostname").write_text("colab-vm\n", encoding="utf-8")
    (etc / "hosts").write_text("127.0.0.1 localhost\n", encoding="utf-8")
    _finetune_earlier_run(ns)
    line = _refuse(ns, _source_with(FINETUNE_CELL, FINETUNE_NOTEBOOK, True, str(etc)))
    assert line.startswith("Refused: etc: BYOD needs exactly one .csv with file,text columns; found none among 2 file(s)"), line
    _assert_finetune_results_removed(ns)


@pytest.mark.skipif(not Path("/etc").is_dir() or os.name == "nt", reason="needs /etc (POSIX, as on Colab)")
def test_finetune_scripted_refusal_input_as_set_by_the_suite(monkeypatch, tmp_path):
    ns = _namespace(monkeypatch, tmp_path)
    line = _refuse(ns, _source_with(FINETUNE_CELL, FINETUNE_NOTEBOOK, True, "/etc"))
    assert line.startswith("Refused: etc: ")


def _bad_zip(root: Path) -> Path:
    path = root / "clips.zip"
    path.write_bytes(b"not a zip")
    return path


def _zip_without_audio(root: Path) -> Path:
    path = root / "short.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("clips.csv", "file,text\nclip00.wav,hello\n")
    return path


@pytest.mark.parametrize(
    ("make", "expected"),
    [
        (lambda root: root / "notes.txt", "Refused: notes.txt: BYOD_PATH "),
        (_bad_zip, "Refused: clips.zip: BadZipFile: File is not a zip file"),
        (_zip_without_audio, "Refused: clips.csv: names 1 audio file(s) that were not supplied"),
        (lambda root: _clip_folder(root, header=("path", "sentence")), "Refused: clips.csv: needs a header with file,text columns"),
        (lambda root: _clip_folder(root, count=6), "Refused: mine: BYOD needs at least 10 usable clips"),
    ],
    ids=["not-a-folder-or-zip", "bad-zip", "missing-audio", "wrong-columns", "too-few-clips"],
)
def test_finetune_byod_refusals_are_one_line(monkeypatch, tmp_path, make, expected):
    ns = _namespace(monkeypatch, tmp_path)
    (tmp_path / "notes.txt").write_text("hello", encoding="utf-8")
    path = make(tmp_path)
    _finetune_earlier_run(ns)
    line = _refuse(ns, _source_with(FINETUNE_CELL, FINETUNE_NOTEBOOK, True, str(path)))
    assert line.startswith(expected), line
    _assert_finetune_results_removed(ns)


def test_finetune_undecodable_audio_is_refused_by_name(monkeypatch, tmp_path):
    ns = _namespace(monkeypatch, tmp_path)
    folder = _clip_folder(tmp_path)
    (folder / "clip03.wav").write_bytes(b"not audio")
    line = _refuse(ns, _source_with(FINETUNE_CELL, FINETUNE_NOTEBOOK, True, str(folder)))
    assert line.startswith("Refused: mine: clip03.wav: not an audio file soundfile can read"), line


@pytest.mark.parametrize(("uploads", "expected"), [({}, "Refused: (no file): the upload was cancelled or empty"), ("no colab", "Refused: (no file): USE_BYOD = True but BYOD_PATH is empty")])
def test_finetune_upload_refusals_are_one_line(monkeypatch, tmp_path, uploads, expected):
    ns = _namespace(monkeypatch, tmp_path)
    if uploads == "no colab":
        _no_colab(monkeypatch)
    else:
        _colab(monkeypatch, uploads)
    _finetune_earlier_run(ns)
    line = _refuse(ns, _source_with(FINETUNE_CELL, FINETUNE_NOTEBOOK, True, ""))
    assert line.startswith(expected), line
    _assert_finetune_results_removed(ns)


def test_finetune_valid_byod_folder_still_splits(monkeypatch, tmp_path):
    ns = _namespace(monkeypatch, tmp_path)
    exec(compile(_source_with(FINETUNE_CELL, FINETUNE_NOTEBOOK, True, str(_clip_folder(tmp_path))), "<section 4>", "exec"), ns)  # noqa: S102
    assert (len(ns["train_clips"]), len(ns["eval_clips"])) == (10, 2) and ns["sample_kind"] == "BYOD"


def test_finetune_genuine_error_inside_the_wrap_keeps_its_traceback(monkeypatch, tmp_path):
    ns = _namespace(monkeypatch, tmp_path)
    source = _source_with(FINETUNE_CELL, FINETUNE_NOTEBOOK, True, str(_clip_folder(tmp_path))).replace(
        "        sample_kind = 'BYOD'\n", "        raise KeyError('bug')\n", 1)
    with pytest.raises(KeyError, match="bug") as caught:
        exec(compile(source, "<section 4>", "exec"), ns)  # noqa: S102
    assert not getattr(caught.value, "dimer_refusal", False)


def test_finetune_public_sample_failure_is_not_a_refusal(monkeypatch, tmp_path):
    def offline(*args, **kwargs):
        raise OSError("the Hub is unreachable")

    ns = _namespace(monkeypatch, tmp_path, load_dataset=offline)
    with pytest.raises(OSError, match="unreachable") as caught:
        exec(compile(_cell(FINETUNE_CELL, FINETUNE_NOTEBOOK), "<section 4>", "exec"), ns)  # noqa: S102
    assert not getattr(caught.value, "dimer_refusal", False)


def test_finetune_per_clip_report_is_kept():
    """Row 112: the per-clip report of Section 7 is unchanged by the refusal work."""
    report = next(_source(c) for c in _code_cells(FINETUNE_NOTEBOOK) if "PER_CLIP = " in _source(c))
    assert "PER_CLIP" in report and "adapted_clip_records" in report
