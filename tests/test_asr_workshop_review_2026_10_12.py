"""Regression tests for the 2026-10-12 speech-recognition workshop review (ASR2-m1..m4, ASR2-S1, ASR2-S2).

The carried runner (`workshop.py`) is extracted from the committed notebook and run with synthetic 16 kHz
audio, so only CI's lightweight dependencies (numpy, soundfile, scipy) are needed. No Whisper model, torch
or GPU is used. The notebook's own `run_stage` (code-04) is exercised with a stand-in subprocess.
"""

from __future__ import annotations

import ast
import io
import json
import os
import re
import subprocess
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
    return ast.literal_eval(ast.parse(_cells()["code-03"]).body[0].value)


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


@pytest.fixture
def carried_root(tmp_path: Path) -> Path:
    root = tmp_path / "root"
    for name, text in _carried().items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text(text, encoding="utf-8")
    return root


def _run_workshop(root: Path, *args: str) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME")}
    env.update(CUDA_VISIBLE_DEVICES="-1", PYTHONUTF8="1")
    return subprocess.run(
        [sys.executable, str(root / "workshop.py"), "--root", str(root), *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=env,
        timeout=300,
    )


# ASR2-m1 --------------------------------------------------------------------------------------------


def test_refused_byod_zip_prints_one_line_and_exits_3(carried_root: Path, tmp_path: Path) -> None:
    archive = _zip(tmp_path / "refused.zip", {"good_clip.wav": _wav(2, 300), "short.wav": _wav(0.4, 500)})
    result = _run_workshop(carried_root, "--stage", "byod", "--byod", str(archive))
    output = result.stdout + result.stderr
    assert result.returncode == 3, output
    assert "Traceback" not in output
    lines = [line for line in output.splitlines() if line.strip()]
    assert len(lines) == 1 and lines[0].startswith("BYOD refused: short.wav: Expected 1")
    assert "run the BYOD cell again" in lines[0] and "canonical reports are kept" in lines[0]


def test_unreadable_zip_is_a_refusal_not_a_crash(carried_root: Path, tmp_path: Path) -> None:
    broken = tmp_path / "broken.zip"
    broken.write_bytes(b"not a zip archive")
    result = _run_workshop(carried_root, "--stage", "byod", "--byod", str(broken))
    assert result.returncode == 3 and "Traceback" not in result.stdout + result.stderr
    assert (
        "BYOD refused: Not a readable ZIP archive. Fix the input, then run the BYOD cell again."
        in result.stdout
    )


def test_genuine_failure_still_fails_loudly(carried_root: Path, tmp_path: Path) -> None:
    archive = _zip(tmp_path / "ok.zip", {"a.wav": _wav(2, 300)})
    (carried_root / "sample.json").write_text("{not json", encoding="utf-8")
    result = _run_workshop(carried_root, "--stage", "byod", "--byod", str(archive))
    assert result.returncode not in (0, 3)
    assert "Traceback" in result.stderr and "BYOD refused" not in result.stdout


def _load_byod_error(carried_root: Path, tmp_path: Path, members: dict[str, bytes]) -> str:
    sys.path.insert(0, str(carried_root))
    try:
        namespace: dict = {"__name__": "asr2_workshop"}
        exec(
            compile((carried_root / "workshop.py").read_text(encoding="utf-8"), "workshop.py", "exec"),
            namespace,
        )
        with pytest.raises(ValueError) as caught:
            namespace["load_byod"](_zip(tmp_path / "in.zip", members), tmp_path / "validation")
        return str(caught.value)
    finally:
        sys.path.remove(str(carried_root))


def test_declared_once_refusal_names_the_members(carried_root: Path, tmp_path: Path) -> None:
    csv_text = "id,file,reference_text\nq1,a.wav,pay my bill\nq2,ghost.wav,hello\n"
    message = _load_byod_error(
        carried_root,
        tmp_path,
        {"a.wav": _wav(2, 300), "b.wav": _wav(2, 400), "transcripts.csv": csv_text.encode()},
    )
    assert "audio not listed in transcripts.csv: b.wav" in message
    assert "listed but not in the ZIP: ghost.wav" in message and "run the BYOD cell again" in message


def test_clip_count_refusal_says_how_many(carried_root: Path, tmp_path: Path) -> None:
    members = {f"clip{i:02d}.wav": _wav(1.2, 200 + 10 * i) for i in range(21)}
    message = _load_byod_error(carried_root, tmp_path, members)
    assert "Expected 1–20 audio clips; found 21: clip00.wav, clip01.wav" in message and " ..." in message


def _notebook_run_stage(root: Path, returncode: int, log_text: str):
    tree = ast.parse(_cells()["code-04"])
    functions = ast.Module([n for n in tree.body if isinstance(n, ast.FunctionDef)], type_ignores=[])

    def fake_run(command, env, stdout, stderr):
        stdout.write(log_text)
        return types.SimpleNamespace(returncode=returncode)

    namespace = {
        "json": json,
        "Path": Path,
        "ROOT": root,
        "PYTHON": Path("python"),
        "ENV": {},
        "subprocess": types.SimpleNamespace(run=fake_run, STDOUT=subprocess.STDOUT),
    }
    exec(compile(functions, "code-04", "exec"), namespace)
    return namespace["run_stage"]


def test_notebook_run_stage_shows_a_refusal_without_raising(tmp_path: Path, capsys) -> None:
    run_stage = _notebook_run_stage(tmp_path, 3, "BYOD refused: short.wav: too short. Fix it.\n")
    assert run_stage("byod", "--byod", "x.zip") is False
    printed = capsys.readouterr().out
    assert printed.splitlines()[-1] == "BYOD refused: short.wav: too short. Fix it."


@pytest.mark.parametrize(
    ("stage", "code", "log"),
    [
        ("byod", 1, "Traceback (most recent call last):\nValueError: Integrity failure\n"),
        ("byod", 3, "no refusal line\n"),
        ("evaluate", 3, "BYOD refused: impossible here\n"),
    ],
)
def test_notebook_run_stage_still_raises_on_genuine_failures(
    tmp_path: Path, stage: str, code: int, log: str
) -> None:
    with pytest.raises(RuntimeError, match=f"{stage} failed"):
        _notebook_run_stage(tmp_path, code, log)(stage)


def test_notebook_run_stage_success_prints_the_log_tail_and_returns_true(tmp_path: Path, capsys) -> None:
    assert _notebook_run_stage(tmp_path, 0, "x" * 8000 + "\nDONE\n")("evaluate") is True
    assert capsys.readouterr().out.rstrip().endswith("DONE")


def test_byod_cell_has_no_bare_raise_and_skips_results_after_a_refusal() -> None:
    cell = _cells()["code-18"]
    assert "raise ValueError" not in cell
    assert "if selected and run_stage('byod'" in cell and "BYOD refused: upload exactly one" in cell


# ASR2-m2 --------------------------------------------------------------------------------------------


def test_notebook_declares_spec_2_5_kind_and_use_notice() -> None:
    notebook = _notebook()
    meta = notebook["metadata"]["dimer"]
    assert meta["notebook_spec"] == "2.5" and meta["notebook_kind"] == "WORKSHOP"
    first_code = next(i for i, c in enumerate(notebook["cells"]) if c["cell_type"] == "code")
    preamble = "".join("".join(c["source"]) for c in notebook["cells"][:first_code])
    assert "Not validated for production" in preamble and '"as is"' in preamble
    assert "`2.5`" in _cells()["md-00"] and "Gerz et al., 2021" not in json.dumps(notebook["cells"])


# ASR2-m3 --------------------------------------------------------------------------------------------


def test_opening_quotes_the_recorded_run_with_its_revision() -> None:
    opening = _cells()["md-00"]
    assert not re.search(
        r"fresh GPU run is still needed|no measured performance promise|awaiting fresh Colab", opening
    )
    assert "158 s" in opening and "T4" in opening and "`fc8acbb6`" in opening
    assert "needs its own hosted run" in opening


# ASR2-m4, ASR2-S1, ASR2-S2 --------------------------------------------------------------------------


def test_paired_counts_and_exact_match_rate_are_defined() -> None:
    cells = _cells()
    assert "Unchanged means equal per-clip WER, not identical text" in cells["md-13"]
    assert "subtracts its clean WER from its noisy WER" in cells["md-13"]
    assert "**Exact-match rate (`exact_match_rate`):**" in cells["md-19"]
    workshop = _carried()["workshop.py"]
    assert '"unchanged": sum(d == 0 for d in deltas)' in workshop
    assert 'r["normalized_reference"] == r["normalized_hypothesis"]' in workshop


def test_exploration_prints_the_canonical_10_db_rates_beside_the_chosen_setting(
    tmp_path: Path, capsys
) -> None:
    def metrics(wer: float, paired: tuple[int, int, int]) -> dict:
        row = {"wer": wer, "cer": wer / 2, "S": 1, "D": 0, "I": 2}
        return {
            "clean": {**row, "wer": 0.185},
            "noisy": row,
            "paired": dict(zip(("improved", "unchanged", "worsened"), paired, strict=True)),
        }

    for stage, values in (("activity", metrics(0.2081, (0, 12, 4))), ("explore", metrics(0.31, (1, 9, 6)))):
        out = tmp_path / stage
        out.mkdir()
        (out / "metrics.json").write_text(json.dumps(values))
        (tmp_path / f"{stage}_output.json").write_text(json.dumps({"path": str(out)}))
    source = _cells()["code-16"].replace("RUN_EXPLORATION = False", "RUN_EXPLORATION = True", 1)
    calls = []
    namespace = {
        "json": json,
        "ROOT": tmp_path,
        "run_stage": lambda *a: calls.append(a) or True,
        "show_results": lambda stage: None,
        "result_dir": lambda stage: Path(json.loads((tmp_path / f"{stage}_output.json").read_text())["path"]),
    }
    exec(compile(source, "code-16", "exec"), namespace)
    printed = capsys.readouterr().out.splitlines()
    assert calls == [("explore", "--snr", "5")]
    assert printed[1].startswith("clean") and printed[2].startswith("10 dB") and "0.2081" in printed[2]
    assert printed[2].endswith("0/12/4") and printed[3].startswith("5 dB") and printed[3].endswith("1/9/6")
