"""Regression tests for the 2026-10-12 capstone review fixes (FAS-M1, FAS-M2, FAS-m1 to FAS-m4).

No model runs: the BYOD refusal path stops before any model loads, and the notebook's own command()
helper and BYOD cell are executed against the carried source in a subprocess.
"""

import contextlib
import io
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import build_audio_capstone as builder  # noqa: E402
from audio_byod import InputRefused, validate_input  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


def _cells(kind):
    return ["".join(c["source"]) for c in builder.build()["cells"] if c["cell_type"] == kind]


def _code(fragment):
    matches = [text for text in _cells("code") if fragment in text]
    assert len(matches) == 1, fragment
    return matches[0]


def _manifest(folder, rate=16000, channels=1, seconds=5.0, path="clip_01.wav"):
    folder.mkdir(parents=True, exist_ok=True)
    t = np.linspace(0, seconds, int(rate * seconds), endpoint=False)
    wave = (0.1 * np.sin(2 * np.pi * 220 * t)).astype("float32")
    sf.write(folder / "clip_01.wav", np.stack([wave, wave], 1) if channels == 2 else wave, rate)
    value = {
        "rights_confirmed": True,
        "source_notes": "synthetic test tone",
        "records": [{"doc_id": "clip_01", "path": path}],
    }
    (folder / "manifest.json").write_text(json.dumps(value), encoding="utf-8")
    return folder / "manifest.json"


def _env():
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP")}
    return {**env, "CUDA_VISIBLE_DEVICES": "-1", "PYTHONIOENCODING": "utf-8"}


@pytest.fixture(scope="module")
def carried_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("carried")
    builder.materialize(root)
    return root


REFUSALS = {
    "stereo_44k1": ({"rate": 44100, "channels": 2}, "44100 Hz, 2 channel(s)"),
    "mono_48k": ({"rate": 48000}, "48000 Hz, 1 channel(s)"),
    "too_long": ({"seconds": 30.0}, "30.0 s long"),
    "missing_file": ({"path": "clip_02.wav"}, "file not found"),
}


@pytest.mark.parametrize("case", sorted(REFUSALS))
def test_refused_byod_input_is_one_actionable_line(case, carried_root, tmp_path):
    kwargs, problem = REFUSALS[case]
    manifest = _manifest(tmp_path / case, **kwargs)
    result = subprocess.run(
        [sys.executable, str(carried_root / "audio_byod.py"), "--root", str(carried_root)]
        + ["--manifest", str(manifest)],
        cwd=carried_root,
        env=_env(),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    lines = (result.stdout + result.stderr).strip().splitlines()
    assert result.returncode == 3
    assert len(lines) == 1 and "Traceback" not in lines[0], lines
    assert lines[0].startswith("BYOD input refused: clip_01 (") and problem in lines[0] and "Fix:" in lines[0]


def test_refusals_stay_value_errors_and_accepted_inputs_are_unchanged(tmp_path):
    _, records = validate_input(_manifest(tmp_path / "ok"))
    assert records[0]["sample_rate"] == 16000 and records[0]["channels"] == 1
    with pytest.raises(InputRefused, match="16 kHz mono is required"):
        validate_input(_manifest(tmp_path / "stereo", rate=44100, channels=2))
    assert issubclass(InputRefused, ValueError)
    with pytest.raises(InputRefused, match="manifest not found"):
        validate_input(tmp_path / "absent" / "manifest.json")


def _notebook_command():
    install = builder.INSTALL
    return install[install.index("def command(") : install.index("\ncommand([UV")]


def test_byod_cell_reports_refusal_without_runtime_error_or_stale_archive(carried_root, tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    for path in carried_root.iterdir():
        (root / path.name).write_bytes(path.read_bytes())
    _manifest(root / "my_audio", rate=44100, channels=2)
    cell = _code("RUN_BYOD = False").replace("RUN_BYOD = False", "RUN_BYOD = True", 1)
    namespace = {"ROOT": root, "PY": Path(sys.executable), "ENV": _env(), "BYOD": {"stale": True}}
    exec("import json, subprocess\nfrom pathlib import Path\n" + _notebook_command(), namespace)
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        exec(cell, namespace)  # must not raise RuntimeError
    text = out.getvalue()
    assert "Traceback" not in text and text.count("BYOD input refused: clip_01 (clip_01.wav)") == 1
    assert "16 kHz mono is required" in text and "Nothing was built" in text
    assert "BYOD" not in namespace, "a refused build must not leave an earlier archive searchable"


def test_command_still_raises_for_genuine_failures(tmp_path):
    namespace = {"ROOT": tmp_path, "ENV": _env()}
    exec("import subprocess\n" + _notebook_command(), namespace)
    with pytest.raises(RuntimeError, match="Command failed"):
        namespace["command"]([sys.executable, "-c", "raise SystemExit(1)"], "fail.log", allowed=(3,))
    refused = [sys.executable, "-c", "raise SystemExit(3)"]
    assert namespace["command"](refused, "refused.log", allowed=(3,)) == 3


def test_byod_markdown_states_the_input_contract_and_upload_route():
    text = next(t for t in _cells("markdown") if t.startswith("### 11. Optional bring-your-own"))
    assert "16 kHz, mono" in text and "2–25 seconds" in text and "4 MB" in text
    assert "does **not** resample or downmix" in text and "-ac 1 -ar 16000" in text
    assert "Files** pane" in text and "my_audio/" in text and "If an input is refused" in text


def test_ceiling_interpretation_query_drafting_and_development_tracing():
    markdown = "\n".join(_cells("markdown"))
    assert "paraphrase of its own target (anchor) recording's reference sentence" in markdown
    assert "41.9%" in markdown and "56 of the 60" in markdown
    assert "not recomputed by this notebook" in markdown
    assert "Reading a ceiling result" in markdown and "cannot distinguish the conditions" in markdown
    assert "Conclude within the design" in markdown
    trace = _code("def anchor_missed(")
    assert "dev_queries" in trace and "diagnosis only" in trace
    categories = _code("def failure_categories(")
    assert "('development (diagnosis only)', dev_queries)" in categories


def test_spec_declaration_and_write_up_layer():
    notebook = builder.build()
    meta = notebook["metadata"]["dimer"]
    assert meta["notebook_spec"] == "2.2" and meta["notebook_kind"] == "CAPSTONE"
    assert meta["notebook_profile"] == "E2E" and meta["notebook_mode"] == "GUIDED"
    assert not {"spec_version", "profile", "mode"} & set(meta)
    opening = _cells("markdown")[0]
    assert "#### Problem statement" in opening and "> **Use notice:**" in opening
    markdown = "\n".join(_cells("markdown"))
    assert "### References" in markdown and "arXiv:2506.05176" in markdown
    trouble = next(t for t in _cells("markdown") if t.startswith("### Troubleshooting"))
    assert "T4" in trouble and "disk" in trouble and "BYOD" in trouble


def test_status_text_names_the_real_open_gates():
    opening = _cells("markdown")[0]
    assert "a fresh hosted run are required" not in opening and "unverified targets" not in opening
    assert "615 s" in opening and "not re-run for this revision" in opening
    readme = (REPO / "README.md").read_text(encoding="utf-8")
    assert "Hosted execution is also pending" not in readme


def test_summary_records_are_not_dumped_whole():
    code = "\n".join(_cells("code"))
    assert "show_json('activity_summary.json')" not in code and "show_json('run_summary.json')" not in code
    assert "RESULTS / 'activity_summary.json'" in code and "RESULTS / 'run_summary.json'" in code
