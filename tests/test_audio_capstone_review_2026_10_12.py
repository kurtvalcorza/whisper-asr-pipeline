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


def _manifest(folder, rate=16000, channels=1, seconds=5.0, path="clip_01.wav", garbage=False):
    folder.mkdir(parents=True, exist_ok=True)
    t = np.linspace(0, seconds, int(rate * seconds), endpoint=False)
    wave = (0.1 * np.sin(2 * np.pi * 220 * t)).astype("float32")
    sf.write(folder / "clip_01.wav", np.stack([wave, wave], 1) if channels == 2 else wave, rate)
    if garbage:
        (folder / "clip_01.wav").write_bytes(b"this is not audio" * 100)
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
    # Sample rate and channels are converted since 2026-10-13; these inputs are still refused.
    "too_long_48k_stereo": ({"rate": 48000, "channels": 2, "seconds": 30.0}, "30.0 s long"),
    "unreadable": ({"garbage": True}, "could not be decoded"),
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
    with pytest.raises(InputRefused, match="2-25 seconds is required"):
        validate_input(_manifest(tmp_path / "long", seconds=30.0))
    assert issubclass(InputRefused, ValueError)
    with pytest.raises(InputRefused, match="manifest not found"):
        validate_input(tmp_path / "absent" / "manifest.json")


def _notebook_command():
    install = builder.INSTALL
    return install[install.index("def command(") : install.index("\ncommand([UV")]


def test_byod_cell_reports_refusal_without_runtime_error_or_stale_archive(tmp_path):
    root = tmp_path / "root"
    builder.materialize(root)
    _manifest(root / "my_audio", seconds=30.0)
    cell = _code("RUN_BYOD = False").replace("RUN_BYOD = False", "RUN_BYOD = True", 1)
    namespace = {"ROOT": root, "PY": Path(sys.executable), "ENV": _env(), "BYOD": {"stale": True}}
    exec("import json, subprocess\nfrom pathlib import Path\n" + _notebook_command(), namespace)
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        exec(cell, namespace)  # must not raise RuntimeError
    text = out.getvalue()
    assert "Traceback" not in text and text.count("BYOD input refused: clip_01 (clip_01.wav)") == 1
    assert "30.0 s long" in text and "Nothing was built" in text
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
    assert "**16 kHz mono**" in text and "2–25 seconds" in text and "16 MB" in text
    assert "scipy.signal.resample_poly" in text and "replays the conversion" in text
    assert "Files** pane" in text and "my_audio/" in text and "If an input is refused" in text


def test_wording_ceiling_reading_is_conditional_and_labelled():
    # Wording test: the interpretation is generated or labelled with its revision (CX2, CX4).
    markdown = "\n".join(_cells("markdown"))
    assert "query version `ai-draft-v2`" in markdown and "never from the sentence itself" in markdown
    assert "draft pending native-speaker review" in markdown and "41.9%" not in markdown
    assert "previous query version** `ai-draft-v1`" in markdown
    assert "this bootstrap interval is degenerate and does not establish equivalence" in markdown
    assert "cannot distinguish the conditions" not in markdown and "verified in this run" not in markdown
    assert "revision `e4b3fed` (notebook blob `f890d97d`" in markdown
    assert "starts at transcription" not in markdown
    evaluate = _code("stage('evaluate')")
    assert "additional_comparisons" in evaluate and "does not establish equivalence" in evaluate


def _trace_namespace(ranked_runs, queries):
    shown = []
    return shown, {
        "ROOT": Path("."), "json": json, "ranked_runs": ranked_runs, "record": lambda name: ranked_runs,
        "corpus_rows": [{"doc_id": d, "reference": d, "path": d} for d in DOCS],
        "automatic": {d: d for d in DOCS}, "display": shown.append,
        "Markdown": lambda text: text, "Audio": lambda **kwargs: "[audio]",
        "draft_queries": queries,
    }


DOCS = [f"doc-{i:02d}" for i in range(16)]
SYSTEMS = [f"{c}_{m}" for c in ("reference", "asr") for m in ("bm25", "dense", "rerank")]


def _runs(placements):
    """placements: {query_id: {system: ranked doc IDs}}; unspecified systems put the anchor first."""
    runs = {}
    for query_id, (anchor, overrides) in placements.items():
        others = [d for d in DOCS if d != anchor]
        runs[query_id] = {s: [{"doc_id": d, "score": 0.0} for d in overrides.get(s, [anchor, *others])]
                          for s in SYSTEMS}
    return runs


def test_observations_separate_cutoff_misses_from_rank_demotions():
    # Behaviour test (CX7) on a synthetic ranking.
    others = DOCS[3:]
    placements = {
        # candidate miss
        "test-01": ("doc-00", {"asr_dense": [*others, "doc-00"], "asr_rerank": others[:10]}),
        # demotion from rank 1 to rank 2, still a top-five hit
        "test-02": ("doc-01", {"asr_dense": ["doc-01", *others],
                               "asr_rerank": [others[0], "doc-01", *others[1:9]]}),
        "test-03": ("doc-02", {"asr_dense": [*others[:6], "doc-02", *others[6:]],
                               "asr_rerank": [*others[:6], "doc-02", *others[6:9]]}),  # unchanged rank 7
    }
    queries = [{"query_id": q, "anchor_doc_id": a, "role": "test", "text": q}
               for q, (a, _) in placements.items()]
    shown, ns = _trace_namespace(_runs(placements), queries)
    trace = _code("def anchor_missed(")
    exec("def anchor_missed" + trace.split("def anchor_missed", 1)[1].split("\nSYSTEM_ORDER", 1)[0], ns)
    cell = _code("def stage_observations(")
    exec(cell[cell.index("def rank_of("):cell.index("def candidate_table(")], ns)
    exec(cell[cell.index("def stage_observations("):cell.index("\ncounts =")], ns)
    observed = {label.split(":")[0]: [q["query_id"] for q in cases]
                for label, cases in ns["stage_observations"](queries).items()}
    assert observed == {
        "Candidate miss": ["test-01"],
        "Top-five miss after ASR reranking": ["test-01", "test-03"],
        "Anchor rank demotion": ["test-02"],
    }


def test_trace_picks_an_evaluation_miss_under_any_system_before_development():
    # Behaviour test (CX7): a BM25-only evaluation miss is traced and named; development is not used.
    others = DOCS[2:]
    placements = {
        "test-01": ("doc-00", {"asr_bm25": [*others[:7], "doc-00", *others[7:]]}),
        "dev-01": ("doc-01", {"asr_rerank": others[:10]}),
    }
    queries = [{"query_id": q, "anchor_doc_id": a, "role": q.split("-")[0], "text": q}
               for q, (a, _) in placements.items()]
    shown, ns = _trace_namespace(_runs(placements), queries)
    cell = _code("def anchor_missed(")
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        exec(cell.split("\n", 2)[2], ns)  # skip the two record-loading lines
    assert ns["missed_by"] == "asr_bm25" and ns["trace_scope"] == "evaluation"
    assert ns["example"]["query_id"] == "test-01"
    assert "missed from the top five by `asr_bm25` (evaluation query)" in shown[0]


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
