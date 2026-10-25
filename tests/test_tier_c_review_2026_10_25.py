"""Tier C review findings T111-1..T111-3 and T112-1..T112-5 (2026-10-25)."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ASR = ROOT / "tutorials" / "whisper_asr_colab.ipynb"
FT = ROOT / "tutorials" / "whisper_asr_finetune_colab.ipynb"


def _nb(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _markdown(path):
    md = "\n".join("".join(c["source"]) for c in _nb(path)["cells"] if c["cell_type"] == "markdown")
    return " ".join(md.split())


def _spec_items(path, first_ref):
    nb = _nb(path)
    dimer = nb["metadata"]["dimer"]
    assert dimer["notebook_spec"] == "2.5" and dimer["notebook_kind"] == "TUTORIAL"
    first_code = next(i for i, c in enumerate(nb["cells"]) if c["cell_type"] == "code")
    head = "\n".join("".join(c["source"]) for c in nb["cells"][:first_code])
    assert "Notice.** For research, training and evaluation only." in head
    md = _markdown(path)
    assert first_ref in md and "`DIMER_ISOLATED_ENV`" in md


def test_t111_1_no_stale_no_run_claims():
    md = _markdown(ASR)
    for stale in ("no run of this revision is recorded yet", "No run of this revision is recorded yet"):
        assert stale not in md, stale
    assert "`75cc521b`" in md


def test_t111_2_dat9_librispeech():
    md = _markdown(ASR)
    assert "**Overlap with the training data:**" in md and "LibriVox" in md


def test_t111_3_spec_items():
    _spec_items(ASR, "[1] A. Radford")


def test_t112_1_numbers_from_the_current_blob_run():
    md = _markdown(FT)
    for stale in ("0.4387", "0.4066", "0.578", "0.612", "0.517", "no run of this revision is recorded yet",
                  "has not been timed yet"):
        assert stale not in md, stale
    assert "0.4211" in md and "0.3707" in md and "`5f91ee7d`" in md


def test_t112_2_edit_breakdown_is_read_correctly():
    md = _markdown(FT)
    assert "count as substitutions even when the words were understood" not in md
    assert "424 of the 518 edits" in md and "insertions fell from 424 to 346" in md


def test_t112_3_dat9_minds14():
    md = _markdown(FT)
    assert "**Overlap with the training data:**" in md and "MINDS-14" in md


def test_t112_4_decoding_is_greedy():
    md = _markdown(FT)
    assert "5-beam" not in md and "beam search actually writes" not in md and "greedy decoding" in md


def test_t112_5_spec_items():
    _spec_items(FT, "[1] A. Radford")
