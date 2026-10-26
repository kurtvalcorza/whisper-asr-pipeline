"""Run-derived prose is a record of named runs, never an expectation (Kurt's ruling, 2026-10-26).

Fine-tune runs: Colab T4 of blobs 5f91ee7d (2026-10-10), f74abf5c and 472225cf (2026-10-25).
Inference runs: Colab T4 of blobs 75cc521b (2026-10-10), 41314708 and acd03796 (2026-10-25).
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "tutorials"
NB_FT = ROOT / "whisper_asr_finetune_colab.ipynb"
NB_INF = ROOT / "whisper_asr_colab.ipynb"


def _markdown(path):
    nb = json.loads(path.read_text(encoding="utf-8"))
    return "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "markdown")


def test_finetune_expectations_are_gone_and_the_confirming_run_is_recorded():
    md = _markdown(NB_FT)
    for stale in ("this version takes about 13 to 15 minutes", "Expect a high number",
                  "That is about five to seven points", "so expect a smaller change",
                  "two of WER is within what another seed can produce"):
        assert stale not in md, stale
    assert "0.3707, 0.3553 and 0.2943 on 100 clips" in md
    assert "insertions fell from 424 to 346, 333 and 239" in md
    assert "0.5077 → 0.2817 in the Colab T4 run of blob `472225cf`" in md


def test_inference_timing_promise_and_run_status_are_gone():
    md = _markdown(NB_INF)
    for stale in ("in about two minutes", "neither was re-run for this revision"):
        assert stale not in md, stale
    assert "124.5 s for blob `acd03796`" in md
