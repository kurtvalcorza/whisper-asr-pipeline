"""Prose numbers stay true across every recorded hosted run (Kurt's ruling, 2026-10-25).

Fine-tune runs: Colab T4 of blobs 5f91ee7d (2026-10-10) and f74abf5c (2026-10-25).
Inference runs: Colab T4 of blobs 75cc521b (2026-10-10) and 41314708 (2026-10-25).
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB_ASR = ROOT / "tutorials" / "whisper_asr_colab.ipynb"
NB_FT = ROOT / "tutorials" / "whisper_asr_finetune_colab.ipynb"


def _markdown(path):
    nb = json.loads(path.read_text(encoding="utf-8"))
    return "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "markdown")


def test_finetune_claims_hold_for_both_runs():
    md = _markdown(NB_FT)
    for stale in ("loss can rise in epoch 2 (0.4814 → 0.5016 there)", "That is about five points",
                  "so loss alone would favour one epoch", "it was not re-run for this revision"):
        assert stale not in md, stale
    assert "rose in epoch 2 in the first (0.4814 → 0.5016) and fell in the second (0.5267 → 0.5084)" in md
    assert "0.3707 and to 0.3553" in md and "861.8 s and 766.1 s" in md


def test_inference_timing_is_stated_with_its_runs():
    md = _markdown(NB_ASR)
    assert "this version has not been timed yet" not in md
    assert "120.4 s for blob `75cc521b` on 2026-10-10, 117.3 s for blob `41314708` on 2026-10-25" in md
