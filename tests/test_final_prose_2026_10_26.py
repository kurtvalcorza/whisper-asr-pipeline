"""Final prose pass (Kurt's ruling, 2026-10-26): the prose-rule items flagged on the confirming run are fixed.

Run-status sentences, timing promises and unlabelled expectations are removed or recast
as records of named runs.
Fails on PR #20 head e876b97.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "tutorials" / "whisper_asr_finetune_colab.ipynb"
NB2 = ROOT / "tutorials" / "whisper_asr_colab.ipynb"


def _markdown(path):
    nb = json.loads(path.read_text(encoding="utf-8"))
    return "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "markdown")


def test_final_prose_pass():
    md = _markdown(NB)
    for gone in (
        'budget about 13 to 15 minutes',
        'or the training cell will take hours.',
    ):
        assert gone not in md, gone
    for kept in (
        '861.8 s, 766.1 s and 825.9 s of session wall',
        'blob `472225cf`, 2026-10-25)); your run may take a different time.',
        'or the training cell will take hours (an estimate).',
    ):
        assert kept in md, kept


def test_final_prose_pass_nb2():
    md = _markdown(NB2)
    for gone in (
        'it has not been timed on CPU',
    ):
        assert gone not in md, gone
    for kept in (
        '(120.4 s for blob `75cc521b` on 2026-10-10, 117.3 s for blob `41314708` on 2026-10-25).',
    ):
        assert kept in md, kept
