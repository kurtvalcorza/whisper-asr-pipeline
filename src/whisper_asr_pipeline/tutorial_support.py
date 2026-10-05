"""Tutorial helpers carried by the two standalone notebooks (2026-10-05 review fixes).

Kept apart from ``pipeline.py`` so the workshop notebook, which carries ``pipeline.py`` byte for byte, is
unchanged:
``decode_audio`` (BYOD decoding that names the file it refuses, WSP-m1 / WSF-m2), ``normalise_transcript`` and
``word_error_breakdown`` (what a WER is made of, WSF-m3) and ``with_empty_transcript_baseline`` (the trivial
reference point of a single-utterance WER, WSP-m2).
"""

from __future__ import annotations

import io
from collections.abc import Mapping, Sequence
from typing import Any

from .pipeline import _tokens, word_error_rate


def normalise_transcript(text: str) -> str:
    """The text exactly as the WER helpers compare it: case-folded, punctuation removed, single spaces."""
    return " ".join(_tokens(text))


def word_error_breakdown(references: Sequence[str], hypotheses: Sequence[str]) -> dict[str, int]:
    """Corpus totals of substitutions, insertions and deletions after the same normalisation as
    ``word_error_count`` (one minimum-edit alignment per pair; ties prefer substitution). Their sum over
    ``reference_words`` is the corpus WER, so a change in WER can be read as a change in edit type."""
    if len(references) != len(hypotheses):
        raise ValueError(f"{len(references)} references but {len(hypotheses)} hypotheses")
    totals = {"substitutions": 0, "insertions": 0, "deletions": 0, "reference_words": 0}
    for reference, hypothesis in zip(references, hypotheses, strict=True):
        ref, hyp = _tokens(reference), _tokens(hypothesis)
        cost = [[0] * (len(hyp) + 1) for _ in range(len(ref) + 1)]
        for i in range(len(ref) + 1):
            cost[i][0] = i
        for j in range(len(hyp) + 1):
            cost[0][j] = j
        for i in range(1, len(ref) + 1):
            for j in range(1, len(hyp) + 1):
                cost[i][j] = min(
                    cost[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]), cost[i - 1][j] + 1, cost[i][j - 1] + 1
                )
        i, j = len(ref), len(hyp)
        while i or j:
            if i and j and cost[i][j] == cost[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]):
                totals["substitutions"] += ref[i - 1] != hyp[j - 1]
                i, j = i - 1, j - 1
            elif i and cost[i][j] == cost[i - 1][j] + 1:
                totals["deletions"] += 1
                i -= 1
            else:
                totals["insertions"] += 1
                j -= 1
        totals["reference_words"] += len(ref)
    return totals


def with_empty_transcript_baseline(report: Mapping[str, Any], reference: str | None) -> dict[str, Any]:
    """Return ``report`` with the empty-transcript baseline under ``baselines`` when a reference exists: the
    WER of writing nothing (every reference word deleted, 1.0), the trivial point a single WER is read
    against."""
    out = dict(report)
    if reference is not None:
        out["baselines"] = [
            *report.get("baselines", []),
            {
                "id": "empty_transcript",
                "word_error_rate": word_error_rate(reference, ""),
                "meaning": "a system that outputs nothing; every reference word counts as a deletion",
            },
        ]
    return out


def decode_audio(name: str, data: bytes) -> tuple[Any, int]:
    """Decode one audio file's bytes with the pinned ``soundfile`` into a mono float32 waveform and its
    sampling rate. A file that is empty, unreadable or holds no samples is refused with a ``ValueError``
    that names it, so a BYOD upload fails at loading rather than inside the model call."""
    import soundfile as sf

    if not data:
        raise ValueError(f"{name}: the file is empty")
    try:
        waveform, rate = sf.read(io.BytesIO(data), dtype="float32")
    except Exception as exc:  # libsndfile raises its own error types
        raise ValueError(
            f"{name}: not an audio file soundfile can read (WAV, FLAC, OGG, MP3 ...): {exc}"
        ) from exc
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)
    if len(waveform) == 0:
        raise ValueError(f"{name}: the file holds no audio samples")
    return waveform, int(rate)
