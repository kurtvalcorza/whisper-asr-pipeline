"""Tutorial helpers carried by the two standalone notebooks (2026-10-05 review fixes).

Kept apart from ``pipeline.py`` so the workshop notebook, which carries ``pipeline.py`` byte for byte, is
unchanged:
``decode_audio`` (BYOD decoding that names the file it refuses, WSP-m1 / WSF-m2), ``normalise_transcript`` and
``word_error_breakdown`` (what a WER is made of, WSF-m3) and ``with_empty_transcript_baseline`` (the trivial
reference point of a single-utterance WER, WSP-m2), and ``clip_error_records`` / ``top_insertion_clips`` (one
edit record per clip, so a corpus edit count can be traced to the clips that produce it, row 112).
"""

from __future__ import annotations

import io
from collections.abc import Mapping, Sequence
from typing import Any

from .pipeline import _tokens, word_error_rate


def normalise_transcript(text: str) -> str:
    """The text exactly as the WER helpers compare it: case-folded, punctuation removed, single spaces."""
    return " ".join(_tokens(text))


def _edit_operations(ref: Sequence[str], hyp: Sequence[str]) -> list[str]:
    """One minimum-edit alignment of two token lists as operations in reading order: ``C`` (match),
    ``S``, ``I`` or ``D``. Ties prefer substitution, then deletion, as in ``word_error_breakdown``."""
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
    operations = []
    i, j = len(ref), len(hyp)
    while i or j:
        if i and j and cost[i][j] == cost[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]):
            operations.append("S" if ref[i - 1] != hyp[j - 1] else "C")
            i, j = i - 1, j - 1
        elif i and cost[i][j] == cost[i - 1][j] + 1:
            operations.append("D")
            i -= 1
        else:
            operations.append("I")
            j -= 1
    return operations[::-1]


def word_error_breakdown(references: Sequence[str], hypotheses: Sequence[str]) -> dict[str, int]:
    """Corpus totals of substitutions, insertions and deletions after the same normalisation as
    ``word_error_count`` (one minimum-edit alignment per pair; ties prefer substitution). Their sum over
    ``reference_words`` is the corpus WER, so a change in WER can be read as a change in edit type."""
    if len(references) != len(hypotheses):
        raise ValueError(f"{len(references)} references but {len(hypotheses)} hypotheses")
    totals = {"substitutions": 0, "insertions": 0, "deletions": 0, "reference_words": 0}
    for reference, hypothesis in zip(references, hypotheses, strict=True):
        ref = _tokens(reference)
        operations = _edit_operations(ref, _tokens(hypothesis))
        totals["substitutions"] += operations.count("S")
        totals["insertions"] += operations.count("I")
        totals["deletions"] += operations.count("D")
        totals["reference_words"] += len(ref)
    return totals


def longest_repeated_ngram(tokens: Sequence[str]) -> dict[str, Any]:
    """The longest word n-gram that occurs at least twice in ``tokens`` (occurrences may overlap), with its
    count. ``n`` is 0 when no word repeats. A long repeated n-gram is the signature of a decoding loop."""
    for n in range(len(tokens) - 1, 0, -1):
        counts: dict[tuple[str, ...], int] = {}
        for start in range(len(tokens) - n + 1):
            gram = tuple(tokens[start : start + n])
            counts[gram] = counts.get(gram, 0) + 1
        repeated = [(count, gram) for gram, count in counts.items() if count > 1]
        if repeated:
            count, gram = max(repeated, key=lambda item: item[0])
            return {"n": n, "count": count, "text": " ".join(gram)}
    return {"n": 0, "count": 0, "text": ""}


def clip_error_records(
    ids: Sequence[str], seconds: Sequence[float], references: Sequence[str], hypotheses: Sequence[str]
) -> list[dict[str, Any]]:
    """One record per clip: id, duration, reference and hypothesis word counts (after the WER normalisation),
    substitutions, insertions and deletions from the same alignment as ``word_error_breakdown`` (so the
    records sum to its totals), the insertions that come after the last reference word in that alignment
    (``insertions_after_reference_end``: words written past where the reference stops), and the longest
    repeated word n-gram in the hypothesis."""
    if not len(ids) == len(seconds) == len(references) == len(hypotheses):
        raise ValueError(
            f"{len(ids)} ids, {len(seconds)} durations, {len(references)} references and "
            f"{len(hypotheses)} hypotheses must match"
        )
    records = []
    for clip_id, duration, reference, hypothesis in zip(ids, seconds, references, hypotheses, strict=True):
        ref, hyp = _tokens(reference), _tokens(hypothesis)
        operations = _edit_operations(ref, hyp)
        last_reference_op = max((k for k, op in enumerate(operations) if op != "I"), default=-1)
        records.append(
            {
                "id": clip_id,
                "seconds": round(float(duration), 2),
                "reference_words": len(ref),
                "hypothesis_words": len(hyp),
                "substitutions": operations.count("S"),
                "insertions": operations.count("I"),
                "deletions": operations.count("D"),
                "insertions_after_reference_end": len(operations) - 1 - last_reference_op,
                "longest_repeated_ngram": longest_repeated_ngram(hyp),
            }
        )
    return records


def top_insertion_clips(records: Sequence[Mapping[str, Any]], k: int = 5) -> list[Mapping[str, Any]]:
    """The ``k`` records with the most insertions, most first (ties keep the input order)."""
    return sorted(records, key=lambda record: -record["insertions"])[:k]


def per_clip_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Corpus totals of the per-clip records, plus one clearly diagnostic number: the WER with the
    insertions after the reference end left out (``wer_without_insertions_after_reference_end``). It is
    not a corrected WER; it shows how much of the corpus WER is words written past where the reference
    stops, which is speech a truncated reference left out or text the model added at the end."""
    reference_words = sum(record["reference_words"] for record in records)
    totals = {
        key: sum(record[key] for record in records)
        for key in ("substitutions", "insertions", "deletions", "insertions_after_reference_end")
    }
    if reference_words == 0:
        raise ValueError("the per-clip summary is undefined for empty references")
    edits = totals["substitutions"] + totals["insertions"] + totals["deletions"]
    return {
        "clips": len(records),
        "reference_words": reference_words,
        **totals,
        "wer": round(edits / reference_words, 4),
        "wer_without_insertions_after_reference_end": round(
            (edits - totals["insertions_after_reference_end"]) / reference_words, 4
        ),
    }


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
