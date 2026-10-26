"""Row 112 (2026-10-26): the frozen baseline's insertions are mostly speech the MINDS-14 references leave out.

The references are machine ASR output (Google ASR 1-best, dataset paper [3]) that often leave out speech or
stop early, so the notebook says so, and it keeps one edit record per held-out clip for the baseline and the
adapted model in the evaluation report and in the executed notebook's output.
"""

import json
from pathlib import Path

import pytest

from whisper_asr_pipeline import pipeline as pipeline_module
from whisper_asr_pipeline.tutorial_support import (
    clip_error_records,
    longest_repeated_ngram,
    per_clip_summary,
    top_insertion_clips,
    word_error_breakdown,
)

NB_FT = Path(__file__).resolve().parents[1] / "tutorials" / "whisper_asr_finetune_colab.ipynb"


def _cells(kind):
    nb = json.loads(NB_FT.read_text(encoding="utf-8"))
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == kind]


def _record(ref, hyp, clip_id="c", seconds=1.0):
    return clip_error_records([clip_id], [seconds], [ref], [hyp])[0]


# --- helper unit tests -------------------------------------------------------------------------------------


def test_trailing_words_count_as_insertions_after_the_reference_end():
    record = _record(
        "i want to pay my bill", "I want to pay my bill, please, right now.", "en-US~a.wav", 4.257
    )
    assert record == {
        "id": "en-US~a.wav",
        "seconds": 4.26,
        "reference_words": 6,
        "hypothesis_words": 9,
        "substitutions": 0,
        "insertions": 3,
        "deletions": 0,
        "insertions_after_reference_end": 3,
        "longest_repeated_ngram": {"n": 0, "count": 0, "text": ""},
    }


def test_an_insertion_inside_the_reference_is_not_after_its_end():
    record = _record("a b c", "a x b c")
    assert (record["insertions"], record["insertions_after_reference_end"]) == (1, 0)


def test_deletions_and_empty_sides():
    assert _record("a b c d", "a b")["deletions"] == 2
    empty_hyp = _record("a b", "")
    assert (
        empty_hyp["deletions"],
        empty_hyp["hypothesis_words"],
        empty_hyp["insertions_after_reference_end"],
    ) == (2, 0, 0)
    empty_ref = _record("", "a b")
    assert (empty_ref["insertions"], empty_ref["insertions_after_reference_end"]) == (2, 2)


def test_normalisation_matches_the_wer_helpers():
    record = _record("Hello, World.", "hello world")
    assert record["substitutions"] + record["insertions"] + record["deletions"] == 0


def test_longest_repeated_ngram_finds_decoding_loops():
    assert longest_repeated_ngram(["thank", "you", "thank", "you", "thank", "you"]) == {
        "n": 4,
        "count": 2,
        "text": "thank you thank you",
    }
    assert longest_repeated_ngram(["the", "the"]) == {"n": 1, "count": 2, "text": "the"}
    assert longest_repeated_ngram(["a", "b", "c"])["n"] == 0
    assert longest_repeated_ngram([])["n"] == 0


def test_records_sum_to_the_corpus_breakdown_and_wer():
    references = [
        "the cat sat on the mat",
        "one hundred dollars",
        "I'd like to pay",
        "hi there I just wanted",
    ]
    hypotheses = [
        "the cat sat mat today",
        "100 dollars",
        "I like to pay my bill",
        "hi there I just wanted to ask",
    ]
    records = clip_error_records(list("abcd"), [1.0, 2.0, 3.0, 4.0], references, hypotheses)
    totals = word_error_breakdown(references, hypotheses)
    for key in ("substitutions", "insertions", "deletions", "reference_words"):
        assert sum(r[key] for r in records) == totals[key], key
    summary = per_clip_summary(records)
    assert summary["wer"] == round(pipeline_module.corpus_word_error_rate(references, hypotheses), 4)
    assert [r["insertions_after_reference_end"] for r in records] == [0, 0, 2, 2]
    assert summary["clips"] == 4 and summary["insertions_after_reference_end"] == 4
    edits = totals["substitutions"] + totals["insertions"] + totals["deletions"]
    assert summary["wer_without_insertions_after_reference_end"] == round(
        (edits - 4) / totals["reference_words"], 4
    )


def test_top_insertion_clips_orders_by_insertions_and_keeps_ties_stable():
    records = [{"id": i, "insertions": n} for i, n in enumerate([1, 5, 3, 5, 0, 2, 4])]
    assert [r["id"] for r in top_insertion_clips(records, 5)] == [1, 3, 6, 2, 5]
    assert len(top_insertion_clips(records[:2])) == 2


def test_mismatched_inputs_are_refused():
    with pytest.raises(ValueError, match="must match"):
        clip_error_records(["a"], [1.0, 2.0], ["x"], ["x"])
    with pytest.raises(ValueError, match="undefined"):
        per_clip_summary([_record("", "")])


# --- acceptance: notebook wording and output ---------------------------------------------------------------


def test_section_6_reads_the_baseline_as_disagreement_with_machine_references():
    md = "\n".join(_cells("markdown"))
    assert "written by people" not in md
    assert "was not checked clip by clip" not in md
    assert "machine ASR output, the 1-best hypothesis of Google's recogniser [3]" in md
    assert (
        "Most of the baseline's insertions are speech the reference left out, not text Whisper invented" in md
    )
    assert "read the baseline WER as disagreement with another recogniser" in md
    assert "Gerz *et al.*" in md and "MINDS-14" in md  # [3] is still the cited source


def test_section_9_names_the_learned_omissions_as_part_of_the_gain():
    md = "\n".join(_cells("markdown"))
    assert (
        "part of that measured gain is the adapter learning to leave speech out the way the references do"
        in md
    )
    assert "deletions rose while insertions fell" in md and "can vary between runs" in md


def test_no_inferred_share_or_run_status_is_written_into_the_notebook():
    md = "\n".join(_cells("markdown"))
    for stale in ("75", "85%", "next run", "will settle", "pending"):
        assert stale not in md.split("## 6.")[1].split("## 10.")[0], stale


def test_per_clip_records_are_printed_and_kept_in_the_report():
    code = "\n".join(_cells("code"))
    assert (
        "baseline_clip_records = clip_error_records("
        "eval_ids, eval_seconds, eval_references, baseline_transcripts)" in code
    )
    assert (
        "adapted_clip_records = clip_error_records("
        "eval_ids, eval_seconds, eval_references, adapted_transcripts)" in code
    )
    assert code.count("report['per_clip'] = PER_CLIP") == 2  # Section 9 report and the Section 10 rebuild
    assert (
        "top_insertion_clips(baseline_clip_records, 5)" in code
        and "top_insertion_clips(adapted_clip_records, 5)" in code
    )
    for model in ("baseline", "adapted"):
        assert f"for record in {model}_clip_records:\n    print(json.dumps(record" in code
