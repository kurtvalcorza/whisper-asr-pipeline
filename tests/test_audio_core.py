"""Known-answer and refusal checks, without model downloads."""

import copy
import importlib.util
from pathlib import Path

import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location("audio_core", Path(__file__).parents[1] / "tools/audio_core.py")
core = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(core)


def test_unicode_normalization_preserves_diacritics_and_words():
    assert core.normalize("  Áraw—GABI,  'ISA'! a\u0301 ") == "áraw gabi isa á"


def test_wer_can_exceed_one_and_counts_insertions():
    result = core.asr_metrics(["isa"], ["isa dalawa tatlo apat"])
    assert result["wer"] == 3
    assert result["word"]["insertions"] == 3


def test_blank_hypothesis_counts_deletions():
    result = core.asr_metrics(["isa dalawa"], [""])
    assert result["wer"] == result["cer"] == 1
    assert result["word"]["deletions"] == 2


def test_corpus_rate_uses_reference_totals():
    result = core.asr_metrics(["a", "b c d"], ["", "b c d"])
    assert result["wer"] == 0.25
    assert result["cer"] == pytest.approx(1 / 6)


@pytest.mark.parametrize("references,hypotheses", [([], []), (["!"], ["a"]), (["a"], [])])
def test_asr_rejects_invalid_reference_collections(references, hypotheses):
    with pytest.raises(ValueError):
        core.asr_metrics(references, hypotheses)


def test_known_edit_counts():
    assert core.edit_counts("kitten", "sitting") == dict(
        errors=3, substitutions=2, deletions=0, insertions=1, reference_units=6
    )


def test_bm25_known_score_and_blank_retention():
    model = core.BM25(["z", "a", "blank"], ["ulan", "ulan", ""])
    ranked = model.rank("ulan")
    assert [row["doc_id"] for row in ranked] == ["a", "z", "blank"]
    expected = np.log(1 + 1.5 / 2.5) * 2.2 / (1 + 1.2 * (0.25 + 0.75 * 1.5))
    assert ranked[0]["score"] == pytest.approx(expected)
    assert ranked[-1]["score"] == 0


def test_bm25_blank_corpus_query_and_missing_term():
    model = core.BM25(["z", "a"], ["", ""])
    assert model.rank("isa") == [{"doc_id": "a", "score": 0.0}, {"doc_id": "z", "score": 0.0}]
    assert core.BM25(["a"], ["ulan"]).rank("!")[0]["score"] == 0
    assert core.BM25(["a"], ["ulan"]).rank("araw")[0]["score"] == 0


def test_condition_specific_bm25_statistics():
    a = core.BM25(["a", "b"], ["ulan", "araw"])
    b = core.BM25(["a", "b"], ["ulan", "ulan"])
    assert a.rank("ulan")[0]["score"] != b.rank("ulan")[0]["score"]


def test_bm25_full_length_normalization_retains_blank_document():
    model = core.BM25(["a", "b"], ["ulan", ""], b=1)
    assert model.rank("ulan")[-1] == {"doc_id": "b", "score": 0.0}


def test_exact_cosine_normalization_and_ties():
    ranked = core.dense_rank(["z", "a", "b"], np.array([[1, 0], [2, 0], [0, 3]]), np.array([5, 0]))
    assert [row["doc_id"] for row in ranked] == ["a", "z", "b"]
    assert [row["score"] for row in ranked] == [1, 1, 0]


@pytest.mark.parametrize(
    "docs,query", [([[0, 0]], [1, 1]), ([[1, 1]], [0, 0]), ([[float("nan"), 0]], [1, 1]), ([[1, 0]], [1])]
)
def test_bad_embeddings_refused(docs, query):
    with pytest.raises(ValueError):
        core.dense_rank(["a"], np.array(docs), np.array(query))


def test_multirelevant_metrics_use_all_positives():
    labels = {"a": 2, "b": 2, "c": 1, "d": 0}
    metrics = core.retrieval_metrics(["c", "a", "d"], labels)
    assert metrics["recall_at_5"] == 0.5
    assert metrics["mrr_at_10"] == 0.5
    assert metrics["relevant_count"] == 2
    expected = (1 + 3 / np.log2(3)) / (3 + 3 / np.log2(3) + 1 / np.log2(4))
    assert metrics["ndcg_at_10"] == pytest.approx(expected)


def test_candidate_miss_cannot_be_rescued_and_cutoffs_apply():
    labels = {str(i): int(i == 11) * 2 for i in range(12)}
    assert core.retrieval_metrics(list(labels), labels)["mrr_at_10"] == 0
    assert core.retrieval_metrics(list(labels)[:10], labels)["recall_at_5"] == 0
    assert core.retrieval_metrics(list(labels)[:10][::-1], labels)["ndcg_at_10"] == 0


@pytest.mark.parametrize(
    "rank,labels",
    [(["a", "a"], {"a": 2}), (["b"], {"a": 2}), ([], {"a": 0}), ([], {"a": None}), ([], {"a": True})],
)
def test_invalid_qrels_refused(rank, labels):
    with pytest.raises(ValueError):
        core.retrieval_metrics(rank, labels)


def annotations():
    docs = [
        dict(doc_id="a", family_id="fa", role="development"),
        dict(doc_id="b", family_id="fb", role="evaluation"),
    ]
    queries = [dict(query_id="q", text="Anong ulan?", target_family_id="fa", role="development")]
    labels = [
        dict(query_id="q", doc_id="a", grade=2, rationale="Draft intended anchor", status="draft"),
        dict(query_id="q", doc_id="b", grade=None, rationale="", status="unjudged"),
    ]
    return docs, queries, labels, dict(status="draft")


def test_preview_null_labels_never_become_zero():
    result = core.validate_annotations(*annotations())
    assert result["benchmark_qualified"] is False
    assert result["matrix_entries"] == 2 and result["judgements"] == 1
    assert result["qrels_by_query"]["q"]["b"] is None


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "role", "grade", "rationale", "fake_review"])
def test_annotation_integrity(mutation):
    docs, queries, labels, manifest = annotations()
    if mutation == "missing":
        labels.pop()
    elif mutation == "duplicate":
        labels.append(copy.deepcopy(labels[0]))
    elif mutation == "role":
        queries[0]["role"] = "evaluation"
    elif mutation == "grade":
        labels[0]["grade"] = 3
    elif mutation == "rationale":
        labels[0]["rationale"] = ""
    elif mutation == "fake_review":
        manifest["human_review_complete"] = True
    with pytest.raises(ValueError):
        core.validate_annotations(docs, queries, labels, manifest)


def test_reviewed_claim_requires_complete_labels_and_provenance():
    docs, queries, labels, manifest = annotations()
    manifest["status"] = "human_reviewed"
    with pytest.raises(ValueError):
        core.validate_annotations(docs, queries, labels, manifest)
    labels[1]["grade"] = 0
    with pytest.raises(ValueError):
        core.validate_annotations(docs, queries, labels, manifest)
    for label in labels:
        label["status"] = "reviewed"
    manifest.update(
        human_review_complete=True,
        reviewer_ids=["fixture-reviewer-a", "fixture-reviewer-b"],
        review_evidence="fixture review record",
        independent_review=True,
        reconciliation="fixture disposition",
    )
    assert core.validate_annotations(docs, queries, labels, manifest)["benchmark_qualified"] is True


def _reviewed(reviewers, independent, **extra):
    docs, queries, labels, manifest = annotations()
    labels[1]["grade"] = 0
    for label in labels:
        label["status"] = "reviewed"
    manifest.update(
        status="human_reviewed",
        human_review_complete=True,
        reviewer_ids=reviewers,
        independent_review=independent,
        **extra,
    )
    return docs, queries, labels, manifest


def test_independent_review_claim_needs_two_reviewers():
    evidence = dict(review_evidence="Completed review record", reconciliation="Adjudicated disagreements")
    with pytest.raises(ValueError, match="at least two"):
        core.validate_annotations(*_reviewed(["reviewer-a"], True, **evidence))
    single = _reviewed(
        ["reviewer-a"], False, single_reviewer_limitation="One reviewer; no independent check.", **evidence
    )
    assert core.validate_annotations(*single)["benchmark_qualified"] is True
    assert core.validate_annotations(*_reviewed(["reviewer-a", "reviewer-b"], True, **evidence))[
        "benchmark_qualified"
    ]
    with pytest.raises(ValueError, match="evidence"):
        core.validate_annotations(*_reviewed(["reviewer-a", "reviewer-b"], True))


def test_anchor_diagnostic_is_not_relevance_metric():
    result = core.anchor_metrics(["b", "a"], "a")
    assert result == dict(
        anchor_hit_at_5=1, anchor_reciprocal_rank=0.5, anchor_rank=2, benchmark_qualified=False
    )
    assert "recall_at_5" not in result


@pytest.mark.parametrize(
    "reviewers,limitation,accepted",
    [
        (["reviewer-a"], "One Filipino-language reviewer; no independent review available.", True),
        (["reviewer-a"], "", False),
        (["reviewer-a"], None, False),
        (["reviewer-a", "reviewer-b"], "No independent review", False),
        (["reviewer-a", "reviewer-a"], "Duplicate identity", False),
        ("reviewer-a", "String is not an identity list", False),
    ],
)
def test_single_reviewer_exception_is_explicit(reviewers, limitation, accepted):
    docs, queries, labels, manifest = annotations()
    labels[1]["grade"] = 0
    for label in labels:
        label["status"] = "reviewed"
    manifest.update(
        status="human_reviewed",
        human_review_complete=True,
        reviewer_ids=reviewers,
        review_evidence="Completed review record",
        independent_review=False,
        single_reviewer_limitation=limitation,
        reconciliation="Single reviewer resolved ambiguous cases; no consensus statistic claimed",
    )
    if accepted:
        result = core.validate_annotations(docs, queries, labels, manifest)
        assert result["benchmark_qualified"] is True
        assert result["single_reviewer_limitation"] == limitation
        manifest.pop("review_evidence")
        with pytest.raises(ValueError, match="evidence"):
            core.validate_annotations(docs, queries, labels, manifest)
    else:
        with pytest.raises(ValueError):
            core.validate_annotations(docs, queries, labels, manifest)


def test_cluster_bootstrap_paired_direction_and_repeatability():
    result = core.paired_bootstrap([1, 1, 0], [0, 0, 0], ["family1", "family1", "family2"])
    assert result == core.paired_bootstrap([1, 1, 0], [0, 0, 0], ["family1", "family1", "family2"])
    assert result["difference"] == pytest.approx(2 / 3)
    assert result["family_count"] == 2
    assert result["ci95"][2] == [0, 1]


def test_single_family_resampled_as_unit():
    result = core.paired_bootstrap([0, 1], [0, 0], ["same", "same"])
    assert result["ci95"][2] == [0.5, 0.5]


@pytest.mark.parametrize(
    "a,b,groups", [([], [], []), ([1], [1, 2], ["a"]), ([float("nan")], [0], ["a"]), ([1], [0], [""])]
)
def test_bad_bootstrap_inputs(a, b, groups):
    with pytest.raises(ValueError):
        core.paired_bootstrap(a, b, groups)
