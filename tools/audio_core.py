"""Model-independent scientific contracts for the Filipino audio-search capstone."""

from __future__ import annotations

import math
import unicodedata
from collections import Counter
from collections.abc import Sequence

import numpy as np

NORMALIZATION_VERSION = "nfc-lower-punctuation-space-collapse-v1"


def normalize(text: str) -> str:
    """Preserve diacritics and lexical content; map Unicode punctuation to spaces."""
    if not isinstance(text, str):
        raise ValueError("Text must be a string")
    text = unicodedata.normalize("NFC", text).lower()
    return " ".join("".join(" " if unicodedata.category(c).startswith("P") else c for c in text).split())


def edit_counts(reference: Sequence, hypothesis: Sequence) -> dict:
    """Levenshtein alignment; equal-cost precedence is substitution, deletion, insertion."""
    previous = [(j, 0, 0, j) for j in range(len(hypothesis) + 1)]
    for i, token in enumerate(reference, 1):
        current = [(i, 0, i, 0)]
        for j, other in enumerate(hypothesis, 1):
            if token == other:
                current.append(previous[j - 1])
            else:
                cost, sub, delete, insert = previous[j - 1]
                candidates = [(cost + 1, sub + 1, delete, insert)]
                cost, sub, delete, insert = previous[j]
                candidates.append((cost + 1, sub, delete + 1, insert))
                cost, sub, delete, insert = current[j - 1]
                candidates.append((cost + 1, sub, delete, insert + 1))
                current.append(min(candidates, key=lambda item: item[0]))
        previous = current
    errors, substitutions, deletions, insertions = previous[-1]
    return dict(
        errors=errors,
        substitutions=substitutions,
        deletions=deletions,
        insertions=insertions,
        reference_units=len(reference),
    )


def asr_metrics(references: list[str], hypotheses: list[str]) -> dict:
    """Corpus edit totals, not the mean of individual error rates; CER includes spaces."""
    if len(references) != len(hypotheses) or not references:
        raise ValueError("ASR requires equal, nonempty collections")
    per_clip = []
    for reference, hypothesis in zip(references, hypotheses, strict=True):
        ref, hyp = normalize(reference), normalize(hypothesis)
        if not ref:
            raise ValueError("Empty normalized reference")
        word, char = edit_counts(ref.split(), hyp.split()), edit_counts(ref, hyp)
        per_clip.append(
            dict(
                reference=ref,
                hypothesis=hyp,
                word=word,
                character=char,
                wer=word["errors"] / word["reference_units"],
                cer=char["errors"] / char["reference_units"],
            )
        )
    totals = {
        kind: {key: sum(row[kind][key] for row in per_clip) for key in per_clip[0][kind]}
        for kind in ("word", "character")
    }
    return dict(
        normalization=NORMALIZATION_VERSION,
        cer_spaces_included=True,
        per_clip=per_clip,
        **totals,
        wer=totals["word"]["errors"] / totals["word"]["reference_units"],
        cer=totals["character"]["errors"] / totals["character"]["reference_units"],
    )


def rank_scores(doc_ids: list[str], scores: Sequence[float]) -> list[dict]:
    """Descending score, then ascending stable document ID; all documents remain present."""
    if len(doc_ids) != len(scores) or len(set(doc_ids)) != len(doc_ids) or not doc_ids:
        raise ValueError("Unique document IDs and matching scores required")
    values = np.asarray(scores, dtype=np.float64)
    if values.shape != (len(doc_ids),) or not np.isfinite(values).all():
        raise ValueError("Scores must be finite scalars")
    return [
        dict(doc_id=doc_ids[i], score=float(values[i]))
        for i in sorted(range(len(doc_ids)), key=lambda i: (-values[i], doc_ids[i]))
    ]


class BM25:
    """Robertson BM25 with positive log(1 + (N-df+.5)/(df+.5)) IDF.

    Repeated query tokens contribute once per occurrence. Empty documents and
    empty queries have zero lexical score. Each condition fits its own corpus.
    """

    def __init__(self, doc_ids: list[str], texts: list[str], k1: float = 1.2, b: float = 0.75):
        rank_scores(doc_ids, np.zeros(len(texts)))
        if not math.isfinite(k1) or k1 <= 0 or not math.isfinite(b) or not 0 <= b <= 1:
            raise ValueError("Invalid BM25 parameters")
        self.doc_ids, self.k1, self.b = list(doc_ids), k1, b
        self.tokens = [Counter(normalize(text).split()) for text in texts]
        self.lengths = np.array([sum(tokens.values()) for tokens in self.tokens], dtype=float)
        self.average_length = float(self.lengths.mean())
        self.df = Counter(term for tokens in self.tokens for term in tokens)

    def rank(self, query: str) -> list[dict]:
        """Rank all documents, retaining blank hypotheses as zero-score candidates."""
        scores = np.zeros(len(self.doc_ids))
        if self.average_length:
            norm = self.k1 * (1 - self.b + self.b * self.lengths / self.average_length)
            for term in normalize(query).split():
                df = self.df[term]
                idf = math.log1p((len(self.doc_ids) - df + 0.5) / (df + 0.5))
                tf = np.array([tokens[term] for tokens in self.tokens], dtype=float)
                scores += np.divide(
                    idf * tf * (self.k1 + 1),
                    tf + norm,
                    out=np.zeros_like(tf),
                    where=(tf + norm) > 0,
                )
        return rank_scores(self.doc_ids, scores)


def dense_rank(doc_ids: list[str], embeddings: np.ndarray, query_embedding: np.ndarray) -> list[dict]:
    """Exact cosine ranking, rejecting invalid/zero vectors rather than hiding them."""
    docs, query = np.asarray(embeddings, dtype=float), np.asarray(query_embedding, dtype=float)
    if docs.ndim != 2 or docs.shape[0] != len(doc_ids) or query.shape != (docs.shape[1],):
        raise ValueError("Embedding dimensions disagree")
    if not np.isfinite(docs).all() or not np.isfinite(query).all():
        raise ValueError("Embeddings must be finite")
    norms, query_norm = np.linalg.norm(docs, axis=1), np.linalg.norm(query)
    if np.any(norms == 0) or query_norm == 0:
        raise ValueError("Zero embeddings have undefined cosine")
    return rank_scores(doc_ids, (docs / norms[:, None]) @ (query / query_norm))


def retrieval_metrics(ranked_ids: list[str], qrels: dict[str, int]) -> dict:
    """Grade-2 Recall@5/MRR@10 and exponential-gain nDCG@10."""
    if not qrels or any(type(grade) is not int or grade not in (0, 1, 2) for grade in qrels.values()):
        raise ValueError("Relevance grades must be integers 0, 1, 2")
    if len(set(ranked_ids)) != len(ranked_ids) or not set(ranked_ids) <= qrels.keys():
        raise ValueError("Unknown or repeated ranked document")
    positives = {doc for doc, grade in qrels.items() if grade == 2}
    if not positives:
        raise ValueError("Every query needs a directly relevant reference")
    grades = [qrels[doc] for doc in ranked_ids[:10]]
    dcg = sum((2**grade - 1) / math.log2(i + 2) for i, grade in enumerate(grades))
    ideal = sorted(qrels.values(), reverse=True)[:10]
    idcg = sum((2**grade - 1) / math.log2(i + 2) for i, grade in enumerate(ideal))
    return dict(
        recall_at_5=len(positives.intersection(ranked_ids[:5])) / len(positives),
        mrr_at_10=next((1 / (i + 1) for i, grade in enumerate(grades) if grade == 2), 0.0),
        ndcg_at_10=dcg / idcg,
        relevant_count=len(positives),
        retrieved_relevant_at_5=len(positives.intersection(ranked_ids[:5])),
    )


def validate_annotations(
    documents: list[dict], queries: list[dict], qrels: list[dict], manifest: dict
) -> dict:
    """Validate complete labels while keeping engineering drafts explicitly unqualified.

    documents: doc_id, family_id, role. queries: query_id, text, target_family_id,
    role. qrels: query_id, doc_id, grade, rationale. A reviewed manifest additionally
    requires reviewer_ids, review_evidence, reconciliation and human_review_complete=True.
    Independent review may be unavailable only with one named reviewer and an explicit
    single_reviewer_limitation. These are recorded claims, not identity proof.
    """
    docs = {row["doc_id"]: row for row in documents}
    qs = {row["query_id"]: row for row in queries}
    if not docs or not qs or len(docs) != len(documents) or len(qs) != len(queries):
        raise ValueError("Duplicate or empty document/query IDs")
    families = {}
    for row in documents:
        if row["role"] not in ("development", "evaluation"):
            raise ValueError("Unknown document role")
        family = row["family_id"]
        if not family or families.setdefault(family, row["role"]) != row["role"]:
            raise ValueError("Text family crosses roles")
    for row in queries:
        if not normalize(row["text"]) or families.get(row["target_family_id"]) != row["role"]:
            raise ValueError("Query target role mismatch or empty text")
    status = manifest.get("status")
    if status not in ("engineering_preview", "draft", "human_reviewed"):
        raise ValueError("Explicit annotation status required")
    reviewed = status == "human_reviewed"
    seen, matrix = set(), {query: {} for query in qs}
    for row in qrels:
        pair = row["query_id"], row["doc_id"]
        if pair in seen or pair[0] not in qs or pair[1] not in docs:
            raise ValueError("Duplicate or unknown qrel pair")
        grade = row["grade"]
        if reviewed and row.get("status") != "reviewed":
            raise ValueError("Every graded pair must have completed reviewed status")
        if grade is None and not reviewed:
            if row.get("status") != "unjudged":
                raise ValueError("Null relevance must be explicitly unjudged")
            seen.add(pair)
            matrix[pair[0]][pair[1]] = None
            continue
        if type(grade) is not int or grade not in (0, 1, 2):
            raise ValueError("Invalid relevance grade")
        if grade and not str(row.get("rationale", "")).strip():
            raise ValueError("Nonzero relevance needs a rationale")
        seen.add(pair)
        matrix[pair[0]][pair[1]] = grade
    if len(seen) != len(docs) * len(qs):
        raise ValueError("Incomplete query-document judgement matrix")
    if reviewed:
        for labels in matrix.values():
            retrieval_metrics([], labels)
        if (
            manifest.get("human_review_complete") is not True
            or not manifest.get("reviewer_ids")
            or not manifest.get("review_evidence")
            or not manifest.get("reconciliation")
        ):
            raise ValueError("Human review lacks evidence records")
        reviewers = manifest["reviewer_ids"]
        if (
            not isinstance(reviewers, list)
            or any(not isinstance(reviewer, str) or not reviewer.strip() for reviewer in reviewers)
            or len(set(reviewers)) != len(reviewers)
        ):
            raise ValueError("Reviewer IDs must be distinct nonempty strings")
        if manifest.get("independent_review") is True and len(reviewers) < 2:
            # Counting IDs cannot prove independence, but one reviewer cannot be independent review.
            raise ValueError("Independent review requires at least two distinct reviewers")
        if manifest.get("independent_review") is not True:
            limitation = manifest.get("single_reviewer_limitation")
            if (
                manifest.get("independent_review") is not False
                or len(reviewers) != 1
                or not isinstance(limitation, str)
                or not limitation.strip()
            ):
                raise ValueError("Independent review or explicit single-reviewer limitation required")
    elif manifest.get("human_review_complete") or manifest.get("benchmark_qualified"):
        raise ValueError("Draft annotations cannot claim completed human review")
    return dict(
        benchmark_qualified=reviewed,
        annotation_status=status,
        queries=len(qs),
        documents=len(docs),
        matrix_entries=len(seen),
        judgements=sum(grade is not None for labels in matrix.values() for grade in labels.values()),
        single_reviewer_limitation=(
            manifest.get("single_reviewer_limitation")
            if reviewed and manifest.get("independent_review") is False
            else None
        ),
        qrels_by_query=matrix,
    )


def anchor_metrics(ranked_ids: list[str], anchor_doc_id: str) -> dict:
    """Engineering diagnostic only; an intended anchor is not exhaustive relevance."""
    if not anchor_doc_id or len(set(ranked_ids)) != len(ranked_ids):
        raise ValueError("Unique ranked IDs and a nominated anchor required")
    rank = ranked_ids.index(anchor_doc_id) + 1 if anchor_doc_id in ranked_ids else None
    return dict(
        anchor_hit_at_5=int(rank is not None and rank <= 5),
        anchor_reciprocal_rank=1 / rank if rank else 0.0,
        anchor_rank=rank,
        benchmark_qualified=False,
    )


def paired_bootstrap(
    a: Sequence[float], b: Sequence[float], groups: Sequence[str], n_boot: int = 2000, seed: int = 42
) -> dict:
    """Resample whole target families; query-weighted paired difference is A minus B."""
    left, right = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if (
        left.ndim != 1
        or left.shape != right.shape
        or len(left) != len(groups)
        or not len(left)
        or not np.isfinite(left).all()
        or not np.isfinite(right).all()
        or n_boot < 1
    ):
        raise ValueError("Invalid paired observations")
    if any(not isinstance(group, str) or not group for group in groups):
        raise ValueError("Nonempty source family IDs required")
    unique = sorted(set(groups))
    members = [np.flatnonzero(np.asarray(groups) == group) for group in unique]
    rng = np.random.default_rng(seed)
    samples = np.empty((n_boot, 3))
    for i in range(n_boot):
        idx = np.concatenate([members[j] for j in rng.integers(len(unique), size=len(unique))])
        av, bv = float(left[idx].mean()), float(right[idx].mean())
        samples[i] = av, bv, av - bv
    return dict(
        a=float(left.mean()),
        b=float(right.mean()),
        difference=float((left - right).mean()),
        ci95=np.quantile(samples, [0.025, 0.975], axis=0).T.tolist(),
        ci95_order=["a", "b", "a_minus_b"],
        family_count=len(unique),
        query_count=len(left),
        replicates=n_boot,
        seed=seed,
        resampling_unit="target_source_family",
    )
