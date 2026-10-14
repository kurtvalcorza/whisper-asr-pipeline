# Filipino audio-search annotation review

The current notebook is an **engineering preview**, with 60 AI-authored query drafts (version `ai-draft-v2`, a draft pending native-speaker review) and **zero human relevance judgements**. All 7,200 query/document cells remain `grade: null, status: unjudged`, including the nominated anchors. Never interpret null as unrelated, or a nominated anchor as the only relevant document.

The frozen 120-document corpus is in `tools/audio_sample.json`. Drafts are in `tools/audio_queries.json`, the full review matrix in `tools/audio_qrels.json`, and draft provenance/hash records in `tools/audio_annotations.json`. Runtime copies are named `audio_manifest.json`, `queries.json`, and `qrels.json`. Preserve UTF-8 and LF when updating these records.

## How drafts were prepared

From document-ID-sorted role lists, Python `random.Random(42)` selected 20 development anchors, followed by 40 evaluation anchors, without replacement. Each selected source family has one query. No audio/reference agreement or human naturalness review has been completed. Questions are curated rather than sampled from real search logs.

**Version `ai-draft-v2` (current, 2026-10-13).** Same IDs, anchors, families and split as v1. Each query was written from a one-line summary of its anchor's information (recorded in `tools/audio_queries_v2_construction.json`), never from the sentence, as a topic or information need. Its lexical overlap with the anchor is audited (`tools/audit_audio_queries.py`; normalised tokens longer than three characters) against a declared threshold in the annotation manifest: mean anchor overlap at most 0.10 and per-query at most 0.25. The prepare stage recomputes the audit and refuses a query set over the threshold. Measured: mean 0.0101, maximum 0.1429, mean overlap with other references 0.0104; the anchor is the strictly highest-overlap reference for 0 of 60 queries. Limitation: one AI-assisted author saw each sentence when writing its summary, so the summary step is not an independent elicitation. **Native-speaker review is pending**: check that each query is natural Filipino and still points to its anchor, and record that in the review evidence.

**Version `ai-draft-v1` (previous; kept as `tools/audio_queries_v1.json` with `tools/audio_annotations_v1.json`).** Query text was drafted directly from each anchor's reference sentence (20 relationship/action, 17 paraphrased facts, 23 distinctive terms). Audit: mean anchor overlap 0.4193, maximum 0.8000, other references 0.0126; anchor strictly highest for 50 of 60. This made anchor recovery nearly automatic; hosted runs up to 2026-10-12 used this version.

## Reviewer workflow

1. A competent Filipino-language reviewer listens to each target recording, checks its reference, and assesses naturalness, answerability and ambiguity. Edit or replace inadequate queries before running models. Preserve stable query IDs and a revision log.
2. Judge every query against all 120 corpus references, listening whenever the reference/audio match is doubtful. Assign integer grades: **0** unrelated, **1** partially useful, **2** directly answers. Every query needs at least one grade-2 document. Keep a rationale for each nonzero label, and record reviewer identity and date in review evidence.
3. Independently review all nonzero labels and ambiguous cases, plus a deterministic 20% sample of zero labels. Record the sampled IDs, initial decisions, disagreements and reconciliation. Do not claim multiple reviewers when only one contributed; record incomplete review explicitly.
4. Confirm that development and evaluation queries target disjoint source-text families. Do not modify the frozen corpus or queries in response to evaluation rankings.
5. Replace the null grades only after actual judgement and set each completed pair's status to `reviewed`. Retain original drafts separately in the review record. Write a new annotation version, detailed provenance, reviewer identifiers, reconciliation record and file checksums. Updating only a status flag is insufficient. If exactly one reviewer is available, declare `independent_review: false` and a nonempty `single_reviewer_limitation`; no independent consensus is claimed.
6. Validate the entire matrix and re-freeze the experiment before any benchmark-quality evaluation. Regenerate the notebook using its source builder so embedded hashes and annotation records agree. Import reviewed records into the source files before generation; do not edit embedded notebook strings manually. No automatic human-review claim or one-click status promotion is provided.

The public preview can report nominated-anchor hit/rank diagnostics. Recall@5, MRR@10 and nDCG@10 require the completed relevance layer. Other documents may also answer a draft query; the engineering diagnostics cannot estimate exhaustive recall.

## Known drafting concerns to review

- Several source clips lack surrounding context. `dev-02`, `dev-03`, `dev-07`, `dev-15`, `test-06`, `test-10`, `test-12`, `test-16`, `test-17`, `test-26`, `test-30`, `test-32`, `test-34`, `test-35` and `test-38` deliberately ask only about the available passage. A reviewer should decide whether these remain too generic or ambiguous for search.
- Preserve source wording for diagnosis, but review apparent names or transcription oddities such as **Mojarcan**, **Newt Gindrich** and **Springbooks**. Do not silently correct the reference and then compare ASR against the altered text.
- `test-08` asks what the passage says about Ebola research; the historical passage is not current medical advice. `test-12` likewise describes a visa statement explicitly dated 2009, not present travel requirements.
- Short generic passages may have more than one relevant corpus recording. Full-corpus judgement is required even when a nominated anchor looks obvious.

## Rights and provenance

Source: Google FLEURS `fil_ph`, Conneau et al. (2022), *FLEURS: Few-shot Learning Evaluation of Universal Representations of Speech*, <https://arxiv.org/abs/2205.12446>, CC BY 4.0. The drafts transform reference statements into search questions; preserve this attribution and the corpus revision/hash manifest when redistributing them. AI assistance is disclosed in each draft and in the annotation manifest. Human review is pending.
