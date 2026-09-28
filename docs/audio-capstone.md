# Filipino Audio Archive Search capstone

**Status: Engineering preview / Candidate.** Built for DIMER NOTEBOOK_SPEC 2.2,
`E2E` / `GUIDED`. The composed workflow is transcription → index construction →
evaluation → search → export → fresh-process reconstruction. No weight training
or adaptation is claimed. Hosted execution and human relevance review remain
open acceptance gates.

## Build and run

```powershell
python tools/build_audio_capstone.py
python tools/build_audio_capstone.py --check
```

Open `tutorials/DIMER_Filipino_Audio_Archive_Search_Capstone.ipynb` in Colab,
select a T4 GPU and choose **Run all**. The notebook carries all pipeline source
and creates an isolated Python 3.12 environment from its hashed Linux lock;
it does not clone a repository or import DIMER worker code. Public source data
and model snapshots are acquired at immutable revisions and checked by size and
SHA-256, including cached files. No paid API, secret or manual restart is needed
by design; a clean hosted run is still needed to verify that path.

Targets, not measured results: 60 minutes, 20 GiB free disk before setup, and
12 GiB peak allocated GPU memory. The model families load sequentially. Do not
reduce the sample after seeing scores to meet these targets.

## Frozen sample and annotation boundary

FLEURS `fil_ph` at `70bb2e84b976b7e960aa89f1c648e09c59f894dd`:

| Item | Frozen value |
| --- | ---: |
| Source validation/test rows audited | 1,382 |
| Source Parquet bytes verified | 1,557,013,923 |
| Selected development / evaluation clips | 40 / 80 |
| Selected audio duration | 1,715.46 seconds |
| Development / evaluation query drafts | 20 / 40 |
| Human-reviewed relevance judgements | 0 of 7,200 |

`tools/freeze_audio_data.py` reproduces the cohort from the source Parquet
files. Selection excludes cross-role connected ID/text/audio families and takes
one clip per family; source sentence IDs are not speaker IDs. Audio remains at
the source's mono 16 kHz rate without clipping or truncation. The audit records
every exclusion and the realized duration distribution.

All relevance grades are **null / unjudged**, including nominated anchors.
The preview reports `anchor_hit_at_5` and `anchor_reciprocal_rank_at_10` only.
They measure recovery of a nominated engineering target; they do not establish
search relevance. Full Recall@5, MRR@10 and nDCG@10 require the complete reviewed
annotation layer and provenance described in [the review protocol](audio-annotation-review.md).
Any annotation change needs a new frozen version and a complete rerun.

## Experiment

All six systems search the same 120 IDs: reference/automatic transcripts crossed
with BM25, normalized Qwen3 dense retrieval, and dense top-10 → Qwen3 reranking.
Each condition fits its own lexical statistics. Blank automatic transcripts keep
their IDs and use the documented tokenizer special-token embedding policy.
No input exceeds a token ceiling silently.

WER and CER use corpus edit totals, retaining insertions, deletions and
substitutions. Normalization is NFC, lowercase, punctuation-to-space and
whitespace collapse, preserving diacritics; CER includes normalized spaces.
WER may exceed 100%. Search scores are ranking values, not calibrated
probabilities. Grouped paired bootstrap intervals use source families, 2,000
replicates and seed 42; these intervals do not correct unreviewed annotations or
make curated queries representative of user behaviour.

The controlled activity changes only candidate depth from 10 to 20 on the
20 development queries. Canonical evaluation results remain separate. A
candidate absent from the shortlist cannot be recovered by reranking. Its cost
comparison times depth 10 and depth 20 on the same 20 development queries, in one
process with the reranker loaded, after one untimed warm-up call and with the
depth order alternating per query (`activity_latency.csv`,
`activity_summary.json → paired_latency`). The canonical run's 60-query timings
are a different workload and are not compared with it. The canonical run also
records per-query reranking time.

## Artifacts and BYOD

`results.zip` holds the evaluation evidence: provenance, references, draft
annotations, outputs, scores, stage receipts, plots, checksums and replay results.
`search_index.zip` is separate and contains automatic transcripts and numeric
embeddings, model/settings identity, safe reconstruction code and audio
reacquisition information. Reference text and qrels must not enter this search
artifact. Both exclude model weights and source audio.

Fresh-process verification replays the first four queries, re-embeds the first
four documents and retranscribes three clips. The four are exactly the first
embedding batch of the original index and evaluation calls: fp16 embeddings
depend on batch composition (left-padding length), so replaying a smaller batch
would be a different computation, not a reconstruction. Rankings must match exactly; numeric score/vector
tolerances are `atol=1e-5, rtol=1e-4`. No tolerance is widened automatically.

Optional BYOD is disabled by default. Supply an authorized local JSON manifest
and 1–120 mono 16 kHz WAV/FLAC recordings, each 2–25 seconds and at most 4 MB:

```json
{
  "rights_confirmed": true,
  "source_notes": "Describe the source and permission to use these recordings.",
  "probe_query": "Ano ang paksa ng talakayan?",
  "records": [
    {"doc_id": "clip-1", "path": "audio/clip-1.wav", "reference": "Optional verified transcript"}
  ]
}
```

Paths are relative to the manifest directory and cannot escape it. Omit
`reference` if unavailable. This path performs the same ASR, indexing, search,
export and fresh-process reconstruction in a separate output directory,
`byod/<run id>/`:

- `search/` and `search_index.zip`: a search package in the **same
  `dimer_audio_archive` v1 format** as the default index, verified by the same
  `verify_package` code and searched by the same `search_archive.py` consumer.
  It holds automatic transcripts, embeddings, audio identities (relative path and
  SHA-256), model identity, the lock and `RECONSTRUCT.md`; no references,
  queries, qrels or evaluation records.
- `evaluation.json`: kept outside the search package; holds supplied references
  and ASR metrics.
- `probe.json`, `build_receipt.json`, `verification.json`: the fresh-process
  reconstruction check, which needs the original manifest and audio.

In the notebook, a separate query cell searches the saved BYOD package with
`audio_runtime.py --query ... --index <package>` without re-running ASR, shows
top-five transcript evidence and plays only local files whose SHA-256 matches the
package. The package can be moved to a clean directory and searched with its own
`search_archive.py`. It reports ASR metrics only for supplied references. BYOD v1 does not import
reviewed relevance labels; retrieval evaluation is explicitly `not_measurable`.
The replay probe is a mechanical check, never an accuracy score.

## Review fixes (2026-09-28)

The Notebook Review Framework v1 review of PR #10 at `5b9f2ab` found one major
and four minor issues. Fixes, all made in the generator or carried modules:

| Finding | Change |
| --- | --- |
| F1 BYOD search/reuse | Shared `write_search_package` / `verify_package` / `search_archive.py` for default and BYOD packages; `--index` selects a package; BYOD query cell with transcript evidence and hash-checked playback; `evaluation.json` moved out of the search package; BYOD `search_index.zip`. |
| F2 latency cohort | Paired depth-10/20 timing on the same development queries (see Experiment). |
| F3 competing evidence | Failure tracing shows dense candidates with transcript excerpts, cosine and reranker scores and ranks for one candidate miss and one reranker demotion, or states that a category did not occur; optional competitor playback. |
| F4 reviewer declaration | `independent_review: true` now requires at least two distinct reviewer IDs; the one-reviewer route still needs its disclosed limitation. |
| F5 collapse | The embedded-source carrier cell starts collapsed in Colab (form view) and Jupyter. |

User-visible contract changes: the default search package manifest gains a
`source` field; search results (runtime and exported consumer) gain
`audio_path` and `audio_sha256`; `interactive_search.json` records the searched
package; the BYOD `dimer_byod_audio_search_v1` bundle format is replaced by the
`byod/<run id>/` layout above; `independent_review: true` with one reviewer is
refused.

Offline verification: the regenerated notebook's own code cells were executed
in order with synthetic constant-valued audio (not speech) and deterministic
stand-in models (not pretrained inference); the T4 preflight and locked install
were skipped. Default outputs (rankings, scores, per-query CSV, paired
comparisons, transcripts, embeddings) were byte-identical to the reviewed
notebook under the same stand-ins, and unchanged when every optional toggle was
enabled. BYOD with and without references, a repeat query without rebuilding,
hash-checked playback of only BYOD recordings, clean-directory search by the
exported consumer in a fresh process, and refusal of modified embeddings,
identifiers and model settings were observed. None of this is hosted-runtime,
pretrained-model or relevance evidence.

## Hosted runs

| Date (UTC) | Commit / notebook blob | Result |
| --- | --- | --- |
| 2026-09-28 | `ad42d18` (`main`) / `47da982901db` | **PASSED on Colab Tesla T4** (maintainer-supplied). Default `Run all`, toggles off, 18/18 cells. Reload passed; stage-output digests are byte-identical to the Kaggle run of `5664f61`. See [release verification](release-verification.md#maintainer-supplied-colab-execution-of-ad42d18--2026-09-28). |
| 2026-09-28 | `5664f61` / `47da982901db` | **PASSED on Kaggle Tesla T4.** Blob-verified serial test-suite execution, 21/21 cells, with search, competitor playback and BYOD on plus harness cells for BYOD re-query, export re-run, a no-reference build and clean-directory reuse. Reload passed (4 queries, 4 documents, 3 clips replayed; tolerances unchanged); metrics equal the `d58ab28` run; peak allocated GPU 1.61 GiB. BYOD used FLEURS train clips as stand-ins. See [release verification](release-verification.md#kaggle-execution-of-5664f61--2026-09-28). |
| 2026-09-28 | `d58ab28` / `284adf543e21` | **FAILED at reload.** Default Run all on a fresh Colab Tesla T4 passed `prepare`, `asr`, `index`, `evaluate` and `activity`; `reload` refused because 3 re-embedded documents missed `atol=1e-5` by up to 4.2e-4 against an index embedded in batches of 4. Fixed in the next commit by replaying the original first batch; tolerances unchanged. See [release verification](release-verification.md#filipino-audio-archive-search-capstone). |

## Release evidence still required

Local validation on 2026-09-27: the pre-change suite had **53 passed, 1 skipped**.
The integrated suite has **141 passed, 5 skipped** in the lightweight CPU
environment. The 18 adapter tests also pass separately in an existing CPU
PyTorch/Transformers environment, including tiny random Whisper generation;
this does not load pretrained weights or establish model quality. Ruff, notebook
schema/Python parsing, source-generator parity and release-asset validation pass.
The two original tutorial notebooks remain byte-identical to their generators.
Actual source Parquet hashes and all 120 selected audio files were reverified.

The integration tests use deterministic fake models to exercise all seven CLI
stages, reruns, stage-receipt invalidation, index export/replay, reference/qrel
noninterference and artifact-tamper refusal. Those synthetic outputs are test
fixtures and are not stored as notebook outputs or scientific results.

1. Filipino-language query review, full relevance annotation, independent review
   or disclosed single-reviewer limitation, and a frozen review record.
2. Fresh Colab T4 default Run all against the exact notebook bytes, retaining the
   executed notebook, all six real-model outputs and complete evidence ZIP.
3. Real-model BYOD execution (with and without references, a repeat query and
   clean-directory reuse), artifact replay and resource measurements.

Model superiority is not required. Do not claim production readiness, verified
speaker independence, real meeting performance, or a causal relationship between
WER and search errors. Source overlap with model pretraining is unknown.

See [data attribution and AI assistance disclosure](audio-data-license.md).
