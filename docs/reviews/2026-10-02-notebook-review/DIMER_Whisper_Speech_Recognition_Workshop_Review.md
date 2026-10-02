# DIMER Speech Recognition and Transcription Notebook — Review

**Verdict: Needs revision**  
**Review date:** 2 October 2026  
**Repository:** `kurtvalcorza/whisper-asr-pipeline`  
**Notebook:** `tutorials/DIMER_Whisper_Speech_Recognition_Workshop.ipynb`  
**Reviewed commit:** `e81a265a259ec0d416f949f7add7fbb3939a5b55` (`main`, confirmed against the GitHub API at review time)  
**Notebook Git blob:** `56749100397df879cea2ad4804bb3523ff3e3eac`  
**Finding prefix:** `ASR`

## Executive assessment

The notebook is a carefully engineered, self-contained lesson. Its carried source is hash-checked, the
runtime is isolated, the default path ran cleanly on a fresh Colab T4 for this exact notebook blob, and the
metric, noise and export code behaves as described on the paths probed here. Its scope statements are
unusually honest.

Three problems keep it from delivering its central lesson reliably:

1. The "references" that every WER/CER is measured against are **another commercial speech recognizer's
   1-best output** (Google ASR, per the MINDS-14 paper), and the notebook never says so. A lesson titled
   *When can we trust an automatically generated transcript?* scores one automatic transcript against another
   and presents the disagreement as Whisper's word errors.
2. The only worked alignment example (objective 2) explains a pairing that the printed alignment contradicts.
3. Unlabelled bring-your-own-audio output loses the file names, so transcripts of a multi-clip ZIP cannot be
   mapped back to their recordings (`OUT4`).

No Blocker was found. The default journey is backed by documented hosted execution; the exploration and BYOD
journeys have not been run on a hosted GPU by anyone.

## 1. Review contract and evidence

| Item | Scope |
|---|---|
| Declared profile / mode / spec | `TASK-INFERENCE` / `WORKSHOP` / NOTEBOOK_SPEC `2.2` (`metadata.dimer`) |
| Specification baseline | `ml-worker` `origin/main` `b1cfe13`, `NOTEBOOK_SPEC.md` 2.2 (2026-09-26), blob `7428d5be` |
| Design document | `docs/speech-recognition-workshop-spec.md` (repo's own acceptance criteria ACC-01..12) |
| Intended learner | Level 2 Applied Tasks; basic Python and Colab; no prior ML or audio experience |
| Supported runtime | Fresh Colab T4 GPU; isolated `uv` CPython 3.12.12 environment from a carried hash lock |
| Default task | English transcription with `openai/whisper-large-v3-turbo` @ `41f01f3f`, greedy, batch 1, float16 |
| Default data | MINDS-14 `en-US` parquet @ `40ce77cb`; 80 frozen recordings (64 evaluation / 16 activity) |
| Promised outcomes | Listen/validate audio; explain S/D/I with an alignment; interpret corpus WER/CER vs an empty baseline; paired clean vs 10 dB noise comparison; export a report; evidence-based conclusion; optional 5/20 dB exploration; optional BYOD |
| Teaching mode | Self-paced; predict → run → what to notice → collapsible worked interpretation |
| Release status | Candidate |

Scope: every markdown and code cell (20 cells), the six carried files (`whisper_reference.py`,
`workshop.py`, `sample.json`, `requirements.txt`, model manifest, `source.json`), the cell and notebook
metadata, the troubleshooting table and the exported report contract.

### Evidence actually obtained

| Basis | What it covers |
|---|---|
| **Source inspection** | All 20 cells and all carried files at `e81a265`; the design spec; `tools/validate_release_assets.py` workshop checks |
| **Documented execution evidence** | `docs/release-verification.md`: a maintainer-supplied fresh Colab T4 `Run all` of commit `aca0e06`, notebook blob `56749100397d` — **the same blob reviewed here** — PASS, 173 s, all 10 code cells, evaluate WER 0.1981 / CER 0.1646 (S 56, D 5, I 122, N 924), activity clean 0.1850 → noisy 0.2081, paired 0 improved / 12 unchanged / 4 worsened. The executed `.ipynb` itself is not stored under `docs/execution-evidence/`; this review read the record, not the run. Exploration and BYOD: **not exercised** (the record says so) |
| **Direct execution (CPU only, no model)** | `run_probes.py` (in the probe ZIP) extracts the carried files from the notebook and runs the carried `workshop.py` metric, noise, BYOD-validation and export code on synthetic audio in Python 3.12.12 on Windows. No Whisper inference, no GPU, no network. Results in `results.json`; source digests in `source_manifest.json` |
| **External source** | Gerz et al. 2021, *Multilingual and Cross-Lingual Intent Detection from Spoken Data* (arXiv 2104.08524), §4 "Speech Transcription" — read for ASR-M1 |
| **Learner observation** | None. Statements about learner confusion are reviewer judgment, not measured effects |

Repo checks at the reviewed commit (py3.12.12, CI's dependency set without torch, `PYTHONPATH=src`):
pytest 158 passed / 5 skipped (all 5 need torch); `ruff check src tests tools` clean; validator and all three
`--check` builds pass. These are static checks, not execution evidence.

### Journeys

| Journey | Basis | Result |
|---|---|---|
| First-time learner | Source inspection | Mostly well guided; ASR-M1, ASR-M2, ASR-m2, ASR-m4, ASR-m5 apply |
| Clean default | Documented execution (same blob) | **Passed** (2026-09-27 Colab T4 record) |
| Active learning (5/20 dB exploration) | Source inspection + CPU probe of `noise_pair` | Code path reaches the computation, writes a separate directory, shared gain verified invariant across 5/10/20 dB (P03). Hosted run: **not verified** |
| Reuse and recovery (BYOD) | Source inspection + CPU probe of `load_byod` | Validation works and rejects invalid input; ASR-M3 and ASR-m1 apply. Hosted labelled/unlabelled BYOD run: **not verified** (`REL12`) |

## 2. Separate judgments

- **Technical correctness:** good. Integrity checks, bounded decoding, deterministic noise with a shared gain,
  strict-JSON reload-verified exports and fail-stop diagnostics all hold up under the probes. Defects: BYOD
  rows lose their file identity (ASR-M3), BYOD refusals don't name the offending member (ASR-m1), and a stale
  `workshop.py` digest sits in the carrier cell's metadata (ASR-m3).
- **Promise fulfilment:** the default promises are met and hosted-verified. The exploration and BYOD promises
  are implemented but not yet hosted-verified.
- **Scientific validity:** the WER/CER computation is correct, but what it measures is misdescribed:
  agreement with a second ASR system's output, not error against a human transcript (ASR-M1). The SNR is
  computed over whole-recording power, while the prose says speech power (ASR-m2).
- **Learner experience:** strong structure (prediction prompts, what-to-notice notes, worked
  interpretations, a conclusion template). The alignment worked answer contradicts the printed output
  (ASR-M2). There is no glossary for about a dozen new terms (ASR-m4). The learner is told to inspect the
  worst errors but can only hear the first clip (ASR-m5).
- **Specification conformance:** unresolved applicable `MUST`s are `OUT4` (ASR-M3), `DAT19` (ASR-m1) and
  `REL12` (BYOD not release-verified). `EVAL3` ("principal metrics MUST be explained in terms of what they
  measure") is weakened by ASR-M1. `SHOULD` gaps: `GDL6` (ASR-m4) and `UX10` (ASR-m1).

## 3. Findings

### ASR-M1 — Major: the references are Google ASR output, and the notebook never says so

- **Cell/section:** md-05 (§2 data), md-07, md-09/md-11 (metric interpretation), md-19 (limitations), References; carried `workshop.py` `export_report` summary.
- **Observed issue:** MINDS-14's `transcription` field is not a human transcript. The dataset paper states that
  for all recordings the authors ran the Google ASR model to obtain written transcriptions, and they work with
  the 1-best hypothesis (Gerz et al. 2021, §4). The notebook calls these "supplied reference" text, adds only "the
  reference itself may contain errors", and labels the disagreement *Whisper's* WER/CER. The frozen sample
  shows the consequence. Evaluation reference 4 reads "hello I just ate only lost my card at the airport will
  you be able to stop listen to increase the card" (P02). The hosted run's errors are insertion-dominated
  (I 122 of 183 edits; D 5): Whisper "inserting" words the reference ASR dropped is the likely explanation, but
  that is *inferred* — no hypotheses are stored to confirm it.
- **Consequence:** the central question is *when can we trust an automatically generated transcript?* As
  shipped, the notebook teaches the learner to treat one automatic transcript as ground truth for another. A
  learner who follows the worked interpretations will attribute the reference system's omissions to Whisper,
  and will read "WER 0.198" as Whisper's error rate. Neither conclusion is supported. This misleads on the
  lesson's core idea (dimension 3, scientific validity; `EVAL3`).
- **Evidence:** source inspection (no cell mentions how the references were produced); external source (arXiv
  2104.08524 §4); direct execution P02 (`notebook_discloses_references_are_asr_output: false`); documented
  execution (S/D/I split).
- **Recommended correction:** say plainly, where the data is introduced and again where WER is interpreted,
  that the references are 1-best output of a commercial ASR system (Google) collected by the dataset authors.
  Reframe the score as *word disagreement with the reference transcript*, which counts errors made by either
  system. Give the learner a way to arbitrate: listen to the worst-agreement example and decide which
  transcript is right (see ASR-m5). Carry the same caveat into the exported `summary.md` and into the
  Limitations and References sections. A human-transcribed corpus is the larger alternative; per the design spec
  (§12), replacing the dataset is the maintainer's decision.
- **Acceptance check:** (a) md-05 or md-07 names the reference source as Google ASR / commercial ASR 1-best
  output with the citation; (b) the WER interpretation (md-11 worked interpretation) states that a disagreement
  can be an error in either transcript, and that a lower rate means closer agreement with that system's
  output; (c) the troubleshooting/limitations section and References repeat it; (d) the exported `summary.md`
  text produced by `export_report` names the reference provenance for the canonical dataset; (e) the default
  computation is unchanged (same S/D/I for identical inputs).

### ASR-M2 — Major: the worked alignment answer contradicts the alignment the notebook prints

- **Cell/section:** md-07 (prediction prompt), code-08 (metric demo), md-09 (worked interpretation).
- **Observed issue:** for reference "pay my bill" and prediction "pay the bills now", the worked interpretation
  says "two substitutions (my→the, bill→bills) and one insertion (now)". The carried `alignment()` backtrace
  prefers substitution at the end. It prints `I(the), S(my→bills), S(bill→now)` (P01). The counts (S 2, I 1,
  WER 1.0) match, but the printed pairing doesn't, and the notebook never explains that several minimal
  alignments can exist.
- **Consequence:** objective 2 (*explain substitutions, deletions and insertions using one aligned example*) has
  exactly one worked example. The learner who compares their answer with the output sees a nonsensical
  pairing ("my"→"bills") presented as the result, and a worked answer that disagrees with it. That is
  material confusion on a core objective. Every per-clip alignment later in the notebook uses the same
  tie-break.
- **Evidence:** direct execution P01; source inspection of md-09 and `alignment()`'s docstring ("ties prefer
  match/substitution, deletion, insertion", applied from the end).
- **Recommended correction:** keep the scoring contract (the counts and WER are correct and exported). Explain
  in the worked interpretation that the edit count is unique but the pairing is not, that "my→the, bill→bills,
  +now" and the printed "+the, my→bills, bill→now" are both minimal three-edit alignments, and that the code
  breaks ties deterministically from the end of the sentence. Show the learner how to read the printed `op`
  list, so the demo output and the worked answer agree.
- **Acceptance check:** the md-09 worked interpretation (i) states the edit counts S 2, D 0, I 1 and WER 1;
  (ii) reproduces the exact pairing the code prints for this pair; (iii) explains that more than one minimal
  alignment exists and the counts, not the pairing, determine WER. The printed alignment for the demo pair is
  unchanged.

### ASR-M3 — Major: BYOD outputs lose the file name, so unlabelled transcripts can't be mapped back (`OUT4`, `INF4`)

- **Cell/section:** md-17, code-18; carried `workshop.py` `load_byod` and `run_stage` (BYOD branch).
- **Observed issue:** without `transcripts.csv`, `load_byod` assigns IDs `"0".."n-1"` over sorted file names
  and then drops the file name. Neither the records, `input_manifest.json`, `transcripts.json` nor
  `transcripts.csv` contains the file (P04: a ZIP of `zeta_call.wav` and `alpha_call.wav` yields IDs `0`, `1`
  and no file field). The learner is never told about the sort order. With a CSV, the learner's own `id`
  survives but `file` is still dropped.
- **Consequence:** the promised "bring your own audio" path produces transcripts the learner can't reliably
  attribute to recordings. That defeats transfer (dimension 8) and violates `OUT4` ("outputs MUST retain
  identifiers mapping outputs back to relevant rows…").
- **Evidence:** direct execution P04; source inspection.
- **Recommended correction:** carry the archive member name (or the single file's name) into each BYOD record
  and row as `file`, show it in the BYOD display, and leave the canonical stages' rows unchanged.
- **Acceptance check:** for an unlabelled two-file ZIP, every BYOD record and every row written to
  `transcripts.json` / `transcripts.csv` carries `file` equal to its archive member name; labelled CSV input
  keeps both `id` and `file`; evaluate/activity rows gain no new column.

### ASR-m1 — Minor (spec `MUST` `DAT19`, `UX10`): BYOD refusals don't name the offending clip or member

- **Cell/section:** carried `workshop.py` `load_byod` / `decode_audio`; md-19 ("Check the named duration, format, schema or path issue").
- **Observed issue:** a 0.4 s clip inside a three-clip ZIP is refused with "Expected 1–30 seconds; never
  silently cropped", which doesn't say which clip. An archive with `notes.txt` and a `__MACOSX/` entry is
  refused with "ZIP has unsupported or undeclared files", which doesn't say which files (P05). The duplicate
  check has the same gap.
- **Consequence:** for a 20-clip archive the learner has to bisect by hand. The troubleshooting table promises
  a *named* issue.
- **Evidence:** direct execution P05.
- **Recommended correction:** prefix per-clip decode and duplicate errors with the member name, and list the
  undeclared members (bounded), with the corrective action.
- **Acceptance check:** the short-clip refusal message contains `short.wav`; the undeclared-file refusal
  contains `notes.txt` (or the `__MACOSX` entry); a duplicate refusal names both members.

### ASR-m2 — Minor: "SNR compares speech power to noise power", but the code uses whole-recording power

- **Cell/section:** md-11 (§5); carried `noise_pair`.
- **Observed issue:** `noise_pair` scales noise to the RMS of the whole clip, including pauses and leading
  silence. P03 uses a clip that is half silent: the speech portion then sits about 3 dB *above* the stated
  target (7.995 dB at 5 dB, 12.995 at 10, 22.995 at 20). The construction is valid and its exported
  `actual_snr` is the whole-clip value. Only the prose is imprecise.
- **Consequence:** the learner may read 10 dB as the speech-to-noise ratio during speech. For clips with long
  pauses, the effective noise during speech is milder than stated.
- **Evidence:** direct execution P03; source inspection.
- **Recommended correction:** say the SNR is computed over the whole recording, pauses included, so the speech
  portions of clips with long pauses experience a somewhat higher SNR.
- **Acceptance check:** md-11 states that SNR is measured over the whole recording (including pauses); the
  noise construction is unchanged.

### ASR-m3 — Minor: the carrier cell's metadata records a stale `workshop.py` digest

- **Cell/section:** code-03 `metadata.dimer.sources`.
- **Observed issue:** the cell metadata gives `workshop.py` as `009a5509…`, which is the failed draft recorded
  in the release-verification table. `CARRIED_HASHES`, `source.json` and `metadata.dimer.generated_from` all
  give `6763e433…` (P06). The validator doesn't compare the cell metadata.
- **Consequence:** provenance inside the artifact disagrees with itself. Anyone auditing by cell metadata
  identifies the wrong (failed) source.
- **Evidence:** direct execution P06.
- **Recommended correction:** make the cell metadata equal `generated_from.sources`, and have the validator
  check it.
- **Acceptance check:** code-03 `metadata.dimer.sources` equals `metadata.dimer.generated_from.sources`, and
  the validator rejects a mismatch.

### ASR-m4 — Minor (`GDL6` `SHOULD`; design spec §4): no glossary

- **Cell/section:** whole notebook.
- **Observed issue:** WER, CER, substitution/deletion/insertion, alignment, normalization, sample rate,
  resampling, mono, SNR, dB, real-time factor, token ceiling, greedy decoding, reference and BYOD are each
  introduced once, in different sections. There is no glossary, although the repository's own design spec says
  to "collect them in a short glossary" (P07).
- **Consequence:** the declared audience (no prior ML or audio experience) has no place to re-find a term when
  it recurs in a later section or in the exported report.
- **Recommended correction:** add a short glossary section with one-line definitions consistent with the code.
- **Acceptance check:** a markdown cell titled "Glossary" defines at least WER, CER, S/D/I, alignment,
  normalization, sample rate, SNR, real-time factor, token ceiling and reference transcript, without
  contradicting the code.

### ASR-m5 — Minor: the learner is asked to inspect the best and worst errors but can only hear the first clip

- **Cell/section:** md-11, code-04 `show_results`, code-10/code-12.
- **Observed issue:** `show_results` prints the lowest- and highest-WER examples as text, but only the first
  recording (`listen_*.wav`) can be played (P08). The design spec's stage 8 says "listen to successes and
  failures".
- **Consequence:** the learner can't check the worst example by ear, which is the one move that would show
  whether the reference or the prediction is wrong (ASR-M1).
- **Recommended correction:** play the prepared audio for the selected lowest- and highest-WER examples (the
  clean prepared waveform; for a noisy row, say that the clean source is played).
- **Acceptance check:** `show_results` offers an audio player for the highest-WER example (and the
  lowest-WER one) of the evaluate and activity stages, resolved through `prepared.json` by `id`, and handles a
  BYOD stage without error.

### Suggestions (optional, not release requirements)

- **ASR-S1:** render the paired improved/unchanged/worsened counts as a small table instead of inside the
  "Coverage and timing" JSON dump.
- **ASR-S2:** in the exploration branch, print the canonical 10 dB rates next to the 5/20 dB rates, so the
  learner compares the conditions directly.
- **ASR-S3:** store the 2026-09-27 executed Colab notebook byte for byte under `docs/execution-evidence/`, as is
  done for the capstone, so the default-run record can be re-inspected.

## 4. Positive findings and non-findings

- The shared-gain noise construction is invariant across 5/10/20 dB and never clips; the achieved SNR equals
  the target to 1e-4 (P03).
- The 563 / 533 / 80 recordings, 64/16 roles, 597 s total and 1.7–18.4 s per clip claimed in md-05 match the
  frozen manifest (P02).
- The report export round-trips JSON and CSV, re-aggregates the metrics and verifies the ZIP inventory on
  synthetic rows (P09).
- BYOD validation rejects traversal, symlinks, collisions, undeclared files, oversized members, mixed labels
  and out-of-range clips before any model loads (source inspection; P05 shows refusals happen pre-model).
- The default path needs no token, upload or edit. HF tokens are removed from the stage environment.

## 5. Promise-to-evidence matrix

| Claim | Implementation | Observable result | Learner interpretation | Status |
|---|---|---|---|---|
| 80 frozen recordings, 64/16, none cropped | `sample_records`, `sample.json` | prepare log counts | md-05/07 | Met (P02 + hosted run) |
| WER/CER with S/D/I alignment | `score`/`alignment` | code-08 JSON | md-09 | Counts correct; worked pairing contradicts output (ASR-M2) |
| Whisper vs empty baseline | `aggregate`, `empty_transcript_baseline` | code-10 table | md-11 | Executed; meaning of "error" misdescribed (ASR-M1) |
| Paired 10 dB noise comparison | `noise_pair`, activity stage | code-12 table + CSV | md-13 | Met (hosted); SNR prose imprecise (ASR-m2) |
| Optional 5/20 dB exploration, separate directory | `explore` stage | code-16 | md-15 | Source-verified; hosted not verified |
| BYOD transcribe/evaluate/export | `load_byod`, `byod` stage | code-18 | md-17 | Validation probed; file identity lost (ASR-M3); hosted not verified |
| Reload-verified report ZIP | `export_report` | code-14 | md-13 | Met (P09 + hosted) |

| Objective | Learner activity | Evidence it was exercised |
|---|---|---|
| Identify waveform/sample rate/transcript/reference | code-06 playback + metadata | Default run output |
| Explain S/D/I with one aligned example | md-07 prediction + code-08 | Undermined by ASR-M2 |
| Interpret corpus WER/CER, not as confidence | md-11 worked interpretation | Present; ASR-M1 changes what it means |
| Predict and measure one controlled noise change | md-11 prediction + code-12 | Default run output |
| Distinguish execution, agreement and reliability | md-15 self-check + template | Present (prose) |
| Apply to new audio | BYOD | Not hosted-verified; ASR-M3 |

## 6. Readiness

**Needs revision.** Open Majors: ASR-M1, ASR-M2 and ASR-M3. Unresolved applicable `MUST`s: `OUT4`,
`DAT19`, `REL12`. Remaining gates after the fixes: a hosted Colab T4 run on the fix head that covers the
default path, the 5 or 20 dB exploration, and BYOD with labelled, unlabelled and one rejected input. The
status stays Candidate.

## 7. Verified versus inferred

- **Verified:** P01–P09 by direct CPU execution of the carried source; the default run by a documented hosted
  record for this exact notebook blob; that the references are Google ASR output, from the dataset paper.
- **Inferred:** that the insertion-dominated S/D/I split comes from omissions in the reference ASR (no stored
  hypotheses to confirm); every statement about how a learner will react (no learner observation).
- **Most likely to be wrong:** ASR-M2's severity. A reader could treat a correct-count example with a
  different-but-valid pairing as Minor; I rated it Major because it is the only worked example for a stated
  objective.
