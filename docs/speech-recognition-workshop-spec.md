# DIMER Speech Recognition and Transcription — Notebook Specification

Status: **Draft for implementation; no new notebook or runtime qualification is claimed.**

Prepared: 2026-09-27. Owner: maintainer of `kurtvalcorza/whisper-asr-pipeline`.

## 1. Product decision

Build **DIMER Notebook: Speech Recognition and Transcription with Whisper**, a self-paced Level 2 Applied Tasks unit immediately after OCR and Structured Document Extraction.

The main question is: **When can we trust an automatically generated transcript?**

The curriculum currently covers text, tables, images, documents, and scientific grids but has no dedicated speech unit. Learners should listen, predict likely errors, inspect transcripts, measure agreement against references, and explain the limits of that measurement.

| Field | Decision |
|---|---|
| Proposed learning artifact | `tutorials/DIMER_Whisper_Speech_Recognition_Workshop.ipynb` |
| Learner-facing title | DIMER Notebook: Speech Recognition and Transcription with Whisper |
| Profile / mode | `TASK-INFERENCE` / `WORKSHOP`; implements the guided layer for independent use |
| Normative basis | DIMER NOTEBOOK_SPEC 2.2; applicable fleet model, data, artifact and release contracts |
| Default model | `openai/whisper-large-v3-turbo`, revision `41f01f3fe87f28c78e2fbf8b568835947dd65ed9` |
| Default task / language | `transcribe` / English (`en`); no language auto-detection experiment in v1 |
| Default hardware | Fresh Colab T4-class GPU runtime; fail early and clearly if absent |
| Default data | Bounded, pinned English MINDS-14 sample; exact selection and bytes frozen before release |
| Learner requirements | Basic Python and Colab familiarity; no prior ML or audio-processing knowledge |
| Main output | Raw transcripts, error alignments, aggregate metrics, paired noise results, and a reproducible report |
| Adaptation | Separate optional continuation through the existing `whisper_asr_finetune_colab.ipynb` after the upgrades in §10 |

The core notebook is complete without running another notebook. Do not label the core `E2E`: training is not part of its default workflow. Conversely, the E2E continuation must actually train, export, and reload on its own default Run all path (RUN7). Do not hide training behind a default-off flag and still claim E2E conformance.

Proposed catalogue description:

> Explore automatic speech recognition with Whisper, compare transcripts against reference text, measure word and character errors, and investigate how background noise affects recognition. An optional continuation introduces domain adaptation and verified adapter reload.

## 2. Source baseline and verified constraints

Repository main was checked at `1eb99622614d547170675b24c051510168a694da`. Existing generated notebooks are `whisper_asr_colab.ipynb` and `whisper_asr_finetune_colab.ipynb`; neither is the proposed learning artifact.

Confirmed from source:

- Both existing tutorials carry pipeline code and model manifests. Reuse their acquisition, validation, transcription, safetensors adapter, and reload logic; maintain generator parity rather than hand-editing generated notebooks.
- `pyproject.toml` requires Python `>=3.12,<3.13`. Existing pins include torch 2.6.0, torchvision 0.21.0, torchaudio 2.6.0, transformers 4.52.1, NumPy 1.26.4, soundfile 0.13.1, and accelerate 1.3.0; tutorial extras include datasets 4.4.0 and pandas 2.2.3. These are reuse candidates, not a new Colab compatibility claim.
- Existing fine-tuning uses MINDS-14 `en-US` at revision `40ce77cb32a384e4d50a568e1ec39ac804019d33`, 400 train and 100 held-out clips, and LoRA rank 32. Replace the two-role evaluation design before presenting the continuation as validation-selected adaptation.
- The existing WER helper returns edit counts but not a full substitution/deletion/insertion alignment. The instructional metric layer needs an independently tested alignment implementation and explicit empty-reference behavior.
- Existing source exposes `translate` and some documentation advertises it. The upstream Whisper README explicitly says Turbo is not trained for translation. The new notebook must expose only transcription and explain this capability boundary; do not infer support from an API accepting a task string.
- Existing release records concern specific earlier notebook blobs. Their measured timings and results must not be reused as qualification of this new notebook.

The inspected NOTEBOOK_SPEC 2.2 content has Git blob `046d866eac7cc67b1539a4ba370ebc965341c87b` in `ml-worker` at `ce276e10197a1b9f1c6ebe72c8effc730f761d34`.

## 3. Outcomes and scope

Participants should be able to:

1. Identify waveform, sample rate, transcript, reference, and prediction in **Audio → Whisper → Transcript**.
2. Explain substitutions, deletions, and insertions using one aligned example.
3. Interpret corpus WER and CER without treating either as a calibrated confidence score.
4. Predict and measure the effect of one controlled noise change on the same recordings.
5. Distinguish successful execution, agreement with supplied references, and practical transcription reliability.
6. Explain one plausible application and what additional evidence it requires.

Out of scope for v1: translation, speaker identification/diarization, live microphone capture, streaming, timestamp accuracy, long-recording chunking, clinical/legal transcription guarantees, a cross-language benchmark, and a second ASR model. These require different data, evaluation, or runtime contracts. A Whisper-only comparison of conditions is intentional; do not call this a multi-model benchmark.

## 4. Canonical learner flow

Every principal experiment includes a prediction prompt before execution, an expected-output description, a “what to notice” note, and a collapsible worked interpretation. Answers remain optional prose; they never block Run all.

| Stage | Learner-facing content | Executable result / checkpoint |
|---|---|---|
| 1. Orient | Task, prerequisites, roadmap, runtime, limitations, AI disclosure | Controls with working defaults; no edit required |
| 2. Audio basics | Listen to an activity clip; explain seconds, samples, sample rate and channels | Audio player, duration and waveform; no prediction shown yet |
| 3. Predict | Identify a word or phrase likely to be difficult | Optional personal note, not a graded response |
| 4. Prepare | Explain resampling versus changing playback speed | Validated dataset manifest and role counts |
| 5. Baselines and metrics | Work one short reference/hypothesis alignment by hand | Correct S/D/I, WER, CER and empty-transcript baseline |
| 6. Freeze | Explain what is fixed before final evaluation | Persist model, data, normalization, decoding and role identities |
| 7. Transcribe | Run the pinned frozen model on the clean evaluation set | Raw transcripts and complete per-clip evaluation records |
| 8. Interpret | Compare model with the trivial baseline; listen to successes and failures | Corpus table, error alignment examples, timings and explicit coverage |
| 9. Controlled noise | Predict the effect of adding noise to the disjoint activity set | Paired clean/noisy outputs at fixed 10 dB SNR |
| 10. Try one change | Optional 5 or 20 dB comparison using the same activity recordings and noise realization | Separate exploratory results; canonical exports unchanged |
| 11. New input / BYOD | Optional validated audio, with or without references | Transcripts; measured results only where appropriate references exist |
| 12. Conclude and export | Evidence-based conclusion and practical application | Summary, manifest, metrics, CSVs and report ZIP |
| 13. Continue | Explain when adaptation may help and how it can overfit | Link to separately qualified LoRA continuation; no dependency on it |

Infrastructure cells—runtime setup, carried modules, manifests and runners—must be labelled and collapsed. Keep the audio, prediction, metric, comparison and interpretation cells visible. Explain terms at first use and collect them in a short glossary.

## 5. Default data and split contract

**DATA-01.** Use `PolyAI/minds14`, locale `en-US`, at the existing immutable revision above, subject to verifying anonymous availability and a bounded fetch mechanism. Retain CC-BY-4.0 attribution and record that supplied transcriptions can be noisy. Agreement with these references is not authoritative transcription accuracy.

**DATA-02.** Proposed release sample: **80 eligible recordings**, with **64 clean evaluation** and **16 activity** recordings, selected deterministically with seed 42 from stable IDs. Eligibility: 1–20 seconds, decodable, finite audio, nonempty normalized English reference. Freeze exact IDs and counts in a manifest. If filtering leaves too few examples, stop the build and revise the documented sample—do not quietly substitute or resample at runtime.

**DATA-03.** Pin raw files or archive/parquet members by immutable location, SHA-256 and byte count. Include original IDs, license, reference text, input rate/channels/duration, decoded audio hash, eligibility exclusions, and role digest. Do not execute downloaded dataset scripts or fetch DIMER Python source at runtime.

**DATA-04.** Decode WAV/FLAC with soundfile and explicit bounded allocation; cast dataset audio to `decode=False` when using datasets. Reject invalid/oversized headers before allocating a full waveform. Convert mono/stereo deterministically to mono float32 and resample to 16 kHz. Record changes. Never silently crop overlong speech or truncate references.

**DATA-05.** Keep all copies, noise variants and any segments of a source recording in one role. Reject exact decoded-audio duplicates across roles. If reliable speaker/session IDs exist, group by them; otherwise state that the split is recording-disjoint, not speaker-disjoint. Do not infer speaker identity from filenames or claim generalization to unseen speakers.

**DATA-06.** The 16 activity clips are reusable for learner exploration; final evaluation clips are not used to choose noise severity, decoding settings or normalization. Display a fixed set of examples plus clearly labelled best/worst-error examples; retain every evaluated recording in the exported table.

## 6. Runtime, model and credential contract

**RUN-01.** Default Run all completes on a fresh selected T4 runtime without manual restart, upload dialog, token prompt, configuration edit, repository clone, DIMER service, or separate notebook. Public model/data downloads must succeed without credentials. An optional `HF_TOKEN` Colab secret may support authorized Hub access but must not become a default-path requirement, and must never enter outputs or provenance.

**RUN-02.** Use an isolated Python 3.12 worker environment launched from the notebook. Carry the worker source inside the notebook; materialize it locally and invoke it by stage. Keep the host kernel responsible for lightweight controls, displaying audio/results, and subprocess orchestration. Do not downgrade preloaded Colab NumPy/torch or delete modules from `sys.modules`.

**RUN-03.** The build must freeze the Python patch version, bootstrap tool/version/integrity mechanism and complete dependency lock. Start with the existing compatible 3.12 pins, then qualify the exact environment. Runtime fetches may acquire public dependencies and interpreter assets, not DIMER application source. The worker's input/output protocol must be versioned and JSON/CSV based, with explicit nonzero-exit handling and concise diagnostic tails. Expose real audio examples and intermediate results after each stage.

**RUN-04.** Verify every model snapshot member against the existing carried manifest before loading. Load from the verified local snapshot with remote code disabled. Pin generation settings: transcription, `en`, greedy decoding, no sampling or previous-transcript prompt, and an explicit generation token ceiling within model limits. Record the effective generation config. Detect token-ceiling termination and report it; retain affected rows rather than silently excluding failures.

**RUN-05.** Batch size 1 is the initial memory-safe reference. Float16 on the T4 is the proposed canonical compute dtype; record actual device, dtype, CUDA and package versions. Declare any CPU run a separate reduced smoke configuration. GPU absence must fail at preflight before large downloads.

**RUN-06.** Targets, not measured guarantees: default execution within 20 minutes after connection, peak GPU allocation below 12 GiB, peak host RAM below 10 GiB, total download below 3 GiB and working storage below 8 GiB. Report cold-download and inference time separately. Preflight must verify feasibility; adjust the sample or documented envelope before release if needed.

## 7. Evaluation and controlled experiment

**EVAL-01.** Preserve raw reference and hypothesis strings. Version one normalization policy shared by all conditions: Unicode NFC, curly-apostrophe normalization, case-folding, punctuation-to-space consistent with the existing `_tokens` behavior, and whitespace collapse. Do not normalize numbers or abbreviations silently. Show raw and normalized examples. CER operates on normalized Unicode code points with spaces removed; state this explicitly.

**EVAL-02.** WER = `(S + D + I) / N_reference_words`; CER uses character edits/reference characters. Corpus rates use summed numerators and denominators, not a mean of utterance rates. Return deterministic alignments with a documented tie-break order. Validate edit counts against an independent reference implementation on fixtures. Rates may exceed 1; never clip them or label `1 − WER` as accuracy.

**EVAL-03.** Empty-transcript baseline has corpus WER/CER 1 on nonempty references. Label it a trivial sanity reference, not a competitive recognizer. Report normalized exact-match rate, S/D/I totals, eligible/evaluated/failed/truncated counts, audio seconds, inference seconds and real-time factor (processing seconds/audio seconds). Do not fabricate probabilities or use average token likelihood as a calibrated confidence score.

**EVAL-04.** Missing reference means `not-measurable`, with null rates and a reason. Distinguish a deliberately supplied empty/silence reference: its per-clip WER/CER denominator is zero, so report null and the insertion count. Silence probes get their own false-transcription report, not a manufactured 0% or 100% WER. Operational failures must fail the release acceptance run; diagnostic exports must not hide a reduced denominator.

**EXP-01.** Use the 16 activity recordings for a paired frozen-model clean versus synthetic Gaussian-noise experiment. Do not call Gaussian noise representative of real environments. Hold waveform, language, normalization and decoding fixed; only noise power changes.

**EXP-02.** Generate one fixed zero-mean noise vector per source ID with a versioned hash-derived RNG seed. Normalize its measured RMS; for target SNR `s`, use `noise_rms = clean_rms / 10**(s/20)`. Use the same realization when varying SNR. Reject near-zero-RMS inputs from this speech experiment and report the reason separately. Do not clip the noisy waveform. If headroom requires scaling, apply one common gain to both clean and noisy inputs for the pair; verify achieved SNR and export gain, seed and waveform hashes. Playback gain must match the scored waveforms.

**EXP-03.** Canonical activity SNR is 10 dB. The optional learner control offers 5 or 20 dB in an explicit exploratory branch. Report paired per-clip and corpus changes, including improved/unchanged/worsened counts. Improvement under noise is a valid observation, not a test failure. No statistical significance or universal robustness claim follows from 16 clips.

## 8. BYOD contract

**BYOD-01.** Disabled by default. Accept a WAV/FLAC file for transcription, or a ZIP containing audio files and optional UTF-8 `transcripts.csv`. Noninteractive callers set a path; an upload UI may exist only behind the explicit optional switch. V1 supports English evaluation; other languages are a separately specified extension, not a hidden dropdown promise.

**BYOD-02.** Bounds: 1–20 clips, 1–30 seconds each, no more than 300 seconds decoded audio total; mono or stereo, supported rates 8–48 kHz; maximum 100 MiB compressed archive and 200 MiB expanded contents, 100 members. CSV schema: `id,file,reference_text`; IDs unique, files relative, every referenced file present exactly once. Permit either fully labelled or wholly unlabelled sets; reject mixed/ambiguous labels clearly. The continuation uses the stronger schema in §10.

**BYOD-03.** Refuse traversal, absolute/drive paths, symlink members, case-insensitive path collisions, undeclared audio files, duplicate IDs/audio, unsupported channels/rates, nonfinite samples, corrupt formats and excessive resources before model execution. Use unique extraction/output directories and never delete or overwrite canonical results. Failed optional runs must leave canonical exports intact.

**BYOD-04.** The supported branch must actually decode, validate, transcribe, evaluate when labelled, and export results. It must not stop at loading a path. Unlabelled audio produces transcripts and operational measurements with `not-measurable` quality results. Optional noise exploration requires suitable speech clips and references for comparative metrics.

**BYOD-05.** Explain consent for recorded voices and permitted hosted processing. No microphone recording or third-party inference API. Keep user audio out of the default report ZIP; include hashes and metadata, with a clear option for the learner to retain local audio separately.

## 9. Outputs, interpretation and learner controls

Each execution creates a unique run directory under `outputs/whisper_speech_recognition/` containing:

- `input_manifest.json`: source identities, roles, preprocessing, exclusions, reference status.
- `frozen_experiment.json`: model/asset hashes, data-role digest, normalization, decoding, experiment settings, seeds and configuration hash; persisted before final evaluation.
- `transcripts.csv`: ID, role, condition, raw and normalized text, S/D/I, denominators, rates/null reasons, duration, timing, truncation/failure fields.
- `metrics.json`: clean evaluation and paired activity results in separate sections, complete coverage counts and evidence-scope labels.
- `noise_experiment.csv`: paired IDs, target/actual SNR, gain, seed, clean/noisy metrics and deltas.
- `provenance.json`: notebook/source identity, exact runtime lock, model/data pins, runtime/device and all acquisition/processing decisions; no secrets.
- `summary.md` and a report ZIP with an inventory of artifact byte counts and SHA-256 hashes. Use strict JSON (`allow_nan=False`).

BYOD and exploratory runs use their own subdirectories and manifests. The report builder must allowlist the intended run's files, not zip the entire working directory. Verify that CSV/JSON reload preserves row counts, text and aggregate metrics; record report-ZIP identity outside the ZIP to avoid self-referential hashes.

Conclusion template: “On [sample/scope], Whisper produced [metric] compared with [baseline]. Changing only [noise setting] produced [paired observation]. A notable error was [example]. This does not establish [generalization claim]. Before using this for [application], I would test [specific requirement].”

Include the agreed **AI Use Disclosure**, and state: “Personal learning notes are optional and do not need to be submitted.” No mandatory completion upload, grading or survey response in Run all.

## 10. Separate optional LoRA continuation

Reuse and improve the existing `whisper_asr_finetune_colab.ipynb`; do not create an unqualified second implementation of adapter loading. This is a subsequent deliverable in the same learning unit, not a requirement to finish the core unit.

Before advertising the upgraded continuation as ready:

1. Retain profile `E2E`; default Run all performs bounded adaptation. Apply the same isolated runtime and guided explanations. Preserve existing tutorial history rather than rewriting old evidence.
2. Proposed starting configuration: 200 train / 40 validation / 60 final test / 12 new-input probes, English, 1–20 seconds, seed 42. Freeze source groups and duplicates across all roles. Verify actual eligible counts before adopting the configuration. Do not let validation reuse final test labels.
3. Start from the existing rank-32, alpha-64 q/v LoRA recipe, two epochs, microbatch 4, accumulation 2, learning rate 1e-3. These are preflight candidates, not promised optimal settings. Assert optimizer-step counts including the final partial accumulation window; mask padding labels correctly. Reuse package-level helpers wherever semantics match.
4. Select the epoch by lowest validation corpus WER, with earlier epoch winning ties; include the frozen checkpoint as a valid candidate. Evaluate the frozen and selected models on the same untouched test recordings only after persisting the selection/freeze record. Training loss is optimization evidence, not ASR quality.
5. Report worsening, ties and a selected epoch of zero honestly. Never train longer merely to force improvement or a nonzero adapter delta. If the frozen candidate wins, export the base selection and a clearly labelled non-selected trained adapter for demonstrating serialization; verify that trained adapter separately without presenting it as the selected winner.
6. Export safetensors adapter/config/provenance with exact base identity and file hashes; verify before deserialization. Fresh-process reconstruction must reproduce live adapted validation predictions under the fixed decoding contract. Also verify nonzero LoRA weight application against `W_base + (alpha/r) BA` using measured tolerances; a null delta must be explicitly reported and must not pass as proof of adaptation. Final reported artifact metrics must refer to the persisted/reloaded selected state.
7. BYOD adaptation requires nonempty references and explicit `train`, `validation`, `test`, `probe` roles (`id,file,reference_text,split,group_id`), at least 8/2/2/1 clips respectively, with groups kept disjoint. No automatic random split of potentially related personal recordings. Use the same train → validation → freeze → test → export → fresh reload route; inference-only BYOD does not satisfy the adaptation contract.
8. Qualify a fresh GPU default run and a real-model BYOD adaptation run. Initial resource target: under 45 minutes and 12 GiB peak GPU allocation, to be measured and revised before release.

## 11. Acceptance criteria

| ID | Required evidence |
|---|---|
| ACC-01 | Generated notebook declares 2.2, TASK-INFERENCE, WORKSHOP, standalone, no worker, and Candidate until qualification; source/output contains no unfinished default cells. |
| ACC-02 | Exact committed notebook completes fresh Colab T4 Run all, no token, clean model/data caches, no restart, manual rerun, file edit or preloaded repository. Save executed notebook, start/end times, cell counts, device/runtime inventory, output hashes and any bootstrap actions. |
| ACC-03 | Pin/integrity refusal tests fail on deliberately altered model/data bytes; no silent fallback to another checkpoint or source. |
| ACC-04 | Data tests reject corrupt, oversized and duplicate recordings, role overlap and unsupported BYOD schemas before inference; canonical outputs survive optional failures. |
| ACC-05 | Metric tests cover exact match, substitutions, deletions, insertions, ties, empty hypotheses, absent/empty references, Unicode, rates above 1, and corpus weighting. Cross-check against independent expected values. |
| ACC-06 | Experiment tests establish deterministic same-noise reuse, achieved SNR, pair alignment, no clipping, common-gain behavior and isolation from the evaluation set. |
| ACC-07 | Source and generator parity checks pass; carried reusable functions match package semantics. Any deliberate wrapper restrictions (transcribe-only, duration ceilings, empty-reference policy) are documented and tested. |
| ACC-08 | Real-model labelled BYOD reaches metrics and exports; unlabelled BYOD reaches transcripts with null quality metrics; at least one invalid example is rejected clearly (REL12). Mocks alone do not establish this. |
| ACC-09 | Default output includes the clean baseline/model table, complete paired activity results, audible examples, error alignments, explicit limitations and reload-verified report data. No required playback click for Run all. |
| ACC-10 | GDL1–GDL15 checklist reviewed: audience, roadmap, I/O, terms, predictions, observations, worked interpretations, one-variable exercise, infrastructure separation, troubleshooting, conclusion and notebook terminology. |
| ACC-11 | Timings, memory, downloads and disk use measured for the exact candidate; targets met or accurately revised. No unsupported multilingual, translation, robustness or benchmark claim. |
| ACC-12 | Qualify every advertised path separately; core release is independent of continuation readiness. Record all remaining gates explicitly. |

## 12. Implementation sequence and deliverables

1. **Preflight:** verify anonymous pinned MINDS-14 acquisition, eligible counts, resource bounds, Python worker bootstrap and a single transcription on the selected T4 class. Do not run full training just to write this specification.
2. **Core build:** proposed `tools/whisper_speech_workshop_source.py`, `tools/build_whisper_speech_workshop.py`, focused shared audio/metric helpers as needed, generated notebook, and meaningful tests. Preserve both existing notebooks and their generator checks.
3. **Core qualification:** fresh Run all, labelled/unlabelled/invalid BYOD, noise activity, report verification; update `tutorials/README.md` and `docs/release-verification.md` for the new blob. A public release claim requires this evidence.
4. **Continuation:** implement §10 as a separate reviewed change and qualification record. Core readiness does not promote this path automatically.

Implementation-time decisions with named evidence:

- Builder resolves exact worker/bootstrap lock from fresh-runtime preflight; no unpinned interpreter or arbitrary current wheel selection in the release.
- Builder inspects the pinned corpus metadata for speaker/session groups and documents absent information; maintainer approves any dataset replacement if the proposed public sample cannot satisfy the contract.
- Maintainer reviews sample counts and runtime targets after measurement. Do not reduce counts silently to make a run pass.
- Existing API translation claims are a separate repository correction to track; the new notebook explicitly refuses translation regardless of the existing task tuple.

## 13. References

- [DIMER NOTEBOOK_SPEC 2.2 at the inspected fleet revision](https://github.com/kurtvalcorza/ml-worker/blob/ce276e10197a1b9f1c6ebe72c8effc730f761d34/integrations/dimer/fleet-specs/NOTEBOOK_SPEC.md) — especially RUN1–14, standalone/parity, BYOD/REL12 and GDL1–15.
- [Whisper pipeline source at the inspected revision](https://github.com/kurtvalcorza/whisper-asr-pipeline/blob/1eb99622614d547170675b24c051510168a694da/src/whisper_asr_pipeline/pipeline.py).
- [Existing fine-tuning template](https://github.com/kurtvalcorza/whisper-asr-pipeline/blob/1eb99622614d547170675b24c051510168a694da/tools/notebook_template_finetune.py) and [release evidence](https://github.com/kurtvalcorza/whisper-asr-pipeline/blob/1eb99622614d547170675b24c051510168a694da/docs/release-verification.md).
- [Upstream Whisper task guidance](https://github.com/openai/whisper#available-models-and-languages) — Turbo translation exclusion.
- [MINDS-14 dataset card](https://huggingface.co/datasets/PolyAI/minds14) — source corpus, schema and license; release selection must use the immutable revision and measured file hashes above.
