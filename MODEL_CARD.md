---
license: mit
model_card_spec: "1.1"
pipeline_tag: automatic-speech-recognition
base_model: openai/whisper-large-v3-turbo
date_published: "2024-10-01"
date_published_source: "Hugging Face Hub repository creation date of the exact hosted checkpoint (`createdAt`, https://huggingface.co/api/models/openai/whisper-large-v3-turbo)"
---

# Whisper large-v3-turbo — Speech Recognition Model (Inference & LoRA Fine-Tuning)

[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-openai%2Fwhisper--large--v3--turbo-ffcc4d?style=flat)](https://huggingface.co/openai/whisper-large-v3-turbo)
[![Upstream GitHub](https://img.shields.io/badge/Upstream%20GitHub-openai%2Fwhisper-181717?style=flat&logo=github&logoColor=white)](https://github.com/openai/whisper)
[![arXiv Paper](https://img.shields.io/badge/arXiv-2212.04356-b31b1b.svg)](https://arxiv.org/abs/2212.04356)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://huggingface.co/openai/whisper-large-v3-turbo/blob/41f01f3fe87f28c78e2fbf8b568835947dd65ed9/LICENSE)

> [!WARNING]
> ⚠️ **Provided for research, training, and evaluation purposes only.** Model weights are redistributed unmodified under their upstream license, which controls your use, including any commercial use or redistribution; the accompanying code and notebooks are released under this repository's license. All of it is supplied **"as is"**, without warranty of any kind, and has not been validated for production, clinical, or safety-critical use. Running the notebooks downloads third-party weights and datasets governed by their own licenses and consumes compute on your own Colab/Kaggle account. To the maximum extent permitted by law, the maintainers of this repository and the DIMER platform accept no liability for any damages arising from their use. Hosting implies no affiliation with or endorsement by the original authors.

---

## Interactive Colab Tutorials

This repository ships standalone Google Colab tutorials that exercise its public pipeline API:

- **Task Inference Tutorial**: \
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kurtvalcorza/whisper-asr-pipeline/blob/main/tutorials/whisper_asr_colab.ipynb) [`whisper_asr_colab.ipynb`](https://github.com/kurtvalcorza/whisper-asr-pipeline/blob/main/tutorials/whisper_asr_colab.ipynb) \
  *Verify the pinned Whisper snapshot, validate and transcribe a public speech sample or an uploaded clip, compute word error rate when a reference transcript is available, and export the transcript, evaluation report and provenance.*

- **LoRA Fine-Tuning Tutorial**: \
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kurtvalcorza/whisper-asr-pipeline/blob/main/tutorials/whisper_asr_finetune_colab.ipynb) [`whisper_asr_finetune_colab.ipynb`](https://github.com/kurtvalcorza/whisper-asr-pipeline/blob/main/tutorials/whisper_asr_finetune_colab.ipynb) \
  *Validate a public labelled speech set or your own clips, record zero-shot word error rate, train LoRA adapters, evaluate the adapted model, and export and reload the adapter bundle against the verified base weights.*

> [!NOTE]
> The LoRA tutorial defaults require a 16 GiB-class CUDA GPU; CPU smoke runs require reducing `TRAIN_CLIPS`, `EVAL_CLIPS` and `EPOCHS`. These tutorials remain release candidates; see [release verification](docs/release-verification.md) for execution records and promotion requirements.

---

#### Description

Whisper large-v3-turbo is the OpenAI Whisper speech-sequence-to-sequence model published as `openai/whisper-large-v3-turbo`, pinned here to revision `41f01f3fe87f28c78e2fbf8b568835947dd65ed9`. Upstream describes Turbo as a fine-tuned/pruned large-v3 variant with the decoder reduced from 32 layers to 4 for faster inference. This repository does not adapt weights; it adds immutable resolution, input checks, normalized ASR output, WER evaluation, provenance, and DIMER tutorial packaging. The `finetune` extra and the E2E tutorial let a user train a LoRA adapter on their own labelled speech and load it through the same public API (`adapter_dir=`); such adapters are user artifacts produced in the user's runtime, are not distributed by this repository, and inherit the license obligations of both these weights and the training data.

#### Intended Use and Limitations

###### Primary Intended Uses

The primary task is automatic speech recognition: audio is converted to text using the pinned multilingual Whisper checkpoint. A secondary supported path exposes Whisper's speech-translation task when the caller explicitly requests it. Intended applications include transcription of meetings, interviews, lectures, field recordings, and other authorized speech where operators can review errors. The pipeline is intended as an inference component or evaluation baseline, not an autonomous decision system.

###### Primary Intended Users

Primary users are ML engineers, data scientists, software developers, researchers, and public-service technical teams integrating speech transcription into larger systems. Operators are expected to understand audio sampling and encoding, language selection, word-error-rate interpretation, privacy obligations for recorded speech, and the difference between a transcript produced by a model and a verified human record. Deployment owners remain responsible for domain validation and access controls.

###### Out-of-scope use cases

1. **Capability boundary:** speaker identification, diarization, voice biometrics, emotion inference, and audio event classification are not implemented by this pipeline.
2. **Input boundary:** the public wrapper accepts audio understood by the Transformers ASR pipeline and rejects missing local paths; chunk size is constrained to 1–30 seconds per inference chunk.
3. **Decision boundary:** unreviewed transcripts must not be treated as authoritative evidence for health, legal, disciplinary, employment, security, or other high-consequence decisions.

#### Factors

###### Groups

This pipeline is human-centric because speech comes from people and transcription quality can vary across languages, accents, dialects, age groups, speaking styles, background conditions, and other characteristics. This repository does not claim a demographic fairness audit of the upstream pretraining corpus or checkpoint. Downstream operators therefore must measure error rates on representative local speaker groups and investigate materially different error rates before deployment.

###### Instrumentation

Input audio may originate from microphones, telephony systems, meeting platforms, recorders, broadcast sources, or digitized archives. Sampling rate, codec, clipping, microphone distance, reverberation, channel mixing, packet loss, and background noise can all alter the acoustic evidence reaching the model. The wrapper checks file existence and delegates decoding/resampling to the supported ASR stack, but it cannot detect every upstream capture defect or reconstruct information absent from the recording.

###### Environment

The reference runtime is Python 3.12 with pinned PyTorch, torchvision, Transformers, Accelerate, NumPy, and SoundFile versions. torchvision and torchaudio are pinned to the builds that match the pinned PyTorch because hosted runtimes ship copies compiled against their own PyTorch; once PyTorch is pinned, an orphaned torchvision breaks the Transformers processor imports (`operator torchvision::nms does not exist`) and an orphaned torchaudio would break resampling of in-memory audio supplied at a rate other than 16 kHz. CUDA is recommended for practical large-v3-turbo inference; CPU is permitted but may be substantially slower. The data environment assumes intelligible speech reasonably represented by Whisper's upstream training distribution. Domain jargon, heavy overlap, severe noise, unusual recording conditions, or languages with weaker upstream coverage can increase transcription error and require local evaluation.

#### Metrics

###### Performance Measures

The repository reports `word_error_rate` when a reference transcript is supplied. The public `evaluation_report` stage writes it to a machine-readable report whose verdict is `sample-sanity` on the single referenced tutorial utterance and `not-measurable` when no reference exists. WER measures token-level substitutions, insertions, and deletions relative to reference words after basic normalization (case-folding, curly-to-straight apostrophes and punctuation removal, with hyphenated compounds kept as one token; numbers, abbreviations and spelled-out forms are not normalized, so `Mr.` versus `Mister` counts as an error), making it interpretable for ASR but insensitive to some semantic differences. The tutorial labels its WER as sample/tutorial evidence from a single public example; the repository does not reproduce or claim upstream benchmark scores as measurements made by this pipeline.

###### Decision thresholds

No acceptance threshold is shipped. `transcribe()` returns model text rather than converting a confidence score into an accept/reject decision, and this wrapper does not claim calibrated token or utterance probabilities. A deployment that needs automatic acceptance must establish its own review or confidence policy against representative labeled audio, balancing the cost of missed transcription errors against the operational cost of human review.

###### Approaches to uncertainty and variability

The tutorial uses one deterministic public sample and reports a single WER value; it does not compute a confidence interval or dispersion estimate. Model decoding, floating-point kernels, hardware, chunking, language hints, and audio decoding can change results. The reference tutorial favors reproducible settings but does not claim bitwise determinism across devices. No calibrated per-transcript probability is produced; deployments needing uncertainty estimates must validate an appropriate confidence or review mechanism.

#### Ethical considerations and biases

###### Data

Upstream Whisper documentation describes large-scale multilingual weakly supervised speech data; the full composition and sensitivity status of every upstream example are not enumerated by this repository, so sensitive content cannot be ruled out by us. This repository distributes code, documentation, and tutorial logic but not the upstream multi-gigabyte model weights or user recordings. Operators must audit inference audio for personal, confidential, copyrighted, classified, or otherwise restricted information before processing.

###### Human Life

This pipeline is not intended or certified for autonomous decisions central to health, safety, criminal justice, employment, credit, housing, or similar high-impact domains. No external board or regulator has validated this repository for those purposes. Where transcription is used inside a sensitive workflow, admissible use requires appropriate human review, independent domain validation of error patterns, data-governance controls, and any regulatory or institutional approval applicable to the recording and decision process.

###### Mitigations

Implemented mitigations include an immutable upstream model revision; a committed `dimer-base-manifest.json` whose per-file SHA-256 digests `verify_snapshot` re-checks before every load; the public `validate_inputs` stage, which applies the same task, path and chunk-length checks as `transcribe` and writes an input manifest with any rejection recorded as a finding; standard Transformers loading with `trust_remote_code=False`; a SafeTensors upstream checkpoint; explicit task and chunk-length validation; missing-file rejection; normalized model/revision fields in every result; exact dependency pins for the model-facing runtime; a repository WER implementation for labeled checks; source-level notebook/model-card validation in CI; and a documented rule that static CI cannot be promoted as notebook runtime evidence.

###### Risks and harms

Transcription errors can omit, substitute, or hallucinate words, harming speakers or operators when text is treated as exact. Bias may appear as higher error rates for underrepresented languages, accents, or recording conditions. Automation bias can cause reviewers to overlook plausible-looking mistakes. Processing recordings may expose private or restricted information. Long or noisy audio can also create resource or latency failures. These risks increase when local validation and human review are absent.

###### Use cases

The pipeline must not be used for covert or unlawful surveillance, voice-biometric identification, demographic profiling, social scoring, unlawful discrimination, deceptive impersonation, or to fabricate supposedly verbatim evidence. It must not be used in ways that violate recording consent, privacy, copyright, data-protection requirements, the upstream model license, or DIMER deployment terms. High-consequence actions based solely on unreviewed generated transcripts are prohibited by this repository's intended-use contract.

## Immutable provenance

- Model: `openai/whisper-large-v3-turbo`
- Revision: `41f01f3fe87f28c78e2fbf8b568835947dd65ed9`
- Weight format: SafeTensors
- Snapshot manifest: `weights/whisper-large-v3-turbo/dimer-base-manifest.json` — `model.safetensors` SHA-256 `542566a422ae4f3fd23f1ba11add198fca01bbf82e66e6a2857b3f608b1eb9d1` (1617824864 bytes)
- Upstream reference: https://huggingface.co/openai/whisper-large-v3-turbo
- Whisper paper: https://arxiv.org/abs/2212.04356
