---
license: mit
model_card_spec: "1.0"
pipeline_tag: automatic-speech-recognition
base_model: openai/whisper-large-v3-turbo
---

# Whisper large-v3-turbo (DIMER package v0.1.0)

[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-openai%2Fwhisper--large--v3--turbo-ffcc4d)](https://huggingface.co/openai/whisper-large-v3-turbo)
[![Weight license](https://img.shields.io/badge/weights-mit-blue)](https://huggingface.co/openai/whisper-large-v3-turbo/blob/41f01f3fe87f28c78e2fbf8b568835947dd65ed9/LICENSE)

###### Description

Whisper large-v3-turbo is the OpenAI Whisper speech-sequence-to-sequence model published as `openai/whisper-large-v3-turbo`, pinned here to revision `41f01f3fe87f28c78e2fbf8b568835947dd65ed9`. Upstream describes Turbo as a fine-tuned/pruned large-v3 variant with the decoder reduced from 32 layers to 4 for faster inference. This repository does not adapt weights; it adds immutable resolution, input checks, normalized ASR output, WER evaluation, provenance, and DIMER tutorial packaging.

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

The reference runtime is Python 3.12 with pinned PyTorch, torchvision, Transformers, Accelerate, NumPy, and SoundFile versions. torchvision is pinned to the build that matches the pinned PyTorch because hosted runtimes ship a torchvision compiled against their own PyTorch; once PyTorch is pinned, an orphaned torchvision breaks the Transformers processor imports (`operator torchvision::nms does not exist`). CUDA is recommended for practical large-v3-turbo inference; CPU is permitted but may be substantially slower. The data environment assumes intelligible speech reasonably represented by Whisper's upstream training distribution. Domain jargon, heavy overlap, severe noise, unusual recording conditions, or languages with weaker upstream coverage can increase transcription error and require local evaluation.

#### Metrics

###### Performance Measures

The repository reports `word_error_rate` when a reference transcript is supplied. WER measures token-level substitutions, insertions, and deletions relative to reference words after basic normalization (lower-casing and punctuation removal; numbers, abbreviations and spelled-out forms are not normalized, so `Mr.` versus `Mister` counts as an error), making it interpretable for ASR but insensitive to some semantic differences. The tutorial labels its WER as sample/tutorial evidence from a single public example; the repository does not reproduce or claim upstream benchmark scores as measurements made by this pipeline.

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

Implemented mitigations include an immutable upstream model revision; standard Transformers loading with `trust_remote_code=False`; a SafeTensors upstream checkpoint; explicit task and chunk-length validation; missing-file rejection; normalized model/revision fields in every result; exact dependency pins for the model-facing runtime; a repository WER implementation for labeled checks; source-level notebook/model-card validation in CI; and a documented rule that static CI cannot be promoted as notebook runtime evidence.

###### Risks and harms

Transcription errors can omit, substitute, or hallucinate words, harming speakers or operators when text is treated as exact. Bias may appear as higher error rates for underrepresented languages, accents, or recording conditions. Automation bias can cause reviewers to overlook plausible-looking mistakes. Processing recordings may expose private or restricted information. Long or noisy audio can also create resource or latency failures. These risks increase when local validation and human review are absent.

###### Use cases

The pipeline must not be used for covert or unlawful surveillance, voice-biometric identification, demographic profiling, social scoring, unlawful discrimination, deceptive impersonation, or to fabricate supposedly verbatim evidence. It must not be used in ways that violate recording consent, privacy, copyright, data-protection requirements, the upstream model license, or DIMER deployment terms. High-consequence actions based solely on unreviewed generated transcripts are prohibited by this repository's intended-use contract.

## Immutable provenance

- Model: `openai/whisper-large-v3-turbo`
- Revision: `41f01f3fe87f28c78e2fbf8b568835947dd65ed9`
- Weight format: SafeTensors
- Upstream reference: https://huggingface.co/openai/whisper-large-v3-turbo
- Whisper paper: https://arxiv.org/abs/2212.04356
