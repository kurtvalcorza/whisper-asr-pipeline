# Tutorials

Notebook specification: **DIMER Notebook Specification 1.0**

| Notebook | Profile | Capability | Default runtime | BYOD | Release status |
|---|---|---|---|---|---|
| `whisper_asr_colab.ipynb` | `TASK-INFERENCE` | Whisper large-v3-turbo multilingual ASR with WER against the public sample reference | CPU (CUDA used automatically when available) | single audio file, gated off by default | **Candidate** — static checks pass; clean-runtime execution evidence is recorded in `../docs/release-verification.md` and must be reviewed for the exact notebook revision before promotion |
| `whisper_asr_finetune_colab.ipynb` | `E2E` | LoRA domain adaptation of Whisper large-v3-turbo on one locale of the public `PolyAI/minds14` corpus (8 kHz telephone speech), with held-out WER before/after and a manifested adapter bundle that is reloaded through the public API | CUDA (T4 class; ~6 GiB, 10–15 min). CPU only as a reduced smoke configuration | `transcripts.csv` + audio files, gated off by default | **Candidate** — static checks pass; clean-runtime execution evidence must be recorded in `../docs/release-verification.md` for the exact notebook revision before promotion |

## Conformance notes

- The notebook exercises `WhisperASRPipeline` from the repository public API rather than reimplementing model loading; the pipeline pins the immutable upstream revision and refuses remote model code.
- The default sample is the public `hf-internal-testing/librispeech_asr_dummy` clip; its WER is tutorial evidence for one utterance, not a fleet benchmark.
- `USE_BYOD` defaults to `False` so the sample path never opens an upload dialog.
- The fine-tuning notebook loads the base model and the exported adapter through `load_model` /
  `WhisperASRPipeline.from_pretrained(adapter_dir=...)`; only the LoRA training loop lives in the
  notebook. Its default corpus is resampled from 8 kHz to 16 kHz with the same pinned `torchaudio`
  resampler the pipeline applies at inference. The baseline/adapted/reloaded WER figures are
  corpus-level numbers on one small held-out split and are tutorial evidence, not a benchmark.
- The exported adapter (`outputs/whisper-asr-lora-adapter.zip`) is a user artifact produced in the
  learner's runtime; the repository ships no adapted weights.
- `tools/validate_release_assets.py` performs source validation only. It does not satisfy the
  clean-runtime execution requirement; a release review must confirm that a recorded clean run in
  `docs/release-verification.md` matches the notebook revision under review before the status is
  promoted to `Release-grade`.
