# Tutorials

Notebook specification: **DIMER Notebook Specification 1.0**

| Notebook | Profile | Capability | Default runtime | BYOD | Release status |
|---|---|---|---|---|---|
| `whisper_asr_colab.ipynb` | `TASK-INFERENCE` | Whisper large-v3-turbo multilingual ASR with WER against the public sample reference | CPU (CUDA used automatically when available) | single audio file, gated off by default | **Candidate** — static checks pass; clean-runtime execution evidence is recorded in `../docs/release-verification.md` and must be reviewed for the exact notebook revision before promotion |

## Conformance notes

- The notebook exercises `WhisperASRPipeline` from the repository public API rather than reimplementing model loading; the pipeline pins the immutable upstream revision and refuses remote model code.
- The default sample is the public `hf-internal-testing/librispeech_asr_dummy` clip; its WER is tutorial evidence for one utterance, not a fleet benchmark.
- `USE_BYOD` defaults to `False` so the sample path never opens an upload dialog.
- `tools/validate_release_assets.py` performs source validation only. It does not satisfy the
  clean-runtime execution requirement; a release review must confirm that a recorded clean run in
  `docs/release-verification.md` matches the notebook revision under review before the status is
  promoted to `Release-grade`.
