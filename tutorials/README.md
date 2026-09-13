# Tutorials

[![GitHub](https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white)](https://github.com/kurtvalcorza/whisper-asr-pipeline)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kurtvalcorza/whisper-asr-pipeline/blob/main/tutorials/whisper_asr_colab.ipynb)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-openai%2Fwhisper--large--v3--turbo-ffcc4d?style=flat)](https://huggingface.co/openai/whisper-large-v3-turbo)
[![Upstream](https://img.shields.io/badge/Upstream-openai%2Fwhisper-181717?style=flat&logo=github&logoColor=white)](https://github.com/openai/whisper)
[![arXiv](https://img.shields.io/badge/arXiv-2212.04356-b31b1b.svg)](https://arxiv.org/abs/2212.04356)

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
