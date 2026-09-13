# Whisper ASR Pipeline

DIMER-oriented inference wrapper for **OpenAI Whisper large-v3-turbo**, pinned to an immutable Hugging Face revision. The repository exposes automatic speech recognition/transcription as the primary capability, optional speech-to-English translation through Whisper's supported `task` switch, WER evaluation when a reference transcript is available, and machine-readable provenance.

## Upstream alignment

- Model: `openai/whisper-large-v3-turbo`
- Revision: `41f01f3fe87f28c78e2fbf8b568835947dd65ed9`
- Upstream weight license: MIT
- Upstream task: automatic speech recognition / speech translation
- Repository adaptation: **none**; the packaged weights are inference-only. The optional `finetune`
  extra and the E2E tutorial let a user train a LoRA adapter locally; adapters are user artifacts
  loaded through the same public API and are not shipped by this repository

## Public API

```python
from whisper_asr_pipeline import WhisperASRPipeline
pipe = WhisperASRPipeline.from_pretrained()
result = pipe.transcribe("audio.wav", language="en")
print(result["text"])
```

A LoRA adapter exported by the fine-tuning tutorial is merged into the pinned base at load time
(`pip install ".[finetune]"` adds `peft`):

```python
adapted = WhisperASRPipeline.from_pretrained(adapter_dir="outputs/whisper-asr-lora-adapter")
result = adapted.transcribe("audio.wav", language="en")
print(result["adapter"], result["adapter_sha256"])  # bundle path and the SHA-256 of its weights
```

Only safetensors bundles are accepted (`adapter_model.safetensors` must be present); a pickle-only
`adapter_model.bin` bundle is refused rather than deserialized.

`load_model(device=None, adapter_dir=None)` returns the underlying `(model, processor)` pair for
callers that need the raw Transformers objects; `WhisperASRPipeline.from_model(model, processor)` wraps
such a model (plain or PEFT-wrapped) in the same decoding path, and `.model` exposes the loaded model
for inspection; and `word_error_count` returns the
`(edits, reference_words)` pair from which a corpus WER is summed.

## Tutorials

`tutorials/whisper_asr_colab.ipynb` is declared `TASK-INFERENCE`. Its default path uses a public LibriSpeech dummy sample, validates runtime/model provenance, transcribes through the public API, computes tutorial WER against the supplied reference (case-folded, punctuation removed, curly apostrophes folded; abbreviations are not normalized), and exports JSON. BYOD is optional and gated off by default.

`tutorials/whisper_asr_finetune_colab.ipynb` is declared `E2E`. Its default path loads one locale of
the public `PolyAI/minds14` corpus (CC-BY-4.0, 8 kHz telephone speech, resampled to 16 kHz), records
a zero-shot corpus WER on a held-out split through the public API, trains LoRA adapters on the
attention query/value projections with float16 mixed precision, measures the adapted WER, exports a
manifested adapter bundle, and proves the bundle reloads through
`WhisperASRPipeline.from_pretrained(adapter_dir=...)` with the same effect. It needs a 16 GiB-class
CUDA GPU for the default configuration (measured: 4.5 GiB peak, 27 minutes on a Kaggle P100). BYOD (`transcripts.csv` + audio files) is optional and gated
off by default.

## Release status

**Candidate.** Static/unit checks do not constitute clean-runtime notebook evidence. Complete `docs/release-verification.md` against the exact release revision of each notebook before calling it release-grade.

## Licensing

This repository's code is Apache-2.0. The packaged upstream Whisper weights are MIT-licensed; see `docs/WEIGHTS.md` and `MODEL_CARD.md`.
