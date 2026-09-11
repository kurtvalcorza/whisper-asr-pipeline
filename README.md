# Whisper ASR Pipeline

DIMER-oriented inference wrapper for **OpenAI Whisper large-v3-turbo**, pinned to an immutable Hugging Face revision. The repository exposes automatic speech recognition/transcription as the primary capability, optional speech-to-English translation through Whisper's supported `task` switch, WER evaluation when a reference transcript is available, and machine-readable provenance.

## Upstream alignment

- Model: `openai/whisper-large-v3-turbo`
- Revision: `41f01f3fe87f28c78e2fbf8b568835947dd65ed9`
- Upstream weight license: MIT
- Upstream task: automatic speech recognition / speech translation
- Repository adaptation: **none**; inference only

## Public API

```python
from whisper_asr_pipeline import WhisperASRPipeline
pipe = WhisperASRPipeline.from_pretrained()
result = pipe.transcribe("audio.wav", language="en")
print(result["text"])
```

## Tutorial

`tutorials/whisper_asr_colab.ipynb` is declared `TASK-INFERENCE`. Its default path uses a public LibriSpeech dummy sample, validates runtime/model provenance, transcribes through the public API, computes tutorial WER against the supplied reference (case-folded, punctuation removed, curly apostrophes folded; abbreviations are not normalized), and exports JSON. BYOD is optional and gated off by default.

## Release status

**Candidate.** Static/unit checks do not constitute clean-runtime notebook evidence. Complete `docs/release-verification.md` against the exact release revision before calling the notebook release-grade.

## Licensing

This repository's code is Apache-2.0. The packaged upstream Whisper weights are MIT-licensed; see `docs/WEIGHTS.md` and `MODEL_CARD.md`.
