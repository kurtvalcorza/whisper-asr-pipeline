from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MODEL_ID = "openai/whisper-large-v3-turbo"
MODEL_REVISION = "41f01f3fe87f28c78e2fbf8b568835947dd65ed9"
MODEL_LICENSE = "MIT"
# Basic WER normalization: case-fold and drop punctuation so that "classes," and "gospel."
# match an unpunctuated reference. Word-internal apostrophes and hyphens are kept. Numbers,
# abbreviations and spelled-out forms are NOT normalized ("Mr." vs "Mister" is an error).
_PUNCTUATION = re.compile(r"[^\w\s'-]|(?<!\w)['-]|['-](?!\w)", re.UNICODE)


def _tokens(text: str) -> list[str]:
    return _PUNCTUATION.sub(" ", text.lower()).split()


def word_error_rate(reference: str, hypothesis: str) -> float:
    """Word error rate after basic normalization (lowercase, punctuation removed)."""
    reference_tokens = _tokens(reference)
    hypothesis_tokens = _tokens(hypothesis)
    if not reference_tokens:
        return 0.0 if not hypothesis_tokens else 1.0

    previous = list(range(len(hypothesis_tokens) + 1))
    for row_index, reference_token in enumerate(reference_tokens, 1):
        current = [row_index]
        for column_index, hypothesis_token in enumerate(hypothesis_tokens, 1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column_index] + 1,
                    previous[column_index - 1] + (reference_token != hypothesis_token),
                )
            )
        previous = current
    return previous[-1] / len(reference_tokens)


@dataclass
class WhisperASRPipeline:
    _runner: Callable[..., dict[str, Any]]
    device: str

    @classmethod
    def from_pretrained(cls, device: str | None = None) -> WhisperASRPipeline:
        import torch
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

        resolved_device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if resolved_device.startswith("cuda") else torch.float32
        processor = AutoProcessor.from_pretrained(
            MODEL_ID,
            revision=MODEL_REVISION,
            trust_remote_code=False,
        )
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            MODEL_ID,
            revision=MODEL_REVISION,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=False,
        )
        if resolved_device.startswith("cuda"):
            model = model.to(resolved_device)
        runner = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            torch_dtype=dtype,
            device=resolved_device,
        )
        return cls(runner, resolved_device)

    def transcribe(
        self,
        audio: str | Path | dict[str, Any],
        *,
        language: str | None = None,
        task: str = "transcribe",
        return_timestamps: bool = False,
        chunk_length_s: int = 30,
    ) -> dict[str, Any]:
        if task not in {"transcribe", "translate"}:
            raise ValueError("task must be 'transcribe' or 'translate'")
        if (
            isinstance(audio, str | Path)
            and not str(audio).startswith(("http://", "https://"))
            and not Path(audio).is_file()
        ):
            raise FileNotFoundError(f"audio file not found: {audio}")
        if not 1 <= chunk_length_s <= 30:
            raise ValueError("chunk_length_s must be between 1 and 30 seconds")

        generate_kwargs: dict[str, Any] = {"task": task}
        if language:
            generate_kwargs["language"] = language
        raw = self._runner(
            str(audio) if isinstance(audio, Path) else audio,
            return_timestamps=return_timestamps,
            chunk_length_s=chunk_length_s,
            generate_kwargs=generate_kwargs,
        )
        if not isinstance(raw, dict) or "text" not in raw:
            raise RuntimeError("ASR backend returned an invalid result")
        return {
            "text": str(raw["text"]).strip(),
            "chunks": raw.get("chunks"),
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "task": task,
            "language": language,
        }
