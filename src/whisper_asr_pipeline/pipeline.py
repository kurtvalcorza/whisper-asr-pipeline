from __future__ import annotations

import hashlib
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MODEL_ID = "openai/whisper-large-v3-turbo"
MODEL_REVISION = "41f01f3fe87f28c78e2fbf8b568835947dd65ed9"
MODEL_LICENSE = "MIT"
# A LoRA bundle is accepted only in the safetensors format; peft would otherwise fall back to a
# pickle-based adapter_model.bin, which this repository's trust boundary refuses.
ADAPTER_WEIGHTS = "adapter_model.safetensors"
# Basic WER normalization: case-fold and drop punctuation so that "classes," and "gospel."
# match an unpunctuated reference. Curly apostrophes are folded to the straight form first;
# word-internal apostrophes and hyphens are kept (a hyphenated compound stays one token).
# Numbers, abbreviations and spelled-out forms are NOT normalized ("Mr." vs "Mister" is an
# error).
_APOSTROPHES = str.maketrans({"\u2019": "'", "\u2018": "'", "\u02bc": "'"})
_PUNCTUATION = re.compile(r"[^\w\s'-]|(?<!\w)['-]|['-](?!\w)", re.UNICODE)


def _tokens(text: str) -> list[str]:
    return _PUNCTUATION.sub(" ", text.casefold().translate(_APOSTROPHES)).split()


def word_error_count(reference: str, hypothesis: str) -> tuple[int, int]:
    """Word-level edit distance and reference length after basic normalization.

    Summing the pairs over a corpus and dividing gives the corpus WER; ``word_error_rate``
    is the single-utterance ratio.
    """
    reference_tokens = _tokens(reference)
    hypothesis_tokens = _tokens(hypothesis)
    if not reference_tokens:
        return len(hypothesis_tokens), 0

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
    return previous[-1], len(reference_tokens)


def word_error_rate(reference: str, hypothesis: str) -> float:
    """Word error rate after basic normalization (lowercase, punctuation removed)."""
    errors, reference_length = word_error_count(reference, hypothesis)
    if reference_length == 0:
        return 0.0 if errors == 0 else 1.0
    return errors / reference_length


def adapter_digest(adapter_dir: str | Path) -> str:
    """SHA-256 of the bundle's ``adapter_model.safetensors``: the adapter's identity."""
    digest = hashlib.sha256()
    with open(Path(adapter_dir) / ADAPTER_WEIGHTS, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_model(device: str | None = None, adapter_dir: str | Path | None = None) -> tuple[Any, Any]:
    """Return ``(model, processor)`` for the pinned upstream revision, on ``device``.

    Weights are loaded in float16 on CUDA and float32 on CPU. With ``adapter_dir``, a PEFT
    LoRA adapter saved by ``PeftModel.save_pretrained`` (for example by the fine-tuning
    tutorial) is attached and merged into the weights, so the result is a plain Whisper
    model; ``peft`` is only imported on that path, and only a safetensors bundle is accepted.
    Remote model code is always refused.
    """
    adapter_path = None if adapter_dir is None else Path(adapter_dir)
    if adapter_path is not None:
        for required in ("adapter_config.json", ADAPTER_WEIGHTS):
            if not (adapter_path / required).is_file():
                raise FileNotFoundError(f"{required} not found in {adapter_path}")

    import torch
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

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
    if adapter_path is not None:
        try:
            from peft import PeftModel
        except ImportError as exc:  # pragma: no cover - depends on the optional extra
            raise ImportError("loading an adapter requires the 'finetune' extra (peft)") from exc
        model = PeftModel.from_pretrained(model, str(adapter_path), is_trainable=False)
        model = model.merge_and_unload()
    if resolved_device.startswith("cuda"):
        model = model.to(resolved_device)
    return model, processor


@dataclass
class WhisperASRPipeline:
    _runner: Callable[..., dict[str, Any]]
    device: str
    adapter: str | None = None
    adapter_sha256: str | None = None

    @property
    def model(self) -> Any:
        """The underlying Transformers model, for inspection (weights, config, dtype)."""
        return self._runner.model

    @classmethod
    def from_pretrained(
        cls, device: str | None = None, adapter_dir: str | Path | None = None
    ) -> WhisperASRPipeline:
        model, processor = load_model(device=device, adapter_dir=adapter_dir)
        if adapter_dir is None:
            return cls.from_model(model, processor)
        return cls.from_model(
            model, processor, adapter=str(adapter_dir), adapter_sha256=adapter_digest(adapter_dir)
        )

    @classmethod
    def from_model(
        cls, model: Any, processor: Any, adapter: str | None = None, adapter_sha256: str | None = None
    ) -> WhisperASRPipeline:
        """Wrap an already-loaded Whisper model (plain or PEFT-wrapped) in the same decoding path.

        Lets a caller that holds a live model — the fine-tuning tutorial, between training and
        export — transcribe through exactly the pipeline that ``from_pretrained`` builds, so
        in-memory and reloaded results are comparable.
        """
        from transformers import pipeline

        resolved_device = str(model.device)
        runner = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            torch_dtype=model.dtype,
            device=resolved_device,
        )
        return cls(runner, resolved_device, adapter, adapter_sha256)

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
            "adapter": self.adapter,
            "adapter_sha256": self.adapter_sha256,
        }
