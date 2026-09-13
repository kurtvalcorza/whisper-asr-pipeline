"""Multilingual speech recognition with the pinned ``openai/whisper-large-v3-turbo`` checkpoint.

The class loads the processor and model only from a digest-verified local snapshot (``weights/<key>/``)
or, when explicitly allowed, from the Hugging Face Hub at the pinned revision — always with
``trust_remote_code=False``: the architecture comes from the pinned ``transformers`` release, the
weights are SafeTensors, and no model-repository code is executed.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MODEL_ID = "openai/whisper-large-v3-turbo"
MODEL_REVISION = "41f01f3fe87f28c78e2fbf8b568835947dd65ed9"
MODEL_LICENSE = "MIT"
MODEL_KEY = "whisper-large-v3-turbo"
DEFAULT_WEIGHTS_DIR = Path(__file__).resolve().parents[2] / "weights" / MODEL_KEY
MANIFEST_NAME = "dimer-base-manifest.json"
WEIGHTS_FILE = "model.safetensors"
CONFIG_FILE = "config.json"

TASKS = ("transcribe", "translate")
MIN_CHUNK_LENGTH_S = 1
MAX_CHUNK_LENGTH_S = 30  # Whisper's receptive field; longer audio is chunked by the transformers pipeline
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_snapshot(path: str | Path | None = None) -> dict[str, Any]:
    """Check a local snapshot against its manifest; raise naming the first mismatch."""
    root = Path(path or DEFAULT_WEIGHTS_DIR)
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"snapshot manifest not found: {manifest_path}")
    with open(manifest_path, encoding="utf-8") as fh:
        manifest = json.load(fh)
    if manifest.get("modelId") != MODEL_ID:
        raise ValueError(f"manifest modelId {manifest.get('modelId')!r} != {MODEL_ID!r}")
    if manifest.get("revision") != MODEL_REVISION:
        raise ValueError(f"manifest revision {manifest.get('revision')!r} != {MODEL_REVISION!r}")
    for entry in manifest.get("files", []):
        file_path = root / entry["path"]
        if not file_path.is_file():
            raise FileNotFoundError(f"snapshot file missing: {file_path}")
        size = file_path.stat().st_size
        if size != entry["bytes"]:
            raise ValueError(f"{entry['path']}: size {size} != manifest {entry['bytes']}")
        digest = _sha256(file_path)
        if digest != entry["sha256"]:
            raise ValueError(f"{entry['path']}: sha256 {digest} != manifest {entry['sha256']}")
    return {"path": str(root), **manifest}


def _hub_download(relative_path: str, root: Path) -> None:
    """Fetch one manifest-listed file at MODEL_REVISION straight into the snapshot directory."""
    from huggingface_hub import hf_hub_download

    hf_hub_download(MODEL_ID, relative_path, revision=MODEL_REVISION, local_dir=str(root))


def stage_missing_files(
    path: str | Path | None = None,
    *,
    allow_download: bool = False,
    downloader: Callable[[str, Path], None] | None = None,
) -> list[str]:
    """Fetch manifest-listed files that are absent locally (a fresh clone commits the manifest and the
    tokenizer/config files but git-ignores the weights). Returns the relative paths fetched;
    `verify_snapshot` still runs after."""
    root = Path(path) if path is not None else DEFAULT_WEIGHTS_DIR
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    with open(manifest_path, encoding="utf-8") as fh:
        manifest = json.load(fh)
    if manifest.get("modelId") != MODEL_ID or manifest.get("revision") != MODEL_REVISION:
        raise ValueError(
            f"manifest names {manifest.get('modelId')}@{manifest.get('revision')}, "
            f"package pins {MODEL_ID}@{MODEL_REVISION}; refusing to stage"
        )
    missing = [entry["path"] for entry in manifest["files"] if not (root / entry["path"]).is_file()]
    if not missing:
        return []
    if not allow_download:
        raise FileNotFoundError(
            f"snapshot at {root} is missing {missing}; "
            f"pass allow_download=True to fetch them at {MODEL_REVISION}"
        )
    fetch = downloader or _hub_download
    for relative_path in missing:
        fetch(relative_path, root)
    return missing


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


def _load_base(
    device: str | None,
    weights_dir: str | Path | None,
    allow_download: bool,
    adapter_dir: str | Path | None,
) -> tuple[Any, Any, str, str, Any]:
    """Shared loader: ``(model, processor, source, device, dtype)`` from a digest-verified local
    snapshot or, only when ``allow_download`` is set and no snapshot exists, from the Hub at the
    pinned revision. There is no silent fallback: a missing or unverified snapshot raises unless
    downloading was explicitly allowed (MOD8). ``trust_remote_code`` is always False."""
    import torch
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

    adapter_path = None if adapter_dir is None else Path(adapter_dir)
    if adapter_path is not None:
        for required in ("adapter_config.json", ADAPTER_WEIGHTS):
            if not (adapter_path / required).is_file():
                raise FileNotFoundError(f"{required} not found in {adapter_path}")
    resolved_device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if resolved_device.startswith("cuda") else torch.float32
    root = Path(weights_dir or DEFAULT_WEIGHTS_DIR)
    if (root / MANIFEST_NAME).is_file():
        stage_missing_files(root, allow_download=allow_download)
        verify_snapshot(root)
        # A directory argument makes transformers read config/tokenizer/weights from it directly
        # (no Hub resolution, no cache lookup).
        location: dict[str, Any] = {"pretrained_model_name_or_path": str(root)}
        source = "local-snapshot"
    elif allow_download:
        location = {"pretrained_model_name_or_path": MODEL_ID, "revision": MODEL_REVISION}
        source = "hf-hub"
    else:
        raise FileNotFoundError(
            f"no verified snapshot at {root} and allow_download=False; "
            f"stage it with: hf download {MODEL_ID} --revision {MODEL_REVISION} --local-dir {root}"
        )
    processor = AutoProcessor.from_pretrained(**location, trust_remote_code=False)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        **location,
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
    return model, processor, source, resolved_device, dtype


def load_model(
    device: str | None = None,
    adapter_dir: str | Path | None = None,
    *,
    weights_dir: str | Path | None = None,
    allow_download: bool = True,
) -> tuple[Any, Any]:
    """Return ``(model, processor)`` for the pinned upstream revision, on ``device``.

    Weights are loaded in float16 on CUDA and float32 on CPU, from the digest-verified snapshot
    under ``weights_dir`` when one exists (the fine-tuning tutorial passes the notebook's staged
    directory) and otherwise from the Hub at the pinned revision. With ``adapter_dir``, a PEFT
    LoRA adapter saved by ``PeftModel.save_pretrained`` is attached and merged into the weights,
    so the result is a plain Whisper model; ``peft`` is only imported on that path, and only a
    safetensors bundle is accepted. Remote model code is always refused.
    """
    model, processor, _source, _device, _dtype = _load_base(device, weights_dir, allow_download, adapter_dir)
    return model, processor


def corpus_word_error_rate(references: Sequence[str], hypotheses: Sequence[str]) -> float:
    """Corpus WER: total word edits over total reference words (``word_error_count`` summed)."""
    if len(references) != len(hypotheses):
        raise ValueError("references and hypotheses must have the same length")
    counts = [
        word_error_count(reference, hypothesis)
        for reference, hypothesis in zip(references, hypotheses, strict=True)
    ]
    total_words = sum(length for _, length in counts)
    if total_words == 0:
        raise ValueError("corpus WER is undefined for empty references")
    return sum(errors for errors, _ in counts) / total_words


ADAPTER_BUNDLE_FORMAT = "peft_adapter"
ADAPTER_BUNDLE_FORMAT_VERSION = 1
ARTIFACT_MANIFEST_NAME = "artifact-manifest.json"


def export_adapter_bundle(
    model: Any, adapter_dir: str | Path, *, metrics: Mapping[str, Any], provenance: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Write the deployable adapter bundle and return its file manifest.

    The bundle is what ``from_pretrained(adapter_dir=...)`` consumes: ``adapter_config.json`` and
    ``adapter_model.safetensors`` from ``PeftModel.save_pretrained`` (safetensors only), plus
    ``metrics.json``, ``provenance.json`` and ``artifact-manifest.json`` (every file with its size
    and SHA-256). Saved LoRA ``B`` matrices that are all zero are refused: such an adapter is a
    no-op and exporting it would hide a training run that never updated anything.
    """
    from safetensors.torch import load_file

    root = Path(adapter_dir)
    if root.exists():
        for stale in sorted(root.rglob("*"), reverse=True):
            stale.unlink() if stale.is_file() else stale.rmdir()
    root.mkdir(parents=True)
    model.save_pretrained(str(root), safe_serialization=True)
    if not (root / ADAPTER_WEIGHTS).is_file():
        raise RuntimeError(f"{ADAPTER_WEIGHTS} was not written; only safetensors adapters are accepted")
    saved_b = [tensor for name, tensor in load_file(str(root / ADAPTER_WEIGHTS)).items() if "lora_B" in name]
    if not saved_b or max(float(tensor.abs().max()) for tensor in saved_b) == 0:
        raise RuntimeError("saved adapter weights are zero or missing")
    (root / "metrics.json").write_text(json.dumps(dict(metrics), indent=2), encoding="utf-8")
    (root / "provenance.json").write_text(json.dumps(dict(provenance), indent=2), encoding="utf-8")
    manifest = [
        {"path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size, "sha256": _sha256(path)}
        for path in sorted(root.rglob("*"))
        if path.is_file()
    ]
    (root / ARTIFACT_MANIFEST_NAME).write_text(
        json.dumps(
            {
                "format": ADAPTER_BUNDLE_FORMAT,
                "formatVersion": ADAPTER_BUNDLE_FORMAT_VERSION,
                "files": manifest,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return manifest


def verify_adapter_merge(
    adapter_dir: str | Path,
    module_name: str,
    base_weight: Any,
    reloaded_weight: Any,
    *,
    rank: int,
    alpha: int,
    tolerance: float,
) -> dict[str, Any]:
    """Weight-level proof that a fresh load applied the saved adapter to the base weights.

    For one probe module, ``W_reloaded`` must equal ``W_base + (alpha / rank) * B @ A`` read back
    from the bundle, within ``tolerance``; an adapter whose delta is below the verification
    resolution is rejected too, because the check would then pass on an unmodified base.
    """
    import torch
    from safetensors.torch import load_file

    saved = load_file(str(Path(adapter_dir) / ADAPTER_WEIGHTS))
    lora_a = saved[f"base_model.model.{module_name}.lora_A.weight"].to(torch.float32)
    lora_b = saved[f"base_model.model.{module_name}.lora_B.weight"].to(torch.float32)
    base = base_weight.detach().to("cpu", torch.float32)
    expected = base + (alpha / rank) * (lora_b @ lora_a)
    reloaded = reloaded_weight.detach().to("cpu", torch.float32)
    adapter_delta = float((expected - base).abs().max())
    merge_error = float((reloaded - expected).abs().max())
    if adapter_delta <= 4 * tolerance:
        raise RuntimeError(
            f"adapter delta {adapter_delta:.2e} on {module_name} is below the {tolerance:.0e} verification "
            "resolution; train longer or with a higher learning rate"
        )
    if merge_error > tolerance:
        raise RuntimeError(
            f"reloaded weights differ from base + scaled B@A by {merge_error:.2e} "
            f"(> {tolerance:.0e}) on {module_name}"
        )
    return {
        "module": module_name,
        "adapter_delta_max": adapter_delta,
        "merge_error_max": merge_error,
        "tolerance": tolerance,
    }


def adaptation_report(
    *,
    baseline_wer: float,
    adapted_wer: float,
    reloaded_wer: float | None,
    n_eval: int,
    history: Sequence[Mapping[str, Any]],
    dataset: Mapping[str, Any],
    sample_kind: str = "public-sample",
) -> dict[str, Any]:
    """Evaluation stage for the fine-tuning tutorial: corpus WER before/after adaptation.

    The verdict is always ``sample-sanity``: one held-out split of one corpus says whether the
    adapter helped *here*, not how it generalises, and ``history`` (teacher-forced losses) is
    optimisation evidence only (FT7).
    """
    metrics = [
        {
            "id": "baseline_wer",
            "value": baseline_wer,
            "estimation": f"corpus WER over {n_eval} held-out utterances, zero-shot",
        },
        {
            "id": "adapted_wer",
            "value": adapted_wer,
            "estimation": f"corpus WER over the same {n_eval} utterances, in-memory adapter",
        },
    ]
    if reloaded_wer is not None:
        metrics.append(
            {
                "id": "reloaded_wer",
                "value": reloaded_wer,
                "estimation": "same split, adapter reloaded from the bundle",
            }
        )
    return {
        "task": "automatic speech recognition (LoRA adaptation)",
        "score_semantics": "generated transcript; the pipeline exposes no confidence score or threshold",
        "sample_kind": sample_kind,
        "n_utterances": n_eval,
        "dataset": dict(dataset),
        "baselines": [
            {"id": "zero_shot_base_model", "word_error_rate": baseline_wer},
        ],
        "metrics": metrics,
        "optimisation_history": [dict(row) for row in history],
        "verdict": "sample-sanity",
        "reason": (
            "one seeded held-out split of one corpus; a few points of WER can change sign with another seed"
        ),
        "needs": (
            "a referenced evaluation set from the deployment domain (speakers, microphones, noise) of "
            "several "
            "hundred utterances, and a check outside the adaptation distribution for forgetting, before any "
            "generalisable claim"
        ),
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
    }


INPUT_SCHEMA: dict[str, Any] = {
    "input": (
        "one audio input: a local file path readable by the ASR stack, an http(s) URL, or a dict "
        "{'array': float waveform, 'sampling_rate': int}"
    ),
    "task": list(TASKS),
    "language": "optional ISO language name/code forwarded to Whisper; None lets the model detect it",
    "chunk_length_s": [MIN_CHUNK_LENGTH_S, MAX_CHUNK_LENGTH_S],
    "preprocessing": (
        "the transformers ASR pipeline resamples to 16 kHz, computes 128-bin log-Mel features in 30 s "
        "windows and chunks longer audio; nothing is altered by this module"
    ),
}


def _check_inputs(audio: Any, task: str, chunk_length_s: int) -> None:
    """Raise ValueError/FileNotFoundError naming the first violated rule (shared with transcribe)."""
    if task not in TASKS:
        raise ValueError("task must be 'transcribe' or 'translate'")
    if (
        isinstance(audio, str | Path)
        and not str(audio).startswith(("http://", "https://"))
        and not Path(audio).is_file()
    ):
        raise FileNotFoundError(f"audio file not found: {audio}")
    if not MIN_CHUNK_LENGTH_S <= chunk_length_s <= MAX_CHUNK_LENGTH_S:
        raise ValueError(
            f"chunk_length_s must be between {MIN_CHUNK_LENGTH_S} and {MAX_CHUNK_LENGTH_S} seconds"
        )


def _observe(audio: Any) -> dict[str, Any]:
    if isinstance(audio, Mapping):
        array = audio.get("array")
        rate = audio.get("sampling_rate")
        samples = len(array) if array is not None and hasattr(array, "__len__") else None
        has_rate = isinstance(rate, int | float) and bool(rate)
        seconds = round(samples / rate, 3) if samples is not None and has_rate else None
        return {"kind": "waveform", "samples": samples, "sampling_rate": rate, "seconds": seconds}
    if isinstance(audio, str | Path) and str(audio).startswith(("http://", "https://")):
        return {"kind": "url", "value": str(audio)}
    path = Path(audio)
    return {"kind": "file", "name": path.name, "bytes": path.stat().st_size}


def validate_inputs(
    audio: str | Path | Mapping[str, Any],
    *,
    language: str | None = None,
    task: str = "transcribe",
    chunk_length_s: int = 30,
    names: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Validation stage: return the input manifest (schema, observed input, request, verdict).

    Rejection is reported by raising exactly as ``transcribe`` would; a caller that wants the
    finding recorded catches the exception and stores ``str(exc)`` under ``findings``.
    """
    _check_inputs(audio, task, chunk_length_s)
    if names is not None and len(names) != 1:
        raise ValueError("names must have exactly one entry (one audio input per call)")
    return {
        "schema": dict(INPUT_SCHEMA),
        "inputs": [{"id": names[0] if names else "audio-0", **_observe(audio)}],
        "task": task,
        "language": language,
        "chunk_length_s": chunk_length_s,
        "verdict": "accepted",
        "findings": [],
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
    }


def evaluation_report(
    result: Mapping[str, Any], reference: str | None = None, *, sample_kind: str = "public-sample"
) -> dict[str, Any]:
    """Evaluation stage: a machine-readable report even when nothing is measurable.

    With a ``reference`` transcript the report carries ``word_error_rate`` (the repository's own
    normalised WER) as sample-sanity evidence for that one utterance; without one the verdict is
    ``not-measurable`` and the report says what would make the task measurable.
    """
    base = {
        "task": f"automatic speech recognition ({result.get('task', 'transcribe')})",
        "score_semantics": "generated transcript; the pipeline exposes no confidence score or threshold",
        "sample_kind": sample_kind,
        "n_utterances": 1,
        "language": result.get("language"),
        "baselines": [],
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
    }
    if reference is None:
        return {
            **base,
            "metrics": [],
            "verdict": "not-measurable",
            "reason": "no reference transcript was supplied for the evaluated audio",
            "needs": (
                "reference transcripts for audio from the deployment domain (speakers, microphones, noise), "
                "scored with word_error_rate after the same normalisation; several hundred utterances before "
                "any rate is quoted"
            ),
        }
    return {
        **base,
        "metrics": [
            {
                "id": "word_error_rate",
                "value": word_error_rate(reference, str(result["text"])),
                "normalisation": "casefold, punctuation removed, curly apostrophes folded",
                "estimation": "single utterance, no dispersion estimate",
            }
        ],
        "verdict": "sample-sanity",
        "reason": "one referenced utterance from the tutorial sample; not a benchmark",
        "needs": "a referenced evaluation set from the deployment domain for any generalisable WER claim",
    }


@dataclass
class WhisperASRPipeline:
    _runner: Callable[..., dict[str, Any]]
    device: str
    adapter: str | None = None
    adapter_sha256: str | None = None
    source: str = "injected"

    @property
    def model(self) -> Any:
        """The underlying Transformers model, for inspection (weights, config, dtype)."""
        return self._runner.model

    @classmethod
    def from_pretrained(
        cls,
        device: str | None = None,
        weights_dir: str | Path | None = None,
        allow_download: bool = False,
        adapter_dir: str | Path | None = None,
    ) -> WhisperASRPipeline:
        """Build the pipeline from a digest-verified local snapshot (or, only when
        ``allow_download`` is set, from the Hub at the pinned revision), optionally merging a
        PEFT LoRA adapter (safetensors only). There is no silent fallback: a missing or
        unverified snapshot raises unless downloading was explicitly allowed."""
        model, processor, source, resolved_device, dtype = _load_base(
            device, weights_dir, allow_download, adapter_dir
        )
        adapter_path = None if adapter_dir is None else Path(adapter_dir)
        return cls.from_model(
            model,
            processor,
            adapter=None if adapter_path is None else str(adapter_path),
            adapter_sha256=None if adapter_path is None else adapter_digest(adapter_path),
            source=source,
            device=resolved_device,
            dtype=dtype,
        )

    @classmethod
    def from_model(
        cls,
        model: Any,
        processor: Any,
        adapter: str | None = None,
        adapter_sha256: str | None = None,
        source: str = "injected",
        device: str | None = None,
        dtype: Any | None = None,
    ) -> WhisperASRPipeline:
        """Wrap an already-loaded Whisper model (plain or PEFT-wrapped) in the same decoding path.

        Lets a caller that holds a live model — the fine-tuning tutorial, between training and
        export — transcribe through exactly the pipeline that ``from_pretrained`` builds, so
        in-memory and reloaded results are comparable.
        """
        from transformers import pipeline

        resolved_device = device or str(model.device)
        runner = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            torch_dtype=model.dtype if dtype is None else dtype,
            device=resolved_device,
        )
        return cls(runner, resolved_device, adapter, adapter_sha256, source)

    def transcribe(
        self,
        audio: str | Path | dict[str, Any],
        *,
        language: str | None = None,
        task: str = "transcribe",
        return_timestamps: bool = False,
        chunk_length_s: int = 30,
    ) -> dict[str, Any]:
        _check_inputs(audio, task, chunk_length_s)

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
            "device": self.device,
            "adapter": self.adapter,
            "adapter_sha256": self.adapter_sha256,
            "source": self.source,
        }
