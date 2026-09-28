"""Pinned upstream inference, with bounded inputs and no silent truncation.

No package from a DIMER worker is imported. Heavy libraries load only inside
inference functions; contracts can be exercised on CPU with synthetic objects.
"""

from __future__ import annotations

import gc
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np

INSTRUCTION = "Given a web search query, retrieve relevant passages that answer the query"
PREFIX = (
    "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the "
    'Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
)
SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
SETTINGS = {
    "instruction": INSTRUCTION,
    "embedding_pooling": "last_nonpadding_token",
    "embedding_normalization": "L2_float32",
    "text_token_limit": 8192,
    "text_truncation": "refuse",
    "embedding_batch_size": 4,
    "reranker_batch_size": 1,
    "reranker_prefix": PREFIX,
    "reranker_suffix": SUFFIX,
    "reranker_score": "softmax([no,yes])[yes]; not calibrated",
    "empty_document_policy": (
        "retain; embedding tokenizer special-token-only input; reranker empty Document prompt"
    ),
    "asr": {
        "language": "tl",
        "task": "transcribe",
        "do_sample": False,
        "num_beams": 1,
        "max_new_tokens": 440,
        "condition_on_prev_tokens": False,
        "return_timestamps": False,
        "return_dict_in_generate": True,
        "temperature": 0.0,
        "compression_ratio_threshold": None,
        "logprob_threshold": None,
        "no_speech_threshold": None,
    },
}


def file_digest(path: Path) -> str:
    """Hash without reading a multi-gigabyte weight into RAM."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_snapshot(directory: Path, manifest: dict) -> None:
    """Verify byte counts, hashes and bounded relative names, including cache hits."""
    if not re.fullmatch(r"[0-9a-f]{40}", manifest["revision"]):
        raise ValueError("An immutable full model revision is required")
    names = set()
    for row in manifest["files"]:
        name = row["path"]
        if not isinstance(name, str) or "\\" in name or ":" in name:
            raise ValueError("Unsafe snapshot file name")
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or name in names:
            raise ValueError("Unsafe or duplicate snapshot file name")
        names.add(name)
        path = directory / relative
        if not path.resolve().is_relative_to(directory.resolve()):
            raise ValueError("Snapshot path escapes cache")
        if not path.is_file() or path.stat().st_size != row["bytes"] or file_digest(path) != row["sha256"]:
            raise ValueError(f"Snapshot integrity failure: {name}")


def acquire_snapshot(root: Path, manifest: dict) -> Path:
    """Download only inventoried files, then verify every file before loading."""
    from huggingface_hub import snapshot_download

    destination = root / "models" / manifest["modelKey"]
    # Verify before download to avoid overwriting an already corrupt cache silently.
    if destination.exists() and all((destination / f["path"]).exists() for f in manifest["files"]):
        verify_snapshot(destination, manifest)
        return destination
    snapshot_download(
        manifest["modelId"],
        revision=manifest["revision"],
        allow_patterns=[row["path"] for row in manifest["files"]],
        local_dir=destination,
        max_workers=2,
    )
    verify_snapshot(destination, manifest)
    return destination


def load_model(kind: str, root: Path | str) -> dict[str, Any]:
    """Load one verified model on a hosted CUDA device. Caller unloads between families."""
    import torch
    from transformers import (
        AutoModel,
        AutoModelForCausalLM,
        AutoProcessor,
        AutoTokenizer,
        WhisperForConditionalGeneration,
    )

    if kind not in ("whisper", "embedding", "reranker"):
        raise ValueError(f"Unknown model kind: {kind}")
    if not torch.cuda.is_available():
        raise RuntimeError("Model inference requires a hosted Colab/Kaggle GPU; local validation is CPU-only")
    root = Path(root)
    path = root / "model_manifest.json"
    if not path.exists():
        path = Path(__file__).with_name("audio_models.json")
    manifest = json.loads(path.read_text(encoding="utf-8"))[kind]
    snapshot = acquire_snapshot(root, manifest)
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    options = dict(local_files_only=True, trust_remote_code=False)
    constructor = {
        "whisper": WhisperForConditionalGeneration,
        "embedding": AutoModel,
        "reranker": AutoModelForCausalLM,
    }[kind]
    model = (
        constructor.from_pretrained(
            snapshot,
            **options,
            torch_dtype=torch.float16,
            use_safetensors=True,
            attn_implementation="sdpa",
        )
        .to("cuda")
        .eval()
    )
    handle = {"kind": kind, "model": model, "device": "cuda", "manifest": manifest}
    if kind == "whisper":
        processor = AutoProcessor.from_pretrained(snapshot, **options)
        language_id = processor.tokenizer.convert_tokens_to_ids("<|tl|>")
        if model.generation_config.lang_to_id.get("<|tl|>") != language_id:
            raise ValueError("Pinned tokenizer and generation language token disagree")
        prompt = processor.get_decoder_prompt_ids(language="tl", task="transcribe", no_timestamps=True)
        prefix_length = 1 + len(prompt)
        if prefix_length + SETTINGS["asr"]["max_new_tokens"] > model.config.max_target_positions:
            raise ValueError("ASR generation budget exceeds decoder context")
        model.generation_config.forced_decoder_ids = None
        handle.update(
            processor=processor,
            language_token_id=language_id,
            prefix_length=prefix_length,
            generation_config=model.generation_config.to_dict(),
            generation_overrides=dict(SETTINGS["asr"]),
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(snapshot, **options, padding_side="left")
        if kind == "reranker" and [tokenizer.convert_tokens_to_ids(x) for x in ("no", "yes")] != [2152, 9693]:
            raise ValueError("Pinned reranker yes/no token identities changed")
        handle["tokenizer"] = tokenizer
    return handle


def validate_wave(wave: np.ndarray) -> np.ndarray:
    """Accept already resampled, mono, non-silent 2–25 second float waveforms only."""
    wave = np.asarray(wave, dtype=np.float32)
    if wave.ndim != 1 or not 32000 <= len(wave) <= 400000:
        raise ValueError("Expected mono 16 kHz audio lasting 2–25 seconds; no truncation")
    if not np.isfinite(wave).all() or not np.any(wave != 0):
        raise ValueError("Audio must be finite and non-silent")
    return wave


def verify_stopping(sequence: list[int], eos_token_id: int, prefix_length: int = 4) -> None:
    """A real EOS after the forced prefix is required; reaching the cap fails."""
    if eos_token_id not in sequence[prefix_length:]:
        raise ValueError("Whisper generation did not reach EOS; possible length-cap truncation")


def transcribe(handle: dict, waves: list[np.ndarray]) -> list[str]:
    """Greedy, independent Filipino transcription; reject length-cap outputs."""
    import torch

    if handle["kind"] != "whisper":
        raise ValueError("ASR requires a Whisper handle")
    processor, model = handle["processor"], handle["model"]
    result = []
    for wave in waves:
        wave = validate_wave(wave)
        inputs = processor(
            wave,
            sampling_rate=16000,
            return_tensors="pt",
            truncation=False,
            padding="max_length",
            return_attention_mask=True,
        )
        inputs = {
            key: value.to(handle["device"], dtype=torch.float16 if key == "input_features" else value.dtype)
            for key, value in inputs.items()
        }
        with torch.inference_mode():
            generated = model.generate(**inputs, **SETTINGS["asr"])
        ids = generated.sequences[0].detach().cpu().tolist()
        verify_stopping(ids, model.generation_config.eos_token_id, handle.get("prefix_length", 4))
        result.append(processor.batch_decode([ids], skip_special_tokens=True)[0].strip())
    return result


def bounded_text(text: str) -> str:
    if not isinstance(text, str) or len(text) > 100000:
        raise ValueError("Text must be a string of at most 100000 characters")
    return text


def embedding_text(text: str, query: bool = False) -> str:
    text = bounded_text(text)
    if query and not text.strip():
        raise ValueError("Empty search query")
    return f"Instruct: {INSTRUCTION}\nQuery:{text}" if query else text


def pair_ids(tokenizer: Any, query: str, document: str) -> list[int]:
    """Exact upstream three-part tokenization, with an explicit length refusal."""
    bounded_text(query)
    bounded_text(document)
    if not query.strip():
        raise ValueError("Empty search query")
    body = f"<Instruct>: {INSTRUCTION}\n<Query>: {query}\n<Document>: {document}"
    ids = sum((tokenizer.encode(x, add_special_tokens=False) for x in (PREFIX, body, SUFFIX)), [])
    if len(ids) > SETTINGS["text_token_limit"]:
        raise ValueError("Reranker prompt exceeds token limit; truncation forbidden")
    return ids


def embed(handle: dict, texts: list[str], query: bool = False) -> np.ndarray:
    """Official query-only instruction, left padding, last-token pooling and L2 norm."""
    import torch

    if handle["kind"] != "embedding":
        raise ValueError("Embedding requires an embedding handle")
    if not texts:
        return np.empty((0, 1024), dtype=np.float32)
    tokenizer, model = handle["tokenizer"], handle["model"]
    rows = []
    for start in range(0, len(texts), SETTINGS["embedding_batch_size"]):
        batch = [
            embedding_text(text, query) for text in texts[start : start + SETTINGS["embedding_batch_size"]]
        ]
        encoded = tokenizer(batch, padding=False, truncation=False)
        if any(len(ids) > SETTINGS["text_token_limit"] or not ids for ids in encoded["input_ids"]):
            raise ValueError("Empty tokenization or embedding token limit exceeded; truncation forbidden")
        inputs = tokenizer.pad(encoded, padding=True, return_tensors="pt").to(handle["device"])
        if not bool(inputs["attention_mask"][:, -1].all()):
            raise ValueError("Embedding tokenizer must use left padding")
        with torch.inference_mode():
            vectors = model(**inputs).last_hidden_state[:, -1].float()
            if not bool(torch.isfinite(vectors).all()) or bool((vectors.norm(dim=1) == 0).any()):
                raise ValueError("Invalid embedding vector")
            vectors = torch.nn.functional.normalize(vectors, p=2, dim=1)
        rows.append(vectors.cpu().numpy())
    return np.concatenate(rows).astype(np.float32)


def rerank(handle: dict, query: str, docs: list[str]) -> np.ndarray:
    """Score one pair at a time to bound full-vocabulary logits memory on T4."""
    import torch

    if handle["kind"] != "reranker":
        raise ValueError("Reranking requires a reranker handle")
    tokenizer, model = handle["tokenizer"], handle["model"]
    result = []
    for doc in docs:
        ids = pair_ids(tokenizer, query, doc)
        inputs = tokenizer.pad({"input_ids": [ids]}, padding=True, return_tensors="pt").to(handle["device"])
        with torch.inference_mode():
            # Qwen3 supports logits_to_keep: avoid allocating [length, 151936] logits.
            logits = model(**inputs, logits_to_keep=1).logits[0, -1, [2152, 9693]].float()
            if not bool(torch.isfinite(logits).all()):
                raise ValueError("Non-finite reranker logits")
            result.append(torch.softmax(logits, dim=0)[1].item())
    return np.asarray(result, dtype=np.float32)


def unload(handle: dict) -> None:
    """Release references and CUDA allocations before loading the next family."""
    import torch

    handle.clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
