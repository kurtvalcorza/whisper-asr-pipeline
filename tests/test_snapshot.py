"""Offline tests for the fleet snapshot scheme: manifest-driven verification, staging, loading."""

from __future__ import annotations

import hashlib
import json
import sys
import types
from pathlib import Path

import pytest

from whisper_asr_pipeline import (
    MODEL_ID,
    MODEL_KEY,
    MODEL_REVISION,
    WhisperASRPipeline,
    stage_missing_files,
    verify_snapshot,
)
from whisper_asr_pipeline.pipeline import MANIFEST_NAME

ROOT = Path(__file__).resolve().parents[1]
COMMITTED_MANIFEST = ROOT / "weights" / MODEL_KEY / MANIFEST_NAME
LOADER_FILES = {
    "config.json",
    "generation_config.json",
    "preprocessor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "added_tokens.json",
    "special_tokens_map.json",
    "normalizer.json",
    "model.safetensors",
}
FILES = {
    "README.md": b"# Whisper\n",
    "config.json": b'{"model_type": "whisper"}\n',
    "model.safetensors": b"\x00" * 64,
}


def _write_snapshot(root: Path, *, files: dict[str, bytes] | None = None, model_id: str = MODEL_ID) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    files = FILES if files is None else files
    entries = []
    for name, payload in files.items():
        entries.append({"path": name, "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()})
    manifest = {
        "format": "dimer_hf_snapshot",
        "formatVersion": 1,
        "modelKey": MODEL_KEY,
        "modelId": model_id,
        "revision": MODEL_REVISION,
        "files": entries,
        "totalBytes": sum(e["bytes"] for e in entries),
    }
    (root / MANIFEST_NAME).write_text(json.dumps(manifest), encoding="utf-8")
    return manifest


def _materialise(root: Path, files: dict[str, bytes] | None = None) -> None:
    for name, payload in (FILES if files is None else files).items():
        (root / name).write_bytes(payload)


class _Calls:
    def __init__(self) -> None:
        self.processor: list = []
        self.model: list = []
        self.pipeline: list = []


def _stub_stack(monkeypatch, calls: _Calls, cuda: bool = False) -> None:
    fake_torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=lambda: cuda), float16="fp16", float32="fp32"
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    class AutoProcessor:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            calls.processor.append((args, kwargs))
            return types.SimpleNamespace(tokenizer="tok", feature_extractor="fe")

    class AutoModelForSpeechSeq2Seq:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            calls.model.append((args, kwargs))
            return types.SimpleNamespace(to=lambda device: "model-on-" + device)

    def pipeline(*args, **kwargs):
        calls.pipeline.append((args, kwargs))
        return lambda audio, **kw: {"text": " hi ", "chunks": None}

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        types.SimpleNamespace(
            AutoProcessor=AutoProcessor,
            AutoModelForSpeechSeq2Seq=AutoModelForSpeechSeq2Seq,
            pipeline=pipeline,
        ),
    )


def test_committed_manifest_names_the_pinned_identity_and_loader_files() -> None:
    manifest = json.loads(COMMITTED_MANIFEST.read_text(encoding="utf-8"))
    identity = (manifest["modelId"], manifest["revision"], manifest["modelKey"])
    assert identity == (MODEL_ID, MODEL_REVISION, MODEL_KEY)
    paths = {entry["path"] for entry in manifest["files"]}
    assert paths >= LOADER_FILES, "AutoProcessor/AutoModelForSpeechSeq2Seq need every listed file"
    assert all(len(entry["sha256"]) == 64 for entry in manifest["files"])
    assert manifest["totalBytes"] == sum(entry["bytes"] for entry in manifest["files"])


def test_verify_snapshot_accepts_matching_files(tmp_path: Path) -> None:
    manifest = _write_snapshot(tmp_path)
    _materialise(tmp_path)
    result = verify_snapshot(tmp_path)
    assert result["path"] == str(tmp_path)
    assert result["files"] == manifest["files"]


def test_verify_snapshot_rejects_tampered_digest(tmp_path: Path) -> None:
    _write_snapshot(tmp_path)
    _materialise(tmp_path)
    (tmp_path / "model.safetensors").write_bytes(b"\x01" * 64)  # same size, different bytes
    with pytest.raises(ValueError, match="model.safetensors: sha256"):
        verify_snapshot(tmp_path)


def test_verify_snapshot_rejects_wrong_identity(tmp_path: Path) -> None:
    _write_snapshot(tmp_path, model_id="someone/else")
    _materialise(tmp_path)
    with pytest.raises(ValueError, match="modelId"):
        verify_snapshot(tmp_path)
    with pytest.raises(ValueError, match="refusing to stage"):
        stage_missing_files(tmp_path, allow_download=True, downloader=lambda *_: None)


def test_stage_missing_files_fetches_only_absent_entries(tmp_path: Path) -> None:
    _write_snapshot(tmp_path)
    _materialise(tmp_path, {"README.md": FILES["README.md"], "config.json": FILES["config.json"]})
    with pytest.raises(FileNotFoundError, match="allow_download=True"):
        stage_missing_files(tmp_path)
    fetched: list[str] = []

    def downloader(relative_path: str, root: Path) -> None:
        fetched.append(relative_path)
        (root / relative_path).write_bytes(FILES[relative_path])

    assert stage_missing_files(tmp_path, allow_download=True, downloader=downloader) == ["model.safetensors"]
    assert fetched == ["model.safetensors"]
    assert stage_missing_files(tmp_path, allow_download=True, downloader=downloader) == []
    verify_snapshot(tmp_path)


def test_from_pretrained_loads_verified_directory_no_remote_code(monkeypatch, tmp_path: Path) -> None:
    _write_snapshot(tmp_path)
    _materialise(tmp_path)
    calls = _Calls()
    _stub_stack(monkeypatch, calls)
    pipe = WhisperASRPipeline.from_pretrained(weights_dir=tmp_path)
    assert (pipe.source, pipe.device) == ("local-snapshot", "cpu")
    location = {"pretrained_model_name_or_path": str(tmp_path), "trust_remote_code": False}
    assert calls.processor == [((), location)]
    assert calls.model[0][1]["pretrained_model_name_or_path"] == str(tmp_path)
    assert calls.model[0][1]["trust_remote_code"] is False
    assert calls.model[0][1]["torch_dtype"] == "fp32"
    assert calls.pipeline[0][1]["device"] == "cpu"
    assert pipe.transcribe({"array": [0.0], "sampling_rate": 16_000})["source"] == "local-snapshot"


def test_from_pretrained_hub_path_only_with_allow_download(monkeypatch, tmp_path: Path) -> None:
    calls = _Calls()
    _stub_stack(monkeypatch, calls)
    with pytest.raises(FileNotFoundError, match="allow_download=False"):
        WhisperASRPipeline.from_pretrained(weights_dir=tmp_path)
    pipe = WhisperASRPipeline.from_pretrained(weights_dir=tmp_path, allow_download=True)
    assert pipe.source == "hf-hub"
    expected = {
        "pretrained_model_name_or_path": MODEL_ID,
        "revision": MODEL_REVISION,
        "trust_remote_code": False,
    }
    assert calls.processor == [((), expected)]
    assert calls.model[0][1]["revision"] == MODEL_REVISION


def test_from_pretrained_refuses_a_tampered_snapshot(monkeypatch, tmp_path: Path) -> None:
    _write_snapshot(tmp_path)
    _materialise(tmp_path)
    (tmp_path / "config.json").write_bytes(b'{"model_type": "other"}\n')
    _stub_stack(monkeypatch, _Calls())
    with pytest.raises(ValueError, match="config.json"):
        WhisperASRPipeline.from_pretrained(weights_dir=tmp_path)
