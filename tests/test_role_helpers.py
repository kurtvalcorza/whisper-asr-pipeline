"""Offline tests for the public validation and evaluation stage helpers (DAT24 / EVAL21)."""

from __future__ import annotations

from pathlib import Path

import pytest

from whisper_asr_pipeline import (
    INPUT_SCHEMA,
    MAX_CHUNK_LENGTH_S,
    MIN_CHUNK_LENGTH_S,
    MODEL_ID,
    MODEL_REVISION,
    WhisperASRPipeline,
    evaluation_report,
    validate_inputs,
)

WAVEFORM = {"array": [0.0] * 32_000, "sampling_rate": 16_000}


def _result(text: str = "hello world") -> dict:
    return {"text": text, "chunks": None, "task": "transcribe", "language": "en"}


def test_validate_inputs_returns_manifest_with_schema_and_identity() -> None:
    manifest = validate_inputs(WAVEFORM, language="en", names=["utterance"])
    assert manifest["verdict"] == "accepted"
    assert manifest["findings"] == []
    assert manifest["schema"] == INPUT_SCHEMA
    assert manifest["schema"]["chunk_length_s"] == [MIN_CHUNK_LENGTH_S, MAX_CHUNK_LENGTH_S]
    assert manifest["inputs"] == [
        {"id": "utterance", "kind": "waveform", "samples": 32_000, "sampling_rate": 16_000, "seconds": 2.0}
    ]
    assert (manifest["task"], manifest["language"], manifest["chunk_length_s"]) == ("transcribe", "en", 30)
    assert (manifest["model_id"], manifest["model_revision"]) == (MODEL_ID, MODEL_REVISION)


def test_validate_inputs_observes_files_and_urls(tmp_path: Path) -> None:
    audio = tmp_path / "clip.wav"
    audio.write_bytes(b"RIFF" + b"\x00" * 40)
    manifest = validate_inputs(audio)
    assert manifest["inputs"] == [{"id": "audio-0", "kind": "file", "name": "clip.wav", "bytes": 44}]
    manifest = validate_inputs("https://example.invalid/clip.wav", task="translate")
    assert manifest["inputs"][0]["kind"] == "url"
    assert manifest["task"] == "translate"


def test_validate_inputs_rejects_like_transcribe(tmp_path: Path) -> None:
    pipe = WhisperASRPipeline(lambda audio, **kw: {"text": "x"}, "cpu")
    with pytest.raises(ValueError, match="task must be"):
        validate_inputs(WAVEFORM, task="diarize")
    with pytest.raises(ValueError, match="task must be"):
        pipe.transcribe(WAVEFORM, task="diarize")
    with pytest.raises(ValueError, match="chunk_length_s must be between 1 and 30"):
        validate_inputs(WAVEFORM, chunk_length_s=MAX_CHUNK_LENGTH_S + 1)
    with pytest.raises(ValueError, match="chunk_length_s must be between 1 and 30"):
        pipe.transcribe(WAVEFORM, chunk_length_s=MAX_CHUNK_LENGTH_S + 1)
    with pytest.raises(FileNotFoundError):
        validate_inputs(tmp_path / "missing.wav")
    with pytest.raises(ValueError, match="names must have exactly one entry"):
        validate_inputs(WAVEFORM, names=["a", "b"])


def test_evaluation_report_not_measurable_without_reference() -> None:
    report = evaluation_report(_result(), sample_kind="synthetic")
    assert report["verdict"] == "not-measurable"
    assert report["metrics"] == []
    assert "reference transcripts" in report["needs"]
    assert report["sample_kind"] == "synthetic"
    assert (report["model_id"], report["model_revision"]) == (MODEL_ID, MODEL_REVISION)


def test_evaluation_report_sample_sanity_with_reference() -> None:
    report = evaluation_report(_result("Hello, world."), "hello world")
    assert report["verdict"] == "sample-sanity"
    assert report["metrics"] == [
        {
            "id": "word_error_rate",
            "value": 0.0,
            "normalisation": "casefold, punctuation removed, curly apostrophes folded",
            "estimation": "single utterance, no dispersion estimate",
        }
    ]
    report = evaluation_report(_result("hello"), "hello world")
    assert report["metrics"][0]["value"] == 0.5
