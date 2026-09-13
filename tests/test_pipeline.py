import hashlib

import pytest

from whisper_asr_pipeline import (
    WhisperASRPipeline,
    adapter_digest,
    load_model,
    word_error_count,
    word_error_rate,
)


def test_wer():
    assert word_error_rate("hello world", "hello world") == 0.0
    assert word_error_rate("hello world", "hello") == 0.5


def test_wer_ignores_case_and_punctuation_but_not_spelling():
    reference = "MISTER QUILTER IS THE APOSTLE OF THE MIDDLE CLASSES AND WE ARE GLAD"
    hypothesis = "Mr. Quilter is the apostle of the middle classes, and we are glad."
    assert word_error_rate("hello world", "Hello, world.") == 0.0
    assert word_error_rate("don't stop", "Don't stop!") == 0.0
    assert word_error_rate("don't stop", "Don\u2019t stop") == 0.0
    assert word_error_rate("stra\u00dfe", "STRASSE") == 0.0
    assert word_error_rate("well-known", "well known") == 2.0  # one reference token, two errors
    assert word_error_rate(reference, hypothesis) == 1 / 13


def test_normalized_output():
    calls = {}

    def fake(audio, **kwargs):
        calls.update(kwargs)
        return {"text": " hello world ", "chunks": []}

    pipeline = WhisperASRPipeline(fake, "cpu")
    result = pipeline.transcribe(
        {"array": [0.0], "sampling_rate": 16_000},
        language="en",
    )
    assert result["text"] == "hello world"
    assert result["model_revision"]
    assert calls["generate_kwargs"]["language"] == "en"


def test_word_error_count_sums_to_corpus_wer():
    pairs = [("hello world", "hello world"), ("hello world", "hello"), ("", "extra")]
    counts = [word_error_count(reference, hypothesis) for reference, hypothesis in pairs]
    assert counts == [(0, 2), (1, 2), (1, 0)]
    errors = sum(error for error, _ in counts)
    length = sum(length for _, length in counts)
    assert errors / length == 0.5
    assert word_error_rate("", "extra") == 1.0
    assert word_error_rate("", "") == 0.0


def test_load_model_refuses_directory_without_adapter_config(tmp_path):
    with pytest.raises(FileNotFoundError, match="adapter_config.json"):
        load_model(adapter_dir=tmp_path)


def test_load_model_refuses_pickle_only_adapter_bundle(tmp_path):
    # peft would fall back to adapter_model.bin (pickle); the loader must fail closed instead.
    (tmp_path / "adapter_config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "adapter_model.bin").write_bytes(b"not a safetensors file")
    with pytest.raises(FileNotFoundError, match="adapter_model.safetensors"):
        load_model(adapter_dir=tmp_path)


def test_adapter_digest_is_sha256_of_the_weights_file(tmp_path):
    (tmp_path / "adapter_model.safetensors").write_bytes(b"weights")
    assert adapter_digest(tmp_path) == hashlib.sha256(b"weights").hexdigest()


def test_transcribe_records_adapter_identity():
    def fake(audio, **kwargs):
        return {"text": "hi", "chunks": None}

    audio = {"array": [0.0], "sampling_rate": 16_000}
    plain = WhisperASRPipeline(fake, "cpu").transcribe(audio)
    assert plain["adapter"] is None and plain["adapter_sha256"] is None
    adapted = WhisperASRPipeline(fake, "cpu", adapter="outputs/adapter", adapter_sha256="ab" * 32)
    result = adapted.transcribe(audio)
    assert result["adapter"] == "outputs/adapter" and result["adapter_sha256"] == "ab" * 32
