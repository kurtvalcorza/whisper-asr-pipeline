"""Offline inference contracts; no real weights or GPU execution."""

import hashlib
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import audio_models as am  # noqa: E402


@pytest.mark.parametrize("name", ["../escape", "/root", "C:/outside", "a\\b"])
def test_snapshot_refuses_unsafe_names(tmp_path, name):
    with pytest.raises(ValueError):
        am.verify_snapshot(tmp_path, {"revision": "a" * 40, "files": [{"path": name}]})


def test_snapshot_cache_tamper_and_revision(tmp_path):
    path = tmp_path / "weights"
    path.write_bytes(b"safe")
    manifest = {
        "revision": "a" * 40,
        "files": [{"path": "weights", "bytes": 4, "sha256": hashlib.sha256(b"safe").hexdigest()}],
    }
    am.verify_snapshot(tmp_path, manifest)
    path.write_bytes(b"evil")
    with pytest.raises(ValueError, match="integrity"):
        am.verify_snapshot(tmp_path, manifest)
    manifest["revision"] = "main"
    with pytest.raises(ValueError, match="immutable"):
        am.verify_snapshot(tmp_path, manifest)


@pytest.mark.parametrize(
    "wave", [np.ones(31999), np.ones(400001), np.zeros(32000), np.ones((32000, 2)), np.full(32000, np.nan)]
)
def test_audio_refuses_invalid_or_truncation(wave):
    with pytest.raises(ValueError):
        am.validate_wave(wave)


def test_audio_preserves_full_signal():
    wave = np.linspace(-1, 1, 400000, dtype=np.float32)
    assert np.array_equal(am.validate_wave(wave), wave)


def test_stopping_needs_eos_after_prefix():
    am.verify_stopping([1, 2, 3, 4, 9, 50257], 50257)
    with pytest.raises(ValueError, match="truncation"):
        am.verify_stopping([1, 50257, 3, 4, 9], 50257)


def test_query_instruction_and_empty_documents():
    assert am.embedding_text("texto") == "texto"
    assert am.embedding_text("") == ""
    assert am.embedding_text("tanong", True) == f"Instruct: {am.INSTRUCTION}\nQuery:tanong"
    with pytest.raises(ValueError):
        am.embedding_text("", True)


class CharacterTokenizer:
    def encode(self, text, add_special_tokens):
        assert add_special_tokens is False
        return [ord(char) for char in text]


def test_pair_template_exact_and_no_silent_truncation():
    ids = am.pair_ids(CharacterTokenizer(), "Ano?", "Sagot")
    assert "".join(map(chr, ids)) == (
        am.PREFIX + f"<Instruct>: {am.INSTRUCTION}\n<Query>: Ano?\n<Document>: Sagot" + am.SUFFIX
    )
    assert am.pair_ids(CharacterTokenizer(), "Ano?", "")
    with pytest.raises(ValueError, match="truncation"):
        am.pair_ids(CharacterTokenizer(), "Ano?", "a" * 8192)


class FakeBatch(dict):
    def to(self, _device):
        return self


class FakeTokenizer:
    def __call__(self, texts, padding, truncation):
        assert padding is False and truncation is False
        return {"input_ids": [[1] * (len(text) + 1) for text in texts]}

    def pad(self, encoded, padding, return_tensors):
        import torch

        maximum = max(map(len, encoded["input_ids"]))
        ids = [[0] * (maximum - len(row)) + row for row in encoded["input_ids"]]
        return FakeBatch(input_ids=torch.tensor(ids), attention_mask=torch.tensor(ids) != 0)

    def encode(self, text, add_special_tokens):
        return [1] * len(text)


def test_embedding_last_token_norm_empty_retention_and_refusal():
    torch = pytest.importorskip("torch")

    class Model:
        def __call__(self, input_ids, attention_mask):
            batch, length = input_ids.shape
            hidden = torch.ones(batch, length, 1024)
            hidden[:, -1, 0] = 5
            return SimpleNamespace(last_hidden_state=hidden)

    handle = {"kind": "embedding", "tokenizer": FakeTokenizer(), "model": Model(), "device": "cpu"}
    values = am.embed(handle, ["", "test", "longer"])
    assert values.shape == (3, 1024)
    assert np.allclose(np.linalg.norm(values, axis=1), 1)
    assert values[0, 0] == pytest.approx(5 * values[0, 1])
    with pytest.raises(ValueError, match="truncation"):
        am.embed(handle, ["x" * 8192])


def test_reranker_yes_no_softmax_last_logits():
    torch = pytest.importorskip("torch")

    class Model:
        def __call__(self, input_ids, attention_mask, logits_to_keep):
            assert logits_to_keep == 1
            logits = torch.zeros(1, 1, 10000)
            logits[0, 0, 9693] = np.log(3)
            return SimpleNamespace(logits=logits)

    handle = {"kind": "reranker", "tokenizer": FakeTokenizer(), "model": Model(), "device": "cpu"}
    assert np.allclose(am.rerank(handle, "Tanong?", ["Sagot", ""]), [0.75, 0.75])


def test_asr_passes_independent_settings_and_refuses_cap():
    torch = pytest.importorskip("torch")

    class Processor:
        def __call__(self, wave, **kwargs):
            assert kwargs["truncation"] is False and kwargs["sampling_rate"] == 16000
            return {"input_features": torch.zeros(1, 128, 3000), "attention_mask": torch.ones(1, 3000)}

        def batch_decode(self, rows, **_kwargs):
            return [" hello "]

    class Model:
        generation_config = SimpleNamespace(eos_token_id=50257)
        capped = False

        def generate(self, **kwargs):
            assert kwargs["language"] == "tl" and kwargs["task"] == "transcribe"
            assert kwargs["do_sample"] is False and kwargs["condition_on_prev_tokens"] is False
            return SimpleNamespace(sequences=torch.tensor([[1, 2, 3, 4, 9, 9 if self.capped else 50257]]))

    model = Model()
    handle = {"kind": "whisper", "processor": Processor(), "model": model, "device": "cpu"}
    assert am.transcribe(handle, [np.ones(32000), np.ones(400000)]) == ["hello", "hello"]
    model.capped = True
    with pytest.raises(ValueError, match="truncation"):
        am.transcribe(handle, [np.ones(32000)])


def test_upstream_whisper_generate_contract_tiny_random_cpu():
    """Exercise actual upstream generation kwargs without loading pretrained weights."""
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    config = transformers.WhisperConfig(
        vocab_size=51866,
        num_mel_bins=4,
        d_model=8,
        encoder_layers=1,
        decoder_layers=1,
        encoder_attention_heads=2,
        decoder_attention_heads=2,
        encoder_ffn_dim=16,
        decoder_ffn_dim=16,
        max_source_positions=4,
        max_target_positions=448,
    )
    model = transformers.WhisperForConditionalGeneration(config).eval()
    model.generation_config.lang_to_id = {"<|tl|>": 50348}
    model.generation_config.task_to_id = {"transcribe": 50360}
    model.generation_config.no_timestamps_token_id = 50364
    model.generation_config.is_multilingual = True
    model.generation_config.forced_decoder_ids = None
    settings = dict(am.SETTINGS["asr"], max_new_tokens=2)
    with torch.inference_mode():
        result = model.generate(input_features=torch.zeros(1, 4, 8), **settings)
    assert result.sequences[0, :4].tolist() == [50257, 50348, 50360, 50364]
    assert 4 < result.sequences.shape[1] <= 6
