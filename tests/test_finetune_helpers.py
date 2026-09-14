"""Offline tests for the fine-tuning helpers the standalone E2E notebook carries (no model download)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from whisper_asr_pipeline.pipeline import (
    ADAPTER_WEIGHTS,
    MODEL_ID,
    MODEL_REVISION,
    adaptation_report,
    corpus_word_error_rate,
    export_adapter_bundle,
    verify_adapter_merge,
)

torch = pytest.importorskip("torch")
safetensors_torch = pytest.importorskip("safetensors.torch")


class _FakePeftModel:
    """Writes the two files ``PeftModel.save_pretrained`` writes, with controllable LoRA B weights."""

    def __init__(self, b_scale: float) -> None:
        self.b_scale = b_scale

    def save_pretrained(self, path: str, safe_serialization: bool = True) -> None:
        assert safe_serialization
        root = Path(path)
        (root / "adapter_config.json").write_text(json.dumps({"r": 2, "lora_alpha": 4}), encoding="utf-8")
        tensors = {
            "base_model.model.probe.lora_A.weight": torch.ones(2, 3),
            "base_model.model.probe.lora_B.weight": torch.full((3, 2), self.b_scale),
        }
        safetensors_torch.save_file(tensors, str(root / ADAPTER_WEIGHTS))


def test_corpus_wer_is_edits_over_reference_words() -> None:
    wer = corpus_word_error_rate(["a b c", "d e"], ["a b x", "d e"])
    assert wer == pytest.approx(1 / 5)


def test_corpus_wer_rejects_mismatched_lengths_and_empty_references() -> None:
    with pytest.raises(ValueError, match="same length"):
        corpus_word_error_rate(["a"], [])
    with pytest.raises(ValueError, match="undefined"):
        corpus_word_error_rate(["", ""], ["", ""])


def test_export_adapter_bundle_writes_manifest_and_refuses_zero_adapter(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    manifest = export_adapter_bundle(
        _FakePeftModel(0.5), bundle, metrics={"adapted_wer": 0.1}, provenance={"baseModel": MODEL_ID}
    )
    names = {entry["path"] for entry in manifest}
    assert names == {"adapter_config.json", ADAPTER_WEIGHTS, "metrics.json", "provenance.json"}
    written = json.loads((bundle / "artifact-manifest.json").read_text(encoding="utf-8"))
    assert written["format"] == "peft_adapter" and written["files"] == manifest
    assert all(len(entry["sha256"]) == 64 and entry["bytes"] > 0 for entry in manifest)
    with pytest.raises(RuntimeError, match="zero or missing"):
        export_adapter_bundle(_FakePeftModel(0.0), tmp_path / "zero", metrics={}, provenance={})


def test_verify_adapter_merge_accepts_the_scaled_product_and_rejects_drift(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    export_adapter_bundle(_FakePeftModel(0.5), bundle, metrics={}, provenance={})
    base = torch.zeros(3, 3)
    expected = base + (4 / 2) * (torch.full((3, 2), 0.5) @ torch.ones(2, 3))
    check = verify_adapter_merge(bundle, "probe", base, expected, rank=2, alpha=4, tolerance=1e-5)
    assert check["merge_error_max"] == 0.0 and check["adapter_delta_max"] == pytest.approx(2.0)
    with pytest.raises(RuntimeError, match="differ from base"):
        verify_adapter_merge(bundle, "probe", base, expected + 1.0, rank=2, alpha=4, tolerance=1e-5)
    with pytest.raises(RuntimeError, match="below the"):
        verify_adapter_merge(bundle, "probe", base, expected, rank=2, alpha=4, tolerance=10.0)


def test_adaptation_report_is_sample_sanity_with_pinned_identity() -> None:
    report = adaptation_report(
        baseline_wer=0.4,
        adapted_wer=0.3,
        reloaded_wer=0.3,
        n_eval=10,
        history=[{"epoch": 1, "train_loss": 1.0, "validation_loss": 0.9}],
        dataset={"name": "PolyAI/minds14", "config": "en-US"},
    )
    assert report["verdict"] == "sample-sanity"
    assert [m["id"] for m in report["metrics"]] == ["baseline_wer", "adapted_wer", "reloaded_wer"]
    assert report["model_id"] == MODEL_ID and report["model_revision"] == MODEL_REVISION
    assert report["n_utterances"] == 10 and report["optimisation_history"][0]["epoch"] == 1
