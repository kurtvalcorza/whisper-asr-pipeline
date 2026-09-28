"""Optional BYOD input and artifact validation runs without pretrained models."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from audio_byod import check_bundle, validate_input
from audio_data import sha256


def source(tmp_path, **overrides):
    sf.write(tmp_path / "a.wav", np.ones(32000) * 0.1, 16000)
    value = {
        "rights_confirmed": True,
        "source_notes": "My consented recording",
        "records": [{"doc_id": "my-clip", "path": "a.wav"}],
        **overrides,
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def test_valid_reference_free_byod(tmp_path):
    _, records = validate_input(source(tmp_path))
    assert records[0]["frames"] == 32000
    assert "reference" not in records[0]


@pytest.mark.parametrize(
    "overrides",
    [
        {"rights_confirmed": False},
        {"source_notes": ""},
        {"records": []},
        {"records": [{"doc_id": "bad/id", "path": "a.wav"}]},
        {"records": [{"doc_id": "my-clip", "path": "../a.wav"}]},
        {"records": [{"doc_id": "my-clip", "path": "a.wav", "reference": ""}]},
        {"probe_query": ""},
    ],
)
def test_invalid_byod_refused(tmp_path, overrides):
    with pytest.raises(ValueError):
        validate_input(source(tmp_path, **overrides))


def test_index_tamper_refused(tmp_path):
    docs = tmp_path / "documents.json"
    docs.write_text('[{"doc_id":"one","text":"salita"}]', encoding="utf-8")
    np.save(tmp_path / "embeddings.npy", np.array([[1.0, 0.0]], dtype=np.float32), allow_pickle=False)
    manifest = {
        "format": "dimer_byod_audio_search_v1",
        "doc_ids": ["one"],
        "dimension": 2,
        "files": {p.name: sha256(p) for p in tmp_path.iterdir()},
    }
    (tmp_path / "index_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    check_bundle(tmp_path)
    docs.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="integrity"):
        check_bundle(tmp_path)
