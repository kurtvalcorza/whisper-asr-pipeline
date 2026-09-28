"""CPU refusal tests for the frozen FLEURS acquisition contract."""

import copy
import io
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from audio_data import download, inspect_audio, normalize, safe_path, validate_manifest


def wav(rate=16000, channels=1, duration=2, silent=False):
    stream = io.BytesIO()
    data = (
        np.zeros((int(rate * duration), channels))
        if silent
        else np.full((int(rate * duration), channels), 0.1)
    )
    sf.write(stream, data, rate, format="WAV", subtype="PCM_16")
    return stream.getvalue()


def test_original_mono_validation():
    result = inspect_audio(wav())
    assert result["frames"] == 32000 and result["duration"] == 2


@pytest.mark.parametrize(
    "kwargs", [{"rate": 8000}, {"channels": 2}, {"duration": 1.99}, {"duration": 25.01}, {"silent": True}]
)
def test_bad_audio_refused(kwargs):
    with pytest.raises(ValueError):
        inspect_audio(wav(**kwargs))


def test_normalization_keeps_diacritics():
    assert normalize("Óo—Hindi! 12, tatlo.") == "óo hindi 12 tatlo"


@pytest.mark.parametrize("path", ["../escape.wav", "/absolute.wav"])
def test_unsafe_path(tmp_path, path):
    with pytest.raises(ValueError):
        safe_path(tmp_path, path)


def test_corrupt_cache_refused_without_network(tmp_path):
    path = tmp_path / "source.parquet"
    path.write_bytes(b"wrong")
    with pytest.raises(ValueError, match="Cached source"):
        download({"bytes": 5, "sha256": "0" * 64}, path)


def test_frozen_manifest():
    source = Path(__file__).resolve().parents[1] / "tools/audio_sample.json"
    manifest = json.loads(source.read_text(encoding="utf-8"))
    validate_manifest(manifest)
    broken = copy.deepcopy(manifest)
    broken["records"][1]["family_id"] = broken["records"][0]["family_id"]
    with pytest.raises(ValueError, match="Duplicate"):
        validate_manifest(broken)
    broken = copy.deepcopy(manifest)
    broken["records"][0]["role"] = "unknown"
    with pytest.raises(ValueError, match="role"):
        validate_manifest(broken)
