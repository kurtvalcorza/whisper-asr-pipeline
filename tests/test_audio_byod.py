"""Optional BYOD input and artifact validation runs without pretrained models."""

import hashlib
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from audio_byod import validate_input
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


def _fake_models(monkeypatch):
    import audio_byod as byod

    monkeypatch.setattr(
        byod.models,
        "load_model",
        lambda kind, root: {
            "kind": kind,
            "generation_config": {},
            "generation_overrides": byod.models.SETTINGS["asr"],
            "language_token_id": 50348,
            "prefix_length": 4,
        },
    )
    monkeypatch.setattr(byod.models, "unload", lambda handle: None)
    monkeypatch.setattr(
        byod.models,
        "transcribe",
        lambda handle, waves: [
            f"awit ng ibon {len(w)}" if i else "ulan sa bundok" for i, w in enumerate(waves)
        ],
    )

    def embed(handle, texts, query=False):
        rows = []
        for text in texts:
            seed = int.from_bytes(hashlib.sha256(text.encode()).digest()[:4], "little")
            row = np.random.default_rng(seed).normal(size=8).astype("float32")
            rows.append(row / np.linalg.norm(row))
        return np.stack(rows)

    monkeypatch.setattr(byod.models, "embed", embed)
    monkeypatch.setattr(
        byod.models, "rerank", lambda handle, query, docs: np.array([len(t) % 7 for t in docs], float)
    )
    return byod


def _run_root(tmp_path):
    tools = Path(__file__).resolve().parents[1] / "tools"
    root = tmp_path / "run"
    root.mkdir()
    (root / "model_manifest.json").write_bytes((tools / "audio_models.json").read_bytes())
    (root / "requirements.txt").write_bytes((tools / "audio-requirements.lock").read_bytes())
    return root


def _two_clips(tmp_path, references):
    audio = tmp_path / "audio"
    audio.mkdir()
    sf.write(audio / "001.wav", np.linspace(-0.2, 0.2, 32000), 16000)
    sf.write(audio / "002.flac", np.linspace(0.1, -0.1, 48000), 16000)
    records = [{"doc_id": "001", "path": "001.wav"}, {"doc_id": "002", "path": "002.flac"}]
    for record, reference in zip(records, references, strict=True):
        if reference is not None:
            record["reference"] = reference
    path = audio / "manifest.json"
    path.write_text(
        json.dumps({"rights_confirmed": True, "source_notes": "Consented fixture", "records": records}),
        encoding="utf-8",
    )
    return path


@pytest.mark.parametrize("references", [("REFERENCE SENTINEL ALPHA", None), (None, None)])
def test_byod_search_package_is_shared_portable_and_separated(tmp_path, monkeypatch, references):
    byod = _fake_models(monkeypatch)
    rt = byod.runtime
    root = _run_root(tmp_path)
    manifest = _two_clips(tmp_path, references)
    bundle = byod.build(root, manifest, verify_in_subprocess=False)
    latest = json.loads((root / "byod" / "latest.json").read_text(encoding="utf-8"))
    assert Path(latest["search_dir"]) == bundle / "search"
    assert Path(latest["audio_base"]) == manifest.parent

    # Fresh-process reconstruction check (the build normally runs it in a subprocess).
    receipt = json.loads((bundle / "build_receipt.json").read_text(encoding="utf-8"))
    monkeypatch.setattr(byod.os, "getpid", lambda: receipt["pid"] + 1)
    byod.verify(root, manifest, bundle)
    assert json.loads((bundle / "verification.json").read_text(encoding="utf-8"))["passed"] is True

    evaluation = json.loads((bundle / "evaluation.json").read_text(encoding="utf-8"))
    measured = [r for r in references if r]
    assert evaluation["reference_count"] == len(measured)
    assert evaluation["retrieval_status"] == "not_measurable"
    assert evaluation["asr_status"] == ("measured_on_supplied_references" if measured else "not_measurable")

    # New query on the BYOD package, without rebuilding ASR: only BYOD recordings, with audio identity.
    monkeypatch.setattr(byod.models, "transcribe", lambda *a: pytest.fail("search must not re-run ASR"))
    rt.search(root, "Saan umuulan?", bundle / "search")
    response = json.loads((root / "interactive_search.json").read_text(encoding="utf-8"))
    assert response["source"] == "byod" and response["unscored"] is True
    assert {r["doc_id"] for r in response["results"]} == {"001", "002"}
    for row in response["results"]:
        assert sha256(manifest.parent / row["audio_path"]) == row["audio_sha256"]

    # The portable package carries no references or evaluation records.
    with zipfile.ZipFile(bundle / "search_index.zip") as archive:
        names = set(archive.namelist())
        assert names == rt.PACKAGE_FILES | {"index_manifest.json"}
        assert not {"evaluation.json", "probe.json", "verification.json"} & names
        assert all(b"REFERENCE SENTINEL" not in archive.read(n) for n in names)
        clean = tmp_path / "clean"
        archive.extractall(clean)

    # The same exported consumer as the default archive searches it from a clean directory.
    namespace = {"__name__": "byod_export_test"}
    exec(compile((clean / "search_archive.py").read_text(), "search_archive.py", "exec"), namespace)
    assert namespace["run"](clean, "Saan umuulan?") == [
        {k: v for k, v in r.items()} for r in response["results"]
    ]

    # Modified embeddings, identifiers and model settings are still refused.
    for name, mutate in [
        ("asr_embeddings.npy", lambda p: np.save(p, np.load(p) * -1, allow_pickle=False)),
        ("asr_index_documents.json", lambda p: p.write_text(p.read_text().replace('"001"', '"009"'))),
    ]:
        target = clean / name
        saved = target.read_bytes()
        mutate(target)
        with pytest.raises(ValueError, match="integrity"):
            namespace["run"](clean, "Saan umuulan?")
        with pytest.raises(ValueError, match="hash mismatch"):
            rt.verify_package(clean, root / "model_manifest.json")
        target.write_bytes(saved)
    monkeypatch.setitem(byod.models.SETTINGS, "retrieval_depth_override", 20)
    with pytest.raises(ValueError, match="settings"):
        namespace["run"](clean, "Saan umuulan?")
    with pytest.raises(ValueError, match="settings"):
        rt.verify_package(clean, root / "model_manifest.json")


def test_byod_verify_refuses_changed_input_manifest(tmp_path, monkeypatch):
    byod = _fake_models(monkeypatch)
    root = _run_root(tmp_path)
    manifest = _two_clips(tmp_path, (None, None))
    bundle = byod.build(root, manifest, verify_in_subprocess=False)
    with pytest.raises(ValueError, match="Fresh process"):
        byod.verify(root, manifest, bundle)
    receipt = json.loads((bundle / "build_receipt.json").read_text(encoding="utf-8"))
    monkeypatch.setattr(byod.os, "getpid", lambda: receipt["pid"] + 1)
    changed = manifest.read_text(encoding="utf-8").replace("Consented", "Changed")
    manifest.write_text(changed, encoding="utf-8")
    with pytest.raises(ValueError, match="unchanged input manifest"):
        byod.verify(root, manifest, bundle)


def test_separate_byod_attempts_do_not_overwrite(tmp_path, monkeypatch):
    byod = _fake_models(monkeypatch)
    root = _run_root(tmp_path)
    manifest = _two_clips(tmp_path, (None, None))
    first = byod.build(root, manifest, verify_in_subprocess=False)
    second = byod.build(root, manifest, verify_in_subprocess=False)
    assert first != second and (first / "search" / "index_manifest.json").is_file()
