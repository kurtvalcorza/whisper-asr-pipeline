"""Full-stage integration using deterministic CPU fake models; no performance claims."""

import copy
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import audio_runtime as rt  # noqa: E402


@pytest.fixture
def experiment(tmp_path, monkeypatch):
    tools = Path(rt.__file__).parent
    mappings = {
        "audio_sample.json": "sample_manifest.json",
        "audio_queries.json": "queries.json",
        "audio_qrels.json": "qrels.json",
        "audio_annotations.json": "annotation_manifest.json",
        "audio_models.json": "model_manifest.json",
        "audio_dataset_audit.json": "dataset_audit.json",
        "audio-requirements.lock": "requirements.txt",
    }
    for source, destination in mappings.items():
        (tmp_path / destination).write_bytes((tools / source).read_bytes())
    (tmp_path / "DATA_LICENSE.md").write_text("CC BY 4.0 Google FLEURS attribution")
    rt.write(tmp_path / "source.json", {"files": {}})
    rt.out(tmp_path).mkdir()
    records = rt.read(tmp_path / "sample_manifest.json")["records"]
    monkeypatch.setattr(rt.data, "acquire", lambda root: copy.deepcopy(records))
    monkeypatch.setattr(rt, "waves", lambda root, rows: [r["doc_id"] for r in rows])
    monkeypatch.setattr(
        rt.models,
        "load_model",
        lambda kind, root: {
            "kind": kind,
            "generation_config": {},
            "generation_overrides": rt.models.SETTINGS["asr"],
            "language_token_id": 50348,
            "prefix_length": 4,
        },
    )
    monkeypatch.setattr(rt.models, "unload", lambda handle: None)
    text_by_id = {r["doc_id"]: "Automatic transcript " + r["doc_id"] for r in records}
    monkeypatch.setattr(rt.models, "transcribe", lambda handle, waves: [text_by_id[w] for w in waves])

    def embed(handle, texts, query=False):
        rows = []
        for text in texts:
            seed = int.from_bytes(rt.hashlib.sha256(text.encode()).digest()[:4], "little")
            row = np.random.default_rng(seed).normal(size=16).astype("float32")
            rows.append(row / np.linalg.norm(row))
        return np.stack(rows)

    monkeypatch.setattr(rt.models, "embed", embed)
    monkeypatch.setattr(
        rt.models, "rerank", lambda handle, query, docs: np.array([len(t) % 11 for t in docs], float)
    )
    rt.prepare(tmp_path)
    return tmp_path


def test_real_annotations_compatible_and_remain_unqualified(experiment):
    queries, qrels, qualified = rt.annotations(experiment)
    assert len(queries) == 60 and len(qrels) == 7200 and qualified is False
    assert rt.read(rt.out(experiment) / "qualification.json")["unjudged_cells"] == 7200


def test_annotation_refuses_false_review_and_corpus_change(experiment):
    path = experiment / "annotation_manifest.json"
    manifest = rt.read(path)
    manifest["benchmark_qualified"] = True
    rt.write(path, manifest)
    with pytest.raises(ValueError, match="Draft"):
        rt.annotations(experiment)
    manifest["benchmark_qualified"] = False
    rt.write(path, manifest)
    sample = rt.read(experiment / "sample_manifest.json")
    sample["seed"] = 43
    rt.write(experiment / "sample_manifest.json", sample)
    with pytest.raises(ValueError, match="corpus changed"):
        rt.annotations(experiment)


def test_asr_index_search_reference_and_qrel_noninterference(experiment):
    rt.asr(experiment)
    rt.index(experiment)
    rt.search(experiment, "Ano ang agham?")
    before = rt.read(experiment / "interactive_search.json")
    records = rt.corpus(experiment)
    for row in records:
        row["reference"] = "REF LEAK SENTINEL"
    rt.write(rt.out(experiment) / "corpus.json", records)
    rt.write(experiment / "qrels.json", [{"sentinel": "changed evaluation labels"}])
    rt.index(experiment)
    rt.search(experiment, "Ano ang agham?")
    assert rt.read(experiment / "interactive_search.json") == before
    assert "REF LEAK SENTINEL" not in (rt.out(experiment) / "audio_acquisition.json").read_text()
    acquisition = rt.read(rt.out(experiment) / "audio_acquisition.json")
    assert all(not any("reference" in key for key in row) for row in acquisition["records"])


def test_index_refuses_model_identity_and_embedding_tampering(experiment):
    rt.asr(experiment)
    rt.index(experiment)
    rt.verify_index(experiment)
    path = experiment / "model_manifest.json"
    original = path.read_bytes()
    path.write_bytes(b"{}")
    with pytest.raises(ValueError, match="model identity"):
        rt.verify_index(experiment)
    path.write_bytes(original)
    vectors = rt.out(experiment) / "asr_embeddings.npy"
    vectors.write_bytes(b"wrong")
    with pytest.raises(ValueError, match="hash mismatch"):
        rt.verify_index(experiment)


def test_full_fake_stages_and_portable_export(experiment, monkeypatch):
    root = experiment
    rt.asr(root)
    rt.index(root)
    rt.evaluate(root)
    rt.activity(root)
    rt.write(rt.out(root) / "receipt_index.json", {"pid": -1})
    rt.reload(root)
    assert rt.read(rt.out(root) / "verification.json")["passed"] is True
    candidates = list(rt.csv.DictReader((rt.out(root) / "activity_candidates.csv").open()))
    assert len(candidates) == 80 and {r["candidate_depth"] for r in candidates} == {"10", "20"}
    assert all(r["query_id"].startswith("dev-") for r in candidates)
    latency = list(rt.csv.DictReader((rt.out(root) / "activity_latency.csv").open()))
    dev_ids = [q["query_id"] for q in rt.read(root / "queries.json") if q["role"] == "dev"]
    for condition in ("reference", "asr"):
        rows = [r for r in latency if r["condition"] == condition]
        assert [r["query_id"] for r in rows] == dev_ids
        assert {(r["depth10_pairs"], r["depth20_pairs"]) for r in rows} == {("10", "20")}
        assert all(r["depth10_matches_canonical"] == "True" for r in rows)
        paired = rt.read(rt.out(root) / "activity_summary.json")["paired_latency"][condition]
        assert paired["query_ids"] == dev_ids and paired["depth10_pairs"] == 200
        assert paired["depth20_pairs"] == 400
    canonical_timing = {t["stage"]: t for t in rt.read(rt.out(root) / "retrieval_timing.json")}
    assert len(canonical_timing["asr_rerank"]["per_query"]) == 60
    for stage in rt.STAGES[:-1]:
        rt.write(rt.out(root) / f"receipt_{stage}.json", {"pid": -1})
    import importlib.metadata

    monkeypatch.setattr(importlib.metadata, "version", lambda name: "fake-test-version")
    rt.report(root)
    with zipfile.ZipFile(rt.out(root) / "search_index.zip") as archive:
        names = set(archive.namelist())
        assert {"DATA_LICENSE.md", "audio_acquisition.json", "search_archive.py", "requirements.txt"} <= names
        assert not {"sample_manifest.json", "audio_manifest.json", "queries.json", "qrels.json"} & names
        joined = b"\n".join(archive.read(n) for n in names if n.endswith(".json"))
        for record in rt.corpus(root):
            assert record["reference"].encode() not in joined
        restored = root / "restored-search"
        archive.extractall(restored)
    namespace = {"__name__": "export_test"}
    exec(compile((restored / "search_archive.py").read_text(), "search_archive.py", "exec"), namespace)
    rt.search(root, "Ano ang agham?")
    assert (
        namespace["run"](restored, "Ano ang agham?") == rt.read(root / "interactive_search.json")["results"]
    )
    assert rt.read(rt.out(root) / "run_summary.json")["annotation_qualified"] is False
    (root / "dataset_audit.json").unlink()
    with pytest.raises(ValueError, match="provenance"):
        rt.report(root)


def test_bootstrap_named_direction_is_asr_minus_reference(experiment, monkeypatch):
    queries, _, _ = rt.annotations(experiment)
    runs = {}
    for q in queries:
        anchor = q["anchor_doc_id"]
        distractors = [d["doc_id"] for d in rt.corpus(experiment) if d["doc_id"] != anchor]
        runs[q["query_id"]] = {}
        for system in rt.SYSTEMS:
            ids = [anchor, *distractors] if system.startswith("reference") else [*distractors, anchor]
            runs[q["query_id"]][system] = [{"doc_id": d, "score": float(-i)} for i, d in enumerate(ids)]
    monkeypatch.setattr(rt, "run_search", lambda root, queries, depth: (runs, []))
    rt.evaluate(experiment)
    values = rt.read(rt.out(experiment) / "paired_comparisons.json")["comparisons"]
    contrast = values["asr_rerank_minus_reference_rerank"]
    assert contrast["difference"] == -1 and contrast["a_system"] == "asr_rerank"


def test_prepare_requires_rights_and_audit(experiment):
    (experiment / "DATA_LICENSE.md").unlink()
    with pytest.raises(ValueError, match="provenance"):
        rt.prepare(experiment)


def test_audit_must_bind_exact_sample(experiment):
    audit = rt.read(experiment / "dataset_audit.json")
    audit["manifest_sha256"] = "0" * 64
    rt.write(experiment / "dataset_audit.json", audit)
    with pytest.raises(ValueError, match="audit does not bind"):
        rt.prepare(experiment)


def test_main_receipt_chain_rerun_ownership_and_tampering(experiment, monkeypatch):
    import importlib.metadata

    root = experiment
    # Start with a clean results directory rather than fixture-created stage output.
    for path in rt.out(root).iterdir():
        if path.is_file():
            path.unlink()
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: True, reset_peak_memory_stats=lambda: None, max_memory_allocated=lambda: 0
        )
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(importlib.metadata, "version", lambda name: "fake-test-version")
    process_id = [101]
    monkeypatch.setattr(rt.os, "getpid", lambda: process_id[0])

    def execute(stage):
        monkeypatch.setattr(sys, "argv", ["audio_runtime.py", "--root", str(root), "--stage", stage])
        if stage == "reload":
            process_id[0] += 1
        rt.main()

    for stage in rt.STAGES:
        execute(stage)
    canonical = {stage: rt.sha(rt.out(root) / f"receipt_{stage}.json") for stage in rt.STAGES[:4]}
    expected_owned = {
        stage: set(rt.read(rt.out(root) / f"receipt_{stage}.json")["outputs"])
        for stage in ("activity", "reload", "report")
    }
    assert "verification.json" in expected_owned["reload"]
    execute("activity")
    assert rt.read(rt.out(root) / "receipt_reload.json")["completed"] is False
    with pytest.raises(ValueError, match="Stale experiment receipt"):
        execute("report")
    for stage in ("reload", "report"):
        execute(stage)
    assert canonical == {stage: rt.sha(rt.out(root) / f"receipt_{stage}.json") for stage in rt.STAGES[:4]}
    for stage in ("activity", "reload", "report"):
        assert expected_owned[stage] <= set(rt.read(rt.out(root) / f"receipt_{stage}.json")["outputs"])
    artifact = rt.out(root) / "asr_documents.json"
    saved = artifact.read_bytes()
    artifact.write_text("[]")
    with pytest.raises(ValueError, match="Changed output: asr_documents.json"):
        execute("report")
    artifact.write_bytes(saved)
    rt.write(rt.out(root) / "verification.json", {"passed": True, "forged": "changed"})
    with pytest.raises(ValueError, match="Changed output: verification.json"):
        execute("report")


def test_reload_replays_original_embedding_batch(experiment, monkeypatch):
    """fp16 GPU embeddings depend on batch composition; reload must replay the same batch.

    Regression for the 2026-09-28 hosted T4 run, where re-embedding 3 documents against an
    index built in batches of 4 missed atol=1e-5 by up to 4.2e-4.
    """
    size = rt.models.SETTINGS["embedding_batch_size"]
    plain = rt.models.embed

    def batch_sensitive(handle, texts, query=False):
        rows = []
        for start in range(0, len(texts), size):
            chunk = texts[start : start + size]
            # Batch size and longest member stand in for the left-padding shape of a GPU batch.
            shape = len(chunk) + max(map(len, chunk))
            vectors = plain(handle, chunk, query) + 1e-3 * shape
            rows.append(vectors / np.linalg.norm(vectors, axis=1, keepdims=True))
        return np.concatenate(rows).astype("float32")

    monkeypatch.setattr(rt.models, "embed", batch_sensitive)
    root = experiment
    rt.asr(root)
    rt.index(root)
    rt.evaluate(root)
    docs, vectors = rt.verify_index(root)
    # The stand-in really is batch-sensitive: a 3-document batch is a different computation.
    assert not np.allclose(batch_sensitive(None, [d["text"] for d in docs[:3]]), vectors[:3], atol=1e-5)
    rt.write(rt.out(root) / "receipt_index.json", {"pid": -1})
    rt.reload(root)
    verification = rt.read(rt.out(root) / "verification.json")
    assert verification["passed"] is True
    assert verification["documents_reembedded"] == verification["queries_replayed"] == size
