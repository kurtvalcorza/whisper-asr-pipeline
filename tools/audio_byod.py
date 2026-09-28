"""Optional, explicit-consent BYOD audio search; never called by default Run all."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import uuid
from pathlib import Path

import numpy as np
import soundfile as sf

import audio_core as core
import audio_models as models
from audio_data import inspect_audio, safe_path, sha256


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def validate_input(path: Path) -> tuple[dict, list[dict]]:
    """Validate explicit rights and bounded local inputs before downloading models."""
    if path.stat().st_size > 1_000_000:
        raise ValueError("BYOD manifest exceeds 1 MB")
    manifest = read(path)
    if manifest.get("rights_confirmed") is not True or not str(manifest.get("source_notes", "")).strip():
        raise ValueError("Explicit rights/consent acknowledgement and source notes required")
    records = manifest.get("records", [])
    if not 1 <= len(records) <= 120:
        raise ValueError("BYOD accepts 1–120 recordings")
    ids = [r["doc_id"] for r in records]
    if len(set(ids)) != len(ids) or any(not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", name) for name in ids):
        raise ValueError("Invalid/duplicate document IDs")
    verified = []
    for record in sorted(records, key=lambda r: r["doc_id"]):
        source = safe_path(path.parent, record["path"])
        if source.suffix.lower() not in (".wav", ".flac") or not 0 < source.stat().st_size <= 4_000_000:
            raise ValueError("Only bounded WAV/FLAC inputs (4 MB each) are supported")
        data = source.read_bytes()
        verified.append({**record, **inspect_audio(data), "local_path": str(source)})
        reference = record.get("reference")
        if reference is not None and (not isinstance(reference, str) or not core.normalize(reference)):
            raise ValueError("Supplied references must contain valid nonempty text")
    query = manifest.get("probe_query")
    if query is not None and (not isinstance(query, str) or not query.strip() or len(query) > 2000):
        raise ValueError("Optional probe query must be nonempty and at most 2000 characters")
    return manifest, verified


def waves(records: list[dict]) -> list[np.ndarray]:
    values = []
    for row in records:
        source = Path(row["local_path"])
        if sha256(source) != row["sha256"]:
            raise ValueError("BYOD audio changed after validation")
        values.append(sf.read(source, dtype="float32")[0])
    return values


def rankings(root: Path, docs: list[dict], vectors: np.ndarray, query: str) -> dict:
    model = models.load_model("embedding", root)
    try:
        query_vector = models.embed(model, [query], query=True)[0]
    finally:
        models.unload(model)
    ids = [r["doc_id"] for r in docs]
    dense = core.dense_rank(ids, vectors, query_vector)
    lookup = {r["doc_id"]: r["text"] for r in docs}
    candidates = dense[:10]
    model = models.load_model("reranker", root)
    try:
        scores = models.rerank(model, query, [lookup[r["doc_id"]] for r in candidates])
    finally:
        models.unload(model)
    return {
        "bm25": core.BM25(ids, [r["text"] for r in docs]).rank(query),
        "dense": dense,
        "rerank": core.rank_scores([r["doc_id"] for r in candidates], scores),
    }


def check_bundle(bundle: Path) -> tuple[dict, list[dict], np.ndarray]:
    manifest = read(bundle / "index_manifest.json")
    if manifest.get("format") != "dimer_byod_audio_search_v1":
        raise ValueError("Unsupported BYOD index format")
    for filename, checksum in manifest["files"].items():
        if Path(filename).name != filename or sha256(bundle / filename) != checksum:
            raise ValueError("BYOD artifact integrity failure")
    docs = read(bundle / "documents.json")
    vectors = np.load(bundle / "embeddings.npy", allow_pickle=False)
    if [r["doc_id"] for r in docs] != manifest["doc_ids"] or vectors.shape != (
        len(docs),
        manifest["dimension"],
    ):
        raise ValueError("BYOD shape/order failure")
    if not np.isfinite(vectors).all() or not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-5):
        raise ValueError("BYOD vector normalization failure")
    return manifest, docs, vectors


def verify(root: Path, source: Path, bundle: Path) -> None:
    manifest, docs, vectors = check_bundle(bundle)
    if manifest["pid"] == os.getpid() or sha256(source) != manifest["input_manifest_sha256"]:
        raise ValueError("Fresh process and unchanged input manifest required")
    if sha256(root / "model_manifest.json") != manifest["model_manifest_sha256"]:
        raise ValueError("Model identity changed")
    _, records = validate_input(source)
    expected_audio = read(bundle / "audio_reacquisition.json")["records"]
    if [(r["doc_id"], r["sha256"]) for r in records] != [(r["doc_id"], r["sha256"]) for r in expected_audio]:
        raise ValueError("Audio identity changed")
    count = min(3, len(docs))
    model = models.load_model("embedding", root)
    try:
        rebuilt = models.embed(model, [r["text"] for r in docs[:count]])
    finally:
        models.unload(model)
    np.testing.assert_allclose(rebuilt, vectors[:count], atol=1e-5, rtol=1e-4)
    expected = read(bundle / "probe.json")
    actual = rankings(root, docs, vectors, expected["query"])
    for method, ranked in actual.items():
        previous = expected["rankings"][method]
        if [r["doc_id"] for r in ranked] != [r["doc_id"] for r in previous]:
            raise ValueError("BYOD replay ranking mismatch")
        np.testing.assert_allclose(
            [r["score"] for r in ranked], [r["score"] for r in previous], atol=1e-5, rtol=1e-4
        )
    model = models.load_model("whisper", root)
    try:
        replayed = models.transcribe(model, waves(records[:count]))
    finally:
        models.unload(model)
    if replayed != [r["text"] for r in docs[:count]]:
        raise ValueError("BYOD retranscription mismatch")
    write(
        bundle / "verification.json",
        {
            "passed": True,
            "fresh_process": True,
            "retranscribed": count,
            "reembedded": count,
            "atol": 1e-5,
            "rtol": 1e-4,
        },
    )


def build(root: Path, source: Path) -> Path:
    provided, records = validate_input(source)
    if not (root / "model_manifest.json").is_file():
        raise ValueError("Run root must contain the pinned model_manifest.json")
    bundle = root / "byod" / uuid.uuid4().hex
    bundle.mkdir(parents=True)
    model = models.load_model("whisper", root)
    try:
        texts = models.transcribe(model, waves(records))
    finally:
        models.unload(model)
    docs = [{"doc_id": r["doc_id"], "text": text} for r, text in zip(records, texts, strict=True)]
    model = models.load_model("embedding", root)
    try:
        vectors = models.embed(model, texts)
    finally:
        models.unload(model)
    query = provided.get("probe_query") or next(
        (t for t in texts if t.strip()), "Ano ang nilalaman ng audio?"
    )
    predicted = rankings(root, docs, vectors, query)
    references = [
        (r["reference"], text) for r, text in zip(records, texts, strict=True) if r.get("reference")
    ]
    metrics = core.asr_metrics([r for r, _ in references], [t for _, t in references]) if references else None
    write(bundle / "documents.json", docs)
    np.save(bundle / "embeddings.npy", vectors, allow_pickle=False)
    lexical = core.BM25([r["doc_id"] for r in docs], texts)
    write(
        bundle / "lexical_statistics.json",
        {
            "normalization": "NFC, lowercase, punctuation-to-space, whitespace collapse; no stopword removal",
            "token_counts": [dict(counts) for counts in lexical.tokens],
            "document_frequency": dict(lexical.df),
            "document_lengths": lexical.lengths.tolist(),
            "average_length": lexical.average_length,
            "k1": lexical.k1,
            "b": lexical.b,
        },
    )
    write(
        bundle / "probe.json",
        {
            "query": query,
            "purpose": "Mechanical reconstruction probe; no accuracy claim",
            "rankings": predicted,
        },
    )
    write(
        bundle / "evaluation.json",
        {
            "asr_status": "measured_on_supplied_references" if references else "not_measurable",
            "reference_count": len(references),
            "asr": metrics,
            "retrieval_status": "not_measurable",
            "reason": "No reviewed query/relevance labels supplied; BYOD v1 does not import qrels.",
        },
    )
    write(
        bundle / "audio_reacquisition.json",
        {
            "instructions": "Supply original local audio matching these SHA256 values; audio is not bundled.",
            "source_notes": provided["source_notes"],
            "records": [
                {k: r[k] for k in ("doc_id", "path", "sha256", "duration", "sample_rate")} for r in records
            ],
        },
    )
    (bundle / "model_manifest.json").write_bytes((root / "model_manifest.json").read_bytes())
    write(
        bundle / "settings.json",
        {
            "models": models.SETTINGS,
            "bm25": {"k1": 1.2, "b": 0.75},
            "candidate_depth": 10,
            "tie_rule": "stable document ID",
        },
    )
    write(
        bundle / "index_manifest.json",
        {
            "format": "dimer_byod_audio_search_v1",
            "pid": os.getpid(),
            "input_manifest_sha256": sha256(source),
            "model_manifest_sha256": sha256(root / "model_manifest.json"),
            "doc_ids": [r["doc_id"] for r in docs],
            "dimension": vectors.shape[1],
            "files": {p.name: sha256(p) for p in sorted(bundle.iterdir()) if p.is_file()},
        },
    )
    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "--root",
            str(root),
            "--manifest",
            str(source),
            "--verify",
            str(bundle),
        ],
        check=True,
    )
    return bundle


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args()
    if args.verify:
        verify(args.root.resolve(), args.manifest.resolve(), args.verify.resolve())
    else:
        print(build(args.root.resolve(), args.manifest.resolve()))
