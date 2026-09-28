"""Standalone Filipino audio archive experiment and safe search artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import time
import zipfile
from pathlib import Path

import numpy as np

import audio_core as core
import audio_data as data
import audio_models as models

STAGES = ("prepare", "asr", "index", "evaluate", "activity", "reload", "report")
SYSTEMS = tuple(
    f"{condition}_{method}" for condition in ("reference", "asr") for method in ("bm25", "dense", "rerank")
)
PORTABLE_EXTRA = (
    "audio_core.py",
    "audio_models.py",
    "search_archive.py",
    "requirements.txt",
    "RECONSTRUCT.md",
)
SEARCH_HELPER = '''"""Search the exported automatic-transcript archive; no evaluation files required."""
import argparse, hashlib, json
from pathlib import Path
import numpy as np
import audio_core as core
import audio_models as models

def run(root, query):
    manifest=json.loads((root/'index_manifest.json').read_text())
    if manifest['format']!='dimer_audio_archive' or manifest['version']!=1:
        raise ValueError('Unknown index version')
    for name, expected in manifest['files'].items():
        if Path(name).name!=name or hashlib.sha256((root/name).read_bytes()).hexdigest()!=expected:
            raise ValueError('Archive integrity failure: '+name)
    model_hash=hashlib.sha256((root/'model_manifest.json').read_bytes()).hexdigest()
    if model_hash!=manifest['model_manifest_sha256']:
        raise ValueError('Model identity changed')
    if models.SETTINGS!=manifest['model_settings']:
        raise ValueError('Inference settings changed')
    docs=json.loads((root/'asr_index_documents.json').read_text())
    ids=[d['doc_id'] for d in docs]
    vectors=np.load(root/'asr_embeddings.npy',allow_pickle=False)
    if ids!=manifest['doc_ids'] or vectors.shape!=(len(ids),manifest['dimension']):
        raise ValueError('Index order/shape mismatch')
    model=models.load_model('embedding',root)
    try: vector=models.embed(model,[query],query=True)[0]
    finally: models.unload(model)
    chosen=core.dense_rank(ids,vectors,vector)[:10]
    texts={d['doc_id']:d['text'] for d in docs}
    model=models.load_model('reranker',root)
    try: scores=models.rerank(model,query,[texts[d['doc_id']] for d in chosen])
    finally: models.unload(model)
    ranked=core.rank_scores([d['doc_id'] for d in chosen],scores)[:5]
    return [{**d,'transcript':texts[d['doc_id']]} for d in ranked]

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=Path(__file__).parent)
    parser.add_argument('--query',required=True);args=parser.parse_args()
    print(json.dumps(run(args.root,args.query),ensure_ascii=False,indent=2))
'''


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8", newline="\n"
    )


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
    ).hexdigest()


def out(root):
    return root / "results"


def table(path, rows):
    if not rows:
        raise ValueError("Cannot export empty table: " + str(path))
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def identity(root):
    files = [
        "source.json",
        "sample_manifest.json",
        "queries.json",
        "qrels.json",
        "annotation_manifest.json",
        "model_manifest.json",
    ]
    hashes = {name: sha(root / name) for name in files}
    for name, checksum in read(root / "source.json").get("files", {}).items():
        if Path(name).name != name or sha(root / name) != checksum:
            raise ValueError("Embedded source changed: " + name)
    for module in (core, data, models):
        hashes[Path(module.__file__).name] = sha(module.__file__)
    hashes["runtime"] = sha(__file__)
    return digest(hashes)


def corpus(root):
    rows = read(out(root) / "corpus.json")
    if len({r["doc_id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate document ID")
    return rows


def waves(root, records):
    import soundfile as sf

    audio = []
    for row in records:
        path = (root / row["path"]).resolve()
        if not path.is_relative_to(root.resolve()) or sha(path) != row["sha256"]:
            raise ValueError("Audio path/hash mismatch")
        signal, rate = sf.read(path, dtype="float32")
        if signal.ndim != 1 or rate != 16000 or not 2 <= len(signal) / rate <= 25:
            raise ValueError("Unsupported audio duration/rate/channels")
        if not np.isfinite(signal).all() or not np.any(signal):
            raise ValueError("Nonfinite or silent audio")
        audio.append(signal)
    return audio


def annotations(root):
    docs = corpus(root)
    queries = read(root / "queries.json")
    qrels = read(root / "qrels.json")
    manifest = read(root / "annotation_manifest.json")
    ids = {d["doc_id"] for d in docs}
    qids = {q["query_id"] for q in queries}
    if len(qids) != len(queries) or len(queries) != 60:
        raise ValueError("Expected 60 unique queries")
    if sum(q["role"] == "dev" for q in queries) != 20 or sum(q["role"] == "test" for q in queries) != 40:
        raise ValueError("Expected 20 development and 40 evaluation queries")
    if len(qrels) != len(ids) * len(qids) or {(r["query_id"], r["doc_id"]) for r in qrels} != {
        (q, d) for q in qids for d in ids
    }:
        raise ValueError("Missing/duplicated annotation matrix cells")
    for q in queries:
        if not q["text"].strip() or q["anchor_doc_id"] not in ids or q["role"] not in ("dev", "test"):
            raise ValueError("Invalid query or anchor")
        doc = next(d for d in docs if d["doc_id"] == q["anchor_doc_id"])
        if doc["role"] != q["role"]:
            raise ValueError("Query role crosses corpus role")
        if q.get("target_family_id") != doc["family_id"] or q.get("group_id") != doc["family_id"]:
            raise ValueError("Query anchor and bootstrap family differ")
    roles = {"dev": "development", "test": "evaluation"}
    validation = core.validate_annotations(
        [{**d, "role": roles[d["role"]]} for d in docs],
        [{**q, "role": roles[q["role"]]} for q in queries],
        qrels,
        manifest,
    )
    qualified = validation["benchmark_qualified"]
    for name in ("queries.json", "qrels.json"):
        entry = manifest["files"][name]
        if entry["sha256"] != sha(root / name) or entry["bytes"] != (root / name).stat().st_size:
            raise ValueError("Annotations changed without a versioned manifest")
    sample = manifest["files"]["audio_manifest.json"]
    if (
        sample["sha256"] != sha(root / "sample_manifest.json")
        or sample["bytes"] != (root / "sample_manifest.json").stat().st_size
    ):
        raise ValueError("Annotation source corpus changed")
    return queries, qrels, qualified


def prepare(root):
    for name in ("dataset_audit.json", "DATA_LICENSE.md"):
        if not (root / name).is_file() or not (root / name).read_text(encoding="utf-8").strip():
            raise ValueError("Required provenance absent: " + name)
    if read(root / "dataset_audit.json").get("manifest_sha256") != sha(root / "sample_manifest.json"):
        raise ValueError("Dataset audit does not bind the frozen corpus")
    records = data.acquire(root)
    if (
        len(records) != 120
        or sum(r["role"] == "dev" for r in records) != 40
        or sum(r["role"] == "test" for r in records) != 80
    ):
        raise ValueError("Frozen corpus must contain 40 development and 80 evaluation clips")
    write(out(root) / "corpus.json", records)
    queries, qrels, qualified = annotations(root)
    write(
        out(root) / "qualification.json",
        {
            "human_labels_verified": qualified,
            "mode": "reviewed_benchmark" if qualified else "engineering_preview_anchor_recovery",
            "limitations": [
                "Read speech only",
                "No verified speaker-independent claim",
                "Unknown pretraining overlap",
                "Curated queries are not representative user logs",
            ],
            "queries": len(queries),
            "judgement_cells": len(qrels),
            "unjudged_cells": sum(r.get("grade") is None for r in qrels),
        },
    )
    write(
        out(root) / "experiment_lock.json",
        {
            "identity": identity(root),
            "candidate_depth": 10,
            "activity_depth": 20,
            "seed": 42,
            "bm25": {"k1": 1.2, "b": 0.75},
            "tie_break": "doc_id ascending",
            "models": read(root / "model_manifest.json"),
            "annotation_qualified": qualified,
            "model_settings": models.SETTINGS,
        },
    )
    table(
        out(root) / "annotation_review.csv",
        [
            {
                "query_id": r["query_id"],
                "query": next(q["text"] for q in queries if q["query_id"] == r["query_id"]),
                "doc_id": r["doc_id"],
                "reference": next(d["reference"] for d in records if d["doc_id"] == r["doc_id"]),
                "grade": r.get("grade"),
                "status": r.get("status"),
                "reviewer": "",
                "rationale": "",
            }
            for r in qrels
        ],
    )


def asr(root):
    records = corpus(root)
    model = models.load_model("whisper", root)
    write(
        out(root) / "asr_settings.json",
        {
            "generation_config": model["generation_config"],
            "generation_overrides": model["generation_overrides"],
            "language_token_id": model["language_token_id"],
            "prefix_length": model["prefix_length"],
        },
    )
    hypotheses = []
    started = time.perf_counter()
    try:
        for begin in range(0, len(records), 4):
            batch = records[begin : begin + 4]
            hypotheses.extend(models.transcribe(model, waves(root, batch)))
            print(f"Transcribed {len(hypotheses)}/{len(records)}", flush=True)
        seconds = time.perf_counter() - started
    finally:
        models.unload(model)
    if len(hypotheses) != len(records):
        raise ValueError("ASR output count mismatch")
    documents = [{"doc_id": r["doc_id"], "text": h} for r, h in zip(records, hypotheses, strict=True)]
    write(out(root) / "asr_documents.json", documents)
    metrics = {}
    for role in ("dev", "test"):
        pairs = [(r, h) for r, h in zip(records, hypotheses, strict=True) if r["role"] == role]
        metrics[role] = core.asr_metrics([r["reference"] for r, h in pairs], [h for r, h in pairs])
    write(out(root) / "asr_metrics.json", metrics)
    write(
        out(root) / "asr_timing.json",
        {
            "seconds": seconds,
            "audio_seconds": sum(r["duration"] for r in records),
            "real_time_factor": seconds / sum(r["duration"] for r in records),
            "includes_loading": False,
            "includes_audio_decode": True,
            "includes_unload": False,
            "warm_state": "first clip cold; subsequent clips warm",
        },
    )
    table(
        out(root) / "transcriptions.csv",
        [
            {
                "doc_id": r["doc_id"],
                "role": r["role"],
                "reference": r["reference"],
                "hypothesis": h,
                "empty_hypothesis": not bool(core.normalize(h)),
            }
            for r, h in zip(records, hypotheses, strict=True)
        ],
    )


def index(root):
    records = corpus(root)
    hypotheses = read(out(root) / "asr_documents.json")
    if [d["doc_id"] for d in hypotheses] != [r["doc_id"] for r in records]:
        raise ValueError("Index document order differs")
    documents = {
        "reference": [{"doc_id": r["doc_id"], "text": r["reference"]} for r in records],
        "asr": hypotheses,
    }
    model = models.load_model("embedding", root)
    try:
        for condition, docs in documents.items():
            vectors = models.embed(model, [d["text"] for d in docs])
            if vectors.shape[0] != 120 or not np.isfinite(vectors).all():
                raise ValueError("Bad document embeddings")
            np.save(out(root) / f"{condition}_embeddings.npy", vectors, allow_pickle=False)
            write(out(root) / f"{condition}_index_documents.json", docs)
    finally:
        models.unload(model)
    # This portable artifact contains automatic transcripts only, never references or qrels.
    files = {name: sha(out(root) / name) for name in ("asr_index_documents.json", "asr_embeddings.npy")}
    lexical = core.BM25([d["doc_id"] for d in hypotheses], [d["text"] for d in hypotheses])
    write(
        out(root) / "lexical_statistics.json",
        {
            "doc_ids": lexical.doc_ids,
            "token_counts": [dict(counts) for counts in lexical.tokens],
            "document_frequency": dict(lexical.df),
            "document_lengths": lexical.lengths.tolist(),
            "average_length": lexical.average_length,
            "k1": lexical.k1,
            "b": lexical.b,
            "normalization": core.NORMALIZATION_VERSION,
        },
    )
    files["lexical_statistics.json"] = sha(out(root) / "lexical_statistics.json")
    sample = read(root / "sample_manifest.json")
    audio_fields = (
        "doc_id",
        "source_split",
        "source_id",
        "row_index",
        "path",
        "sha256",
        "sample_rate",
        "channels",
        "frames",
        "duration",
    )
    acquisition = {
        key: sample[key]
        for key in ("dataset", "configuration", "revision", "license", "attribution", "files")
    }
    acquisition["records"] = [{key: r[key] for key in audio_fields} for r in records]
    acquisition["reacquisition"] = (
        "Download the exact hashed Parquet source, select source_split/row_index, "
        "verify source_id and original audio SHA256, then restore the relative audio path. "
        "No reference transcript is needed for search."
    )
    write(out(root) / "audio_acquisition.json", acquisition)
    for name in ("model_manifest.json", "DATA_LICENSE.md"):
        (out(root) / name).write_bytes((root / name).read_bytes())
    for module in (core, models):
        (out(root) / Path(module.__file__).name).write_bytes(Path(module.__file__).read_bytes())
    (out(root) / "requirements.txt").write_bytes((root / "requirements.txt").read_bytes())
    (out(root) / "search_archive.py").write_text(SEARCH_HELPER, encoding="utf-8", newline="\n")
    (out(root) / "RECONSTRUCT.md").write_text(
        "# Reconstruct the automatic-transcript search archive\n\n"
        "Use a fresh hosted Colab/Kaggle T4 with Python 3.12. Install the pinned wheel lock with "
        "`pip install --require-hashes -r requirements.txt`, then run "
        '`python search_archive.py --root . --query "Ano ang hinahanap mo?"`. '
        "The helper verifies all file hashes and loads pinned query/reranking snapshots sequentially. "
        "Search requires no source references, qrels or audio. "
        "It returns document IDs and automatic transcripts. "
        "For playback, reacquire audio using audio_acquisition.json: verify the pinned Parquet, select "
        "the recorded row and source ID, verify the original audio checksum and restore the relative path. "
        "Source audio and pretrained weights are excluded. "
        "Preserve DATA_LICENSE.md and source attribution.\n",
        encoding="utf-8",
        newline="\n",
    )
    for name in ("model_manifest.json", "DATA_LICENSE.md", "audio_acquisition.json", "asr_settings.json"):
        files[name] = sha(out(root) / name)
    for name in PORTABLE_EXTRA:
        files[name] = sha(out(root) / name)
    write(
        out(root) / "index_manifest.json",
        {
            "format": "dimer_audio_archive",
            "version": 1,
            "files": files,
            "doc_ids": [d["doc_id"] for d in hypotheses],
            "dimension": int(vectors.shape[1]),
            "lexical": {
                "k1": 1.2,
                "b": 0.75,
                "normalization": "NFC lower punctuation-to-space whitespace collapse",
            },
            "model_manifest_sha256": sha(root / "model_manifest.json"),
            "model_settings": models.SETTINGS,
            "audio": [{"doc_id": r["doc_id"], "path": r["path"], "sha256": r["sha256"]} for r in records],
        },
    )


def ranked_rerank(model, query, docs, candidates, depth):
    chosen = candidates[:depth]
    texts = {d["doc_id"]: d["text"] for d in docs}
    scores = np.asarray(models.rerank(model, query, [texts[p["doc_id"]] for p in chosen]))
    if scores.shape != (len(chosen),) or not np.isfinite(scores).all():
        raise ValueError("Reranker output mismatch")
    return sorted(
        [{"doc_id": p["doc_id"], "score": float(s)} for p, s in zip(chosen, scores, strict=True)],
        key=lambda p: (-p["score"], p["doc_id"]),
    )


def run_search(root, queries, depth):
    documents = {
        condition: read(out(root) / f"{condition}_index_documents.json") for condition in ("reference", "asr")
    }
    results = {q["query_id"]: {} for q in queries}
    times = []
    embedding = models.load_model("embedding", root)
    try:
        started = time.perf_counter()
        query_vectors = models.embed(embedding, [q["text"] for q in queries], query=True)
        times.append(
            {
                "stage": "query_embeddings",
                "seconds": time.perf_counter() - started,
                "queries": len(queries),
                "includes_load": False,
                "warm_state": "first batch cold; remainder warm",
            }
        )
    finally:
        models.unload(embedding)
    for condition, docs in documents.items():
        ids = [d["doc_id"] for d in docs]
        bm25 = core.BM25(ids, [d["text"] for d in docs])
        vectors = np.load(out(root) / f"{condition}_embeddings.npy", allow_pickle=False)
        started = time.perf_counter()
        for q in queries:
            results[q["query_id"]][f"{condition}_bm25"] = bm25.rank(q["text"])
        times.append(
            {
                "stage": condition + "_bm25",
                "seconds": time.perf_counter() - started,
                "queries": len(queries),
                "includes_load": False,
                "warm_state": "first query cold; remainder warm",
            }
        )
        started = time.perf_counter()
        for q, v in zip(queries, query_vectors, strict=True):
            results[q["query_id"]][f"{condition}_dense"] = core.dense_rank(ids, vectors, v)
        times.append(
            {
                "stage": condition + "_dense",
                "seconds": time.perf_counter() - started,
                "queries": len(queries),
                "includes_load": False,
                "warm_state": "first query cold; remainder warm",
            }
        )
    reranker = models.load_model("reranker", root)
    try:
        for condition, docs in documents.items():
            started = time.perf_counter()
            for q in queries:
                runs = results[q["query_id"]]
                runs[f"{condition}_rerank"] = ranked_rerank(
                    reranker, q["text"], docs, runs[f"{condition}_dense"], depth
                )
            times.append(
                {
                    "stage": condition + "_rerank",
                    "seconds": time.perf_counter() - started,
                    "queries": len(queries),
                    "includes_load": False,
                    "warm_state": "first reference pair cold; subsequent pairs warm",
                }
            )
    finally:
        models.unload(reranker)
    return results, times


def score_runs(queries, qrels, runs, qualified):
    per_query = []
    for q in queries:
        for system in SYSTEMS:
            ranking = [p["doc_id"] for p in runs[q["query_id"]][system]]
            if len(ranking) != len(set(ranking)):
                raise ValueError("Duplicate ranked documents")
            if qualified:
                labels = {r["doc_id"]: r["grade"] for r in qrels if r["query_id"] == q["query_id"]}
                measures = core.retrieval_metrics(ranking, labels)
            else:
                anchor = q["anchor_doc_id"]
                rank = ranking.index(anchor) + 1 if anchor in ranking else None
                measures = {
                    "anchor_hit_at_5": float(rank is not None and rank <= 5),
                    "anchor_reciprocal_rank_at_10": 1.0 / rank if rank is not None and rank <= 10 else 0.0,
                }
            per_query.append(
                {
                    "query_id": q["query_id"],
                    "group": q["group_id"],
                    "role": q["role"],
                    "system": system,
                    **measures,
                }
            )
    summary = {}
    measure_names = (
        ("recall_at_5", "mrr_at_10", "ndcg_at_10")
        if qualified
        else ("anchor_hit_at_5", "anchor_reciprocal_rank_at_10")
    )
    for role in sorted({q["role"] for q in queries}):
        summary[role] = {}
        for system in SYSTEMS:
            selected = [p for p in per_query if p["role"] == role and p["system"] == system]
            summary[role][system] = {
                "n": len(selected),
                **{m: float(np.mean([p[m] for p in selected])) for m in measure_names},
            }
    return per_query, summary


def evaluate(root):
    queries, qrels, qualified = annotations(root)
    lock = read(out(root) / "experiment_lock.json")
    if lock["identity"] != identity(root):
        raise ValueError("Experiment changed after lock")
    runs, timing = run_search(root, queries, 10)
    per_query, summary = score_runs(queries, qrels, runs, qualified)
    write(out(root) / "ranked_runs.json", runs)
    write(out(root) / "retrieval_timing.json", timing)
    table(out(root) / "per_query_results.csv", per_query)
    write(
        out(root) / "retrieval_results.json",
        {
            "mode": "reviewed_benchmark" if qualified else "engineering_preview_anchor_recovery",
            "not_a_benchmark": not qualified,
            "scores": summary,
        },
    )
    measure = "recall_at_5" if qualified else "anchor_hit_at_5"
    comparisons = {}
    for a, b in [("reference_rerank", "asr_rerank"), ("asr_dense", "asr_rerank"), ("asr_bm25", "asr_dense")]:
        ids = [q for q in queries if q["role"] == "test"]

        def find(q, s):
            return next(p[measure] for p in per_query if p["query_id"] == q["query_id"] and p["system"] == s)

        result = core.paired_bootstrap(
            [find(q, b) for q in ids], [find(q, a) for q in ids], [q["group_id"] for q in ids]
        )
        comparisons[b + "_minus_" + a] = {**result, "a_system": b, "b_system": a}
    write(
        out(root) / "paired_comparisons.json",
        {"measure": measure, "provisional": not qualified, "comparisons": comparisons},
    )
    errors = []
    for q in queries:
        if q["role"] != "test":
            continue
        run = runs[q["query_id"]]
        anchor = q["anchor_doc_id"]
        errors.append(
            {
                "query_id": q["query_id"],
                "query": q["text"],
                "nominated_anchor": anchor,
                "asr_dense_candidate_missing": anchor not in [r["doc_id"] for r in run["asr_dense"][:10]],
                "asr_rerank_top5_missing": anchor not in [r["doc_id"] for r in run["asr_rerank"][:5]],
                "reference_rerank_top5_missing": anchor
                not in [r["doc_id"] for r in run["reference_rerank"][:5]],
                "annotation_status": "reviewed" if qualified else "unreviewed_anchor_only",
            }
        )
    table(out(root) / "error_trace.csv", errors)


def activity(root):
    queries, qrels, qualified = annotations(root)
    dev = [q for q in queries if q["role"] == "dev"]
    runs, timing = run_search(root, dev, 20)
    values, summary = score_runs(dev, qrels, runs, qualified)
    candidates = []
    for q in dev:
        targets = (
            {r["doc_id"] for r in qrels if r["query_id"] == q["query_id"] and r["grade"] == 2}
            if qualified
            else {q["anchor_doc_id"]}
        )
        for condition in ("reference", "asr"):
            ranking = [r["doc_id"] for r in runs[q["query_id"]][condition + "_dense"]]
            for depth in (10, 20):
                candidates.append(
                    {
                        "query_id": q["query_id"],
                        "condition": condition,
                        "candidate_depth": depth,
                        "measure": "candidate_recall" if qualified else "nominated_anchor_candidate_hit",
                        "value": len(targets.intersection(ranking[:depth])) / len(targets),
                        "denominator": len(targets),
                        "provisional": not qualified,
                    }
                )
    table(out(root) / "activity_candidates.csv", candidates)
    write(out(root) / "activity_runs.json", runs)
    write(
        out(root) / "activity_summary.json",
        {"candidate_depth": 20, "provisional": not qualified, "scores": summary, "timing": timing},
    )
    table(out(root) / "activity_results.csv", values)


def verify_index(root):
    manifest = read(out(root) / "index_manifest.json")
    if manifest.get("format") != "dimer_audio_archive" or manifest.get("version") != 1:
        raise ValueError("Unknown index format")
    required = {
        "asr_index_documents.json",
        "asr_embeddings.npy",
        "lexical_statistics.json",
        "model_manifest.json",
        "DATA_LICENSE.md",
        "audio_acquisition.json",
        "asr_settings.json",
        *PORTABLE_EXTRA,
    }
    if set(manifest["files"]) != required:
        raise ValueError("Index file inventory differs")
    if manifest["model_manifest_sha256"] != sha(root / "model_manifest.json"):
        raise ValueError("Search model identity changed")
    if manifest["model_settings"] != models.SETTINGS:
        raise ValueError("Search model settings changed")
    for name, checksum in manifest["files"].items():
        if Path(name).name != name or sha(out(root) / name) != checksum:
            raise ValueError("Index artifact hash mismatch")
    docs = read(out(root) / "asr_index_documents.json")
    if any(set(d) != {"doc_id", "text"} or not isinstance(d["text"], str) for d in docs):
        raise ValueError("Index must contain only automatic transcripts and document IDs")
    if len({d["doc_id"] for d in docs}) != len(docs):
        raise ValueError("Duplicate index document ID")
    vectors = np.load(out(root) / "asr_embeddings.npy", allow_pickle=False)
    if [d["doc_id"] for d in docs] != manifest["doc_ids"] or vectors.shape != (
        len(docs),
        manifest["dimension"],
    ):
        raise ValueError("Index shape/order mismatch")
    if not np.isfinite(vectors).all() or not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-5):
        raise ValueError("Invalid embedding vectors")
    return docs, vectors


def reload(root):
    if read(out(root) / "receipt_index.json")["pid"] == os.getpid():
        raise ValueError("Index reload must cross a fresh process boundary")
    docs, vectors = verify_index(root)
    queries = read(root / "queries.json")[:3]
    baseline = read(out(root) / "ranked_runs.json")
    model = models.load_model("embedding", root)
    try:
        qvectors = models.embed(model, [q["text"] for q in queries], query=True)
        rebuilt = models.embed(model, [d["text"] for d in docs[:3]])
        np.testing.assert_allclose(rebuilt, vectors[:3], atol=1e-5, rtol=1e-4)
    finally:
        models.unload(model)
    reranker = models.load_model("reranker", root)
    try:
        lexical = core.BM25([d["doc_id"] for d in docs], [d["text"] for d in docs])
        for q, v in zip(queries, qvectors, strict=True):
            dense = core.dense_rank([d["doc_id"] for d in docs], vectors, v)
            ranked = {
                "asr_bm25": lexical.rank(q["text"]),
                "asr_dense": dense,
                "asr_rerank": ranked_rerank(reranker, q["text"], docs, dense, 10),
            }
            for method, ranking in ranked.items():
                expected = baseline[q["query_id"]][method]
                if [r["doc_id"] for r in ranking] != [r["doc_id"] for r in expected]:
                    raise ValueError("Reload ranking mismatch")
                np.testing.assert_allclose(
                    [r["score"] for r in ranking], [r["score"] for r in expected], atol=1e-5, rtol=1e-4
                )
    finally:
        models.unload(reranker)
    whisper = models.load_model("whisper", root)
    try:
        repeated = models.transcribe(whisper, waves(root, corpus(root)[:3]))
    finally:
        models.unload(whisper)
    if repeated != [d["text"] for d in docs[:3]]:
        raise ValueError("Retranscription differs from frozen ASR run")
    write(
        out(root) / "verification.json",
        {
            "passed": True,
            "queries_replayed": 3,
            "clips_retranscribed": 3,
            "documents_reembedded": 3,
            "fresh_process": True,
            "atol": 1e-5,
            "rtol": 1e-4,
        },
    )


def search(root, query):
    docs, vectors = verify_index(root)
    model = models.load_model("embedding", root)
    try:
        v = models.embed(model, [query], query=True)[0]
    finally:
        models.unload(model)
    dense = core.dense_rank([d["doc_id"] for d in docs], vectors, v)
    reranker = models.load_model("reranker", root)
    try:
        ranking = ranked_rerank(reranker, query, docs, dense, 10)
    finally:
        models.unload(reranker)
    values = {d["doc_id"]: d["text"] for d in docs}
    result = [{**p, "transcript": values[p["doc_id"]]} for p in ranking[:5]]
    write(root / "interactive_search.json", {"query": query, "unscored": True, "results": result})


def figures(root, stage):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if stage == "prepare":
        records = corpus(root)
        fig, ax = plt.subplots(figsize=(9, 4))
        for role in ("dev", "test"):
            ax.hist(
                [r["duration"] for r in records if r["role"] == role],
                bins=[2, 8, 15, 25],
                alpha=0.5,
                label=role,
            )
        ax.set(xlabel="Clip duration (seconds)", ylabel="Recordings", title="Frozen sample · read speech")
        ax.legend()
    elif stage == "evaluate":
        value = read(out(root) / "retrieval_results.json")
        scores = value["scores"]["test"]
        metric = "anchor_hit_at_5" if value["not_a_benchmark"] else "recall_at_5"
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.barh(list(scores), [v[metric] for v in scores.values()])
        ax.set(
            xlim=(0, 1),
            xlabel=metric,
            title="UNREVIEWED ANCHOR PREVIEW — not benchmark relevance"
            if value["not_a_benchmark"]
            else "Reviewed evaluation queries",
        )
    else:
        return
    fig.tight_layout()
    fig.savefig(out(root) / f"{stage}.png", dpi=130)
    plt.close(fig)


def report(root):
    verify_index(root)
    queries, qrels, qualified = annotations(root)
    values, scores = score_runs(queries, qrels, read(out(root) / "ranked_runs.json"), qualified)
    if digest(scores) != digest(read(out(root) / "retrieval_results.json")["scores"]):
        raise ValueError("Retrieval scores fail recomputation")
    with (out(root) / "per_query_results.csv").open(encoding="utf-8", newline="") as handle:
        saved = list(csv.DictReader(handle))
    if saved != [{k: str(v) for k, v in row.items()} for row in values]:
        raise ValueError("Per-query CSV mismatch")
    if not read(out(root) / "verification.json")["passed"]:
        raise ValueError("Reload missing")
    import importlib.metadata

    write(
        out(root) / "environment.json",
        {
            name: importlib.metadata.version(name)
            for name in ("numpy", "torch", "transformers", "soundfile", "pyarrow")
        },
    )
    for name in (
        "sample_manifest.json",
        "model_manifest.json",
        "queries.json",
        "qrels.json",
        "annotation_manifest.json",
        "source.json",
    ):
        (out(root) / name).write_bytes((root / name).read_bytes())
    (out(root) / "audio_manifest.json").write_bytes((root / "sample_manifest.json").read_bytes())
    (out(root) / "queries.jsonl").write_text(
        "".join(json.dumps(q, ensure_ascii=False) + "\n" for q in queries), encoding="utf-8", newline="\n"
    )
    table(out(root) / "qrels.csv", qrels)
    for name in ("DATA_LICENSE.md", "dataset_audit.json"):
        if not (root / name).is_file() or not (root / name).read_text(encoding="utf-8").strip():
            raise ValueError("Required provenance absent: " + name)
        (out(root) / name).write_bytes((root / name).read_bytes())
    write(
        out(root) / "run_summary.json",
        {
            "status": "Candidate",
            "annotation_qualified": qualified,
            "scope": "small read-speech corpus; no operational or speaker independence claim",
            "mode": "reviewed_benchmark" if qualified else "engineering_preview_anchor_recovery",
            "warning": None
            if qualified
            else "Human query/qrel review incomplete. Scores measure nominated-anchor recovery only.",
            "verification": read(out(root) / "verification.json"),
            "stages": {s: read(out(root) / f"receipt_{s}.json") for s in STAGES[:-1]},
        },
    )
    portable = ["index_manifest.json", *read(out(root) / "index_manifest.json")["files"]]
    with zipfile.ZipFile(out(root) / "search_index.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for name in portable:
            archive.write(out(root) / name, name)
    files = {
        p.name: sha(p)
        for p in out(root).iterdir()
        if p.is_file() and p.name not in ("results.zip", "checksums.json", "receipt_report.json")
    }
    write(out(root) / "checksums.json", files)
    with zipfile.ZipFile(out(root) / "results.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for name in [*files, "checksums.json"]:
            archive.write(out(root) / name, name)
    with zipfile.ZipFile(out(root) / "results.zip") as archive:
        if archive.testzip():
            raise ValueError("Corrupt export")
    if (out(root) / "results.zip").stat().st_size > 100 * 1024**2:
        raise ValueError("Export exceeds100MiB")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--stage", choices=STAGES)
    parser.add_argument("--query")
    args = parser.parse_args()
    root = args.root.resolve()
    out(root).mkdir(parents=True, exist_ok=True)
    if args.query is not None:
        search(root, args.query)
        return
    if not args.stage:
        raise ValueError("A stage or query is required")
    previous = None
    for stage in STAGES[: STAGES.index(args.stage)]:
        receipt = read(out(root) / f"receipt_{stage}.json")
        if (
            not receipt.get("completed")
            or receipt["identity"] != identity(root)
            or receipt["previous"] != previous
        ):
            raise ValueError("Stale experiment receipt")
        for name, checksum in receipt["outputs"].items():
            if sha(out(root) / name) != checksum:
                raise ValueError("Changed output: " + name)
        previous = sha(out(root) / f"receipt_{stage}.json")
    prior = out(root) / f"receipt_{args.stage}.json"
    owned = set(read(prior)["outputs"]) if prior.exists() else set()
    prior.unlink(missing_ok=True)
    # Preserve downstream output ownership even when deterministic reruns produce
    # identical bytes. Invalidated receipts can never satisfy the predecessor gate.
    for stage in STAGES[STAGES.index(args.stage) + 1 :]:
        path = out(root) / f"receipt_{stage}.json"
        if path.exists():
            stale = read(path)
            stale.update(completed=False, invalidated_by=args.stage)
            write(path, stale)
    before = {p.name: sha(p) for p in out(root).iterdir() if p.is_file()}
    started = time.monotonic()
    gpu = args.stage in ("asr", "index", "evaluate", "activity", "reload")
    if gpu:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("Use the documented hosted T4 runtime")
        torch.cuda.reset_peak_memory_stats()
    globals()[args.stage](root)
    figures(root, args.stage)
    produced = {
        p.name: sha(p) for p in out(root).iterdir() if p.is_file() and not p.name.startswith("receipt_")
    }
    write(
        out(root) / f"receipt_{args.stage}.json",
        {
            "completed": True,
            "identity": identity(root),
            "previous": previous,
            "pid": os.getpid(),
            "seconds": time.monotonic() - started,
            "peak_allocated_gpu_bytes": int(torch.cuda.max_memory_allocated()) if gpu else None,
            "outputs": {n: h for n, h in produced.items() if before.get(n) != h or n in owned},
        },
    )
    print(args.stage + ": PASS", flush=True)


if __name__ == "__main__":
    main()
