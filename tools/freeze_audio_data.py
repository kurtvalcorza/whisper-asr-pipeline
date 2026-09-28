"""Maintainer-only deterministic source audit and cohort freezing (no model access)."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

from audio_data import inspect_audio, normalize, sha256

REVISION = "70bb2e84b976b7e960aa89f1c648e09c59f894dd"
FILES = [
    {
        "split": "validation",
        "bytes": 454686207,
        "sha256": "693d176879c83b3ce9fa86d350f2351237e41e4f5e85bad1d8d7c0b7137d9f43",
    },
    {
        "split": "test",
        "bytes": 1102327716,
        "sha256": "cf89a30f324ca7c10dcc3d768d24c21966d7c9d66e1e9bc687237413e1cd88e8",
    },
]


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")


def freeze(cache: Path, output: Path) -> None:
    files = []
    candidates = []
    excluded = []
    source_counts = {}
    for base in FILES:
        item = dict(base)
        item["filename"] = f"{item['split']}-00000-of-00001.parquet"
        item["path"] = "parquet-data/fil_ph/" + item["filename"]
        item["url"] = f"https://huggingface.co/datasets/google/fleurs/resolve/{REVISION}/{item['path']}"
        path = cache / item["filename"]
        if path.stat().st_size != item["bytes"] or sha256(path) != item["sha256"]:
            raise ValueError("Source hash failure")
        files.append(item)
        offset = 0
        for batch in pq.ParquetFile(path).iter_batches(batch_size=32):
            for local, row in enumerate(batch.to_pylist()):
                index = offset + local
                key = f"{item['split']}:{index:06d}:{row['id']}"
                try:
                    audio = inspect_audio(row["audio"]["bytes"])
                    reference = row["raw_transcription"]
                    if not normalize(reference):
                        raise ValueError("Empty normalized reference")
                except ValueError as exc:
                    excluded.append({"recording_key": key, "reason": str(exc)})
                    continue
                band = 0 if audio["duration"] <= 8 else 1 if audio["duration"] <= 15 else 2
                candidates.append(
                    {
                        "doc_id": "fil-" + hashlib.sha256(key.encode()).hexdigest()[:16],
                        "recording_key": key,
                        "source_split": item["split"],
                        "row_index": index,
                        "role": "dev" if item["split"] == "validation" else "test",
                        "source_id": str(row["id"]),
                        "reference": reference,
                        "reference_normalized": normalize(reference),
                        "duration_band": band,
                        **audio,
                    }
                )
            offset += len(batch)
        source_counts[item["split"]] = offset
        print(f"Audited {item['split']}: {offset} rows", flush=True)
    # A connected component captures any shared sentence ID, normalized text or exact audio.
    parent = list(range(len(candidates)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    seen = {}
    for i, record in enumerate(candidates):
        for field in ("source_id", "reference_normalized", "sha256"):
            key = (field, record[field])
            if key in seen:
                parent[find(i)] = find(seen[key])
            seen[key] = i
    groups = {}
    for i, record in enumerate(candidates):
        groups.setdefault(find(i), []).append(record)
    eligible = []
    cross_role = 0
    for group in groups.values():
        if len({r["role"] for r in group}) > 1:
            cross_role += len(group)
            excluded.extend(
                {"recording_key": r["recording_key"], "reason": "Cross-role duplicate family"} for r in group
            )
            continue
        ordered = sorted(
            group, key=lambda r: hashlib.sha256(("42:" + r["recording_key"]).encode()).hexdigest()
        )
        record = ordered[0]
        record["family_id"] = (
            "family-"
            + hashlib.sha256("|".join(sorted(r["recording_key"] for r in group)).encode()).hexdigest()[:16]
        )
        eligible.append(record)
        excluded.extend(
            {
                "recording_key": r["recording_key"],
                "reason": "Repeated family; one deterministic representative",
            }
            for r in ordered[1:]
        )
    selected = []
    for role, target in [("dev", 40), ("test", 80)]:
        queues = [
            sorted(
                [r for r in eligible if r["role"] == role and r["duration_band"] == band],
                key=lambda r: hashlib.sha256(("42:" + r["recording_key"]).encode()).hexdigest(),
            )
            for band in range(3)
        ]
        chosen = []
        while len(chosen) < target and any(queues):
            for queue in queues:
                if queue and len(chosen) < target:
                    chosen.append(queue.pop(0))
        if len(chosen) != target:
            raise ValueError(f"Insufficient eligible {role} families")
        selected.extend(chosen)
    selected.sort(key=lambda r: r["doc_id"])
    for record in selected:
        record["path"] = "audio/" + record["doc_id"] + ".wav"
    manifest = {
        "format_version": 1,
        "dataset": "google/fleurs",
        "configuration": "fil_ph",
        "revision": REVISION,
        "license": "CC-BY-4.0",
        "attribution": (
            "Google FLEURS; Conneau et al. (2022), FLEURS: Few-shot Learning Evaluation of Universal "
            "Representations of Speech; https://arxiv.org/abs/2205.12446"
        ),
        "transformations": (
            "Selected 120 original 16 kHz mono recordings; original encoded audio and display transcripts "
            "preserved. NFC/lowercase/punctuation-to-space normalized text stored separately."
        ),
        "seed": 42,
        "selection": (
            "SHA256(seed:stable-recording-key), one per connected ID/text/audio family, "
            "round-robin duration bands"
        ),
        "files": files,
        "records": selected,
    }
    write_json(output / "audio_sample.json", manifest)
    audit = {
        "format_version": 1,
        "source_revision": REVISION,
        "source_files": files,
        "source_rows": source_counts,
        "decoded_eligible_rows": len(candidates),
        "cross_role_duplicate_rows_excluded": cross_role,
        "eligible_unique_families": len(eligible),
        "selected_clips": len(selected),
        "selected_roles": dict(Counter(r["role"] for r in selected)),
        "duration_bands": {
            role: dict(Counter(str(r["duration_band"]) for r in selected if r["role"] == role))
            for role in ("dev", "test")
        },
        "total_audio_seconds": sum(r["duration"] for r in selected),
        "exclusions": excluded,
        "manifest_sha256": sha256(output / "audio_sample.json"),
        "audio_gate_passed": True,
        "benchmark_qualified": False,
        "limitations": [
            "No verified speaker independence or capture location metadata.",
            "Unknown pretraining overlap.",
            "Read speech only.",
            "Search relevance annotations require separate human review.",
        ],
    }
    write_json(output / "audio_dataset_audit.json", audit)
    print(f"Frozen {len(selected)} recordings; {audit['total_audio_seconds']:.2f} seconds", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("tools"))
    args = parser.parse_args()
    freeze(args.cache, args.output)
