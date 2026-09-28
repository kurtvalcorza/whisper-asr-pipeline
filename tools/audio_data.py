"""Acquire the frozen FLEURS cohort without executing remote dataset code."""

from __future__ import annotations

import hashlib
import io
import json
import unicodedata
import urllib.request
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text).lower()
    return " ".join("".join(" " if unicodedata.category(c).startswith("P") else c for c in text).split())


def safe_path(root: Path, relative: str) -> Path:
    candidate = (root / relative).resolve()
    if (
        Path(relative).is_absolute()
        or candidate == root.resolve()
        or not candidate.is_relative_to(root.resolve())
    ):
        raise ValueError("Unsafe relative path")
    return candidate


def inspect_audio(data: bytes) -> dict:
    """Validate entire recording; conversion is deliberately unnecessary for FLEURS."""
    values, rate = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    if rate != 16000 or values.shape[1] != 1:
        raise ValueError("Expected original 16 kHz mono audio")
    duration = len(values) / rate
    if not 2 <= duration <= 25:
        raise ValueError("Duration outside 2–25 seconds")
    if not np.isfinite(values).all() or not np.any(values != 0):
        raise ValueError("Nonfinite or silent audio")
    return {
        "sample_rate": rate,
        "channels": 1,
        "frames": len(values),
        "duration": duration,
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def download(item: dict, destination: Path) -> None:
    """Bounded atomic transfer with verification of both cache and downloads."""
    if destination.exists():
        if destination.stat().st_size == item["bytes"] and sha256(destination) == item["sha256"]:
            return
        raise ValueError(f"Cached source integrity failure: {destination.name}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(item["url"], headers={"User-Agent": "DIMER-audio-capstone/1"})
    try:
        with urllib.request.urlopen(request, timeout=120) as response, partial.open("wb") as out:
            total = 0
            for chunk in iter(lambda: response.read(8 * 1024 * 1024), b""):
                total += len(chunk)
                if total > item["bytes"]:
                    raise ValueError("Source exceeds pinned byte count")
                out.write(chunk)
        if total != item["bytes"] or sha256(partial) != item["sha256"]:
            raise ValueError("Source integrity failure")
        partial.replace(destination)
    finally:
        if partial.exists():
            partial.unlink()


def acquire(root: Path | str) -> list[dict]:
    """Verify two pinned Parquets and extract only the 120 frozen original WAVs."""
    root = Path(root)
    manifest = json.loads((root / "sample_manifest.json").read_text(encoding="utf-8"))
    records = manifest["records"]
    validate_manifest(manifest)
    if sum(f["bytes"] for f in manifest["files"]) > 2_000_000_000:
        raise ValueError("Source download budget exceeded")
    for item in manifest["files"]:
        source = safe_path(root / "cache", item["filename"])
        download(item, source)
        wanted = {r["row_index"]: r for r in records if r["source_split"] == item["split"]}
        offset = 0
        for batch in pq.ParquetFile(source).iter_batches(batch_size=32):
            for local_index, row in enumerate(batch.to_pylist()):
                index = offset + local_index
                if index not in wanted:
                    continue
                record = wanted[index]
                if str(row["id"]) != record["source_id"] or row["raw_transcription"] != record["reference"]:
                    raise ValueError("Source reference/ID drift")
                data = row["audio"]["bytes"]
                actual = inspect_audio(data)
                for field in ("sha256", "sample_rate", "channels", "frames", "duration"):
                    if actual[field] != record[field]:
                        raise ValueError(f"Audio integrity failure: {field}")
                target = safe_path(root, record["path"])
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.exists() and sha256(target) != record["sha256"]:
                    raise ValueError("Cached audio integrity failure")
                target.write_bytes(data)
            offset += len(batch)
    for record in records:
        if sha256(safe_path(root, record["path"])) != record["sha256"]:
            raise ValueError("Missing or corrupt extracted audio")
    return records


def validate_manifest(manifest: dict) -> None:
    records = manifest["records"]
    if len(records) != 120 or len({r["doc_id"] for r in records}) != 120:
        raise ValueError("Expected 120 unique documents")
    for field in ("family_id", "source_id", "reference_normalized", "sha256", "path"):
        if len({r[field] for r in records}) != 120:
            raise ValueError(f"Duplicate {field}")
    for role, split, count in [("dev", "validation", 40), ("test", "test", 80)]:
        selected = [r for r in records if r["role"] == role]
        if len(selected) != count or any(r["source_split"] != split for r in selected):
            raise ValueError("Invalid role/source split")
    for record in records:
        if (
            record["path"] != f"audio/{record['doc_id']}.wav"
            or "/" in record["doc_id"]
            or "\\" in record["doc_id"]
        ):
            raise ValueError("Invalid audio path")
        if (
            normalize(record["reference"]) != record["reference_normalized"]
            or not record["reference_normalized"]
        ):
            raise ValueError("Reference normalization mismatch")
    if {f["split"] for f in manifest["files"]} != {"validation", "test"} or len(manifest["files"]) != 2:
        raise ValueError("Invalid source file inventory")
    for item in manifest["files"]:
        expected = f"https://huggingface.co/datasets/google/fleurs/resolve/{manifest['revision']}/parquet-data/fil_ph/{item['split']}-00000-of-00001.parquet"
        if item["url"] != expected or item["filename"] != f"{item['split']}-00000-of-00001.parquet":
            raise ValueError("Unexpected source URL/path")
