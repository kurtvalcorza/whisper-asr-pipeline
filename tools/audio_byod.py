"""Optional, explicit-consent BYOD audio search; never called by default Run all."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import re
import subprocess
import sys
import uuid
import zipfile
from pathlib import Path

import numpy as np
import scipy
import soundfile as sf
from scipy.signal import resample_poly

import audio_core as core
import audio_models as models
import audio_runtime as runtime
from audio_data import inspect_audio, safe_path, sha256


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


# Exit status of the command-line entry point when an input is refused (see __main__).
REFUSED_EXIT = 3
# Bounded original input: 25 s of 48 kHz 24-bit stereo WAV is about 7.2 MB.
MAX_INPUT_BYTES = 16_000_000
TARGET_RATE = 16000
REEXPORT = "re-export it as WAV or FLAC (e.g. `ffmpeg -i <input> <clip>.wav`) and point the manifest at it"
CONVERSION = {
    "downmix": "arithmetic mean of channels (float64)",
    "resampler": "scipy.signal.resample_poly (up/down reduced by gcd, default Kaiser window)",
    "scipy": scipy.__version__,
    "output": "16 kHz mono 16-bit PCM WAV, samples clipped to [-32768, 32767] after rounding x * 32767",
}


class InputRefused(ValueError):
    """A BYOD input breaks the input contract; the message names the input, the rule and the fix."""

    def __init__(self, where: object, problem: str, fix: str):
        super().__init__(f"{where}: {problem}. Fix: {fix}")


def convert(data: bytes) -> tuple[bytes | None, dict]:
    """Return 16 kHz mono 16-bit WAV bytes (None if the input already is 16 kHz mono) and the
    original identity. Deterministic: the same original bytes give the same converted bytes."""
    values, rate = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    original = {
        "sha256": hashlib.sha256(data).hexdigest(),
        "sample_rate": int(rate),
        "channels": int(values.shape[1]),
        "frames": int(len(values)),
    }
    if rate == TARGET_RATE and values.shape[1] == 1:
        return None, original
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite or silent audio")
    mono = values.astype(np.float64).mean(axis=1)
    divisor = math.gcd(TARGET_RATE, int(rate))
    resampled = resample_poly(mono, TARGET_RATE // divisor, int(rate) // divisor)
    pcm = np.clip(np.round(resampled * 32767.0), -32768, 32767).astype("<i2")
    buffer = io.BytesIO()
    sf.write(buffer, pcm, TARGET_RATE, format="WAV", subtype="PCM_16")
    return buffer.getvalue(), original


def check_audio(where: str, source: Path) -> dict:
    """Convert to 16 kHz mono when needed, then run the eligibility checks shared with the default
    sample, explaining a refusal in terms the learner can act on."""
    if not source.is_file():
        raise InputRefused(
            where, f"file not found at {source}", "upload it next to the manifest or correct `path`"
        )
    if source.suffix.lower() not in (".wav", ".flac"):
        raise InputRefused(where, "only WAV or FLAC files are accepted", REEXPORT)
    size = source.stat().st_size
    if not 0 < size <= MAX_INPUT_BYTES:
        raise InputRefused(
            where, f"file is {size:,} bytes; the limit is {MAX_INPUT_BYTES:,}", "use a shorter clip (2-25 s)"
        )
    try:
        info = sf.info(str(source))
    except Exception as exc:  # undecodable input: soundfile raises its own runtime error types
        raise InputRefused(where, f"could not be decoded as audio ({exc})", REEXPORT) from exc
    duration = info.frames / info.samplerate if info.samplerate else 0.0
    if not 2 <= duration <= 25:
        raise InputRefused(
            where, f"{duration:.1f} s long; 2-25 seconds is required", "trim or split it into 2-25 s clips"
        )
    data = source.read_bytes()
    try:
        converted, original = convert(data)
    except Exception as exc:  # decode failure or nonfinite samples
        raise InputRefused(where, f"could not be converted to 16 kHz mono ({exc})", REEXPORT) from exc
    try:  # the authoritative check shared with the default sample, on what the model will hear
        checked = inspect_audio(converted if converted is not None else data)
    except (ValueError, RuntimeError) as exc:
        raise InputRefused(where, str(exc), "use a recording that contains audible speech") from exc
    return {
        **checked,
        "sha256": original["sha256"],  # identity of the user's file (playback, reacquisition)
        "original": original,
        "converted": None
        if converted is None
        else {
            "sha256": checked["sha256"],
            "sample_rate": TARGET_RATE,
            "channels": 1,
            "frames": checked["frames"],
            "method": CONVERSION,
        },
    }


def validate_input(path: Path) -> tuple[dict, list[dict]]:
    """Validate explicit rights and bounded local inputs before downloading models."""
    if not path.is_file():
        raise InputRefused(
            path, "manifest not found", "upload manifest.json there or set BYOD_MANIFEST to its path"
        )
    if path.stat().st_size > 1_000_000:
        raise InputRefused(path, "BYOD manifest exceeds 1 MB", "list at most 120 recordings")
    try:
        manifest = read(path)
    except (ValueError, UnicodeDecodeError) as exc:
        raise InputRefused(path, f"not valid UTF-8 JSON ({exc})", "fix the JSON syntax") from exc
    if not isinstance(manifest, dict):
        raise InputRefused(path, "the manifest must be a JSON object", "follow the example manifest")
    if manifest.get("rights_confirmed") is not True or not str(manifest.get("source_notes", "")).strip():
        raise InputRefused(
            path,
            "Explicit rights/consent acknowledgement and source notes required",
            'set "rights_confirmed": true and describe the source in "source_notes"',
        )
    records = manifest.get("records", [])
    if not isinstance(records, list) or not 1 <= len(records) <= 120:
        raise InputRefused(path, "BYOD accepts 1–120 recordings", 'list 1–120 entries under "records"')
    if any(not isinstance(r, dict) or "doc_id" not in r or "path" not in r for r in records):
        raise InputRefused(path, "every record needs `doc_id` and `path`", "follow the example manifest")
    ids = [r["doc_id"] for r in records]
    if len(set(map(str, ids))) != len(ids) or any(
        not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", name) for name in ids
    ):
        raise InputRefused(
            path,
            "Invalid/duplicate document IDs",
            "use unique IDs of 1–80 letters, digits, `_` or `-`",
        )
    verified = []
    for record in sorted(records, key=lambda r: r["doc_id"]):
        where = f"{record['doc_id']} ({record['path']})"
        try:
            source = safe_path(path.parent, record["path"])
        except (ValueError, TypeError) as exc:
            raise InputRefused(
                where, "Unsafe relative path", "give a path inside the manifest's folder, e.g. clip_01.wav"
            ) from exc
        verified.append({**record, **check_audio(where, source), "local_path": str(source)})
        reference = record.get("reference")
        if reference is not None and (not isinstance(reference, str) or not core.normalize(reference)):
            raise InputRefused(
                where,
                "Supplied references must contain valid nonempty text",
                "remove `reference` or fill it in",
            )
    query = manifest.get("probe_query")
    if query is not None and (not isinstance(query, str) or not query.strip() or len(query) > 2000):
        raise InputRefused(
            path,
            "Optional probe query must be nonempty and at most 2000 characters",
            "remove `probe_query` or shorten it",
        )
    return manifest, verified


def waves(records: list[dict]) -> list[np.ndarray]:
    """Model input: the original 16 kHz mono file, or its replayed and hash-checked conversion."""
    values = []
    for row in records:
        data = Path(row["local_path"]).read_bytes()
        if hashlib.sha256(data).hexdigest() != row["sha256"]:
            raise ValueError("BYOD audio changed after validation")
        payload = data
        if row.get("converted"):
            payload, _ = convert(data)
            if payload is None or hashlib.sha256(payload).hexdigest() != row["converted"]["sha256"]:
                raise ValueError("BYOD conversion did not reproduce the recorded converted audio")
        values.append(sf.read(io.BytesIO(payload), dtype="float32")[0])
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


def verify(root: Path, source: Path, bundle: Path) -> None:
    """Fresh-process reconstruction check; needs the original manifest and audio, unlike search."""
    receipt = read(bundle / "build_receipt.json")
    if receipt["pid"] == os.getpid() or sha256(source) != receipt["input_manifest_sha256"]:
        raise ValueError("Fresh process and unchanged input manifest required")
    package = bundle / "search"
    if sha256(package / "index_manifest.json") != receipt["search_manifest_sha256"]:
        raise ValueError("BYOD search package changed after build")
    docs, vectors = runtime.verify_package(package, root / "model_manifest.json")
    _, records = validate_input(source)
    expected_audio = read(package / "audio_acquisition.json")["records"]
    def identity(rows):
        return [(r["doc_id"], r["sha256"], (r.get("converted") or {}).get("sha256")) for r in rows]

    # Replays the conversion: validate_input re-converts every original and must reproduce its hash.
    if identity(records) != identity(expected_audio):
        raise ValueError("Audio identity changed (original or converted)")
    count = min(3, len(docs))
    # Re-embed exactly the build's first embedding batch: fp16 vectors depend on batch composition.
    embedded = min(models.SETTINGS["embedding_batch_size"], len(docs))
    model = models.load_model("embedding", root)
    try:
        rebuilt = models.embed(model, [r["text"] for r in docs[:embedded]])
    finally:
        models.unload(model)
    np.testing.assert_allclose(rebuilt, vectors[:embedded], atol=1e-5, rtol=1e-4)
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
            "reembedded": embedded,
            "probe_query_replayed": True,
            "atol": 1e-5,
            "rtol": 1e-4,
        },
    )


LICENSE = """# User-supplied recordings

These recordings were supplied by the person who ran this BYOD build, who confirmed they have
the rights and consent to process them. Source notes supplied with the build:

{notes}

Automatic transcripts in this package are model output. Do not redistribute this package beyond
the permissions that apply to the original recordings.
"""


def build(root: Path, source: Path, verify_in_subprocess: bool = True) -> Path:
    """Transcribe, index and export BYOD recordings in the same format as the default archive.

    Layout: ``search/`` is the portable automatic-transcript package (searchable by
    ``audio_runtime.py --index`` or its own ``search_archive.py``); ``evaluation.json`` holds any
    supplied references and stays outside it. ``search_index.zip`` is ``search/`` zipped.
    """
    provided, records = validate_input(source)
    if not (root / "model_manifest.json").is_file():
        raise ValueError("Run root must contain the pinned model_manifest.json")
    bundle = root / "byod" / uuid.uuid4().hex
    bundle.mkdir(parents=True)
    model = models.load_model("whisper", root)
    try:
        asr_settings = {
            key: model[key]
            for key in ("generation_config", "generation_overrides", "language_token_id", "prefix_length")
        }
        texts = models.transcribe(model, waves(records))
    finally:
        models.unload(model)
    if len(texts) != len(records):
        raise ValueError("ASR output count mismatch")
    docs = [{"doc_id": r["doc_id"], "text": text} for r, text in zip(records, texts, strict=True)]
    model = models.load_model("embedding", root)
    try:
        vectors = models.embed(model, texts)
    finally:
        models.unload(model)
    if vectors.shape[0] != len(docs) or not np.isfinite(vectors).all():
        raise ValueError("Bad document embeddings")
    acquisition = {
        "instructions": (
            "Supply the original local audio at these relative paths; audio is not bundled. sha256 is the "
            "original file; sample_rate/channels/frames/duration describe the 16 kHz mono model input. "
            "Inputs that were not 16 kHz mono are converted with the recorded method; the converted "
            "identity lets the reconstruction check replay the conversion."
        ),
        "records": [
            {
                k: r[k]
                for k in (
                    "doc_id", "path", "sha256", "sample_rate", "channels", "frames", "duration",
                    "original", "converted",
                )
            }
            for r in records
        ],
    }
    package = runtime.write_search_package(
        bundle / "search",
        root,
        docs,
        vectors,
        acquisition,
        LICENSE.format(notes=str(provided["source_notes"]).strip()),
        asr_settings,
        "byod",
    )
    runtime.verify_package(package, root / "model_manifest.json")
    with zipfile.ZipFile(bundle / "search_index.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(package.iterdir()):
            archive.write(path, path.name)
    query = provided.get("probe_query") or next(
        (t for t in texts if t.strip()), "Ano ang nilalaman ng audio?"
    )
    write(
        bundle / "probe.json",
        {
            "query": query,
            "purpose": "Mechanical reconstruction probe; no accuracy claim",
            "rankings": rankings(root, docs, vectors, query),
        },
    )
    references = [
        (r["doc_id"], r["reference"], text)
        for r, text in zip(records, texts, strict=True)
        if r.get("reference")
    ]
    metrics = (
        core.asr_metrics([ref for _, ref, _ in references], [hyp for _, _, hyp in references])
        if references
        else None
    )
    # Evaluation evidence (including supplied references) lives beside, never inside, search/.
    write(
        bundle / "evaluation.json",
        {
            "asr_status": "measured_on_supplied_references" if references else "not_measurable",
            "recordings": len(records),
            "reference_count": len(references),
            "reference_doc_ids": [doc_id for doc_id, _, _ in references],
            "asr": metrics,
            "retrieval_status": "not_measurable",
            "reason": "No reviewed query/relevance labels supplied; BYOD v1 does not import qrels.",
        },
    )
    write(
        bundle / "build_receipt.json",
        {
            "pid": os.getpid(),
            "input_manifest_sha256": sha256(source),
            "model_manifest_sha256": sha256(root / "model_manifest.json"),
            "search_manifest_sha256": sha256(package / "index_manifest.json"),
            "search_index_zip_sha256": sha256(bundle / "search_index.zip"),
        },
    )
    if verify_in_subprocess:
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
    write(
        root / "byod" / "latest.json",
        {
            "bundle": str(bundle),
            "search_dir": str(package),
            "search_index_zip": str(bundle / "search_index.zip"),
            "evaluation": str(bundle / "evaluation.json"),
            "verification": str(bundle / "verification.json"),
            "audio_base": str(source.parent),
            "input_manifest": str(source),
        },
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
        # Validate first: every ValueError raised here is an input refusal (InputRefused), reported
        # as one line naming the input, the rule and the fix. Failures after validation still raise.
        try:
            validate_input(args.manifest.resolve())
        except ValueError as exc:
            print(f"BYOD input refused: {exc}", flush=True)
            sys.exit(REFUSED_EXIT)
        print(build(args.root.resolve(), args.manifest.resolve()))
