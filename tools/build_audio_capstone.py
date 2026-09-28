"""Generate the standalone, guided Filipino audio archive capstone."""

# Ruff permits long notebook prose and embedded source strings.
# ruff: noqa: E501
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
NOTEBOOK = REPO / "tutorials/DIMER_Filipino_Audio_Archive_Search_Capstone.ipynb"


def carried_files() -> dict[str, str]:
    """Freeze plain source/data as UTF-8 LF; no repository imports at runtime."""
    paths = {
        f"audio_{name}.py": f"tools/audio_{name}.py" for name in ("runtime", "core", "data", "models", "byod")
    }
    paths.update(
        {
            "sample_manifest.json": "tools/audio_sample.json",
            "dataset_audit.json": "tools/audio_dataset_audit.json",
            "queries.json": "tools/audio_queries.json",
            "qrels.json": "tools/audio_qrels.json",
            "annotation_manifest.json": "tools/audio_annotations.json",
            "model_manifest.json": "tools/audio_models.json",
            "requirements.txt": "tools/audio-requirements.lock",
            "DATA_LICENSE.md": "docs/audio-data-license.md",
            "ANNOTATION_REVIEW.md": "docs/audio-annotation-review.md",
        }
    )
    files = {name: (REPO / path).read_text(encoding="utf-8") for name, path in paths.items()}
    source = {
        "format_version": 1,
        "builder": "tools/build_audio_capstone.py",
        "builder_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "files": {name: hashlib.sha256(text.encode("utf-8")).hexdigest() for name, text in files.items()},
    }
    files["source.json"] = json.dumps(source, indent=2, ensure_ascii=False) + "\n"
    return files


PREFLIGHT = r"""
from pathlib import Path
import os, sys, json, hashlib, platform, shutil, subprocess, zipfile, urllib.request
from IPython.display import display, Markdown, Audio, Image, HTML

ROOT = Path('/content/dimer_fil_audio_preview')
assert platform.system() == 'Linux', 'Use a fresh hosted Colab T4 runtime.'
assert shutil.disk_usage('/').free >= 20 * 1024**3, 'At least 20 GiB free disk required.'
gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], text=True)
assert 'T4' in gpu, f'This candidate targets a Colab T4; found {gpu.strip()}'
ROOT.mkdir(parents=True, exist_ok=True)
print('GPU:', gpu.strip())
print('Engineering preview: human relevance review is pending. No benchmark-quality claim.')
"""

INSTALL = r'''
for name, content in FILES.items():
    destination = ROOT / name
    destination.write_bytes(content.encode('utf-8'))
source = json.loads(FILES['source.json'])
for name, expected in source['files'].items():
    assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, name

uv_url = 'https://files.pythonhosted.org/packages/1e/fd/432451d732917c49152a291de3ef171aa6b0f1a22d39780fb2c1f085ca4c/uv-0.12.15-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl'
uv_sha = 'aee9802f46bae436bd91751bb33ddeb379ef1596b5c19df193219d545d244b60'
wheel = ROOT / 'uv.whl'
if not wheel.exists():
    with urllib.request.urlopen(uv_url, timeout=120) as response:
        raw = response.read(20_081_405)
    assert len(raw) == 20_081_404 and hashlib.sha256(raw).hexdigest() == uv_sha
    wheel.write_bytes(raw)
assert wheel.stat().st_size == 20_081_404 and hashlib.sha256(wheel.read_bytes()).hexdigest() == uv_sha
with zipfile.ZipFile(wheel) as archive:
    candidates = [name for name in archive.namelist() if name.endswith('/uv')]
    assert len(candidates) == 1, 'Pinned uv wheel layout changed'
    UV = ROOT / 'uv'
    UV.write_bytes(archive.read(candidates[0]))
UV.chmod(0o755)
ENV = dict(os.environ, UV_PYTHON_INSTALL_DIR=str(ROOT / 'python'), TOKENIZERS_PARALLELISM='false',
           MPLBACKEND='Agg', HF_HUB_DISABLE_TELEMETRY='1')

def command(args, log_name):
    """Stream output and preserve the real subprocess exit code."""
    with (ROOT / log_name).open('w', encoding='utf-8') as log:
        process = subprocess.Popen([str(a) for a in args], cwd=ROOT, env=ENV,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                   encoding='utf-8', errors='replace', bufsize=1)
        for line in process.stdout:
            print(line, end='')
            log.write(line)
            log.flush()
        status = process.wait()
    if status:
        raise RuntimeError(f'Command failed ({status}); inspect {ROOT / log_name}')

command([UV, 'python', 'install', '3.12.12'], 'python-install.log')
PY = ROOT / 'venv/bin/python'
if not PY.exists():
    command([UV, 'venv', '--python', '3.12.12', ROOT / 'venv'], 'venv.log')
command([UV, 'pip', 'sync', '--python', PY, '--require-hashes', '--only-binary', ':all:',
         ROOT / 'requirements.txt'], 'dependencies.log')
RESULTS = ROOT / 'results'

def stage(name):
    command([PY, ROOT / 'audio_runtime.py', '--root', ROOT, '--stage', name], name + '.log')

def record(name):
    return json.loads((RESULTS / name).read_text(encoding='utf-8'))

def show_json(name):
    display(HTML('<pre>' + __import__('html').escape(json.dumps(record(name), indent=2, ensure_ascii=False)) + '</pre>'))

def show_csv(name, limit=8):
    import csv, html
    with (RESULTS / name).open(encoding='utf-8', newline='') as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        print('No rows:', name)
        return
    headers = list(rows[0])
    render = lambda value: html.escape(str(value))
    text = '<table><tr>' + ''.join('<th>' + render(h) + '</th>' for h in headers) + '</tr>'
    for row in rows[:limit]:
        text += '<tr>' + ''.join('<td>' + render(row[h]) + '</td>' for h in headers) + '</tr>'
    display(HTML(text + '</table>'))
    print(f'Showing {min(limit, len(rows))}/{len(rows)} rows. Full table: {RESULTS / name}')

def picture(name):
    display(Image(filename=str(RESULTS / name)))
'''


def build() -> dict:
    """Return a deterministic notebook with plain Python code cells."""
    cells = []

    def add(kind: str, text: str) -> None:
        text = text.strip() + "\n"
        cell = {
            "cell_type": kind,
            "id": hashlib.sha256(f"{len(cells)}:{text}".encode()).hexdigest()[:12],
            "metadata": {},
            "source": text.splitlines(keepends=True),
        }
        if kind == "code":
            ast.parse(text)
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)

    def md(text: str) -> None:
        add("markdown", text)

    def code(text: str) -> None:
        add("code", text)

    md("""# Filipino Audio Archive Search
## How do transcription errors affect what we find?

**Engineering preview · Candidate · DIMER composed-systems capstone**

**Standard:** DIMER NOTEBOOK_SPEC 2.2 · **Profile:** `E2E` · **Mode:** `GUIDED` · **Carrier:** standalone.

**Input → System → Output:** Filipino read-speech recordings → Whisper transcripts → lexical or Qwen3 search → ranked, playable recordings with visible evidence.

You will measure transcription errors, compare six search conditions, trace failures through the pipeline, change one development-only setting and export a reproducible search artifact. Basic Python and Colab familiarity is enough; no prior ML experience is assumed.

**Evidence boundary:** the 60 queries are AI-authored drafts. All 7,200 relevance cells are unjudged, including nominated anchors. Default results are **anchor-recovery engineering diagnostics**, not Recall@5, MRR or nDCG benchmark results. Human review and a fresh hosted run are required before qualification. Completion/reflection notes are optional, not required submissions.

Use a **fresh Colab T4** and **Runtime → Run all**. No secret, login, paid API or repository clone is needed. Resource targets are 60 minutes, 20 GiB free disk and ≤12 GiB peak allocated GPU memory; these are unverified targets, not measured guarantees.""")
    md("""### 1. Orient and predict

A transcript can contain many errors yet preserve the key concept. One wrong name can also make a mostly correct transcript hard to find. **Predict:** which matters more for your search query—overall transcription accuracy or preservation of its useful evidence?

| Method | Reference transcripts | Whisper transcripts |
|---|---|---|
| BM25 keyword matching | R-BM25 | A-BM25 |
| Qwen3 cosine similarity | R-Dense | A-Dense |
| Dense top 10 → Qwen3 reranking | R-Rerank | A-Rerank |

All six conditions use the same 120 document IDs and 60 queries. The 40 development and 80 evaluation recordings contribute 20 and 40 query anchors respectively. All recordings are searchable distractors in both roles. Reference text is an oracle **input condition**, not a guaranteed performance ceiling.

The pinned models are [Whisper large-v3-turbo](https://huggingface.co/openai/whisper-large-v3-turbo), [Qwen3 Embedding 0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) and [Qwen3 Reranker 0.6B](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B). Immutable revisions and file hashes are carried in `model_manifest.json`. **No fine-tuning occurs:** you construct and evaluate a search system around frozen pretrained models.

Data: [Google FLEURS](https://huggingface.co/datasets/google/fleurs), `fil_ph`, CC BY 4.0; [Conneau et al. (2022)](https://arxiv.org/abs/2205.12446). Read speech does not establish performance on meetings, regional accents or noisy archives. Sentence families are not verified speaker identities; source/pretraining overlap is unknown.""")
    md("""### 2. Infrastructure — prepare an isolated environment

The next cells verify the target runtime, materialize embedded source and hash-bound manifests, and install a locked Python environment. Models run in sequential subprocesses to release GPU memory. A failure is visible and stops execution; do not skip failed cells or accept incomplete output as a successful run.""")
    code(PREFLIGHT)
    code(
        "# Embedded, inspectable source and manifests; no remote DIMER code import.\nFILES = "
        + repr(carried_files())
        + "\n"
        + INSTALL
    )
    md("""### 3. Inspect the recordings before modelling

The frozen sample uses one recording per audited source-text family. Eligibility was decided before predictions: decodable finite non-silent mono audio, 2–25 seconds, nonempty reference, and no family crossing roles. Exact source files, hashes, exclusions and realised duration bands are preserved.

**Predict:** listen to three development clips and identify one name, number or concept you would need to retain to find each clip. Do not use evaluation clips to tune settings.""")
    code("stage('prepare')\nshow_json('qualification.json')\npicture('prepare.png')")
    code("""corpus_rows = record('corpus.json')
dev_examples = [r for r in corpus_rows if r['role'] == 'dev'][:3]
for row in dev_examples:
    display(Markdown('**' + row['doc_id'] + '** — ' + row['reference']))
    display(Audio(filename=str(ROOT / row['path'])))""")
    md(
        """**What to notice:** duration bands are not speaker groups. The model sees recordings; retrieval later sees only their transcripts. The source text may itself contain unusual names or translation choices. Human review of query naturalness, reference/audio correspondence and full-corpus relevance is still pending. `annotation_review.csv` is a review aid, not completed evidence."""
    )
    md("""### 4. Transcribe and measure edits

Whisper transcribes Filipino/Tagalog using deterministic greedy decoding, without reference prompts or previous-clip context. Blank outputs remain in the experiment; a detected length-cap truncation is a failure.

**WER** counts word substitutions, deletions and insertions divided by reference words. **CER** does the same for characters, including normalized spaces. Corpus rates use summed edit counts and summed reference lengths. WER can exceed 100%; nothing is capped. Scoring applies NFC, lowercase, punctuation-to-space and whitespace collapse while preserving diacritics.""")
    code("stage('asr')\nshow_csv('transcriptions.csv')\nshow_json('asr_timing.json')")
    code("""metrics = record('asr_metrics.json')
for role, values in metrics.items():
    display(Markdown(f"**{role}** — WER {values['wer']:.3f}; CER {values['cer']:.3f}. "
                     f"Word edits: {values['word']}; character edits: {values['character']}"))
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(7, 3))
for role, values in metrics.items():
    ax.hist([r['wer'] for r in values['per_clip']], bins=15, alpha=0.5, label=role)
ax.set(xlabel='Per-recording WER (uncapped)', ylabel='Recordings', title='Where are the transcription errors?')
ax.legend(); plt.show()""")
    code("""import difflib, html
automatic = {r['doc_id']: r['text'] for r in record('asr_documents.json')}
for row in dev_examples:
    reference, hypothesis = row['reference'].split(), automatic[row['doc_id']].split()
    pieces = []
    for tag, i, j, k, l in difflib.SequenceMatcher(None, reference, hypothesis).get_opcodes():
        old, new = html.escape(' '.join(reference[i:j])), html.escape(' '.join(hypothesis[k:l]))
        pieces.append(old if tag == 'equal' else '<del>' + old + '</del> <mark>' + new + '</mark>')
    display(HTML('<strong>' + html.escape(row['doc_id']) + '</strong><p>' + ' '.join(pieces) + '</p>'))
print('Visual word diff only: strikethrough = removed/replaced reference; highlight = ASR addition/replacement.')
print('Scientific WER/CER above use the documented normalized Levenshtein alignment, not this display diff.')""")
    md("""**Interpret:** inspect substitutions involving names and numbers, and compare them with the words you predicted would matter. A correlation between WER and retrieval failure does not establish causation.

<details><summary>Worked interpretation</summary>An insertion-heavy hypothesis can have WER above 1. A blank hypothesis deletes all reference units. Neither case is silently removed. Search may still recover an imperfect transcript if the discriminating concept survives.</details>""")
    md("""### 5. Build lexical and semantic indexes

Each recording is one document. BM25 counts matching normalized words, with fixed `k1=1.2, b=0.75` and no stopword removal; each transcript condition fits its own statistics. Dense search compares normalized Qwen3 embeddings by exact cosine similarity. Ties use ascending document ID.

The reference and automatic indexes stay separate. Blank ASR documents retain their IDs. Query/document token ceilings are checked without silent truncation. Raw lexical, cosine and reranker scores are different ranking values, not comparable calibrated probabilities.""")
    code("stage('index')\nshow_json('index_manifest.json')")
    md("""### 6. Compare the six systems

**Predict:** can a reranker rescue a recording absent from its ten dense candidates?

The experiment lock binds source, annotations, model identities, prompts and candidate depth. Evaluate the untouched query set only after this lock. In this preview, **anchor hit@5** asks whether the nominated draft target appears in the first five results; **anchor reciprocal rank@10** measures where that target appears in the first ten. They do not account for other potentially relevant recordings.

After complete human review, the graded branch reports Recall@5 over all grade-2 references, MRR@10 for the first grade-2 result and nDCG@10 with gains 0/1/3. Do not rename preview diagnostics to these metrics.""")
    code(
        "stage('evaluate')\nshow_json('retrieval_results.json')\npicture('evaluate.png')\nshow_json('paired_comparisons.json')\nshow_csv('error_trace.csv')"
    )
    code("""draft_queries = json.loads((ROOT / 'queries.json').read_text(encoding='utf-8'))
ranked_runs = record('ranked_runs.json')
test_queries = [q for q in draft_queries if q['role'] == 'test']
failed = [q for q in test_queries if q['anchor_doc_id'] not in
          [r['doc_id'] for r in ranked_runs[q['query_id']]['asr_rerank'][:5]]]
example = (failed or test_queries)[0]
anchor = next(r for r in corpus_rows if r['doc_id'] == example['anchor_doc_id'])
display(Markdown('**Inspect a ' + ('missed' if failed else 'recovered') + ' draft anchor:** ' + example['text']))
display(Markdown('**Reference:** ' + anchor['reference']))
display(Markdown('**Whisper:** ' + automatic[anchor['doc_id']]))
display(Audio(filename=str(ROOT / anchor['path'])))
for name in ('reference_rerank', 'asr_dense', 'asr_rerank'):
    ids = [r['doc_id'] for r in ranked_runs[example['query_id']][name]]
    print(name, 'anchor rank:', ids.index(anchor['doc_id']) + 1 if anchor['doc_id'] in ids else 'absent',
          'top ten:', ids[:10])
print('An anchor miss is not proof that every returned recording is irrelevant; all qrels remain unjudged.')""")
    md("""**What to notice:** primary paired differences compare automatic versus reference transcripts under the same method. Rerank-minus-dense isolates the reranking stage; dense-minus-BM25 compares semantic and lexical search. Uncertainty resamples entire target-family query groups, 2,000 times with seed 42. Small curated drafts do not represent real user traffic.

Trace a missed target through its source audio → reference → Whisper transcript → dense candidates → reranked list. A missing candidate cannot be rescued by reranking. A present candidate can be demoted. Good transcription does not guarantee search success. Log potential annotation issues for a new version and complete rerun; do not edit labels after seeing results.

<details><summary>Worked answer</summary>A reranker only reorders the candidates supplied to it. If the target is absent, improving its pairwise scoring cannot recover that target. Increasing candidate depth changes that constraint, but may increase latency.</details>""")
    md("""### 7. Change one thing — development candidate depth

**Predict → run → observe → explain:** increase candidate depth from **10 to 20** on development queries only. Keep models, corpus, prompts and labels fixed. Canonical evaluation files remain unchanged. Compare anchor recovery and measured latency; after label qualification, compare graded relevance too. More candidates do not guarantee improvement.""")
    code(
        "stage('activity')\nshow_json('activity_summary.json')\nshow_csv('activity_results.csv')\nshow_csv('activity_candidates.csv')\nprint('Compare against development scores above; evaluation configuration remains depth 10.')"
    )
    md("""### 8. Reconstruct in a fresh process

Reload the automatic-transcript artifact, verify hashes and dimensions, replay fixed queries, re-embed a document subset and retranscribe three frozen clips. Ranked IDs must match, with declared score tolerance `atol=1e-5, rtol=1e-4`. A mismatch stops the run; tolerances are not silently loosened. This verifies reconstruction beyond reading cached results.""")
    code("stage('reload')\nshow_json('verification.json')")
    md("""### 9. Export evidence and conclude

The automatic-transcript search artifact is separate from evaluation references and qrels. Numeric arrays use safe NumPy formats. The default ZIP excludes source audio and model weights, retaining source IDs, checksums and reacquisition information. Playback uses the validated local corpus, not stale exported Colab paths.

**Optional reflection:** What evidence changed your prediction? Which failure entered at transcription, candidate generation or reranking? What cannot be concluded while relevance labels are unreviewed? Would your conclusion transfer to noisy conversations? Record measured results and limitations rather than declaring a winning model in advance.""")
    code(
        "stage('report')\nshow_json('run_summary.json')\nprint('Evidence bundle:', RESULTS / 'results.zip')\nDOWNLOAD_RESULTS = False\nif DOWNLOAD_RESULTS:\n    from google.colab import files\n    files.download(str(RESULTS / 'results.zip'))"
    )
    md("""### 10. Optional free-form search

Set `RUN_SEARCH=True` and enter a Filipino query. This uses automatic transcripts only. Results have no benchmark score because your new query has no reviewed relevance labels. Listen to the returned recordings and judge the evidence yourself.""")
    code("""RUN_SEARCH = False
SEARCH_QUERY = 'Ano ang epekto ng matinding panahon?'
if RUN_SEARCH:
    command([PY, ROOT / 'audio_runtime.py', '--root', ROOT, '--query', SEARCH_QUERY], 'interactive-search.log')
    response = json.loads((ROOT / 'interactive_search.json').read_text(encoding='utf-8'))
    locations = {r['doc_id']: r for r in corpus_rows}
    for row in response['results']:
        display(Markdown('**' + row['doc_id'] + '** — ' + row['transcript']))
        display(Audio(filename=str(ROOT / locations[row['doc_id']]['path'])))""")
    md("""### 11. Optional bring-your-own recordings

Disabled by default; no microphone or upload dialog opens during Run all. Use recordings you have rights and consent to process. Prepare a local JSON manifest with `rights_confirmed: true`, nonempty `source_notes`, optional `probe_query`, and `records` containing stable `doc_id`, safe relative `path`, and optional nonempty `reference`. Example: `{"rights_confirmed": true, "source_notes": "My consented recording", "records": [{"doc_id": "clip_01", "path": "clip_01.wav"}]}`.

WAV/FLAC inputs are bounded to 1–120 clips, 4 MB each and the same audio eligibility checks. The real BYOD path builds and reloads an independent artifact. WER/CER require references; retrieval metrics require reviewed labels. Missing evidence is reported as **not measurable**, never zero error. BYOD needs separate hosted verification.""")
    code(
        "RUN_BYOD = False\nBYOD_MANIFEST = ROOT / 'my_audio/manifest.json'\nif RUN_BYOD:\n    command([PY, ROOT / 'audio_byod.py', '--root', ROOT, '--manifest', BYOD_MANIFEST], 'byod.log')"
    )
    md("""### AI Assistance Disclosure

Code, instructional text and initial query drafts were developed with generative AI assistance under maintainer direction. Draft annotations are explicitly distinguished from completed human review. The maintainer remains responsible for reviewing implementation, validating results and release decisions. AI assistance is not independent verification, provider endorsement or release approval.

**Current scope:** read-speech engineering preview; no speaker identification, translation, generated answers, weight training or production archive claim. Hosted execution and human relevance qualification remain separate evidence gates.""")
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12.12"},
            "colab": {"name": NOTEBOOK.name, "provenance": []},
            "accelerator": "GPU",
            "dimer": {
                "spec_version": "2.2",
                "profile": "E2E",
                "mode": "GUIDED",
                "status": "Candidate",
                "annotation_status": "engineering_preview",
            },
        },
        "cells": cells,
    }


def materialize(root: Path) -> dict[str, str]:
    """Write the exact carried LF bytes for CPU source checks without executing cells."""
    root.mkdir(parents=True, exist_ok=True)
    files = carried_files()
    for name, text in files.items():
        (root / name).write_bytes(text.encode("utf-8"))
    return files


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    data = (json.dumps(build(), indent=1, ensure_ascii=False) + "\n").encode("utf-8")
    if args.check:
        if not NOTEBOOK.exists() or NOTEBOOK.read_bytes() != data:
            raise SystemExit("Notebook differs from generator")
        print(f"PASS: notebook generator parity ({len(data):,} bytes)")
    else:
        NOTEBOOK.write_bytes(data)
        print(f"Wrote {NOTEBOOK} ({len(data):,} bytes)")


if __name__ == "__main__":
    main()
