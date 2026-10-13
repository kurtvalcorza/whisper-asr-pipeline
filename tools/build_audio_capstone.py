"""Generate the standalone, guided Filipino audio archive capstone."""

# Ruff permits long notebook prose and embedded source strings.
# ruff: noqa: E501
from __future__ import annotations

import argparse
import ast
import base64
import hashlib
import json
import zlib
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
NOTEBOOK = REPO / "tutorials/DIMER_Filipino_Audio_Archive_Search_Capstone.ipynb"
# Colab opens GitHub notebooks through the contents API, which returns no content above 1 MB.
GITHUB_CONTENTS_LIMIT = 1_000_000
# Bulky data (7,200 unjudged relevance cells) is embedded compressed; all source stays plain text.
PACKED = ("qrels.json",)
# Carried strings are written as short implicitly concatenated pieces: one very long notebook
# line makes the Colab editor unresponsive. Python joins the pieces back into identical text.
CARRIER_PIECE = 1000


def carried_literal(value: str | dict[str, str], indent: str = "") -> str:
    """Return a Python literal equal to ``value`` with no line much longer than CARRIER_PIECE."""
    if isinstance(value, dict):
        inner = indent + "    "
        items = [f"{inner}{key!r}: {carried_literal(text, inner)},\n" for key, text in value.items()]
        return "{\n" + "".join(items) + indent + "}"
    pieces = [
        line[start : start + CARRIER_PIECE]
        for line in value.splitlines(keepends=True)
        for start in range(0, len(line), CARRIER_PIECE)
    ]
    if not pieces:
        return repr(value)
    inner = indent + "    "
    return "(\n" + "".join(f"{inner}{piece!r}\n" for piece in pieces) + indent + ")"


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
import base64, zlib
for name, packed in PACKED_FILES.items():
    FILES[name] = zlib.decompress(base64.b64decode(packed)).decode('utf-8')
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

def command(args, log_name, allowed=()):
    """Stream output and preserve the real subprocess exit code; statuses in `allowed` are returned."""
    with (ROOT / log_name).open('w', encoding='utf-8') as log:
        process = subprocess.Popen([str(a) for a in args], cwd=ROOT, env=ENV,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                   encoding='utf-8', errors='replace', bufsize=1)
        for line in process.stdout:
            print(line, end='')
            log.write(line)
            log.flush()
        status = process.wait()
    if status and status not in allowed:
        raise RuntimeError(f'Command failed ({status}); inspect {ROOT / log_name}')
    return status

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

    def add(kind: str, text: str, metadata: dict | None = None) -> None:
        text = text.strip() + "\n"
        cell = {
            "cell_type": kind,
            "id": hashlib.sha256(f"{len(cells)}:{text}".encode()).hexdigest()[:12],
            "metadata": metadata or {},
            "source": text.splitlines(keepends=True),
        }
        if kind == "code":
            ast.parse(text)
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)

    def md(text: str) -> None:
        add("markdown", text)

    def code(text: str, metadata: dict | None = None) -> None:
        add("code", text, metadata)

    md("""# Filipino Audio Archive Search
## How do transcription errors affect what we find?

**Engineering preview · Candidate · DIMER composed-systems CAPSTONE**

**Standard:** DIMER NOTEBOOK_SPEC 2.2 · **Profile:** `E2E` · **Mode:** `GUIDED` · **Kind:** `CAPSTONE` · **Carrier:** standalone.

> **Use notice:** this is an educational engineering preview, not a validated archive-search product. Its scores are anchor-recovery diagnostics on draft queries (pending native-speaker review) with unjudged relevance and must not be quoted as search quality. The optional bring-your-own-recordings path is for recordings you have the rights and consent to process; they stay in this Colab runtime, and you remain responsible for what you upload and export.

#### Problem statement

Archivists, librarians and researchers hold Filipino speech recordings that nobody has transcribed by hand. Automatic transcription makes them searchable, but transcription errors may hide a recording from the person looking for it. A useful outcome is evidence about *where* in the pipeline (transcription, candidate retrieval or reranking) a wanted recording gets lost, plus a reproducible search artifact you can rebuild and query. Scope: 120 read-speech FLEURS clips, three frozen pretrained models, draft queries; no fine-tuning, no speaker identification and no claim about meetings, regional accents or noisy archives.

**Input → System → Output:** Filipino read-speech recordings → Whisper transcripts → lexical or Qwen3 search → ranked, playable recordings with visible evidence.

You will measure transcription errors, compare six search conditions, trace failures through the pipeline, change one development-only setting and export a reproducible search artifact. Basic Python and Colab familiarity is enough; no prior ML experience is assumed.

**Evidence boundary:** the 60 queries are AI-authored drafts. All 7,200 relevance cells are unjudged, including nominated anchors. Default results are **anchor-recovery engineering diagnostics**, not Recall@5, MRR or nDCG benchmark results. Human relevance review is required before qualification. Completion/reflection notes are optional, not required submissions.

**How the queries were made (read before predicting):** this is query version `ai-draft-v2`. Each query was written from a one-line summary of the information its target (anchor) recording carries, never from the sentence itself, as a topic or information need a searcher could ask without knowing the wording. The earlier version `ai-draft-v1` paraphrased each anchor's reference sentence, which made anchor recovery nearly automatic; it stays in the repository (`tools/audio_queries_v1.json`) for comparison. §3 prints this version's lexical-overlap audit, computed from the frozen files in this run, against a declared threshold. **The v2 queries are a draft pending native-speaker review:** their Filipino naturalness, and whether each still identifies its anchor, have not been checked by a competent speaker. Queries, anchors and the development/evaluation split were frozen before any system ran on this version.

Use a **fresh Colab T4** and **Runtime → Run all**. No secret, login, paid API or repository clone is needed. Measured for revision `e4b3fed` (notebook blob `f890d97d`, query version `ai-draft-v1`) and not re-run for this revision: 615 s wall on a fresh Colab T4 via the Colab CLI (2026-10-03), with a peak of 1.61 GiB allocated GPU memory in the transcription stage ([release verification](https://github.com/kurtvalcorza/whisper-asr-pipeline/blob/main/docs/release-verification.md#filipino-audio-archive-search-capstone)). Budget 60 minutes, 20 GiB free disk and ≤12 GiB peak allocated GPU memory; the run checks disk and GPU type before installing anything.""")
    md("""### 1. Orient and predict

A transcript can contain many errors yet preserve the key concept. One wrong name can also make a mostly correct transcript hard to find. **Predict:** which matters more for your search query—overall transcription accuracy or preservation of its useful evidence?

| Method | Reference transcripts | Whisper transcripts |
|---|---|---|
| BM25 keyword matching | R-BM25 | A-BM25 |
| Qwen3 cosine similarity | R-Dense | A-Dense |
| Dense top 10 → Qwen3 reranking | R-Rerank | A-Rerank |

All six conditions use the same 120 document IDs and 60 queries. The 40 development and 80 evaluation recordings contribute 20 and 40 query anchors respectively. All recordings are searchable distractors in both roles. Reference text is an oracle **input condition**, not a guaranteed performance ceiling. The queries no longer repeat their anchors' wording (see above), so predict also: which method and which transcript condition will lose the most anchors, and why?

The pinned models are [Whisper large-v3-turbo](https://huggingface.co/openai/whisper-large-v3-turbo), [Qwen3 Embedding 0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) and [Qwen3 Reranker 0.6B](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B). Immutable revisions and file hashes are carried in `model_manifest.json`. **No fine-tuning occurs:** you construct and evaluate a search system around frozen pretrained models.

Data: [Google FLEURS](https://huggingface.co/datasets/google/fleurs), `fil_ph`, CC BY 4.0; [Conneau et al. (2022)](https://arxiv.org/abs/2205.12446). Read speech does not establish performance on meetings, regional accents or noisy archives. Sentence families are not verified speaker identities; source/pretraining overlap is unknown.""")
    md("""### 2. Infrastructure — prepare an isolated environment

The next cells verify the target runtime, materialize embedded source and hash-bound manifests, and install a locked Python environment. The second cell carries several hundred thousand characters of embedded source and data, so it starts collapsed: select **Show code** (Colab) or the collapsed-input bar (Jupyter) to inspect it. Models run in sequential subprocesses to release GPU memory. A failure is visible and stops execution; do not skip failed cells or accept incomplete output as a successful run.""")
    code(PREFLIGHT)
    files = carried_files()
    packed = {
        name: base64.b64encode(zlib.compress(files.pop(name).encode("utf-8"), 9)).decode("ascii")
        for name in PACKED
    }
    code(
        "# @title Infrastructure: embedded source, manifests and locked install (expand to inspect)\n"
        "# Embedded, inspectable source and manifests; no remote DIMER code import.\nFILES = "
        + carried_literal(files)
        + "\n# Relevance matrix (7,200 unjudged cells), zlib+base64 to keep the notebook openable from\n"
        "# GitHub in Colab; expanded below and verified against source.json like every other file.\n"
        "PACKED_FILES = "
        + carried_literal(packed)
        + "\n"
        + INSTALL,
        # Collapsed in Colab (form view) and Jupyter; the source stays one click away.
        {"cellView": "form", "jupyter": {"source_hidden": True}},
    )
    md("""### 3. Inspect the recordings before modelling

The frozen sample uses one recording per audited source-text family. Eligibility was decided before predictions: decodable finite non-silent mono audio, 2–25 seconds, nonempty reference, and no family crossing roles. Exact source files, hashes, exclusions and realised duration bands are preserved.

**Predict:** listen to three development clips and identify one name, number or concept you would need to retain to find each clip. Do not use evaluation clips to tune settings.""")
    code("stage('prepare')\nshow_json('qualification.json')\npicture('prepare.png')")
    code("""# Frozen query-set audit, computed by the prepare stage from queries.json and the corpus references.
audit = record('query_audit.json')
print(f"Query version {audit['annotation_version']}: {audit['queries']} queries x {audit['documents']} references; "
      f"unit: {audit['unit']}")
print(f"Mean share of a query's tokens found in its anchor's reference: {audit['mean_anchor_overlap']:.3f} "
      f"(maximum {audit['max_anchor_overlap']:.3f}); in each other reference: {audit['mean_other_overlap']:.3f}")
print(f"Anchor is the strictly highest-overlap reference for {audit['anchor_strictly_highest']} queries; "
      f"highest or tied (nonzero) for {audit['anchor_highest_or_tied']}")
if audit.get('declared_threshold'):
    print('Declared threshold:', audit['declared_threshold'], '-> within threshold:', audit['within_threshold'])
print("Low overlap means BM25 cannot simply match the anchor's words; it does not show the query is good "
      "Filipino or that the anchor is its only relevant recording (native-speaker review pending).")""")
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
    md("""#### Worked example by hand: candidates, embeddings and reranking

- **Keyword search (BM25)** scores a document by the query words it shares, giving rare words more weight.
- **An embedding** turns a text into a list of numbers so that texts with similar meaning lie close together. **Cosine similarity** measures how closely two such lists point the same way (1 means the same direction).
- **A reranker** reads the query together with each *supplied* candidate and scores that pair. It can only reorder what it is given.

Five recordings A–E; the nominated anchor is **D**. Dense search ranks them A, B, C, E, D, and suppose the candidate depth is 3.
- Dense alone: D is at rank 5, so **hit@5 = 1** and **reciprocal rank = 1/5 = 0.2**.
- Dense top 3 → reranker: the reranker sees only A, B, C and returns, say, B, A, C. D is absent, so **hit@5 = 0** and **reciprocal rank = 0**: a candidate miss that no reranker can fix.
- With depth 5 the reranker would also see D and might put it first: **hit@5 = 1**, **reciprocal rank = 1.0**.

Hit@5 counts an anchor at rank 1 and at rank 5 the same; reciprocal rank does not. That is why a metric can saturate while another still moves.

**Building an index is not training.** No model weight changes anywhere in this notebook: Whisper, the embedder and the reranker stay frozen. The index is their stored output (transcripts, word statistics and embedding vectors), kept so queries can be answered quickly.""")
    code("stage('index')\nshow_json('index_manifest.json')")
    md("""### 6. Compare the six systems

**Predict:** can a reranker rescue a recording absent from its ten dense candidates?

The experiment lock binds source, annotations, model identities, prompts and candidate depth. Evaluate the untouched query set only after this lock. In this preview, **anchor hit@5** asks whether the nominated draft target appears in the first five results; **anchor reciprocal rank@10** measures where that target appears in the first ten. They do not account for other potentially relevant recordings.

After complete human review, the graded branch reports Recall@5 over all grade-2 references, MRR@10 for the first grade-2 result and nDCG@10 with gains 0/1/3. Grades: **0** unrelated, **1** partially useful, **2** directly answers; every query needs at least one grade-2 recording. Do not rename preview diagnostics to these metrics. The branch switches on when the annotation manifest validates: `benchmark_qualified` and `human_labels_verified` then mean the labels are complete and schema-valid and a human review is *declared* with reviewer metadata. They do not mean the notebook checked the review itself, nor that this query design suits the question; graded metrics would still describe this constructed task.

**Can a transcription effect show up now?** With queries that do not repeat their anchors' wording, a difference between reference and Whisper transcripts *can* appear in these scores; whether it does is decided by the numbers below, not in advance, and no effect is promised.

**Reading a ceiling result.** The next cell prints, from this run's files, which systems reach 1.000 on each metric, and every paired difference with its interval: ASR minus reference for all three methods, rerank minus dense, and dense minus BM25, on anchor hit@5 and on anchor reciprocal rank@10. Saturation is per metric: two systems can both score hit@5 = 1.000 while their reciprocal ranks differ. Where a difference is 0.000 with a zero-width interval, read it as: *no difference was observed on this metric and cohort; this bootstrap interval is degenerate and does not establish equivalence.* It does not show that transcription errors leave search unaffected. All of this concerns anchor recovery; graded relevance, once reviewed, must be read on its own metrics.""")
    code(
        """stage('evaluate')
show_json('retrieval_results.json')
picture('evaluate.png')
import html
retrieval, paired = record('retrieval_results.json'), record('paired_comparisons.json')
comparisons = [(paired['measure'], name, v) for name, v in paired['comparisons'].items()]
comparisons += [(m, name, v) for m, block in paired.get('additional_comparisons', {}).items()
                for name, v in block.items()]
head = ['measure', 'A − B', 'A', 'B', 'difference', '95% interval', 'interval']
body = ''
for measure, name, v in comparisons:
    low, high = v['ci95'][2]
    row = [measure, name.replace('_minus_', ' − '), f"{v['a']:.3f}", f"{v['b']:.3f}", f"{v['difference']:+.3f}",
           f'[{low:+.3f}, {high:+.3f}]', 'zero width' if low == high else f'width {high - low:.3f}']
    body += '<tr>' + ''.join('<td>' + html.escape(c) + '</td>' for c in row) + '</tr>'
display(HTML('<table><tr>' + ''.join('<th>' + h + '</th>' for h in head) + '</tr>' + body + '</table>'))
print('Evaluation queries only; whole target families resampled 2,000 times, seed 42. Exploratory differences.')
print('Read from this run (generated, not written in advance):')
for measure in dict.fromkeys(m for m, _, _ in comparisons):
    top = [s for s, values in retrieval['scores']['test'].items() if values.get(measure) == 1.0]
    print(f'- {measure}: systems at 1.000 on evaluation queries: ' + (', '.join(top) or 'none'))
for measure, name, v in comparisons:
    low, high = v['ci95'][2]
    if low == high:
        observed = ('no difference was observed on this metric and cohort'
                    if v['difference'] == 0 else f'every resampled difference equals {low:+.3f}')
        print(f'- {measure}, {name}: {observed}; this bootstrap interval is degenerate '
              'and does not establish equivalence.')
show_csv('error_trace.csv')
print('error_trace.csv: evaluation queries with three selected diagnostics only. '
      'Complete rankings for every query and system:', RESULTS / 'ranked_runs.json')"""
    )
    code("""draft_queries = json.loads((ROOT / 'queries.json').read_text(encoding='utf-8'))
ranked_runs = record('ranked_runs.json')
test_queries = [q for q in draft_queries if q['role'] == 'test']
dev_queries = [q for q in draft_queries if q['role'] == 'dev']

def anchor_missed(query, system='asr_rerank'):
    return query['anchor_doc_id'] not in [r['doc_id'] for r in ranked_runs[query['query_id']][system][:5]]

SYSTEM_ORDER = ('asr_rerank', 'asr_dense', 'asr_bm25', 'reference_rerank', 'reference_dense', 'reference_bm25')
# First top-five anchor miss: evaluation queries under any system, else development (diagnosis only).
example = missed_by = trace_scope = None
for scope, queries in (('evaluation', test_queries),
                       ('development (diagnosis only, no setting is changed)', dev_queries)):
    for system in SYSTEM_ORDER:
        misses = [q for q in queries if anchor_missed(q, system)]
        if misses:
            example, missed_by, trace_scope = misses[0], system, scope
            break
    if example is not None:
        break
if example is None:
    example = test_queries[0]
anchor = next(r for r in corpus_rows if r['doc_id'] == example['anchor_doc_id'])
display(Markdown(('**Inspect a draft anchor missed from the top five by `' + missed_by + '` (' + trace_scope
                  + ' query):** ' if missed_by else
                  '**No top-five anchor miss under any system in this run; inspect a recovered anchor:** ')
                 + example['text']))
display(Markdown('**Reference:** ' + anchor['reference']))
display(Markdown('**Whisper:** ' + automatic[anchor['doc_id']]))
display(Audio(filename=str(ROOT / anchor['path'])))
for name in ('reference_bm25', 'asr_bm25', 'reference_dense', 'asr_dense', 'reference_rerank', 'asr_rerank'):
    ids = [r['doc_id'] for r in ranked_runs[example['query_id']][name]]
    print(name, 'anchor rank:', ids.index(anchor['doc_id']) + 1 if anchor['doc_id'] in ids else 'absent',
          'top ten:', ids[:10])
print('These ranks locate where the anchor was lost; they do not show that a transcription edit caused it.')
print('An anchor miss is not proof that every returned recording is irrelevant; all qrels remain unjudged.')""")
    code("""# Competing evidence: what outranked the anchor? Automatic transcripts are what search saw.
PLAY_COMPETITORS = False  # set True to hear the top competing recordings
locations = {r['doc_id']: r for r in corpus_rows}

def rank_of(run, doc_id):
    ids = [r['doc_id'] for r in run]
    return ids.index(doc_id) + 1 if doc_id in ids else None

def candidate_table(query):
    runs, anchor_id = ranked_runs[query['query_id']], query['anchor_doc_id']
    dense = runs['asr_dense'][:10]
    rerank = {r['doc_id']: (i + 1, r['score']) for i, r in enumerate(runs['asr_rerank'])}
    rows = []
    for position, row in enumerate(dense, 1):
        rerank_rank, rerank_score = rerank.get(row['doc_id'], (None, None))
        rows.append([position, row['doc_id'] + (' ← anchor' if row['doc_id'] == anchor_id else ''),
                     f"{row['score']:.4f}", rerank_rank, '' if rerank_score is None else f'{rerank_score:.4f}',
                     automatic[row['doc_id']][:110]])
    if rank_of(dense, anchor_id) is None:
        full = rank_of(runs['asr_dense'], anchor_id)
        rows.append([full, anchor_id + ' ← anchor (outside candidates)', '', None, '', automatic[anchor_id][:110]])
    head = ['dense rank', 'doc_id', 'cosine', 'rerank rank', 'reranker score', 'Whisper transcript (excerpt)']
    cell = lambda v: '<td>' + html.escape('' if v is None else str(v)) + '</td>'
    display(HTML('<table><tr>' + ''.join('<th>' + h + '</th>' for h in head) + '</tr>'
                 + ''.join('<tr>' + ''.join(cell(v) for v in r) + '</tr>' for r in rows) + '</table>'))
    print('Scores are raw ranking values of different kinds (cosine vs reranker), not probabilities.')
    if PLAY_COMPETITORS:
        for row in runs['asr_rerank'][:2]:
            if row['doc_id'] != anchor_id:
                display(Markdown('Competitor **' + row['doc_id'] + '**'))
                display(Audio(filename=str(ROOT / locations[row['doc_id']]['path'])))

def stage_observations(queries):
    return {
        'Candidate miss: anchor absent from the 10 ASR dense candidates (reranking cannot recover it)':
            [q for q in queries if rank_of(ranked_runs[q['query_id']]['asr_dense'][:10], q['anchor_doc_id']) is None],
        'Top-five miss after ASR reranking: anchor not in the first five (including anchors whose rank did not change)':
            [q for q in queries if anchor_missed(q, 'asr_rerank')],
        'Anchor rank demotion: ASR reranking placed the anchor lower than ASR dense did (not necessarily a top-five miss)':
            [q for q in queries
             if (d := rank_of(ranked_runs[q['query_id']]['asr_dense'][:10], q['anchor_doc_id'])) is not None
             and rank_of(ranked_runs[q['query_id']]['asr_rerank'], q['anchor_doc_id']) > d],
    }

counts = ', '.join(f'`{s}` {sum(anchor_missed(q, s) for q in test_queries)}/{len(test_queries)}' for s in SYSTEM_ORDER)
display(Markdown('**Top-five anchor misses on evaluation queries, per system:** ' + counts))
for scope, queries in (('evaluation', test_queries), ('development (diagnosis only)', dev_queries)):
    observations = stage_observations(queries)
    for label, cases in observations.items():
        display(Markdown(f'#### {label}: {len(cases)} of {len(queries)} {scope} queries'))
        if not cases:
            print('Not observed in this run.')
            continue
        display(Markdown('**Query:** ' + cases[0]['text'] + ' · **draft anchor:** ' + cases[0]['anchor_doc_id']))
        candidate_table(cases[0])
    if any(observations.values()):
        break
    if scope == 'evaluation':
        print('None of these ASR-rerank observations occurred on evaluation queries in this run (other systems '
              'may still miss; see the counts above). Development queries follow as diagnosis; they were not used '
              'to change any setting.')
print('Competing recordings are unjudged, not negatives: a competitor may also be relevant.')""")
    md("""**What to notice:** primary paired differences compare automatic versus reference transcripts under the same method, for BM25, dense and rerank, on anchor hit@5 and anchor reciprocal rank@10. Rerank-minus-dense isolates the reranking stage; dense-minus-BM25 compares semantic and lexical search. Uncertainty resamples entire target-family query groups, 2,000 times with seed 42. Small curated drafts do not represent real user traffic.

Trace a missed target through its source audio → reference → Whisper transcript → dense candidates → reranked list. Keep four observations apart:
- the anchor is **absent from the dense candidates**, so reranking cannot recover it;
- the anchor is **below the top-five cutoff after reranking** (a rank demotion is not necessarily a miss, and with unjudged competitors not necessarily worse search);
- the **reference and ASR conditions differ** for the same method (for BM25 the whole corpus's word statistics change, not only the anchor's text);
- a **suspected transcription contribution**, which stays a hypothesis until you have listened to the audio and checked which query words the transcript lost.

Good transcription does not guarantee search success. Log potential annotation issues for a new version and complete rerun; do not edit labels after seeing results.

**Which query is traced.** The trace takes the first top-five anchor miss among evaluation queries under any of the six systems and names that system. Only if no evaluation query has one does it trace a development query, labelled as diagnosis: no setting, label or query is changed after seeing it, and the evaluation results stay the reported ones. `error_trace.csv` covers the 40 evaluation queries with three selected diagnostics (ASR dense candidate missing, ASR and reference rerank top-five missing). Complete rankings for every query and system are in `ranked_runs.json`, and per-query anchor scores in `per_query_results.csv`.

<details><summary>Worked answer</summary>A reranker only reorders the candidates supplied to it. If the target is absent, improving its pairwise scoring cannot recover that target. Increasing candidate depth changes that constraint, but may increase latency.</details>""")
    md("""### 7. Change one thing — development candidate depth

**Predict → run → observe → explain:** increase candidate depth from **10 to 20** on development queries only. Keep models, corpus, prompts and labels fixed. Canonical evaluation files remain unchanged. Compare anchor recovery and measured latency; after label qualification, compare graded relevance too. More candidates do not guarantee improvement.

**Same candidates, two depths.** The activity re-embeds the development queries, and re-batched embeddings can reorder near-tied dense candidates. So the second table scores the depth-10 and depth-20 rerankings of the *same* dense candidates from one computation, side by side, and counts how many depth-10 rankings reproduce the canonical development rankings. Compare depth-20 results with the development scores above only when every depth-10 ranking matched; otherwise compare the two depths within that table.

**Fair cost comparison:** the canonical run reranked 60 queries and this activity reranks 20, so their raw durations measure different workloads. The activity therefore times depth 10 and depth 20 on the **same 20 development queries**, in the same process with the reranker already loaded, alternating which depth runs first. Depth changes only reranking, so query embedding and dense search are excluded from both columns.""")
    code(
        """stage('activity')
activity = record('activity_summary.json')
for role, systems in activity['scores'].items():
    print(f"Depth {activity['candidate_depth']}, {role} queries (provisional: {activity['provisional']}):")
    for system, values in systems.items():
        print('  ' + system.ljust(18) + '  '.join(
            f'{k}={v:.3f}' if isinstance(v, float) else f'{k}={v}' for k, v in values.items()))
print('Full record, including per-query timings:', RESULTS / 'activity_summary.json')
show_csv('activity_results.csv')
show_csv('activity_candidates.csv')
same = activity.get('same_candidate_depth_quality', {})
differs = {c: v['queries'] - v['depth10_matches_canonical'] for c, v in same.items()}
if same and not any(differs.values()):
    print('Every depth-10 activity ranking equals the canonical development ranking, '
          'so these depth-20 scores can be compared with the development scores above.')
else:
    print('Depth-10 activity rankings differ from canonical for', differs,
          'queries per condition: compare depth 10 and 20 only in the same-candidate table below.')
print('Evaluation configuration remains depth 10.')"""
    )
    code("""import csv
summary = record('activity_summary.json')
with (RESULTS / 'activity_candidates.csv').open(encoding='utf-8', newline='') as handle:
    candidate_rows = list(csv.DictReader(handle))
rows = []
for condition in ('reference', 'asr'):
    paired = summary['paired_latency'][condition]
    quality = {}
    for depth in (10, 20):
        values = [float(r['value']) for r in candidate_rows
                  if r['condition'] == condition and int(r['candidate_depth']) == depth]
        assert [r['query_id'] for r in candidate_rows if r['condition'] == condition
                and int(r['candidate_depth']) == depth] == paired['query_ids'], 'query cohorts differ'
        quality[depth] = sum(values) / len(values)
    rows.append([condition, paired['queries'], f"{quality[10]:.3f}", f"{quality[20]:.3f}",
                 paired['depth10_pairs'], paired['depth20_pairs'],
                 f"{paired['depth10_mean_seconds_per_query']:.4f}", f"{paired['depth20_mean_seconds_per_query']:.4f}"])
head = ['transcripts', 'same dev queries', 'anchor in candidates @10', 'anchor in candidates @20',
        'reranked pairs @10', 'reranked pairs @20', 'rerank s/query @10', 'rerank s/query @20']
display(HTML('<table><tr>' + ''.join('<th>' + h + '</th>' for h in head) + '</tr>' + ''.join(
    '<tr>' + ''.join('<td>' + html.escape(str(v)) + '</td>' for v in r) + '</tr>' for r in rows) + '</table>'))
print(summary['paired_latency']['conditions'])
print('Per-query timings:', RESULTS / 'activity_latency.csv')
same = summary.get('same_candidate_depth_quality')
if same:
    names = list(same['reference']['depth10'])
    head = ['transcripts', 'depth-10 ranking = canonical'] + [f'{m} @ depth {d}' for m in names for d in (10, 20)]
    rows = [[c, f"{v['depth10_matches_canonical']}/{v['queries']}"]
            + [f"{v[f'depth{d}'][m]:.3f}" for m in names for d in (10, 20)] for c, v in same.items()]
    display(HTML('<table><tr>' + ''.join('<th>' + h + '</th>' for h in head) + '</tr>' + ''.join(
        '<tr>' + ''.join('<td>' + html.escape(str(v)) + '</td>' for v in r) + '</tr>' for r in rows) + '</table>'))
    print('Final ranking quality after reranking the same dense candidates at depth 10 and 20 '
          '(one computation; development anchors only; not a benchmark).')""")
    md("""### 8. Reconstruct in a fresh process

Reload the automatic-transcript artifact, verify hashes and dimensions, replay fixed queries, re-embed a document subset and retranscribe three frozen clips. Ranked IDs must match, with declared score tolerance `atol=1e-5, rtol=1e-4`. A mismatch stops the run; tolerances are not silently loosened. This verifies reconstruction beyond reading cached results.""")
    code("stage('reload')\nshow_json('verification.json')")
    md("""### 9. Export evidence and conclude

The automatic-transcript search artifact is separate from evaluation references and qrels. Numeric arrays use safe NumPy formats. The default ZIP excludes source audio and model weights, retaining source IDs, checksums and reacquisition information. Playback uses the validated local corpus, not stale exported Colab paths.

**Conclude from this run's evidence.** Base your conclusion on what §6 printed for this run: which systems reach 1.000 on each metric, and which paired differences have zero-width intervals. For orientation only, the recorded run of the **previous query version** `ai-draft-v1` (paraphrase queries; revision `e4b3fed`, notebook blob `f890d97d`, Colab CLI T4, 2026-10-03; not re-run for this revision or query version) gave evaluation anchor hit@5 of 1.000 for dense and rerank in both transcript conditions and 0.975 for BM25 in both, so the hit@5 ASR-minus-reference rerank difference was 0.000 with a degenerate interval. Anchor reciprocal rank@10 was not saturated there except for rerank: 0.938 (reference) vs 0.906 (ASR) for BM25 and 0.946 vs 0.963 for dense. Development hit@5 for BM25 was 0.95 (reference) vs 0.90 (ASR). Those numbers belong to the paraphrase queries and are not expected to match this run: the v2 queries are meant to make anchor recovery harder, so compare your §6 output with them only to see how much the query construction mattered.

A trace shows *where* an anchor was lost (absent from the candidates, or below the cutoff after reranking) and whether the reference and ASR conditions differ. A transcription contribution stays a hypothesis until you check the audio and the query words. This version replaces paraphrases with information-need drafts; native-speaker review of those drafts, topic-level relevance judgements over all recordings and a larger distractor pool remain future work. These are anchor-recovery statements: if the reviewed-label branch is ever active, read Recall@5, MRR@10 and nDCG@10 on their own; anchor saturation says nothing about graded relevance.

**Recommendation, with the evidence level of each part:**
- Reuse the exported automatic-transcript search package as a *reproducible engineering artifact* only if the export cell below reports `Fresh-process reload passed: True` for your run. That check covers the scope it lists (the replayed queries, re-embedded documents and retranscribed clips in `verification.json`), not every query or document (engineering evidence).
- Treat the six-condition scores as anchor-recovery diagnostics on draft queries only (preview evidence, unreviewed labels); do not quote them as search quality or as the effect of transcription errors.
- Before choosing a method for a real archive, obtain native-speaker-reviewed queries and human relevance judgements (not yet available; no evidence either way).

**Optional reflection:** What evidence changed your prediction? Where was each missed anchor lost (candidates or cutoff), and what would you check before attributing it to transcription? What cannot be concluded while relevance labels are unreviewed? Would your conclusion transfer to noisy conversations? Record measured results and limitations rather than declaring a winning model in advance.""")
    code(
        """stage('report')
run_summary = record('run_summary.json')
for key in ('status', 'annotation_qualified', 'scope', 'mode', 'warning'):
    print(f'{key}: {run_summary[key]}')
check = run_summary['verification']
print(f"Fresh-process reload passed: {check['passed']} (tested scope: {check.get('queries_replayed')} queries "
      f"replayed, {check.get('documents_reembedded')} documents re-embedded, "
      f"{check.get('clips_retranscribed')} clips retranscribed)")
for name, receipt in run_summary['stages'].items():
    peak = receipt.get('peak_allocated_gpu_bytes')
    print(f"  {name}: completed={receipt['completed']}, {receipt['seconds']:.1f} s, peak allocated GPU "
          + ('none' if peak is None else f'{peak / 1024**3:.2f} GiB'))
print('Stage identities and output hashes:', RESULTS / 'run_summary.json')
print('Evidence bundle:', RESULTS / 'results.zip')
DOWNLOAD_RESULTS = False
if DOWNLOAD_RESULTS:
    from google.colab import files
    files.download(str(RESULTS / 'results.zip'))"""
    )
    md("""### 10. Optional free-form search

Set `RUN_SEARCH=True` and enter a Filipino query. This uses automatic transcripts only. Results have no benchmark score because your new query has no reviewed relevance labels. Listen to the returned recordings and judge the evidence yourself.

The same search view serves the default archive and a BYOD archive (§11): each search package is verified (hashes, model identity and settings) before it is queried, and a recording plays only when its local file matches the SHA256 recorded in the package.""")
    code("""def show_search(query, index_dir=None, audio_base=ROOT):
    \"\"\"Search a verified package and show top-five transcript evidence with hash-checked playback.\"\"\"
    args = [PY, ROOT / 'audio_runtime.py', '--root', ROOT, '--query', query]
    if index_dir is not None:
        args += ['--index', index_dir]
    command(args, 'interactive-search.log')
    response = json.loads((ROOT / 'interactive_search.json').read_text(encoding='utf-8'))
    display(Markdown(f"**Query:** {response['query']} · **archive:** `{response['source']}` · "
                     'unscored: no reviewed relevance labels exist for a new query'))
    for rank, row in enumerate(response['results'], 1):
        display(Markdown(f"**{rank}. {row['doc_id']}** (reranker score {row['score']:.4f}) — {row['transcript']}"))
        audio_path = (Path(audio_base) / row['audio_path']).resolve()
        if audio_path.is_file() and hashlib.sha256(audio_path.read_bytes()).hexdigest() == row['audio_sha256']:
            display(Audio(filename=str(audio_path)))
        else:
            print('   audio unavailable or changed; supply/reacquire it via audio_acquisition.json')

RUN_SEARCH = False
SEARCH_QUERY = 'Ano ang epekto ng matinding panahon?'
if RUN_SEARCH:
    show_search(SEARCH_QUERY)""")
    md("""### 11. Optional bring-your-own recordings

Disabled by default; no microphone or upload dialog opens during Run all. Use recordings you have rights and consent to process. Prepare a local JSON manifest with `rights_confirmed: true`, nonempty `source_notes`, optional `probe_query`, and `records` containing stable `doc_id`, safe relative `path`, and optional nonempty `reference`. Example: `{"rights_confirmed": true, "source_notes": "My consented recording", "records": [{"doc_id": "clip_01", "path": "clip_01.wav"}]}`.

**Input contract** (checked before any model loads): WAV or FLAC at any sample rate, mono or stereo, 2–25 seconds and at most 16 MB per clip; 1–120 clips; each `path` relative to the manifest's folder. The models need **16 kHz mono**, so the BYOD build converts other inputs itself: it averages the channels, resamples to 16 kHz with SciPy's polyphase resampler (`scipy.signal.resample_poly`, pinned in the locked environment) and stores 16-bit PCM. `audio_acquisition.json` records both identities (original and converted SHA-256, sample rate and channels), and the fresh-process reconstruction check replays the conversion and must reproduce the converted hash. The same eligibility checks as the default sample then run on the converted audio. Your original file is never modified. Split anything longer than 25 seconds into shorter clips; other formats (MP3, M4A) must be re-exported as WAV or FLAC, for example `ffmpeg -i my_recording.m4a clip_01.wav` (run in a new cell prefixed with `!`).

**Getting files into Colab:** after Run all has created `/content/dimer_fil_audio_preview/`, open the **Files** pane (folder icon), create the folder `dimer_fil_audio_preview/my_audio/`, upload `manifest.json` and your clips into it, set `RUN_BYOD = True` and run the next cell. Uploaded files disappear when the runtime is recycled.

**If an input is refused**, the cell prints one line naming the clip (`doc_id` and path), the rule it broke and how to fix it, builds nothing and clears the active notebook BYOD handle, so the query cell cannot search an earlier archive by mistake (earlier bundles under `byod/` and `byod/latest.json` stay on disk). Fix the file or manifest and re-run the cell. Other failures still stop with an error.

The BYOD build transcribes and embeds your recordings once, then exports a search package in **the same format as the default archive**, with the same `search_archive.py` consumer. A fresh process then re-embeds, replays a probe query and retranscribes up to three clips against your original audio.

Each build gets its own folder under `byod/`; the default index is never touched:
- `search/` and `search_index.zip`: the portable automatic-transcript package (transcripts, embeddings, audio identities, consumer, lock, `RECONSTRUCT.md`). It contains no references or evaluation records.
- `evaluation.json`: kept separate. WER/CER require references; retrieval metrics require reviewed labels. Missing evidence is reported as **not measurable**, never zero error.
- `verification.json`: the fresh-process reconstruction check.

Hosted evidence so far: the BYOD build, re-query and clean-directory reuse passed on a Kaggle T4 at revision `5664f61` using public FLEURS 16 kHz clips as stand-ins (not re-run for this revision). A BYOD run on your own recordings and a hosted run with a refused input have not been recorded yet.""")
    code("""RUN_BYOD = False
BYOD_MANIFEST = ROOT / 'my_audio/manifest.json'
DOWNLOAD_BYOD = False
BYOD_REFUSED = 3  # audio_byod.py exit status for a refused input; its one-line reason is printed above
if RUN_BYOD:
    globals().pop('BYOD', None)  # clear the active handle; earlier bundles and byod/latest.json stay on disk
    BYOD_STATUS = command([PY, ROOT / 'audio_byod.py', '--root', ROOT, '--manifest', BYOD_MANIFEST], 'byod.log',
                          allowed=(BYOD_REFUSED,))
if not RUN_BYOD:
    print('BYOD disabled (RUN_BYOD = False).')
elif BYOD_STATUS == BYOD_REFUSED:
    print('Nothing was built. Fix the input named above, then re-run this cell.')
else:
    BYOD = json.loads((ROOT / 'byod/latest.json').read_text(encoding='utf-8'))
    evaluation = json.loads(Path(BYOD['evaluation']).read_text(encoding='utf-8'))
    display(Markdown(
        f"**BYOD archive built:** {evaluation['recordings']} recordings; references supplied for "
        f"{evaluation['reference_count']} of {evaluation['recordings']}. ASR: `{evaluation['asr_status']}`; "
        f"retrieval: `{evaluation['retrieval_status']}`."))
    if evaluation['asr']:
        print(f"WER {evaluation['asr']['wer']:.3f}, CER {evaluation['asr']['cer']:.3f} "
              f"on the {evaluation['reference_count']} referenced recordings only")
    for label, key in (('Search package', 'search_dir'), ('Portable ZIP', 'search_index_zip'),
                       ('Evaluation (separate)', 'evaluation'), ('Reconstruction check', 'verification')):
        print(f'{label}: {BYOD[key]}')
    if DOWNLOAD_BYOD:
        from google.colab import files
        files.download(BYOD['search_index_zip'])""")
    md("""**Query your archive.** Change `BYOD_QUERY` and re-run only the next cell as often as you like: it searches the saved BYOD package without re-transcribing, and plays only your own recordings. To reuse the archive in a later session, keep `search_index.zip` and follow its `RECONSTRUCT.md`, or upload it with your audio and point `show_search` at the extracted folder.""")
    code("""BYOD_QUERY = ''
if 'BYOD' in globals() and BYOD_QUERY.strip():
    show_search(BYOD_QUERY, BYOD['search_dir'], Path(BYOD['audio_base']))
elif 'BYOD' in globals():
    print('Enter a BYOD_QUERY and re-run this cell to search your archive.')
else:
    print('No BYOD archive in this session; enable RUN_BYOD above first.')""")
    md("""### Troubleshooting

| Symptom | What to do |
|---|---|
| `This candidate targets a Colab T4` or `nvidia-smi` not found | Runtime → Change runtime type → **T4 GPU**, then Runtime → Run all. CPU, L4 and A100 runtimes are outside the verified runtime and are refused. |
| `At least 20 GiB free disk required` | Start a fresh runtime (Runtime → Disconnect and delete runtime) so earlier files are gone, then Run all. |
| A download times out, or an `assert` on a size or SHA-256 fails | Re-run the cell once (a network hiccup is the usual cause). Never edit or bypass a hash: a persistent mismatch means the pinned source changed, so stop and report it. |
| `Command failed (N); inspect …/<stage>.log` | Open that log in the Files pane (under `/content/dimer_fil_audio_preview/`). Stages are sequential; re-run from the failed stage's cell after fixing the cause, or Run all on a fresh runtime. |
| `reload` reports a mismatch | Do not loosen the tolerances. Run all again on a fresh T4; a repeat mismatch is a reproducibility failure to report. |
| BYOD prints `BYOD input refused: …` | The line names the clip, the rule and the fix (a missing file, a clip outside 2–25 s or over 16 MB, an unreadable or silent recording; sample rate and channels are converted, not refused). Fix the file or manifest and re-run the BYOD cell only. |
| BYOD manifest not found | Upload `manifest.json` to `/content/dimer_fil_audio_preview/my_audio/` or set `BYOD_MANIFEST` to its path. |""")
    md("""### AI Assistance Disclosure

Code, instructional text and initial query drafts were developed with generative AI assistance under maintainer direction. Draft annotations are explicitly distinguished from completed human review. The maintainer remains responsible for reviewing implementation, validating results and release decisions. AI assistance is not independent verification, provider endorsement or release approval.

**Current scope:** read-speech engineering preview; no speaker identification, translation, generated answers, weight training or production archive claim. Open evidence gates: native-speaker review of the `ai-draft-v2` queries, human relevance review (0 of 7,200 cells judged), a hosted run of this revision and query version, a BYOD run on the maintainer's own recordings and a hosted run with a refused BYOD input. Default-path hosted runs are recorded in the repository's `docs/release-verification.md`.

### References

1. A. Radford, J. W. Kim, T. Xu, G. Brockman, C. McLeavey and I. Sutskever, "Robust speech recognition via large-scale weak supervision," in *Proc. 40th Int. Conf. Machine Learning (ICML)*, 2023. arXiv:2212.04356. Model: [openai/whisper-large-v3-turbo](https://huggingface.co/openai/whisper-large-v3-turbo).
2. Y. Zhang *et al.*, "Qwen3 Embedding: Advancing text embedding and reranking through foundation models," arXiv:2506.05176, 2025. Models: [Qwen/Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B), [Qwen/Qwen3-Reranker-0.6B](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B).
3. A. Conneau *et al.*, "FLEURS: Few-shot learning evaluation of universal representations of speech," in *Proc. IEEE Spoken Language Technology Workshop (SLT)*, 2022. arXiv:2205.12446. Dataset: [google/fleurs](https://huggingface.co/datasets/google/fleurs), `fil_ph`, CC BY 4.0.
4. S. Robertson and H. Zaragoza, "The probabilistic relevance framework: BM25 and beyond," *Foundations and Trends in Information Retrieval*, vol. 3, no. 4, pp. 333–389, 2009.""")
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12.12"},
            "colab": {"name": NOTEBOOK.name, "provenance": []},
            "accelerator": "GPU",
            "dimer": {
                "notebook_spec": "2.2",
                "notebook_profile": "E2E",
                "notebook_mode": "GUIDED",
                "notebook_kind": "CAPSTONE",
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
