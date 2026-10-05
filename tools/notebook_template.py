"""Per-repository template for tools/build_notebook.py (NOTEBOOK_SPEC 2.2 §4 standalone carrier).

Only the task-specific prose and stage cells live here. Runtime install, the embedded pipeline
module, and the model pin/stage/verify cells are produced by the generator from repository
sources so they cannot drift from the package.
"""
# ruff: noqa: E501  -- markdown prose and code-cell text are kept on single lines for readable rendering

TEMPLATE = {
    "package": "whisper_asr_pipeline",
    "repo_name": "whisper-asr-pipeline",
    "stem": "whisper_asr",
    "notebook_name": "whisper_asr_colab.ipynb",
    "profile": "TASK-INFERENCE",
    "mode": "GUIDED",
    # WSP-M2: the inline pins are pyproject's runtime dependencies plus `datasets` (the default sample loader).
    "pins_file": "tools/inference-pins.txt",
    # NOTEBOOK_SPEC 2.2 §5 (WSP-M1): the pins are installed into an isolated uv environment and every later cell runs in
    # a persistent worker there, so a hosted runtime's preloaded NumPy never forces a restart. The lock is compiled with
    # `uv pip compile tools/inference-pins.txt --python-version 3.12 --python-platform x86_64-manylinux_2_28
    # --generate-hashes --only-binary :all: -o tutorials/requirements-colab.lock.txt`.
    "isolated_runtime": True,
    "infrastructure_labels": True,
    "managed_python": "3.12.12",
    "uv": {
        "version": "0.12.15",
        "url": "https://files.pythonhosted.org/packages/1e/fd/432451d732917c49152a291de3ef171aa6b0f1a22d39780fb2c1f085ca4c/uv-0.12.15-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
        "bytes": 20081404,
        "sha256": "aee9802f46bae436bd91751bb33ddeb379ef1596b5c19df193219d545d244b60",
    },
    "lock": "tutorials/requirements-colab.lock.txt",
    "run_all": (
        "Selecting **Run all** in a fresh supported runtime (CPU is enough; a T4 GPU is faster) builds an isolated Python environment from the "
        "hash-locked pins (the kernel's own packages, such as a preloaded NumPy, are left alone, so no restart is needed; a second Run all reuses it), "
        "stages and digest-verifies the pinned snapshot, loads one public referenced utterance at a pinned dataset revision, validates it into an "
        "input manifest before the model runs, transcribes it through the carried pipeline, writes the evaluation report (word error rate beside "
        "the empty-transcript baseline), exports machine-readable outputs with provenance, and finally runs a short activity on generated silence. "
        "No repository clone, DIMER worker or service, credential, upload dialog, configuration edit or runtime restart is required "
        "(NOTEBOOK_SPEC 2.2 §5). No run of this revision is recorded yet: the recorded Kaggle runs (2026-09-11) executed an earlier, "
        "repository-installing version of this notebook."
    ),
    "byod": (
        "Optional and off by default: set `USE_BYOD = True` in Section 4 and either give a file path in `BYOD_PATH` or upload exactly one "
        "audio file (WAV/FLAC/OGG/MP3 readable by the pinned `soundfile`). It is decoded by the same `soundfile` call as the sample and handed "
        "to the model as a waveform, so it enters the same validation, transcription, evaluation-report and export cells; paste its transcript "
        "into `REFERENCE_TEXT` to have it scored. The file stays inside this runtime."
    ),
    "pipeline_class": "WhisperASRPipeline",
    "weights_key": "whisper-large-v3-turbo",
    # pipeline.py is shared byte for byte with the workshop notebook; the review-fix helpers live in their own module.
    "modules": ["pipeline.py", "tutorial_support.py"],
    "runtime_imports": ["torch", "transformers"],
    "title": "Whisper large-v3-turbo — DIMER ASR tutorial (standalone)",
    "badges": [
        (
            "GitHub",
            "https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white",
            "https://github.com/kurtvalcorza/whisper-asr-pipeline",
        ),
        (
            "Open In Colab",
            "https://colab.research.google.com/assets/colab-badge.svg",
            "https://colab.research.google.com/github/kurtvalcorza/whisper-asr-pipeline/blob/main/tutorials/whisper_asr_colab.ipynb",
        ),
        (
            "Hugging Face",
            "https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-openai%2Fwhisper--large--v3--turbo-ffcc4d?style=flat",
            "https://huggingface.co/openai/whisper-large-v3-turbo",
        ),
        (
            "Upstream",
            "https://img.shields.io/badge/Upstream-openai%2Fwhisper-181717?style=flat&logo=github&logoColor=white",
            "https://github.com/openai/whisper",
        ),
        ("arXiv", "https://img.shields.io/badge/arXiv-2212.04356-b31b1b.svg", "https://arxiv.org/abs/2212.04356"),
    ],
    "capability": "multilingual automatic speech recognition (transcribe or translate-to-English) using the pinned `openai/whisper-large-v3-turbo` weights",
    "intro": (
        "At inference the encoder maps 30-second windows of 128-bin log-Mel features to hidden states and the "
        "4-layer decoder generates text tokens autoregressively; the transformers ASR pipeline handles resampling, "
        "feature extraction and chunking of longer audio, and the carried pipeline module returns the normalised "
        "transcript plus provenance. **No adaptation occurs:** no training, fine-tuning, in-context conditioning, or "
        "preprocessing fitting happens in this notebook — the upstream checkpoint supplies the weights, tokenizer and "
        "feature-extractor configuration, and the carried module adds snapshot verification, the input contract, a "
        "fixed output contract, a BYOD decoder that names the file it refuses (`decode_audio`), and the `word_error_rate`, `validate_inputs` and `evaluation_report` helpers. The "
        "default sample is one public LibriSpeech utterance with its reference transcript, fetched at a pinned "
        "dataset revision; its WER is demonstration (plumbing) evidence for one utterance, not a benchmark claim."
    ),
    "learning_objectives": (
        "install the pinned runtime, read what the carried pipeline module guarantees, resolve and digest-verify the "
        "immutable upstream model revision, load one public referenced utterance (or upload your own audio) and "
        "validate it into an input manifest, run the supported task (transcribe or translate, with or without "
        "timestamps, all from form fields), read the generated transcript correctly, produce an evaluation report that is "
        "`sample-sanity` with `word_error_rate` beside the empty-transcript baseline only when a reference transcript exists "
        "and `not-measurable` otherwise, see Whisper generate text from silence, exercise an optional BYOD path, and export "
        "machine-readable outputs plus provenance."
    ),
    "exclusions": (
        "speaker diarization, speaker identification or any biometric inference, word-level confidence or a "
        "transcript-acceptance threshold, streaming/real-time recognition, text-to-speech, or any training. Whisper "
        "can hallucinate fluent text on silence, music or non-speech audio, and the pipeline does not detect it."
    ),
    "guided_opening": [
        (
            "## How to use this notebook\n\n"
            "**Who this notebook is for.** Learners who can open a hosted notebook (Google Colab, Kaggle or Jupyter), run cells in order and read "
            "short Python, and who want to see how a pretrained speech-recognition model is driven safely and how its transcript is checked "
            "against a reference. No prior experience with Whisper is assumed: each term is explained where it is first needed and again in the "
            "glossary below. A CPU runtime is enough. The **Prerequisites** give the details.\n\n"
            "**Running it.** Choose *Runtime → Run all*. The default path needs no edit, no upload, no account, no token and no runtime restart. "
            "Section 1 builds the isolated environment (the pinned `torch` is the largest download, so it is the slowest step); a second Run all in "
            "the same runtime reuses it. You can also run the notebook one cell at a time with *Shift + Enter*.\n\n"
            "**Where the code runs.** The first two code cells run in the notebook kernel: they build the environment and start one Python process "
            "inside it. Every later cell is sent to that process, so the pinned `torch`, `transformers`, `numpy` and `datasets` are used without "
            "replacing anything the hosted runtime had already loaded. Printed output and errors come back to the notebook as usual, and variables "
            "persist from cell to cell.\n\n"
            "**Two kinds of cell.** *Infrastructure cells* (Sections 1–3: the isolated install and router, the carried pipeline module and the "
            "pinned-model staging) are collapsed and labelled **Infrastructure**; you may run them without studying their implementation. *Learner "
            "cells* (Sections 4–9) are the workflow.\n\n"
            "**Form controls.** The Section 4 cell starts with fields Colab renders as a form: `USE_BYOD`, `BYOD_PATH`, `REFERENCE_TEXT`, `TASK`, "
            "`LANGUAGE` and `RETURN_TIMESTAMPS`; Section 9 has `ACTIVITY_AUDIO`. Leave them at their defaults for the first run: the notes and sample "
            "answers describe the default path. Every experiment in this notebook changes only a form field.\n\n"
            "**Section tags.** Each learner heading carries one tag. **[Concept]** — what the model does and why. **[Evaluation practice]** — how the "
            "evidence is produced and how to read it. **[Engineering]** — reproducibility, provenance and packaging.\n\n"
            "**Predict, then check.** Before each principal result a **Predict before running** prompt asks you to commit to an expectation; after "
            "it, **What to notice** describes normal output and a collapsed **Check your reasoning** block gives a worked answer. Write your own answer "
            "first, then open it. The transcript and WER the answers quote come from the recorded runs of an earlier version of this notebook on the "
            "same pinned model and utterance (Kaggle CPU, 11 September 2026); no run of this revision is recorded yet."
        ),
        (
            "## The task: Input → Model/System → Output\n\n"
            "| Stage | Input | Model / system | Output |\n"
            "|---|---|---|---|\n"
            "| **Validate** | a waveform and its sampling rate, plus the request (task, language, chunk length) | the carried `validate_inputs` | an input manifest: samples, rate, seconds, request, verdict |\n"
            "| **Transcribe** | the waveform | Whisper large-v3-turbo: 128-bin log-Mel features → 32-layer encoder → 4-layer decoder, greedy decoding | `text`, optional `chunks` with timestamps, task, language |\n"
            "| **Evaluate** | the transcript and a reference transcript | `word_error_rate` after case and punctuation normalisation | an evaluation report beside the empty-transcript baseline |\n"
            "| **Export** | everything above | — | JSON + a plain-text transcript with provenance |\n\n"
            "## Roadmap\n\n"
            "| Section | Tag | What happens | What you read |\n"
            "|---|---|---|---|\n"
            "| 1. Install the pinned runtime | [Engineering] | isolated environment built; later cells routed to it | versions, CUDA |\n"
            "| 2. Pipeline code | [Engineering] | the repository's module, carried verbatim | nothing to run by hand |\n"
            "| 3. Pin, stage and verify the model | [Engineering] | snapshot downloaded and digest-checked | the verified files |\n"
            "| 4. Load the sample or BYOD | [Concept] | one referenced utterance (or your file) | duration, rate, digest |\n"
            "| 5. Validate | [Evaluation practice] | input manifest and a rejection probe | the manifest and the finding |\n"
            "| 6. Transcribe | [Concept] | one call through the pipeline | the transcript beside the reference |\n"
            "| 7. Evaluate | [Evaluation practice] | WER beside the empty-transcript baseline | the report and its limits |\n"
            "| 8. Export | [Engineering] | JSON and text outputs with provenance | the file list |\n"
            "| 9. Activity: silence | [Concept] | the model hears seconds of generated silence or noise | what it writes |\n"
            "| Interpretation, Troubleshooting, Conclusion | — | limits, recovery, your notes | when needed |\n\n"
            "**Fast path.** Short on time? Run all, then read Sections 6, 7 and 9 and the conclusion."
        ),
        (
            "<details>\n<summary><strong>Glossary</strong> — open when a term is unfamiliar</summary>\n\n"
            "| Term | Meaning in this notebook |\n"
            "|---|---|\n"
            "| **Waveform / sampling rate** | The audio as a list of amplitude values, and how many values make one second (Whisper works at 16 kHz; the pipeline resamples). |\n"
            "| **Log-Mel features** | A spectrogram on a perceptual frequency scale, in 30-second windows: what the encoder actually reads. |\n"
            "| **Encoder / decoder** | The encoder turns the audio features into hidden states; the decoder writes text one token at a time while attending to them. |\n"
            "| **Decoder language token** | A token at the start of the output that tells the decoder which language to write in; `LANGUAGE = ''` lets the model detect it. |\n"
            "| **Task: transcribe / translate** | Write the speech in its own language, or write an English translation of it. |\n"
            "| **Greedy decoding** | At every step, emit the most likely token. Deterministic, but with no confidence attached. |\n"
            "| **Chunking** | Audio longer than 30 s is cut into windows and the transcripts are joined; `chunk_length_s` sets the window. |\n"
            "| **Timestamps (`chunks`)** | With `RETURN_TIMESTAMPS = True`, segments of the transcript with their start and end times. |\n"
            "| **Word error rate (WER)** | (substitutions + insertions + deletions) / reference words, after lower-casing and removing punctuation. 0 is perfect; it can exceed 1. |\n"
            "| **Normalisation** | What is made equal before counting errors. Here: case and punctuation only, so `Mr.` against `MISTER` and `100` against `one hundred` still count as errors. |\n"
            "| **Empty-transcript baseline** | The WER of writing nothing: 1.0, every reference word deleted. The trivial reference point for any WER. |\n"
            "| **Hallucination** | Fluent text the model writes that was never spoken, typically on silence, music or noise. |\n"
            "| **Isolated environment** | A separate Python built from hash-locked pins, in which every learner cell runs. |\n"
            "| **BYOD** | Bring Your Own Data: the optional switch that runs the same cells on your recording. |\n\n"
            "</details>"
        ),
    ],
    "prerequisites": [
        "- **Runtime:** a fresh **Linux x86_64** runtime (Google Colab, Kaggle or Linux Jupyter). The default path runs on CPU (float32) and uses CUDA automatically when available (float16). The kernel's own Python version does not matter: Section 1 builds a separate environment with CPython 3.12.12 from the hash-locked pins and every later cell runs there. The lock carries the Linux build of `torch==2.6.0` with its CUDA libraries (several GB of disk) and the checkpoint is 1.6 GB; these are the largest downloads. Timing: the earlier, repository-installing version of this notebook took 273.6–328.9 s end to end on a Kaggle CPU (11 September 2026), about 210–260 s of it the install, 35 s the snapshot fetch and load and 22–34 s the transcription; this version has not been timed yet.",
        "- **Knowledge:** basic Python; what a waveform, a sampling rate and a word error rate are. The glossary above covers the rest.",
        "- **Data:** the default sample is the first validation utterance of the public `hf-internal-testing/librispeech_asr_dummy` dataset (a few seconds of read English speech with its reference transcript), fetched from the Hugging Face Hub at a pinned dataset revision with the pinned `datasets==4.4.0` and decoded with the pinned `soundfile` — no private data is needed. Optional BYOD is gated off by default so the sample path runs top-to-bottom without interaction. Expected BYOD input: one audio file `soundfile` can read (WAV/FLAC/OGG/MP3, mono or stereo, any rate), by path (`BYOD_PATH`) or through the upload dialog; if you know its transcript, paste it into `REFERENCE_TEXT` so the evaluation step can compute `word_error_rate`. Do not upload confidential or restricted data to a hosted notebook environment unless you are authorized to do so. Uploaded inputs remain in the notebook runtime; this pipeline does not send them to a third-party inference API.",
        "- **Credentials:** none. The pinned model and the sample are public.",
    ],
    "cells": [
        {
            "md": (
                "## 4. Load the public sample or optional BYOD · [Concept]\n\n"
                "The default sample is **public**: the first `validation` row of `hf-internal-testing/librispeech_asr_dummy` "
                "(`clean` config), loaded with the pinned `datasets` at the pinned dataset revision `5be91486e11a2d616f4ec5db8d3fd248585ac07a` so the "
                "bytes cannot drift. The stored audio bytes are decoded with the pinned `soundfile` dependency into a float32 waveform plus sampling "
                "rate (the datasets audio feature would decode through torchcodec/FFmpeg, which this runtime does not pin), stereo is averaged to "
                "mono, and the waveform's SHA-256 is printed for the record. Its reference transcript gives the evaluation step a ground truth.\n\n"
                "BYOD is optional and disabled by default. Set `USE_BYOD = True` and either put a file path in `BYOD_PATH` (any Jupyter runtime) or "
                "leave it empty and upload **exactly one** audio file in Colab. `decode_audio` reads it with the same `soundfile` call and refuses an "
                "empty or unreadable file with a message that names it. The model always receives a waveform and its rate, never a file name, so your "
                "audio goes through exactly the decoder and resampler the sample does, and no `ffmpeg` is needed. If you know the transcript, set "
                "`REFERENCE_TEXT` (leave it empty when unknown).\n\n"
                "Three more form fields set the request for Sections 5 and 6: `TASK` (`transcribe` writes the speech in its own language, `translate` "
                "writes English), `LANGUAGE` (the decoder language token, `en` by default; leave it empty to let the model detect the language) and "
                "`RETURN_TIMESTAMPS` (add timed segments to the result). Look for a dictionary naming the sample kind, its duration and digest, and "
                "whether a reference exists."
            ),
            "code": (
                "import hashlib\n"
                "import io\n"
                "from pathlib import Path\n\n"
                "import numpy as np\n"
                "import soundfile as sf\n"
                "from datasets import Audio, load_dataset\n\n"
                "USE_BYOD = False  # @param {{type:\"boolean\"}}\n"
                "BYOD_PATH = ''  # @param {{type:\"string\"}}\n"
                "REFERENCE_TEXT = ''  # @param {{type:\"string\"}}\n"
                "TASK = 'transcribe'  # @param ['transcribe', 'translate']\n"
                "LANGUAGE = 'en'  # @param {{type:\"string\"}}\n"
                "RETURN_TIMESTAMPS = False  # @param {{type:\"boolean\"}}\n"
                "SAMPLE_DATASET = 'hf-internal-testing/librispeech_asr_dummy'\n"
                "SAMPLE_DATASET_REVISION = '5be91486e11a2d616f4ec5db8d3fd248585ac07a'\n"
                "language = LANGUAGE.strip() or None\n\n"
                "if USE_BYOD:\n"
                "    if BYOD_PATH:\n"
                "        sample_name, audio_bytes = Path(BYOD_PATH).name, Path(BYOD_PATH).read_bytes()\n"
                "    else:\n"
                "        from google.colab import files\n"
                "        uploaded = files.upload()\n"
                "        if len(uploaded) != 1:\n"
                "            raise RuntimeError(f'Upload exactly one audio file (received {{len(uploaded)}}); or set BYOD_PATH and run this cell again.')\n"
                "        ((sample_name, audio_bytes),) = uploaded.items()\n"
                "    waveform, sampling_rate = decode_audio(sample_name, audio_bytes)\n"
                "    reference = REFERENCE_TEXT.strip() or None\n"
                "    sample_kind = 'BYOD'\n"
                "else:\n"
                "    ds = load_dataset(SAMPLE_DATASET, 'clean', split='validation', revision=SAMPLE_DATASET_REVISION)\n"
                "    ds = ds.cast_column('audio', Audio(decode=False))\n"
                "    row = ds[0]\n"
                "    waveform, sampling_rate = sf.read(io.BytesIO(row['audio']['bytes']), dtype='float32')\n"
                "    if waveform.ndim > 1:\n"
                "        waveform = waveform.mean(axis=1)\n"
                "    sample_name = f\"{{SAMPLE_DATASET}}:clean:validation[0] ({{row['id']}})\"\n"
                "    reference = row['text']\n"
                "    sample_kind = 'public-sample'\n"
                "audio_input = {{'array': waveform, 'sampling_rate': sampling_rate}}\n\n"
                "sample_sha256 = hashlib.sha256(np.ascontiguousarray(waveform, dtype=np.float32).tobytes()).hexdigest()\n"
                "print({{'sample_kind': sample_kind, 'name': sample_name, 'seconds': round(len(waveform) / sampling_rate, 2), 'sampling_rate': sampling_rate, 'waveform_sha256': sample_sha256, 'has_reference': reference is not None, 'task': TASK, 'language': language, 'return_timestamps': RETURN_TIMESTAMPS}})"
            ),
        },
        {
            "md": (
                "## 5. Validate the input → input manifest · [Evaluation practice]\n\n"
                "`validate_inputs` is the pipeline's public validation stage: it applies exactly the checks "
                "`transcribe` applies — the task must be `transcribe` or `translate`, a path must exist, "
                "`chunk_length_s` must be 1..`MAX_CHUNK_LENGTH_S` — and returns an **input manifest** naming the "
                "schema and ceilings, the observed input (kind, samples, sampling rate, duration), the request "
                "(task, language, chunk length) and the verdict. The manifest is written to "
                "`outputs/{stem}_input_manifest.json`. To show what rejection looks like, the cell also validates a "
                "deliberately oversized chunk length and records the pipeline's own error message as a finding. "
                "Inside the transformers pipeline the audio is resampled to 16 kHz and turned into 30-second log-Mel "
                "windows; nothing else is dropped or altered.\n\n"
                "**Predict before running:** the probe asks for a chunk length of `MAX_CHUNK_LENGTH_S + 1` = 31 seconds. Will the pipeline "
                "refuse it, or quietly use 30?"
            ),
            "code": (
                "import json\n"
                "import os\n\n"
                "os.makedirs('outputs', exist_ok=True)\n"
                "print({{'ceilings': {{'TASKS': list(TASKS), 'MIN_CHUNK_LENGTH_S': MIN_CHUNK_LENGTH_S, 'MAX_CHUNK_LENGTH_S': MAX_CHUNK_LENGTH_S}}}})\n"
                "input_manifest = validate_inputs(audio_input, language=language, task=TASK, names=[sample_name])\n"
                "# Demonstrate rejection on a request that breaks a ceiling; the finding is recorded, not swallowed.\n"
                "try:\n"
                "    validate_inputs(audio_input, chunk_length_s=MAX_CHUNK_LENGTH_S + 1)\n"
                "except ValueError as exc:\n"
                "    input_manifest['findings'].append({{'input': 'oversized-chunk-probe', 'verdict': 'rejected', 'message': str(exc)}})\n"
                "with open('outputs/{stem}_input_manifest.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(input_manifest, handle, indent=2, ensure_ascii=False)\n"
                "print(json.dumps(input_manifest, indent=2))"
            ),
        },
        {
            "md": (
                "**What to notice (Section 5):** the input is a `waveform` with its samples, rate and seconds, the verdict is `accepted`, and "
                "`findings` holds one `rejected` entry for the oversized-chunk probe with the pipeline's message.\n\n"
                "<details><summary>Check your reasoning</summary>It is refused. The pipeline never silently changes a request: a value outside its "
                "ceilings raises a `ValueError` naming the rule, and the notebook records that message as a finding rather than hiding it. Because "
                "BYOD audio is also passed as a waveform, its manifest records seconds and rate too.</details>\n\n"
                "## 6. Transcribe · [Concept]\n\n"
                "`transcribe` returns the generated `text` (whitespace-stripped, otherwise as decoded), optional "
                "`chunks` when timestamps are requested, the task and language, and the model identity. The text is a "
                "**generated transcript**: greedy autoregressive decoding with the upstream generation config, no "
                "confidence score, no acceptance threshold, and no guarantee against hallucinated words on silence or "
                "noise — a deployment that needs an abstain option must build its own detector on its own labelled "
                "audio. `LANGUAGE` fixes the decoder's language token; leave it empty to let the model detect "
                "the language. Decoding is deterministic given the same weights, device and library versions; float16 "
                "on CUDA and float32 on CPU can differ in near-tied tokens.\n\n"
                "**Predict before running:** the reference reads `MISTER QUILTER IS THE APOSTLE OF THE MIDDLE CLASSES AND WE ARE GLAD TO WELCOME "
                "HIS GOSPEL` (17 words). Will the transcript match it word for word? If not, which word do you expect to differ, and why?"
            ),
            "code": (
                "result = pipe.transcribe(audio_input, language=language, task=TASK, return_timestamps=RETURN_TIMESTAMPS)\n"
                "print({{'task': result['task'], 'language': result['language'], 'device': result['device'], 'source': result['source']}})\n"
                "print('transcript:', result['text'])\n"
                "print('reference: ', reference)\n"
                "for chunk in result['chunks'] or []:\n"
                "    print(chunk)"
            ),
        },
        {
            "md": (
                "**What to notice (Section 6):** the transcript next to the reference; with `RETURN_TIMESTAMPS = True`, timed segments below it.\n\n"
                "<details><summary>Check your reasoning</summary>The recorded runs of the earlier notebook on the same model and utterance wrote "
                "\"Mr. Quilter is the apostle of the middle classes, and we are glad to welcome his gospel.\" Every word is right; it differs from "
                "the reference in style — case, punctuation and `Mr.` for `MISTER`. Case and punctuation are removed before scoring, `Mr.` is not, "
                "so Section 7 will count exactly one error. A transcript can therefore be correct and still score above zero.</details>\n\n"
                "## 7. Evaluate → evaluation report · [Evaluation practice]\n\n"
                "`evaluation_report` is the pipeline's public evaluation stage and always produces a report. When a "
                "reference transcript exists — the public sample's, or `REFERENCE_TEXT` for BYOD — it carries "
                "`word_error_rate` (the repository's metric helper: case-folded, punctuation removed, curly apostrophes "
                "folded; abbreviations and numbers are **not** normalised, so `Mr.` versus `MISTER` counts as an error) "
                "with the verdict `sample-sanity` — one utterance, no dispersion estimate — and, under `baselines`, the "
                "**empty-transcript baseline**: the WER of writing nothing, 1.0 (every reference word deleted). That is "
                "the reference point a single WER is read against: 0 is a perfect match after normalisation, 1.0 is no "
                "better than silence, and values above 1 mean the model added more words than the reference holds. Without "
                "a reference the verdict is `not-measurable` and the report states what would make the task measurable: "
                "reference transcripts for several hundred utterances from the deployment domain. The report is written to "
                "`outputs/{stem}_evaluation_report.json`.\n\n"
                "**Predict before running:** with one error in 17 words, what WER will the report show?"
            ),
            "code": (
                "report = with_empty_transcript_baseline(evaluation_report(result, reference, sample_kind=sample_kind), reference)\n"
                "with open('outputs/{stem}_evaluation_report.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(report, handle, indent=2, ensure_ascii=False)\n"
                "print(json.dumps(report, indent=2))\n"
                "if report['verdict'] == 'not-measurable':\n"
                "    print('No reference transcript was supplied, so word_error_rate is not computed; the transcript above is sanity evidence only.')"
            ),
        },
        {
            "md": (
                "**What to notice (Section 7):** `word_error_rate` beside the `empty_transcript` baseline of 1.0, and the verdict `sample-sanity`.\n\n"
                "<details><summary>Check your reasoning</summary>1/17 ≈ 0.0588, the figure the earlier runs recorded. Against the 1.0 of an empty "
                "transcript that looks excellent, and here it is: the one \"error\" is a spelling convention. But it is one utterance of clean read "
                "speech, so it says nothing about accents, noise, other languages or your microphones — and a WER near 0.06 on such audio would not "
                "stop the same model from hallucinating on silence (Section 9).</details>\n\n"
                "## 8. Export outputs and provenance · [Engineering]\n\n"
                "Machine-readable JSON preserves the full transcription result, the evaluation report, the input "
                "manifest, the sample identity and waveform digest, the request (task, language, timestamps), the notebook's "
                "source (repository, revision, embedded module digest, generator), the model identifier, the immutable model "
                "revision, the model licence, and the runtime identity (Python, `torch`, `transformers`, device). The transcript "
                "is also written as a plain-text file. No credentials are recorded."
            ),
            "code": (
                "payload = {{\n"
                "    'prediction': result,\n"
                "    'evaluation_report': report,\n"
                "    'input_manifest': input_manifest,\n"
                "    'request': {{'task': TASK, 'language': language, 'return_timestamps': RETURN_TIMESTAMPS}},\n"
                "    'sample': {{'kind': sample_kind, 'name': sample_name, 'seconds': round(len(waveform) / sampling_rate, 3), 'sampling_rate': sampling_rate, 'waveform_sha256': sample_sha256, 'reference': reference}},\n"
                "    'notebook_source': NOTEBOOK_SOURCE,\n"
                "    'repository_revision': NOTEBOOK_SOURCE['repository_revision'],\n"
                "    'model_id': MODEL_ID,\n"
                "    'model_revision': MODEL_REVISION,\n"
                "    'model_license': MODEL_LICENSE,\n"
                "    'runtime': {{\n"
                "        'python': platform.python_version(),\n"
                "        'torch': torch.__version__,\n"
                "        'transformers': transformers.__version__,\n"
                "        'device': pipe.device,\n"
                "    }},\n"
                "}}\n"
                "with open('outputs/{stem}_result.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(payload, handle, indent=2, ensure_ascii=False)\n"
                "with open('outputs/{stem}_transcript.txt', 'w', encoding='utf-8') as handle:\n"
                "    handle.write(result['text'] + '\\n')\n"
                "print(sorted(os.listdir('outputs')))"
            ),
        },
        {
            "md": (
                "## 9. Activity: what does Whisper write for silence? · [Concept]\n\n"
                "**Predict → Change one thing → Run → Observe → Explain.** The cell below generates five seconds of audio in NumPy — "
                "digital silence (all zeros) by default, or quiet random noise with `ACTIVITY_AUDIO = 'noise'` — and transcribes it with the same "
                "request as Section 6. Nobody speaks. It writes nothing to `outputs/`, needs no upload, and runs as part of Run all.\n\n"
                "1. **Predict:** will the transcript be empty? If not, what kind of text do you expect?\n"
                "2. **Run** the cell as it is (silence) and **observe** the transcript.\n"
                "3. **Change one thing:** set `ACTIVITY_AUDIO = 'noise'` and run only this cell again.\n"
                "4. **Observe** the second transcript and compare.\n"
                "5. **Explain:** in one sentence, why can a model with a low WER on speech still produce text here, and what would a deployment "
                "need before it could trust a transcript of an unknown recording?"
            ),
            "code": (
                "ACTIVITY_AUDIO = 'silence'  # @param ['silence', 'noise']\n"
                "ACTIVITY_SECONDS = 5\n\n"
                "if ACTIVITY_AUDIO == 'noise':\n"
                "    activity_waveform = (0.01 * np.random.default_rng(0).standard_normal(16_000 * ACTIVITY_SECONDS)).astype(np.float32)\n"
                "else:\n"
                "    activity_waveform = np.zeros(16_000 * ACTIVITY_SECONDS, dtype=np.float32)\n"
                "activity = pipe.transcribe({{'array': activity_waveform, 'sampling_rate': 16_000}}, language=language, task=TASK)\n"
                "print({{'activity_audio': ACTIVITY_AUDIO, 'seconds': ACTIVITY_SECONDS, 'words_written': len(activity['text'].split())}})\n"
                "print('transcript:', repr(activity['text']))"
            ),
        },
    ],
    "closing": (
        "<details><summary>Check your reasoning (Section 9)</summary>Whisper is a language model conditioned on audio: it always writes the "
        "most likely continuation, and it was trained on subtitles in which silent stretches often carry text such as \"Thank you.\" or "
        "credits. So it may write a short, fluent phrase for audio in which nobody spoke, or nothing at all; which one you see depends on the "
        "audio and the device. WER on speech measures how well it transcribes words that were said; it does not measure whether it invents words "
        "that were not. A deployment needs a separate voice-activity or abstain check, tested on its own silent and noisy recordings.</details>\n\n"
        "## Interpretation and limits\n\n"
        "The ASR text is a model-generated transcript with no confidence score; the pipeline ships no threshold and "
        "cannot tell a correct word from a fluent hallucination, as Section 9 lets you see. `word_error_rate`, when shown, is tied to the single "
        "demonstrated reference after the stated normalisation and is read against the empty-transcript baseline of 1.0; it must not be "
        "generalised to other languages, speakers, accents, domains, microphones or noise conditions; one utterance is not an error rate. Silence, "
        "music, overlapping speakers, code-switching and heavy accents degrade results in ways the pipeline does not "
        "detect. The pipeline provides no diarization, speaker identity, biometric inference, streaming or training "
        "capability.\n\n"
        "Successful execution proves that the recorded repository revision's pipeline module, carried in this notebook, "
        "can acquire and digest-verify the pinned model, validate the demonstrated input, execute the public pipeline "
        "path in an isolated, hash-locked environment, and emit the shown machine-readable outputs in the tested runtime — without the repository "
        "being reachable. It does **not** establish benchmark superiority, deployment calibration, safety for high-consequence "
        "decisions, or production fitness on an unseen domain.\n\n"
        "**Next experiments** (each changes only a form field in Section 4; then choose *Runtime → Run after* from that cell): enable "
        "`USE_BYOD` with a recording you have transcribed yourself and paste the transcript into `REFERENCE_TEXT` to see your own WER beside "
        "the baseline; set `TASK = 'translate'` with `LANGUAGE` empty on non-English audio and compare the English output with what was said; "
        "set `RETURN_TIMESTAMPS = True` to get `chunks` with time offsets; set `LANGUAGE = ''` on the default sample and check which language "
        "the model detects.\n\n"
        "## Troubleshooting\n\n"
        "| Symptom | Likely cause | What to do |\n"
        "|---|---|---|\n"
        "| Section 1 stops with `This notebook needs a Linux x86_64 runtime` | a local Windows or macOS kernel, or an ARM machine | Use Google Colab, Kaggle or a Linux x86_64 Jupyter; the lock holds manylinux x86_64 wheels. |\n"
        "| Section 1 fails while downloading, or `The pinned uv wheel failed its size/SHA-256 check` | a network failure, or an altered download | Run the Section 1 install cell again; a repeated mismatch means the download is being altered — never edit the digest. |\n"
        "| `holds Python …, not 3.12.12` in Section 1 | an older `dimer_isolated_env/` folder from another notebook version | Delete that folder (or start a fresh runtime) and run the install cell again. |\n"
        "| `The isolated environment's Python process exited` | the worker ran out of memory | Restart the session and choose *Runtime → Run all* again. |\n"
        "| A Hub download fails in Section 3 or 4, or `sha256 … != manifest` | a transient Hugging Face failure, or a changed file | Run that cell again; a digest mismatch is never loaded — do not edit the manifest. |\n"
        "| `RuntimeError: Upload exactly one audio file` in Section 4 | the upload dialog was cancelled, or several files were chosen | Run the cell again and choose one file, or set `BYOD_PATH`. |\n"
        "| `…: not an audio file soundfile can read` or `the file is empty` in Section 4 | the file is not audio, is corrupt, or uses a format libsndfile cannot read | Convert it to WAV or FLAC and try again. |\n"
        "| `FileNotFoundError` in Section 4 | `BYOD_PATH` does not point to a file in this runtime | Check the path (relative paths start from the notebook's working directory). |\n"
        "| `word_error_rate` is high on your own audio | spelling conventions (numbers, abbreviations) or a wrong `LANGUAGE` | Compare the transcript with the reference word by word before trusting the number. |\n\n"
        "## Conclusion (your notes)\n\n"
        "Optional. Fill in from your own run, one sentence each:\n\n"
        "1. The transcript of the sample was ___; its WER was ___ against the empty-transcript baseline of 1.0.\n"
        "2. The word(s) counted as errors were ___, and they were / were not real recognition errors because ___.\n"
        "3. On five seconds of silence the model wrote ___; on noise it wrote ___.\n"
        "4. What one referenced utterance can show about this model, and what it cannot: ___.\n"
        "5. What I would need before trusting its transcripts of my own recordings: ___.\n\n"
        "## References\n\n"
        "- Repository README: https://github.com/kurtvalcorza/whisper-asr-pipeline/blob/main/README.md\n"
        "- Repository model card: https://github.com/kurtvalcorza/whisper-asr-pipeline/blob/main/MODEL_CARD.md\n"
        "- Weight provenance: https://github.com/kurtvalcorza/whisper-asr-pipeline/blob/main/docs/WEIGHTS.md\n"
        "- Upstream model: https://huggingface.co/{MODEL_ID}\n"
        "- Upstream code: https://github.com/openai/whisper\n"
        "- Whisper paper: https://arxiv.org/abs/2212.04356\n"
        "- Public sample: https://huggingface.co/datasets/hf-internal-testing/librispeech_asr_dummy"
    ),
}
