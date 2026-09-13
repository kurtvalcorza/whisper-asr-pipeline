"""Per-repository template for tools/build_notebook.py (NOTEBOOK_SPEC 1.1 §3.6 standalone carrier).

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
    "pipeline_class": "WhisperASRPipeline",
    "weights_key": "whisper-large-v3-turbo",
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
        "fixed output contract and the `word_error_rate`, `validate_inputs` and `evaluation_report` helpers. The "
        "default sample is one public LibriSpeech utterance with its reference transcript, fetched at a pinned "
        "dataset revision; its WER is demonstration (plumbing) evidence for one utterance, not a benchmark claim."
    ),
    "learning_objectives": (
        "install the pinned runtime, read what the carried pipeline module guarantees, resolve and digest-verify the "
        "immutable upstream model revision, load one public referenced utterance (or upload your own audio) and "
        "validate it into an input manifest, run the supported task, read the generated transcript correctly, "
        "exercise an optional BYOD path, produce an evaluation report that is `sample-sanity` with `word_error_rate` "
        "only when a reference transcript exists and `not-measurable` otherwise, and export machine-readable outputs "
        "plus provenance."
    ),
    "exclusions": (
        "speaker diarization, speaker identification or any biometric inference, word-level confidence or a "
        "transcript-acceptance threshold, streaming/real-time recognition, text-to-speech, or any training. Whisper "
        "can hallucinate fluent text on silence, music or non-speech audio, and the pipeline does not detect it."
    ),
    "prerequisites": [
        "- **Runtime:** a fresh supported runtime (Google Colab or Jupyter, Python 3.12). The default path runs on CPU (float32) and uses CUDA automatically when available (float16); the pinned `torch==2.6.0` install and the 1.6 GB checkpoint fetch are the largest downloads of the run.",
        "- **Knowledge:** basic Python; what a waveform, a sampling rate and a word error rate are.",
        "- **Data:** the default sample is the first validation utterance of the public `hf-internal-testing/librispeech_asr_dummy` dataset (a few seconds of read English speech with its reference transcript), fetched from the Hugging Face Hub at a pinned dataset revision and decoded with the pinned `soundfile` dependency — no private data is needed. Optional BYOD upload is gated off by default so the sample path can run top-to-bottom without interaction. Expected BYOD input: one audio file readable by the ASR stack (WAV/FLAC/MP3 and similar, mono or stereo); if you know its transcript, paste it into `REFERENCE_TEXT` so the evaluation step can compute `word_error_rate`. Do not upload confidential or restricted data to a hosted notebook environment unless you are authorized to do so. Uploaded inputs remain in the notebook runtime; this pipeline does not send them to a third-party inference API.",
    ],
    "cells": [
        {
            "md": (
                "## 4. Load the public sample or optional BYOD\n\n"
                "The default sample is **public**: the first `validation` row of `hf-internal-testing/librispeech_asr_dummy` "
                "(`clean` config), loaded at the pinned dataset revision `5be91486e11a2d616f4ec5db8d3fd248585ac07a` so the "
                "bytes cannot drift; the stored audio bytes are decoded with the pinned `soundfile` dependency into a "
                "float32 waveform plus sampling rate (the datasets audio feature would decode through torchcodec/FFmpeg, "
                "which this runtime does not pin), stereo is averaged to mono, and the waveform's SHA-256 is printed for "
                "the record. Its reference transcript gives the evaluation step a ground truth. BYOD is optional and "
                "disabled by default; when enabled, upload one audio file and, if you know its transcript, set "
                "`REFERENCE_TEXT` (leave it empty when unknown). Look for a dictionary naming the sample kind, its "
                "duration and digest, and whether a reference exists."
            ),
            "code": (
                "import hashlib\n"
                "import io\n\n"
                "import numpy as np\n"
                "import soundfile as sf\n"
                "from datasets import Audio, load_dataset\n\n"
                "USE_BYOD = False  # @param {{type:\"boolean\"}}\n"
                "REFERENCE_TEXT = ''  # @param {{type:\"string\"}}\n"
                "SAMPLE_DATASET = 'hf-internal-testing/librispeech_asr_dummy'\n"
                "SAMPLE_DATASET_REVISION = '5be91486e11a2d616f4ec5db8d3fd248585ac07a'\n\n"
                "if USE_BYOD:\n"
                "    from google.colab import files\n"
                "    uploaded = files.upload()\n"
                "    sample_name = next(iter(uploaded))\n"
                "    audio_input = sample_name\n"
                "    waveform, sampling_rate = sf.read(io.BytesIO(uploaded[sample_name]), dtype='float32')\n"
                "    if waveform.ndim > 1:\n"
                "        waveform = waveform.mean(axis=1)\n"
                "    reference = REFERENCE_TEXT.strip() or None\n"
                "    sample_kind = 'BYOD'\n"
                "else:\n"
                "    ds = load_dataset(SAMPLE_DATASET, 'clean', split='validation', revision=SAMPLE_DATASET_REVISION)\n"
                "    ds = ds.cast_column('audio', Audio(decode=False))\n"
                "    row = ds[0]\n"
                "    waveform, sampling_rate = sf.read(io.BytesIO(row['audio']['bytes']), dtype='float32')\n"
                "    if waveform.ndim > 1:\n"
                "        waveform = waveform.mean(axis=1)\n"
                "    audio_input = {{'array': waveform, 'sampling_rate': sampling_rate}}\n"
                "    sample_name = f\"{{SAMPLE_DATASET}}:clean:validation[0] ({{row['id']}})\"\n"
                "    reference = row['text']\n"
                "    sample_kind = 'public-sample'\n\n"
                "sample_sha256 = hashlib.sha256(np.ascontiguousarray(waveform, dtype=np.float32).tobytes()).hexdigest()\n"
                "print({{'sample_kind': sample_kind, 'name': sample_name, 'seconds': round(len(waveform) / sampling_rate, 2), 'sampling_rate': sampling_rate, 'waveform_sha256': sample_sha256, 'has_reference': reference is not None}})"
            ),
        },
        {
            "md": (
                "## 5. Validate the input → input manifest\n\n"
                "`validate_inputs` is the pipeline's public validation stage: it applies exactly the checks "
                "`transcribe` applies — the task must be `transcribe` or `translate`, a path must exist, "
                "`chunk_length_s` must be 1..`MAX_CHUNK_LENGTH_S` — and returns an **input manifest** naming the "
                "schema and ceilings, the observed input (kind, samples, sampling rate, duration), the request "
                "(task, language, chunk length) and the verdict. The manifest is written to "
                "`outputs/{stem}_input_manifest.json`. To show what rejection looks like, the cell also validates a "
                "deliberately oversized chunk length and records the pipeline's own error message as a finding. "
                "Inside the transformers pipeline the audio is resampled to 16 kHz and turned into 30-second log-Mel "
                "windows; nothing else is dropped or altered."
            ),
            "code": (
                "import json\n"
                "import os\n\n"
                "os.makedirs('outputs', exist_ok=True)\n"
                "print({{'ceilings': {{'TASKS': list(TASKS), 'MIN_CHUNK_LENGTH_S': MIN_CHUNK_LENGTH_S, 'MAX_CHUNK_LENGTH_S': MAX_CHUNK_LENGTH_S}}}})\n"
                "input_manifest = validate_inputs(audio_input, language='en', task='transcribe', names=[sample_name])\n"
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
                "## 6. Transcribe\n\n"
                "`transcribe` returns the generated `text` (whitespace-stripped, otherwise as decoded), optional "
                "`chunks` when timestamps are requested, the task and language, and the model identity. The text is a "
                "**generated transcript**: greedy autoregressive decoding with the upstream generation config, no "
                "confidence score, no acceptance threshold, and no guarantee against hallucinated words on silence or "
                "noise — a deployment that needs an abstain option must build its own detector on its own labelled "
                "audio. `language='en'` fixes the decoder's language token; leave it `None` to let the model detect "
                "the language. Decoding is deterministic given the same weights, device and library versions; float16 "
                "on CUDA and float32 on CPU can differ in near-tied tokens. Look for the transcript next to the "
                "reference."
            ),
            "code": (
                "result = pipe.transcribe(audio_input, language='en', task='transcribe')\n"
                "print({{'task': result['task'], 'language': result['language'], 'device': result['device'], 'source': result['source']}})\n"
                "print('transcript:', result['text'])\n"
                "print('reference: ', reference)"
            ),
        },
        {
            "md": (
                "## 7. Evaluate → evaluation report\n\n"
                "`evaluation_report` is the pipeline's public evaluation stage and always produces a report. When a "
                "reference transcript exists — the public sample's, or `REFERENCE_TEXT` for BYOD — it carries "
                "`word_error_rate` (the repository's metric helper: case-folded, punctuation removed, curly apostrophes "
                "folded; abbreviations and numbers are **not** normalised, so `Mr.` versus `MISTER` counts as an error) "
                "with the verdict `sample-sanity` — one utterance, no dispersion estimate. Without a reference the "
                "verdict is `not-measurable` and the report states what would make the task measurable: reference "
                "transcripts for several hundred utterances from the deployment domain. The report is written to "
                "`outputs/{stem}_evaluation_report.json`."
            ),
            "code": (
                "report = evaluation_report(result, reference, sample_kind=sample_kind)\n"
                "with open('outputs/{stem}_evaluation_report.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(report, handle, indent=2, ensure_ascii=False)\n"
                "print(json.dumps(report, indent=2))\n"
                "if report['verdict'] == 'not-measurable':\n"
                "    print('No reference transcript was supplied, so word_error_rate is not computed; the transcript above is sanity evidence only.')"
            ),
        },
        {
            "md": (
                "## 8. Export outputs and provenance\n\n"
                "Machine-readable JSON preserves the full transcription result, the evaluation report, the input "
                "manifest, the sample identity and waveform digest, the notebook's source (repository, revision, "
                "embedded module digest, generator), the model identifier, the immutable model revision, the model "
                "licence, and the runtime identity (Python, `torch`, `transformers`, device). The transcript is also "
                "written as a plain-text file. No credentials are recorded."
            ),
            "code": (
                "payload = {{\n"
                "    'prediction': result,\n"
                "    'evaluation_report': report,\n"
                "    'input_manifest': input_manifest,\n"
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
    ],
    "closing": (
        "## Interpretation and limits\n\n"
        "The ASR text is a model-generated transcript with no confidence score; the pipeline ships no threshold and "
        "cannot tell a correct word from a fluent hallucination. `word_error_rate`, when shown, is tied to the single "
        "demonstrated reference after the stated normalisation and must not be generalised to other languages, "
        "speakers, accents, domains, microphones or noise conditions; one utterance is not an error rate. Silence, "
        "music, overlapping speakers, code-switching and heavy accents degrade results in ways the pipeline does not "
        "detect. The pipeline provides no diarization, speaker identity, biometric inference, streaming or training "
        "capability.\n\n"
        "Successful execution proves that the recorded repository revision's pipeline module, carried in this notebook, "
        "can acquire and digest-verify the pinned model, validate the demonstrated input, execute the public pipeline "
        "path, and emit the shown machine-readable outputs in the tested runtime — without the repository being "
        "reachable. It does **not** establish benchmark superiority, deployment calibration, safety for high-consequence "
        "decisions, or production fitness on an unseen domain.\n\n"
        "**Next experiments:** enable `USE_BYOD` with a recording you have transcribed yourself and paste the transcript "
        "into `REFERENCE_TEXT` to see the report switch to `sample-sanity`; set `task='translate'` on non-English audio "
        "and compare; pass `return_timestamps=True` to `pipe.transcribe` to get `chunks` with time offsets; record a "
        "few seconds of silence and observe what the decoder generates.\n\n"
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
