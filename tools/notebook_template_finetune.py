"""E2E template for tools/build_notebook.py (NOTEBOOK_SPEC 2.0 §4 standalone carrier): LoRA fine-tuning.

Only the task-specific prose and stage cells live here. Runtime install (pins from
tools/finetune-pins.txt), the embedded pipeline module, and the model pin/stage/verify cells are
produced by the generator from repository sources so they cannot drift from the package.
"""
# ruff: noqa: E501  -- markdown prose and code-cell text are kept on single lines for readable rendering

TEMPLATE = {
    "package": "whisper_asr_pipeline",
    "repo_name": "whisper-asr-pipeline",
    "stem": "whisper_asr_finetune",
    "notebook_name": "whisper_asr_finetune_colab.ipynb",
    "profile": "E2E",
    "mode": "GUIDED",
    "pipeline_class": "WhisperASRPipeline",
    "weights_key": "whisper-large-v3-turbo",
    "pins_file": "tools/finetune-pins.txt",
    "runtime_imports": ["torch", "transformers", "peft"],
    "title": "Whisper large-v3-turbo — DIMER LoRA fine-tuning tutorial (standalone)",
    "badges": [
        (
            "GitHub",
            "https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white",
            "https://github.com/kurtvalcorza/whisper-asr-pipeline",
        ),
        (
            "Open In Colab",
            "https://colab.research.google.com/assets/colab-badge.svg",
            "https://colab.research.google.com/github/kurtvalcorza/whisper-asr-pipeline/blob/main/tutorials/whisper_asr_finetune_colab.ipynb",
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
    "capability": "parameter-efficient (LoRA) domain adaptation of the pinned `openai/whisper-large-v3-turbo` weights for automatic speech recognition, evaluated by corpus word error rate before and after adaptation and after a fresh reload of the exported adapter bundle",
    "run_all": (
        "This notebook needs a 16 GiB-class CUDA GPU (Colab: Runtime > Change runtime type > T4 GPU; Kaggle P100/T4 also works) — the default configuration was measured at 4.5 GiB peak GPU memory and about 27 minutes on a Kaggle P100 (RUN11/RUN12; on CPU it runs but takes hours, so reduce `TRAIN_CLIPS`, `EVAL_CLIPS` and `EPOCHS` first). Once that runtime is selected, **Run all** installs the pinned dependencies, stages and digest-verifies the pinned snapshot, downloads one locale of the public `PolyAI/minds14` corpus at a pinned dataset revision and decodes/resamples it with the pinned `soundfile`/`torchaudio`, draws a seeded train/held-out split and validates every clip into an input manifest, records the zero-shot **baseline** corpus WER through the carried pipeline, attaches LoRA adapters to the verified base weights and trains them for two bounded epochs with AdamW under mixed precision, evaluates the adapted model on the held-out split through the same pipeline and writes the evaluation report, exports the manifested adapter bundle, and reloads it against the verified base snapshot with a weight-level merge check and a transcript agreement check. No repository clone, DIMER worker or service, credential, upload dialog or configuration edit is required (NOTEBOOK_SPEC 2.0 §5)."
    ),
    "byod": (
        "After the sample workflow completes, set `USE_BYOD = True` in Section 4 and re-run from that cell to upload a `transcripts.csv` (`file,text` columns) plus the referenced audio files; they enter the same decoding, validation, seeded split, baseline, LoRA training, evaluation, export and fresh-reload cells as the public sample (DAT14). Expected format, the 30-second clip ceiling and the privacy guidance are stated in the Prerequisites and in Section 4; uploads stay inside this runtime. BYOD is optional and never part of the default path."
    ),
    "intro": (
        "**What is trained and what is not.** LoRA keeps every original weight matrix $W_0$ frozen and learns a "
        "low-rank correction $W = W_0 + \\frac{\\alpha}{r} B A$; $B$ starts at zero, so at step 0 the adapted model "
        "*is* the base model. Adapters go on the query and value projections of every attention block in both the "
        "32-layer encoder and the 4-layer decoder (the standard Whisper recipe); with rank 32 that is under 1% of "
        "the 809M parameters. What the upstream checkpoint supplies is the weights, tokenizer and feature-extractor "
        "configuration; what the carried pipeline module adds is snapshot verification, the input contract, the "
        "`corpus_word_error_rate`, `validate_inputs`, `export_adapter_bundle`, `verify_adapter_merge` and "
        "`adaptation_report` helpers, and the single decoding path (`WhisperASRPipeline`) that the baseline, the "
        "in-memory adapted model and the reloaded bundle all go through, so their transcripts are comparable. "
        "**Training loss and validation loss are optimisation evidence only** (FT7); the metric that matters is the "
        "corpus WER on the held-out split, and it is tutorial evidence for one small split of one locale, not a "
        "benchmark."
    ),
    "learning_objectives": (
        "install the pinned runtime, read what the carried pipeline module guarantees, resolve and digest-verify the "
        "immutable upstream model revision, load a public labelled speech corpus at a pinned revision (or upload your "
        "own) and validate it into an input manifest, record a zero-shot baseline through the pipeline, attach LoRA "
        "adapters and train them with a plain PyTorch loop whose every knob is explicit, evaluate the adapted model on "
        "the held-out split and read the evaluation report correctly, export a manifested adapter bundle, prove that a "
        "fresh load applies the saved adapter to the base weights, and export machine-readable outputs plus provenance."
    ),
    "exclusions": (
        "full-parameter fine-tuning, encoder-only or decoder-only adaptation, learning-rate schedules, speaker "
        "diarization or identification, character-level metrics for languages without whitespace-delimited words "
        "(`zh-CN`, `ko-KR`), a forgetting check outside the adaptation distribution, or any calibrated "
        "transcript-confidence threshold. Whisper can hallucinate fluent text on silence or music, and the pipeline "
        "does not detect it."
    ),
    "prerequisites": [
        "- **Runtime:** a fresh supported runtime with a 16 GiB-class CUDA GPU (Google Colab **T4** or Kaggle P100/T4; Python 3.12 on Colab). Measured on a Kaggle P100 with the default configuration by the previous, clone-based revision of this notebook: 4.5 GiB peak GPU memory and about 27 minutes end to end, of which about 18 minutes is training (100 optimizer steps) and 4 minutes is the pinned install. CPU execution is supported only as a reduced smoke test: lower `TRAIN_CLIPS`, `EVAL_CLIPS` and `EPOCHS` first, or the training cell will take hours.",
        "- **Knowledge:** basic Python and PyTorch; what a waveform, a sampling rate, a log-mel spectrogram, teacher forcing and a word error rate are; the idea of a low-rank adapter.",
        "- **Data:** the default sample is one locale of the public `PolyAI/minds14` banking-intent corpus (CC-BY-4.0), fetched from the Hugging Face Hub at the pinned dataset revision `40ce77cb32a384e4d50a568e1ec39ac804019d33` and decoded with the pinned `soundfile`/`torchaudio` — no private data is needed. Optional BYOD upload is gated off by default so the sample path can run top-to-bottom without interaction. Expected BYOD input: a `transcripts.csv` with `file,text` columns plus the referenced audio files (WAV/FLAC/OGG readable by `soundfile`, any rate, mono or stereo, each at most 30 seconds), all selected in the same upload dialog; set `BYOD_LANGUAGE` to the ISO code Whisper should decode in. Do not upload confidential or restricted data to a hosted notebook environment unless you are authorized to do so. Uploaded inputs remain in the notebook runtime; this pipeline does not send them to a third-party inference API. An adapter trained on the default sample inherits CC-BY-4.0 attribution obligations.",
    ],
    "cells": [
        {
            "md": (
                "## 4. Load a public labelled speech set or optional BYOD\n\n"
                "The default path loads one locale of the public `PolyAI/minds14` banking-intent corpus (CC-BY-4.0) at the "
                "pinned dataset revision `40ce77cb32a384e4d50a568e1ec39ac804019d33` so the bytes cannot drift: short "
                "telephone-quality utterances stored at **8 kHz**, each with a cased, punctuated transcription. Whisper "
                "expects 16 kHz input, so every clip is decoded with the pinned `soundfile` and resampled with the pinned "
                "`torchaudio` — the same resampler the transformers ASR pipeline applies at inference time, so training and "
                "inference see identical audio. A seeded shuffle then takes `TRAIN_CLIPS` clips for adaptation and a disjoint "
                "`EVAL_CLIPS` for the held-out baseline/adapted comparison (SPL1; random splitting assumes the utterances are "
                "independent, SPL3); the split is recorded as a digest in the exported provenance.\n\n"
                "The locale list is restricted to languages whose transcripts are whitespace-delimited: the package's word "
                "error rate splits on whitespace, so it is not a meaningful metric for the corpus's `zh-CN` and `ko-KR` "
                "locales. BYOD is optional and disabled by default; expected input is a `transcripts.csv` with `file,text` "
                "columns plus the referenced audio files, all selected in the same upload dialog, and `BYOD_LANGUAGE` names "
                "the decoding language. Look for a dictionary naming the corpus, the language, the clip counts, the split "
                "digest and the clip-length statistics."
            ),
            "code": (
                "import csv\n"
                "import hashlib\n"
                "import io\n"
                "import random\n\n"
                "import soundfile as sf\n"
                "import torchaudio\n"
                "from datasets import Audio, load_dataset\n\n"
                "LOCALE = 'en-US'  # @param ['en-US', 'en-GB', 'en-AU', 'de-DE', 'fr-FR', 'es-ES', 'it-IT', 'nl-NL', 'pl-PL', 'pt-PT', 'ru-RU', 'cs-CZ']\n"
                "TRAIN_CLIPS = 400  # @param {{type:\"integer\"}}\n"
                "EVAL_CLIPS = 100  # @param {{type:\"integer\"}}\n"
                "SEED = 0  # @param {{type:\"integer\"}}\n"
                "USE_BYOD = False  # @param {{type:\"boolean\"}}\n"
                "BYOD_LANGUAGE = 'en'  # @param {{type:\"string\"}}\n"
                "SAMPLE_DATASET = 'PolyAI/minds14'\n"
                "SAMPLE_DATASET_REVISION = '40ce77cb32a384e4d50a568e1ec39ac804019d33'\n"
                "TARGET_RATE = 16_000\n\n\n"
                "def decode_clip(raw_bytes):\n"
                "    # Decode with the pinned soundfile, then resample with the pinned torchaudio: the datasets\n"
                "    # audio feature would decode through torchcodec/FFmpeg, which this runtime does not pin.\n"
                "    waveform, rate = sf.read(io.BytesIO(raw_bytes), dtype='float32')\n"
                "    if waveform.ndim > 1:\n"
                "        waveform = waveform.mean(axis=1)\n"
                "    if rate != TARGET_RATE:\n"
                "        waveform = torchaudio.functional.resample(torch.from_numpy(waveform), rate, TARGET_RATE).numpy()\n"
                "    return waveform\n\n\n"
                "if USE_BYOD:\n"
                "    from google.colab import files\n"
                "    uploaded = files.upload()\n"
                "    manifest_name = next(name for name in uploaded if name.lower().endswith('.csv'))\n"
                "    rows = list(csv.DictReader(io.StringIO(uploaded[manifest_name].decode('utf-8-sig'))))\n"
                "    clips = [{{'id': row['file'], 'audio': decode_clip(uploaded[row['file']]), 'text': row['text'].strip()}} for row in rows]\n"
                "    LANGUAGE = BYOD_LANGUAGE.strip().lower()\n"
                "    DATASET = {{'name': 'BYOD', 'config': manifest_name, 'revision': None, 'license': None}}\n"
                "    sample_kind = 'BYOD'\n"
                "else:\n"
                "    ds = load_dataset(SAMPLE_DATASET, LOCALE, split='train', revision=SAMPLE_DATASET_REVISION)\n"
                "    ds = ds.cast_column('audio', Audio(decode=False))\n"
                "    clips = [{{'id': row['path'], 'audio': decode_clip(row['audio']['bytes']), 'text': row['transcription'].strip()}} for row in ds]\n"
                "    LANGUAGE = LOCALE.split('-')[0]\n"
                "    DATASET = {{'name': SAMPLE_DATASET, 'config': LOCALE, 'revision': SAMPLE_DATASET_REVISION, 'license': 'CC-BY-4.0'}}\n"
                "    sample_kind = 'public-sample'\n\n"
                "clips = [c for c in clips if len(c['audio']) / TARGET_RATE <= MAX_CHUNK_LENGTH_S]\n"
                "if TRAIN_CLIPS < 1 or EVAL_CLIPS < 1 or TRAIN_CLIPS + EVAL_CLIPS > len(clips):\n"
                "    raise ValueError(f'TRAIN_CLIPS + EVAL_CLIPS must fit in the {{len(clips)}} available clips')\n"
                "order = list(range(len(clips)))\n"
                "random.Random(SEED).shuffle(order)\n"
                "train_clips = [clips[i] for i in order[:TRAIN_CLIPS]]\n"
                "eval_clips = [clips[i] for i in order[TRAIN_CLIPS:TRAIN_CLIPS + EVAL_CLIPS]]\n"
                "SPLIT_DIGEST = hashlib.sha256('\\n'.join(c['id'] + '\\t' + c['text'] for c in train_clips + eval_clips).encode('utf-8')).hexdigest()\n"
                "seconds = [len(c['audio']) / TARGET_RATE for c in train_clips + eval_clips]\n"
                "print({{'dataset': DATASET, 'language': LANGUAGE, 'train_clips': len(train_clips), 'eval_clips': len(eval_clips), 'split_digest': SPLIT_DIGEST[:16], 'seconds_mean': round(sum(seconds) / len(seconds), 2), 'seconds_max': round(max(seconds), 2)}})\n"
                "print({{'example_text': train_clips[0]['text']}})"
            ),
        },
        {
            "md": (
                "## 5. Validate every clip → input manifest\n\n"
                "`validate_inputs` is the pipeline's public validation stage and is applied to **every** train and held-out "
                "clip before any model runs (VAL1): each waveform must satisfy the same contract `transcribe` enforces "
                "(`TASKS`, the `MIN_CHUNK_LENGTH_S`–`MAX_CHUNK_LENGTH_S` window), and this notebook adds the two rules the "
                "training loop needs — a non-empty transcript, and a clip no longer than `MAX_CHUNK_LENGTH_S` seconds, "
                "because Whisper's fixed 30-second window would silently truncate a longer clip's supervision (VAL6/VAL7: "
                "nothing is truncated; an offending clip is a rejection finding and stops the run). The per-clip manifests "
                "are folded into one input manifest written to `outputs/{stem}_input_manifest.json`. To show what rejection "
                "looks like, the cell also validates a request that breaks the chunk ceiling and records the pipeline's own "
                "error message as a finding."
            ),
            "code": (
                "import json\n\n"
                "os.makedirs('outputs', exist_ok=True)\n"
                "print({{'ceilings': {{'TASKS': list(TASKS), 'MIN_CHUNK_LENGTH_S': MIN_CHUNK_LENGTH_S, 'MAX_CHUNK_LENGTH_S': MAX_CHUNK_LENGTH_S, 'TARGET_RATE': TARGET_RATE}}}})\n"
                "input_manifest = {{'schema': dict(INPUT_SCHEMA), 'inputs': [], 'task': 'transcribe', 'language': LANGUAGE, 'chunk_length_s': MAX_CHUNK_LENGTH_S, 'split': {{'train': len(train_clips), 'eval': len(eval_clips), 'seed': SEED, 'digest': SPLIT_DIGEST}}, 'verdict': 'accepted', 'findings': [], 'model_id': MODEL_ID, 'model_revision': MODEL_REVISION}}\n"
                "for role, clip_list in (('train', train_clips), ('eval', eval_clips)):\n"
                "    for clip in clip_list:\n"
                "        audio_input = {{'array': clip['audio'], 'sampling_rate': TARGET_RATE}}\n"
                "        entry = validate_inputs(audio_input, language=LANGUAGE, task='transcribe', names=[clip['id']])['inputs'][0]\n"
                "        if not clip['text']:\n"
                "            raise ValueError(f\"{{clip['id']}}: empty transcript; every training and evaluation clip needs a reference\")\n"
                "        if entry['seconds'] > MAX_CHUNK_LENGTH_S:\n"
                "            raise ValueError(f\"{{clip['id']}}: {{entry['seconds']}} s exceeds the {{MAX_CHUNK_LENGTH_S}} s training window; trim it rather than letting the window truncate its supervision\")\n"
                "        input_manifest['inputs'].append({{**entry, 'role': role, 'reference_words': len(clip['text'].split())}})\n"
                "# Demonstrate rejection on a request that breaks a ceiling; the finding is recorded, not swallowed.\n"
                "try:\n"
                "    validate_inputs({{'array': eval_clips[0]['audio'], 'sampling_rate': TARGET_RATE}}, chunk_length_s=MAX_CHUNK_LENGTH_S + 1)\n"
                "except ValueError as exc:\n"
                "    input_manifest['findings'].append({{'input': 'oversized-chunk-probe', 'verdict': 'rejected', 'message': str(exc)}})\n"
                "with open('outputs/{stem}_input_manifest.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(input_manifest, handle, indent=2, ensure_ascii=False)\n"
                "print({{'validated_clips': len(input_manifest['inputs']), 'verdict': input_manifest['verdict'], 'findings': input_manifest['findings']}})"
            ),
        },
        {
            "md": (
                "## 6. Record the zero-shot baseline\n\n"
                "`pipe` was built in Section 3 from the digest-verified snapshot (float16 on CUDA, float32 on CPU). Before "
                "anything is trained, the unmodified model transcribes the held-out clips through `WhisperASRPipeline.transcribe`, "
                "and `corpus_word_error_rate` (total word edits over total reference words, after the package's case and "
                "punctuation normalisation) is recorded as the **baseline** (EVAL10: the naive comparison every adapted number "
                "is read against). The pipeline is then released so the training model has the GPU to itself. Look for the "
                "baseline WER and three reference/hypothesis pairs."
            ),
            "code": (
                "import gc\n"
                "import time\n\n\n"
                "def transcribe_all(asr, clip_list):\n"
                "    return [asr.transcribe({{'array': c['audio'], 'sampling_rate': TARGET_RATE}}, language=LANGUAGE, task='transcribe')['text'] for c in clip_list]\n\n\n"
                "eval_references = [c['text'] for c in eval_clips]\n"
                "DEVICE = pipe.device\n"
                "started = time.time()\n"
                "baseline_transcripts = transcribe_all(pipe, eval_clips)\n"
                "baseline_wer = corpus_word_error_rate(eval_references, baseline_transcripts)\n"
                "print({{'device': DEVICE, 'source': pipe.source, 'baseline_wer': round(baseline_wer, 4), 'seconds': round(time.time() - started, 1)}})\n"
                "for reference, hypothesis in list(zip(eval_references, baseline_transcripts))[:3]:\n"
                "    print({{'reference': reference, 'baseline': hypothesis, 'wer': round(word_error_rate(reference, hypothesis), 3)}})\n"
                "del pipe\n"
                "gc.collect()\n"
                "if DEVICE.startswith('cuda'):\n"
                "    torch.cuda.empty_cache()"
            ),
        },
        {
            "md": (
                "## 7. Attach LoRA adapters to the verified base weights\n\n"
                "`load_model(weights_dir=WEIGHTS_DIR)` re-reads the **same digest-verified snapshot** Section 3 staged (no "
                "second download, no Hub call) and the model is promoted to float32 master weights (mixed precision runs the "
                "matmuls in float16); gradient checkpointing is enabled so the encoder's 30-second activations fit a 16 GiB "
                "card at batch size 4. The forced decoder prompt is cleared so training and generation both derive the "
                "language/task prefix from the labels and the `generate` arguments rather than a config default. One frozen "
                "weight (`PROBE_MODULE`) is copied before the adapters are attached so Section 10 can prove, independently of "
                "how much the transcripts moved, that a fresh load applies the saved adapter to the base weights. Look for the "
                "trainable-parameter count and its share of the total (FT5)."
            ),
            "code": (
                "from peft import LoraConfig, get_peft_model\n\n"
                "LORA_RANK = 32  # @param {{type:\"integer\"}}\n"
                "LORA_ALPHA = 64  # @param {{type:\"integer\"}}\n"
                "LORA_TARGETS = ['q_proj', 'v_proj']\n\n"
                "base_model, processor = load_model(device=DEVICE, weights_dir=WEIGHTS_DIR, allow_download=False)\n"
                "base_model = base_model.float()\n"
                "PROBE_MODULE = 'model.decoder.layers.0.self_attn.q_proj'\n"
                "probe_base_weight = base_model.get_submodule(PROBE_MODULE).weight.detach().to('cpu', torch.float32).clone()\n"
                "base_model.config.forced_decoder_ids = None\n"
                "base_model.generation_config.forced_decoder_ids = None\n"
                "base_model.config.use_cache = False\n"
                "base_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={{'use_reentrant': False}})\n\n"
                "lora_config = LoraConfig(r=LORA_RANK, lora_alpha=LORA_ALPHA, lora_dropout=0.05, bias='none', target_modules=LORA_TARGETS)\n"
                "model = get_peft_model(base_model, lora_config)\n"
                "trainable_params, all_params = model.get_nb_trainable_parameters()\n"
                "print({{'lora_targets': LORA_TARGETS, 'trainable_parameters': trainable_params, 'all_parameters': all_params, 'trainable_percent': round(100 * trainable_params / all_params, 3), 'dtype': str(next(model.parameters()).dtype)}})"
            ),
        },
        {
            "md": (
                "## 8. Train\n\n"
                "The loop is deliberately plain PyTorch rather than a `Trainer`, so every moving part is visible (FT4/FT6):\n\n"
                "- **Features and labels.** Each clip becomes a 128-bin log-mel spectrogram padded to Whisper's fixed 30-second window by the pinned processor. The label sequence is the tokenizer's rendering of the transcript with its language/task prefix, minus the leading start token (the model prepends it when it shifts labels right), padded with `-100` so padding is ignored by the loss.\n"
                "- **Mixed precision.** On CUDA the forward pass runs under float16 autocast with a gradient scaler; the LoRA weights and optimizer state stay in float32. On CPU everything is float32 (ENV5).\n"
                "- **AdamW at a constant learning rate** (`1e-3` is the usual LoRA starting point for Whisper) with `GRAD_ACCUM` micro-batches per optimizer step; there is no warmup or decay schedule. `SEED` drives the split and the epoch shuffles; non-deterministic CUDA kernels remain a source of run-to-run variability (ENV7/ENV8).\n"
                "- **Validation loss** is the teacher-forced cross-entropy on the held-out clips after each epoch. It tracks whether the adapter is still learning; WER, the metric that matters, is measured in the next section (FT7)."
            ),
            "code": (
                "EPOCHS = 2  # @param {{type:\"integer\"}}\n"
                "BATCH_SIZE = 4  # @param {{type:\"integer\"}}\n"
                "GRAD_ACCUM = 2  # @param {{type:\"integer\"}}\n"
                "LEARNING_RATE = 1e-3  # @param {{type:\"number\"}}\n\n"
                "processor.tokenizer.set_prefix_tokens(language=LANGUAGE, task='transcribe')\n"
                "decoder_start = model.config.decoder_start_token_id\n"
                "use_amp = DEVICE.startswith('cuda')\n\n\n"
                "def make_batch(clip_list):\n"
                "    features = processor.feature_extractor([c['audio'] for c in clip_list], sampling_rate=TARGET_RATE, return_tensors='pt').input_features\n"
                "    label_rows = []\n"
                "    for c in clip_list:\n"
                "        ids = processor.tokenizer(c['text']).input_ids\n"
                "        if ids and ids[0] == decoder_start:\n"
                "            ids = ids[1:]\n"
                "        label_rows.append(ids)\n"
                "    width = max(len(ids) for ids in label_rows)\n"
                "    labels = torch.full((len(label_rows), width), -100, dtype=torch.long)\n"
                "    for row, ids in enumerate(label_rows):\n"
                "        labels[row, :len(ids)] = torch.tensor(ids)\n"
                "    return features.to(DEVICE), labels.to(DEVICE)\n\n\n"
                "def batches(clip_list, size):\n"
                "    for start in range(0, len(clip_list), size):\n"
                "        yield clip_list[start:start + size]\n\n\n"
                "def evaluate_loss(clip_list):\n"
                "    model.eval()\n"
                "    total_loss, total_tokens = 0.0, 0\n"
                "    with torch.inference_mode():\n"
                "        for batch in batches(clip_list, BATCH_SIZE):\n"
                "            features, labels = make_batch(batch)\n"
                "            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):\n"
                "                loss = model(input_features=features, labels=labels).loss\n"
                "            count = int((labels != -100).sum())\n"
                "            total_loss += loss.item() * count\n"
                "            total_tokens += count\n"
                "    return total_loss / total_tokens\n\n\n"
                "optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LEARNING_RATE)\n"
                "scaler = torch.amp.GradScaler('cuda', enabled=use_amp)\n"
                "shuffler = random.Random(SEED)\n"
                "if use_amp:\n"
                "    torch.cuda.reset_peak_memory_stats()\n"
                "started = time.time()\n"
                "history = []\n"
                "OPTIMIZER_STEPS = 0\n"
                "for epoch in range(1, EPOCHS + 1):\n"
                "    model.train()\n"
                "    epoch_clips = train_clips[:]\n"
                "    shuffler.shuffle(epoch_clips)\n"
                "    optimizer.zero_grad(set_to_none=True)\n"
                "    epoch_loss, epoch_tokens, pending, epoch_steps = 0.0, 0, 0, 0\n"
                "    micro_batches = list(batches(epoch_clips, BATCH_SIZE))\n"
                "    for step, batch in enumerate(micro_batches, start=1):\n"
                "        features, labels = make_batch(batch)\n"
                "        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):\n"
                "            loss = model(input_features=features, labels=labels).loss\n"
                "        # The window is measured from its own first micro-batch, so the last (possibly short)\n"
                "        # window is scaled by its true size and still takes its optimizer step.\n"
                "        window_start = ((step - 1) // GRAD_ACCUM) * GRAD_ACCUM\n"
                "        window = min(GRAD_ACCUM, len(micro_batches) - window_start)\n"
                "        scaler.scale(loss / window).backward()\n"
                "        count = int((labels != -100).sum())\n"
                "        epoch_loss += loss.item() * count\n"
                "        epoch_tokens += count\n"
                "        pending += 1\n"
                "        if pending == window:\n"
                "            scaler.step(optimizer)\n"
                "            scaler.update()\n"
                "            optimizer.zero_grad(set_to_none=True)\n"
                "            pending = 0\n"
                "            epoch_steps += 1\n"
                "    expected_steps = -(-len(micro_batches) // GRAD_ACCUM)\n"
                "    if epoch_steps != expected_steps or pending:\n"
                "        raise RuntimeError(f'epoch {{epoch}} took {{epoch_steps}} optimizer steps, expected {{expected_steps}} ({{pending}} micro-batches left unstepped)')\n"
                "    OPTIMIZER_STEPS += epoch_steps\n"
                "    train_loss = epoch_loss / epoch_tokens\n"
                "    validation_loss = evaluate_loss(eval_clips)\n"
                "    history.append({{'epoch': epoch, 'train_loss': round(train_loss, 4), 'validation_loss': round(validation_loss, 4), 'optimizer_steps': epoch_steps}})\n"
                "    print(history[-1])\n"
                "TRAINING_SECONDS = round(time.time() - started, 1)\n"
                "PEAK_GPU_GIB = round(torch.cuda.max_memory_allocated() / 1024 ** 3, 2) if use_amp else None\n"
                "print({{'training_seconds': TRAINING_SECONDS, 'peak_gpu_gib': PEAK_GPU_GIB, 'optimizer_steps_total': OPTIMIZER_STEPS}})"
            ),
        },
        {
            "md": (
                "## 9. Evaluate the adapted model → evaluation report\n\n"
                "The live LoRA model is wrapped in the same pipeline that produced the baseline (`WhisperASRPipeline.from_model`), "
                "so the adapted transcripts come from exactly the decoding path a user of the package gets (5-beam search, the "
                "model's 448-token limit) — not from a hand-rolled `generate` call with different settings, which can disagree "
                "with the pipeline on near-tie clips (EVAL8). On CUDA the model is first cast to float16, the precision the "
                "pipeline serves at. `adaptation_report` then writes `outputs/{stem}_evaluation_report.json`: baseline and "
                "adapted corpus WER, the zero-shot baseline as the comparison, the loss history as optimisation evidence, and "
                "the verdict `sample-sanity` (EVAL6) — one seeded split of one locale says whether the adapter helped *here*, "
                "not how it generalises. A few changed clips are shown side by side."
            ),
            "code": (
                "model.eval()\n"
                "model.config.use_cache = True\n"
                "if use_amp:\n"
                "    model = model.half()\n"
                "adapted = WhisperASRPipeline.from_model(model, processor, adapter='in-memory LoRA')\n"
                "adapted_transcripts = transcribe_all(adapted, eval_clips)\n"
                "adapted_wer = corpus_word_error_rate(eval_references, adapted_transcripts)\n"
                "del adapted\n"
                "report = adaptation_report(baseline_wer=round(baseline_wer, 4), adapted_wer=round(adapted_wer, 4), reloaded_wer=None, n_eval=len(eval_clips), history=history, dataset=DATASET, sample_kind=sample_kind)\n"
                "with open('outputs/{stem}_evaluation_report.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(report, handle, indent=2, ensure_ascii=False)\n"
                "print({{'baseline_wer': report['baselines'][0]['word_error_rate'], 'adapted_wer': adapted_wer, 'wer_delta': round(adapted_wer - baseline_wer, 4), 'verdict': report['verdict']}})\n"
                "changed = [(r, b, a) for r, b, a in zip(eval_references, baseline_transcripts, adapted_transcripts) if b != a]\n"
                "print({{'clips_with_changed_transcript': len(changed)}})\n"
                "for reference, before, after in changed[:3]:\n"
                "    print({{'reference': reference, 'baseline': before, 'adapted': after}})"
            ),
        },
        {
            "md": (
                "## 10. Export the adapter bundle, then prove it reloads against the verified base\n\n"
                "The deliverable is the **adapter bundle**, not a copy of the base model (ART1/ART4): `export_adapter_bundle` "
                "writes `adapter_model.safetensors` (a few tens of MB) and `adapter_config.json`, `metrics.json`, "
                "`provenance.json` (notebook source revision, base model identifier and immutable revision, dataset identity "
                "and split digest, hyperparameters, runtime) and an `artifact-manifest.json` listing every file with its size "
                "and SHA-256, and refuses an adapter whose saved `B` matrices are all zero. The bundle is zipped under `outputs/`.\n\n"
                "Then the notebook proves the bundle is usable from disk (VER1–VER5): the training model is released, "
                "`WhisperASRPipeline.from_pretrained(weights_dir=WEIGHTS_DIR, adapter_dir=...)` loads the **verified base "
                "snapshot** and merges the saved adapter, `verify_adapter_merge` checks on the probe module that "
                "`W_reloaded == W_base + (alpha / r) * B @ A` read back from the bundle (this proves the adapter was applied "
                "to the fresh weights however small its effect on the transcripts), and the held-out clips are transcribed "
                "again through the same pipeline — the run fails unless the reloaded transcripts agree with the in-memory "
                "adapted transcripts on at least 95% of the clips (float16 decoding is not bit-exact across two loads). The "
                "reloaded WER is added to the evaluation report, and the result, metrics and provenance are exported."
            ),
            "code": (
                "import math\n"
                "import shutil\n"
                "import zipfile\n"
                "from pathlib import Path\n\n"
                "ADAPTER_DIR = Path('outputs/whisper-asr-lora-adapter')\n"
                "ARTIFACT_ZIP = Path('outputs/whisper-asr-lora-adapter.zip')\n"
                "metrics = {{'baseline_wer': round(baseline_wer, 4), 'adapted_wer': round(adapted_wer, 4), 'wer_delta': round(adapted_wer - baseline_wer, 4), 'eval_clips': len(eval_clips), 'history': history}}\n"
                "PROVENANCE = {{\n"
                "    'artifactFormat': ADAPTER_BUNDLE_FORMAT,\n"
                "    'artifactFormatVersion': ADAPTER_BUNDLE_FORMAT_VERSION,\n"
                "    'notebook_source': NOTEBOOK_SOURCE,\n"
                "    'repository_revision': NOTEBOOK_SOURCE['repository_revision'],\n"
                "    'baseModel': MODEL_ID,\n"
                "    'baseModelRevision': MODEL_REVISION,\n"
                "    'baseModelLicense': MODEL_LICENSE,\n"
                "    'trustRemoteCode': False,\n"
                "    'dataset': {{**DATASET, 'language': LANGUAGE, 'train_clips': len(train_clips), 'eval_clips': len(eval_clips), 'seed': SEED, 'split_digest': SPLIT_DIGEST}},\n"
                "    'training': {{'method': 'lora', 'epochs': EPOCHS, 'batch_size': BATCH_SIZE, 'grad_accum': GRAD_ACCUM, 'learning_rate': LEARNING_RATE, 'lora_rank': LORA_RANK, 'lora_alpha': LORA_ALPHA, 'target_modules': LORA_TARGETS, 'mixed_precision': 'float16' if use_amp else None, 'seconds': TRAINING_SECONDS, 'peak_gpu_gib': PEAK_GPU_GIB, 'optimizer_steps': OPTIMIZER_STEPS}},\n"
                "    'runtime': {{'python': platform.python_version(), 'torch': torch.__version__, 'transformers': transformers.__version__, 'peft': peft.__version__, 'device': DEVICE}},\n"
                "}}\n"
                "manifest = export_adapter_bundle(model, ADAPTER_DIR, metrics=metrics, provenance=PROVENANCE)\n"
                "ARTIFACT_ZIP.unlink(missing_ok=True)\n"
                "with zipfile.ZipFile(ARTIFACT_ZIP, 'w', zipfile.ZIP_STORED) as archive:\n"
                "    for p in sorted(ADAPTER_DIR.rglob('*')):\n"
                "        if p.is_file():\n"
                "            archive.write(p, p.relative_to(ADAPTER_DIR).as_posix())\n"
                "zip_sha256 = hashlib.sha256(ARTIFACT_ZIP.read_bytes()).hexdigest()\n"
                "print({{'adapter_dir': str(ADAPTER_DIR), 'files': len(manifest), 'zip_mib': round(ARTIFACT_ZIP.stat().st_size / 1024 ** 2, 1), 'zip_sha256': zip_sha256}})\n\n"
                "del model, base_model, optimizer, scaler\n"
                "gc.collect()\n"
                "if use_amp:\n"
                "    torch.cuda.empty_cache()\n"
                "reloaded = WhisperASRPipeline.from_pretrained(device=DEVICE, weights_dir=WEIGHTS_DIR, allow_download=False, adapter_dir=ADAPTER_DIR)\n"
                "weight_tolerance = 1e-3 if reloaded.model.dtype == torch.float16 else 1e-5\n"
                "reload_weight_check = verify_adapter_merge(ADAPTER_DIR, PROBE_MODULE, probe_base_weight, reloaded.model.get_submodule(PROBE_MODULE).weight, rank=LORA_RANK, alpha=LORA_ALPHA, tolerance=weight_tolerance)\n"
                "reloaded_transcripts = transcribe_all(reloaded, eval_clips)\n"
                "reloaded_wer = corpus_word_error_rate(eval_references, reloaded_transcripts)\n"
                "agreement = sum(a == r for a, r in zip(adapted_transcripts, reloaded_transcripts))\n"
                "if agreement < math.ceil(0.95 * len(eval_clips)):\n"
                "    raise RuntimeError(f'Reloaded adapter reproduces only {{agreement}}/{{len(eval_clips)}} in-memory adapted transcripts')\n"
                "metrics['reloaded_wer'] = round(reloaded_wer, 4)\n"
                "metrics['reload_agreement'] = f'{{agreement}}/{{len(eval_clips)}}'\n"
                "metrics['clips_changed_vs_baseline'] = sum(b != r for b, r in zip(baseline_transcripts, reloaded_transcripts))\n"
                "metrics['reload_weight_check'] = reload_weight_check\n"
                "report = adaptation_report(baseline_wer=metrics['baseline_wer'], adapted_wer=metrics['adapted_wer'], reloaded_wer=metrics['reloaded_wer'], n_eval=len(eval_clips), history=history, dataset=DATASET, sample_kind=sample_kind)\n"
                "with open('outputs/{stem}_evaluation_report.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(report, handle, indent=2, ensure_ascii=False)\n"
                "print({{'reloaded_wer': metrics['reloaded_wer'], 'reload_agreement': metrics['reload_agreement'], 'clips_changed_vs_baseline': metrics['clips_changed_vs_baseline'], 'adapter': reloaded.adapter, 'adapter_sha256': reloaded.adapter_sha256, 'source': reloaded.source, 'reload_weight_check': reload_weight_check}})\n\n"
                "payload = {{\n"
                "    'metrics': metrics,\n"
                "    'evaluation_report': report,\n"
                "    'input_manifest': {{k: v for k, v in input_manifest.items() if k != 'inputs'}},\n"
                "    'examples': [{{'reference': r, 'baseline': b, 'reloaded': a}} for r, b, a in list(zip(eval_references, baseline_transcripts, reloaded_transcripts))[:5]],\n"
                "    'artifact': {{'zip': str(ARTIFACT_ZIP), 'sha256': zip_sha256, 'adapter_sha256': reloaded.adapter_sha256, 'files': manifest}},\n"
                "    'provenance': PROVENANCE,\n"
                "    'notebook_source': NOTEBOOK_SOURCE,\n"
                "    'repository_revision': NOTEBOOK_SOURCE['repository_revision'],\n"
                "    'model_id': MODEL_ID,\n"
                "    'model_revision': MODEL_REVISION,\n"
                "    'model_license': MODEL_LICENSE,\n"
                "    'runtime': {{'python': platform.python_version(), 'torch': torch.__version__, 'transformers': transformers.__version__, 'peft': peft.__version__, 'device': reloaded.device}},\n"
                "    'sample': {{'kind': sample_kind, 'dataset': DATASET, 'language': LANGUAGE, 'split_digest': SPLIT_DIGEST}},\n"
                "}}\n"
                "with open('outputs/{stem}_result.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(payload, handle, indent=2, ensure_ascii=False)\n"
                "shutil.copy(ADAPTER_DIR / 'metrics.json', 'outputs/{stem}_metrics.json')\n"
                "print(sorted(os.listdir('outputs')))"
            ),
        },
    ],
    "closing": (
        "## Interpretation and limits\n\n"
        "The adapted transcripts are model-generated. The baseline, adapted and reloaded WER figures are corpus-level "
        "numbers on one seeded held-out split of one locale of one public corpus (or of the uploaded BYOD set) and must "
        "not be generalized to other languages, speakers, domains, or capture conditions; a WER that moved by a few points "
        "on 100 short utterances is within the range where a different seed can change the sign, and no dispersion "
        "estimate is computed (EVAL5/ENV8). The adapter specializes the model toward the training distribution and can "
        "degrade it elsewhere (catastrophic forgetting); the tutorial measures nothing outside the held-out split. Training "
        "and validation loss are optimisation evidence only. The pipeline provides no diarization, speaker identity, "
        "biometric inference, or calibrated transcript-confidence threshold, and the adapter inherits the license "
        "obligations of both the base weights (MIT) and the training data (CC-BY-4.0 for the default sample).\n\n"
        "Successful execution proves that the recorded repository revision's pipeline module, carried in this notebook, "
        "can acquire and digest-verify the pinned model, load and resample the demonstrated data, validate it, record a "
        "zero-shot baseline through the public pipeline, train LoRA adapters with the shown configuration, export a "
        "manifested adapter bundle, and reload that bundle against the verified base with a weight-level merge check and "
        "transcript agreement — in the tested runtime, without the repository being reachable. It does **not** establish "
        "benchmark superiority, deployment calibration, safety for high-consequence use, or that the adapter generalises "
        "beyond the split it was measured on.\n\n"
        "**Try next:** change `LOCALE` to another whitespace-delimited language and compare the baseline/adapted delta; halve "
        "`LEARNING_RATE` or set `EPOCHS = 1` and watch whether the validation loss and the WER move together; upload a few "
        "minutes of your own domain speech through BYOD and read the report's `needs` field before trusting the number.\n\n"
        "## References\n\n"
        "- Repository README: https://github.com/kurtvalcorza/whisper-asr-pipeline/blob/main/README.md\n"
        "- Repository model card: https://github.com/kurtvalcorza/whisper-asr-pipeline/blob/main/MODEL_CARD.md\n"
        "- Weight provenance: https://github.com/kurtvalcorza/whisper-asr-pipeline/blob/main/docs/WEIGHTS.md\n"
        "- Upstream model: https://huggingface.co/{MODEL_ID}\n"
        "- Upstream code: https://github.com/openai/whisper\n"
        "- Whisper paper: https://arxiv.org/abs/2212.04356\n"
        "- LoRA paper: https://arxiv.org/abs/2106.09685\n"
        "- Public sample: https://huggingface.co/datasets/PolyAI/minds14\n"
    ),
}
