# Release verification

`tutorials/whisper_asr_colab.ipynb` (`TASK-INFERENCE`, **standalone** carrier) is a **release candidate** until
the exact notebook revision has executed top-to-bottom in a clean supported runtime. Unit tests, JSON
validation, code-cell compilation, and `tools/validate_release_assets.py` are necessary checks but are **not**
runtime evidence under DIMER Notebook Specification 1.1. This file is the durable release-gate record for the
notebook.

## Automatic coverage (static, every pull request)

CI runs `tools/validate_release_assets.py`, which checks:

- notebook JSON parses; every code cell compiles as plain Python (no `%`/`!` magics); no
  persisted outputs or execution counts; no unresolved placeholder markers; every code cell
  is preceded by an explanatory markdown cell;
- exactly one tutorial notebook, named in `tutorials/README.md` with its `TASK-INFERENCE`
  profile, the notebook-spec version and the standalone carrier; `metadata.dimer` declares that profile, spec `1.1`,
  `standalone: true` and `generated_from` (repository, generating revision, module, module SHA-256, generator);
- the standalone carrier (ST1–ST6, PAR1–PAR3): no clone, repository install or repository import on the
  primary path; exactly one cell tagged `embedded_module` equal to `src/whisper_asr_pipeline/pipeline.py`
  after the generator's documented rewrites; the inline `MANIFEST` equal to the committed snapshot manifest and the
  inline `PINS` equal to the `pyproject.toml` runtime pins; the notebook byte-identical to `tools/build_notebook.py`
  output; the pinned-install cell with its restart-on-stale-import guard; `NOTEBOOK_SOURCE` recorded in exports;
- `MODEL_ID`/`MODEL_REVISION` are bound only in the carried module cell (and repeated in the inline manifest,
  which the notebook asserts against the module before fetching), the revision is a 40-hex immutable commit, and the
  same identity string appears in `README.md`, `MODEL_CARD.md`, and `docs/WEIGHTS.md` with no stray revisions (the
  pinned public-sample dataset revision is the one other 40-hex value allowed);
- the profile-specific public-API calls (`stage_missing_files`, `verify_snapshot`,
  `WhisperASRPipeline.from_pretrained(weights_dir=...)`, `validate_inputs`, `transcribe`, `evaluation_report`), the
  ceiling print (`TASKS`, `MIN_CHUNK_LENGTH_S`, `MAX_CHUNK_LENGTH_S`), the pinned dataset load with
  `Audio(decode=False)` + `soundfile` decoding, the exports, the learner-facing ASR statements (generated transcript,
  no confidence or threshold, hallucination risk, WER normalisation, no diarization) and the gated-off BYOD default
  listed in the validator; forbidden patterns (credential-in-URL, any `git clone` / `github.com` / repository import
  on the primary path, a mutable `revision='main'`, direct `transformers` / `safetensors` / `huggingface_hub` use
  **outside the carried module cell**, any worker process or subprocess outside the generator-owned install cell,
  `trust_remote_code=True`, `pickle.load`, `torch.load(`, `extractall(`);
- `STATUS.md`, `README.md` and `tutorials/README.md` agree on one release-status token and no
  document makes an unsupported release-grade, production-readiness or benchmark claim;
- `MODEL_CARD.md` front matter, single H1, required heading order, and immutable provenance.

CI also runs `ruff`, `tools/build_notebook.py --check`, and the offline unit suite (`tests/test_pipeline.py`,
`tests/test_snapshot.py`, `tests/test_role_helpers.py`, `tests/test_notebook_parity.py`, `tests/test_release_assets.py`;
stubbed `transformers`/`torch`, no weights). These are source/provenance and unit checks. They are **not** execution
evidence.

## Executor paths

| Path | Runtime | Role |
|---|---|---|
| Google Colab (supported user path) | Colab CPU runtime (CUDA used automatically when present) | The runtime the tutorial is written for; a clean top-to-bottom run here is promotion evidence |
| Kaggle CLI kernel | Kaggle CPU kernel, Python 3.12 image | Reproducible clean-room executor of the same class; the notebook is pushed verbatim plus one leading shim cell that provides `google.colab` and chdirs to a scratch directory (no repository checkout is needed — the notebook is standalone) |
| Kaggle CLI kernel, fresh-interpreter harness | Same Kaggle container; the committed notebook is executed verbatim, cell by cell, by `run_nb.py` in a subprocess of the container Python | Used when the kernel pre-imports a distribution the pinned install replaces (numpy 2.0.2 vs the pinned 1.26.4): the stale-import guard correctly halts the in-kernel path, so the verbatim notebook runs in a fresh interpreter instead; the evidence cell proves the executed file equals the committed blob |
| Local WSL harness (pre-flight only) | Workstation, `run_nb.py` sequential cell executor with a `google.colab` shim | Builder pre-flight to catch defects before spending cloud runs; **not** a supported runtime and not promotion evidence |

## Supported release verification procedure

Before changing the registry status from `Candidate` to `Release-grade`:

1. resolve the exact PR/commit head under review and confirm static CI is green;
2. open that exact notebook revision in a new CPU (or CUDA) runtime (Colab, or the Kaggle executor above) with
   **no repository checkout** and a clean model cache;
3. run the notebook top-to-bottom without editing implementation cells (form parameters at their
   defaults for the sample path: `USE_BYOD = False`, `REFERENCE_TEXT = ''`);
4. verify that Section 1 reports `NOTEBOOK_SOURCE.repository_revision` equal to the revision recorded in
   `metadata.dimer.generated_from` and that the installed core package versions equal the inline `PINS` (= `pyproject.toml`);
5. verify every default-path stage completes:
   - pinned runtime installed from the inline `PINS` with no GitHub access;
   - the carried module cell executes (defines `WhisperASRPipeline`, `validate_inputs`, `evaluation_report`,
     `word_error_rate` and the ceilings) with no import of the repository package;
   - pinned `openai/whisper-large-v3-turbo` acquisition at the immutable revision through the package: the inline
     `MANIFEST` is asserted against the module identity and written to `weights/whisper-large-v3-turbo/`,
     `stage_missing_files(WEIGHTS_DIR, allow_download=True)` reports all twelve manifest entries on a clean runtime,
     `verify_snapshot` returns the manifest dict, and `from_pretrained(weights_dir=WEIGHTS_DIR)` reports
     `source == 'local-snapshot'`;
   - the public LibriSpeech dummy utterance loaded at dataset revision `5be91486e11a2d616f4ec5db8d3fd248585ac07a`,
     decoded by `soundfile`, with its waveform SHA-256 printed and `has_reference: True`;
   - `validate_inputs` writes `outputs/whisper_asr_input_manifest.json` (verdict `accepted`, one recorded
     rejection finding from the oversized-chunk probe);
   - transcription through `transcribe(audio_input, language='en', task='transcribe')` (the previous notebook's
     runs recorded "Mr. Quilter is the apostle of the middle classes, and we are glad to welcome his gospel." on the
     same utterance; the standalone path must be measured, not assumed to reproduce it);
   - `evaluation_report` writes `outputs/whisper_asr_evaluation_report.json` with verdict `sample-sanity` carrying
     `word_error_rate` (previously 0.0588 = 1/17, `Mr.` vs `MISTER`);
   - `outputs/whisper_asr_result.json` and `outputs/whisper_asr_transcript.txt` written with `NOTEBOOK_SOURCE`,
     model revision, model licence, runtime versions and device;
6. verify the exports exist and the interpretation section matches the observed path;
7. record the notebook Git blob id, commit, runtime (platform, Python, PyTorch, transformers, device),
   model identifier and immutable revision, whether the model cache was clean, outcome, produced
   outputs, and any warning or applicable `SHOULD` deviation in the table below;
8. record no access tokens or other secrets.

A known-failing default path in the supported runtime blocks release.

## Recorded executions

Notebook identity is the Git blob id of `tutorials/whisper_asr_colab.ipynb` (verify with
`git rev-parse <commit>:tutorials/whisper_asr_colab.ipynb`). Wall times are the sum of per-cell
times reported by the executor and include installs and the model download; they are
measurements for the stated runtime, not general estimates.

### Standalone carrier (NOTEBOOK_SPEC 1.1 §3.6) — current notebook

| Date (UTC) | Commit / notebook blob | Executor | Path exercised | Wall | Outcome |
|---|---|---|---|---|---|
| | | | Default sample path | | pending — queued to the GPU lane |

### Previous repository-installing notebook (NOTEBOOK_SPEC 1.0) — audit trail, does not cover the standalone carrier

Pre-flight runtime: WSL2 Ubuntu 24.04 (kernel 6.18.33), Python 3.12.3, Intel Core Ultra 9 275HX (24 threads), 15 GiB RAM, NVIDIA GeForce RTX 5070 Ti Laptop GPU (12,227 MiB, driver 610.88). The harness executes the working-copy notebook cell by cell with the package installed non-editably from the same tree, under an empty `HF_HOME`. **Not a supported user runtime and not promotion evidence.**

| Date (UTC) | Commit / notebook blob | Executor | Path exercised | Wall | Outcome |
|---|---|---|---|---|---|
| 2026-09-11 | `ca4c5ae82a77` / blob `a5b13f9bbfe2` (the notebook blob at this revision; the two later commits change `tools/validate_release_assets.py` lint and documentation only) | Kaggle kernel `kurtvalcorza/dimer-whisper-asr-verify-v2` v5 — fresh CPU container, Python 3.12.13, Linux 6.12.90; committed notebook executed verbatim, cell by cell, by `run_nb.py` in a fresh interpreter (executed blob == committed blob, measured in-run); the Kaggle kernel itself pre-imports numpy 2.0.2 and Pillow 11.3.0, which the generalized stale-import guard of this revision would reject after the pinned install, hence the fresh interpreter | Default sample path, all 5 code cells, clean HF cache (`models--openai--whisper-large-v3-turbo` + the LibriSpeech dummy dataset are the only cache entries afterwards) | 328.9 s (256 s install, 35 s snapshot fetch + load, 34 s transcription) | **PASS** — `repository_revision` equals the candidate commit; recorded versions torch 2.6.0, torchvision 0.21.0, torchaudio 2.6.0, transformers 4.52.1, numpy 1.26.4, pandas 2.2.3, datasets 4.4.0, accelerate 1.3.0, soundfile 0.13.1 (all pins); transcript "Mr. Quilter is the apostle of the middle classes, and we are glad to welcome his gospel."; WER 0.0588 (1/17, `Mr.` vs `MISTER`); `outputs/whisper_asr_result.json` sha256 `e4a4cf21…fdc96` |
| 2026-09-11 | `8fba91c20b2b` / blob `c308c7b1d40e` (this revision; executed file verified equal to the committed blob) | Kaggle kernel `kurtvalcorza/dimer-whisper-asr-verify-v2` v4 — fresh CPU container, Python 3.12.13, Linux 6.12.90; committed notebook executed verbatim, cell by cell, by `run_nb.py` in a fresh interpreter (the Kaggle kernel itself pre-imports numpy 2.0.2, which the tutorial's stale-import guard correctly rejects after the pinned numpy 1.26.4 install — kernel v1 halted there by design; kernel v2 at `ab00a28` failed with `operator torchvision::nms does not exist` from the orphaned runtime torchvision, fixed by the `torchvision==0.21.0` pin in this revision) | Default sample path, all 5 code cells, clean HF cache (`models--openai--whisper-large-v3-turbo` and the LibriSpeech dummy dataset are the only cache entries afterwards) | 273.6 s (209 s install, 37 s snapshot fetch + load, 22 s transcription) | **PASS** — `repository_revision` equals the candidate commit; torch 2.6.0+cu124, transformers 4.52.1, accelerate 1.3.0, datasets 4.4.0, soundfile 0.13.1 (recorded), torchvision 0.21.0 (pinned; not in that run's recorded versions — inferred from the successful pinned install); transcript "Mr. Quilter is the apostle of the middle classes, and we are glad to welcome his gospel."; WER 0.0588 (1/17, `Mr.` vs reference `MISTER`) — identical to the local pre-flight; `outputs/whisper_asr_result.json` sha256 `3b6ead78…759de`; only warning: transformers' internal `inputs` FutureWarning |
| 2026-09-11 | working tree equal to `8fba91c` (torchvision pin) / blob `c308c7b1d40e` | Local WSL harness, CPU (`CUDA_VISIBLE_DEVICES=''`), torch 2.6.0+cu124, torchvision 0.21.0, transformers 4.52.1 (pins) | Default sample path, all 5 code cells, clean cache | 213.0 s | PASS — same transcript, WER 0.0588; `outputs/whisper_asr_result.json` sha256 `45763436…dfd47` |
| 2026-09-11 | blob `c308c7b1d40e` (this revision) | Local WSL harness, CPU (`CUDA_VISIBLE_DEVICES=''`), torch 2.6.0+cu124, transformers 4.52.1, datasets 4.4.0, soundfile 0.13.1 (pins) | Default sample path, all 5 code cells, clean cache | 206.9 s (174 s of it the 1.6 GB snapshot fetch) | PASS — `repository_revision` recorded; `openai/whisper-large-v3-turbo` acquired at the pinned revision into an empty cache (`models--openai--whisper-large-v3-turbo` only); transcript "Mr. Quilter is the apostle of the middle classes, and we are glad to welcome his gospel."; WER 0.0588 (1/17: `Mr.` vs reference `MISTER`); `outputs/whisper_asr_result.json` sha256 `8b786056…7ec290`; only warning: transformers' internal `inputs` FutureWarning |
| 2026-09-11 | pre-fix working tree of `0b76b2a` | Local WSL harness, CPU | Default sample path | — | FAILED at cell 5 — `ImportError: To support decoding audio data, please install 'torchcodec'` (`datasets==4.4.0` decodes audio through torchcodec, which is not pinned). Fixed: the sample is loaded with `Audio(decode=False)` and decoded by the pinned `soundfile`. The following PASS run also showed WER 0.176 because punctuation and case counted as errors; `word_error_rate` now applies basic normalization |
| 2026-09-11 | pre-fix working tree of `0b76b2a` | Local WSL harness, CPU | Default sample path | — | FAILED at cell 3 — the stale-import guard compared `torch.__version__` (`2.6.0+cu124`) with the distribution version (`2.6.0`) and raised the restart error whenever torch was already imported. Fixed: distribution metadata is compared with itself before/after installation |

## Current status

**No clean-runtime execution of the standalone notebook has been recorded yet**; the run is **pending** and
queued to the GPU lane. The rows above under the previous notebook prove that the pipeline's transcription path,
the pinned snapshot fetch through the Hub loader and the public sample produced a stable transcript and WER in a
clean Kaggle container, but they executed the earlier repository-installing carrier: the standalone path (carried
module cell, inline manifest, `stage_missing_files` through `hf_hub_download` for all twelve files, `verify_snapshot`
over the real 1.6 GB checkpoint, and `AutoProcessor`/`AutoModelForSpeechSeq2Seq` on the verified directory) has been
validated statically only (parity PASS, carrier probe with the repository package blocked) and never run. Static
validation (`tools/validate_release_assets.py`), nbformat validation, a `compile()` sweep over every code cell, and
the offline unit suite passed on the tutorial source at the candidate revision, which is necessary but not
sufficient. The registry status remains **Candidate** until a reviewer confirms a recorded run against the notebook
blob under review and an integrator promotes it; promotion is not performed by the builder. Two facts a reviewer
should weigh: `stage_missing_files` was exercised only with an injected downloader in the unit suite, and
`from_pretrained(weights_dir=...)` only with stubbed `transformers` classes; the clean run will be the first execution
of the standalone path, of the staging path, and of the local-directory loading path against the real weights.
