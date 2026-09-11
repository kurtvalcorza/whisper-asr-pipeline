# Release verification

`tutorials/whisper_asr_colab.ipynb` (`TASK-INFERENCE`) is a **release candidate** until the exact
notebook revision has executed top-to-bottom in a clean supported runtime. Unit tests, JSON
validation, code-cell compilation, and `tools/validate_release_assets.py` are necessary
checks but are **not** runtime evidence under DIMER Notebook Specification 1.0. This file is
the durable release-gate record for the notebook.

## Automatic coverage (static, every pull request)

CI runs `tools/validate_release_assets.py`, which checks:

- notebook JSON parses; every code cell compiles as plain Python (no `%`/`!` magics); no
  persisted outputs or execution counts; no unresolved placeholder markers; every code cell
  is preceded by an explanatory markdown cell;
- exactly one tutorial notebook, named in `tutorials/README.md` with its `TASK-INFERENCE`
  profile and the notebook-spec version; `metadata.dimer` declares that profile and spec `1.0`;
- the fresh-runtime bootstrap (clone by canonical URL, `DIMER_TUTORIAL_REF`, detached checkout of
  the requested revision, restart-on-stale-import guard) and the recorded `REPO_SHA` in exports;
- `MODEL_ID`/`MODEL_REVISION` are imported from the package rather than hard-coded, the revision is
  a 40-hex immutable commit, and the same identity string appears in `README.md`,
  `MODEL_CARD.md`, and `docs/WEIGHTS.md` with no stray revisions;
- the profile-specific public-API calls, exports, learner-facing statements and gated-off BYOD
  default listed in the validator; forbidden patterns (credential-in-URL, direct `transformers`
  loading that bypasses the pipeline, `trust_remote_code=True` outside the pipeline boundary,
  `pickle.load`, `torch.load(`, `extractall(`);
- `STATUS.md`, `README.md` and `tutorials/README.md` agree on one release-status token and no
  document makes an unsupported release-grade, production-readiness or benchmark claim;
- `MODEL_CARD.md` front matter, single H1, required heading order, and immutable provenance.

These are source/provenance checks. They are **not** execution evidence.

## Executor paths

| Path | Runtime | Role |
|---|---|---|
| Google Colab (supported user path) | Colab CPU runtime (CUDA used automatically when present) | The runtime the tutorial is written for; a clean top-to-bottom run here is promotion evidence |
| Kaggle CLI kernel | Kaggle CPU kernel, Python 3.12 image | Reproducible clean-room executor of the same class; the notebook is pushed verbatim plus one leading shim cell that provides `google.colab`, sets `DIMER_TUTORIAL_REF`, and chdirs to a scratch directory so the bootstrap clones the candidate |
| Kaggle CLI kernel, fresh-interpreter harness | Same Kaggle container; the committed notebook is executed verbatim, cell by cell, by `run_nb.py` in a subprocess of the container Python | Used because the Kaggle kernel pre-imports numpy 2.0.2 and this repository pins numpy 1.26.4: the tutorial's fail-closed stale-import guard correctly halts the in-kernel path after the pinned install, so the verbatim notebook runs in a fresh interpreter instead; the evidence cell proves the executed file equals the committed blob |
| Local WSL harness (pre-flight only) | Workstation, `run_nb.py` sequential cell executor with a `google.colab` shim | Builder pre-flight to catch defects before spending cloud runs; **not** a supported runtime and not promotion evidence |

## Supported release verification procedure

Before changing the registry status from `Candidate` to `Release-grade`:

1. resolve the exact PR/commit head under review and confirm static CI is green;
2. open that exact notebook revision in a new CPU (or CUDA) runtime (Colab, or the Kaggle
   executor above) with `DIMER_TUTORIAL_REF` set to the candidate commit and a clean model cache;
3. run the notebook top-to-bottom without editing implementation cells (form parameters at their
   defaults for the sample path);
4. verify that Section 1 reports `repository_revision` equal to the candidate commit and that the
   installed core package versions equal the `pyproject.toml` pins;
5. verify every default-path stage completes:
   - fresh bootstrap from GitHub at the candidate revision;
   - pinned `openai/whisper-large-v3-turbo` acquisition at the immutable revision;
   - public LibriSpeech dummy sample ingestion with its reference transcript;
   - ASR through `WhisperASRPipeline.transcribe`;
   - WER computed against the reference text;
   - `outputs/whisper_asr_result.json` written with repository SHA, model revision, runtime versions and device;
6. verify the exports exist and the interpretation section matches the observed path;
7. record the notebook Git blob id, commit, runtime (platform, Python, PyTorch, Transformers, device),
   model identifier and immutable revision, whether the model cache was clean, outcome, produced
   outputs, and any warning or applicable `SHOULD` deviation in the table below;
8. record no access tokens or other secrets.

A known-failing default path in the supported runtime blocks release.

## Recorded executions

Notebook identity is the Git blob id of `tutorials/whisper_asr_colab.ipynb` (verify with
`git rev-parse <commit>:tutorials/whisper_asr_colab.ipynb`). Wall times are the sum of per-cell
times reported by the executor and include installs and the model download; they are
measurements for the stated runtime, not general estimates.

Pre-flight runtime: WSL2 Ubuntu 24.04 (kernel 6.18.33), Python 3.12.3, Intel Core Ultra 9 275HX (24 threads), 15 GiB RAM, NVIDIA GeForce RTX 5070 Ti Laptop GPU (12,227 MiB, driver 610.88). The harness executes the working-copy notebook cell by cell with the package installed non-editably from the same tree, under an empty `HF_HOME`. **Not a supported user runtime and not promotion evidence.**

| Date (UTC) | Commit / notebook blob | Executor | Path exercised | Wall | Outcome |
|---|---|---|---|---|---|
| 2026-09-11 | `8fba91c20b2b` / blob `c308c7b1d40e` (this revision; executed file verified equal to the committed blob) | Kaggle kernel `kurtvalcorza/dimer-whisper-asr-verify-v2` v4 — fresh CPU container, Python 3.12.13, Linux 6.12.90; committed notebook executed verbatim, cell by cell, by `run_nb.py` in a fresh interpreter (the Kaggle kernel itself pre-imports numpy 2.0.2, which the tutorial's stale-import guard correctly rejects after the pinned numpy 1.26.4 install — kernel v1 halted there by design; kernel v2 at `ab00a28` failed with `operator torchvision::nms does not exist` from the orphaned runtime torchvision, fixed by the `torchvision==0.21.0` pin in this revision) | Default sample path, all 5 code cells, clean HF cache (`models--openai--whisper-large-v3-turbo` and the LibriSpeech dummy dataset are the only cache entries afterwards) | 273.6 s (209 s install, 37 s snapshot fetch + load, 22 s transcription) | **PASS** — `repository_revision` equals the candidate commit; torch 2.6.0+cu124, transformers 4.52.1, accelerate 1.3.0, datasets 4.4.0, soundfile 0.13.1 (recorded), torchvision 0.21.0 (pinned; not in that run's recorded versions — inferred from the successful pinned install); transcript "Mr. Quilter is the apostle of the middle classes, and we are glad to welcome his gospel."; WER 0.0588 (1/17, `Mr.` vs reference `MISTER`) — identical to the local pre-flight; `outputs/whisper_asr_result.json` sha256 `3b6ead78…759de`; only warning: transformers' internal `inputs` FutureWarning |
| 2026-09-11 | working tree equal to `8fba91c` (torchvision pin) / blob `c308c7b1d40e` | Local WSL harness, CPU (`CUDA_VISIBLE_DEVICES=''`), torch 2.6.0+cu124, torchvision 0.21.0, transformers 4.52.1 (pins) | Default sample path, all 5 code cells, clean cache | 213.0 s | PASS — same transcript, WER 0.0588; `outputs/whisper_asr_result.json` sha256 `45763436…dfd47` |
| 2026-09-11 | blob `c308c7b1d40e` (this revision) | Local WSL harness, CPU (`CUDA_VISIBLE_DEVICES=''`), torch 2.6.0+cu124, transformers 4.52.1, datasets 4.4.0, soundfile 0.13.1 (pins) | Default sample path, all 5 code cells, clean cache | 206.9 s (174 s of it the 1.6 GB snapshot fetch) | PASS — `repository_revision` recorded; `openai/whisper-large-v3-turbo` acquired at the pinned revision into an empty cache (`models--openai--whisper-large-v3-turbo` only); transcript "Mr. Quilter is the apostle of the middle classes, and we are glad to welcome his gospel."; WER 0.0588 (1/17: `Mr.` vs reference `MISTER`); `outputs/whisper_asr_result.json` sha256 `8b786056…7ec290`; only warning: transformers' internal `inputs` FutureWarning |
| 2026-09-11 | pre-fix working tree of `0b76b2a` | Local WSL harness, CPU | Default sample path | — | FAILED at cell 5 — `ImportError: To support decoding audio data, please install 'torchcodec'` (`datasets==4.4.0` decodes audio through torchcodec, which is not pinned). Fixed: the sample is loaded with `Audio(decode=False)` and decoded by the pinned `soundfile`. The following PASS run also showed WER 0.176 because punctuation and case counted as errors; `word_error_rate` now applies basic normalization |
| 2026-09-11 | pre-fix working tree of `0b76b2a` | Local WSL harness, CPU | Default sample path | — | FAILED at cell 3 — the stale-import guard compared `torch.__version__` (`2.6.0+cu124`) with the distribution version (`2.6.0`) and raised the restart error whenever torch was already imported. Fixed: distribution metadata is compared with itself before/after installation |

## Current status

A clean supported-class CPU execution of this exact revision is recorded above (Kaggle container, fresh interpreter, clean cache, pinned wheels, all stages of the default path, transcript and WER identical to the local pre-flight). Static CI is green on the same commit. The registry status remains **Candidate** until a reviewer confirms the recorded run against the notebook blob under review and an integrator promotes it; promotion is not performed by the builder. The commit that adds a recorded-execution row changes documentation only; the executed source is the commit named in the row.
