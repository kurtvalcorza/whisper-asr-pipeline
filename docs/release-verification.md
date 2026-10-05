# Release verification

`tutorials/whisper_asr_colab.ipynb` (`TASK-INFERENCE`) and `tutorials/whisper_asr_finetune_colab.ipynb`
(`E2E`) are each a **release candidate** until the exact notebook revision has executed
top-to-bottom in a clean supported runtime. Unit tests, JSON
validation, code-cell compilation, and `tools/validate_release_assets.py` are necessary
checks but are **not** runtime evidence under DIMER Notebook Specification 2.2. This file is
the durable release-gate record for both notebooks; each is promoted on its own evidence.

## Automatic coverage (static, every pull request)

CI runs `tools/validate_release_assets.py`, which checks:

- notebook JSON parses; every code cell compiles as plain Python (no `%`/`!` magics); no
  persisted outputs or execution counts; no unresolved placeholder markers; every code cell
  is preceded by an explanatory markdown cell;
- every notebook in `tutorials/` is one of the two declared generated notebooks or the declared workshop notebook (checked separately, see the workshop section below), each named in
  `tutorials/README.md` with its profile (`TASK-INFERENCE`, `E2E`) and the notebook-spec version;
  `metadata.dimer` declares that profile, spec `2.0` and a pedagogical mode;
- the standalone carrier (NOTEBOOK_SPEC 2.0 §4): each notebook is byte-identical to its generator's output
  (`tools/build_notebook.py --check`, PAR3), carries the package module verbatim (PAR1) and the committed
  manifest and pins inline (PAR2), performs no clone or repository install, and records `NOTEBOOK_SOURCE`
  (repository revision, module SHA-256) in exports; the restart-on-stale-import guard must raise;
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
| Kaggle CLI GPU kernel, fresh-interpreter harness | Kaggle GPU kernel (P100 or T4 as assigned), Python 3.12 image; same verbatim `run_nb.py` executor as the CPU row | Executor for `whisper_asr_finetune_colab.ipynb`, whose default configuration needs CUDA; the evidence cell records the assigned GPU |
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

For `whisper_asr_finetune_colab.ipynb` the same procedure applies with a 16 GiB-class CUDA runtime
(Colab T4, or a Kaggle GPU kernel, which is assigned a P100 or T4), form parameters at their defaults (`en-US`, 400/100 clips, 2 epochs, rank 32), and the
default-path stages are instead:

- pinned install from `tools/finetune-pins.txt` (runtime dependencies plus the `tutorial` and `finetune` extras), no repository checkout;
- pinned base acquisition, `PolyAI/minds14` ingestion with 8→16 kHz resampling and the recorded
  `split_digest`;
- zero-shot baseline corpus WER on the held-out clips through `WhisperASRPipeline`;
- LoRA attachment through `load_model`, mixed-precision training with per-epoch train/validation
  loss, peak GPU memory and wall time;
- adapted corpus WER on the same held-out clips;
- `outputs/whisper-asr-lora-adapter.zip` with `artifact-manifest.json`, non-zero saved LoRA
  weights, and a fresh reload through `WhisperASRPipeline.from_pretrained(adapter_dir=...)` whose
  probe-module weight equals base + scaled `B @ A` from the bundle, and whose transcripts agree
  with the in-memory adapted transcripts (same pipeline, via `WhisperASRPipeline.from_model`) on
  at least 95% of the held-out clips; the number of clips changed versus the baseline is recorded;
- `outputs/whisper_asr_finetune_result.json` with baseline/adapted/reloaded WER, training
  configuration, repository SHA, model revision, runtime and device.

A known-failing default path in the supported runtime blocks release.

## Recorded executions

Notebook identity is the Git blob id of the notebook file (verify with
`git rev-parse <commit>:tutorials/<notebook>.ipynb`); rows name the notebook they cover. Wall times are the sum of per-cell
times reported by the executor and include installs and the model download; they are
measurements for the stated runtime, not general estimates.

Pre-flight runtime: WSL2 Ubuntu 24.04 (kernel 6.18.33), Python 3.12.3, Intel Core Ultra 9 275HX (24 threads), 15 GiB RAM, NVIDIA GeForce RTX 5070 Ti Laptop GPU (12,227 MiB, driver 610.88). The harness executes the working-copy notebook cell by cell with the package installed non-editably from the same tree, under an empty `HF_HOME`. **Not a supported user runtime and not promotion evidence.**

| Date (UTC) | Commit / notebook blob | Executor | Path exercised | Wall | Outcome |
|---|---|---|---|---|---|
| 2026-09-14 | `b3a123e` / blob `818d2e556b44` of `tutorials/whisper_asr_finetune_colab.ipynb` | Kaggle kernel `kurtvalcorza/dimer-nb2-whisper-asr-finetune` v2 — GPU container assigned **Tesla T4** (driver 580.159.04), Python 3.12, torch 2.10.0+cu128; clean HF cache | Default LoRA finetune path, all 10 code cells: mind14 sample dataset, staged snapshot (26 files, 1622 MB), training, evaluation report, adapter bundle zip export and reload verification | 1002.0 s | **PASS** — 10/10 ok code cells executed cleanly (1 restart after install cell) |
| 2026-09-11 | `db4daf856e82` / blob `e916da1c9270` of `tutorials/whisper_asr_finetune_colab.ipynb` (the executed file was verified equal to this committed blob in-run; HEAD changes only learner-facing Prerequisites markdown content plus regenerated markdown-cell ID metadata; no code cell differs, verify with `git diff e916da1c9270 <head>:tutorials/whisper_asr_finetune_colab.ipynb`) | Kaggle kernel `kurtvalcorza/dimer-whisper-finetune-t4-verify` v1 — fresh GPU container assigned a **Tesla P100-PCIE-16GB** (driver 580.159.04), Python 3.12.13, Linux 6.12.90; committed notebook executed verbatim, cell by cell, by `run_nb.py` in a fresh interpreter under a clean HF cache (`models--openai--whisper-large-v3-turbo` and `datasets--PolyAI--minds14` are the only cache entries afterwards) | Default path, all 7 code cells: `en-US`, 400 train / 100 held-out clips (`split_digest` `686ef3a3d988…`), LoRA r=32 α=64 on q/v, 2 epochs × 50 measured optimizer steps, float16 autocast | 1617.7 s (236 s install, 109 s baseline, 1106.7 s training, 72 s adapted eval, 75 s export + reload); peak CUDA allocation 4.48 GiB | **PASS** — `repository_revision` equals the candidate commit; recorded versions torch 2.6.0+cu124, torchvision 0.21.0, torchaudio 2.6.0, transformers 4.52.1, peft 0.17.1, numpy 1.26.4, datasets 4.4.0, accelerate 1.3.0, soundfile 0.13.1 (all pins); baseline corpus WER 0.4387 → adapted 0.4066 → reloaded 0.4066, reload agreement 100/100, 100/100 transcripts moved toward the corpus's lowercase unpunctuated reference style; weight probe on `model.decoder.layers.0.self_attn.q_proj`: merge error 1.2e-4 (tolerance 1e-3) against an adapter delta of 1.5e-2; train loss 0.517 → 0.283 while validation loss 0.578 → 0.612 (mild overfitting in epoch 2; WER still improved); `outputs/whisper-asr-lora-adapter.zip` sha256 `d07dcffad2cb…`, `adapter_model.safetensors` sha256 `0005c56e7691…`, `outputs/whisper_asr_finetune_result.json` sha256 `e067b63793eb…`. The corpus's own references are noisy transcriptions, so the absolute WER measures agreement with them, not speech-recognition accuracy |
| 2026-09-11 | `ca4c5ae82a77` / blob `a5b13f9bbfe2` (the notebook blob at this revision; the two later commits change `tools/validate_release_assets.py` lint and documentation only) | Kaggle kernel `kurtvalcorza/dimer-whisper-asr-verify-v2` v5 — fresh CPU container, Python 3.12.13, Linux 6.12.90; committed notebook executed verbatim, cell by cell, by `run_nb.py` in a fresh interpreter (executed blob == committed blob, measured in-run); the Kaggle kernel itself pre-imports numpy 2.0.2 and Pillow 11.3.0, which the generalized stale-import guard of this revision would reject after the pinned install, hence the fresh interpreter | Default sample path, all 5 code cells, clean HF cache (`models--openai--whisper-large-v3-turbo` + the LibriSpeech dummy dataset are the only cache entries afterwards) | 328.9 s (256 s install, 35 s snapshot fetch + load, 34 s transcription) | **PASS** — `repository_revision` equals the candidate commit; recorded versions torch 2.6.0, torchvision 0.21.0, torchaudio 2.6.0, transformers 4.52.1, numpy 1.26.4, pandas 2.2.3, datasets 4.4.0, accelerate 1.3.0, soundfile 0.13.1 (all pins); transcript "Mr. Quilter is the apostle of the middle classes, and we are glad to welcome his gospel."; WER 0.0588 (1/17, `Mr.` vs `MISTER`); `outputs/whisper_asr_result.json` sha256 `e4a4cf21…fdc96` |
| 2026-09-11 | `8fba91c20b2b` / blob `c308c7b1d40e` (this revision; executed file verified equal to the committed blob) | Kaggle kernel `kurtvalcorza/dimer-whisper-asr-verify-v2` v4 — fresh CPU container, Python 3.12.13, Linux 6.12.90; committed notebook executed verbatim, cell by cell, by `run_nb.py` in a fresh interpreter (the Kaggle kernel itself pre-imports numpy 2.0.2, which the tutorial's stale-import guard correctly rejects after the pinned numpy 1.26.4 install — kernel v1 halted there by design; kernel v2 at `ab00a28` failed with `operator torchvision::nms does not exist` from the orphaned runtime torchvision, fixed by the `torchvision==0.21.0` pin in this revision) | Default sample path, all 5 code cells, clean HF cache (`models--openai--whisper-large-v3-turbo` and the LibriSpeech dummy dataset are the only cache entries afterwards) | 273.6 s (209 s install, 37 s snapshot fetch + load, 22 s transcription) | **PASS** — `repository_revision` equals the candidate commit; torch 2.6.0+cu124, transformers 4.52.1, accelerate 1.3.0, datasets 4.4.0, soundfile 0.13.1 (recorded), torchvision 0.21.0 (pinned; not in that run's recorded versions — inferred from the successful pinned install); transcript "Mr. Quilter is the apostle of the middle classes, and we are glad to welcome his gospel."; WER 0.0588 (1/17, `Mr.` vs reference `MISTER`) — identical to the local pre-flight; `outputs/whisper_asr_result.json` sha256 `3b6ead78…759de`; only warning: transformers' internal `inputs` FutureWarning |
| 2026-09-11 | working tree equal to `8fba91c` (torchvision pin) / blob `c308c7b1d40e` | Local WSL harness, CPU (`CUDA_VISIBLE_DEVICES=''`), torch 2.6.0+cu124, torchvision 0.21.0, transformers 4.52.1 (pins) | Default sample path, all 5 code cells, clean cache | 213.0 s | PASS — same transcript, WER 0.0588; `outputs/whisper_asr_result.json` sha256 `45763436…dfd47` |
| 2026-09-11 | blob `c308c7b1d40e` (this revision) | Local WSL harness, CPU (`CUDA_VISIBLE_DEVICES=''`), torch 2.6.0+cu124, torchvision 0.21.0, transformers 4.52.1, datasets 4.4.0, accelerate 1.3.0, soundfile 0.13.1 (pins) | Default sample path, all 5 code cells, clean cache | 206.9 s (174 s of it the 1.6 GB snapshot fetch) | PASS — `repository_revision` recorded; `openai/whisper-large-v3-turbo` acquired at the pinned revision into an empty cache (`models--openai--whisper-large-v3-turbo` only); transcript "Mr. Quilter is the apostle of the middle classes, and we are glad to welcome his gospel."; WER 0.0588 (1/17: `Mr.` vs `MISTER`); `outputs/whisper_asr_result.json` sha256 `8b786056…7ec290`; only warning: transformers' internal `inputs` FutureWarning |
| 2026-09-11 | pre-fix working tree of `0b76b2a` | Local WSL harness, CPU | Default sample path | — | FAILED at cell 5 — `ImportError: To support decoding audio data, please install 'torchcodec'` (`datasets==4.4.0` decodes audio through torchcodec, which is not pinned). Fixed: the sample is loaded with `Audio(decode=False)` and decoded by the pinned `soundfile`. The following PASS run also showed WER 0.176 because punctuation and case counted as errors; `word_error_rate` now applies basic normalization |
| 2026-09-11 | pre-fix working tree of `0b76b2a` | Local WSL harness, CPU | Default sample path | — | FAILED at cell 3 — the stale-import guard compared `torch.__version__` (`2.6.0+cu124`) with the distribution version (`2.6.0`) and raised the restart error whenever torch was already imported. Fixed: distribution metadata is compared with itself before/after installation |

## Current status

**2026-10-05 — review fixes; no run of the current blobs is recorded.** The 2026-10-05 reviews (`docs/reviews/2026-10-05-notebook-review/`) found that the recorded rows do not cover the standalone notebooks as they stand: every `whisper_asr_colab.ipynb` row executed the earlier, repository-installing notebook (blobs `a5b13f9b`, `c308c7b1`, 5 code cells) in a fresh-interpreter harness, and the one `whisper_asr_finetune_colab.ipynb` standalone row (2026-09-14, blob `818d2e55`, Kaggle T4, 1002.0 s) needed one manual restart after the install cell, which is not REL2 evidence, and recorded no metrics or versions. Both notebooks were then regenerated: an isolated, hash-locked `uv` environment replaces the in-kernel install and its restart guard (spec 2.2), the inference pins now include `datasets==4.4.0`, and the guided layer and the BYOD and evaluation fixes were added (details in the two `*_Fixes.md` files). **A one-pass hosted Run all of each regenerated blob, recorded with `restarted: false`, is the open gate.**

**2026-09-14 — `whisper_asr_finetune_colab.ipynb` was regenerated as a standalone NOTEBOOK_SPEC 2.0 notebook and run on Kaggle** (Tesla T4, 1002.0 s, 10/10 code cells after one restart, kernel `dimer-nb2-whisper-asr-finetune v2`). The task-inference standalone notebook was not run.

Before the standalone migration: a clean supported-class execution of the notebook blob at that revision is recorded in the rows above (Kaggle container, fresh interpreter, clean cache, pinned wheels, all stages of the default path, verbatim blob measured in-run, every version recorded); it supersedes the earlier rows, which remain as the audit trail of the review round. Static CI is green on the same branch. The registry status remains **Candidate** until a reviewer confirms the recorded run against the notebook blob under review and an integrator promotes it; promotion is not performed by the builder. The commit that adds a recorded-execution row changes documentation only; the executed source is the commit named in the row.
## Speech-recognition workshop notebook

`tutorials/DIMER_Whisper_Speech_Recognition_Workshop.ipynb` (`TASK-INFERENCE` / `WORKSHOP`, DIMER Notebook Specification 2.2) is a **Candidate**. It carries its reference module, runner, frozen sample manifest, model manifest and hash-pinned dependency lock, and installs them into an isolated `uv` Python 3.12.12 environment. Its design is `docs/speech-recognition-workshop-spec.md`.

| Check | Automatic (every pull request) | Manual (before promotion) |
|---|---|---|
| Metadata, opening declaration, no persisted outputs, every code cell plain Python | `tools/validate_release_assets.py` | — |
| Each carried file matches `CARRIED_HASHES`; carried `whisper_reference.py` and model manifest equal the package; `source.json` agrees with the metadata | `tools/validate_release_assets.py`, `tests/test_release_assets.py` | — |
| Default `Run all` on a fresh Colab T4 runtime without a restart, with total time, peak memory and disk recorded | — | recorded 2026-09-27 (PASS, 173 s; peak memory and disk not captured) |
| Optional exploration and bring-your-own-audio branches | — | not yet exercised |

| Date (UTC) | Notebook source | Executor | Path exercised | Wall | Outcome |
|---|---|---|---|---|---|
| 2026-09-27 | `aca0e06` / blob `56749100397d` (the executed file's code cells equal this blob; Colab added only a `# @title` line to cells 3 and 4, and the carried files and hashes are unchanged) | Google Colab, fresh **Tesla T4** runtime (reported by cell 2); isolated `uv` Python 3.12.12 environment built from the carried hash-pinned lock | Default `Run all` without a restart: all 10 code cells executed in order (execution counts 1–10, no errors). Stages: `prepare` (64 evaluation / 16 activity recordings), `model` (snapshot staged at the pinned revision and verified), `evaluate`, `activity`, then report export. Exploration and BYOD left at their defaults (off) | 173.0 s elapsed from setup (80.9 s bootstrap, 25.6 s model acquisition); inference 17.5 s for 494 s of evaluation audio | **PASS**. Evaluate: WER 0.1981, CER 0.1646 (S 56, D 5, I 122, N 924), exact-match 0.453, 0 truncated, against 1.0 for the empty-transcript baseline. Activity at 10 dB SNR: clean WER 0.1850, noisy 0.2081; paired 0 improved / 12 unchanged / 4 worsened. These equal the CPU pre-flight figures. Report ZIPs reload-verified: evaluate sha256 `c598d9d16aaf…` (41,532 bytes), activity sha256 `34b433b0c98d…` (19,362 bytes). Peak GPU memory and disk use were not captured in the visible outputs |
| 2026-09-27 | Uploaded draft (carried `workshop.py` sha256 `009a5509487f…`) | Builder pre-flight in a Linux container, CPU only (4 cores); exact hash-pinned lock installed with `uv` 0.12.15 into managed Python 3.12.12 (torch 2.6.0+cu124, transformers 4.52.1); the runner's CUDA-only lines (device check, `cuda:0`, float16, synchronize, device name, peak memory) patched in a copy for CPU | `prepare`, `model`, `evaluate` | — | **FAILED** at `evaluate`: `Audio amplitude must be normalized PCM-like samples within [-1.01, 1.01]`. `prepare` stores the 16 kHz resampled waveform, and resampling the 8 kHz clip `en-US~APP_ERROR/602ba642963e11ccd901cccc.wav` overshoots to a peak of 1.0225; `evaluate` then re-ran the learner-audio amplitude check on it. Deterministic, so a Colab run would fail the same way. Fixed: `evaluate` and `activity` reload the prepared waveform as stored (16 kHz required) and identify it by its recorded SHA-256 |
| 2026-09-27 | This branch (carried `workshop.py` sha256 `6763e433336c…`) | Same pre-flight environment | `prepare` (64 evaluation / 16 activity recordings, 597 s of audio), `model` (1.6 GB snapshot staged at the pinned revision and verified), `evaluate`, `activity`; then the notebook's display, metric-demo and export cells run against those outputs; exploration and BYOD left at their defaults (off) | prepare 6 s, model 14 s, evaluate 414 s, activity 171 s (CPU float32) | **PASS**. Evaluate: WER 0.1981, CER 0.1646 (S 56, D 5, I 122, N 924), exact-match 0.453, 0 truncated, against 1.0 for the empty-transcript baseline. Activity at 10 dB SNR: clean WER 0.1850, noisy 0.2081; paired 0 improved / 12 unchanged / 4 worsened. Both report ZIPs exported and reload-verified. **Not a supported runtime and not promotion evidence**: CPU float32 differs from the notebook's T4 float16 path, so Colab figures may differ |
| 2026-10-03 | `1b0a7de` / blob `fc8acbb6a3df` (the 2026-10-03 source-layout revision), downloaded from GitHub at the PR head and blob-verified before the session; executed file `docs/execution-evidence/2026-10-03/DIMER_Whisper_Speech_Recognition_Workshop_1b0a7de_colab-cli-t4.ipynb`, sha256 `e70b8f614f4a32c8753224ecdadfe0b79bd308bb2b6571812c9836093426a356` | Google Colab CLI 0.7.4 on a fresh Colab **Tesla T4** session (reported by cell 1) via the workspace `colab-cli-serial-test-suite` (`colab new --gpu T4`, `colab exec -f`, `colab stop`); isolated `uv` Python 3.12.12 environment built from the carried hash-pinned lock | Default path: all 10 code cells ran in order in one kernel; this is not a browser Run all, and the CLI records no execution counts, so order is evidenced by its `Executing cell k/N` log (1/10–10/10). Stages: `prepare` (64 evaluation / 16 activity recordings, 597 s of audio), `model` (pinned snapshot verified), `evaluate`, `activity`, then report export. Optional 5/20 dB exploration and BYOD not run (left at their defaults, off) | 175.4 s session wall; notebook-reported 158.1 s elapsed from setup (66.3 s bootstrap, 23.0 s model acquisition); inference 17.3 s for 494 s of evaluation audio | **PASSED 10/10**, no errors. Evaluate: WER 0.1981, CER 0.1646 (S 56, D 5, I 122, N 924), exact-match 0.453, 0 failed, 0 truncated, against 1.0 for the empty-transcript baseline. Activity at 10 dB SNR: clean WER 0.1850, noisy 0.2081; paired 0 improved / 12 unchanged / 4 worsened. Compared with the 2026-09-27 Colab run of blob `56749100397d`: every metric and count above is equal. Report ZIPs reload-verified but their digests and sizes differ, as expected from the 2026-10-02 `summary.md` changes: evaluate sha256 `56422df6b588…` (41,626 bytes, +94), activity sha256 `a118b9e79c7b…` (19,486 bytes, +124). Timing differs: elapsed 158.1 s vs 173.0 s, bootstrap 66.3 s vs 80.9 s, model acquisition 23.0 s vs 25.6 s, inference 17.3 s vs 17.5 s. Boundary: saved outputs inspected; peak GPU memory and disk not captured; the exploration and BYOD journeys (REL12) were not exercised and remain open. Status stays **Candidate** |

A fresh Colab T4 `Run all` of the committed blob is recorded in the first row above. The notebook stays **Candidate** until a reviewer confirms that run against the blob under review and an integrator promotes it; the optional branches remain unexercised.

**2026-10-02 notebook review fixes (ASR-M1..M3, ASR-m1..m5).** The review in `reviews/2026-10-02-notebook-review/` changed the notebook (prose, the `show_results` helper, and the carried `workshop.py`: BYOD file identity and named refusals, reference provenance in `summary.md`). The 2026-09-27 run above covers blob `56749100397d` only and is **not** evidence for the revised blob. Before promotion, a fresh Colab T4 run of the revised blob must cover the default path, the 5 or 20 dB exploration, and BYOD with labelled, unlabelled and one rejected input (REL12). Local checks for the fix were CPU-only with a stand-in transcriber and are not clean-runtime evidence.

**2026-10-03 source layout change.** The workshop's `CARRIED_FILES` literal in cell 3 was one 203,177-character line. `tools/split_workshop_carrier.py` rewrote it as parenthesised runs of short string pieces (at most 1,000 characters each), and the change is logged in `metadata.dimer.revisions`. Python joins the pieces back into the same text: the carried files, `CARRIED_HASHES`, the carried `source.json` and `generated_from` are unchanged, and no cell line is now longer than 2,000 characters (`python tools/split_workshop_carrier.py --check`). The notebook blob changes from `baa5be541c92` (the 2026-10-02 review-fix revision, which has no hosted run) to `fc8acbb6a3df`. A hosted re-run of the new blob passed on the default path (Colab CLI T4, 2026-10-03, last row of the table above); the exploration and BYOD runs required before promotion are still open. Status stays **Candidate**.

## Filipino Audio Archive Search capstone

Status: **Engineering Preview / Candidate.** Human relevance review is pending
(0 of 7,200 cells judged); scores below are nominated-anchor recovery diagnostics,
not benchmark Recall, MRR or nDCG.

| Date (UTC) | Commit / notebook blob | Executor | Path exercised | Wall | Outcome |
|---|---|---|---|---|---|
| 2026-10-03 | `e4b3fed` / blob `f890d97d43fd` (downloaded from GitHub at the commit and blob-verified before the session; all cells equal the PR head) | Google Colab CLI 0.7.4 on a fresh Colab **Tesla T4** session via the workspace `colab-cli-serial-test-suite` (`colab new --gpu T4`, `colab exec -f`, `colab stop`); cells run in order in one kernel, not a browser Run all; order evidenced by the CLI's `Executing cell k/18` log (the CLI records no execution counts) | Default path, every optional toggle off | 615.3 s wall | **PASS**: 18/18 code cells, no errors. Outputs equal the `ad42d18` run except download progress bars. Executed notebook: `execution-evidence/2026-10-03/DIMER_Filipino_Audio_Archive_Search_Capstone_e4b3fed_colab-cli-t4.ipynb`, SHA-256 `2d3e18e7adce2135b87d23672be09b77b58468404b177654173ab90b505f0005`. Saved outputs inspected; BYOD and search toggles not exercised |
| 2026-09-28 | `ad42d18` (`main`) / blob `47da982901db` (executed file's 35 cell sources and IDs equal this blob; no toggle changed) | Google Colab, fresh **Tesla T4**, maintainer-supplied; isolated `uv` Python 3.12.12 environment from the carried lock | Default `Run all`, every optional toggle off | Stage total 483.8 s (Colab saved no per-cell timings) | **PASS**: 18/18 code cells, execution counts 1–18, no errors; see below |
| 2026-09-28 | `5664f61` / blob `47da982901db` (fetched from GitHub at the commit and blob-verified in-run; four toggle lines edited and three harness cells appended, listed below) | Kaggle kernel `kurtvalcorza/dimer-nb2-filipino-audio-archive-search-capstone` v1, serial test-suite executor: fresh `nbclient` interpreter, **Tesla T4** ×2 (driver 580.159.04; the runtime uses `cuda:0`), clean HF cache; isolated `uv` CPython 3.12.12 venv with torch 2.11.0+cu130 from the carried lock | `Run all` with search, competitor playback and BYOD on; harness cells re-queried BYOD, re-ran the export, built a no-reference BYOD package and reused it from a clean directory | 733.2 s runner wall (installs, 1.56 GB data and model downloads included); stage total 462.5 s | **PASS**: 21/21 code cells, execution counts 1–21, no errors; see below |
| 2026-09-28 | `d58ab28` / blob `284adf543e21` (executed file's 35 cell sources and IDs equal this blob; no `# @param` value changed) | Google Colab, fresh **Tesla T4** (reported by the preflight cell); isolated `uv` Python 3.12.12 environment from the carried hash-pinned lock | Default `Run all`; every optional toggle left at its default (off). Code cells executed in order 1–14; `reload` raised; export and the optional search/BYOD cells did not run | Not recorded (Colab saved no per-cell timings); ASR 52.8 s for 1,715.46 s of audio | **FAILED at `reload`** — see below |

### Maintainer-supplied Colab execution of `ad42d18` — 2026-09-28

- **File:** [`execution-evidence/2026-09-28/DIMER_Filipino_Audio_Archive_Search_Capstone_ad42d18_colab.ipynb`](execution-evidence/2026-09-28/DIMER_Filipino_Audio_Archive_Search_Capstone_ad42d18_colab.ipynb),
  SHA-256 `eef0cf97c3f0ec54235011ad24aafc9a0f89fa317bec3991d9bcc8a64e7881d4` (5,052,980 bytes),
  stored byte for byte.
- **Source match:** all 35 cells have the IDs, order and source of blob `47da9829…`, the
  notebook on `main` at the merge commit `ad42d18` and at `5664f61`. No toggle was changed.
- **Runtime:** fresh Colab Tesla T4; the stages ran in the notebook's `uv` CPython 3.12.12
  venv from the carried lock.
- **Per-stage seconds / peak allocated GPU memory:** prepare 103.3 / none; asr 92.2 / 1.61 GiB;
  index 26.7 / 1.17 GiB; evaluate 98.1 / 1.16 GiB; activity 134.5 / 1.16 GiB; reload 28.8 /
  1.61 GiB.
- **Results (engineering preview):** ASR dev WER 0.140 / CER 0.036, test WER 0.136 / CER 0.042,
  real-time factor 0.032. Test anchor hit@5: BM25 0.975, dense and rerank 1.000 (reference and
  ASR). Paired latency: mean rerank time per query 0.571 → 1.038 s (reference) and 0.551 →
  1.102 s (ASR) from depth 10 to 20. Failure tracing: 0 candidate misses, 0 reranker demotions.
- **Reload:** `passed: true`, 4 queries replayed, 4 documents re-embedded, 3 clips
  retranscribed, tolerances unchanged. This is the step that failed on Colab at `d58ab28`.
- **Cross-host check:** every stage-output digest in this run's `run_summary.json` equals the
  Kaggle run of `5664f61` below (transcripts, embeddings, rankings, evaluation, reload
  verification); only the timing files differ. The Colab and Kaggle T4 hosts produced
  byte-identical model outputs.
- **Evidence boundary:** Saved outputs were inspected; execution was not independently
  repeated.

| Journey | Verdict |
|---|---|
| Setup, preflight, locked install (Colab T4) | Pass |
| Prepare, ASR, index, evaluate (six systems) | Pass; outputs byte-identical to the Kaggle run |
| Failure tracing (F3) | Ran; neither category occurred |
| Development depth activity with paired latency (F2) | Pass |
| Fresh-process reload | **Pass** on Colab |
| Export (`report`) | Pass |
| Free-form search, BYOD | Not assessed in this run (toggles off by default); covered by the Kaggle run below |

### Kaggle execution of `5664f61` — 2026-09-28

Run by the agent on the maintainer's Kaggle account with the workspace serial
test-suite executor, not supplied by the maintainer.

- **File:** [`execution-evidence/2026-09-28/DIMER_Filipino_Audio_Archive_Search_Capstone_5664f61_kaggle.ipynb`](execution-evidence/2026-09-28/DIMER_Filipino_Audio_Archive_Search_Capstone_5664f61_kaggle.ipynb),
  SHA-256 `b6fdbc398c4bd1c83cf35fdde2e2c2509c5335a499f8f9b3211aed4e69713aa2` (18,570,318 bytes;
  larger than the `d58ab28` record because the search and BYOD cells embed 11 more audio
  players), stored byte for byte. Executor summary:
  [`execution-evidence/2026-09-28/DIMER_Filipino_Audio_Archive_Search_Capstone_5664f61_kaggle_run_summary.json`](execution-evidence/2026-09-28/DIMER_Filipino_Audio_Archive_Search_Capstone_5664f61_kaggle_run_summary.json); applied source diff:
  [`execution-evidence/2026-09-28/DIMER_Filipino_Audio_Archive_Search_Capstone_5664f61_kaggle_harness.diff`](execution-evidence/2026-09-28/DIMER_Filipino_Audio_Archive_Search_Capstone_5664f61_kaggle_harness.diff).
- **Source match:** the executor fetched the notebook from GitHub at `5664f61` and asserted
  blob `47da9829…` before running. Cells 1–35 keep the IDs, order and source of that blob,
  except four toggle cells, each edited only after its committed source matched a recorded
  SHA-256:
  `PLAY_COMPETITORS = True`; `RUN_SEARCH = True`; `RUN_BYOD = True` with
  `BYOD_MANIFEST = next(Path('/kaggle/input').rglob('manifest.json'))`; and
  `BYOD_QUERY = 'Anong mga dagat ang nakapalibot sa Turkey?'`. Three harness cells
  (`kaggleharness1`–`3`, tagged `kaggle-harness`) were appended; they are executed evidence,
  not notebook source. `DOWNLOAD_RESULTS` / `DOWNLOAD_BYOD` stayed `False`.
- **BYOD input:** three unmodified FLEURS `fil_ph` **train**-split WAVs at the pinned revision
  `70bb2e84` (CC-BY-4.0), attached as a private Kaggle dataset. The default archive indexes only
  validation and test clips, so the BYOD archive shares no recording with it. Two records carried
  their FLEURS transcript as `reference`; a second manifest carried none. These are public
  licensed stand-ins; no maintainer-owned or field recording has been tested.
- **Runtime:** Kaggle GPU image `sha256:37c64f7d…`, Linux 6.12.90, kernel Python 3.12.13; the
  preflight saw `Tesla T4` twice and 1,108.7 GiB free on `/`; `/content` was creatable. The stages
  ran in the notebook's own `uv` venv: torch 2.11.0 (CUDA 13.0), transformers 4.57.6, numpy 2.5.3,
  pyarrow 23.0.1, soundfile 0.13.1 (`environment.json`).
- **Per-stage seconds / peak allocated GPU memory:** prepare 119.6 / none; asr 81.6 / 1.61 GiB;
  index 18.7 / 1.17 GiB; evaluate 88.9 / 1.16 GiB; activity 126.5 / 1.16 GiB; reload 27.1 /
  1.61 GiB. Well inside the unverified 60-minute / 12-GiB targets.
- **Results (engineering preview):** identical to the `d58ab28` Colab run where comparable. ASR
  dev WER 0.140 / CER 0.036, test WER 0.136 / CER 0.042, real-time factor 0.031. Test anchor
  hit@5: reference and ASR BM25 0.975; dense and rerank 1.000. Paired latency (20 dev queries,
  same process): mean rerank time per query 0.519 → 1.032 s (reference) and 0.518 → 1.031 s (ASR)
  from depth 10 to 20. Failure tracing: 0 candidate misses and 0 reranker demotions in 40
  evaluation queries, so `PLAY_COMPETITORS` had no competitor to play.
- **Reload:** `verification.json` shows `passed: true`, `queries_replayed: 4`,
  `documents_reembedded: 4`, `clips_retranscribed: 3`, `replay_batch: "first original embedding
  batch, identical composition"`, `atol` 1e-5, `rtol` 1e-4 (unchanged). The `d58ab28` failure is
  fixed on hosted hardware.
- **BYOD:** 3 recordings, references for 2; ASR `measured_on_supplied_references`, WER 0.138 /
  CER 0.040 on those 2 (Whisper heard "Turkey" as "perky" and "Aegean" as "egan"); retrieval
  `not_measurable`; bundle `verification.json` passed (3 re-embedded, 3 retranscribed, probe
  replayed). Both BYOD queries returned only the three BYOD `doc_id`s with hash-matched audio.
  The second query created no new `byod/<uuid>` folder. The no-reference build reported
  `asr_status` and `retrieval_status` `not_measurable` and verified. Its `search_index.zip`,
  extracted to an empty directory, ran `search_archive.py` with the venv Python (exit 0) and
  ranked `fleurs_train_fish` first (score 0.9998) for "Bakit namamatay ang mga isda?".
- **Export:** `report` passed in the notebook's export cell and again in the harness re-run;
  `results.zip` lists 56 members. Free-form search showed five transcripts, each with playback
  after its SHA-256 matched.
- **Figures:** `prepare.png`, the per-recording WER histogram and `evaluate.png` rendered and
  were inspected.
- **Repeat:** an earlier direct Kaggle kernel of the same edited source
  (`dimer-fil-audio-capstone-5664f61-verify` v1, same day) passed every stage. Kaggle kept none of
  its executed outputs, so it is not recorded here, but its stage-output digests equal this run's
  for every model output (transcripts, embeddings, rankings, reload verification). Only timing files
  differ.
- **Evidence boundary:** Saved outputs were inspected. The recorded run was not independently
  repeated by a human; the repeat above was agent-run, and its executed notebook was not retained.

| Journey | Verdict |
|---|---|
| Setup, preflight, locked install (Kaggle T4) | Pass |
| Prepare, ASR, index, evaluate (six systems) | Pass; metrics equal the `d58ab28` run |
| Failure tracing with competing evidence (F3) | Ran; neither category occurred, so there was no competitor audio to play |
| Development depth activity with paired latency (F2) | Pass |
| Fresh-process reload | **Pass** (fix confirmed) |
| Export and export re-run | Pass |
| Free-form search with hash-checked playback | Pass |
| BYOD build with references, query, re-query without rebuild | Pass (FLEURS stand-in audio) |
| BYOD build without references (`not_measurable`) | Pass (harness) |
| Search package reuse from a clean directory | Pass (harness) |
| Colab-only download buttons; opening from GitHub in Colab | Not assessed in this run |

Open: human relevance review (0 of 7,200 judged); a BYOD run on the maintainer's own or field
recordings; a Colab run of this head if Colab-specific behaviour must be qualified.

### Maintainer-supplied Colab execution of `d58ab28` — 2026-09-28

- **File:** [`execution-evidence/2026-09-28/DIMER_Filipino_Audio_Archive_Search_Capstone_d58ab28.ipynb`](execution-evidence/2026-09-28/DIMER_Filipino_Audio_Archive_Search_Capstone_d58ab28.ipynb),
  SHA-256 `7bf7717f0e310a6601f02a8f9021c18165bf3aabf2baa1ba4a3d007ea0594834`, stored byte for byte.
- **Source match:** all 35 cells have the IDs, order and source of blob `284adf543e21`;
  the carrier cell kept its collapsed (`cellView: form`) metadata.
- **Runtime:** Tesla T4; `uv` CPython 3.12.12 virtual environment. Peak GPU memory and
  disk use are in the stage receipts, which were not exported because the run stopped
  before `report`.
- **Results (engineering preview):** 120 recordings (40 dev / 80 test); 60 queries;
  7,200 unjudged cells. ASR: dev WER 0.140 / CER 0.036, test WER 0.136 / CER 0.042;
  real-time factor 0.031 (model load excluded). Test anchor hit@5: reference BM25 0.975,
  dense 1.000, rerank 1.000; ASR BM25 0.975, dense 1.000, rerank 1.000. Paired
  ASR-rerank minus reference-rerank difference 0.0 (95% interval 0.0–0.0, 40 families).
  Activity (20 dev queries, same process): anchor-in-candidates 1.000 → 1.000
  (reference) and 0.950 → 0.950 (ASR) from depth 10 to 20; mean rerank time per query
  0.548 → 0.988 s (reference) and 0.538 → 1.058 s (ASR). Failure tracing: 0 candidate
  misses and 0 reranker demotions among 40 evaluation queries.
- **Failure:** `reload` re-embedded 3 documents and compared them with the index at
  `atol=1e-5, rtol=1e-4`; 2,737 of 3,072 elements differed, by up to 4.2e-4.
  **Cause:** Qwen3 embeddings run in float16 in batches of 4 with left padding, so a
  vector depends on its batch's composition; the index embedded documents 0–3 together
  and reload embedded 0–2. The evaluation queries had the same mismatch (batches of 4,
  replayed as 3). **Fixed in** the commit that follows this record: reload (and BYOD
  verification) replay exactly the first original embedding batch; tolerances are
  unchanged. A regression test with a batch-sensitive stand-in reproduces the failure
  on the old code and passes on the fix.
- **Evidence boundary:** Saved outputs were inspected; execution was not independently
  repeated. This run does not qualify the fix, the export, the free-form search or BYOD.

| Journey | Verdict |
|---|---|
| Setup, preflight, locked install | Pass |
| Prepare, ASR, index, evaluate (six systems) | Pass |
| Failure tracing with competing evidence (F3) | Ran; neither category occurred |
| Development depth activity with paired latency (F2) | Pass |
| Fresh-process reload | **Fail** (fixed; needs a new hosted run) |
| Export, free-form search, BYOD build and re-query | Not assessed in this run (not reached; toggles off) |

Open: a hosted T4 run of the fixed head with the optional journeys enabled; human
relevance review.
