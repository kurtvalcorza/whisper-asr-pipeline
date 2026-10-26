# Release verification

`tutorials/whisper_asr_colab.ipynb` (`TASK-INFERENCE`) and `tutorials/whisper_asr_finetune_colab.ipynb`
(`E2E`) are each a **release candidate** until the exact notebook revision has executed
top-to-bottom in a clean supported runtime. Unit tests, JSON
validation, code-cell compilation, and `tools/validate_release_assets.py` are necessary
checks but are **not** runtime evidence under DIMER Notebook Specification 2.5. This file is
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
| 2026-10-25 | `7116f71` / `5e302edc` (PR #20 head; `whisper_asr_finetune_colab.ipynb`; historical wording, notebook commit `5ed1812`; confirming run) | Colab CLI sequential execution, fresh Colab Tesla T4 VM (`colab exec -f` on a fresh session; fresh VM, empty Hugging Face cache, no repository checkout) — not a browser `Run all`; no execution counts, order from `exec.log` | Default path, every form field at its default, no overrides; notebook bytes fetched at the full commit SHA `7116f719cc0899ffc2e05df47114192f6c9d8ae3`, git blob SHA-1 `5e302edcc0980ad73541c2a8546c7d50eb0d3e95` checked, input unchanged; captured executed-notebook SHA-256 `b11e4f374f27f0b1bc44f5fe97ac6f870c83ddd2a421cec17e2c8dd0e07ed934` equals `run_summary.json` `executed_sha256` | 769.1 s | **PASSED in one pass, no restart, 0 errors** — 13/13 code cells (`exec.log`); isolated environment 61 s (70 locked packages); `PolyAI/minds14` en-US, 400 / 100 clips (5 over 30 s dropped); baseline WER 0.4211 (82 substitutions, 424 insertions, 12 deletions, 1,230 reference words) in 37.0 s; two epochs (100 optimizer steps) in 512.0 s, train loss 0.5306 → 0.2884, validation loss 0.5106 → 0.4683, peak 4.48 GiB; adapted WER 0.2528 (Δ −0.1683; 87 / 137 / 87 edits); 3 held-out transcripts also in training; reloaded WER 0.2585, agreement 99/100. Drift against the prose's recorded values (recorded, not fixed, per the 2026-10-26 ruling): adapted WER 0.2528 and its edit counts; train and validation losses; training 512.0 s. Evidence in `docs/execution-evidence/2026-10-25-7116f71/` |
| 2026-10-25 | `7116f71` / `10bd35ec` (PR #20 head; `whisper_asr_colab.ipynb`; historical wording, notebook commit `5ed1812`; confirming run) | Colab CLI sequential execution, fresh Colab Tesla T4 VM (`colab exec -f` on a fresh session; fresh VM, empty Hugging Face cache, no repository checkout) — not a browser `Run all`; no execution counts, order from `exec.log` | Default path, every form field at its default, no overrides; notebook bytes fetched at the full commit SHA `7116f719cc0899ffc2e05df47114192f6c9d8ae3`, git blob SHA-1 `10bd35ec3e5a485ee6879952bf8fd6525d5df0dc` checked, input unchanged; captured executed-notebook SHA-256 `4d3220d35700eed12c8f4ca566b69dbef356aa7b474fa86ac2a0f02492f8e432` equals `run_summary.json` `executed_sha256` | 123.5 s | **PASSED in one pass, no restart, 0 errors** — 12/12 code cells (`exec.log`); isolated environment 66 s (67 locked packages); LibriSpeech dummy utterance 1272-128104-0000 (5.86 s) → `Mr. Quilter is the apostle of the middle classes, and we are glad to welcome his gospel.`; WER 0.058823529411764705 against the empty-transcript baseline 1.0; silence activity wrote `Thank you.` (2 words). No drift against the prose's recorded values. Evidence in `docs/execution-evidence/2026-10-25-7116f71/` |
| 2026-10-25 | `e162ef6` (PR #20 head) / blob `472225cff86b` of `tutorials/whisper_asr_finetune_colab.ipynb` (prose kept true across runs, notebook commit `72ee88c`), fetched at the full SHA `e162ef627caab84a3ff6fefa1afec392c396ca47` and blob-verified before the session; captured executed-notebook SHA-256 equals `run_summary.json` `executed_sha256`; evidence `docs/execution-evidence/2026-10-25-e162ef6/` — SHA-256 captured → committed: `whisper_asr_finetune_colab_e162ef6_colab-cli-t4.ipynb` `a19ada6491e7960a4d5c9030004614d1d8f546a819bb0e4c97c2ed4e722a7584` → unchanged (no local path); `whisper_asr_finetune_colab_e162ef6_colab-cli-t4_exec.log` `9991d730f127bffb2970f7a6bde817b1cfe227c64cbef333f69abb097badf7e4` → `dd08484e63378639091e666c88eef720caf047a7f2e3560dad82a0b3e5a67416` (2 local-path replacements); `whisper_asr_finetune_colab_e162ef6_colab-cli-t4_run_summary.json` `22a6ffaca1f9c69945bb7554307d9e8b8e22b063d3c0c2426fff1d9ac04d582c` → unchanged (no local path) | Google Colab CLI sequential execution on a fresh Colab **Tesla T4** session (`colab exec -f`); isolated `uv` Python 3.12.12 environment built from the carried hash lock (70 packages, setup 56 s); kernel Python 3.13.16. Not a browser Run all; order from the CLI's `Executing cell k/N` log (1/13–13/13) | Default path only, no overrides, input unchanged: `PolyAI/minds14` `en-US`, 400 train / 100 held-out clips (`split_digest` `1d7ec0f47f34c578`; 5 clips over 30 s dropped; 3 held-out transcripts also in train), LoRA on q/v (6,553,600 trainable parameters, 0.804%), 2 epochs × 50 optimizer steps, export and fresh reload. BYOD not exercised | 825.9 s session wall (training 562.6 s, baseline evaluation 37.5 s); peak GPU 4.48 GiB | **PASS — one pass, no restart, 0 errors, 13/13 code cells.** Baseline corpus WER 0.4211 (S 82, I 424, D 12, N 1230) → adapted 0.2943 (S 92, I 239, D 31), delta −0.1268, verdict `sample-sanity`; 100/100 transcripts changed; reloaded WER 0.3024, reload agreement 97/100; weight probe on `model.decoder.layers.0.self_attn.q_proj`: merge error 1.2e-4 against an adapter delta of 1.4e-2 (tolerance 1e-3); train loss 0.5077 → 0.2817, validation loss 0.498 → 0.5329 (rose in epoch 2). Prose check (notebook not edited): the Section 9 answer "WER fell from 0.4211 to 0.3707 and to 0.3553 … about five to seven points" does not cover this run (adapted **0.2943**, about 12.7 points); every other quoted figure holds (baseline 0.4211 and 424 insertions; the loss values are labelled per run; the validation-loss direction matches the first run's; session wall inside about 13 to 15 minutes; reload agreement at least 95 of 100) |
| 2026-10-25 | `e162ef6` (PR #20 head) / blob `acd037962a09` of `tutorials/whisper_asr_colab.ipynb` (prose kept true across runs, notebook commit `72ee88c`), fetched at the full SHA `e162ef627caab84a3ff6fefa1afec392c396ca47` and blob-verified before the session; captured executed-notebook SHA-256 equals `run_summary.json` `executed_sha256`; evidence `docs/execution-evidence/2026-10-25-e162ef6/` — SHA-256 captured → committed: `whisper_asr_colab_e162ef6_colab-cli-t4.ipynb` `44d3d2bb317eb2a0e0e11ec63aef3245c05baad72000129535fde67dbb111a30` → unchanged (no local path); `whisper_asr_colab_e162ef6_colab-cli-t4_exec.log` `43918059ba488161566cd19f1799ab811ffea0f8ec76023de3053ebe22ed669c` → `8688acadc82087d98c97e8fdd98df5733fd4f096615b9975965e6fdbb195c695` (2 local-path replacements); `whisper_asr_colab_e162ef6_colab-cli-t4_run_summary.json` `c307c07590b1129795bf215238b91cf96913ca2a92df2e64cd109d903d48b01d` → unchanged (no local path) | Google Colab CLI sequential execution on a fresh Colab **Tesla T4** session (`colab exec -f`); isolated `uv` Python 3.12.12 environment built from the carried hash lock (67 packages, setup 65 s); kernel Python 3.13.16. Not a browser Run all; order from the CLI's `Executing cell k/N` log (1/12–12/12) | Default path only, no overrides, input unchanged: the pinned LibriSpeech dummy utterance `1272-128104-0000` (5.86 s), `transcribe`, `en`; the oversized-chunk probe rejected; the Section 9 activity at its default (`silence`, 5 s). BYOD not exercised | 124.5 s session wall | **PASS — one pass, no restart, 0 errors, 12/12 code cells.** Transcript "Mr. Quilter is the apostle of the middle classes, and we are glad to welcome his gospel."; `word_error_rate` 0.0588 (1/17) beside the empty-transcript baseline 1.0, verdict `sample-sanity`; silence activity wrote 2 words. Prose check: every quoted figure holds (WER 0.0588 in all recorded Colab T4 runs; about two minutes on a Colab T4) |
| 2026-10-25 | `bca2542` (PR #20 head) / blob `f74abf5cb565` of `tutorials/whisper_asr_finetune_colab.ipynb` (tier C review fixes generated at `4085772`), downloaded from GitHub at the PR head and blob-verified before the session; executed file `docs/execution-evidence/2026-10-25/whisper_asr_finetune_colab_bca2542_colab-cli-t4.ipynb`, sha256 `3d11c5e00f2f3521d040e145f6c0f6b7c29e88e8e646cec4240994dca7d21411` (with `…_run_summary.json` sha256 `0ef318aa9a91…` and `…_exec.log` sha256 `fd37c79d9764…` committed, `24781238189e…` captured; two local run-directory paths redacted to `<local-run-dir>`) | Google Colab CLI sequential execution on a fresh Colab **Tesla T4** session via the workspace `colab-cli-serial-test-suite` (`colab new --gpu T4`, `colab exec -f`, `colab stop`); isolated `uv` Python 3.12.12 environment built from the carried hash lock (70 packages, setup 65 s); kernel Python 3.13.16; torch 2.6.0+cu124, transformers 4.52.1, peft 0.17.1. Not a browser Run all; order from the CLI's `Executing cell k/N` log (1/13–13/13) | Default path only, no overrides, input unchanged: all 13 code cells ran in order in one kernel. `PolyAI/minds14` `en-US` at revision `40ce77cb32a3`, 400 train / 100 held-out clips (`split_digest` `1d7ec0f47f34c578`; 5 clips over 30 s dropped before the split; 3 held-out transcripts also occur in train), LoRA on q/v (6,553,600 trainable parameters, 0.804%), 2 epochs × 50 optimizer steps, export and fresh reload. Cells 4–5 print nothing by design. BYOD (`USE_BYOD`) and upload not exercised | 766.1 s session wall (training 512.4 s, baseline evaluation 35.9 s); peak GPU 4.48 GiB | **PASS — one pass, no restart, 0 errors, 13/13 code cells.** Device `cuda:0`, snapshot from the local pinned copy (12 files verified). Baseline corpus WER 0.4211 (S 82, I 424, D 12, N 1230) → adapted 0.3553 (S 80, I 333, D 24), delta −0.0659, verdict `sample-sanity`; 100/100 transcripts changed; reloaded WER 0.3553, reload agreement 100/100; weight probe on `model.decoder.layers.0.self_attn.q_proj`: merge error 1.2e-4 against an adapter delta of 1.9e-2 (tolerance 1e-3); train loss 0.5054 → 0.2793, validation loss 0.5267 → 0.5084. `outputs/whisper-asr-lora-adapter.zip` 12.5 MiB, sha256 `482eb6f9436a…`; printed `adapter_sha256` `7a2ca2663e4b…`. Prose check (REL13, notebook not edited): the worked answers quote the 2026-10-10 run of blob `5f91ee7d`, and this run printed different training and adaptation numbers: train loss 0.5054 → 0.2793 (quoted 0.511 → 0.2832); validation loss 0.5267 → 0.5084, which **fell** in epoch 2 (quoted 0.4814 → 0.5016, and the Section 9 and 10 notes that validation loss can rise in epoch 2 and that the epoch-1 loss was lower, so loss alone would favour one epoch); adapted WER 0.3553, delta −0.0659 (quoted 0.4211 → 0.3707, "about five points"); training 512.4 s and 766.1 s session wall (quoted 595.1 s and 861.8 s for `5f91ee7d`). The baseline (WER 0.4211, S 82 / I 424 / D 12 / N 1230), the split, the LoRA parameter counts and the 4.48 GiB peak equal the quoted values. Status stays **Candidate** |
| 2026-10-25 | `bca2542` (PR #20 head) / blob `413147082a7b` of `tutorials/whisper_asr_colab.ipynb` (tier C review fixes generated at `4085772`), downloaded from GitHub at the PR head and blob-verified before the session; executed file `docs/execution-evidence/2026-10-25/whisper_asr_colab_bca2542_colab-cli-t4.ipynb`, sha256 `15e722d9d5862c478483ff5b582f1912aa5efeb1cc6e399437b90ed277ad9eea` (with `…_run_summary.json` sha256 `3fcce1b65df8…` and `…_exec.log` sha256 `1cff4016d9de…` committed, `32e8354b12ca…` captured; two local run-directory paths redacted to `<local-run-dir>`) | Google Colab CLI sequential execution on a fresh Colab **Tesla T4** session via the workspace `colab-cli-serial-test-suite`; isolated `uv` Python 3.12.12 environment from the carried hash lock (67 packages, setup 57 s); kernel Python 3.13.16; torch 2.6.0+cu124, transformers 4.52.1. Not a browser Run all; order from the CLI's `Executing cell k/N` log (1/12–12/12) | Default sample path only, no overrides, input unchanged: all 12 code cells ran in order in one kernel, including the input-validation probe and the silence activity. Cells 4–5 print nothing by design. BYOD (`BYOD_PATH` / upload) and the noise activity not exercised | 117.3 s session wall | **PASS — one pass, no restart, 0 errors, 12/12 code cells.** Device `cuda:0`; 12 snapshot files verified; transcript "Mr. Quilter is the apostle of the middle classes, and we are glad to welcome his gospel."; WER 0.0588 (1/17, `Mr.` vs `MISTER`) against 1.0 for the empty-transcript baseline, verdict `sample-sanity`; oversized-chunk probe rejected; silence activity wrote 2 words ("Thank you."). Transcript and WER equal the 2026-10-10 row; every number the notebook's prose quotes from a recorded run equals this run's output (timings are labelled as other runs'). Status stays **Candidate** |
| 2026-10-10 | `9a9c39f` / blob `5f91ee7df26a` of `tutorials/whisper_asr_finetune_colab.ipynb`, downloaded from GitHub at the PR head and blob-verified before the session; executed file `docs/execution-evidence/2026-10-10/whisper_asr_finetune_colab_9a9c39f_colab-cli-t4.ipynb`, sha256 `53307c09e89a23ec29f204063d316d2cd12ff15a61129deeb6bad652b527b535` (with `…_run_summary.json` sha256 `480a1691b00d…` and `…_exec.log` sha256 `2253f15e46c0…`) | Google Colab CLI 0.7.4 sequential execution on a fresh Colab **Tesla T4** session via the workspace `colab-cli-serial-test-suite` (`colab new --gpu T4`, `colab exec -f`, `colab stop`); isolated `uv` Python 3.12.12 environment built from the carried hash lock (70 packages, setup 62 s); kernel Python 3.13.15. Not a browser Run all; the CLI records no execution counts, so order is evidenced by its `Executing cell k/N` log (1/13–13/13) | Default path only: all 13 code cells ran in order in one kernel. `PolyAI/minds14` `en-US` at revision `40ce77cb32a3`, 400 train / 100 held-out clips (`split_digest` `1d7ec0f47f34c578`; 5 clips over 30 s dropped before the split; 3 held-out transcripts also occur in train), LoRA on q/v (6,553,600 trainable parameters, 0.804%), 2 epochs × 50 optimizer steps, export and fresh reload. Cells 4–5 (carried module definitions) print nothing by design. BYOD (`USE_BYOD`) and upload not exercised | 861.8 s session wall (training 595.1 s, baseline evaluation 40.1 s); peak GPU 4.48 GiB | **PASS — one pass, no restart, 0 errors, 13/13 code cells.** Device `cuda:0`, snapshot from the local pinned copy. Baseline corpus WER 0.4211 (S 82, I 424, D 12, N 1230) → adapted 0.3707 (S 81, I 346, D 29), delta −0.0504, verdict `sample-sanity`; 100/100 transcripts changed; reloaded WER 0.3707, reload agreement 99/100; weight probe on `model.decoder.layers.0.self_attn.q_proj`: merge error 1.2e-4 against an adapter delta of 1.8e-2; train loss 0.511 → 0.2832, validation loss 0.4814 → 0.5016 (mild overfitting in epoch 2; WER still improved). `outputs/whisper-asr-lora-adapter.zip` 12.5 MiB, sha256 `2904999fb211…`; printed `adapter_sha256` `061ee9739300…`. No worked numeric answer exists for this blob; against the 2026-09-11 P100 row (different split: no 30 s drop then) the direction is the same (adapted better than baseline), with lower absolute WERs here. Status stays **Candidate** |
| 2026-10-10 | `9a9c39f` / blob `75cc521be949` of `tutorials/whisper_asr_colab.ipynb`, downloaded from GitHub at the PR head and blob-verified before the session; executed file `docs/execution-evidence/2026-10-10/whisper_asr_colab_9a9c39f_colab-cli-t4.ipynb`, sha256 `93889874e3cb104822e1161d6381aa1942a6564eb203d42ac5d87b0e15c294c5` (with `…_run_summary.json` sha256 `24f8bb2268bc…` and `…_exec.log` sha256 `874d69908748…`) | Google Colab CLI 0.7.4 sequential execution on a fresh Colab **Tesla T4** session via the workspace `colab-cli-serial-test-suite`; isolated `uv` Python 3.12.12 environment from the carried hash lock (67 packages, setup 61 s); torch 2.6.0+cu124, transformers 4.52.1. Not a browser Run all; order from the CLI's `Executing cell k/N` log (1/12–12/12) | Default sample path only: all 12 code cells ran in order in one kernel, including the input-validation probe and the silence activity. Cells 4–5 (carried module definitions) print nothing by design. BYOD (`BYOD_PATH` / upload) and the noise activity not exercised | 120.4 s session wall | **PASS — one pass, no restart, 0 errors, 12/12 code cells.** Device `cuda:0`; transcript "Mr. Quilter is the apostle of the middle classes, and we are glad to welcome his gospel."; WER 0.0588 (1/17, `Mr.` vs `MISTER`) against 1.0 for the empty-transcript baseline, verdict `sample-sanity`; oversized-chunk probe rejected (`chunk_length_s must be between 1 and 30 seconds`); silence activity wrote 2 words ("Thank you."). Transcript and WER equal the 2026-09-11 rows. Status stays **Candidate** |
| 2026-09-14 | `b3a123e` / blob `818d2e556b44` of `tutorials/whisper_asr_finetune_colab.ipynb` | Kaggle kernel `kurtvalcorza/dimer-nb2-whisper-asr-finetune` v2 — GPU container assigned **Tesla T4** (driver 580.159.04), Python 3.12, torch 2.10.0+cu128; clean HF cache | Default LoRA finetune path, all 10 code cells: mind14 sample dataset, staged snapshot (26 files, 1622 MB), training, evaluation report, adapter bundle zip export and reload verification | 1002.0 s | **PASS** — 10/10 ok code cells executed cleanly (1 restart after install cell) |
| 2026-09-11 | `db4daf856e82` / blob `e916da1c9270` of `tutorials/whisper_asr_finetune_colab.ipynb` (the executed file was verified equal to this committed blob in-run; HEAD changes only learner-facing Prerequisites markdown content plus regenerated markdown-cell ID metadata; no code cell differs, verify with `git diff e916da1c9270 <head>:tutorials/whisper_asr_finetune_colab.ipynb`) | Kaggle kernel `kurtvalcorza/dimer-whisper-finetune-t4-verify` v1 — fresh GPU container assigned a **Tesla P100-PCIE-16GB** (driver 580.159.04), Python 3.12.13, Linux 6.12.90; committed notebook executed verbatim, cell by cell, by `run_nb.py` in a fresh interpreter under a clean HF cache (`models--openai--whisper-large-v3-turbo` and `datasets--PolyAI--minds14` are the only cache entries afterwards) | Default path, all 7 code cells: `en-US`, 400 train / 100 held-out clips (`split_digest` `686ef3a3d988…`), LoRA r=32 α=64 on q/v, 2 epochs × 50 measured optimizer steps, float16 autocast | 1617.7 s (236 s install, 109 s baseline, 1106.7 s training, 72 s adapted eval, 75 s export + reload); peak CUDA allocation 4.48 GiB | **PASS** — `repository_revision` equals the candidate commit; recorded versions torch 2.6.0+cu124, torchvision 0.21.0, torchaudio 2.6.0, transformers 4.52.1, peft 0.17.1, numpy 1.26.4, datasets 4.4.0, accelerate 1.3.0, soundfile 0.13.1 (all pins); baseline corpus WER 0.4387 → adapted 0.4066 → reloaded 0.4066, reload agreement 100/100, 100/100 transcripts moved toward the corpus's lowercase unpunctuated reference style; weight probe on `model.decoder.layers.0.self_attn.q_proj`: merge error 1.2e-4 (tolerance 1e-3) against an adapter delta of 1.5e-2; train loss 0.517 → 0.283 while validation loss 0.578 → 0.612 (mild overfitting in epoch 2; WER still improved); `outputs/whisper-asr-lora-adapter.zip` sha256 `d07dcffad2cb…`, `adapter_model.safetensors` sha256 `0005c56e7691…`, `outputs/whisper_asr_finetune_result.json` sha256 `e067b63793eb…`. The corpus's own references are noisy transcriptions, so the absolute WER measures agreement with them, not speech-recognition accuracy |
| 2026-09-11 | `ca4c5ae82a77` / blob `a5b13f9bbfe2` (the notebook blob at this revision; the two later commits change `tools/validate_release_assets.py` lint and documentation only) | Kaggle kernel `kurtvalcorza/dimer-whisper-asr-verify-v2` v5 — fresh CPU container, Python 3.12.13, Linux 6.12.90; committed notebook executed verbatim, cell by cell, by `run_nb.py` in a fresh interpreter (executed blob == committed blob, measured in-run); the Kaggle kernel itself pre-imports numpy 2.0.2 and Pillow 11.3.0, which the generalized stale-import guard of this revision would reject after the pinned install, hence the fresh interpreter | Default sample path, all 5 code cells, clean HF cache (`models--openai--whisper-large-v3-turbo` + the LibriSpeech dummy dataset are the only cache entries afterwards) | 328.9 s (256 s install, 35 s snapshot fetch + load, 34 s transcription) | **PASS** — `repository_revision` equals the candidate commit; recorded versions torch 2.6.0, torchvision 0.21.0, torchaudio 2.6.0, transformers 4.52.1, numpy 1.26.4, pandas 2.2.3, datasets 4.4.0, accelerate 1.3.0, soundfile 0.13.1 (all pins); transcript "Mr. Quilter is the apostle of the middle classes, and we are glad to welcome his gospel."; WER 0.0588 (1/17, `Mr.` vs `MISTER`); `outputs/whisper_asr_result.json` sha256 `e4a4cf21…fdc96` |
| 2026-09-11 | `8fba91c20b2b` / blob `c308c7b1d40e` (this revision; executed file verified equal to the committed blob) | Kaggle kernel `kurtvalcorza/dimer-whisper-asr-verify-v2` v4 — fresh CPU container, Python 3.12.13, Linux 6.12.90; committed notebook executed verbatim, cell by cell, by `run_nb.py` in a fresh interpreter (the Kaggle kernel itself pre-imports numpy 2.0.2, which the tutorial's stale-import guard correctly rejects after the pinned numpy 1.26.4 install — kernel v1 halted there by design; kernel v2 at `ab00a28` failed with `operator torchvision::nms does not exist` from the orphaned runtime torchvision, fixed by the `torchvision==0.21.0` pin in this revision) | Default sample path, all 5 code cells, clean HF cache (`models--openai--whisper-large-v3-turbo` and the LibriSpeech dummy dataset are the only cache entries afterwards) | 273.6 s (209 s install, 37 s snapshot fetch + load, 22 s transcription) | **PASS** — `repository_revision` equals the candidate commit; torch 2.6.0+cu124, transformers 4.52.1, accelerate 1.3.0, datasets 4.4.0, soundfile 0.13.1 (recorded), torchvision 0.21.0 (pinned; not in that run's recorded versions — inferred from the successful pinned install); transcript "Mr. Quilter is the apostle of the middle classes, and we are glad to welcome his gospel."; WER 0.0588 (1/17, `Mr.` vs reference `MISTER`) — identical to the local pre-flight; `outputs/whisper_asr_result.json` sha256 `3b6ead78…759de`; only warning: transformers' internal `inputs` FutureWarning |
| 2026-09-11 | working tree equal to `8fba91c` (torchvision pin) / blob `c308c7b1d40e` | Local WSL harness, CPU (`CUDA_VISIBLE_DEVICES=''`), torch 2.6.0+cu124, torchvision 0.21.0, transformers 4.52.1 (pins) | Default sample path, all 5 code cells, clean cache | 213.0 s | PASS — same transcript, WER 0.0588; `outputs/whisper_asr_result.json` sha256 `45763436…dfd47` |
| 2026-09-11 | blob `c308c7b1d40e` (this revision) | Local WSL harness, CPU (`CUDA_VISIBLE_DEVICES=''`), torch 2.6.0+cu124, torchvision 0.21.0, transformers 4.52.1, datasets 4.4.0, accelerate 1.3.0, soundfile 0.13.1 (pins) | Default sample path, all 5 code cells, clean cache | 206.9 s (174 s of it the 1.6 GB snapshot fetch) | PASS — `repository_revision` recorded; `openai/whisper-large-v3-turbo` acquired at the pinned revision into an empty cache (`models--openai--whisper-large-v3-turbo` only); transcript "Mr. Quilter is the apostle of the middle classes, and we are glad to welcome his gospel."; WER 0.0588 (1/17: `Mr.` vs `MISTER`); `outputs/whisper_asr_result.json` sha256 `8b786056…7ec290`; only warning: transformers' internal `inputs` FutureWarning |
| 2026-09-11 | pre-fix working tree of `0b76b2a` | Local WSL harness, CPU | Default sample path | — | FAILED at cell 5 — `ImportError: To support decoding audio data, please install 'torchcodec'` (`datasets==4.4.0` decodes audio through torchcodec, which is not pinned). Fixed: the sample is loaded with `Audio(decode=False)` and decoded by the pinned `soundfile`. The following PASS run also showed WER 0.176 because punctuation and case counted as errors; `word_error_rate` now applies basic normalization |
| 2026-09-11 | pre-fix working tree of `0b76b2a` | Local WSL harness, CPU | Default sample path | — | FAILED at cell 3 — the stale-import guard compared `torch.__version__` (`2.6.0+cu124`) with the distribution version (`2.6.0`) and raised the restart error whenever torch was already imported. Fixed: distribution metadata is compared with itself before/after installation |

## Current status

**Candidate (2026-10-26, final prose pass).** The current notebook blob of `tutorials/whisper_asr_finetune_colab.ipynb` is `d1f3be6e03cdbff10f73a59bb3dc801bcfe5c347` (commit `0c9714f`), regenerated from `38957c2` under Kurt's 2026-10-26 ruling to fix only the prose-rule items flagged on the confirming run of `5e302edc`: the Prerequisites timing no longer says 'budget about 13 to 15 minutes'; it lists all three recorded Colab T4 runs of this version (861.8 s, 766.1 s and 825.9 s, adding the omitted run of `472225cf`), and the CPU training time is labelled as an estimate. Markdown only; no code or output changed, and the drift recorded for that run below is kept, not fixed. **Its hosted Colab T4 run is pending (REL14);** the confirming run of `5e302edc` and the earlier records below stay as history.

**Candidate (2026-10-26, final prose pass).** The current notebook blob of `tutorials/whisper_asr_colab.ipynb` is `d9b28823c9668832538f8f5423bfe70c13a4c64f` (commit `0c9714f`), regenerated from `38957c2` under Kurt's 2026-10-26 ruling to fix only the prose-rule items flagged on the confirming run of `10bd35ec`: the Prerequisites clause 'and it has not been timed on CPU' is removed. Markdown only; no code or output changed, and the drift recorded for that run below is kept, not fixed. **Its hosted Colab T4 run is pending (REL14);** the confirming run of `10bd35ec` and the earlier records below stay as history.

**Historical wording (2026-10-26, code commit `b3ad7e6`, notebook commit `5ed1812`).** Previous blobs (confirming runs recorded): `tutorials/whisper_asr_finetune_colab.ipynb` `5e302edcc0980ad73541c2a8546c7d50eb0d3e95` and `tutorials/whisper_asr_colab.ipynb` `10bd35ec3e5a485ee6879952bf8fd6525d5df0dc`, both **Candidate**, regenerated under Kurt's 2026-10-26 ruling: every run-derived number in the prose names the hosted runs that printed it (fine-tune: Colab T4 of `5f91ee7d` 2026-10-10, `f74abf5c` and `472225cf` 2026-10-25; inference: Colab T4 of `75cc521b` 2026-10-10, `41314708` and `acd03796` 2026-10-25), with no expectation ranges, timing promises or run-status sentences; it resolves the fine-tune prose mismatch recorded for the `472225cf` run (adapted WER 0.2943, about 12.7 points). Markdown only. Both exact blobs (PR #20 head `7116f71`) **passed** their confirming hosted runs: Colab CLI sequential execution on fresh Colab Tesla T4 VMs, 2026-10-25, one pass each, no restart, 0 errors — fine-tune 13/13 code cells, 769.1 s, WER 0.4211 → 0.2528, reload agreement 99/100; inference 12/12 code cells, 123.5 s, WER 0.0588; evidence in `docs/execution-evidence/2026-10-25-7116f71/`. This PR's runs cover the default path of both notebooks only. Remaining gates: REL12 (a hosted BYOD run with one accepted and one refused input) and REL2 (a browser Run all). Runs of earlier blobs are history (REL14). The records below stay as history.

**Drift recorded, not fixed (2026-10-26, Kurt's 2026-10-26 "historical wording, then stop" ruling).** The notebook prose of the fine-tune blob `5e302edc` and the inference blob `10bd35ec` labels every run-derived number with the named recorded runs that printed it, so these differences are not prose errors. This confirming run printed these values outside the recorded values and spans the prose names. The inference run printed no value outside its recorded values. The notebooks are not edited:

- Fine-tune Section 9 worked answer: adapted WER 0.2528 (about 16.8 points below the 0.4211 baseline); the prose records 0.3707, 0.3553 and 0.2943 (about five, seven and thirteen points) for the three named Colab T4 runs, and a spread of up to 7.6 points.
- Fine-tune Section 9 worked answer: adapted edits 87 substitutions, 137 insertions, 87 deletions; the prose records insertions 346, 333, 239 and deletions 29, 24, 31 (substitutions 81, 80, 92). Insertions still fell most.
- Fine-tune Section 8 worked answer: train loss 0.5306 → 0.2884 and validation loss 0.5106 → 0.4683 (fell in epoch 2, as in the run of `f74abf5c`); the prose records train loss 0.5054–0.511 → 0.2793–0.2832 and an epoch-2 validation loss of 0.5016, 0.5084 and 0.5329.
- Fine-tune Run all / Prerequisites: training took 512.0 s, just below the recorded 512.4–595.1 s (session wall 769.1 s lies inside the recorded 766.1–861.8 s).

Evidence files for the 2026-10-25 finetune run of blob `5e302edc`, copied from the run directory and redacted byte-wise (local run paths replaced by `<local-run-dir>`, the Colab CLI install path by `<colab-cli-tool-dir>`; no JSON re-serialised; the executed notebook's captured hash equals `run_summary.json` `executed_sha256`):

| File (under `docs/execution-evidence/2026-10-25-7116f71/`) | Captured SHA-256 | Committed SHA-256 |
|---|---|---|
| `whisper_asr_finetune_colab_7116f71_colab-cli-t4.ipynb` | `b11e4f374f27f0b1bc44f5fe97ac6f870c83ddd2a421cec17e2c8dd0e07ed934` | `b11e4f374f27f0b1bc44f5fe97ac6f870c83ddd2a421cec17e2c8dd0e07ed934` (unchanged, no local path) |
| `whisper_asr_finetune_colab_7116f71_colab-cli-t4_exec.log` | `013fbd54474e1788b55be12a99f8d58e31a79bae1620d804c980bd0721889387` | `af0e48727bae44a13368631f0fde784851c5982c7eb3acd8d93cd70a487b8652` (2 local-path replacements) |
| `whisper_asr_finetune_colab_7116f71_colab-cli-t4_run_summary.json` | `fc0c235ff6a721520d72442b9b56097b17b5328f82d199b431d9b6a0d5c64067` | `fc0c235ff6a721520d72442b9b56097b17b5328f82d199b431d9b6a0d5c64067` (unchanged, no local path) |

Evidence files for the 2026-10-25 asr run of blob `10bd35ec`, copied from the run directory and redacted byte-wise (local run paths replaced by `<local-run-dir>`, the Colab CLI install path by `<colab-cli-tool-dir>`; no JSON re-serialised; the executed notebook's captured hash equals `run_summary.json` `executed_sha256`):

| File (under `docs/execution-evidence/2026-10-25-7116f71/`) | Captured SHA-256 | Committed SHA-256 |
|---|---|---|
| `whisper_asr_colab_7116f71_colab-cli-t4.ipynb` | `4d3220d35700eed12c8f4ca566b69dbef356aa7b474fa86ac2a0f02492f8e432` | `4d3220d35700eed12c8f4ca566b69dbef356aa7b474fa86ac2a0f02492f8e432` (unchanged, no local path) |
| `whisper_asr_colab_7116f71_colab-cli-t4_exec.log` | `908e285644872b80dd9d51eeb33714f0b0c80142ea0918103618b9fd7f35da48` | `378c5c4ce7ffdbded2efeb990b0ce73149532dfeabee2ab9937cb0f9d916e1dc` (2 local-path replacements) |
| `whisper_asr_colab_7116f71_colab-cli-t4_run_summary.json` | `8d124db5e3dbe32d29e96ac76423231b05f3fa498f473279853c3c15b10929bf` | `8d124db5e3dbe32d29e96ac76423231b05f3fa498f473279853c3c15b10929bf` (unchanged, no local path) |

**Prose kept true across runs (2026-10-25, code commit `08a714a`, notebook commit `72ee88c`).** Previous blobs: `tutorials/whisper_asr_finetune_colab.ipynb` `472225cff86ba9d4d94e967e35e4a7bf6d5b4dd2` and `tutorials/whisper_asr_colab.ipynb` `acd037962a0978fec2d21d6574f202762a307ce0`, both **Candidate**; each ran its default path in one pass with no restart and 0 errors on a fresh Colab Tesla T4 on 2026-10-25 at PR #20 head `e162ef6` (first two rows of the table above: fine-tune 13/13 code cells, 825.9 s, WER 0.4211 → 0.2943, reload agreement 97/100; inference 12/12 code cells, 124.5 s, WER 0.0588). These runs cover the default path of both notebooks only; remaining gates REL12 (a hosted BYOD run) and REL2 (a browser `Run all`). Prose check of the fine-tune run: the Section 9 answer's adapted WER (0.3707 and 0.3553, "about five to seven points") does not cover this run's 0.2943 (about 12.7 points); the notebook is not edited by this record. Markdown only. Fine-tune: the worked answers now quote both recorded Colab T4 runs of this version (blob `5f91ee7d`, 2026-10-10, and blob `f74abf5c`, 2026-10-25): train loss 0.511 → 0.2832 and 0.5054 → 0.2793; validation loss rose in epoch 2 in the first (0.4814 → 0.5016) and fell in the second (0.5267 → 0.5084), so the claim that loss alone would favour one epoch is stated for the first run only; adapted WER 0.3707 and 0.3553 (about five to seven points); timing 861.8 s and 766.1 s. Inference: the session time is stated as about two minutes on a Colab T4 (120.4 s for `75cc521b`, 117.3 s for `41314708`) instead of "not timed yet", and WER 0.0588 is labelled with both runs. The records of earlier blobs below stay as history (REL14).

**Tier C review fixes, 2026-10-25:** `tutorials/whisper_asr_finetune_colab.ipynb` blob `f74abf5cb56495aa20174cabc28990d255e84705`, generated at `4085772` (T112-1..T112-5). It changes markdown, the use notice, `metadata.dimer` and the `NOTEBOOK_SOURCE` record only. Status **Candidate**; its default path ran in one pass on a fresh Colab T4 on 2026-10-25 (first row of the table above: WER 0.4211 → 0.3553, reload agreement 100/100). The worked answers quote the 2026-10-10 run of `5f91ee7d`, whose training numbers differ from this run's (listed in that row).

**Tier C review fixes, 2026-10-25:** `tutorials/whisper_asr_colab.ipynb` blob `413147082a7bdcfdd8e404b413cca60984b569c2`, generated at `4085772` (T111-1..T111-3). It changes markdown, the use notice, `metadata.dimer` and the `NOTEBOOK_SOURCE` record only. Status **Candidate**; its default sample path ran in one pass on a fresh Colab T4 on 2026-10-25 (second row of the table above: WER 0.0588, equal to the 2026-10-10 run). Both PR #20 runs cover the default paths only; BYOD of both notebooks and the noise activity were not exercised. Remaining gates: REL12 (a hosted BYOD run with one accepted and one refused input) and REL2 (a browser run of the interactive journeys). The runs described below are of earlier blobs and are history (REL14).

**2026-10-10 — one-pass hosted runs of both regenerated blobs are recorded.** `whisper_asr_colab.ipynb` (blob `75cc521be949`) and `whisper_asr_finetune_colab.ipynb` (blob `5f91ee7df26a`) at `9a9c39f` each ran once on a fresh Colab Tesla T4 through the Colab CLI 0.7.4 (sequential execution, not a browser Run all), with no restart and no error output (first two rows of the table above). That closes the one-pass default-path gate below for these blobs. Both notebooks stay **Candidate**: the remaining gates are reviewer confirmation of these runs against the blobs under review, the BYOD journeys of both notebooks, and the optional activity branches not exercised here (the noise activity of `whisper_asr_colab.ipynb`); promotion is an integrator decision.

**2026-10-05 — review fixes; no run of the then-current blobs was recorded at that date (the 2026-10-10 Colab T4 rows above cover them).** The 2026-10-05 reviews (`docs/reviews/2026-10-05-notebook-review/`) found that the recorded rows do not cover the standalone notebooks as they stand: every `whisper_asr_colab.ipynb` row executed the earlier, repository-installing notebook (blobs `a5b13f9b`, `c308c7b1`, 5 code cells) in a fresh-interpreter harness, and the one `whisper_asr_finetune_colab.ipynb` standalone row (2026-09-14, blob `818d2e55`, Kaggle T4, 1002.0 s) needed one manual restart after the install cell, which is not REL2 evidence, and recorded no metrics or versions. Both notebooks were then regenerated: an isolated, hash-locked `uv` environment replaces the in-kernel install and its restart guard (spec 2.2), the inference pins now include `datasets==4.4.0`, and the guided layer and the BYOD and evaluation fixes were added (details in the two `*_Fixes.md` files). **A one-pass hosted Run all of each regenerated blob, recorded with `restarted: false`, is the open gate.**

**2026-09-14 — `whisper_asr_finetune_colab.ipynb` was regenerated as a standalone NOTEBOOK_SPEC 2.0 notebook and run on Kaggle** (Tesla T4, 1002.0 s, 10/10 code cells after one restart, kernel `dimer-nb2-whisper-asr-finetune v2`). The task-inference standalone notebook was not run.

Before the standalone migration: a clean supported-class execution of the notebook blob at that revision is recorded in the rows above (Kaggle container, fresh interpreter, clean cache, pinned wheels, all stages of the default path, verbatim blob measured in-run, every version recorded); it supersedes the earlier rows, which remain as the audit trail of the review round. Static CI is green on the same branch. The registry status remains **Candidate** until a reviewer confirms the recorded run against the notebook blob under review and an integrator promotes it; promotion is not performed by the builder. The commit that adds a recorded-execution row changes documentation only; the executed source is the commit named in the row.
## Speech-recognition workshop notebook

`tutorials/DIMER_Whisper_Speech_Recognition_Workshop.ipynb` (`TASK-INFERENCE` / `WORKSHOP`, kind `WORKSHOP`, DIMER Notebook Specification 2.5 since the 2026-10-12 review fixes; 2.2 before) is a **Candidate**. It carries its reference module, runner, frozen sample manifest, model manifest and hash-pinned dependency lock, and installs them into an isolated `uv` Python 3.12.12 environment. Its design is `docs/speech-recognition-workshop-spec.md`.

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
| 2026-10-15 | `904cbe7` / blob `378259d9d0de` (the 2026-10-12 review-fix revision, PR #19 head), downloaded from GitHub at the PR head and blob-verified before the session; the executed file's 20 cell IDs and sources equal the blob, no toggle changed; executed file `docs/execution-evidence/2026-10-15/DIMER_Whisper_Speech_Recognition_Workshop_904cbe7_colab-cli-t4.ipynb`, sha256 `baca8eebd87f6f84831efa6c100fd8f1f2f150028ac3e3e5bb647835694118c6` (with `…_run_summary.json` sha256 `b6318d6d9831…` and `…_exec.log` sha256 `5a22e967eac2…`) | Colab CLI 0.7.4, fresh Colab **Tesla T4** session (reported by cell 1) via the workspace `colab-cli-serial-test-suite` (`colab new --gpu T4`, `colab exec -f`, `colab stop`); isolated `uv` Python 3.12.12 environment built from the carried hash-pinned lock. CLI sequential execution, not a browser Run all; the CLI records no execution counts, so order is evidenced by its `Executing cell k/N` log (1/10–10/10) | Default path: all 10 code cells ran in order in one kernel (cell 3, the bootstrap, prints nothing by design). Stages `prepare` (64 evaluation / 16 activity recordings, 597 s of audio), `model` (pinned snapshot verified), `evaluate`, `activity`, then report export. Optional 5/20 dB exploration and BYOD (including the new refused-input message) not run; forms and uploads not exercised | 212.7 s session wall; notebook-reported 169.4 s elapsed from setup (72.3 s bootstrap, 18.2 s model acquisition); inference 16.7 s for 494 s of evaluation audio | **PASSED 10/10**, no errors. Evaluate: WER 0.1981, CER 0.1646 (S 56, D 5, I 122, N 924), exact-match 0.453, 0 failed, 0 truncated, against 1.0 for the empty-transcript baseline. Activity at 10 dB SNR: clean WER 0.1850, noisy 0.2081; paired 0 improved / 12 unchanged / 4 worsened. Compared with the 2026-10-03 run of blob `fc8acbb6a3df`: every metric and count above is equal, as the 2026-10-12 fixes intend (default-path computation unchanged). Differences, each explained: the evaluate stage now shows **Fixed first example: listen.** with its own audio player before the lowest- and highest-WER players (the peer-review playback fix); the run directory hash (`79e4e2407330` → `0e6725c3df4e`) and stage directory names change because the carried `workshop.py` digest in `source.json` changed; report ZIPs reload-verified but their digests and sizes differ (evaluate sha256 `f9e39eafe595…`, 41,667 bytes, +41; activity sha256 `378005e6d2c4…`, 19,512 bytes, +26), consistent with the new `workshop.py` digest and run-specific paths and timings in the carried provenance (inner files not compared; the VM was released); timing differs run to run (elapsed 169.4 s vs 158.1 s, bootstrap 72.3 s vs 66.3 s, model acquisition 18.2 s vs 23.0 s, inference 16.7 s vs 17.3 s). Boundary: saved outputs inspected; peak GPU memory and disk not captured; the exploration and BYOD journeys (REL12, ASR2-m5) were not exercised and remain open. Status stays **Candidate** |

A fresh Colab T4 `Run all` of the committed blob is recorded in the first row above. The notebook stays **Candidate** until a reviewer confirms that run against the blob under review and an integrator promotes it; the optional branches remain unexercised.

**2026-10-02 notebook review fixes (ASR-M1..M3, ASR-m1..m5).** The review in `reviews/2026-10-02-notebook-review/` changed the notebook (prose, the `show_results` helper, and the carried `workshop.py`: BYOD file identity and named refusals, reference provenance in `summary.md`). The 2026-09-27 run above covers blob `56749100397d` only and is **not** evidence for the revised blob. Before promotion, a fresh Colab T4 run of the revised blob must cover the default path, the 5 or 20 dB exploration, and BYOD with labelled, unlabelled and one rejected input (REL12). Local checks for the fix were CPU-only with a stand-in transcriber and are not clean-runtime evidence.

**2026-10-03 source layout change.** The workshop's `CARRIED_FILES` literal in cell 3 was one 203,177-character line. `tools/split_workshop_carrier.py` rewrote it as parenthesised runs of short string pieces (at most 1,000 characters each), and the change is logged in `metadata.dimer.revisions`. Python joins the pieces back into the same text: the carried files, `CARRIED_HASHES`, the carried `source.json` and `generated_from` are unchanged, and no cell line is now longer than 2,000 characters (`python tools/split_workshop_carrier.py --check`). The notebook blob changes from `baa5be541c92` (the 2026-10-02 review-fix revision, which has no hosted run) to `fc8acbb6a3df`. A hosted re-run of the new blob passed on the default path (Colab CLI T4, 2026-10-03, last row of the table above); the exploration and BYOD runs required before promotion are still open. Status stays **Candidate**.

**2026-10-12 review fixes (ASR2).** The W22 review of blob `fc8acbb6a3df` found no Blocker or Major. The fixes, applied as exact-once anchored edits because the workshop generator is not in this repository (logged in `metadata.dimer.revisions`): a refused BYOD input now prints one `BYOD refused: ...` line and the carried runner exits 3, with no traceback and no `RuntimeError` (ASR2-m1); the notebook declares NOTEBOOK_SPEC 2.5 and `notebook_kind` `WORKSHOP`, carries the DOC17 use notice and cites in IEEE numbered style, and `tools/validate_release_assets.py` now pins `WORKSHOP_SPEC = "2.5"` and checks the notice and citation correspondence (ASR2-m2, owner decision A); the opening quotes the measured 158 s T4 run with its revision instead of saying no run exists (ASR2-m3); the paired improved/unchanged/worsened counts and the exact-match rate are defined (ASR2-m4, ASR2-S1); and the exploration cell prints the canonical 10 dB rates beside the chosen setting (ASR2-S2). The default-path computation is unchanged, but the blob changes, so the 2026-10-03 run is not evidence for it (REL14). A hosted run of the new blob is required: the default path, then in a browser the 5 dB exploration and BYOD with a labelled ZIP, an unlabelled ZIP and one refused input (REL12, ASR2-m5). Status stays **Candidate**.

**2026-10-15 hosted run of the review-fix revision.** Blob `378259d9d0de` (PR #19 head `904cbe7`) passed the default path on a fresh Colab CLI T4 session (last row of the table above): 10/10 code cells, no errors, every metric and count equal to the 2026-10-03 run of `fc8acbb6a3df`, and the fixed first example now has its own audio player in the evaluate stage. This run is CLI sequential execution, not a browser Run all; forms and uploads were not exercised. The 5 dB exploration and BYOD with a labelled ZIP, an unlabelled ZIP and one refused input (REL12, ASR2-m5) are still required before promotion. Status stays **Candidate**.

## Filipino Audio Archive Search capstone

Status: **Engineering Preview / Candidate.** Human relevance review is pending
(0 of 7,200 cells judged); scores below are nominated-anchor recovery diagnostics,
not benchmark Recall, MRR or nDCG.

**Query version note (2026-10-13).** Every hosted run below used query version
`ai-draft-v1` (paraphrases of each anchor sentence). The owner decision FAS-M1 = B replaced
it with `ai-draft-v2` (information-need drafts, pending native-speaker review), which changes
every retrieval number, and BYOD now converts non-16 kHz or stereo input. The runs below are
evidence for the previous query version and BYOD contract only, except the 2026-10-14 run of
`816b8dc`, which used `ai-draft-v2` and is recorded in the first row and the section below.

| Date (UTC) | Commit / notebook blob | Executor | Path exercised | Wall | Outcome |
|---|---|---|---|---|---|
| 2026-10-14 | `816b8dc` / blob `00cb2948c6e9` (downloaded from GitHub at the PR head and blob-verified before the session; the executed file's 38 cell IDs and sources equal the blob, no toggle changed) | Google Colab CLI 0.7.4 sequential execution on a fresh Colab **Tesla T4** session via the workspace `colab-cli-serial-test-suite` (`colab new --gpu T4`, `colab exec -f`, `colab stop`); isolated `uv` Python 3.12.12 environment from the carried lock; cells run in order in one kernel, not a browser Run all; order evidenced by the CLI's `Executing cell k/19` log (1/19–19/19; the CLI records no execution counts) | Default path, query version `ai-draft-v2`, every optional toggle off; forms and uploads not exercised (the BYOD cells printed `BYOD disabled (RUN_BYOD = False).`) | 452.6 s session wall; stages prepare 24.4 s, asr 91.1 s, index 22.9 s, evaluate 87.3 s, activity 124.4 s, reload 25.5 s; peak allocated GPU 1.61 GiB | **PASS**: 19/19 code cells, no errors; every stage and the report printed PASS, fresh-process reload passed (4 queries, 4 documents, 3 clips). Executed notebook: `execution-evidence/2026-10-14/DIMER_Filipino_Audio_Archive_Search_Capstone_816b8dc_colab-cli-t4.ipynb`, SHA-256 `8c7e4696f46c69a1574df3b66967ca743b70152ed1911374419573c9792f4b74`, with `_exec.log` and `_run_summary.json` alongside. Status stays Candidate; see the section below |
| 2026-10-12 | `ca5cffb` / blob `6cc750453096` (downloaded from GitHub at the PR head and blob-verified before the session; the executed file's cell IDs and sources equal the blob, no toggle changed) | Google Colab CLI 0.7.4 sequential execution on a fresh Colab **Tesla T4** session via the workspace `colab-cli-serial-test-suite` (`colab new --gpu T4`, `colab exec -f`, `colab stop`); isolated `uv` Python 3.12.12 environment from the carried lock; cells run in order in one kernel, not a browser Run all; order evidenced by the CLI's `Executing cell k/18` log (1/18–18/18; the CLI records no execution counts) | Default path, every optional toggle off; forms and uploads not exercised (the BYOD cells printed `BYOD disabled (RUN_BYOD = False).`) | 589.2 s session wall; stages prepare 139.3 s, asr 88.0 s, index 25.9 s, evaluate 104.3 s, activity 130.3 s, reload 28.0 s; peak allocated GPU 1.61 GiB | **PASS**: 18/18 code cells, no error outputs; every stage printed PASS; fresh-process reload passed (4 queries replayed, 4 documents re-embedded, 3 clips retranscribed). Compared with the 2026-10-03 run of `e4b3fed`: WER 0.140 / CER 0.036 and WER 0.136 / CER 0.042, every per-system retrieval score, the three existing hit@5 comparisons in `paired_comparisons.json` (asr_rerank − reference_rerank 1.0 vs 1.0, difference 0.0; asr_rerank − asr_dense 0.0; asr_dense − asr_bm25 +0.025) and the depth-20 development scores are equal. New outputs, as the 2026-10-12 fixes intend: seven `additional_comparisons` (ASR − reference for BM25 and dense on hit@5; anchor reciprocal rank@10 for all five), `same_candidate_depth_quality` (20/20 depth-10 rankings equal canonical; reference hit@5 1.000/1.000 and RR@10 0.967/0.963 at depth 10/20; ASR 0.950/0.950 and 0.917/0.917), the per-system top-five miss count (`asr_bm25` 1/40 and `reference_bm25` 1/40 on evaluation queries, others 0/40) with its traced query, and compact printed summaries; latency timings differ run to run. Executed notebook: `execution-evidence/2026-10-12/DIMER_Filipino_Audio_Archive_Search_Capstone_ca5cffb_colab-cli-t4.ipynb`, SHA-256 `7ba3568adec65d09cf34bd1cbff7c3c2a71dc28b6be429af051403ecea012eab` (with `…_run_summary.json` sha256 `aca7331eba57…` and `…_exec.log` sha256 `3a571f5e390a…`). BYOD (including a refused input) and the search toggles not exercised. Status stays **Candidate** |
| 2026-10-03 | `e4b3fed` / blob `f890d97d43fd` (downloaded from GitHub at the commit and blob-verified before the session; all cells equal the PR head) | Google Colab CLI 0.7.4 on a fresh Colab **Tesla T4** session via the workspace `colab-cli-serial-test-suite` (`colab new --gpu T4`, `colab exec -f`, `colab stop`); cells run in order in one kernel, not a browser Run all; order evidenced by the CLI's `Executing cell k/18` log (the CLI records no execution counts) | Default path, every optional toggle off | 615.3 s wall | **PASS**: 18/18 code cells, no errors. Outputs equal the `ad42d18` run except download progress bars. Executed notebook: `execution-evidence/2026-10-03/DIMER_Filipino_Audio_Archive_Search_Capstone_e4b3fed_colab-cli-t4.ipynb`, SHA-256 `2d3e18e7adce2135b87d23672be09b77b58468404b177654173ab90b505f0005`. Saved outputs inspected; BYOD and search toggles not exercised |
| 2026-09-28 | `ad42d18` (`main`) / blob `47da982901db` (executed file's 35 cell sources and IDs equal this blob; no toggle changed) | Google Colab, fresh **Tesla T4**, maintainer-supplied; isolated `uv` Python 3.12.12 environment from the carried lock | Default `Run all`, every optional toggle off | Stage total 483.8 s (Colab saved no per-cell timings) | **PASS**: 18/18 code cells, execution counts 1–18, no errors; see below |
| 2026-09-28 | `5664f61` / blob `47da982901db` (fetched from GitHub at the commit and blob-verified in-run; four toggle lines edited and three harness cells appended, listed below) | Kaggle kernel `kurtvalcorza/dimer-nb2-filipino-audio-archive-search-capstone` v1, serial test-suite executor: fresh `nbclient` interpreter, **Tesla T4** ×2 (driver 580.159.04; the runtime uses `cuda:0`), clean HF cache; isolated `uv` CPython 3.12.12 venv with torch 2.11.0+cu130 from the carried lock | `Run all` with search, competitor playback and BYOD on; harness cells re-queried BYOD, re-ran the export, built a no-reference BYOD package and reused it from a clean directory | 733.2 s runner wall (installs, 1.56 GB data and model downloads included); stage total 462.5 s | **PASS**: 21/21 code cells, execution counts 1–21, no errors; see below |
| 2026-09-28 | `d58ab28` / blob `284adf543e21` (executed file's 35 cell sources and IDs equal this blob; no `# @param` value changed) | Google Colab, fresh **Tesla T4** (reported by the preflight cell); isolated `uv` Python 3.12.12 environment from the carried hash-pinned lock | Default `Run all`; every optional toggle left at its default (off). Code cells executed in order 1–14; `reload` raised; export and the optional search/BYOD cells did not run | Not recorded (Colab saved no per-cell timings); ASR 52.8 s for 1,715.46 s of audio | **FAILED at `reload`** — see below |

### Colab CLI execution of `816b8dc` (query version `ai-draft-v2`) — 2026-10-14

First hosted run after the owner decisions FAS-M1 = B and BYOD = B. The overlap audit printed
mean anchor overlap 0.010 (maximum 0.143; threshold 0.1 / 0.25: within). ASR WER/CER output is
unchanged from `ca5cffb`; every retrieval number changed with the query set. Anchor recovery
by system, `ca5cffb` (`ai-draft-v1`) → `816b8dc` (`ai-draft-v2`):

| System | Evaluation hit@5 | Evaluation RR@10 | Development hit@5 (depth 20) |
|---|---|---|---|
| reference_bm25 | 0.975 → 0.075 | 0.938 → 0.036 | 0.950 → 0.100 |
| asr_bm25 | 0.975 → 0.075 | 0.906 → 0.038 | 0.900 → 0.100 |
| reference_dense | 1.000 → 0.425 | 0.946 → 0.327 | 1.000 → 0.600 |
| asr_dense | 1.000 → 0.425 | 0.963 → 0.331 | 0.950 → 0.650 |
| reference_rerank | 1.000 → 0.450 | 1.000 → 0.375 | 1.000 → 0.550 |
| asr_rerank | 1.000 → 0.475 | 1.000 → 0.339 | 0.950 → 0.500 |

Keyword search falls to near chance, as the redesign intended; dense and reranked search
still recover roughly half the anchors (`asr_dense − asr_bm25` hit@5 +0.350, 95% interval
[+0.175, +0.525]). No system reaches 1.000. These are nominated-anchor recovery scores on
unreviewed `ai-draft-v2` queries, not benchmark relevance. Evidence boundary: CLI sequential
execution, not a browser Run all; forms, uploads, BYOD and search toggles not exercised.

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
