# Whisper ASR LoRA fine-tuning notebook — review fixes

**Review:** `whisper_asr_finetune_colab_Review.md` (5 October 2026, WSF-M1..M3, WSF-m1..m3).

**Fixed in:**
- the generator (`tools/build_notebook.py` 2.1-swc) and `tools/notebook_template_finetune.py`;
- the new carried module `src/whisper_asr_pipeline/tutorial_support.py`;
- the hash lock `tutorials/requirements-finetune.lock.txt`;
- the validator, `tutorials/README.md`, `docs/release-verification.md` and the tests.

The notebook was regenerated and `--check` passes. The task-inference notebook was fixed in the same change (`whisper_asr_colab_Fixes.md`).

**Readiness:** **Verification pending** until the hosted gates below are recorded. `STATUS.md` and every release label are unchanged.

## Findings

| ID | Status | Change | Cells / files | Evidence |
|---|---|---|---|---|
| WSF-M1 | Fixed in source; the run is the remaining gate | **Install.** Isolated `uv` runtime: managed CPython 3.12.12, hash lock of 70 packages compiled from `tools/finetune-pins.txt`. There is no kernel `pip` and no restart guard; spec 2.2 is declared.<br>**Registry.** The Run-all cell names the 2026-09-14 T4 run's blob (`818d2e55`), says it needed one restart, and states that it is not REL2 evidence. The record's *Current status* says no run of the current blob exists. | Sections 1–3; README; record | `test_wsp_M1_wsf_M1_*` |
| WSF-M2 | Fixed | **Orientation:** How to use, I/M/O table, roadmap, and a glossary (LoRA rank and alpha, q/v, teacher forcing, `-100`, scaler, corpus WER, edit breakdown, merge check, forgetting).<br>**Predictions with collapsed answers:** before the baseline (Section 6), training (8) and adapted evaluation (9). The Section 8 answer explains why validation loss can rise while WER falls; the Section 10 notice follows.<br>**Activity:** Section 11 is a Predict → Change → Run → Observe → Explain exercise with `EPOCHS = 1`.<br>**Support:** troubleshooting table and conclusion template. Infrastructure cells are collapsed. | template | `test_wsf_M2_*`, `test_wsp_M3_wsf_M2_infrastructure_cells_are_collapsed[WSF]` |
| WSF-M3 | Fixed | **Pipeline rebuilt on re-run.** Section 6 starts with `if 'pipe' not in globals(): pipe = WhisperASRPipeline.from_pretrained(weights_dir=WEIGHTS_DIR, allow_download=False)`. A re-run from Section 4 (BYOD, the activity, Try next) therefore rebuilds the baseline pipeline from the verified snapshot.<br>**Prose.** How to use, the activity and Troubleshooting say to re-run from the changed cell. | Section 6; prose | `test_wsf_M3_rerun_after_pipe_was_released_rebuilds_it` (executes the notebook's Section 6 cell twice: the second run rebuilds `pipe` with `allow_download=False`) |
| WSF-m1 | Fixed (kept the filter, recorded it) | **Recording.** Clips over 30 s are still dropped before the split, but their ids are kept in `DROPPED_OVER_30S`. They are printed as a count and recorded in Section 5 as a `clips-over-30s` finding with every id.<br>**Prose.** Sections 4 and 5 and the BYOD declaration now say so. | Sections 4–5 | `test_wsf_m1_m2_long_clip_is_dropped_and_recorded_and_the_byod_split_is_derived` (real 35 s WAV) |
| WSF-m2 | Fixed | **Input sources.** `BYOD_PATH` accepts a folder or a `.zip`; macOS metadata is skipped.<br>**Named refusals.** A missing or doubled `.csv`, a missing `file,text` header, and every audio file the CSV names that was not supplied. Undecodable audio is refused by name (`decode_audio`).<br>**Split and size.** For BYOD the split is derived (`BYOD_TRAIN_FRACTION` 0.8). At least 10 usable clips are required, a minimum stated in the Prerequisites, the BYOD declaration and Section 4. The public-sample error names `TRAIN_CLIPS` and `EVAL_CLIPS`. | Section 4; prose | `test_wsf_m1_m2_*`, `test_wsf_m2_missing_audio_and_too_few_clips_are_refused_by_name` |
| WSF-m3 | Fixed | **Edit breakdown.** `word_error_breakdown` gives substitution, insertion and deletion counts for the baseline, adapted and reloaded transcripts. They are printed and written to the evaluation report (`edit_breakdown`).<br>**Duplicates.** Section 4 counts the held-out transcripts that also occur, normalised, in training (`eval_transcripts_also_in_train`).<br>**Caveat.** Section 9 carries the record's reference-style caveat as **Read the gain carefully.**<br>**Timing.** The Prerequisites and Run-all quote the 2026-09-14 T4 run (1002 s; earlier blob, after one restart) and the 2026-09-11 P100 run with its stage breakdown. The "about 27 minutes" figure is gone. | Sections 4, 6, 9, 10; header | `test_wsf_m3_*` |

Suggestions applied:
- **S1:** Section 8 says the held-out clips select nothing.
- **S4:** spec 2.2 is declared.

Suggestions not taken: S2 and S3.

## User-visible changes

- Section 1 builds an isolated environment (several GB, the CUDA `torch`). Every later cell runs there.
- Section 4 has new form fields: `BYOD_PATH` and `BYOD_TRAIN_FRACTION`. For BYOD, `TRAIN_CLIPS` and `EVAL_CLIPS` are derived.
- The Section 4 printout gains `dropped_over_30s` and `eval_transcripts_also_in_train`.
- Section 5 may record a `clips-over-30s` finding.
- `evaluation_report.json` gains `edit_breakdown`.
- Re-running from Section 4 works.
- The notebook carries a second module cell, `tutorial_support.py`.

## Verification (offline, not clean-runtime evidence)

- **Static checks:**
  - `build_notebook.py --check --template tools/notebook_template_finetune.py`: OK.
  - `validate_release_assets.py`: PASS.
  - `ruff check src tests tools`: clean.
- **Tests:** the repository suite went from 176 passed / 5 skipped to **215 passed / 5 skipped**.
- **Cells executed:** the notebook's Sections 4, 5 and 6 were run on real WAV files with stand-ins for the model and `torchaudio`.
- **Not run here:** training, inference and the export path. The Hub is unreachable, there is no torch and no GPU.

## Remaining gates

1. A hosted one-pass **Run all** of the regenerated blob on a fresh T4, recording:
   - `restarted: false`;
   - baseline, adapted and reloaded WER, with the edit breakdowns;
   - the losses and peak memory;
   - the duplicate count.

   Then re-run the export cell.
2. The REL12 BYOD journey:
   - a small folder or zip by path, and one through the dialog;
   - a refusal for a missing file;
   - a re-run from Section 4 reaching Section 10.
3. Maintainer: `STATUS.md`; whether to commit the executed notebook under `docs/execution-evidence/`.
