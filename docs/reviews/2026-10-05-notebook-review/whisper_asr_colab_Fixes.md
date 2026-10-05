# Whisper ASR task-inference notebook — review fixes

**Review:** `whisper_asr_colab_Review.md` (5 October 2026, WSP-M1..M3, WSP-m1..m2).
**Fixed in:** the generator and templates:
- `tools/build_notebook.py`, now the isolated-runtime generator `build_notebook.py/2.1-swc`;
- `tools/notebook_template.py`;
- `tools/inference-pins.txt`.

Also changed: a new carried module, `src/whisper_asr_pipeline/tutorial_support.py`; the hash lock `tutorials/requirements-colab.lock.txt`; the validator and its negative controls; `tutorials/README.md`; `docs/release-verification.md`; and the tests. `pipeline.py` is unchanged, because the workshop notebook carries it byte for byte. The notebook was regenerated and `--check` passes.
**Readiness:** **Verification pending** until the hosted gates below are recorded. `STATUS.md` and every release label are unchanged.

## Findings

| ID | Status | Change | Cells / files | Evidence |
|---|---|---|---|---|
| WSP-M1 | Fixed in source; the run is the remaining gate | The in-kernel `pip` install and restart guard are replaced by the generator's isolated runtime: pinned `uv` 0.12.15 (checked by size and SHA-256), managed CPython 3.12.12, and the carried hash lock (67 packages, `--require-hashes --only-binary :all:`). A router sends every later cell to one worker. The notebook declares spec 2.2. The registry's Run-all cell and the record's *Current status* no longer call this notebook verified. They name the blobs the 2026-09-11 runs covered (`a5b13f9b`, `c308c7b1`, the earlier repository-installing notebook) and state that no run of the current blob exists. The validator requires exactly two kernel cells and forbids kernel `pip` and restart text; its negative controls were rewritten for that. | Sections 1–3; `tutorials/README.md`; `docs/release-verification.md`; validator | `test_wsp_M1_wsf_M1_*`; `test_control_isolated_install_without_hash_check_is_rejected`, `test_control_restart_instruction_is_rejected`; `grep "Restart the runtime"` returns nothing |
| WSP-M2 | Fixed | The inline pins are `tools/inference-pins.txt`: pyproject's dependencies plus `datasets==4.4.0` from the `tutorial` extra. They are compiled into the lock, so the default loader runs on the pinned version in any runtime. | `PINS`, lock | `test_wsp_M2_inference_pins_are_pyproject_dependencies_plus_datasets` |
| WSP-M3 | Fixed | The notebook now has:<ul><li>How to use, with the audience statement, cell kinds, form controls and the predict/check convention</li><li>an I/M/O table and a roadmap</li><li>a glossary covering log-Mel, language token, greedy, chunking, WER normalisation, hallucination and more</li><li>predictions with collapsed answers for Sections 5, 6 and 7</li><li>new form fields `TASK`, `LANGUAGE` and `RETURN_TIMESTAMPS`, read by Sections 5–6, so every experiment changes only a form field</li><li>a Section 9 silence activity on generated silence or noise (`ACTIVITY_AUDIO`), with no upload</li><li>a troubleshooting table and a conclusion template</li></ul>Infrastructure cells are titled and collapsed. | template | `test_wsp_M3_*` (marker, collapse, form-field and silence-activity tests) |
| WSP-m1 | Fixed | BYOD comes from `BYOD_PATH` or exactly one upload. It is decoded by `decode_audio` (the pinned `soundfile`; empty or unreadable files are refused by name) and passed to the model as `{'array', 'sampling_rate'}`, like the sample, so no `ffmpeg` is needed. The manifest records seconds and rate. | Section 4; `tutorial_support.decode_audio` | `test_wsp_m1_*` (path run through Sections 4–7 with a stand-in model that rejects file names; corrupt/empty refusals; zero/two uploads) |
| WSP-m2 | Fixed | `with_empty_transcript_baseline` adds the empty-transcript baseline (WER 1.0) to the report when a reference exists. Section 7 explains what 0, 1.0 and values above 1 mean. Scoring the rest of the dummy split was not added. | Section 7; `tutorial_support` | `test_wsp_m2_*` |

Suggestions: S4 applied (spec 2.2). S1, S2 and S3 not taken.

## User-visible changes

- Section 1 builds an isolated environment (Linux x86_64 only; the CUDA `torch` build is a larger download). Every later cell runs there.
- Section 4 has new form fields: `BYOD_PATH`, `TASK`, `LANGUAGE`, `RETURN_TIMESTAMPS`. BYOD audio reaches the model as a waveform.
- New Section 9 (silence activity), which runs during Run all and writes nothing to `outputs/`.
- `evaluation_report.json` gains a baseline entry when a reference exists; `result.json` gains `request`.
- The notebook carries a second module cell, `tutorial_support.py`.

## Verification (offline, not clean-runtime evidence)

- `build_notebook.py --check` OK for both notebooks; `build_audio_capstone.py --check` OK; `validate_release_assets.py` PASS (the workshop notebook is unchanged); `ruff check src tests tools` clean.
- pytest with CI's dependencies and no torch: 176 passed / 5 skipped before → **215 passed / 5 skipped** after. This covers both notebooks; `tests/test_review_fixes.py` holds 38 tests.
- The Section 4–7 and 9 learner cells were executed from the generated notebook with a stand-in model and stand-in `datasets`, on real WAV files written with `soundfile`.
- No Whisper inference ran: the Hugging Face Hub is unreachable here and torch is not installed. The transcript and WER quoted in the answers come from the recorded 2026-09-11 runs of the earlier notebook.

## Remaining gates

1. A hosted one-pass **Run all** of the regenerated blob on a fresh runtime, recorded with the blob, `restarted: false`, the transcript, the WER and the silence-activity output. Then re-run the export cell.
2. The REL12 BYOD journey: one WAV by path and one through the dialog, plus one corrupt file.
3. Maintainer: `STATUS.md`; whether to score more utterances of the dummy split.
