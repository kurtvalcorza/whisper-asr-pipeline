# Release status

Current status: **Candidate / source-complete**. Unit tests and static conformance checks are expected to run in GitHub Actions. Each tutorial notebook remains non-release-grade until clean-runtime execution evidence for the exact candidate commit is recorded according to `docs/release-verification.md`.

The DAT20 refusal fix (2026-10-29, code commit `957404b`) produced the current blobs `4b74ff57` (`tutorials/whisper_asr_colab.ipynb`) and `5ca181d9` (`tutorials/whisper_asr_finetune_colab.ipynb`). The runs of `b824f685` and `36bb92fc` are history for them (REL14); on 2026-10-30 the default path and a scripted DAT20 refusal of each new blob passed on a hosted Colab Tesla T4 (Colab CLI sequential executions, not a browser `Run all`; see `docs/release-verification.md`); the remaining gates are REL12 and REL2.
