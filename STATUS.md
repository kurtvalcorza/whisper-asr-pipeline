# Release status

Current status: **Candidate / source-complete**. Unit tests and static conformance checks are expected to run in GitHub Actions. Each tutorial notebook remains non-release-grade until clean-runtime execution evidence for the exact candidate commit is recorded according to `docs/release-verification.md`.

The DAT20 refusal fix (2026-10-29, code commit `957404b`) produced the current blobs `f51a8b1a` (`tutorials/whisper_asr_colab.ipynb`) and `f2c619cd` (`tutorials/whisper_asr_finetune_colab.ipynb`). The runs of `b824f685` and `36bb92fc` are history for them (REL14); hosted runs of the new blobs (the default path and a scripted BYOD refusal) are pending.
