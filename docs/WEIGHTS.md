# Weight provenance and DIMER hosting

- Upstream: `openai/whisper-large-v3-turbo`
- Immutable revision: `41f01f3fe87f28c78e2fbf8b568835947dd65ed9`
- Weight format: SafeTensors (`model.safetensors`, 1617824864 bytes, SHA-256 `542566a422ae4f3fd23f1ba11add198fca01bbf82e66e6a2857b3f608b1eb9d1`)
- Upstream weight license: MIT
- Local snapshot: `weights/whisper-large-v3-turbo/` with `dimer-base-manifest.json` (per-file bytes + SHA-256 for the 12 files the loader reads — configs, tokenizer files, `README.md`, `model.safetensors`; `totalBytes` 1622464535); the Git repository commits the manifest and the small files and git-ignores the checkpoint.
- Load-time check: `stage_missing_files()` fetches only absent manifest entries at the immutable revision and `verify_snapshot()` in `src/whisper_asr_pipeline/pipeline.py` re-hashes every entry and refuses on any mismatch before `AutoProcessor` / `AutoModelForSpeechSeq2Seq` read the directory.
- DIMER hosting: the MIT license permits use, modification, distribution, sublicensing, and commercial use subject to preservation of the copyright and license notice. The Git repository does not vendor the multi-gigabyte checkpoint; DIMER may mirror the pinned checkpoint in its model store under the upstream license.
- Loader trust boundary: Hugging Face/Transformers standard Whisper implementation (`transformers==4.52.1`) with `trust_remote_code=False`; SafeTensors weights execute no code on load.
