# Filipino audio capstone: data attribution and transformations

Source: Google FLEURS, Filipino (`fil_ph`), revision
`70bb2e84b976b7e960aa89f1c648e09c59f894dd`.
[Pinned dataset card](https://huggingface.co/datasets/google/fleurs/blob/70bb2e84b976b7e960aa89f1c648e09c59f894dd/README.md).

The source card declares **Creative Commons Attribution 4.0 International**
([CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)). Retain attribution,
the licence link, and this description of changes when sharing derived material.

Credit: Alexis Conneau, Min Ma, Simran Khanuja, Yu Zhang, Vera Axelrod,
Siddharth Dalmia, Jason Riesa, Clara Rivera and Ankur Bapna (2022),
[*FLEURS: Few-shot Learning Evaluation of Universal Representations of Speech*](https://arxiv.org/abs/2205.12446).

Changes: deterministic selection of 120 original mono 16 kHz clips from the
published validation/test splits; source-ID, normalized-text and audio-hash
family grouping; duration filtering to 2–25 seconds. Original encoded audio
and display references are preserved. A separate NFC/lowercase,
punctuation-to-space normalization is used for scoring. Automatic transcripts,
numeric embeddings, ranks, plots and draft query annotations are derived outputs.

The 60 AI-assisted Filipino query drafts are maintainer-authored adaptations
of source reference material, with source document IDs retained. They have not
been reviewed by a Filipino-language human annotator. All 7,200 relevance
judgements remain unjudged; a nominated anchor is not a relevance label.

Recordings are read speech. The selected corpus does not establish independent
speakers or capture locations, and it is not a recording of public-service
meetings or a representative Philippine population sample. Model-pretraining
overlap is unknown. No speaker identity is inferred.

Default exports exclude audio and pretrained model weights. Audio can be
reacquired from the pinned source files and verified hashes in the manifests.
Any optional redistribution of source audio must retain the source attribution,
CC BY 4.0 link and transformation notice. Model licences remain separate:
Whisper large-v3-turbo is MIT; Qwen3 Embedding/Reranker 0.6B are Apache-2.0.

AI assistance disclosure: Code, documentation and draft queries were developed
with generative AI assistance under maintainer direction. The maintainer is
responsible for reviewing the implementation, validating results and making
release decisions. AI assistance is not independent verification, provider
endorsement, human annotation review or release approval.
