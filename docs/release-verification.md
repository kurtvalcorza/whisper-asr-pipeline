# Release verification

This repository treats the tutorial notebook as a **release candidate** until the exact revision has executed top-to-bottom in a clean supported runtime. Unit tests, JSON validation, and code-cell compilation are necessary checks but are not runtime evidence under DIMER Notebook Specification v1.0.

Before release, record a durable verification entry containing: repository commit SHA; notebook filename; Python and principal framework/library versions; accelerator/device; upstream model identifier and immutable revision; whether the model cache was clean; outcome of every default-path stage; output/provenance artifact names; and any applicable SHOULD deviation.

For heavyweight GPU notebooks, manual Colab/Kaggle or controlled GPU-runner evidence is acceptable when CI cannot execute the complete model path. The tested revision must match the proposed release revision. A known-failing default Colab path blocks release.
