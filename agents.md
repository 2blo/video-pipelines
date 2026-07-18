# Agents instructions

- Do not build retry logic in scripts, prefer failing fast with extensive error messages.
- Require CUDA to be used for all models, do not build fallbacks to CPU, fail fast instead.
- Prefer preserving input video quality, never reduce framerate, reducing resolution is acceptable if developer explicitly requests it / specifies it in a pipeline chart.
