# Video pipelines

```bash
make build
```

## RIFE Docker video interpolation

Build + run with your own input/output paths (WSL paths, for example `/mnt/c/...`, work):

```bash
make rife-upscale INPUT=/path/to/input.mp4 SCALE=2 OUTPUT=/path/to/output.mp4
```

- `SCALE` is temporal interpolation scale: `2` = 2x fps, `4` = 4x fps, etc. (power of two).
- GPU is enabled by default with `--gpus all`.

Run with an auto-generated example video:

```bash
make rife-example
```

Outputs are written to `sandbox/outputs/example_2x.mp4`.
