# Video pipelines

```bash
make build
```

## Video2X (Docker) example: 4K + 120 FPS

This repo now includes [Dockerfile.video2x](Dockerfile.video2x), which pins the
official Video2X container image from GitHub Container Registry.

Build the local wrapper image:

```bash
make build-video2x-image
```

On Linux with `/dev/dxg` (WSL GPU), the Make targets automatically use a
software Vulkan fallback to avoid Video2X container crashes (`Error 139`).
You can override runtime flags explicitly, for example:

```bash
make VIDEO2X_DOCKER_FLAGS="--gpus all" video2x-4k-pass
```

For `./sandbox/inputs/second_step_0_Trim.mp4`, run this in two passes:

1) Upscale to 4K (3840x2160)

```bash
make video2x-4k-pass
```

1) Interpolate to 120 FPS (uses `-m 4`, suitable for ~30 FPS source)

```bash
make video2x-120fps-pass
```

Final output:

`./sandbox/outputs/second_step_0_Trim_4k_120fps.mp4`

If your source is not ~30 FPS, adjust `-m` in the second pass.
Target is approximately:

$$
f_{target} \approx f_{source} \times m
$$
