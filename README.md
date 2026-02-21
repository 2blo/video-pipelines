# Video pipelines

```bash
make build
```

## SeedVR2 Docker upscaler (folder → folder)

This builds a CUDA-enabled image that runs the official SeedVR2-3B inference script.

Build:

```bash
make seedvr2-docker-build
```

Run (GPU required):

```bash
make seedvr2-docker-run
```

If Docker prints `could not select device driver "" with capabilities: [[gpu]]`, your host has NVIDIA drivers but Docker lacks the NVIDIA runtime. Run:

```bash
make setup-docker-gpu
make seedvr2-doctor
```

If your Docker `Context` is `rootless` (shown by `make seedvr2-doctor`), you may need to restart the rootless daemon:

```bash
systemctl --user restart docker
```

Notes:

- The container auto-downloads the SeedVR2-3B checkpoint into `/opt/seedvr/ckpts` if missing.
- You can set explicit output size instead of `--scale` via `--res-h` and `--res-w`.
- Multi-GPU: add `--num-gpus N --sp-size N` (SeedVR2 uses sequence parallelism).
