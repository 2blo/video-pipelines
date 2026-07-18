import json
import os
import shutil
import subprocess
from argparse import ArgumentParser
from pathlib import Path
from typing import Any, Dict, List

import cv2  # type: ignore[import-not-found]
import numpy as np
import yaml  # type: ignore[import-untyped]


def get_cuda_diagnostics() -> str:
    try:
        import torch  # type: ignore[import-not-found]
    except Exception as exc:
        return f"torch import failed: {exc}"

    parts: List[str] = [
        f"torch={torch.__version__}",
        f"torch_cuda={torch.version.cuda}",
        f"cuda_available={torch.cuda.is_available()}",
    ]

    if torch.cuda.is_available():
        try:
            idx = torch.cuda.current_device()
            name = torch.cuda.get_device_name(idx)
            capability = torch.cuda.get_device_capability(idx)
            parts.append(f"gpu={name}")
            parts.append(f"capability={capability[0]}.{capability[1]}")
            if hasattr(torch.cuda, "get_arch_list"):
                parts.append(f"torch_arch_list={torch.cuda.get_arch_list()}")
            if hasattr(torch.backends, "cudnn"):
                parts.append(f"cudnn={torch.backends.cudnn.version()}")
        except Exception as exc:
            parts.append(f"gpu_query_error={exc}")

    return ", ".join(parts)


def validate_cuda_runtime(device: str) -> None:
    try:
        import torch  # type: ignore[import-not-found]
        import torch.nn.functional as F  # type: ignore[import-not-found]
    except Exception:
        return

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for depth_anything_v3_streaming but torch.cuda.is_available() is false. "
            f"Diagnostics: {get_cuda_diagnostics()}"
        )

    try:
        x = torch.randn(1, 3, 32, 32, device="cuda", dtype=torch.float16)
        w = torch.randn(8, 3, 3, 3, device="cuda", dtype=torch.float16)
        _ = F.conv2d(x, w, padding=1)
        torch.cuda.synchronize()
    except RuntimeError as exc:
        message = str(exc)
        diagnostics = get_cuda_diagnostics()
        if "no kernel image" in message.lower():
            raise RuntimeError(
                "CUDA kernel compatibility check failed before DA3 inference. "
                "The installed torch CUDA kernels do not match this GPU architecture. "
                f"Diagnostics: {diagnostics}"
            ) from exc
        raise RuntimeError(
            "CUDA preflight conv2d check failed. "
            f"Diagnostics: {diagnostics}. Original error: {message}"
        ) from exc


def run_command(command: List[str], error_prefix: str) -> None:
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as exc:
        details: List[str] = [
            error_prefix,
            f"Exit code: {exc.returncode}",
            f"Command: {' '.join(command)}",
        ]
        raise RuntimeError("\n\n".join(details)) from exc


def ensure_weights(model_dir: Path) -> Dict[str, str]:
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / "model.safetensors"
    config_path = model_dir / "config.json"
    salad_path = model_dir / "dino_salad.ckpt"

    if model_path.exists() and config_path.exists() and salad_path.exists():
        return {
            "model": str(model_path),
            "config": str(config_path),
            "salad": str(salad_path),
        }

    base_url = (
        "https://huggingface.co/depth-anything/DA3NESTED-GIANT-LARGE-1.1/resolve/main"
    )
    download_commands: List[List[str]] = []
    if not config_path.exists():
        download_commands.append(
            ["curl", "-L", f"{base_url}/config.json", "-o", str(config_path)]
        )
    if not model_path.exists():
        download_commands.append(
            ["curl", "-L", f"{base_url}/model.safetensors", "-o", str(model_path)]
        )
    if not salad_path.exists():
        download_commands.append(
            [
                "curl",
                "-L",
                "https://github.com/serizba/salad/releases/download/v1.0.0/dino_salad.ckpt",
                "-o",
                str(salad_path),
            ]
        )

    for command in download_commands:
        run_command(command, "Failed downloading DA3 streaming weights.")

    return {
        "model": str(model_path),
        "config": str(config_path),
        "salad": str(salad_path),
    }


def extract_frames(
    input_video: Path, frames_dir: Path, fps: float, max_res: int
) -> None:
    frames_dir.mkdir(parents=True, exist_ok=True)
    for old_frame in frames_dir.glob("frame_*.png"):
        old_frame.unlink()

    scale_filter = (
        "scale=w='if(gte(iw,ih),min(iw,{max_res}),-2)':"
        "h='if(gte(ih,iw),min(ih,{max_res}),-2)'"
    ).format(max_res=max_res)
    vf = f"fps={fps},{scale_filter}"

    command: List[str] = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_video),
        "-vf",
        vf,
        str(frames_dir / "frame_%06d.png"),
    ]
    run_command(command, "Failed to extract frames for DA3 streaming.")


def load_base_config() -> Dict[str, Any]:
    config_path = Path("/opt/depth-anything-3/da3_streaming/configs/base_config.yaml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def build_config(
    base_config: Dict[str, Any],
    weights: Dict[str, str],
    chunk_size: int,
    overlap: int,
    device: str,
    loop_enable: bool,
    save_depth_conf_result: bool,
    delete_temp_files: bool,
    align_lib: str,
) -> Dict[str, Any]:
    config = json.loads(json.dumps(base_config))
    config["Weights"]["DA3"] = weights["model"]
    config["Weights"]["DA3_CONFIG"] = weights["config"]
    config["Weights"]["SALAD"] = weights["salad"]

    model_cfg = config["Model"]
    model_cfg["chunk_size"] = chunk_size
    model_cfg["overlap"] = overlap
    model_cfg["loop_enable"] = loop_enable
    model_cfg["save_depth_conf_result"] = save_depth_conf_result
    # DA3 can delete temporary outputs that include depth/conf npz artifacts.
    # Keep temp files when depth/conf results are requested so post-processing can run.
    effective_delete_temp_files = (
        False if save_depth_conf_result and delete_temp_files else delete_temp_files
    )
    model_cfg["delete_temp_files"] = effective_delete_temp_files
    effective_align_lib = align_lib
    if device == "cpu" and align_lib in ["triton", "torch"]:
        effective_align_lib = "numba"
    model_cfg["align_lib"] = effective_align_lib

    return config


def depth_to_u8(depth_map: np.ndarray) -> np.ndarray:
    depth = np.asarray(depth_map, dtype=np.float32)
    depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)

    depth_min = float(depth.min())
    depth_max = float(depth.max())
    if depth_max > depth_min:
        normalized = (depth - depth_min) / (depth_max - depth_min)
    else:
        normalized = np.zeros_like(depth)

    return (normalized * 255.0).astype(np.uint8)


def convert_npz_results_to_depth_maps(output_dir: Path, fps: float) -> int:
    depth_maps_dir = output_dir / "depth_maps"
    depth_maps_dir.mkdir(parents=True, exist_ok=True)

    npz_files = sorted(output_dir.rglob("frame_*.npz"))
    if not npz_files:
        return 0

    writer = None
    preview_path = output_dir / "depth_preview.mp4"
    written = 0

    for index, npz_file in enumerate(npz_files, start=1):
        data = np.load(npz_file)
        if "depth" not in data:
            continue

        depth_u8 = depth_to_u8(data["depth"])
        png_path = depth_maps_dir / f"depth_{index:06d}.png"
        cv2.imwrite(str(png_path), depth_u8)

        if writer is None:
            height, width = depth_u8.shape
            writer = cv2.VideoWriter(
                str(preview_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps,
                (width, height),
            )

        preview_frame = np.repeat(depth_u8[:, :, np.newaxis], 3, axis=2)
        writer.write(preview_frame)
        written += 1

    if writer is not None:
        writer.release()

    return written


def collect_depth_pngs_to_depth_maps(output_dir: Path) -> int:
    depth_maps_dir = output_dir / "depth_maps"
    depth_maps_dir.mkdir(parents=True, exist_ok=True)

    existing = sorted(depth_maps_dir.glob("*.png"))
    if existing:
        return len(existing)

    candidates: List[Path] = []
    for png_path in output_dir.rglob("*.png"):
        if png_path.parent == depth_maps_dir:
            continue
        rel_parent = str(png_path.parent.relative_to(output_dir)).lower()
        name = png_path.name.lower()
        if "depth" in rel_parent or "depth" in name:
            candidates.append(png_path)

    for index, source_path in enumerate(sorted(candidates), start=1):
        target_path = depth_maps_dir / f"depth_{index:06d}.png"
        shutil.copy2(source_path, target_path)

    return len(candidates)


def to_bool(value: int) -> bool:
    return value != 0


def run_da3_streaming(command: List[str], env: Dict[str, str]) -> None:
    subprocess.run(command, check=True, env=env)


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--input-video", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--max-res", type=int, default=640)
    parser.add_argument("--fps", type=float, default=5.0)
    parser.add_argument("--device", choices=["auto", "cuda"], default="auto")
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--overlap", type=int, default=24)
    parser.add_argument("--loop-enable", type=int, default=1)
    parser.add_argument("--save-depth-conf-result", type=int, default=1)
    parser.add_argument("--delete-temp-files", type=int, default=1)
    parser.add_argument(
        "--align-lib",
        choices=["triton", "torch", "numba", "numpy"],
        default="triton",
    )
    args = parser.parse_args()

    if args.max_res <= 0:
        raise RuntimeError("max_res must be positive")
    if args.fps <= 0:
        raise RuntimeError("fps must be positive")
    if args.chunk_size <= 0:
        raise RuntimeError("chunk_size must be positive")
    if args.overlap < 0 or args.overlap >= args.chunk_size:
        raise RuntimeError("overlap must be >= 0 and < chunk_size")

    validate_cuda_runtime(args.device)

    input_video = Path(args.input_video).resolve()
    output_dir = Path(args.output_dir).resolve()
    model_dir = Path(args.model_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    weights = ensure_weights(model_dir)

    frames_dir = output_dir / "frames"
    extract_frames(input_video, frames_dir, args.fps, args.max_res)

    config = build_config(
        base_config=load_base_config(),
        weights=weights,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        device=args.device,
        loop_enable=to_bool(args.loop_enable),
        save_depth_conf_result=to_bool(args.save_depth_conf_result),
        delete_temp_files=to_bool(args.delete_temp_files),
        align_lib=args.align_lib,
    )

    config_path = output_dir / "da3_streaming_config.yaml"
    with open(config_path, "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)

    command: List[str] = [
        "python3",
        "/opt/depth-anything-3/da3_streaming/da3_streaming.py",
        "--image_dir",
        str(frames_dir),
        "--config",
        str(config_path),
        "--output_dir",
        str(output_dir),
    ]
    env = os.environ.copy()
    env.setdefault(
        "PYTORCH_CUDA_ALLOC_CONF",
        "expandable_segments:True,max_split_size_mb:128,garbage_collection_threshold:0.8",
    )
    env.setdefault("CUDA_MODULE_LOADING", "EAGER")
    env.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "1")
    print(f"[da3-streaming] CUDA preflight: {get_cuda_diagnostics()}")
    if os.environ.get("VIDEO_PIPELINES_DA3_DEBUG_CUDA", "0") == "1":
        env["CUDA_LAUNCH_BLOCKING"] = "1"

    try:
        run_da3_streaming(command, env)
    except subprocess.CalledProcessError as exc:
        details: List[str] = [
            "DA3 streaming execution failed.",
            f"Exit code: {exc.returncode}",
            f"Command: {' '.join(command)}",
        ]
        details.append(
            "Hint: this can be caused by unsupported CUDA kernels on your GPU."
        )
        details.append(
            "Hint: for CUDA OOM, reduce max_res, fps, chunk_size, or overlap in the "
            "depth_anything_v3_streaming variant."
        )
        raise RuntimeError("\n\n".join(details)) from exc

    if to_bool(args.save_depth_conf_result):
        n_depth_maps = convert_npz_results_to_depth_maps(
            output_dir=output_dir, fps=args.fps
        )
        if n_depth_maps == 0:
            n_depth_maps = collect_depth_pngs_to_depth_maps(output_dir=output_dir)
            if n_depth_maps == 0:
                raise RuntimeError(
                    "DA3 streaming completed but produced no depth npz files under results_output."
                )


if __name__ == "__main__":
    main()
