#!/usr/bin/env python3

import argparse
import os
import re
import subprocess
import tempfile
from pathlib import Path


VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}


def _ceil_to_multiple(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def _find_first_video(input_dir: Path) -> Path | None:
    if not input_dir.exists():
        return None
    for path in sorted(input_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in VIDEO_EXTS:
            return path
    return None


def _list_videos(input_dir: Path) -> list[Path]:
    if not input_dir.exists():
        return []
    return [
        p
        for p in sorted(input_dir.iterdir())
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS
    ]


def _ffprobe_size(video_path: Path) -> tuple[int, int]:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height",
        "-of",
        "csv=p=0:s=x",
        str(video_path),
    ]
    out = subprocess.check_output(cmd, text=True).strip()
    if "x" not in out:
        raise RuntimeError(f"ffprobe returned unexpected output: {out!r}")
    w_str, h_str = out.split("x", 1)
    return int(w_str), int(h_str)


def _run_seedvr_cmd(
    *,
    input_dir: Path,
    output_dir: Path,
    num_gpus: int,
    seed: int,
    res_h: int,
    res_w: int,
    sp_size: int,
    out_fps: float | None,
) -> int:
    cmd = [
        "torchrun",
        f"--nproc-per-node={num_gpus}",
        "projects/inference_seedvr2_3b.py",
        "--video_path",
        str(input_dir),
        "--output_dir",
        str(output_dir),
        "--seed",
        str(seed),
        "--res_h",
        str(res_h),
        "--res_w",
        str(res_w),
        "--sp_size",
        str(sp_size),
    ]
    if out_fps is not None:
        cmd.extend(["--out_fps", str(out_fps)])
    print("Running:", " ".join(cmd), flush=True)
    return subprocess.call(cmd)


def _tile_extents_grid(
    width: int,
    height: int,
    overlap: int,
    tile_cols: int,
    tile_rows: int,
) -> list[tuple[int, int, int, int, int, int, int, int, int, int]]:
    x_edges = [int(round(i * width / tile_cols)) for i in range(tile_cols + 1)]
    y_edges = [int(round(i * height / tile_rows)) for i in range(tile_rows + 1)]
    out = []
    for r in range(tile_rows):
        for c in range(tile_cols):
            core_x0 = x_edges[c]
            core_x1 = x_edges[c + 1]
            core_y0 = y_edges[r]
            core_y1 = y_edges[r + 1]
            x0 = max(0, core_x0 - overlap)
            x1 = min(width, core_x1 + overlap)
            y0 = max(0, core_y0 - overlap)
            y1 = min(height, core_y1 + overlap)
            out.append((x0, x1, y0, y1, core_x0, core_x1, core_y0, core_y1, c, r))
    return out


def _run_tiled_grid(
    *,
    video_path: Path,
    output_path: Path,
    final_w: int,
    final_h: int,
    overlap: int,
    tile_cols: int,
    tile_rows: int,
    num_gpus: int,
    seed: int,
    sp_size: int,
    out_fps: float | None,
) -> int:
    in_w, in_h = _ffprobe_size(video_path)
    tiles = _tile_extents_grid(in_w, in_h, overlap, tile_cols, tile_rows)
    upscaled_tiles: list[
        tuple[Path, tuple[int, int, int, int, int, int, int, int, int, int]]
    ] = []

    with tempfile.TemporaryDirectory(prefix="seedvr2_tiles_") as td:
        td_path = Path(td)
        for idx, (
            x0,
            x1,
            y0,
            y1,
            core_x0,
            core_x1,
            core_y0,
            core_y1,
            c,
            r,
        ) in enumerate(tiles):
            tile_in_dir = td_path / f"tile_{idx}" / "in"
            tile_out_dir = td_path / f"tile_{idx}" / "out"
            tile_in_dir.mkdir(parents=True, exist_ok=True)
            tile_out_dir.mkdir(parents=True, exist_ok=True)

            tile_name = f"{video_path.stem}_tile{idx}{video_path.suffix}"
            tile_in_path = tile_in_dir / tile_name
            crop_w = x1 - x0
            crop_h = y1 - y0

            subprocess.check_call(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    str(video_path),
                    "-vf",
                    f"crop={crop_w}:{crop_h}:{x0}:{y0}",
                    "-an",
                    str(tile_in_path),
                ]
            )

            tile_target_w = _ceil_to_multiple(
                max(16, int(round(crop_w * (final_w / float(in_w))))), 16
            )
            tile_target_h = _ceil_to_multiple(
                max(16, int(round(crop_h * (final_h / float(in_h))))), 16
            )

            rc = _run_seedvr_cmd(
                input_dir=tile_in_dir,
                output_dir=tile_out_dir,
                num_gpus=num_gpus,
                seed=seed,
                res_h=tile_target_h,
                res_w=tile_target_w,
                sp_size=sp_size,
                out_fps=out_fps,
            )
            if rc != 0:
                return rc

            upscaled_tile = tile_out_dir / tile_name
            if not upscaled_tile.exists():
                raise RuntimeError(f"Missing tiled output: {upscaled_tile}")
            upscaled_tiles.append((upscaled_tile, (x0, x1, y0, y1, core_x0, core_x1, core_y0, core_y1, c, r)))

        crops: list[tuple[int, int, int, int, int, int]] = []
        for tile_path, (x0, x1, y0, y1, core_x0, core_x1, core_y0, core_y1, c, r) in upscaled_tiles:
            out_w, out_h = _ffprobe_size(tile_path)
            sx = out_w / float(x1 - x0)
            sy = out_h / float(y1 - y0)
            crop_x = max(0, int(round((core_x0 - x0) * sx)))
            crop_y = max(0, int(round((core_y0 - y0) * sy)))
            crop_w = max(16, int(round((core_x1 - core_x0) * sx)))
            crop_h = max(16, int(round((core_y1 - core_y0) * sy)))
            if crop_x + crop_w > out_w:
                crop_w = out_w - crop_x
            if crop_y + crop_h > out_h:
                crop_h = out_h - crop_y
            crop_h = max(16, int(round((core_y1 - core_y0) * sy)))
            if crop_y + crop_h > out_h:
                crop_h = out_h - crop_y
            place_x = int(round(core_x0 * final_w / float(in_w)))
            place_y = int(round(core_y0 * final_h / float(in_h)))
            crops.append(
                (
                    crop_x,
                    crop_y,
                    crop_w if crop_w > 0 else out_w,
                    crop_h if crop_h > 0 else out_h,
                    place_x,
                    place_y,
                )
            )

        crop_filters = []
        stack_inputs = []
        layout_parts = []
        for i, (cx, cy, cw, ch, px, py) in enumerate(crops):
            crop_filters.append(f"[{i}:v]crop={cw}:{ch}:{cx}:{cy}[c{i}]")
            stack_inputs.append(f"[c{i}]")
            layout_parts.append(f"{px}_{py}")
        xstack = "".join(stack_inputs) + f"xstack=inputs={len(crops)}:layout=" + "|".join(layout_parts) + "[stacked]"
        filter_complex = ";".join(crop_filters + [xstack, f"[stacked]scale={final_w}:{final_h}[outv]"])

        stitch_cmd = ["ffmpeg", "-y"]
        for tile_path, _ in upscaled_tiles:
            stitch_cmd.extend(["-i", str(tile_path)])
        stitch_cmd.extend(
            [
                "-filter_complex",
                filter_complex,
                "-map",
                "[outv]",
                "-c:v",
                "libx264",
                "-crf",
                "12",
                "-preset",
                "medium",
                "-pix_fmt",
                "yuv420p",
                str(output_path),
            ]
        )
        subprocess.check_call(stitch_cmd)
    return 0


def _cap_size_by_pixels(width: int, height: int, max_pixels: int) -> tuple[int, int]:
    if max_pixels <= 0:
        return width, height
    if width * height <= max_pixels:
        return width, height
    scale = (max_pixels / float(width * height)) ** 0.5
    capped_w = _ceil_to_multiple(max(16, int(width * scale)), 16)
    capped_h = _ceil_to_multiple(max(16, int(height * scale)), 16)
    while capped_w * capped_h > max_pixels and (capped_w > 16 and capped_h > 16):
        capped_w = max(16, capped_w - 16)
        capped_h = max(16, capped_h - 16)
    return capped_w, capped_h


def _apply_vae_runtime_tuning(
    config_path: Path,
    split_size: int,
    memory_device: str,
    conv_max_mem: float,
    norm_max_mem: float,
) -> None:
    if not config_path.exists():
        return
    text = config_path.read_text()
    text = re.sub(r"(?m)^(\s*split_size:\s*).*$", rf"\g<1>{split_size}", text)
    text = re.sub(
        r"(?m)^(\s*memory_device:\s*).*$", rf"\g<1>{memory_device}", text
    )
    text = re.sub(r"(?m)^(\s*conv_max_mem:\s*).*$", rf"\g<1>{conv_max_mem}", text)
    text = re.sub(r"(?m)^(\s*norm_max_mem:\s*).*$", rf"\g<1>{norm_max_mem}", text)
    config_path.write_text(text)


def _sanitize_vae_tuning(
    split_size: int,
    conv_max_mem: float,
    norm_max_mem: float,
    tile_mode: str,
) -> tuple[int, float, float]:
    safe_split = max(1, split_size)
    if tile_mode == "on" and safe_split < 4:
        print(
            f"Adjusting --vae-split-size from {safe_split} to 4 for tile mode stability",
            flush=True,
        )
        safe_split = 4

    safe_conv = conv_max_mem
    if safe_conv < 0.05:
        print(
            f"Adjusting --vae-conv-max-mem from {safe_conv} to 0.05 GiB to avoid invalid over-splitting",
            flush=True,
        )
        safe_conv = 0.05

    safe_norm = norm_max_mem
    if safe_norm < 0.01:
        print(
            f"Adjusting --vae-norm-max-mem from {safe_norm} to 0.01 GiB",
            flush=True,
        )
        safe_norm = 0.01

    return safe_split, safe_conv, safe_norm


def _ensure_seedvr2_ckpt(ckpts_dir: Path, repo_id: str) -> None:
    ckpts_dir.mkdir(parents=True, exist_ok=True)
    expected = ckpts_dir / "seedvr2_ema_3b.pth"
    if expected.exists():
        return

    try:
        from huggingface_hub import snapshot_download
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "huggingface_hub is required to auto-download checkpoints"
        ) from e

    print(f"Checkpoint {expected} not found; downloading from {repo_id}...", flush=True)
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(ckpts_dir),
        allow_patterns=[
            "*.json",
            "*.safetensors",
            "*.pth",
            "*.bin",
            "*.py",
            "*.md",
            "*.txt",
            "*.pt",
        ],
    )

    if not expected.exists():
        raise RuntimeError(
            f"Downloaded from {repo_id} but did not find {expected.name} in {ckpts_dir}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Upscale videos in a folder using SeedVR2 (3B) via torchrun.",
    )
    parser.add_argument("--input", required=True, help="Input folder containing videos")
    parser.add_argument("--output", required=True, help="Output folder")
    parser.add_argument(
        "--scale",
        type=float,
        default=2.0,
        help="Scale factor used when --res-h/--res-w are not provided (default: 2.0)",
    )
    parser.add_argument("--res-h", type=int, default=None, help="Output height")
    parser.add_argument("--res-w", type=int, default=None, help="Output width")
    parser.add_argument("--seed", type=int, default=666)
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=int(os.environ.get("NUM_GPUS", "1")),
        help="torchrun --nproc-per-node (default: $NUM_GPUS or 1)",
    )
    parser.add_argument(
        "--sp-size",
        type=int,
        default=int(os.environ.get("SP_SIZE", "1")),
        help="SeedVR2 sequence parallel size (default: $SP_SIZE or 1)",
    )
    parser.add_argument(
        "--out-fps",
        type=float,
        default=None,
        help="Override output FPS (optional)",
    )
    parser.add_argument(
        "--ckpt-repo",
        default=os.environ.get("SEEDVR2_REPO_ID", "ByteDance-Seed/SeedVR2-3B"),
        help="HuggingFace repo id used to auto-download checkpoints",
    )
    parser.add_argument(
        "--ckpts-dir",
        default=os.environ.get("SEEDVR2_CKPTS_DIR", "./ckpts"),
        help="SeedVR ckpts directory inside container (default: ./ckpts)",
    )
    parser.add_argument(
        "--max-pixels",
        type=int,
        default=int(os.environ.get("SEEDVR2_MAX_PIXELS", "0")),
        help="Cap output resolution by total pixels to avoid OOM; 0 disables cap (default: $SEEDVR2_MAX_PIXELS or 0)",
    )
    parser.add_argument(
        "--vae-split-size",
        type=int,
        default=int(os.environ.get("SEEDVR2_VAE_SPLIT_SIZE", "1")),
        help="Override configs_3b/main.yaml vae.slicing.split_size (default: $SEEDVR2_VAE_SPLIT_SIZE or 1)",
    )
    parser.add_argument(
        "--vae-conv-max-mem",
        type=float,
        default=float(os.environ.get("SEEDVR2_VAE_CONV_MAX_MEM", "0.05")),
        help="Override configs_3b/main.yaml vae.memory_limit.conv_max_mem (default: $SEEDVR2_VAE_CONV_MAX_MEM or 0.05)",
    )
    parser.add_argument(
        "--vae-norm-max-mem",
        type=float,
        default=float(os.environ.get("SEEDVR2_VAE_NORM_MAX_MEM", "0.05")),
        help="Override configs_3b/main.yaml vae.memory_limit.norm_max_mem (default: $SEEDVR2_VAE_NORM_MAX_MEM or 0.05)",
    )
    parser.add_argument(
        "--vae-memory-device",
        choices=["same", "cpu"],
        default=os.environ.get("SEEDVR2_VAE_MEMORY_DEVICE", "cpu"),
        help="Override configs_3b/main.yaml vae.slicing.memory_device (default: $SEEDVR2_VAE_MEMORY_DEVICE or cpu)",
    )
    parser.add_argument(
        "--tile-mode",
        choices=["off", "on"],
        default=os.environ.get("SEEDVR2_TILE_MODE", "off"),
        help="Enable 2x2 tiled inference to reduce VRAM (default: $SEEDVR2_TILE_MODE or off)",
    )
    parser.add_argument(
        "--tile-overlap",
        type=int,
        default=int(os.environ.get("SEEDVR2_TILE_OVERLAP", "64")),
        help="Tile overlap in input pixels when --tile-mode=on (default: $SEEDVR2_TILE_OVERLAP or 64)",
    )
    parser.add_argument(
        "--tile-grid",
        type=int,
        default=int(os.environ.get("SEEDVR2_TILE_GRID", "3")),
        help="Tile grid size N for NxN tiling when --tile-mode=on (default: $SEEDVR2_TILE_GRID or 3)",
    )

    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    ckpts_dir = Path(args.ckpts_dir)
    _ensure_seedvr2_ckpt(ckpts_dir=ckpts_dir, repo_id=args.ckpt_repo)

    res_h = args.res_h
    res_w = args.res_w

    if res_h is None or res_w is None:
        first_video = _find_first_video(input_dir)
        if first_video is None:
            raise SystemExit(f"No video files found in: {input_dir}")
        in_w, in_h = _ffprobe_size(first_video)
        res_w = _ceil_to_multiple(int(round(in_w * args.scale)), 16)
        res_h = _ceil_to_multiple(int(round(in_h * args.scale)), 16)
        print(
            f"Auto output size from {first_video.name}: {in_w}x{in_h} -> {res_w}x{res_h}",
            flush=True,
        )

    capped_w, capped_h = _cap_size_by_pixels(res_w, res_h, args.max_pixels)
    if (capped_w, capped_h) != (res_w, res_h):
        print(
            f"Capping output size to avoid OOM: {res_w}x{res_h} -> {capped_w}x{capped_h} (max_pixels={args.max_pixels})",
            flush=True,
        )
        res_w, res_h = capped_w, capped_h

    if args.sp_size < 1:
        raise SystemExit("--sp-size must be >= 1")
    if args.num_gpus < 1:
        raise SystemExit("--num-gpus must be >= 1")

    safe_split, safe_conv, safe_norm = _sanitize_vae_tuning(
        split_size=args.vae_split_size,
        conv_max_mem=args.vae_conv_max_mem,
        norm_max_mem=args.vae_norm_max_mem,
        tile_mode=args.tile_mode,
    )

    _apply_vae_runtime_tuning(
        config_path=Path("./configs_3b/main.yaml"),
        split_size=safe_split,
        memory_device=args.vae_memory_device,
        conv_max_mem=safe_conv,
        norm_max_mem=safe_norm,
    )
    print(
        "VAE runtime tuning: "
        f"split_size={safe_split}, "
        f"memory_device={args.vae_memory_device}, "
        f"conv_max_mem={safe_conv} GiB, "
        f"norm_max_mem={safe_norm} GiB",
        flush=True,
    )

    if args.tile_mode == "off":
        return _run_seedvr_cmd(
            input_dir=input_dir,
            output_dir=output_dir,
            num_gpus=args.num_gpus,
            seed=args.seed,
            res_h=res_h,
            res_w=res_w,
            sp_size=args.sp_size,
            out_fps=args.out_fps,
        )

    videos = _list_videos(input_dir)
    if not videos:
        raise SystemExit(f"No video files found in: {input_dir}")
    grid = max(2, args.tile_grid)
    print(
        f"Tile mode enabled: processing {len(videos)} video(s) in {grid}x{grid} tiles with overlap={args.tile_overlap}px",
        flush=True,
    )
    for video in videos:
        out_path = output_dir / video.name
        print(f"Tiled processing: {video.name}", flush=True)
        rc = _run_tiled_grid(
            video_path=video,
            output_path=out_path,
            final_w=res_w,
            final_h=res_h,
            overlap=max(0, args.tile_overlap),
            tile_cols=grid,
            tile_rows=grid,
            num_gpus=args.num_gpus,
            seed=args.seed,
            sp_size=args.sp_size,
            out_fps=args.out_fps,
        )
        if rc != 0:
            return rc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
