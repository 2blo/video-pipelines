from datetime import timedelta
from typing import Annotated, Any, Dict, List, Literal

from pydantic import BaseModel, Field


class ManualDownload(BaseModel):
    link: str


class Path(BaseModel):
    path: str


class Episode(BaseModel):
    show: str
    season: int
    episode: int


Input = Path | ManualDownload | Episode


class Trim(BaseModel):
    type: Literal["trim"]
    start: timedelta
    end: timedelta


class Encode(BaseModel):
    type: Literal["encode"]
    codec: Literal["prores"]
    profile: Literal["proxy", "lt", "422", "hq", "4444", "4444xq"] = "422"


class Interpolate(BaseModel):
    type: Literal["interpolate"]
    fps: int


class Upscale(BaseModel):
    type: Literal["upscale"]
    width: int
    variant: "UpscaleVariant"


class EsrganUpscaleVariant(BaseModel):
    type: Literal["esrgan"]


class SeedVR2UpscaleVariant(BaseModel):
    type: Literal["seedvr2"]
    model: str = "seedvr2_ema_3b_fp8_e4m3fn.safetensors"
    batch_size: int = 5
    temporal_overlap: int = 3
    max_resolution: int = 2160
    blocks_to_swap: int = 16
    swap_io_components: bool = True
    dit_offload_device: Literal["none", "cpu"] = "cpu"
    vae_offload_device: Literal["none", "cpu"] = "cpu"
    tensor_offload_device: Literal["none", "cpu"] = "cpu"
    vae_encode_tiled: bool = True
    vae_decode_tiled: bool = True
    vae_encode_tile_size: int = 1024
    vae_encode_tile_overlap: int = 128
    vae_decode_tile_size: int = 1024
    vae_decode_tile_overlap: int = 128
    cache_dit: bool = True
    cache_vae: bool = True


UpscaleVariant = Annotated[
    EsrganUpscaleVariant | SeedVR2UpscaleVariant,
    Field(discriminator="type"),
]


class Colmap(BaseModel):
    type: Literal["colmap"]
    mapper: Literal["colmap", "glomap"] = "colmap"
    frame_rate: float | None = None
    max_frames: int | None = None
    image_format: Literal["jpg", "png"] = "jpg"
    max_image_size: int = 4096
    camera_mode: Literal["auto", "single"] = "single"
    sequential_overlap: int = 10
    loop_detection: bool = True
    min_num_matches: int = 15
    max_num_models: int = 1
    min_model_size: int = 10
    random_seed: int = 0
    use_gpu: bool = True
    extract_depth: bool = False


class DepthAnythingV2Variant(BaseModel):
    type: Literal["depth_anything_v2"]
    encoder: Literal["vits", "vitb", "vitl", "vitg"] = "vitl"
    input_size: int = 518
    precision: Literal["fp32", "fp16"] = "fp32"
    fast_resize_height: int | None = None
    temporal_smoothing_alpha: float = 0.0


class DepthAnythingV3Variant(BaseModel):
    type: Literal["depth_anything_v3"]
    model: Literal["small"] = "small"
    max_res: int = 2160


class DepthAnythingV3StreamingVariant(BaseModel):
    type: Literal["depth_anything_v3_streaming"]
    max_res: int | None = None
    fps: float | None = None
    device: Literal["auto", "cuda"] = "auto"
    chunk_size: int = 64
    overlap: int = 24
    loop_enable: bool = False
    save_depth_conf_result: bool = True
    delete_temp_files: bool = True
    align_lib: Literal["triton", "torch", "numba", "numpy"] = "torch"


class DepthCrafterVariant(BaseModel):
    type: Literal["depth_crafter"]
    max_res: int | None = None
    process_length: int | None = None
    target_fps: int | None = None
    max_megapixel_frames: float | None = None


class DepthProVariant(BaseModel):
    type: Literal["depth_pro"]
    precision: Literal["fp16", "fp32"] = "fp16"


DepthVariant = Annotated[
    DepthAnythingV2Variant
    | DepthAnythingV3Variant
    | DepthAnythingV3StreamingVariant
    | DepthCrafterVariant
    | DepthProVariant,
    Field(discriminator="type"),
]


class Depth(BaseModel):
    type: Literal["depth"]
    variant: DepthVariant


class NormalCrafterVariant(BaseModel):
    type: Literal["normal_crafter"]
    cpu_offload: Literal["model", "sequential"] = "model"
    unet_path: str = "Yanrui95/NormalCrafter"
    pre_train_path: str = "stabilityai/stable-video-diffusion-img2vid-xt"
    max_res: int | None = None
    process_length: int | None = None
    target_fps: int | None = None
    window_size: int = 14
    time_step_size: int = 10
    decode_chunk_size: int = 7


class DktNormalsVariant(BaseModel):
    type: Literal["dkt"]
    model_id: str = "Daniellesry/DKT-Normal-14B"
    height: int = 480
    width: int = 832
    num_inference_steps: int = 3
    window_size: int = 6
    overlap: int = 1


NormalsVariant = Annotated[
    NormalCrafterVariant | DktNormalsVariant,
    Field(discriminator="type"),
]


class Normals(BaseModel):
    type: Literal["normals"]
    variant: NormalsVariant


class NoOp(BaseModel):
    type: Literal["noop", "no_op"]


class CopyTracks(BaseModel):
    type: Literal["copy_tracks"]
    source_path: str


FfmpegOperation = Annotated[Trim | CopyTracks | Encode, Field(discriminator="type")]


class Ffmpeg(BaseModel):
    type: Literal["ffmpeg"]
    operations: List[FfmpegOperation]


BranchStep = Annotated[
    Ffmpeg | Interpolate | Upscale | Colmap | Depth | Normals | NoOp,
    Field(discriminator="type"),
]


class Branch(BaseModel):
    type: Literal["branch"]
    branches: List[BranchStep]


Step = Annotated[
    Ffmpeg
    | Interpolate
    | Upscale
    | Colmap
    | Depth
    | Normals
    | NoOp
    | Branch,
    Field(discriminator="type"),
]


class Pipeline(BaseModel):
    metadata: Dict[str, Any]
    input: Input
    steps: List[Step]


class Job(BaseModel):
    name: str
    pipelines: Dict[str, Pipeline]


class Show(BaseModel):
    path: str
    episode_pattern: str


class Config(BaseModel):
    full_refresh: bool = False
    shows: Dict[str, Show]
    windows_downloads_dir: str
    database_file_path: str
    artifact_dir: str
    output_dir: str
    job: Job
