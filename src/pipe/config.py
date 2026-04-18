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


class CopyTracks(BaseModel):
    type: Literal["copy_tracks"]
    source_path: str


FfmpegOperation = Annotated[Trim | CopyTracks | Encode, Field(discriminator="type")]


class Ffmpeg(BaseModel):
    type: Literal["ffmpeg"]
    operations: List[FfmpegOperation]


Step = Annotated[Ffmpeg | Interpolate | Upscale, Field(discriminator="type")]


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
