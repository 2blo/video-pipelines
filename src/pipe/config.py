from datetime import timedelta
from pydantic import BaseModel
from typing import List, Literal, Dict


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


class Interpolate(BaseModel):
    type: Literal["interpolate"]
    fps: int


class Upscale(BaseModel):
    type: Literal["upscale"]
    width: int


class CopyTracks(BaseModel):
    type: Literal["copy_tracks"]
    source_path: str


class Pipeline(BaseModel):
    metadata: Dict[str, str]
    input: Input
    steps: List[Trim | Interpolate | Upscale | CopyTracks]


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
