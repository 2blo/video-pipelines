from pydantic import BaseModel
from datetime import timedelta
from typing import List, Literal, Dict, Any


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


class Upscale(BaseModel):
    type: Literal["upscale"]
    width: int


class Pipeline(BaseModel):
    metadata: Dict[str, Any]
    input: Input
    steps: List[Trim | Upscale]


class Show(BaseModel):
    path: str
    episode_pattern: str


class Config(BaseModel):
    full_refresh: bool = False
    shows: Dict[str, Show]
    windows_downloads_dir: str
    artifact_dir: str
    output_dir: str
    pipelines: Dict[str, Pipeline]
