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
    op: Literal["trim"]
    start: timedelta
    end: timedelta


class Pipeline(BaseModel):
    metadata: Dict[str, Any]
    input: Input
    steps: List[Trim]


class Show(BaseModel):
    path: str
    episode_pattern: str


class Config(BaseModel):
    shows: Dict[str, Show]
    windows_downloads_dir: str
    output_dir: str
    pipelines: Dict[str, Pipeline]
