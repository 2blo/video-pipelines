from pydantic import BaseModel
from datetime import timedelta
from typing import List, Literal, Dict


class ManualDownload(BaseModel):
    link: str


class Path(BaseModel):
    path: str


Input = Path | ManualDownload


class Trim(BaseModel):
    op: Literal["trim"]
    start: timedelta
    end: timedelta


class Pipeline(BaseModel):
    input: Input
    steps: List[Trim]


class Config(BaseModel):
    windows_downloads_dir: str
    output_dir: str
    pipelines: Dict[str, Pipeline]
