from pydantic import BaseModel
from datetime import timedelta
from typing import List, Literal


class Input(BaseModel):
    path: str


class Output(BaseModel):
    path: str


class Trim(BaseModel):
    op: Literal["trim"]
    start: timedelta
    end: timedelta


class Config(BaseModel):
    input: Input
    steps: List[Trim]
    output: Output
