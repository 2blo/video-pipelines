from pydantic import BaseModel

from pipe.config import Ffmpeg
from pipe.operations.base import Operation
from pipe.ops import ExecutedStep, execute_ffmpeg


class FfmpegOperation(Operation):
    def run(
        self,
        step: BaseModel,
        previous_step: ExecutedStep,
        output_path: str,
    ) -> ExecutedStep:
        if not isinstance(step, Ffmpeg):
            raise ValueError(f"Expected Ffmpeg step, got {step.__class__.__name__}")
        return execute_ffmpeg(
            step=step, previous_step=previous_step, output_path=output_path
        )

    def kill(self, step: BaseModel) -> None:
        return

    def build(self, step: BaseModel) -> None:
        return
