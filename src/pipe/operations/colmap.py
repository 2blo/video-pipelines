from pydantic import BaseModel

from pipe.config import Colmap
from pipe.operations.base import Operation
from pipe.ops import ExecutedStep, execute_colmap


class ColmapOperation(Operation):
    def run(
        self,
        step: BaseModel,
        previous_step: ExecutedStep,
        output_path: str,
    ) -> ExecutedStep:
        if not isinstance(step, Colmap):
            raise ValueError(f"Expected Colmap step, got {step.__class__.__name__}")
        return execute_colmap(
            step=step, previous_step=previous_step, output_path=output_path
        )

    def kill(self, step: BaseModel) -> None:
        return

    def build(self, step: BaseModel) -> None:
        return
