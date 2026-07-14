from pydantic import BaseModel

from pipe.config import NoOp
from pipe.operations.base import Operation
from pipe.ops import ExecutedStep


class NoOpOperation(Operation):
    def run(
        self,
        step: BaseModel,
        previous_step: ExecutedStep,
        output_path: str,
    ) -> ExecutedStep:
        if not isinstance(step, NoOp):
            raise ValueError(f"Expected NoOp step, got {step.__class__.__name__}")
        return ExecutedStep(
            output_path=previous_step.output_path,
            extension=previous_step.extension,
        )

    def kill(self, step: BaseModel) -> None:
        return

    def build(self, step: BaseModel) -> None:
        return
