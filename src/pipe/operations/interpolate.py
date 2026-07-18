from pydantic import BaseModel

from pipe.config import Interpolate
from pipe.operations.base import Operation
from pipe.ops import (
    ExecutedStep,
    _build_docker_image,
    _kill_docker_containers_by_image,
    execute_interpolate,
    get_docker_image,
)


class InterpolateOperation(Operation):
    def run(
        self,
        step: BaseModel,
        previous_step: ExecutedStep,
        output_path: str,
    ) -> ExecutedStep:
        if not isinstance(step, Interpolate):
            raise ValueError(
                f"Expected Interpolate step, got {step.__class__.__name__}"
            )
        return execute_interpolate(
            step=step,
            previous_step=previous_step,
            output_path=output_path,
        )

    def kill(self, step: BaseModel) -> None:
        _kill_docker_containers_by_image(get_docker_image("rife"))

    def build(self, step: BaseModel) -> None:
        _build_docker_image("rife")
