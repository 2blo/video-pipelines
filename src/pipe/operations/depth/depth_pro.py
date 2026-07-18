from pydantic import BaseModel

from pipe.config import Depth, DepthProVariant
from pipe.ops import (
    ExecutedStep,
    _build_docker_image,
    _kill_docker_containers_by_image,
    depth_pro,
    get_docker_image,
)


class DepthProOperation:
    def run(
        self,
        step: BaseModel,
        previous_step: ExecutedStep,
        output_path: str,
    ) -> ExecutedStep:
        if not isinstance(step, Depth):
            raise ValueError(f"Expected Depth step, got {step.__class__.__name__}")
        if not isinstance(step.variant, DepthProVariant):
            raise ValueError(
                f"Expected DepthProVariant, got {step.variant.__class__.__name__}"
            )
        variant = step.variant
        depth_pro(
            input_path=previous_step.output_path,
            output_path=output_path,
            precision=variant.precision,
        )
        return ExecutedStep(output_path=output_path, extension=".json")

    def kill(self, step: BaseModel) -> None:
        _kill_docker_containers_by_image(get_docker_image("depth_pro"))

    def build(self, step: BaseModel) -> None:
        _build_docker_image("depth_pro")
