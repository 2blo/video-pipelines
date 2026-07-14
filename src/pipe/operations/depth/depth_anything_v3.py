from pydantic import BaseModel

from pipe.config import Depth, DepthAnythingV3Variant
from pipe.ops import (
    ExecutedStep,
    _build_docker_image,
    _kill_docker_containers_by_image,
    depth_anything_v3,
    get_docker_image,
)


class DepthAnythingV3Operation:
    def run(
        self,
        step: BaseModel,
        previous_step: ExecutedStep,
        output_path: str,
    ) -> ExecutedStep:
        if not isinstance(step, Depth):
            raise ValueError(f"Expected Depth step, got {step.__class__.__name__}")
        if not isinstance(step.variant, DepthAnythingV3Variant):
            raise ValueError(
                f"Expected DepthAnythingV3Variant, got {step.variant.__class__.__name__}"
            )
        variant = step.variant
        depth_anything_v3(
            input_path=previous_step.output_path,
            output_path=output_path,
            model=variant.model,
            max_res=variant.max_res,
        )
        return ExecutedStep(output_path=output_path, extension=".json")

    def kill(self, step: BaseModel) -> None:
        _kill_docker_containers_by_image(get_docker_image("depth_anything_v3"))

    def build(self, step: BaseModel) -> None:
        _build_docker_image("depth_anything_v3")
