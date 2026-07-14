from pydantic import BaseModel

from pipe.config import Depth, DepthAnythingV2Variant
from pipe.ops import (
    ExecutedStep,
    _build_docker_image,
    _kill_docker_containers_by_image,
    depth_anything_v2,
    get_docker_image,
)


class DepthAnythingV2Operation:
    def run(
        self,
        step: BaseModel,
        previous_step: ExecutedStep,
        output_path: str,
    ) -> ExecutedStep:
        if not isinstance(step, Depth):
            raise ValueError(f"Expected Depth step, got {step.__class__.__name__}")
        if not isinstance(step.variant, DepthAnythingV2Variant):
            raise ValueError(
                f"Expected DepthAnythingV2Variant, got {step.variant.__class__.__name__}"
            )
        variant = step.variant
        depth_anything_v2(
            input_path=previous_step.output_path,
            output_path=output_path,
            encoder=variant.encoder,
            input_size=variant.input_size,
            precision=variant.precision,
            fast_resize_height=variant.fast_resize_height,
            temporal_smoothing_alpha=variant.temporal_smoothing_alpha,
        )
        return ExecutedStep(output_path=output_path, extension=".json")

    def kill(self, step: BaseModel) -> None:
        _kill_docker_containers_by_image(get_docker_image("depth_anything_v2"))

    def build(self, step: BaseModel) -> None:
        _build_docker_image("depth_anything_v2")
