from typing import Any, cast

from pydantic import BaseModel

from pipe.config import Depth, DepthCrafterVariant
from pipe.ops import (
    ExecutedStep,
    _build_docker_image,
    _kill_docker_containers_by_image,
    depth_crafter,
    get_docker_image,
)


class DepthCrafterOperation:
    def run(
        self,
        step: BaseModel,
        previous_step: ExecutedStep,
        output_path: str,
    ) -> ExecutedStep:
        if not isinstance(step, Depth):
            raise ValueError(f"Expected Depth step, got {step.__class__.__name__}")
        if not isinstance(step.variant, DepthCrafterVariant):
            raise ValueError(
                f"Expected DepthCrafterVariant, got {step.variant.__class__.__name__}"
            )
        variant = step.variant
        depth_crafter_fn = cast(Any, depth_crafter)
        depth_crafter_fn(
            input_path=previous_step.output_path,
            output_path=output_path,
            max_res=variant.max_res,
            process_length=variant.process_length,
            target_fps=variant.target_fps,
            max_megapixel_frames=variant.max_megapixel_frames,
            enable_chunk_hack=variant.enable_chunk_hack,
            save_exr=variant.save_exr,
        )
        return ExecutedStep(output_path=output_path, extension=".json")

    def kill(self, step: BaseModel) -> None:
        _kill_docker_containers_by_image(get_docker_image("depth_crafter"))

    def build(self, step: BaseModel) -> None:
        _build_docker_image("depth_crafter")
