import os

from pydantic import BaseModel

from pipe.config import SeedVR2UpscaleVariant, Upscale
from pipe.ops import (
    ExecutedStep,
    _build_docker_image,
    _kill_docker_containers_by_image,
    get_docker_image,
    resolve_video_input_path,
    upscale_seedvr2,
)


class SeedVR2UpscaleOperation:
    def run(
        self,
        step: BaseModel,
        previous_step: ExecutedStep,
        output_path: str,
    ) -> ExecutedStep:
        if not isinstance(step, Upscale):
            raise ValueError(f"Expected Upscale step, got {step.__class__.__name__}")
        if not isinstance(step.variant, SeedVR2UpscaleVariant):
            raise ValueError(
                f"Expected SeedVR2UpscaleVariant, got {step.variant.__class__.__name__}"
            )
        upscale_seedvr2(
            input_path=previous_step.output_path,
            output_path=output_path,
            width=step.width,
            variant=step.variant,
        )
        resolved_input_path = resolve_video_input_path(previous_step.output_path)
        return ExecutedStep(
            output_path=output_path,
            extension=os.path.splitext(resolved_input_path)[1],
        )

    def kill(self, step: BaseModel) -> None:
        _kill_docker_containers_by_image(get_docker_image("seedvr2"))

    def build(self, step: BaseModel) -> None:
        _build_docker_image("seedvr2")
