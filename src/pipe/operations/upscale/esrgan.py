import os

from pydantic import BaseModel

from pipe.config import EsrganUpscaleVariant, Upscale
from pipe.ops import (
    ExecutedStep,
    _build_docker_image,
    _kill_docker_containers_by_image,
    get_docker_image,
    resolve_video_input_path,
    upscale_esrgan,
)


class EsrganUpscaleOperation:
    def run(
        self,
        step: BaseModel,
        previous_step: ExecutedStep,
        output_path: str,
    ) -> ExecutedStep:
        if not isinstance(step, Upscale):
            raise ValueError(f"Expected Upscale step, got {step.__class__.__name__}")
        if not isinstance(step.variant, EsrganUpscaleVariant):
            raise ValueError(
                f"Expected EsrganUpscaleVariant, got {step.variant.__class__.__name__}"
            )
        upscale_esrgan(
            input_path=previous_step.output_path,
            output_path=output_path,
            width=step.width,
        )
        resolved_input_path = resolve_video_input_path(previous_step.output_path)
        return ExecutedStep(
            output_path=output_path,
            extension=os.path.splitext(resolved_input_path)[1],
        )

    def kill(self, step: BaseModel) -> None:
        _kill_docker_containers_by_image(get_docker_image("esrgan"))

    def build(self, step: BaseModel) -> None:
        _build_docker_image("esrgan")
