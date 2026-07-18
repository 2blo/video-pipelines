import os
from typing import Any, cast

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
        variant = step.variant
        upscale_esrgan_fn = cast(Any, upscale_esrgan)
        upscale_esrgan_fn(
            input_path=previous_step.output_path,
            output_path=output_path,
            width=step.width,
            output_mode=variant.output_mode,
            video_codec=variant.video_codec,
            prores_profile=variant.prores_profile,
            x264_crf=variant.x264_crf,
            x264_preset=variant.x264_preset,
            exr_type=variant.exr_type,
            exr_compression=variant.exr_compression,
        )
        if variant.output_mode == "exr_sequence":
            return ExecutedStep(output_path=output_path, extension=".json")
        resolved_input_path = resolve_video_input_path(previous_step.output_path)
        return ExecutedStep(
            output_path=output_path,
            extension=os.path.splitext(resolved_input_path)[1],
        )

    def kill(self, step: BaseModel) -> None:
        _kill_docker_containers_by_image(get_docker_image("esrgan"))

    def build(self, step: BaseModel) -> None:
        _build_docker_image("esrgan")
