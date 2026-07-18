from pydantic import BaseModel

from pipe.config import Depth, DepthAnythingV3StreamingVariant
from pipe.ops import (
    ExecutedStep,
    _build_docker_image,
    _kill_docker_containers_by_image,
    depth_anything_v3_streaming,
    get_docker_image,
)


class DepthAnythingV3StreamingOperation:
    def run(
        self,
        step: BaseModel,
        previous_step: ExecutedStep,
        output_path: str,
    ) -> ExecutedStep:
        if not isinstance(step, Depth):
            raise ValueError(f"Expected Depth step, got {step.__class__.__name__}")
        if not isinstance(step.variant, DepthAnythingV3StreamingVariant):
            raise ValueError(
                "Expected DepthAnythingV3StreamingVariant, "
                f"got {step.variant.__class__.__name__}"
            )
        variant = step.variant
        depth_anything_v3_streaming(
            input_path=previous_step.output_path,
            output_path=output_path,
            max_res=variant.max_res,
            fps=variant.fps,
            device=variant.device,
            chunk_size=variant.chunk_size,
            overlap=variant.overlap,
            loop_enable=variant.loop_enable,
            save_depth_conf_result=variant.save_depth_conf_result,
            delete_temp_files=variant.delete_temp_files,
            align_lib=variant.align_lib,
        )
        return ExecutedStep(output_path=output_path, extension=".json")

    def kill(self, step: BaseModel) -> None:
        _kill_docker_containers_by_image(
            get_docker_image("depth_anything_v3_streaming")
        )

    def build(self, step: BaseModel) -> None:
        _build_docker_image("depth_anything_v3_streaming")
