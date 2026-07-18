from pydantic import BaseModel

from pipe.config import NormalCrafterVariant, Normals
from pipe.ops import (
    ExecutedStep,
    _build_docker_image,
    _kill_docker_containers_by_image,
    get_docker_image,
    normal_crafter,
)


class NormalCrafterOperation:
    def run(
        self,
        step: BaseModel,
        previous_step: ExecutedStep,
        output_path: str,
    ) -> ExecutedStep:
        if not isinstance(step, Normals):
            raise ValueError(f"Expected Normals step, got {step.__class__.__name__}")
        if not isinstance(step.variant, NormalCrafterVariant):
            raise ValueError(
                f"Expected NormalCrafterVariant, got {step.variant.__class__.__name__}"
            )
        variant = step.variant
        normal_crafter(
            input_path=previous_step.output_path,
            output_path=output_path,
            cpu_offload=variant.cpu_offload,
            unet_path=variant.unet_path,
            pre_train_path=variant.pre_train_path,
            max_res=variant.max_res,
            process_length=variant.process_length,
            target_fps=variant.target_fps,
            window_size=variant.window_size,
            time_step_size=variant.time_step_size,
            decode_chunk_size=variant.decode_chunk_size,
        )
        return ExecutedStep(output_path=output_path, extension=".json")

    def kill(self, step: BaseModel) -> None:
        _kill_docker_containers_by_image(get_docker_image("normal_crafter"))

    def build(self, step: BaseModel) -> None:
        _build_docker_image("normal_crafter")
