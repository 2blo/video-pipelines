from pydantic import BaseModel

from pipe.config import DktNormalsVariant, Normals
from pipe.ops import (
    ExecutedStep,
    _build_docker_image,
    _kill_docker_containers_by_image,
    dkt_normals,
    get_docker_image,
)


class DktNormalsOperation:
    def run(
        self,
        step: BaseModel,
        previous_step: ExecutedStep,
        output_path: str,
    ) -> ExecutedStep:
        if not isinstance(step, Normals):
            raise ValueError(f"Expected Normals step, got {step.__class__.__name__}")
        if not isinstance(step.variant, DktNormalsVariant):
            raise ValueError(
                f"Expected DktNormalsVariant, got {step.variant.__class__.__name__}"
            )
        variant = step.variant
        dkt_normals(
            input_path=previous_step.output_path,
            output_path=output_path,
            model_id=variant.model_id,
            height=variant.height,
            width=variant.width,
            num_inference_steps=variant.num_inference_steps,
            window_size=variant.window_size,
            overlap=variant.overlap,
        )
        return ExecutedStep(output_path=output_path, extension=".json")

    def kill(self, step: BaseModel) -> None:
        _kill_docker_containers_by_image(get_docker_image("dkt_normal"))

    def build(self, step: BaseModel) -> None:
        _build_docker_image("dkt_normal")
