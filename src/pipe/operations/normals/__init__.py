from pydantic import BaseModel

from pipe.config import DktNormalsVariant, NormalCrafterVariant, Normals
from pipe.operations.base import Operation
from pipe.operations.normals.dkt import DktNormalsOperation
from pipe.operations.normals.normal_crafter import NormalCrafterOperation
from pipe.ops import ExecutedStep


class NormalsOperation(Operation):
    def _impl(self, step: Normals):
        variant = step.variant
        if isinstance(variant, NormalCrafterVariant):
            return NormalCrafterOperation()
        if isinstance(variant, DktNormalsVariant):
            return DktNormalsOperation()
        raise ValueError(f"Unsupported normals variant: {variant.__class__.__name__}")

    def run(
        self, step: BaseModel, previous_step: ExecutedStep, output_path: str
    ) -> ExecutedStep:
        if not isinstance(step, Normals):
            raise ValueError(f"Expected Normals step, got {step.__class__.__name__}")
        return self._impl(step).run(step, previous_step, output_path)

    def kill(self, step: BaseModel) -> None:
        if not isinstance(step, Normals):
            raise ValueError(f"Expected Normals step, got {step.__class__.__name__}")
        self._impl(step).kill(step)

    def build(self, step: BaseModel) -> None:
        if not isinstance(step, Normals):
            raise ValueError(f"Expected Normals step, got {step.__class__.__name__}")
        self._impl(step).build(step)
