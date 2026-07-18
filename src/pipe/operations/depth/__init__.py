from pydantic import BaseModel

from pipe.config import (
    Depth,
    DepthAnythingV2Variant,
    DepthAnythingV3StreamingVariant,
    DepthAnythingV3Variant,
    DepthCrafterVariant,
    DepthProVariant,
)
from pipe.operations.base import Operation
from pipe.operations.depth.depth_anything_v2 import DepthAnythingV2Operation
from pipe.operations.depth.depth_anything_v3 import DepthAnythingV3Operation
from pipe.operations.depth.depth_anything_v3_streaming import (
    DepthAnythingV3StreamingOperation,
)
from pipe.operations.depth.depth_crafter import DepthCrafterOperation
from pipe.operations.depth.depth_pro import DepthProOperation
from pipe.ops import ExecutedStep


class DepthOperation(Operation):
    def _impl(self, step: Depth):
        variant = step.variant
        if isinstance(variant, DepthAnythingV2Variant):
            return DepthAnythingV2Operation()
        if isinstance(variant, DepthAnythingV3Variant):
            return DepthAnythingV3Operation()
        if isinstance(variant, DepthAnythingV3StreamingVariant):
            return DepthAnythingV3StreamingOperation()
        if isinstance(variant, DepthCrafterVariant):
            return DepthCrafterOperation()
        if isinstance(variant, DepthProVariant):
            return DepthProOperation()
        raise ValueError(f"Unsupported depth variant: {variant.__class__.__name__}")

    def run(
        self, step: BaseModel, previous_step: ExecutedStep, output_path: str
    ) -> ExecutedStep:
        if not isinstance(step, Depth):
            raise ValueError(f"Expected Depth step, got {step.__class__.__name__}")
        return self._impl(step).run(step, previous_step, output_path)

    def kill(self, step: BaseModel) -> None:
        if not isinstance(step, Depth):
            raise ValueError(f"Expected Depth step, got {step.__class__.__name__}")
        self._impl(step).kill(step)

    def build(self, step: BaseModel) -> None:
        if not isinstance(step, Depth):
            raise ValueError(f"Expected Depth step, got {step.__class__.__name__}")
        self._impl(step).build(step)
