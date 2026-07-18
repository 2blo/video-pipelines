from pydantic import BaseModel

from pipe.config import EsrganUpscaleVariant, SeedVR2UpscaleVariant, Upscale
from pipe.operations.base import Operation
from pipe.operations.upscale.esrgan import EsrganUpscaleOperation
from pipe.operations.upscale.seedvr2 import SeedVR2UpscaleOperation
from pipe.ops import ExecutedStep


class UpscaleOperation(Operation):
    def _impl(self, step: Upscale):
        variant = step.variant
        if isinstance(variant, EsrganUpscaleVariant):
            return EsrganUpscaleOperation()
        if isinstance(variant, SeedVR2UpscaleVariant):
            return SeedVR2UpscaleOperation()
        raise ValueError(f"Unsupported upscale variant: {variant.__class__.__name__}")

    def run(
        self, step: BaseModel, previous_step: ExecutedStep, output_path: str
    ) -> ExecutedStep:
        if not isinstance(step, Upscale):
            raise ValueError(f"Expected Upscale step, got {step.__class__.__name__}")
        return self._impl(step).run(step, previous_step, output_path)

    def kill(self, step: BaseModel) -> None:
        if not isinstance(step, Upscale):
            raise ValueError(f"Expected Upscale step, got {step.__class__.__name__}")
        self._impl(step).kill(step)

    def build(self, step: BaseModel) -> None:
        if not isinstance(step, Upscale):
            raise ValueError(f"Expected Upscale step, got {step.__class__.__name__}")
        self._impl(step).build(step)
