from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from pydantic import BaseModel

if TYPE_CHECKING:
    from pipe.ops import ExecutedStep


class Operation(ABC):
    @abstractmethod
    def run(
        self,
        step: BaseModel,
        previous_step: "ExecutedStep",
        output_path: str,
    ) -> "ExecutedStep":
        raise NotImplementedError

    @abstractmethod
    def kill(self, step: BaseModel) -> None:
        raise NotImplementedError

    @abstractmethod
    def build(self, step: BaseModel) -> None:
        raise NotImplementedError
