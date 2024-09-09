from typing import Any, Callable, List, NamedTuple, TypeVar
from api.common.enums import ResourceType

import api.pipeline_repr as repr

# type def
Infer = List[float]
Callback = Callable[[bool, float, ResourceType], bool]

T = TypeVar("T")


class InferCache(NamedTuple):
    latency: float
    infers: Any


class Allocation(NamedTuple):
    app: repr.PipelineRepr
    is_succeed: bool
    amount: float

    def __str__(self):
       return f"{self.is_succeed} {self.amount}"

    def __repr__(self):
        return self.__str__()


class ResRequest(NamedTuple):
    app: repr.PipelineRepr
    diff: repr.InferDiff
    amount: float


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
