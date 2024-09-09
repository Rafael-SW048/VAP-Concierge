from google.protobuf import empty_pb2 as _empty_pb2
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class ResourceChange(_message.Message):
    __slots__ = ("change", "can_proceed")
    CHANGE_FIELD_NUMBER: _ClassVar[int]
    CAN_PROCEED_FIELD_NUMBER: _ClassVar[int]
    change: float
    can_proceed: bool
    def __init__(self, change: _Optional[float] = ..., can_proceed: bool = ...) -> None: ...

class ResourceType(_message.Message):
    __slots__ = ("resource_t",)
    RESOURCE_T_FIELD_NUMBER: _ClassVar[int]
    resource_t: str
    def __init__(self, resource_t: _Optional[str] = ...) -> None: ...

class Diff(_message.Message):
    __slots__ = ("infer_diff", "latency_diff", "infer_diff_high", "infer_diff_low", "bw_lower_bound", "is_min_bw", "curr_frame", "curr_f1", "curr_budget")
    INFER_DIFF_FIELD_NUMBER: _ClassVar[int]
    LATENCY_DIFF_FIELD_NUMBER: _ClassVar[int]
    INFER_DIFF_HIGH_FIELD_NUMBER: _ClassVar[int]
    INFER_DIFF_LOW_FIELD_NUMBER: _ClassVar[int]
    BW_LOWER_BOUND_FIELD_NUMBER: _ClassVar[int]
    IS_MIN_BW_FIELD_NUMBER: _ClassVar[int]
    CURR_FRAME_FIELD_NUMBER: _ClassVar[int]
    CURR_F1_FIELD_NUMBER: _ClassVar[int]
    CURR_BUDGET_FIELD_NUMBER: _ClassVar[int]
    infer_diff: float
    latency_diff: float
    infer_diff_high: float
    infer_diff_low: float
    bw_lower_bound: float
    is_min_bw: bool
    curr_frame: int
    curr_f1: float
    curr_budget: float
    def __init__(self, infer_diff: _Optional[float] = ..., latency_diff: _Optional[float] = ..., infer_diff_high: _Optional[float] = ..., infer_diff_low: _Optional[float] = ..., bw_lower_bound: _Optional[float] = ..., is_min_bw: bool = ..., curr_frame: _Optional[int] = ..., curr_f1: _Optional[float] = ..., curr_budget: _Optional[float] = ...) -> None: ...

class Notification(_message.Message):
    __slots__ = ("is_succeed", "actual_amount", "resource_t", "can_clear_profiling_flag", "is_min_bw", "can_proceed")
    IS_SUCCEED_FIELD_NUMBER: _ClassVar[int]
    ACTUAL_AMOUNT_FIELD_NUMBER: _ClassVar[int]
    RESOURCE_T_FIELD_NUMBER: _ClassVar[int]
    CAN_CLEAR_PROFILING_FLAG_FIELD_NUMBER: _ClassVar[int]
    IS_MIN_BW_FIELD_NUMBER: _ClassVar[int]
    CAN_PROCEED_FIELD_NUMBER: _ClassVar[int]
    is_succeed: bool
    actual_amount: float
    resource_t: str
    can_clear_profiling_flag: bool
    is_min_bw: bool
    can_proceed: bool
    def __init__(self, is_succeed: bool = ..., actual_amount: _Optional[float] = ..., resource_t: _Optional[str] = ..., can_clear_profiling_flag: bool = ..., is_min_bw: bool = ..., can_proceed: bool = ...) -> None: ...

class IsNotified(_message.Message):
    __slots__ = ("notified",)
    NOTIFIED_FIELD_NUMBER: _ClassVar[int]
    notified: bool
    def __init__(self, notified: bool = ...) -> None: ...

class NewTokens(_message.Message):
    __slots__ = ("amount",)
    AMOUNT_FIELD_NUMBER: _ClassVar[int]
    amount: int
    def __init__(self, amount: _Optional[int] = ...) -> None: ...
