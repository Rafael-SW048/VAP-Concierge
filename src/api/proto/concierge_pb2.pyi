from google.protobuf import empty_pb2 as _empty_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class AppInfo(_message.Message):
    __slots__ = ("container_id", "server_uri", "client_uri", "server_port", "is_faked", "pid")
    CONTAINER_ID_FIELD_NUMBER: _ClassVar[int]
    SERVER_URI_FIELD_NUMBER: _ClassVar[int]
    CLIENT_URI_FIELD_NUMBER: _ClassVar[int]
    SERVER_PORT_FIELD_NUMBER: _ClassVar[int]
    IS_FAKED_FIELD_NUMBER: _ClassVar[int]
    PID_FIELD_NUMBER: _ClassVar[int]
    container_id: str
    server_uri: str
    client_uri: str
    server_port: int
    is_faked: bool
    pid: int
    def __init__(self, container_id: _Optional[str] = ..., server_uri: _Optional[str] = ..., client_uri: _Optional[str] = ..., server_port: _Optional[int] = ..., is_faked: bool = ..., pid: _Optional[int] = ...) -> None: ...

class BaselineInfo(_message.Message):
    __slots__ = ("container_id", "f1_scores")
    CONTAINER_ID_FIELD_NUMBER: _ClassVar[int]
    F1_SCORES_FIELD_NUMBER: _ClassVar[int]
    container_id: str
    f1_scores: _containers.RepeatedScalarFieldContainer[float]
    def __init__(self, container_id: _Optional[str] = ..., f1_scores: _Optional[_Iterable[float]] = ...) -> None: ...

class PID(_message.Message):
    __slots__ = ("container_id", "pid", "resource_t")
    CONTAINER_ID_FIELD_NUMBER: _ClassVar[int]
    PID_FIELD_NUMBER: _ClassVar[int]
    RESOURCE_T_FIELD_NUMBER: _ClassVar[int]
    container_id: str
    pid: int
    resource_t: str
    def __init__(self, container_id: _Optional[str] = ..., pid: _Optional[int] = ..., resource_t: _Optional[str] = ...) -> None: ...

class ResRequest(_message.Message):
    __slots__ = ("container_id", "resource_t", "amount")
    CONTAINER_ID_FIELD_NUMBER: _ClassVar[int]
    RESOURCE_T_FIELD_NUMBER: _ClassVar[int]
    AMOUNT_FIELD_NUMBER: _ClassVar[int]
    container_id: str
    resource_t: str
    amount: float
    def __init__(self, container_id: _Optional[str] = ..., resource_t: _Optional[str] = ..., amount: _Optional[float] = ...) -> None: ...

class Status(_message.Message):
    __slots__ = ("is_succeed",)
    IS_SUCCEED_FIELD_NUMBER: _ClassVar[int]
    is_succeed: bool
    def __init__(self, is_succeed: bool = ...) -> None: ...
