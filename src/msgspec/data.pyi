from typing import Any, Literal, TypeAlias

# The runtime builds these as distinct `TypeAliasType` objects (so msgspec can
# read a dtype name off each and tell them apart). Type checkers only need to
# know the underlying type they stand for, which the classic PEP 613 spelling
# conveys portably.
UInt8: TypeAlias = int
UInt16: TypeAlias = int
UInt32: TypeAlias = int
UInt64: TypeAlias = int
Int8: TypeAlias = int
Int16: TypeAlias = int
Int32: TypeAlias = int
Int64: TypeAlias = int
Float32: TypeAlias = float
Float64: TypeAlias = float
Bool: TypeAlias = bool

Scalar: TypeAlias = int | float | bool

DType: TypeAlias = Literal[
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "int8",
    "int16",
    "int32",
    "int64",
    "float32",
    "float64",
    "bool",
]

DTYPE_ALIASES: tuple[Any, ...]
DTYPE_STRINGS: tuple[DType, ...]

def parse_dtype(dtype: Any) -> DType | None: ...
def parse_shape(
    shape: Any,
) -> tuple[int | None, tuple[int | None, ...] | None]: ...

class TensorMeta(type):
    ndims: int | None
    sizes: tuple[int | None, ...] | None
    dtype: DType | None
    def __getitem__(cls, args: Any) -> type[Tensor]: ...

class Tensor(metaclass=TensorMeta):
    def __class_getitem__(cls, item: Any) -> type[Tensor]: ...

class TensorHandle:
    native: Any
    dtype: str | None
    shape: tuple[int, ...] | None
    def __init__(
        self,
        native: Any,
        dtype: str | None = None,
        shape: tuple[int, ...] | None = None,
    ) -> None: ...
