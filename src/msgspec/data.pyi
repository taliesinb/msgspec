from typing import Any, Literal, TypeAlias, overload

__all__ = [
    'Int',
    'Float',
    'UInt8',
    'UInt16',
    'UInt32',
    'UInt64',
    'Int8',
    'Int16',
    'Int32',
    'Int64',
    'Float32',
    'Float64',
    'Bool',
    'Scalar',
    'DType',
    'Tensor',
    'TensorHandle',
]

# The runtime builds these as distinct `TypeAliasType` objects (so msgspec can
# read a dtype name off each and tell them apart). Type checkers only need to
# know the underlying type they stand for, which the classic PEP 613 spelling
# conveys portably.

Int:  TypeAlias = int
Float: TypeAlias = float

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

# The dtype argument of `Tensor[shape, dtype]`: a dtype alias / builtin
# (`Float32`, `float`, ...), a dtype string, or `None` for "any".
_Dtype: TypeAlias = type[int] | type[float] | type[bool] | DType | None

# `Tensor` is subscripted at runtime via `TensorMeta.__getitem__` (below).
# Type checkers, however, only consult `TensorMeta.__getitem__` for subscripts
# in *value* position (`x = Tensor[3, Float32]`), where they do validate the
# arguments; in *annotation* position (`field: Tensor[3, Float32]`) they instead
# require `Tensor.__class_getitem__`, and there the arguments are accepted but
# not validated (`shape` being a value, not a type). So both are declared: the
# metaclass form (matching the runtime) gives value-position checking, and the
# `__class_getitem__` form makes annotations resolve.
class TensorMeta(type):
    ndims: int | None
    sizes: tuple[int | None, ...] | None
    dtype: DType | None
    # `Tensor[rank, dtype]` - a rank (or None for any rank)
    @overload
    def __getitem__(cls, args: tuple[int | None, _Dtype]) -> type[Tensor]: ...
    # `Tensor[(size, ...), dtype]` - explicit per-axis sizes (each may be None)
    @overload
    def __getitem__(
        cls, args: tuple[tuple[int | None, ...], _Dtype]
    ) -> type[Tensor]: ...

class Tensor(metaclass=TensorMeta):
    @overload
    def __class_getitem__(cls, args: tuple[int | None, _Dtype]) -> type[Tensor]: ...
    @overload
    def __class_getitem__(
        cls, args: tuple[tuple[int | None, ...], _Dtype]
    ) -> type[Tensor]: ...

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
