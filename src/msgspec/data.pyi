from typing import Any, Literal, TypeAlias, overload

__all__ = [
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

class TensorMeta(type):
    ndims: int | None
    sizes: tuple[int | None, ...] | None
    dtype: DType | None

class Tensor(metaclass=TensorMeta):
    # `Tensor[shape, dtype]`. Note: `shape` is a value (an int rank or a sizes
    # tuple), not a type, so full validation is limited by the type checker -
    # pyright accepts the subscript but does not check `__class_getitem__` args,
    # while mypy rejects value subscripts outright. The overloads document the
    # accepted forms and are honored by checkers that validate them.
    #
    # `Tensor[rank, dtype]` - a rank (or None for any rank)
    @overload
    def __class_getitem__(cls, args: tuple[int | None, _Dtype]) -> type[Tensor]: ...
    # `Tensor[(size, ...), dtype]` - explicit per-axis sizes (each may be None)
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
