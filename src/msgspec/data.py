from __future__ import annotations

from typing import Any, Literal, TypeAliasType, cast

__all__ = [  # noqa: F822  (TensorHandle is provided via module __getattr__)
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

type UInt8 = int   # turns into ScalarType('uint8')
type UInt16 = int
type UInt32 = int
type UInt64 = int
type Int8 = int
type Int16 = int
type Int32 = int
type Int64 = int
type Float32 = float
type Float64 = float
type Bool = bool

type Scalar = int | float | bool

type DType = Literal[
    'uint8',
    'uint16',
    'uint32',
    'uint64',
    'int8',
    'int16',
    'int32',
    'int64',
    'float32',
    'float64',
    'bool'
]

# ---

type NumAxes = int
type AxisSize = int | None
type AxisSizes = tuple[AxisSize, ...]

# -----

DTYPE_ALIASES: tuple[TypeAliasType, ...] = (
    UInt8,
    UInt16,
    UInt32,
    UInt64,
    Int8,
    Int16,
    Int32,
    Int64,
    Float32,
    Float64,
    Bool,
)

DTYPE_STRINGS: tuple[DType, ...] = (
  'uint8',
  'uint16',
  'uint32',
  'uint64',
  'int8',
  'int16',
  'int32',
  'int64',
  'float32',
  'float64',
  'bool'
)

def parse_dtype(dtype: type | TypeAliasType | None) -> DType | None:
    if dtype is None:
        return None
    if dtype is Scalar:
        return None
    if isinstance(dtype, TypeAliasType):
        name = dtype.__name__.lower()
        assert name in DTYPE_STRINGS
        return name
    if dtype is float:
        return 'float64'
    if dtype is int:
        return 'int64'
    if dtype is bool:
        return 'bool'
    if isinstance(dtype, str):
        assert dtype in DTYPE_STRINGS
        return cast(DType, dtype)
    raise TypeError()

def parse_shape(shape: AxisSizes | NumAxes | None) -> tuple[NumAxes | None, AxisSizes | None]:
    if isinstance(shape, int):
        return shape, None
    if isinstance(shape, tuple):
        assert all(isinstance(s, int) for s in shape)
        return len(shape), shape
    if shape is None:
        return None, None
    raise TypeError()

# ----

class TensorMeta(type):

    ndims: NumAxes | None
    sizes: AxisSizes | None
    dtype: DType | None

    def __getitem__(self, args: tuple[AxisSizes | NumAxes | None, type | TypeAliasType | None]) -> TensorMeta:
        assert isinstance(args, tuple)
        assert len(args) == 2
        if args in TYPE_CACHE:
            return TYPE_CACHE[args]
        s_arg, d_arg = args
        ndims, sizes = parse_shape(s_arg)
        dtype = parse_dtype(d_arg)
        tensor_t = TensorMeta.__new__(TensorMeta, 'Tensor', (Tensor,), dict(ndims=ndims, sizes=sizes, dtype=dtype))
        TYPE_CACHE[args] = tensor_t
        return tensor_t

    def __repr__(cls):
        ndims = cls.ndims
        sizes = cls.sizes
        dtype = cls.dtype
        return f'Tensor[{sizes or ndims}, {dtype!r}]'

TYPE_CACHE: dict[tuple[Any, ...], TensorMeta] = {}

# ---

class Tensor(metaclass=TensorMeta):
    ...

# ---

# `TensorHandle` is an opaque handle to a tensor (e.g. a numpy array). Its
# `native` field is something that supports the buffer protocol / memoryview.
# Once constructed, msgspec serializes it to a custom MessagePack extension
# (code 84 / 'T'): a self-describing payload of dtype + shape + raw bytes.
# On decode, `native` is a memoryview onto a fresh bytes object.
#
# It is implemented in the C extension (see `TensorHandle` in `_core.c`) and
# re-exported here lazily via `__getattr__` so that `_core` can import this
# module during its own initialization without a circular import.
#
# Milestone 2 (not yet implemented):
# - JSON encoding as a struct {'shape': ..., 'dtype': ..., 'data': base64}.
# - a `dec_tensor` hook: a function taking (shape, dtype, data: bytes) that
#   returns the caller's preferred tensor type (e.g. a numpy array).
# - automatically recognizing numpy arrays and wrapping them as TensorHandle,
#   without introducing a dependency on numpy.


def __getattr__(name):
    if name == 'TensorHandle':
        from ._core import TensorHandle
        return TensorHandle
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# ---
# examples:
# print(Tensor[(3, 3), UInt8])
# print(Tensor[None, None])
# print(Tensor[5, None])
# print(Tensor[5, UInt8])
# print(Tensor[5, float])
