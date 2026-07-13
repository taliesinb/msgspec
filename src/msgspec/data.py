from __future__ import annotations

from typing import Any, Literal, cast

try:  # `TypeAliasType` is stdlib on 3.12+, backported by typing_extensions before.
    from typing import TypeAliasType
except ImportError:  # pragma: no cover - Python < 3.12
    from typing_extensions import TypeAliasType

__all__ = [  # noqa: F822  (TensorHandle is provided via module __getattr__)
    'Int',
    'UInt',
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

Int = TypeAliasType("Int", int)      # generic signed integer: Int64 in arrays, bigint in ordinary annotations
UInt = TypeAliasType("UInt", int)    # generic unsigned integer: UInt64 in arrays, bigint in ordinary annotations
Float = TypeAliasType("Float", float)  # treated as Float64 in arrays, and number in ordinary annotations

# `X = TypeAliasType("X", value)` is the back-compatible spelling of the PEP 695
# `type X = value` statement (which is 3.12+ syntax). Each of these turns into a
# `ScalarType(<name>)` when inspected, e.g. `UInt8` -> `ScalarType('uint8')`.
UInt8 = TypeAliasType("UInt8", int)
UInt16 = TypeAliasType("UInt16", int)
UInt32 = TypeAliasType("UInt32", int)
UInt64 = TypeAliasType("UInt64", int)
Int8 = TypeAliasType("Int8", int)
Int16 = TypeAliasType("Int16", int)
Int32 = TypeAliasType("Int32", int)
Int64 = TypeAliasType("Int64", int)
Float32 = TypeAliasType("Float32", float)
Float64 = TypeAliasType("Float64", float)
Bool = TypeAliasType("Bool", bool)

# "Encode this scalar accurately": the marker union, so ints round-trip as
# bigint, floats as number, bools as bool on the JS side.
Scalar = TypeAliasType("Scalar", Int | Float | Bool)

DType = TypeAliasType(
    "DType",
    Literal[
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
        'bool',
    ],
)

# ---

NumAxes = TypeAliasType("NumAxes", int)
AxisSize = TypeAliasType("AxisSize", int | None)
AxisSizes = TypeAliasType("AxisSizes", tuple[AxisSize, ...])

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
        if name in DTYPE_STRINGS:
            return name
        # The generic markers: `Int` -> int64, `UInt` -> uint64, `Float` ->
        # float64. `Int`/`Float` fall through to their `__value__` below;
        # `UInt`'s value is plain `int`, so it needs an explicit mapping.
        if name == 'uint':
            return 'uint64'
        dtype = dtype.__value__
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
# Supported: msgpack + JSON encode/decode, the `dec_tensor` decode hook (both
# formats), automatic numpy recognition on encode, and the self-describing JSON
# object {"type": "tensor", "shape", "dtype", "data": base64}.


def _numpy_to_tensor_handle(arr):
    # Wrap a numpy array as a `TensorHandle`. Called from the C encoder when it
    # encounters a numpy array (numpy is never imported by msgspec itself - if
    # `arr` exists then numpy is already imported, so this is dependency-free).
    from ._core import TensorHandle

    dtype = arr.dtype
    name = dtype.name
    if name not in DTYPE_STRINGS:
        raise TypeError(f"Cannot encode numpy array with unsupported dtype {name!r}")
    # msgpack packs the raw buffer, so it must be C-contiguous and native-endian.
    if dtype.byteorder not in ('=', '|') or not arr.flags['C_CONTIGUOUS']:
        arr = arr.astype(dtype.newbyteorder('='), order='C', copy=False)
        arr = arr if arr.flags['C_CONTIGUOUS'] else arr.copy(order='C')
    return TensorHandle(arr, dtype=name, shape=tuple(arr.shape))


def _check_tensor(info, dtype, shape):
    # Validate a decoded tensor's `dtype`/`shape` against the `TensorInfo`
    # parsed from its annotation (or `None` for an unconstrained target).
    # Raises `ValueError` on mismatch; the C decoder wraps this in a
    # `msgspec.ValidationError` with the decode path appended.
    if info is None:
        return
    want_dtype = info.dtype
    if want_dtype is not None and dtype != want_dtype:
        raise ValueError(f"Expected tensor of dtype {want_dtype!r}, got {dtype!r}")
    if shape is None:
        return
    want_ndims = info.ndims
    want_sizes = info.sizes
    if want_ndims is not None and len(shape) != want_ndims:
        raise ValueError(
            f"Expected tensor of rank {want_ndims}, got rank {len(shape)}"
        )
    if want_sizes is not None:
        if len(shape) != len(want_sizes):
            raise ValueError(
                f"Expected tensor of rank {len(want_sizes)}, got rank {len(shape)}"
            )
        for axis, (want, got) in enumerate(zip(want_sizes, shape)):
            if want is not None and want != got:
                raise ValueError(
                    f"Expected tensor with axis {axis} of size {want}, got {got}"
                )


def _json_object_to_tensor(obj, dec_tensor, info):
    # Convert a decoded JSON tensor object {"type", "shape", "dtype", "data"}
    # into a TensorHandle (or, if `dec_tensor` is given, the caller's tensor),
    # validating against `info` first. Called from the C JSON decoder.
    import base64

    data = obj.get("data")
    raw = base64.b64decode(data) if data is not None else b""
    shape = obj.get("shape")
    shape = tuple(shape) if shape is not None else None
    dtype = obj.get("dtype")
    _check_tensor(info, dtype, shape)
    if dec_tensor is not None:
        return dec_tensor(shape, dtype, raw)
    from ._core import TensorHandle

    return TensorHandle(memoryview(raw), dtype, shape)


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
