import collections
import datetime
import decimal
import enum
import sys
import typing
import uuid
from collections import namedtuple
from copy import deepcopy
from dataclasses import dataclass, field
from typing import (
    Annotated,
    Any,
    Dict,
    Final,
    FrozenSet,
    Generic,
    List,
    Literal,
    NamedTuple,
    NewType,
    Set,
    Tuple,
    TypedDict,
    TypeVar,
    Union,
)

import pytest

import msgspec
import msgspec.inspect as mi
from msgspec import Meta

from .utils import py315_or_later_only, temp_module

if sys.version_info >= (3, 15):
    # This is needed for `ruff` to recognize `frozendict` name
    # and to not raise `F821`:
    from builtins import frozendict

PY312 = sys.version_info[:2] >= (3, 12)
py312_plus = pytest.mark.skipif(not PY312, reason="3.12+ only")

T = TypeVar("T")


@pytest.mark.parametrize(
    "a,b,sol",
    [
        (
            {"a": {"b": {"c": 1}}},
            {"a": {"b": {"d": 2}}},
            {"a": {"b": {"c": 1, "d": 2}}},
        ),
        ({"a": {"b": {"c": 1}}}, {"a": {"b": 2}}, {"a": {"b": 2}}),
        ({"a": [1, 2]}, {"a": [3, 4]}, {"a": [1, 2, 3, 4]}),
        ({"a": {"b": 1}}, {"a2": 3}, {"a": {"b": 1}, "a2": 3}),
        ({"a": 1}, {}, {"a": 1}),
    ],
)
def test_merge_json(a, b, sol):
    a_orig = deepcopy(a)
    b_orig = deepcopy(b)
    res = mi._merge_json(a, b)
    assert res == sol
    assert a == a_orig
    assert b == b_orig


def test_inspect_module_dir():
    assert mi.__dir__() == mi.__all__


def test_any():
    assert mi.type_info(Any) == mi.AnyType()


def test_typevar():
    assert mi.type_info(T) == mi.AnyType()


def test_bound_typevar():
    T = TypeVar("T", bound=int | str)
    assert mi.type_info(T) == mi.UnionType((mi.IntType(), mi.StrType()))


def test_none():
    assert mi.type_info(None) == mi.NoneType()


def test_bool():
    assert mi.type_info(bool) == mi.BoolType()


@pytest.mark.parametrize(
    "kw", [{}, dict(ge=2), dict(gt=2), dict(le=2), dict(lt=2), dict(multiple_of=2)]
)
@pytest.mark.parametrize("typ, info_type", [(int, mi.IntType), (float, mi.FloatType)])
def test_numeric(kw, typ, info_type):
    if kw:
        typ = Annotated[typ, Meta(**kw)]
    assert mi.type_info(typ) == info_type(**kw)


@pytest.mark.parametrize(
    "kw",
    [{}, dict(pattern="[a-z]*"), dict(min_length=0), dict(max_length=3)],
)
def test_string(kw):
    typ = str
    if kw:
        typ = Annotated[typ, Meta(**kw)]
    assert mi.type_info(typ) == mi.StrType(**kw)


@pytest.mark.parametrize(
    "kw",
    [{}, dict(min_length=0), dict(max_length=3)],
)
@pytest.mark.parametrize(
    "typ, info_type",
    [
        (bytes, mi.BytesType),
        (bytearray, mi.ByteArrayType),
        (memoryview, mi.MemoryViewType),
    ],
)
def test_binary(kw, typ, info_type):
    if kw:
        typ = Annotated[typ, Meta(**kw)]
    assert mi.type_info(typ) == info_type(**kw)


@pytest.mark.parametrize(
    "kw",
    [{}, dict(tz=None), dict(tz=True), dict(tz=False)],
)
def test_datetime(kw):
    typ = datetime.datetime
    if kw:
        typ = Annotated[typ, Meta(**kw)]
    assert mi.type_info(typ) == mi.DateTimeType(**kw)


@pytest.mark.parametrize(
    "kw",
    [{}, dict(tz=None), dict(tz=True), dict(tz=False)],
)
def test_time(kw):
    typ = datetime.time
    if kw:
        typ = Annotated[typ, Meta(**kw)]
    assert mi.type_info(typ) == mi.TimeType(**kw)


def test_date():
    assert mi.type_info(datetime.date) == mi.DateType()


def test_timedelta():
    assert mi.type_info(datetime.timedelta) == mi.TimeDeltaType()


def test_uuid():
    assert mi.type_info(uuid.UUID) == mi.UUIDType()


def test_decimal():
    assert mi.type_info(decimal.Decimal) == mi.DecimalType()


def test_raw():
    assert mi.type_info(msgspec.Raw) == mi.RawType()


def test_msgpack_ext():
    assert mi.type_info(msgspec.msgpack.Ext) == mi.ExtType()


def test_newtype():
    UserId = NewType("UserId", str)
    assert mi.type_info(UserId) == mi.StrType()
    assert mi.type_info(Annotated[UserId, Meta(max_length=10)]) == mi.StrType(
        max_length=10
    )

    # Annotated in NewType
    UserId = NewType("UserId", Annotated[str, Meta(max_length=10)])
    assert mi.type_info(UserId) == mi.StrType(max_length=10)
    assert mi.type_info(Annotated[UserId, Meta(min_length=2)]) == mi.StrType(
        min_length=2, max_length=10
    )

    # NewType in NewType
    UserId2 = NewType("UserId2", UserId)
    assert mi.type_info(UserId2) == mi.StrType(max_length=10)
    assert mi.type_info(Annotated[UserId2, Meta(min_length=2)]) == mi.StrType(
        min_length=2, max_length=10
    )


@py312_plus
@pytest.mark.parametrize(
    "src, typ",
    [
        ("type Ex = str | None", str | None),
        ("type Ex[T] = tuple[T, int]", tuple[Any, int]),
        ("type Temp[T] = tuple[T, int]; Ex = Temp[str]", tuple[str, int]),
    ],
)
def test_typealias(src, typ):
    with temp_module(src) as mod:
        assert mi.type_info(mod.Ex) == mi.type_info(typ)


def test_newtype_aliases_true():
    Pixels = NewType("Pixels", int)
    # Default (aliases=False) still resolves through the alias.
    assert mi.type_info(Pixels) == mi.IntType()
    # aliases=True preserves the alias (as `cls`) and its resolved value.
    assert mi.type_info(Pixels, aliases=True) == mi.AliasType(Pixels, mi.IntType())


def test_alias_cache_does_not_leak_between_calls():
    # aliases=True and aliases=False use independent (per-call) caches, so one
    # never affects the other regardless of order.
    Pixels = NewType("Pixels", int)
    assert isinstance(mi.type_info(Pixels, aliases=True), mi.AliasType)
    assert mi.type_info(Pixels) == mi.IntType()
    assert mi.type_info(Pixels, aliases=False) == mi.IntType()
    assert isinstance(mi.type_info(Pixels, aliases=True), mi.AliasType)


def test_newtype_of_struct_aliases_true():
    class Point(msgspec.Struct):
        x: int

    Ref = NewType("Ref", Point)
    info = mi.type_info(Ref, aliases=True)
    assert isinstance(info, mi.AliasType)
    assert info.cls is Ref
    assert isinstance(info.value, mi.StructType)
    assert info.value.cls is Point


def test_alias_nested_and_multi():
    Pixels = NewType("Pixels", int)
    (list_info,) = mi.multi_type_info([list[Pixels]], aliases=True)
    assert list_info == mi.ListType(mi.AliasType(Pixels, mi.IntType()))


@py312_plus
@pytest.mark.parametrize(
    "src, value",
    [
        ("type Ex = int", mi.IntType()),
        ("type Ex = str | None", mi.UnionType((mi.StrType(), mi.NoneType()))),
        ("type Ex = list[int]", mi.ListType(mi.IntType())),
    ],
)
def test_typealias_aliases_true(src, value):
    with temp_module(src) as mod:
        # Default resolves transparently...
        assert mi.type_info(mod.Ex) == value
        # ...aliases=True preserves the alias itself as `cls`.
        info = mi.type_info(mod.Ex, aliases=True)
        assert info == mi.AliasType(mod.Ex, value)
        assert info.cls is mod.Ex


@py312_plus
def test_typealias_generic_specializations_distinct():
    with temp_module("type Vec[T] = list[T]") as mod:
        ints = mi.type_info(mod.Vec[int], aliases=True)
        strs = mi.type_info(mod.Vec[str], aliases=True)
        # Distinct specializations: distinct `cls`, distinct values.
        assert ints.cls == mod.Vec[int]
        assert strs.cls == mod.Vec[str]
        assert ints.value == mi.ListType(mi.IntType())
        assert strs.value == mi.ListType(mi.StrType())
        assert ints != strs


def test_data_scalar_aliases():
    from msgspec import data as md

    # Integer/float markers carry their format inline on IntType/FloatType.
    assert mi.type_info(md.UInt8) == mi.IntType(itype="uint8")
    assert mi.type_info(md.Int64) == mi.IntType(itype="int64")
    assert mi.type_info(md.Float32) == mi.FloatType(ftype="float32")
    assert mi.type_info(md.Float64) == mi.FloatType(ftype="float64")
    # `Bool` is just a bool.
    assert mi.type_info(md.Bool) == mi.BoolType()
    # The generic markers.
    assert mi.type_info(md.Int) == mi.IntType(itype="int")
    assert mi.type_info(md.UInt) == mi.IntType(itype="uint")
    assert mi.type_info(md.Float) == mi.FloatType(ftype="float")
    # `Scalar` is the accurate-scalar union `Int | Float | Bool`.
    assert mi.type_info(md.Scalar) == mi.UnionType(
        (mi.IntType(itype="int"), mi.FloatType(ftype="float"), mi.BoolType())
    )


def test_data_int_float_in_tensors():
    from msgspec import data as md

    assert mi.type_info(md.Tensor[3, md.Int]) == mi.TensorType(
        ndims=3, sizes=None, dtype="int64"
    )
    assert mi.type_info(md.Tensor[(2, 2), md.Float]) == mi.TensorType(
        ndims=2, sizes=(2, 2), dtype="float64"
    )


def test_data_scalar_alias_recognized_regardless_of_aliases_flag():
    from msgspec import data as md

    # data scalars take precedence over generic alias handling.
    assert mi.type_info(md.Int32, aliases=True) == mi.IntType(itype="int32")


def test_unrelated_alias_not_treated_as_scalar():
    # A user's own alias that merely resolves to int/float is untouched.
    Foo = NewType("Foo", int)
    assert mi.type_info(Foo) == mi.IntType()


def test_data_tensor_metas():
    from msgspec import data as md

    assert mi.type_info(md.Tensor[3, md.Float32]) == mi.TensorType(
        ndims=3, sizes=None, dtype="float32"
    )
    assert mi.type_info(md.Tensor[(3, 3), md.UInt8]) == mi.TensorType(
        ndims=2, sizes=(3, 3), dtype="uint8"
    )
    assert mi.type_info(md.Tensor[5, None]) == mi.TensorType(
        ndims=5, sizes=None, dtype=None
    )
    # Bare Tensor -> fully-unspecified tensor.
    assert mi.type_info(md.Tensor) == mi.TensorType(
        ndims=None, sizes=None, dtype=None
    )


def test_data_types_as_struct_fields():
    from msgspec import data as md

    class Layer(msgspec.Struct):
        weights: md.Tensor[3, md.Float32]
        bias: md.Scalar
        count: md.Int64

    info = mi.type_info(Layer)
    assert info.fields[0].type == mi.TensorType(ndims=3, sizes=None, dtype="float32")
    assert info.fields[1].type == mi.UnionType(
        (mi.IntType(itype="int"), mi.FloatType(ftype="float"), mi.BoolType())
    )
    assert info.fields[2].type == mi.IntType(itype="int64")


def test_final():
    cases = [
        (int, mi.IntType()),
        (Annotated[int, Meta(ge=0)], mi.IntType(ge=0)),
        (NewType("UserId", Annotated[int, Meta(ge=0)]), mi.IntType(ge=0)),
    ]
    for typ, sol in cases:

        class Ex(msgspec.Struct):
            x: Final[typ]

        info = mi.type_info(Ex)
        assert info.fields[0].type == sol


def test_custom():
    assert mi.type_info(complex) == mi.CustomType(complex)


@pytest.mark.parametrize(
    "kw",
    [{}, dict(min_length=0), dict(max_length=3)],
)
@pytest.mark.parametrize(
    "typ, info_type",
    [
        (list, mi.ListType),
        (tuple, mi.VarTupleType),
        (set, mi.SetType),
        (frozenset, mi.FrozenSetType),
        (List, mi.ListType),
        (Tuple, mi.VarTupleType),
        (Set, mi.SetType),
        (FrozenSet, mi.FrozenSetType),
    ],
)
@pytest.mark.parametrize("has_item_type", [False, True])
def test_sequence(kw, typ, info_type, has_item_type):
    if has_item_type:
        item_type = mi.IntType()
        if info_type is mi.VarTupleType:
            typ = typ[int, ...]
        else:
            typ = typ[int]
    else:
        item_type = mi.AnyType()

    if kw:
        typ = Annotated[typ, Meta(**kw)]

    sol = info_type(item_type=item_type, **kw)
    assert mi.type_info(typ) == sol


@pytest.mark.parametrize("typ", [Tuple, tuple])
def test_tuple(typ):
    assert mi.type_info(typ[()]) == mi.TupleType(())
    assert mi.type_info(typ[int]) == mi.TupleType((mi.IntType(),))
    assert mi.type_info(typ[int, float]) == mi.TupleType((mi.IntType(), mi.FloatType()))


@pytest.mark.parametrize("typ", [Dict, dict])
@pytest.mark.parametrize("kw", [{}, dict(min_length=0), dict(max_length=3)])
@pytest.mark.parametrize("has_args", [False, True])
def test_dict(typ, kw, has_args):
    if has_args:
        typ = typ[int, float]
        key = mi.IntType()
        val = mi.FloatType()
    else:
        key = val = mi.AnyType()
    if kw:
        typ = Annotated[typ, Meta(**kw)]
    sol = mi.DictType(key_type=key, value_type=val, **kw)
    assert mi.type_info(typ) == sol


@py315_or_later_only
@pytest.mark.parametrize("kw", [{}, dict(min_length=0), dict(max_length=3)])
@pytest.mark.parametrize("has_args", [False, True])
def test_frozendict(kw, has_args):
    if has_args:
        typ = frozendict[int, float]
        key = mi.IntType()
        val = mi.FloatType()
    else:
        typ = frozendict
        key = val = mi.AnyType()
    if kw:
        typ = Annotated[typ, Meta(**kw)]
    sol = mi.FrozenDictType(key_type=key, value_type=val, **kw)
    assert mi.type_info(typ) == sol


@pytest.mark.parametrize(
    "typ",
    [
        typing.Collection,
        typing.MutableSequence,
        typing.Sequence,
        collections.abc.Collection,
        collections.abc.MutableSequence,
        collections.abc.Sequence,
        typing.MutableSet,
        typing.AbstractSet,
        collections.abc.MutableSet,
        collections.abc.Set,
    ],
)
def test_abstract_sequence(typ):
    if "Set" in str(typ):
        col_type = mi.SetType
    else:
        col_type = mi.ListType

    assert mi.type_info(typ) == col_type(mi.AnyType())
    assert mi.type_info(typ[int]) == col_type(mi.IntType())


@pytest.mark.parametrize(
    "typ",
    [
        typing.MutableMapping,
        typing.Mapping,
        collections.abc.MutableMapping,
        collections.abc.Mapping,
    ],
)
def test_abstract_mapping(typ):
    assert mi.type_info(typ) == mi.DictType(mi.AnyType(), mi.AnyType())
    assert mi.type_info(typ[str, int]) == mi.DictType(mi.StrType(), mi.IntType())


@pytest.mark.parametrize("use_union_operator", [False, True])
def test_union(use_union_operator):
    if use_union_operator:
        typ = int | str
    else:
        typ = Union[int, str]

    sol = mi.UnionType((mi.IntType(), mi.StrType()))
    assert mi.type_info(typ) == sol

    assert not sol.includes_none
    assert mi.type_info(Union[int, None]).includes_none
    assert mi.type_info(int | None).includes_none


def test_int_literal():
    assert mi.type_info(Literal[3, 1, 2]) == mi.LiteralType((1, 2, 3))


def test_str_literal():
    assert mi.type_info(Literal["c", "a", "b"]) == mi.LiteralType(("a", "b", "c"))


def test_bool_literal():
    assert mi.type_info(Literal[True]) == mi.LiteralType((True,))
    assert mi.type_info(Literal[True, False]) == mi.LiteralType((False, True))


def test_mixed_literal():
    # Literals may mix value types; `type_info` shouldn't crash trying to sort
    # values of incomparable types (gh#1018).
    assert mi.type_info(Literal[1, None]) == mi.LiteralType((None, 1))
    assert mi.type_info(Literal[True, "yes"]) == mi.LiteralType((True, "yes"))


def test_int_enum():
    class Example(enum.IntEnum):
        B = 3
        A = 2

    assert mi.type_info(Example) == mi.EnumType(Example)


def test_enum():
    class Example(enum.Enum):
        B = "z"
        A = "y"

    assert mi.type_info(Example) == mi.EnumType(Example)


@pytest.mark.parametrize(
    "kw",
    [
        {},
        {"array_like": True},
        {"forbid_unknown_fields": True},
        {"tag": "Example", "tag_field": "type"},
    ],
)
def test_struct(kw):
    def factory():
        return "foo"

    class Example(msgspec.Struct, **kw):
        x: int
        y: int = 0
        z: int = msgspec.field(default_factory=factory)

    sol = mi.StructType(
        cls=Example,
        fields=(
            mi.Field(name="x", encode_name="x", type=mi.IntType()),
            mi.Field(
                name="y", encode_name="y", type=mi.IntType(), required=False, default=0
            ),
            mi.Field(
                name="z",
                encode_name="z",
                type=mi.IntType(),
                required=False,
                default_factory=factory,
            ),
        ),
        **kw,
    )
    assert mi.type_info(Example) == sol


def test_struct_no_fields():
    class Example(msgspec.Struct):
        pass

    sol = mi.StructType(Example, fields=())
    assert mi.type_info(Example) == sol


def test_struct_keyword_only():
    class Example(msgspec.Struct, kw_only=True):
        a: int
        b: int = 1
        c: int
        d: int = 2

    sol = mi.StructType(
        Example,
        fields=(
            mi.Field("a", "a", mi.IntType()),
            mi.Field("b", "b", mi.IntType(), required=False, default=1),
            mi.Field("c", "c", mi.IntType()),
            mi.Field("d", "d", mi.IntType(), required=False, default=2),
        ),
    )
    assert mi.type_info(Example) == sol


def test_struct_encode_name():
    class Example(msgspec.Struct, rename="camel"):
        field_one: int
        field_two: int

    sol = mi.StructType(
        Example,
        fields=(
            mi.Field("field_one", "fieldOne", mi.IntType()),
            mi.Field("field_two", "fieldTwo", mi.IntType()),
        ),
    )
    assert mi.type_info(Example) == sol


def test_generic_struct():
    class Example(msgspec.Struct, Generic[T]):
        a: T
        b: list[T]

    sol = mi.StructType(
        Example,
        fields=(
            mi.Field("a", "a", mi.AnyType()),
            mi.Field("b", "b", mi.ListType(mi.AnyType())),
        ),
    )
    assert mi.type_info(Example) == sol

    sol = mi.StructType(
        Example[int],
        fields=(
            mi.Field("a", "a", mi.IntType()),
            mi.Field("b", "b", mi.ListType(mi.IntType())),
        ),
    )
    assert mi.type_info(Example[int]) == sol


def test_typing_namedtuple():
    class Example(NamedTuple):
        a: str
        b: bool
        c: int = 0

    sol = mi.NamedTupleType(
        Example,
        fields=(
            mi.Field("a", "a", mi.StrType()),
            mi.Field("b", "b", mi.BoolType()),
            mi.Field("c", "c", mi.IntType(), required=False, default=0),
        ),
    )
    assert mi.type_info(Example) == sol


def test_collections_namedtuple():
    Example = namedtuple("Example", ["a", "b", "c"], defaults=(0,))

    sol = mi.NamedTupleType(
        Example,
        fields=(
            mi.Field("a", "a", mi.AnyType()),
            mi.Field("b", "b", mi.AnyType()),
            mi.Field("c", "c", mi.AnyType(), required=False, default=0),
        ),
    )
    assert mi.type_info(Example) == sol


def test_generic_namedtuple():
    NamedTuple = pytest.importorskip("typing_extensions").NamedTuple

    class Example(NamedTuple, Generic[T]):
        a: T
        b: list[T]

    sol = mi.NamedTupleType(
        Example,
        fields=(
            mi.Field("a", "a", mi.AnyType()),
            mi.Field("b", "b", mi.ListType(mi.AnyType())),
        ),
    )
    assert mi.type_info(Example) == sol

    sol = mi.NamedTupleType(
        Example[int],
        fields=(
            mi.Field("a", "a", mi.IntType()),
            mi.Field("b", "b", mi.ListType(mi.IntType())),
        ),
    )
    assert mi.type_info(Example[int]) == sol


@pytest.mark.parametrize("use_typing_extensions", [False, True])
def test_typeddict(use_typing_extensions):
    if use_typing_extensions:
        tex = pytest.importorskip("typing_extensions")
        cls = tex.TypedDict
    else:
        cls = TypedDict

    class Example(cls):
        a: str
        b: bool
        c: int

    sol = mi.TypedDictType(
        Example,
        fields=(
            mi.Field("a", "a", mi.StrType()),
            mi.Field("b", "b", mi.BoolType()),
            mi.Field("c", "c", mi.IntType()),
        ),
    )
    assert mi.type_info(Example) == sol


@pytest.mark.parametrize("use_typing_extensions", [False, True])
def test_typeddict_optional(use_typing_extensions):
    if use_typing_extensions:
        tex = pytest.importorskip("typing_extensions")
        cls = tex.TypedDict
    else:
        cls = TypedDict

    class Base(cls):
        a: str
        b: bool

    class Example(Base, total=False):
        c: int

    sol = mi.TypedDictType(
        Example,
        fields=(
            mi.Field("a", "a", mi.StrType()),
            mi.Field("b", "b", mi.BoolType()),
            mi.Field("c", "c", mi.IntType(), required=False),
        ),
    )
    assert mi.type_info(Example) == sol


def test_generic_typeddict():
    TypedDict = pytest.importorskip("typing_extensions").TypedDict

    class Example(TypedDict, Generic[T]):
        a: T
        b: list[T]

    sol = mi.TypedDictType(
        Example,
        fields=(
            mi.Field("a", "a", mi.AnyType()),
            mi.Field("b", "b", mi.ListType(mi.AnyType())),
        ),
    )
    assert mi.type_info(Example) == sol

    sol = mi.TypedDictType(
        Example[int],
        fields=(
            mi.Field("a", "a", mi.IntType()),
            mi.Field("b", "b", mi.ListType(mi.IntType())),
        ),
    )
    assert mi.type_info(Example[int]) == sol


def test_dataclass():
    @dataclass
    class Example:
        x: int
        y: int = 0
        z: str = field(default_factory=str)

    sol = mi.DataclassType(
        Example,
        fields=(
            mi.Field("x", "x", mi.IntType()),
            mi.Field("y", "y", mi.IntType(), required=False, default=0),
            mi.Field("z", "z", mi.StrType(), required=False, default_factory=str),
        ),
    )
    assert mi.type_info(Example) == sol


def test_attrs():
    attrs = pytest.importorskip("attrs")

    @attrs.define
    class Example:
        x: int
        y: int = 0
        z: str = attrs.field(factory=str)

    sol = mi.DataclassType(
        Example,
        fields=(
            mi.Field("x", "x", mi.IntType()),
            mi.Field("y", "y", mi.IntType(), required=False, default=0),
            mi.Field("z", "z", mi.StrType(), required=False, default_factory=str),
        ),
    )
    assert mi.type_info(Example) == sol


@pytest.mark.parametrize("module", ["dataclasses", "attrs"])
def test_generic_dataclass_or_attrs(module):
    m = pytest.importorskip(module)
    decorator = m.define if module == "attrs" else m.dataclass

    @decorator
    class Example(Generic[T]):
        a: T
        b: list[T]

    sol = mi.DataclassType(
        Example,
        fields=(
            mi.Field("a", "a", mi.AnyType()),
            mi.Field("b", "b", mi.ListType(mi.AnyType())),
        ),
    )
    assert mi.type_info(Example) == sol

    sol = mi.DataclassType(
        Example[int],
        fields=(
            mi.Field("a", "a", mi.IntType()),
            mi.Field("b", "b", mi.ListType(mi.IntType())),
        ),
    )
    assert mi.type_info(Example[int]) == sol


@pytest.mark.parametrize("kind", ["struct", "dataclass", "attrs"])
def test_unset_fields(kind):
    if kind == "struct":

        class Ex(msgspec.Struct):
            x: int | msgspec.UnsetType = msgspec.UNSET

    elif kind == "dataclass":

        @dataclass
        class Ex:
            x: int | msgspec.UnsetType = msgspec.UNSET

    elif kind == "attrs":
        attrs = pytest.importorskip("attrs")

        @attrs.define
        class Ex:
            x: int | msgspec.UnsetType = msgspec.UNSET

    res = mi.type_info(Ex)
    assert res.fields == (mi.Field("x", "x", mi.IntType(), required=False),)


@pytest.mark.parametrize("kind", ["struct", "namedtuple", "typeddict", "dataclass"])
def test_self_referential_objects(kind):
    if kind == "struct":
        code = """
        import msgspec

        class Node(msgspec.Struct):
            child: "Node"
        """
    elif kind == "namedtuple":
        code = """
        from typing import NamedTuple

        class Node(NamedTuple):
            child: "Node"
        """
    elif kind == "typeddict":
        code = """
        from typing import TypedDict

        class Node(TypedDict):
            child: "Node"
        """
    elif kind == "dataclass":
        code = """
        from dataclasses import dataclass

        @dataclass
        class Node:
            child: "Node"
        """

    with temp_module(code) as mod:
        res = mi.type_info(mod.Node)

    assert res.cls is mod.Node
    assert res.fields[0].name == "child"
    assert res.fields[0].type is res


def test_metadata():
    typ = Annotated[int, Meta(gt=1, title="a"), Meta(description="b")]

    assert mi.type_info(typ) == mi.Metadata(
        mi.IntType(gt=1), {"title": "a", "description": "b"}
    )

    typ = Annotated[
        int,
        Meta(extra_json_schema={"title": "a", "description": "b"}),
        Meta(extra_json_schema={"title": "c", "examples": [1, 2]}),
    ]

    assert mi.type_info(typ) == mi.Metadata(
        mi.IntType(), {"title": "c", "description": "b", "examples": [1, 2]}
    )

    typ = Annotated[
        int,
        Meta(extra={"a": 1, "b": 2}),
        Meta(extra={"a": 3, "c": 4}),
    ]

    assert mi.type_info(typ) == mi.Metadata(
        mi.IntType(), extra={"a": 3, "b": 2, "c": 4}
    )


def test_inspect_with_unhashable_metadata():
    typ = Annotated[int, {"unhashable"}]

    assert mi.type_info(typ) == mi.IntType()


def test_multi_type_info():
    class Example(msgspec.Struct):
        x: int
        y: int

    ex_type = mi.StructType(
        Example,
        fields=(
            mi.Field("x", "x", mi.IntType()),
            mi.Field("y", "y", mi.IntType()),
        ),
    )

    assert mi.multi_type_info([]) == ()

    res = mi.multi_type_info([Example, list[Example]])
    assert res == (ex_type, mi.ListType(ex_type))
    assert res[0] is res[1].item_type


def test_type_info_custom_base_class():
    class CustomMeta(msgspec.StructMeta):
        pass

    class Base(metaclass=CustomMeta):
        pass

    class Model(Base):
        foo: str

    assert mi.type_info(Model) == mi.StructType(
        cls=Model,
        fields=(
            mi.Field(
                name="foo",
                encode_name="foo",
                type=mi.StrType(min_length=None, max_length=None, pattern=None),
                required=True,
                default=msgspec.NODEFAULT,
                default_factory=msgspec.NODEFAULT,
            ),
        ),
        tag_field=None,
        tag=None,
        array_like=False,
        forbid_unknown_fields=False,
    )


def test_is_struct_runtime():
    class Base(msgspec.Struct):
        x: int

    class Derived(Base):
        pass

    Generated = msgspec.defstruct("InspectStructRuntime", ["x"])

    class CustomMeta(msgspec.StructMeta):
        pass

    class CustomBase(metaclass=CustomMeta):
        x: int

    class Custom(CustomBase):
        pass

    class NotStruct:
        pass

    assert mi.is_struct(Base(1))
    assert mi.is_struct(Derived(1))
    assert mi.is_struct(Generated(1))
    assert mi.is_struct(Custom(1))
    assert not mi.is_struct(NotStruct())
    assert not mi.is_struct(object())


def test_is_struct_type_runtime():
    class Base(msgspec.Struct):
        x: int

    class Derived(Base):
        pass

    Generated = msgspec.defstruct("InspectStructTypeRuntime", ["x"])

    class CustomMeta(msgspec.StructMeta):
        pass

    class CustomBase(metaclass=CustomMeta):
        pass

    class Custom(CustomBase):
        x: int

    class NotStruct:
        pass

    assert mi.is_struct_type(Base)
    assert mi.is_struct_type(Derived)
    assert mi.is_struct_type(Generated)
    assert mi.is_struct_type(Custom)
    assert not mi.is_struct_type(NotStruct)
    assert not mi.is_struct_type(object)
