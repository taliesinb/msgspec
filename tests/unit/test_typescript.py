import datetime
import enum
import sys
import uuid
from typing import (
    Any,
    Dict,
    List,
    Literal,
    NewType,
    Optional,
    Set,
    Tuple,
    Union,
)

import pytest

import msgspec
from msgspec import Struct

py312_plus = pytest.mark.skipif(
    sys.version_info < (3, 12), reason="3.12+ only"
)


def ts(type):
    return msgspec.typescript.schema(type)


@pytest.mark.parametrize(
    "typ, expected",
    [
        (Any, "any"),
        (msgspec.Raw, "any"),
        (None, "null"),
        (bool, "boolean"),
        (int, "number"),
        (float, "number"),
        (str, "string"),
        (bytes, "string"),
        (bytearray, "string"),
        (datetime.datetime, "string"),
        (datetime.date, "string"),
        (datetime.time, "string"),
        (datetime.timedelta, "string"),
        (uuid.UUID, "string"),
        (List[int], "Array<number>"),
        (Set[int], "Set<number>"),
        (Tuple[int, ...], "Array<number>"),
        (Tuple[int, str], "[number, string]"),
        (Dict[str, int], "Record<string, number>"),
        (Dict[int, int], "Record<number, number>"),
        (Optional[int], "number | null"),
        (Union[int, str], "number | string"),
        (Literal["a", "b"], '"a" | "b"'),
    ],
)
def test_ref_scalar_and_collections(typ, expected):
    (root,), components = msgspec.typescript.schema_components([typ])
    assert root == expected
    assert components == {}


def test_dict_non_string_key_coerced_to_string():
    (root,), _ = msgspec.typescript.schema_components([Dict[bytes, int]])
    assert root == "Record<string, number>"


def test_struct_basic():
    class Point(Struct):
        x: int
        y: int
        label: str = "origin"

    out = ts(Point)
    assert "export class Point {" in out
    assert "  x: number;" in out
    assert "  y: number;" in out
    assert "  label?: string;" in out


def test_struct_docstring_emitted_as_jsdoc():
    class Point(Struct):
        """A 2D point."""

        x: int

    out = ts(Point)
    assert "/** A 2D point. */" in out


def test_nested_struct_emits_component():
    class Point(Struct):
        x: int

    class Line(Struct):
        start: Point
        end: Point

    out = ts(Line)
    assert "export class Line {" in out
    assert "  start: Point;" in out
    assert "export class Point {" in out


def test_enum():
    class Color(enum.Enum):
        RED = "red"
        GREEN = "green"

    out = ts(Color)
    assert "export enum Color {" in out
    assert '  RED = "red",' in out
    assert '  GREEN = "green",' in out


def test_int_enum():
    class Size(enum.IntEnum):
        SMALL = 1
        LARGE = 2

    out = ts(Size)
    assert "  SMALL = 1," in out
    assert "  LARGE = 2," in out


def test_tagged_union():
    class Cat(Struct, tag="cat"):
        meow: int

    class Dog(Struct, tag="dog"):
        woof: str

    out = ts(Union[Cat, Dog])
    assert '  type: "cat";' in out
    assert '  type: "dog";' in out
    assert "export type Root = Cat | Dog;" in out


def test_array_like_struct_is_tuple():
    class Rec(Struct, array_like=True):
        a: int
        b: str = "x"

    out = ts(Rec)
    assert "export type Rec = [number, string];" in out


def test_array_like_tagged_struct_prepends_tag():
    class Rec(Struct, array_like=True, tag="rec"):
        a: int

    out = ts(Rec)
    assert 'export type Rec = ["rec", number];' in out


def test_renamed_field_quoted_when_needed():
    class Model(Struct, rename={"a_field": "a-field"}):
        a_field: int

    out = ts(Model)
    assert '  "a-field": number;' in out


def test_root_alias_for_non_nameable():
    class Point(Struct):
        x: int

    out = ts(List[Point])
    assert "export type Root = Array<Point>;" in out
    assert "export class Point {" in out


def test_custom_type_raises():
    with pytest.raises(TypeError, match="custom type"):
        ts(complex)


def test_ext_type_raises():
    with pytest.raises(TypeError, match="Ext"):
        ts(msgspec.msgpack.Ext)


def test_newtype_alias_emitted_as_type():
    Pixels = NewType("Pixels", int)

    class Sprite(Struct):
        width: Pixels
        height: Pixels

    out = ts(Sprite)
    assert "export type Pixels = number;" in out
    assert "  width: Pixels;" in out
    assert "  height: Pixels;" in out
    # A single alias definition, even though referenced twice.
    assert out.count("export type Pixels = number;") == 1


def test_alias_of_str():
    UserId = NewType("UserId", str)

    class User(Struct):
        id: UserId

    out = ts(User)
    assert "export type UserId = string;" in out
    assert "  id: UserId;" in out


def test_alias_of_list():
    Row = NewType("Row", List[int])

    class Grid(Struct):
        rows: List[Row]

    out = ts(Grid)
    assert "export type Row = Array<number>;" in out
    assert "  rows: Array<Row>;" in out


def test_alias_as_root():
    Pixels = NewType("Pixels", int)
    out = ts(List[Pixels])
    assert "export type Pixels = number;" in out
    assert "export type Root = Array<Pixels>;" in out


def test_alias_referencing_struct():
    class Point(Struct):
        x: int

    Ref = NewType("Ref", Point)

    class Holder(Struct):
        p: Ref

    out = ts(Holder)
    assert "export type Ref = Point;" in out
    assert "export class Point {" in out
    assert "  p: Ref;" in out


@py312_plus
def test_pep695_type_alias():
    from .utils import temp_module

    with temp_module("type Pixels = int") as mod:
        out = ts(List[mod.Pixels])
    assert "export type Pixels = number;" in out
    assert "export type Root = Array<Pixels>;" in out


@pytest.mark.parametrize(
    "dtype_alias, expected",
    [
        ("UInt8", "number"),
        ("Int32", "number"),
        ("Float32", "number"),
        ("Float64", "number"),
        ("Int64", "bigint"),
        ("UInt64", "bigint"),
        ("Bool", "boolean"),
    ],
)
def test_data_scalar_ts(dtype_alias, expected):
    from msgspec import data as md

    (root,), components = msgspec.typescript.schema_components([getattr(md, dtype_alias)])
    assert root == expected
    assert components == {}


def test_data_scalar_any_ts():
    from msgspec import data as md

    (root,), _ = msgspec.typescript.schema_components([md.Scalar])
    assert root == "number | boolean"


@pytest.mark.parametrize(
    "shape, dtype_alias, expected",
    [
        (3, "Float32", "Float32Array"),
        ((3, 3), "UInt8", "Uint8Array"),
        (2, "Int32", "Int32Array"),
        (4, "Int64", "BigInt64Array"),
        (1, "UInt64", "BigUint64Array"),
        (5, "Bool", "Uint8Array"),
    ],
)
def test_data_tensor_ts(shape, dtype_alias, expected):
    from msgspec import data as md

    tensor = md.Tensor[shape, getattr(md, dtype_alias)]
    (root,), _ = msgspec.typescript.schema_components([tensor])
    assert root == expected


def test_data_tensor_any_dtype_ts():
    from msgspec import data as md

    (root,), _ = msgspec.typescript.schema_components([md.Tensor[3, None]])
    assert root == "ArrayBufferView"
    (root2,), _ = msgspec.typescript.schema_components([md.Tensor])
    assert root2 == "ArrayBufferView"


def test_data_types_in_struct_ts():
    from msgspec import data as md

    class Layer(Struct):
        weights: md.Tensor[3, md.Float32]
        bias: md.Scalar
        count: md.Int64

    out = ts(Layer)
    assert "  weights: Float32Array;" in out
    assert "  bias: number | boolean;" in out
    assert "  count: bigint;" in out


@py312_plus
def test_pep695_generic_alias_specializations_distinct():
    from .utils import temp_module

    with temp_module("type Vec[T] = list[T]") as mod:

        class Holder(Struct):
            ints: mod.Vec[int]
            strs: mod.Vec[str]

        out = ts(Holder)
    # Distinct specializations get distinct names (mirrors json.schema's
    # `Box_int_` convention), so no definition is trampled.
    assert "  ints: Vec_int;" in out
    assert "  strs: Vec_str;" in out
    assert "export type Vec_int = Array<number>;" in out
    assert "export type Vec_str = Array<string>;" in out
