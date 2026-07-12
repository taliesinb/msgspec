import datetime
import enum
import uuid
from typing import (
    Any,
    Dict,
    List,
    Literal,
    Optional,
    Set,
    Tuple,
    Union,
)

import pytest

import msgspec
from msgspec import Struct


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
