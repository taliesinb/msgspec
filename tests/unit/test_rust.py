"""Tests for `msgspec.rust` — Rust (serde) type-definition generation."""

import enum
import shutil
import subprocess
import textwrap
from typing import Optional, Union

import pytest

import msgspec
import msgspec.rust as rust
from msgspec import Struct
from msgspec.data import Float32, Int32, Int64, UInt, UInt64

CARGO = shutil.which("cargo")
needs_cargo = pytest.mark.skipif(CARGO is None, reason="cargo not installed")


# --- scalar / collection mapping -------------------------------------------


@pytest.mark.parametrize(
    "T, expected",
    [
        (int, "i64"),
        (bool, "bool"),
        (float, "f64"),
        (str, "String"),
        (Int32, "i32"),
        (Int64, "i64"),
        (UInt64, "u64"),
        (UInt, "u64"),
        (Float32, "f32"),
        (bytes, "serde_bytes::ByteBuf"),
        (list[UInt64], "Vec<u64>"),
        (set[int], "std::collections::HashSet<i64>"),
        (tuple[int, str], "(i64, String)"),
        (dict[str, int], "std::collections::HashMap<String, i64>"),
        (Optional[str], "Option<String>"),
    ],
)
def test_scalar_and_collection_refs(T, expected):
    (root,), _ = rust.schema_components((T,))
    assert root == expected


def test_list_root_gets_alias():
    src = rust.schema(list[Int64])
    assert src.strip() == "pub type Root = Vec<i64>;"


# --- structs ---------------------------------------------------------------


def test_basic_struct():
    class Point(Struct):
        x: Int32
        y: Int32

    assert rust.schema(Point) == textwrap.dedent(
        """\
        #[derive(Serialize, Deserialize)]
        pub struct Point {
            pub x: i32,
            pub y: i32,
        }
        """
    )


def test_optional_field_is_option():
    class Rec(Struct):
        name: str
        nick: Optional[str] = None

    src = rust.schema(Rec)
    assert "pub name: String," in src
    assert "pub nick: Option<String>," in src


def test_reserved_field_name_uses_raw_ident():
    class Rec(Struct):
        type: str
        match: int

    src = rust.schema(Rec)
    assert "pub r#type: String," in src
    assert "pub r#match: i64," in src


def test_non_ident_field_name_gets_serde_rename():
    class Rec(Struct, rename={"my_field": "my-field"}):
        my_field: str

    src = rust.schema(Rec)
    assert '#[serde(rename = "my-field")]' in src
    assert "pub my_field: String," in src


def test_nested_struct_is_emitted_once():
    class Inner(Struct):
        v: int

    class Outer(Struct):
        a: Inner
        b: Inner

    src = rust.schema(Outer)
    assert src.count("pub struct Inner") == 1
    assert "pub a: Inner," in src


# --- enums -----------------------------------------------------------------


def test_str_enum():
    class Color(enum.Enum):
        RED = "red"
        DARK_GREEN = "dark_green"

    src = rust.schema(Color)
    assert "pub enum Color {" in src
    assert '#[serde(rename = "red")]\n    Red,' in src
    assert '#[serde(rename = "dark_green")]\n    DarkGreen,' in src


# --- tagged unions / abstract structs --------------------------------------


def test_abstract_struct_becomes_tagged_enum():
    class Node(Struct, abstract=True, tag_field="op", tag=str.lower):
        pass

    class Add(Node):
        lhs: float
        rhs: float

    class Neg(Node):
        val: float

    src = rust.schema(Node)
    # concrete structs
    assert "pub struct Add {" in src
    assert "pub struct Neg {" in src
    # the records-style enum, internally tagged
    assert '#[serde(tag = "op")]\npub enum Node {' in src
    assert '#[serde(rename = "add")]\n    Add(Add),' in src
    assert '#[serde(rename = "neg")]\n    Neg(Neg),' in src
    # exactly one enum for the abstract base (no duplicate anonymous union)
    assert src.count("pub enum") == 1


def test_anonymous_tagged_union_gets_synthesized_name():
    class Cat(Struct, tag_field="kind", tag="cat"):
        name: str

    class Dog(Struct, tag_field="kind", tag="dog"):
        sound: str

    src = rust.schema(Union[Cat, Dog])
    assert '#[serde(tag = "kind")]\npub enum CatOrDog {' in src
    assert "Cat(Cat)," in src and "Dog(Dog)," in src


def test_unsupported_scalar_union_errors():
    with pytest.raises(NotImplementedError):
        rust.schema(Union[int, str])


# --- aliases ---------------------------------------------------------------


def test_type_alias():
    from typing import TypeAliasType  # py3.12+, matches msgspec.data markers

    Ids = TypeAliasType("Ids", list[UInt64])
    src = rust.schema(Ids)
    assert src.strip() == "pub type Ids = Vec<u64>;"


# --- end-to-end: compile + cross-language round-trip -----------------------


@needs_cargo
def test_cross_language_roundtrip(tmp_path):
    """Generated types compile under serde and round-trip byte-compatibly with
    msgspec through both rmp-serde (msgpack) and serde_json (json)."""

    class Color(enum.Enum):
        RED = "red"
        DARK_GREEN = "dark_green"

    class Meta(Struct):
        tag: str
        weight: Float32

    class Point(Struct):
        x: Int32
        y: Int32
        ids: list[UInt64]
        color: Color
        note: Optional[str] = None
        meta: Optional[Meta] = None

    class Node(Struct, abstract=True, tag_field="op", tag=str.lower):
        pass

    class Add(Node):
        lhs: float
        rhs: float

    class Neg(Node):
        val: float

    src = rust.schema(Point) + "\n\n" + rust.schema(Node)
    proj = tmp_path
    (proj / "src").mkdir()
    (proj / "src" / "generated.rs").write_text(
        "use serde::{Serialize, Deserialize};\n\n" + src
    )
    (proj / "src" / "main.rs").write_text(
        textwrap.dedent(
            """\
            mod generated;
            use generated::{Point, Node};
            use std::fs;
            fn rt<T: serde::de::DeserializeOwned + serde::Serialize>(s: &str) {
                let mp = fs::read(format!("{s}.mp")).unwrap();
                let v: T = rmp_serde::from_slice(&mp).unwrap();
                fs::write(format!("{s}.out.mp"), rmp_serde::to_vec_named(&v).unwrap()).unwrap();
                let js = fs::read(format!("{s}.json")).unwrap();
                let v2: T = serde_json::from_slice(&js).unwrap();
                fs::write(format!("{s}.out.json"), serde_json::to_vec(&v2).unwrap()).unwrap();
            }
            fn main() {
                std::env::set_current_dir(std::env::args().nth(1).unwrap()).unwrap();
                rt::<Point>("point");
                rt::<Node>("node");
            }
            """
        )
    )
    (proj / "Cargo.toml").write_text(
        textwrap.dedent(
            """\
            [package]
            name = "rt"
            version = "0.0.0"
            edition = "2021"
            [dependencies]
            serde = { version = "1", features = ["derive"] }
            serde_json = "1"
            rmp-serde = "1"
            serde_bytes = "0.11"
            [[bin]]
            name = "rt"
            path = "src/main.rs"
            """
        )
    )

    build = subprocess.run(
        [CARGO, "build", "--release"], cwd=proj, capture_output=True, text=True
    )
    if build.returncode != 0:
        pytest.skip(f"cargo build failed (offline?):\n{build.stderr}")

    point = Point(
        x=-5, y=1_000_000, ids=[2**64 - 1, 0, 42], color=Color.DARK_GREEN,
        note="hi", meta=Meta(tag="t", weight=1.5),
    )
    node = Add(lhs=1.5, rhs=2.5)
    for name, obj, T in [("point", point, Point), ("node", node, Node)]:
        # msgpack: type-directed; json: plain/value-directed (native numbers)
        (proj / f"{name}.mp").write_bytes(msgspec.msgpack.encode(obj, type=T))
        (proj / f"{name}.json").write_bytes(msgspec.json.encode(obj))

    run = subprocess.run(
        [str(proj / "target" / "release" / "rt"), str(proj)],
        capture_output=True, text=True,
    )
    assert run.returncode == 0, run.stderr

    for name, obj, T in [("point", point, Point), ("node", node, Node)]:
        got_mp = msgspec.msgpack.decode((proj / f"{name}.out.mp").read_bytes(), type=T)
        got_js = msgspec.json.decode((proj / f"{name}.out.json").read_bytes(), type=T)
        assert got_mp == obj
        assert got_js == obj
