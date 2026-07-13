"""Tests for `msgspec.javascript.codec`.

The structural tests check the generated source. The node-gated tests actually
run the generated JavaScript and assert its MessagePack output is byte-identical
to `msgspec.msgpack` (and that it round-trips).
"""

import enum
import json
import shutil
import subprocess
from pathlib import Path
from typing import Union

import pytest

import msgspec
from msgspec import Struct
from msgspec.data import Int, Int64, UInt64

NODE = shutil.which("node")
needs_node = pytest.mark.skipif(NODE is None, reason="node not installed")

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNTIME_MJS = REPO_ROOT / "javascript" / "src" / "msgpack.mjs"


# --- structure -------------------------------------------------------------


def test_codec_is_self_contained():
    class Point(Struct):
        x: int
        y: int

    out = msgspec.javascript.codec(Point)
    # No external imports - the msgpack runtime is inlined.
    assert "import" not in out
    assert "class Writer" in out and "class Reader" in out
    # Namespaced msgpack + json exports.
    assert "export const msgpack = {" in out
    assert "export const json = {" in out
    assert "export function encodePoint(w, v)" in out
    assert "export function decodePoint(r)" in out
    assert "export function encodeJsonPoint(v)" in out


def test_codec_flags_select_formats():
    class Point(msgspec.Struct):
        x: int

    mp_only = msgspec.javascript.codec(Point, json=False)
    assert "export const msgpack = {" in mp_only
    assert "export const json = {" not in mp_only

    json_only = msgspec.javascript.codec(Point, msgpack=False)
    assert "export const json = {" in json_only
    assert "export const msgpack = {" not in json_only

    with pytest.raises(ValueError):
        msgspec.javascript.codec(Point, msgpack=False, json=False)


def test_codec_optional_and_scalars():
    class S(Struct):
        i: int
        f: float
        s: str
        b: bool
        maybe: str | None

    out = msgspec.javascript.codec(S)
    assert "w.int(v[\"i\"]);" in out
    assert "w.float(v[\"f\"]);" in out
    assert "w.bool(v[\"b\"]);" in out
    assert "r.tryNil() ? null : r.str()" in out


def test_codec_bigint_types():
    class S(Struct):
        a: Int
        b: Int64
        c: UInt64

    out = msgspec.javascript.codec(S)
    assert "w.int64(v[\"a\"]);" in out
    assert "o[\"a\"] = r.int64();" in out


def test_codec_tagged_union_dispatch():
    class Cat(Struct, tag="cat"):
        name: str

    class Dog(Struct, tag="dog"):
        legs: int

    out = msgspec.javascript.codec(Union[Cat, Dog])
    assert "encodeCat(w, value)" in out
    assert 'case "cat": return decodeCat(r);' in out
    # the decoder seeds the tag field so re-encoding works
    assert 'const o = { type: "cat" };' in out


def test_codec_abstract_struct():
    class Shape(Struct, tag_field="kind", abstract=True):
        pass

    class Circle(Shape, tag="circle"):
        r: float

    class Square(Shape, tag="square"):
        s: float

    out = msgspec.javascript.codec(Shape)
    assert "encodeCircle" in out and "encodeSquare" in out
    assert 'case "circle": return decodeCircle(r);' in out


def test_codec_unsupported_type_errors():
    import datetime

    class S(Struct):
        when: datetime.datetime

    with pytest.raises(NotImplementedError):
        msgspec.javascript.codec(S)


def test_bundle_path():
    p = msgspec.javascript.bundle_path()
    # Resolves in both a source checkout (top-level javascript/) and an
    # installed wheel (msgspec/javascript_bundle/).
    assert p.is_dir()
    assert (p / "src" / "msgpack.mjs").is_file()
    assert (p / "package.json").is_file()


def test_embedded_runtime_matches_source():
    """The embedded runtime must stay in sync with the authored .mjs."""
    if not RUNTIME_MJS.exists():
        pytest.skip("javascript/src/msgpack.mjs not present (installed package)")
    from msgspec._msgpack_runtime_js import RUNTIME

    # Mirror scripts/gen_js_runtime.py's `inline()`: strip the ESM exports.
    expected = (
        RUNTIME_MJS.read_text()
        .replace("export class ", "class ")
        .replace("export function ", "function ")
    )
    assert RUNTIME.strip() == expected.strip(), (
        "src/msgspec/_msgpack_runtime_js.py is stale - "
        "re-run scripts/gen_js_runtime.py"
    )


# --- node byte-compat ------------------------------------------------------


def _run_node(codec_src: str, driver_src: str, tmp_path) -> str:
    (tmp_path / "codec.mjs").write_text(codec_src)
    (tmp_path / "driver.mjs").write_text(driver_src)
    res = subprocess.run(
        [NODE, str(tmp_path / "driver.mjs")],
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0, res.stderr
    return res.stdout.strip()


@needs_node
def test_node_byte_compat_roundtrip(tmp_path):
    class Point(Struct):
        x: int
        y: int

    class Cat(Struct, tag="cat"):
        name: str

    class Dog(Struct, tag="dog"):
        legs: int

    class Color(enum.Enum):
        RED = "red"
        BLUE = "blue"

    class Doc(Struct):
        pt: Point
        pets: list[Union[Cat, Dog]]
        scores: dict[str, int]
        color: Color
        note: str | None
        ok: bool
        ratio: float

    value = Doc(
        pt=Point(x=-3, y=70000),
        pets=[Cat(name="Reo"), Dog(legs=4)],
        scores={"a": 1, "b": 2},
        color=Color.BLUE,
        note=None,
        ok=True,
        ratio=1.25,
    )
    expected_hex = msgspec.msgpack.encode(value).hex()

    # The JS object our codec expects (tagged structs carry their tag field).
    js_obj = {
        "pt": {"x": -3, "y": 70000},
        "pets": [
            {"type": "cat", "name": "Reo"},
            {"type": "dog", "legs": 4},
        ],
        "scores": {"a": 1, "b": 2},
        "color": "blue",
        "note": None,
        "ok": True,
        "ratio": 1.25,
    }

    driver = f"""
    import {{ msgpack }} from "./codec.mjs";
    const obj = {json.dumps(js_obj)};
    const bytes = msgpack.encode(obj);
    const hex = Buffer.from(bytes).toString("hex");
    const back = msgpack.decode(bytes);
    const roundtrips = JSON.stringify(back) === JSON.stringify(obj);
    console.log(JSON.stringify({{ hex, roundtrips }}));
    """
    out = json.loads(_run_node(msgspec.javascript.codec(Doc), driver, tmp_path))
    assert out["hex"] == expected_hex
    assert out["roundtrips"] is True


@needs_node
def test_node_json_format_compat(tmp_path):
    """The `json` namespace must produce/consume the same JSON as a type-directed
    `msgspec.json` encoder (hex ints, base64 bytes, Scalar always-hex)."""
    from msgspec.data import Int64, Scalar

    class Cat(Struct, tag="cat"):
        name: str
        lives: Int64

    class Doc(Struct):
        pet: Cat
        blob: bytes
        val: Scalar
        note: str | None

    value = Doc(
        pet=Cat(name="Reo", lives=2**62),
        blob=b"\x00\x01\x02",
        val=2**61,
        note=None,
    )
    expected_json = msgspec.json.encode(value, type=Doc).decode()

    driver = f"""
    import {{ json }} from "./codec.mjs";
    const obj = {{
      pet: {{ type: "cat", name: "Reo", lives: 2n ** 62n }},
      blob: new Uint8Array([0, 1, 2]),
      val: 2n ** 61n,
      note: null,
    }};
    const text = json.encode(obj);
    const back = json.decode({json.dumps(expected_json)});
    const ok =
      back.pet.lives === 2n ** 62n &&
      back.val === 2n ** 61n &&
      [...back.blob].join(",") === "0,1,2" &&
      back.note === null;
    console.log(JSON.stringify({{ text, decode_ok: ok }}));
    """
    out = json.loads(_run_node(msgspec.javascript.codec(Doc), driver, tmp_path))
    assert out["text"] == expected_json
    assert out["decode_ok"] is True


@needs_node
def test_node_bigint_byte_compat(tmp_path):
    class Big(Struct):
        a: Int
        b: Int64
        c: UInt64

    value = Big(a=42, b=-(2**60), c=2**64 - 1)
    expected_hex = msgspec.msgpack.encode(value).hex()

    driver = """
    import { msgpack } from "./codec.mjs";
    const obj = { a: 42n, b: -(2n ** 60n), c: 2n ** 64n - 1n };
    const hex = Buffer.from(msgpack.encode(obj)).toString("hex");
    const back = msgpack.decode(msgpack.encode(obj));
    const roundtrips = back.a === obj.a && back.b === obj.b && back.c === obj.c;
    console.log(JSON.stringify({ hex, roundtrips }));
    """
    out = json.loads(_run_node(msgspec.javascript.codec(Big), driver, tmp_path))
    assert out["hex"] == expected_hex
    assert out["roundtrips"] is True


@needs_node
def test_node_named_tuple(tmp_path):
    import typing

    class Point(typing.NamedTuple):
        x: int
        y: int
        label: str = "p"

    class Holder(Struct):
        pt: Point
        pts: list[Point]

    value = Holder(pt=Point(3, 4, "a"), pts=[Point(5, 6), Point(7, 8, "z")])
    expected_hex = msgspec.msgpack.encode(value).hex()

    # A NamedTuple is positional on the wire but an object keyed by field name
    # in JS (matching the TypeScript codec).
    obj = {
        "pt": {"x": 3, "y": 4, "label": "a"},
        "pts": [
            {"x": 5, "y": 6, "label": "p"},
            {"x": 7, "y": 8, "label": "z"},
        ],
    }
    driver = f"""
    import {{ msgpack }} from "./codec.mjs";
    const obj = {json.dumps(obj)};
    const hex = Buffer.from(msgpack.encode(obj)).toString("hex");
    const back = msgpack.decode(msgpack.encode(obj));
    console.log(JSON.stringify({{ hex, roundtrips: JSON.stringify(back) === JSON.stringify(obj) }}));
    """
    out = json.loads(_run_node(msgspec.javascript.codec(Holder), driver, tmp_path))
    assert out["hex"] == expected_hex
    assert out["roundtrips"] is True


@needs_node
def test_node_array_like_struct(tmp_path):
    class Rec(Struct, array_like=True):
        p: int
        q: str
        r: bool

    value = Rec(p=7, q="hi", r=True)
    expected_hex = msgspec.msgpack.encode(value).hex()

    driver = """
    import { msgpack } from "./codec.mjs";
    const obj = { p: 7, q: "hi", r: true };
    const hex = Buffer.from(msgpack.encode(obj)).toString("hex");
    const back = msgpack.decode(msgpack.encode(obj));
    console.log(JSON.stringify({ hex, roundtrips: JSON.stringify(back) === JSON.stringify(obj) }));
    """
    out = json.loads(_run_node(msgspec.javascript.codec(Rec), driver, tmp_path))
    assert out["hex"] == expected_hex
    assert out["roundtrips"] is True
