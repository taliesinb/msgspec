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
def test_node_scalar_container_union(tmp_path):
    """A union mixing a scalar with a container (e.g. `str | tuple`) dispatches on
    the JS runtime shape (encode) and the msgpack tag / JSON shape (decode)."""

    class Doc(Struct):
        end: Union[str, tuple[str, int]]

    cases = [Doc(end="root"), Doc(end=("a", 3))]
    ref = [
        {
            "mp": msgspec.msgpack.encode(c, type=Doc).hex(),
            "js": msgspec.json.encode(c, type=Doc).decode(),
            "obj": {"end": c.end if isinstance(c.end, str) else list(c.end)},
        }
        for c in cases
    ]

    driver = f"""
    import {{ msgpack, json }} from "./codec.mjs";
    const ref = {json.dumps(ref)};
    const hex = (u) => Buffer.from(u).toString("hex");
    let ok = true;
    for (const c of ref) {{
      const mp = hex(msgpack.encode(c.obj)) === c.mp;
      const js = json.encode(c.obj) === c.js;
      const mb = msgpack.decode(Uint8Array.from(Buffer.from(c.mp, "hex")));
      const jb = json.decode(c.js);
      const dec = JSON.stringify(mb) === JSON.stringify(c.obj)
               && JSON.stringify(jb) === JSON.stringify(c.obj);
      ok = ok && mp && js && dec;
    }}
    console.log(JSON.stringify({{ ok }}));
    """
    out = json.loads(_run_node(msgspec.javascript.codec(Doc), driver, tmp_path))
    assert out["ok"] is True


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


def test_tensor_codec_structure():
    from msgspec.data import Float32, Tensor, UInt8

    class Doc(Struct):
        arr: Tensor[(2, 3), UInt8]
        w: Tensor[1, Float32]

    src = msgspec.javascript.codec(Doc)
    # the runtime path (w.tensor / r.tensor) and the JSON path are wired up, and
    # TensorHandle is re-exported so callers can construct tensor values.
    assert "w.tensor(" in src and "r.tensor()" in src
    assert "encTensorJSON(" in src and "decTensorJSON(" in src
    assert "export { TensorHandle };" in src


def test_tensor_custom_encoder_decoder():
    from msgspec.data import Float32, Tensor

    class Doc(Struct):
        w: Tensor[1, Float32]

    # custom adapters hook a third-party tensor lib: they convert user <->
    # TensorHandle while the runtime still handles the actual bytes.
    src = msgspec.javascript.codec(Doc, tensor_encoder="toH", tensor_decoder="fromH")
    assert "w.tensor(toH(" in src
    assert "fromH(r.tensor())" in src
    assert "encTensorJSON(toH(" in src
    assert "fromH(decTensorJSON(" in src


@needs_node
def test_node_tensor_roundtrip(tmp_path):
    np = pytest.importorskip("numpy")
    from msgspec.data import Bool, Float32, Int64, Tensor, UInt8

    class Doc(Struct):
        name: str
        arr: Tensor[(2, 3), UInt8]
        weights: Tensor[1, Float32]
        big: Tensor[None, Int64]
        mask: Tensor[(2, 2), Bool]
        bare: Tensor

    value = Doc(
        name="x",
        arr=np.arange(6, dtype=np.uint8).reshape(2, 3),
        weights=np.array([1.5, 2.5, -0.5], dtype=np.float32),
        big=np.array([1, -2, 2**62], dtype=np.int64),
        mask=np.array([[True, False], [False, True]], dtype=np.bool_),
        bare=np.array([3.5, 4.5], dtype=np.float64),
    )
    ref_mp = msgspec.msgpack.encode(value, type=Doc).hex()
    ref_json = msgspec.json.encode(value, type=Doc).decode()

    driver = f"""
    import {{ msgpack, json, TensorHandle }} from "./codec.mjs";
    const refMp = Uint8Array.from(Buffer.from({json.dumps(ref_mp)}, "hex"));
    const refJson = {json.dumps(ref_json)};
    const dMp = msgpack.decode(refMp);
    const dJson = json.decode(refJson);
    console.log(JSON.stringify({{
      mpHex: Buffer.from(msgpack.encode(dMp)).toString("hex"),
      jsonOut: json.encode(dJson),
      isHandle: dMp.arr instanceof TensorHandle,
      dtype: dMp.arr.dtype,
      shape: dMp.arr.shape,
      data: [...dMp.arr.array],
      f32: dMp.weights.array instanceof Float32Array,
      bigOk: dMp.big.array[2] === (2n ** 62n),
      big64: dMp.big.array instanceof BigInt64Array,
      bareDtype: dJson.bare.dtype,
    }}));
    """
    out = json.loads(_run_node(msgspec.javascript.codec(Doc), driver, tmp_path))
    assert out["mpHex"] == ref_mp  # msgpack byte-identical
    assert out["jsonOut"] == ref_json  # json byte-identical
    assert out["isHandle"] is True
    assert out["dtype"] == "uint8" and out["shape"] == [2, 3]
    assert out["data"] == [0, 1, 2, 3, 4, 5]
    assert out["f32"] is True and out["big64"] is True and out["bigOk"] is True
    assert out["bareDtype"] == "float64"


@needs_node
def test_node_tensor_decode_from_node_buffer(tmp_path):
    """Regression: decoding a tensor from a raw Node `Buffer` must not throw.

    `Buffer.prototype.slice()` returns an unaligned *view* (not a copy), so the
    packed bytes must be copied to a fresh zero-offset buffer before the
    typed-array view - otherwise `new Float64Array(buf, offset, ...)` raises a
    RangeError when `offset` isn't a multiple of 8.
    """
    np = pytest.importorskip("numpy")
    from msgspec.data import Float64, Tensor

    class Doc(Struct):
        name: str  # shifts the tensor to a non-8-aligned offset
        m: Tensor[(2, 2), Float64]

    value = Doc(name="abc", m=np.array([[1.5, 2.5], [3.5, 4.5]], dtype=np.float64))
    mp_hex = msgspec.msgpack.encode(value, type=Doc).hex()

    driver = f"""
    import {{ msgpack, TensorHandle }} from "./codec.mjs";
    // a real Node Buffer (its .slice() returns a view, unlike Uint8Array)
    const buf = Buffer.from({json.dumps(mp_hex)}, "hex");
    const d = msgpack.decode(buf);
    console.log(JSON.stringify({{
      isBuffer: Buffer.isBuffer(buf),
      handle: d.m instanceof TensorHandle,
      f64: d.m.array instanceof Float64Array,
      data: [...d.m.array].join(","),
    }}));
    """
    out = json.loads(_run_node(msgspec.javascript.codec(Doc), driver, tmp_path))
    assert out == {
        "isBuffer": True,
        "handle": True,
        "f64": True,
        "data": "1.5,2.5,3.5,4.5",
    }


def test_array_codec_structure():
    from msgspec.data import Array, Float32, UInt8

    class Doc(Struct):
        a: Array[UInt8]
        b: Array[Float32]

    src = msgspec.javascript.codec(Doc)
    # flat Arrays use the plain typed-array runtime (no TensorHandle wrapper)
    assert 'w.array(v["a"], "uint8")' in src
    assert 'r.array("float32")' in src
    assert "export { TensorHandle };" not in src  # no handle re-export for arrays


@needs_node
def test_node_array_roundtrip(tmp_path):
    np = pytest.importorskip("numpy")
    from msgspec.data import Array, Float32, Int64, UInt8

    class Doc(Struct):
        name: str
        bytes8: Array[UInt8]
        weights: Array[Float32]
        ids: Array[Int64]

    value = Doc(
        name="x",
        bytes8=np.array([0, 1, 2, 3], dtype=np.uint8),
        weights=np.array([1.5, 2.5, -0.5], dtype=np.float32),
        ids=np.array([1, -2, 2**62], dtype=np.int64),
    )
    ref_mp = msgspec.msgpack.encode(value, type=Doc).hex()
    ref_json = msgspec.json.encode(value, type=Doc).decode()

    driver = f"""
    import {{ msgpack, json }} from "./codec.mjs";
    const refMp = Uint8Array.from(Buffer.from({json.dumps(ref_mp)}, "hex"));
    const dMp = msgpack.decode(refMp);
    const dJson = json.decode({json.dumps(ref_json)});
    console.log(JSON.stringify({{
      mpHex: Buffer.from(msgpack.encode(dMp)).toString("hex"),
      jsonOut: json.encode(dJson),
      u8: dMp.bytes8 instanceof Uint8Array,
      f32: dMp.weights instanceof Float32Array,
      i64: dMp.ids instanceof BigInt64Array,
      data: [...dMp.bytes8].join(","),
      bigOk: dMp.ids[2] === (2n ** 62n),
    }}));
    """
    out = json.loads(_run_node(msgspec.javascript.codec(Doc), driver, tmp_path))
    assert out["mpHex"] == ref_mp and out["jsonOut"] == ref_json
    assert out["u8"] and out["f32"] and out["i64"] and out["bigOk"]
    assert out["data"] == "0,1,2,3"
