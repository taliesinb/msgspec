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


# --- elide_implied_tag ------------------------------------------------------


def test_codec_elide_implied_tag_structure():
    class Cat(Struct, tag="cat"):
        name: str

    class Dog(Struct, tag="dog"):
        legs: int

    class Doc(Struct):
        mono: Cat
        poly: Union[Cat, Dog]

    out = msgspec.javascript.codec(Doc, elide_implied_tag=True)
    # tagged struct encoders take a `tagged` param and write the tag
    # conditionally
    assert "export function encodeCat(w, v, tagged)" in out
    assert "w.mapHeader(tagged ? 2 : 1);" in out
    # monomorphic position passes false, union dispatch passes true
    assert 'encodeCat(w, v["mono"], false);' in out
    assert 'case "cat": encodeCat(w, _u, true); break;' in out or (
        'encodeCat(w, v["poly"], true); break;' in out
    )
    # json namespace mirrors it
    assert "export function encodeJsonCat(v, tagged)" in out
    # decode is unchanged: still seeds the tag on the object
    assert 'const o = { type: "cat" };' in out


def test_codec_elide_implied_tag_off_by_default():
    class Cat(Struct, tag="cat"):
        name: str

    out = msgspec.javascript.codec(Cat)
    assert "export function encodeCat(w, v)" in out
    assert "tagged" not in out


@needs_node
def test_node_elide_implied_tag_byte_compat(tmp_path):
    class Cat(Struct, tag="cat"):
        name: str

    class Dog(Struct, tag="dog"):
        legs: int

    class Doc(Struct):
        mono: Cat
        poly: Union[Cat, Dog]

    value = Doc(mono=Cat(name="Reo"), poly=Dog(legs=4))
    expected = msgspec.msgpack.encode(value, type=Doc, elide_implied_tag=True)
    assert b"mono" in expected and expected.count(b"cat") == 0

    expected_json = msgspec.json.encode(value, type=Doc, elide_implied_tag=True)

    js_obj = {
        "mono": {"type": "cat", "name": "Reo"},
        "poly": {"type": "dog", "legs": 4},
    }
    driver = f"""
    import {{ msgpack, json }} from "./codec.mjs";
    const obj = {json.dumps(js_obj)};
    const bytes = msgpack.encode(obj);
    const hex = Buffer.from(bytes).toString("hex");
    const back = msgpack.decode(bytes);
    const roundtrips = JSON.stringify(back) === JSON.stringify(obj);
    const text = json.encode(obj);
    const jback = json.decode(text);
    const jroundtrips = JSON.stringify(jback) === JSON.stringify(obj);
    console.log(JSON.stringify({{ hex, roundtrips, text, jroundtrips }}));
    """
    codec_src = msgspec.javascript.codec(Doc, elide_implied_tag=True)
    out = json.loads(_run_node(codec_src, driver, tmp_path))
    assert out["hex"] == expected.hex()
    assert out["roundtrips"] is True
    assert out["text"] == expected_json.decode()
    assert out["jroundtrips"] is True
    # and Python can decode the JS bytes
    assert msgspec.msgpack.decode(bytes.fromhex(out["hex"]), type=Doc) == value


# --- schema ----------------------------------------------------------------


def js(type):
    return msgspec.javascript.schema(type)


class TestSchema:
    def test_struct_basic(self):
        class Point(Struct):
            x: int
            y: int

        out = js(Point)
        assert "export class Point {" in out
        assert "  /** @type {number} */\n  x;" in out
        assert "  /** @type {number} */\n  y;" in out
        assert "@param {Object} fields" in out
        assert "@param {number} fields.x" in out
        assert "constructor({ x, y }) {" in out
        assert "this.x = x;" in out

    def test_docstring_emitted_as_jsdoc(self):
        class Point(Struct):
            """A point in 2d."""

            x: int

        out = js(Point)
        assert "/** A point in 2d. */\nexport class Point {" in out

    def test_defaults_in_constructor(self):
        class Config(Struct):
            a: "int | None" = None
            b: list = []
            c: set = set()
            d: dict = {}
            flag: bool = True
            name: str = "hi"

        out = js(Config)
        assert (
            "constructor({ a = null, b = [], c = new Set(), d = {}, "
            "flag = true, name = \"hi\" } = {}) {" in out
        )
        # all-optional: the fields object itself is optional
        assert "@param {Object} [fields]" in out
        assert "@param {number | null} [fields.a]" in out

    def test_tag_field_initializer_not_constructor_param(self):
        class Doc(Struct, tag="doc"):
            body: str

        out = js(Doc)
        assert '  /** @type {"doc"} */\n  type = "doc";' in out
        assert "constructor({ body }) {" in out
        assert "fields.type" not in out

    def test_enum(self):
        class Fruit(enum.Enum):
            """Some fruit."""

            APPLE = "apple"
            BANANA = "banana"

        out = js(Fruit)
        assert "@enum {string}" in out
        assert "Some fruit." in out
        assert "export const Fruit = Object.freeze({" in out
        assert '  APPLE: "apple",' in out
        assert "});" in out

    def test_int_enum(self):
        class Level(enum.IntEnum):
            LOW = 1
            HIGH = 2

        out = js(Level)
        assert "@enum {number}" in out
        assert "  LOW: 1," in out

    def test_enum_default_uses_member_access(self):
        class Fruit(enum.Enum):
            APPLE = "apple"

        class Basket(Struct):
            fruit: Fruit = Fruit.APPLE

        out = js(Basket)
        assert "fruit = Fruit.APPLE" in out

    def test_abstract_struct_is_typedef(self):
        class Animal(Struct, abstract=True, tag_field="kind"):
            name: str

        class Dog(Animal, tag="dog"):
            pass

        class Cat(Animal, tag="cat"):
            pass

        out = js(Animal)
        assert "/** @typedef {Dog | Cat} Animal */" in out
        assert "export class Dog {" in out
        assert '  kind = "dog";' in out

    def test_newtype_alias_is_typedef(self):
        from typing import NewType

        UserId = NewType("UserId", int)

        class User(Struct):
            id: UserId

        out = js(User)
        assert "/** @typedef {number} UserId */" in out
        assert "  /** @type {UserId} */\n  id;" in out

    def test_root_typedef_for_non_nameable(self):
        class Point(Struct):
            x: int

        out = js(list[Point])
        assert "/** @typedef {Array<Point>} Root */" in out

    def test_reserved_word_field_renamed_binding(self):
        class Weird(Struct):
            default: "int | None" = None

        out = js(Weird)
        # `default` is fine as a property, but not as a bare binding.
        assert "  /** @type {number | null} */\n  default;" in out
        assert "constructor({ default: default_ = null } = {}) {" in out
        assert "this.default = default_;" in out

    def test_non_identifier_field_uses_inline_param_type(self):
        class Weird(Struct):
            odd_name: str = msgspec.field(default="hi", name="odd-name")

        out = js(Weird)
        # Dotted @param paths can't hold quoted names; falls back to an
        # inline object type.
        assert '@param {{ "odd-name"?: string }} [fields]' in out
        assert 'constructor({ "odd-name": odd_name = "hi" } = {}) {' in out
        assert 'this["odd-name"] = odd_name;' in out

    def test_bigint_default_gets_n_suffix(self):
        class Big(Struct):
            n: Int64 = 5

        out = js(Big)
        assert "n = 5n" in out
        assert "  /** @type {bigint} */\n  n;" in out

    def test_schema_components_multiple(self):
        class A(Struct):
            x: int

        class B(Struct):
            a: A

        refs, components = msgspec.javascript.schema_components([B, list[A]])
        assert refs == ("B", "Array<A>")
        assert set(components) == {"A", "B"}
        assert "export class A {" in components["A"]

    @needs_node
    def test_node_constructs_and_serializes(self, tmp_path):
        class Fruit(enum.Enum):
            APPLE = "apple"
            BANANA = "banana"

        class Animal(Struct, abstract=True, tag_field="kind"):
            name: str

        class Dog(Animal, tag="dog"):
            barks: bool = True

        class Cat(Animal, tag="cat"):
            lives: int = 9

        class Config(Struct):
            pet: Union[Animal, None] = None
            fruit: Fruit = Fruit.BANANA
            tags: list = []

        src = js(Config)
        driver = """
        const c = new Config({ pet: new Dog({ name: "rex" }) });
        console.log(JSON.stringify(c));
        """
        path = tmp_path / "schema.mjs"
        path.write_text(src + driver)
        out = subprocess.run(
            [NODE, str(path)], capture_output=True, text=True, check=True
        ).stdout.strip()
        assert json.loads(out) == {
            "pet": {"kind": "dog", "name": "rex", "barks": True},
            "fruit": "banana",
            "tags": [],
        }

    @pytest.mark.skipif(
        shutil.which("tsc") is None, reason="tsc not installed"
    )
    def test_tsc_strict_checkjs(self, tmp_path):
        class Fruit(enum.Enum):
            APPLE = "apple"

        class Animal(Struct, abstract=True, tag_field="kind"):
            name: str

        class Dog(Animal, tag="dog"):
            barks: bool = True

        class Weird(Struct):
            default: "int | None" = None
            odd_name: str = msgspec.field(default="hi", name="odd-name")

        class Config(Struct):
            pet: Union[Animal, None] = None
            fruit: Fruit = Fruit.APPLE
            w: Union[Weird, None] = None

        src = js(Config)
        use = """
        const c = new Config({ pet: new Dog({ name: "rex" }) });
        const w = new Weird({ "odd-name": "x" });
        console.log(JSON.stringify(c), w.default);
        """
        path = tmp_path / "schema.mjs"
        path.write_text(src + use)
        res = subprocess.run(
            [
                shutil.which("tsc"),
                "--allowJs",
                "--checkJs",
                "--noEmit",
                "--strict",
                "--target",
                "es2022",
                str(path),
            ],
            capture_output=True,
            text=True,
        )
        assert res.returncode == 0, res.stdout + res.stderr
