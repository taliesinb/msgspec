import datetime
import enum
import json as _json
import shutil
import subprocess
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

NODE = shutil.which("node")
TSC = shutil.which("tsc")
needs_node = pytest.mark.skipif(NODE is None, reason="node not installed")
needs_tsc = pytest.mark.skipif(TSC is None, reason="tsc not installed")


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


def test_array_like_struct_is_class():
    # array_like structs are classes in the schema (the array wire form is a
    # codec concern, not a schema one).
    class Rec(Struct, array_like=True):
        a: int
        b: str = "x"

    out = ts(Rec)
    assert "export class Rec {" in out
    assert "  a: number;" in out
    assert "  b?: string;" in out


def test_array_like_tagged_struct_is_class_with_tag():
    class Rec(Struct, array_like=True, tag="rec"):
        a: int

    out = ts(Rec)
    assert "export class Rec {" in out
    assert '  type: "rec";' in out
    assert "  a: number;" in out


def test_namedtuple_is_class():
    from typing import NamedTuple

    class Point(NamedTuple):
        x: int
        y: int

    out = ts(Point)
    assert "export class Point {" in out
    assert "  x: number;" in out
    assert "  y: number;" in out


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


def cc(type):
    # These tests cover the legacy `@msgpack/msgpack` path.
    return msgspec.typescript.codec(type, embed_msgpack=False)


class TestCodec:
    def test_imports_msgpack(self):
        class P(Struct):
            x: int

        out = cc(P)
        assert 'from "@msgpack/msgpack"' in out

    def test_includes_schema_and_toplevel(self):
        class P(Struct):
            x: int

        out = cc(P)
        assert "export class P {" in out
        assert "export function encode(value: P): Uint8Array {" in out
        assert "return _mpEncode(encodeP(value));" in out
        assert "export function decode(bytes: Uint8Array): P {" in out
        assert "return decodeP(_mpDecode(bytes)) as P;" in out

    def test_struct_encode_decode(self):
        class P(Struct):
            x: int
            y: str

        out = cc(P)
        assert "export function encodeP(value: P): unknown {" in out
        assert 'x: value["x"],' in out
        assert "export function decodeP(data: unknown): P {" in out
        assert 'x: (o["x"] as number),' in out

    def test_tagged_union_emits_tag_on_encode(self):
        class Cat(Struct, tag="cat"):
            n: int

        class Dog(Struct, tag="dog"):
            n: int

        class Holder(Struct):
            pet: Union[Cat, Dog]

        out = cc(Holder)
        # encoder for a tagged struct emits the literal tag
        assert 'type: "cat",' in out
        # union encode/decode dispatch on the tag field
        assert 'switch (v["type"]) {' in out
        assert 'case "cat": return encodeCat(v);' in out
        assert 'case "cat": return decodeCat(v);' in out

    def test_optional_null_check(self):
        class P(Struct):
            x: int

        class Holder(Struct):
            maybe: Optional[P]

        out = cc(Holder)
        assert '=== null ? null : encodeP(value["maybe"])' in out
        assert '=== null ? null : decodeP(o["maybe"])' in out

    def test_list_maps(self):
        class P(Struct):
            x: int

        class Holder(Struct):
            items: List[P]

        out = cc(Holder)
        assert 'value["items"].map((v) => encodeP(v))' in out
        assert '(o["items"] as unknown[]).map((v) => decodeP(v))' in out

    def test_scalar_identity(self):
        class P(Struct):
            a: int
            b: str
            c: bool

        out = cc(P)
        assert 'a: value["a"],' in out  # no transform
        assert 'a: (o["a"] as number),' in out

    def test_non_nameable_root_alias(self):
        class P(Struct):
            x: int

        out = cc(List[P])
        assert "export type Root = Array<P>;" in out
        assert "value.map((v) => encodeP(v))" in out

    def test_array_like_struct_object_array_bridge(self):
        # array_like structs are objects in TS; the codec bridges to an array.
        class Rec(Struct, array_like=True):
            a: int
            b: str

        out = cc(Rec)
        assert "export class Rec {" in out
        assert 'return [value["a"], value["b"]];' in out  # encode: object -> array
        assert "const a = data as unknown[];" in out
        assert "a: (a[0] as number)," in out  # decode: array -> object

    def test_namedtuple_object_array_bridge(self):
        from typing import NamedTuple

        class Point(NamedTuple):
            x: int
            y: int

        out = cc(Point)
        assert "export class Point {" in out
        assert 'return [value["x"], value["y"]];' in out
        assert "x: (a[0] as number)," in out

    def test_tagged_array_like_bridge(self):
        class Rec(Struct, array_like=True, tag="rec"):
            a: int

        out = cc(Rec)
        # tag is the first wire array element; decoded object carries it back
        assert 'return ["rec", value["a"]];' in out
        assert '    type: "rec",' in out
        assert "a: (a[1] as number)," in out

    def test_bigint_scalar_conversion(self):
        from msgspec.data import Float, Int

        class Foo(Struct):
            a: int  # plain int -> number, identity
            b: Int  # -> bigint, converted
            c: Float  # -> number, identity

        out = cc(Foo)
        assert 'a: value["a"],' in out
        assert 'b: _toSafeInt(value["b"]),' in out  # encode: checked bigint->number
        assert 'c: value["c"],' in out
        assert 'a: (o["a"] as number),' in out
        assert 'b: BigInt(o["b"] as number | bigint),' in out  # decode -> bigint
        assert 'c: (o["c"] as number),' in out

    def test_int64_uint64_also_converted(self):
        from msgspec.data import Int64, UInt64

        class Foo(Struct):
            a: Int64
            b: UInt64

        out = cc(Foo)
        assert "Number(value[" not in out  # replaced by the checked helper
        assert '_toSafeInt(value["a"])' in out
        assert "BigInt(o[" in out

    def test_force_int64_false_emits_safe_check(self):
        from msgspec.data import Int

        class Foo(Struct):
            b: Int

        out = msgspec.typescript.codec(Foo, embed_msgpack=False)  # default force_int64=False
        assert "function _toSafeInt(" in out
        assert 'b: _toSafeInt(value["b"]),' in out
        assert "useBigInt64" not in out

    def test_force_int64_true_uses_usebigint64(self):
        from msgspec.data import Int

        class Foo(Struct):
            b: Int

        out = msgspec.typescript.codec(Foo, force_int64=True, embed_msgpack=False)
        assert "_toSafeInt" not in out
        assert "const _codecOptions = { useBigInt64: true };" in out
        assert 'b: value["b"],' in out  # bigint passed through directly
        assert "_mpEncode(encodeFoo(value), _codecOptions)" in out
        assert "_mpDecode(bytes, _codecOptions)" in out

    def test_no_bigint_no_helper(self):
        class Foo(Struct):
            x: int  # plain int -> number, no bigint machinery

        out = cc(Foo)
        assert "_toSafeInt" not in out
        assert "useBigInt64" not in out

    def test_enum_and_alias_identity(self):
        import enum

        class Color(enum.Enum):
            RED = "red"

        class P(Struct):
            c: Color

        out = cc(P)
        assert 'c: value["c"],' in out
        assert 'c: (o["c"] as Color),' in out

    def test_no_tensor_runtime_when_absent(self):
        class P(Struct):
            x: int

        out = cc(P)
        assert "TensorHandle" not in out
        assert "ExtensionCodec" not in out
        assert "_codecOptions" not in out


class TestCodecTensor:
    def _layer(self):
        from msgspec.data import Float32, Tensor

        class Layer(Struct):
            name: str
            weights: Tensor[1, Float32]

        return Layer

    def test_requires_hooks(self):
        Layer = self._layer()
        with pytest.raises(TypeError, match="tensor_encoder"):
            msgspec.typescript.codec(Layer, embed_msgpack=False)
        with pytest.raises(TypeError, match="tensor_decoder"):
            msgspec.typescript.codec(Layer, tensor_encoder="wrap", embed_msgpack=False)

    def test_emits_runtime_and_hooks(self):
        Layer = self._layer()
        out = msgspec.typescript.codec(
            Layer, tensor_encoder="wrapT", tensor_decoder="unwrapT", embed_msgpack=False
        )
        # import + runtime
        assert "ExtensionCodec" in out
        assert "export class TensorHandle {" in out
        assert "data: Uint8Array;" in out
        assert "type: _TENSOR_EXT_TYPE," in out
        # hooks called at the tensor position
        assert 'weights: wrapT(value["weights"]),' in out
        assert 'weights: unwrapT(o["weights"] as TensorHandle),' in out
        # options threaded to encode/decode
        assert "_mpEncode(encodeLayer(value), _codecOptions)" in out
        assert "_mpDecode(bytes, _codecOptions)" in out

    def test_tensor_ext_code_84(self):
        Layer = self._layer()
        out = msgspec.typescript.codec(
            Layer, tensor_encoder="wrapT", tensor_decoder="unwrapT", embed_msgpack=False
        )
        assert "const _TENSOR_EXT_TYPE = 84;" in out


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

    # `Scalar` is the accurate union `Int | Float | Bool` (Int -> bigint).
    (root,), _ = msgspec.typescript.schema_components([md.Scalar])
    assert root == "bigint | number | boolean"


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
    assert "  bias: bigint | number | boolean;" in out
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


class TestCodecEmbed:
    """The default `embed_msgpack=True` path: our own inlined msgpack runtime,
    namespaced `msgpack`/`json` exports, byte/format-compatible with a
    type-directed `msgspec` encoder."""

    def _doc(self):
        from msgspec.data import Float32, Int64, Scalar

        class Cat(Struct, tag="cat"):
            name: str
            lives: Int64

        class Dog(Struct, tag="dog"):
            legs: int

        class Doc(Struct):
            pet: Union[Cat, Dog]
            ratio: Float32
            blob: bytes
            scores: Dict[str, Int64]
            val: Scalar
            maybe: Optional[str]

        return Doc

    def test_structure(self):
        out = msgspec.typescript.codec(self._doc())
        assert "import" not in out  # self-contained, no @msgpack/msgpack
        assert "class Writer" in out and "class Reader" in out
        assert "export const msgpack = {" in out
        assert "export const json = {" in out
        # struct shapes are interfaces; bytes is Uint8Array; Int64 is bigint
        assert "export interface Doc {" in out
        assert "  blob: Uint8Array;" in out
        assert "  lives: bigint;" in out
        assert "w.float32(v[\"ratio\"]);" in out

    def test_flags(self):
        class P(Struct):
            x: int

        assert "export const json = {" not in msgspec.typescript.codec(P, json=False)
        assert "export const msgpack = {" not in msgspec.typescript.codec(
            P, msgpack=False
        )
        with pytest.raises(ValueError):
            msgspec.typescript.codec(P, msgpack=False, json=False)

    @needs_tsc
    def test_tsc_strict_clean(self, tmp_path):
        src = msgspec.typescript.codec(self._doc())
        (tmp_path / "codec.ts").write_text(src)
        res = subprocess.run(
            [TSC, "--strict", "--noEmit", "--target", "es2020",
             "--lib", "es2020,dom", str(tmp_path / "codec.ts")],
            capture_output=True, text=True,
        )
        assert res.returncode == 0, res.stdout + res.stderr

    @needs_node
    def test_node_roundtrip_matches_python(self, tmp_path):
        from msgspec.data import Int64, Scalar

        class Cat(Struct, tag="cat"):
            name: str
            lives: Int64

        class Doc(Struct):
            pet: Cat
            blob: bytes
            val: Scalar

        value = Doc(pet=Cat(name="Reo", lives=2**62), blob=b"\x00\x01", val=2**61)
        mp_hex = msgspec.msgpack.encode(value, type=Doc).hex()
        js = msgspec.json.encode(value, type=Doc).decode()

        (tmp_path / "codec.ts").write_text(msgspec.typescript.codec(Doc))
        driver = f"""
        import {{ msgpack, json }} from "./codec.ts";
        const obj: any = {{ pet: {{ type: "cat", name: "Reo", lives: 2n ** 62n }},
                            blob: new Uint8Array([0, 1]), val: 2n ** 61n }};
        const mpHex = Buffer.from(msgpack.encode(obj)).toString("hex");
        const jsText = json.encode(obj);
        const mb: any = msgpack.decode(Uint8Array.from(Buffer.from({_json.dumps(mp_hex)}, "hex")));
        const jb: any = json.decode({_json.dumps(js)});
        console.log(JSON.stringify({{
          mp: mpHex === {_json.dumps(mp_hex)},
          json: jsText === {_json.dumps(js)},
          dec: mb.pet.lives === 2n ** 62n && jb.val === 2n ** 61n
               && [...jb.blob].join(",") === "0,1",
        }}));
        """
        (tmp_path / "driver.ts").write_text(driver)
        res = subprocess.run(
            [NODE, "--experimental-strip-types", str(tmp_path / "driver.ts")],
            capture_output=True, text=True,
        )
        assert res.returncode == 0, res.stderr
        out = _json.loads(res.stdout.strip())
        assert out == {"mp": True, "json": True, "dec": True}

    def _tensor_doc(self):
        from msgspec.data import Float32, Tensor, UInt8

        class Doc(Struct):
            name: str
            arr: Tensor[(2, 3), UInt8]
            weights: Tensor[1, Float32]

        return Doc

    def test_tensor_structure(self):
        # the embedded codec now handles tensors (no longer raises); a tensor
        # field decodes to a TensorHandle, which is re-exported.
        out = msgspec.typescript.codec(self._tensor_doc())
        assert "export { TensorHandle };" in out
        assert "  arr: TensorHandle;" in out
        assert "w.tensor(" in out and "r.tensor()" in out

    @needs_tsc
    def test_tsc_tensor_clean(self, tmp_path):
        (tmp_path / "codec.ts").write_text(
            msgspec.typescript.codec(self._tensor_doc())
        )
        res = subprocess.run(
            [TSC, "--strict", "--noEmit", "--target", "es2020",
             "--lib", "es2020,dom", str(tmp_path / "codec.ts")],
            capture_output=True, text=True,
        )
        assert res.returncode == 0, res.stdout + res.stderr

    @needs_node
    def test_node_tensor_roundtrip(self, tmp_path):
        np = pytest.importorskip("numpy")
        Doc = self._tensor_doc()
        value = Doc(
            name="x",
            arr=np.arange(6, dtype=np.uint8).reshape(2, 3),
            weights=np.array([1.5, 2.5, -0.5], dtype=np.float32),
        )
        mp_hex = msgspec.msgpack.encode(value, type=Doc).hex()
        js = msgspec.json.encode(value, type=Doc).decode()

        (tmp_path / "codec.ts").write_text(msgspec.typescript.codec(Doc))
        driver = f"""
        import {{ msgpack, json, TensorHandle }} from "./codec.ts";
        const mb: any = msgpack.decode(Uint8Array.from(Buffer.from({_json.dumps(mp_hex)}, "hex")));
        const jb: any = json.decode({_json.dumps(js)});
        console.log(JSON.stringify({{
          mp: Buffer.from(msgpack.encode(mb)).toString("hex") === {_json.dumps(mp_hex)},
          json: json.encode(jb) === {_json.dumps(js)},
          handle: mb.arr instanceof TensorHandle,
          meta: mb.arr.dtype === "uint8" && JSON.stringify(mb.arr.shape) === "[2,3]",
          data: [...mb.arr.array].join(",") === "0,1,2,3,4,5",
          f32: mb.weights.array instanceof Float32Array,
        }}));
        """
        (tmp_path / "driver.ts").write_text(driver)
        res = subprocess.run(
            [NODE, "--experimental-strip-types", str(tmp_path / "driver.ts")],
            capture_output=True, text=True,
        )
        assert res.returncode == 0, res.stderr
        out = _json.loads(res.stdout.strip())
        assert out == {"mp": True, "json": True, "handle": True,
                       "meta": True, "data": True, "f32": True}
