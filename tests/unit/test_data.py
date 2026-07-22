import array
import gc

import pytest

import msgspec
from msgspec import Struct
from msgspec.data import Float32, Tensor, TensorHandle, UInt8
from msgspec.msgpack import Decoder, Ext, decode, encode

DTYPES = [
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "int8",
    "int16",
    "int32",
    "int64",
    "float32",
    "float64",
    "bool",
]


class TestTensorHandleType:
    def test_construct_all_fields(self):
        h = TensorHandle(b"abc", dtype="float32", shape=(1, 3))
        assert h.native == b"abc"
        assert h.dtype == "float32"
        assert h.shape == (1, 3)

    def test_construct_positional(self):
        h = TensorHandle(b"abc", "int8", (3,))
        assert h.dtype == "int8"
        assert h.shape == (3,)

    def test_defaults(self):
        h = TensorHandle(b"abc")
        assert h.dtype is None
        assert h.shape is None

    def test_fields_readonly(self):
        h = TensorHandle(b"abc")
        with pytest.raises(AttributeError):
            h.native = b"xyz"

    def test_repr(self):
        h = TensorHandle(b"abc", dtype="uint8", shape=(3,))
        assert repr(h) == "TensorHandle(native=b'abc', dtype='uint8', shape=(3,))"

    def test_bad_dtype_type(self):
        with pytest.raises(TypeError, match="dtype must be a str or None"):
            TensorHandle(b"abc", dtype=123)

    def test_bad_shape_type(self):
        with pytest.raises(TypeError, match="shape must be a tuple or None"):
            TensorHandle(b"abc", shape=[1, 2])

    def test_missing_native(self):
        with pytest.raises(TypeError):
            TensorHandle()


class TestRoundtrip:
    @pytest.mark.parametrize("dtype", DTYPES)
    def test_dtype_roundtrip(self, dtype):
        h = TensorHandle(b"\x00\x01\x02\x03", dtype=dtype, shape=(4,))
        out = decode(encode(h))
        assert out.dtype == dtype
        assert out.shape == (4,)
        assert bytes(out.native) == b"\x00\x01\x02\x03"

    @pytest.mark.parametrize("size", [0, 1, 2, 4, 8, 16, 17, 255, 256, 70000])
    def test_size_roundtrip(self, size):
        data = bytes(range(256)) * (size // 256) + bytes(range(size % 256))
        h = TensorHandle(data, dtype="uint8", shape=(size,))
        out = decode(encode(h))
        assert bytes(out.native) == data
        assert out.shape == (size,)

    @pytest.mark.parametrize("shape", [(), (0,), (3,), (2, 3), (1, 1, 1, 1), (4, 5, 6)])
    def test_shape_roundtrip(self, shape):
        n = 1
        for s in shape:
            n *= s
        data = bytes(n)
        h = TensorHandle(data, dtype="uint8", shape=shape)
        out = decode(encode(h))
        assert out.shape == shape

    def test_none_dtype_and_shape(self):
        h = TensorHandle(b"xyz")
        out = decode(encode(h))
        assert out.dtype is None
        assert out.shape is None
        assert bytes(out.native) == b"xyz"

    def test_native_is_memoryview_of_bytes(self):
        out = decode(encode(TensorHandle(b"abc", dtype="uint8", shape=(3,))))
        assert isinstance(out.native, memoryview)
        # backed by a bytes object
        assert out.native.obj == b"abc"

    def test_buffer_native(self):
        buf = array.array("f", [1.0, 2.0, 3.0])
        h = TensorHandle(buf, dtype="float32", shape=(3,))
        out = decode(encode(h))
        assert array.array("f", bytes(out.native)).tolist() == [1.0, 2.0, 3.0]


class TestTypedDecode:
    def test_tensor_annotation(self):
        h = TensorHandle(b"\x01\x02\x03", dtype="uint8", shape=(3,))
        out = Decoder(Tensor[1, UInt8]).decode(encode(h))
        assert isinstance(out, TensorHandle)
        assert out.shape == (3,)

    def test_bare_tensor_annotation(self):
        h = TensorHandle(b"\x01\x02", dtype="uint8", shape=(2,))
        out = Decoder(Tensor).decode(encode(h))
        assert isinstance(out, TensorHandle)

    def test_tensorhandle_annotation(self):
        h = TensorHandle(b"\x01\x02", dtype="uint8", shape=(2,))
        out = Decoder(TensorHandle).decode(encode(h))
        assert isinstance(out, TensorHandle)

    def test_struct_field(self):
        class Layer(Struct):
            weights: Tensor[1, Float32]
            name: str

        msg = encode(
            Layer(weights=TensorHandle(b"\x00" * 12, dtype="float32", shape=(3,)), name="w")
        )
        out = Decoder(Layer).decode(msg)
        assert isinstance(out.weights, TensorHandle)
        assert out.name == "w"

    def test_any_decode(self):
        h = TensorHandle(b"\x01\x02", dtype="uint8", shape=(2,))
        out = decode(encode(h))
        assert isinstance(out, TensorHandle)

    def test_tensor_field_rejects_non_tensor(self):
        class Layer(Struct):
            weights: Tensor[3, Float32]

        # encode a struct whose field is an int instead of a tensor
        msg = encode({"weights": 1})
        with pytest.raises(msgspec.ValidationError, match="Expected `tensor`"):
            Decoder(Layer).decode(msg)


class TestEncodeErrors:
    def test_bad_dtype(self):
        with pytest.raises(msgspec.EncodeError, match="Invalid tensor dtype"):
            encode(TensorHandle(b"ab", dtype="nope", shape=(2,)))

    def test_non_buffer_native(self):
        with pytest.raises(TypeError):
            encode(TensorHandle(123, dtype="uint8", shape=(1,)))

    def test_too_many_dims(self):
        with pytest.raises(msgspec.EncodeError, match="254 dimensions"):
            encode(TensorHandle(b"", dtype="uint8", shape=(0,) * 255))


class TestExtInteraction:
    def test_ext_hook_other_codes_unaffected(self):
        def ext_hook(code, buf):
            return ("hook", code, bytes(buf))

        assert decode(encode(Ext(5, b"abc")), ext_hook=ext_hook) == ("hook", 5, b"abc")

    def test_tensor_takes_precedence_over_ext_hook(self):
        def ext_hook(code, buf):  # pragma: no cover - should not fire
            raise AssertionError("ext_hook should not be called for tensor code")

        h = TensorHandle(b"\x01\x02", dtype="uint8", shape=(2,))
        out = decode(encode(h), ext_hook=ext_hook)
        assert isinstance(out, TensorHandle)

    def test_ext_still_roundtrips(self):
        assert decode(encode(Ext(9, b"hello"))) == Ext(9, b"hello")


class TestNumpy:
    def test_numpy_roundtrip(self):
        np = pytest.importorskip("numpy")

        arr = np.arange(6, dtype=np.float32).reshape(2, 3)
        h = TensorHandle(arr, dtype="float32", shape=arr.shape)
        out = decode(encode(h))
        back = np.frombuffer(out.native, dtype=np.float32).reshape(out.shape)
        assert np.array_equal(arr, back)

    def test_numpy_int64(self):
        np = pytest.importorskip("numpy")

        arr = np.array([[1, 2], [3, 4]], dtype=np.int64)
        h = TensorHandle(arr, dtype="int64", shape=arr.shape)
        out = decode(encode(h))
        back = np.frombuffer(out.native, dtype=np.int64).reshape(out.shape)
        assert np.array_equal(arr, back)


class TestNumpyEncode:
    def test_bare_array_auto_wraps(self):
        np = pytest.importorskip("numpy")

        arr = np.arange(6, dtype=np.float32).reshape(2, 3)
        out = decode(encode(arr))
        assert isinstance(out, TensorHandle)
        assert out.dtype == "float32"
        assert out.shape == (2, 3)
        back = np.frombuffer(out.native, dtype=np.float32).reshape(out.shape)
        assert np.array_equal(arr, back)

    def test_non_contiguous_array(self):
        np = pytest.importorskip("numpy")

        arr = np.arange(6, dtype=np.int64).reshape(2, 3).T  # F-contiguous
        out = decode(encode(arr))
        back = np.frombuffer(out.native, dtype=np.int64).reshape(out.shape)
        assert np.array_equal(arr, back)

    def test_array_as_struct_field(self):
        np = pytest.importorskip("numpy")

        class M(Struct):
            w: Tensor[2, Float32]

        arr = np.ones((2, 2), dtype=np.float32)
        out = Decoder(M).decode(encode(M(w=arr)))
        assert isinstance(out.w, TensorHandle)
        assert out.w.shape == (2, 2)

    def test_unsupported_dtype_errors(self):
        np = pytest.importorskip("numpy")

        with pytest.raises(TypeError, match="unsupported dtype"):
            encode(np.array([1 + 2j], dtype=np.complex128))

    def test_non_numpy_unknown_still_errors(self):
        with pytest.raises(TypeError):
            encode(object())


class TestDecTensorHook:
    @staticmethod
    def _to_numpy(shape, dtype, data):
        np = pytest.importorskip("numpy")
        assert isinstance(data, bytes)
        a = np.frombuffer(data, dtype=np.dtype(dtype))
        return a.reshape(shape) if shape is not None else a

    def test_module_decode_hook(self):
        np = pytest.importorskip("numpy")

        arr = np.arange(6, dtype=np.float32).reshape(2, 3)
        out = decode(encode(arr), dec_tensor=self._to_numpy)
        assert isinstance(out, np.ndarray)
        assert np.array_equal(arr, out)

    def test_decoder_hook_typed_field(self):
        np = pytest.importorskip("numpy")

        class M(Struct):
            w: Tensor[2, Float32]

        arr = np.arange(4, dtype=np.float32).reshape(2, 2)
        dec = Decoder(M, dec_tensor=self._to_numpy)
        out = dec.decode(encode(M(w=arr)))
        assert isinstance(out.w, np.ndarray)
        assert np.array_equal(arr, out.w)

    def test_hook_receives_bytes_shape_dtype(self):
        seen = {}

        def hook(shape, dtype, data):
            seen["shape"] = shape
            seen["dtype"] = dtype
            seen["data"] = data
            return "converted"

        h = TensorHandle(b"\x01\x02\x03\x04", dtype="uint8", shape=(4,))
        out = decode(encode(h), dec_tensor=hook)
        assert out == "converted"
        assert seen == {"shape": (4,), "dtype": "uint8", "data": b"\x01\x02\x03\x04"}

    def test_no_hook_yields_tensorhandle(self):
        h = TensorHandle(b"\x01\x02", dtype="uint8", shape=(2,))
        assert isinstance(decode(encode(h)), TensorHandle)

    def test_hook_attribute(self):
        def hook(shape, dtype, data):
            return None

        dec = Decoder(dec_tensor=hook)
        assert dec.dec_tensor is hook
        assert Decoder().dec_tensor is None

    def test_bad_hook(self):
        with pytest.raises(TypeError, match="dec_tensor must be callable"):
            Decoder(dec_tensor=123)
        with pytest.raises(TypeError, match="dec_tensor must be callable"):
            decode(encode(TensorHandle(b"x")), dec_tensor=123)


class TestJsonEncode:
    def test_tensorhandle_json(self):
        import json

        h = TensorHandle(b"\x00\x01\x02\x03", dtype="uint8", shape=(2, 2))
        out = json.loads(msgspec.json.encode(h))
        assert out == {
            "type": "tensor",
            "shape": [2, 2],
            "dtype": "uint8",
            "data": "AAECAw==",
        }

    def test_none_dtype_shape_json(self):
        import json

        h = TensorHandle(b"\x00\x01")
        out = json.loads(msgspec.json.encode(h))
        assert out == {"type": "tensor", "shape": None, "dtype": None, "data": "AAE="}

    def test_numpy_json(self):
        np = pytest.importorskip("numpy")
        import base64
        import json

        arr = np.arange(4, dtype=np.float32).reshape(2, 2)
        out = json.loads(msgspec.json.encode(arr))
        assert out["shape"] == [2, 2]
        assert out["dtype"] == "float32"
        back = np.frombuffer(base64.b64decode(out["data"]), dtype=np.float32).reshape(
            out["shape"]
        )
        assert np.array_equal(arr, back)


class TestJsonDecode:
    def test_typed_field_to_tensorhandle(self):
        class Layer(Struct):
            name: str
            weights: Tensor[1, UInt8]

        h = TensorHandle(b"\x00\x01\x02\x03", dtype="uint8", shape=(4,))
        msg = msgspec.json.encode({"name": "w", "weights": h})
        out = msgspec.json.Decoder(Layer).decode(msg)
        assert isinstance(out.weights, TensorHandle)
        assert out.weights.dtype == "uint8"
        assert out.weights.shape == (4,)
        assert bytes(out.weights.native) == b"\x00\x01\x02\x03"

    def test_bare_tensorhandle_target(self):
        h = TensorHandle(b"\x01\x02", dtype="float32", shape=(1,))
        msg = msgspec.json.encode(h)
        out = msgspec.json.decode(msg, type=TensorHandle)
        assert isinstance(out, TensorHandle)
        assert out.dtype == "float32"
        assert out.shape == (1,)

    def test_none_dtype_shape(self):
        h = TensorHandle(b"\x01\x02\x03")
        out = msgspec.json.decode(msgspec.json.encode(h), type=TensorHandle)
        assert out.dtype is None
        assert out.shape is None
        assert bytes(out.native) == b"\x01\x02\x03"

    def test_dec_tensor_hook(self):
        np = pytest.importorskip("numpy")

        def to_np(shape, dtype, data):
            assert isinstance(data, bytes)
            a = np.frombuffer(data, dtype=np.dtype(dtype))
            return a.reshape(shape) if shape is not None else a

        arr = np.array([1.5, 2.5, 3.5], dtype=np.float32)
        msg = msgspec.json.encode(arr)
        out = msgspec.json.decode(msg, type=Tensor[1, Float32], dec_tensor=to_np)
        assert isinstance(out, np.ndarray)
        assert np.array_equal(arr, out)

    def test_numpy_json_roundtrip(self):
        np = pytest.importorskip("numpy")

        class Layer(Struct):
            weights: Tensor[2, Float32]

        def to_np(shape, dtype, data):
            return np.frombuffer(data, dtype=np.dtype(dtype)).reshape(shape)

        arr = np.arange(6, dtype=np.float32).reshape(2, 3)
        msg = msgspec.json.encode(Layer(weights=arr))
        out = msgspec.json.Decoder(Layer, dec_tensor=to_np).decode(msg)
        assert np.array_equal(arr, out.weights)

    def test_dec_tensor_attribute_and_bad_hook(self):
        def hook(shape, dtype, data):
            return None

        dec = msgspec.json.Decoder(dec_tensor=hook)
        assert dec.dec_tensor is hook
        assert msgspec.json.Decoder().dec_tensor is None
        with pytest.raises(TypeError, match="dec_tensor must be callable"):
            msgspec.json.Decoder(dec_tensor=123)


class TestValidation:
    @pytest.mark.parametrize("fmt", ["msgpack", "json"])
    def test_dtype_mismatch(self, fmt):
        mod = getattr(msgspec, fmt)

        class Layer(Struct):
            w: Tensor[1, Float32]

        bad = Layer(w=TensorHandle(b"\x00\x00", dtype="int8", shape=(2,)))
        with pytest.raises(
            msgspec.ValidationError, match=r"Expected tensor of dtype 'float32'"
        ):
            mod.Decoder(Layer).decode(mod.encode(bad))

    @pytest.mark.parametrize("fmt", ["msgpack", "json"])
    def test_rank_mismatch(self, fmt):
        mod = getattr(msgspec, fmt)

        class Layer(Struct):
            w: Tensor[2, Float32]

        bad = Layer(w=TensorHandle(b"\x00" * 4, dtype="float32", shape=(4,)))
        with pytest.raises(msgspec.ValidationError, match=r"rank 2, got rank 1"):
            mod.Decoder(Layer).decode(mod.encode(bad))

    @pytest.mark.parametrize("fmt", ["msgpack", "json"])
    def test_size_mismatch(self, fmt):
        mod = getattr(msgspec, fmt)

        class Grid(Struct):
            g: Tensor[(2, 3), Float32]

        bad = Grid(g=TensorHandle(b"\x00" * 32, dtype="float32", shape=(2, 4)))
        with pytest.raises(msgspec.ValidationError, match=r"axis 1 of size 3, got 4"):
            mod.Decoder(Grid).decode(mod.encode(bad))

    def test_error_includes_path(self):
        class Layer(Struct):
            w: Tensor[1, Float32]

        bad = Layer(w=TensorHandle(b"\x00", dtype="int8", shape=(1,)))
        with pytest.raises(msgspec.ValidationError, match=r"at `\$.w`"):
            msgspec.msgpack.Decoder(Layer).decode(msgspec.msgpack.encode(bad))

    def test_nested_path(self):
        class Batch(Struct):
            layers: list[Tensor[1, Float32]]

        b = Batch(
            layers=[
                TensorHandle(b"\x00" * 4, dtype="float32", shape=(1,)),
                TensorHandle(b"\x00", dtype="int8", shape=(1,)),
            ]
        )
        with pytest.raises(msgspec.ValidationError, match=r"at `\$.layers\[1\]`"):
            msgspec.msgpack.Decoder(Batch).decode(msgspec.msgpack.encode(b))

    def test_valid_passes(self):
        class Grid(Struct):
            g: Tensor[(2, 3), Float32]

        ok = Grid(g=TensorHandle(b"\x00" * 24, dtype="float32", shape=(2, 3)))
        out = msgspec.msgpack.Decoder(Grid).decode(msgspec.msgpack.encode(ok))
        assert out.g.shape == (2, 3)

    @pytest.mark.parametrize("typ", [Tensor, TensorHandle])
    def test_bare_tensor_and_handle_unconstrained(self, typ):
        # a wildly-shaped/typed tensor decodes fine against an unconstrained target
        h = TensorHandle(b"\x00" * 8, dtype="int8", shape=(2, 2, 2))
        out = msgspec.msgpack.decode(msgspec.msgpack.encode(h), type=typ)
        assert isinstance(out, TensorHandle)

    def test_any_decode_not_validated(self):
        # untyped msgpack decode yields a TensorHandle without any checking
        h = TensorHandle(b"\x00", dtype="int8", shape=(1,))
        out = msgspec.msgpack.decode(msgspec.msgpack.encode(h))
        assert isinstance(out, TensorHandle)

    def test_union_with_non_none_rejected(self):
        with pytest.raises(msgspec.ValidationError):
            # decoding an int where a tensor is expected
            msgspec.msgpack.Decoder(Tensor[1, Float32]).decode(
                msgspec.msgpack.encode(1)
            )

    def test_tensor_union_invariant(self):
        with pytest.raises(TypeError, match="tensor type"):
            msgspec.msgpack.Decoder(Tensor[1, Float32] | int)
        # union with None is allowed
        msgspec.msgpack.Decoder(Tensor[1, Float32] | None)


class TestTensorInfo:
    def test_exposed_and_parsed(self):
        from msgspec._core import TensorInfo

        cls = Tensor[(2, 3), Float32]
        # building a decoder parses + caches the TensorInfo on the class
        msgspec.msgpack.Decoder(cls)
        info = cls.__msgspec_cache__
        assert isinstance(info, TensorInfo)
        assert info.ndims == 2
        assert info.sizes == (2, 3)
        assert info.dtype == "float32"

    def test_bare_tensor_info_all_none(self):
        from msgspec._core import TensorInfo

        msgspec.msgpack.Decoder(Tensor)
        info = Tensor.__msgspec_cache__
        assert isinstance(info, TensorInfo)
        assert info.ndims is None
        assert info.sizes is None
        assert info.dtype is None


def test_decode_no_reference_leak():
    h = TensorHandle(b"x" * 32, dtype="float32", shape=(8,))
    msg = encode(h)
    out = decode(msg)
    native = out.native
    del out
    gc.collect()
    # `native` is a memoryview onto a bytes object; that bytes is its only
    # extra referent and shouldn't be leaked elsewhere.
    assert isinstance(native, memoryview)
    assert bytes(native) == b"x" * 32


# --- Array (flat 1-d array) -------------------------------------------------


class TestArray:
    def test_bytes_to_array_uint8(self):
        from msgspec.data import Array, ArrayHandle, UInt8

        for enc, dec in [
            (msgspec.msgpack.encode, msgspec.msgpack.decode),
            (msgspec.json.encode, msgspec.json.decode),
        ]:
            buf = enc(b"\x00\x01\x02\x03", type=Array[UInt8])
            out = dec(buf, type=Array[UInt8])
            assert isinstance(out, ArrayHandle)
            assert bytes(out.data) == b"\x00\x01\x02\x03"
            assert out.dtype == "uint8" and out.size == 4

    def test_array_flag_distinguishes_from_tensor(self):
        # msgpack: the array flag is the high bit of the ext version byte.
        from msgspec.data import Array, UInt8

        arr = msgspec.msgpack.encode(b"\x01\x02", type=Array[UInt8])
        # ext8: c7 <len> 54 <version-byte> ...; version byte has 0x80 set
        assert arr[3] == 0x81

    def test_memoryview_float32(self):
        from msgspec.data import Array, ArrayHandle, Float32

        a = array.array("f", [1.5, 2.5, -0.5])
        out = msgspec.msgpack.decode(
            msgspec.msgpack.encode(memoryview(a), type=Array[Float32]),
            type=Array[Float32],
        )
        assert isinstance(out, ArrayHandle)
        assert out.dtype == "float32" and out.size == 3
        assert array.array("f", bytes(out.data)).tolist() == [1.5, 2.5, -0.5]

    def test_arrayhandle_roundtrip(self):
        from msgspec.data import Array, ArrayHandle, Int64

        h = ArrayHandle(array.array("q", [1, -2, 2**62]).tobytes(), "int64", 3)
        out = msgspec.json.decode(
            msgspec.json.encode(h, type=Array[Int64]), type=Array[Int64]
        )
        assert array.array("q", bytes(out.data)).tolist() == [1, -2, 2**62]

    def test_struct_field(self):
        from msgspec.data import Array, ArrayHandle, Float32

        class Doc(Struct):
            name: str
            vals: Array[Float32]

        d = Doc(name="x", vals=memoryview(array.array("f", [1.0, 2.0])))
        out = msgspec.msgpack.decode(
            msgspec.msgpack.encode(d, type=Doc), type=Doc
        )
        assert isinstance(out.vals, ArrayHandle)
        assert array.array("f", bytes(out.vals.data)).tolist() == [1.0, 2.0]

    def test_any_decode_follows_wire_flag(self):
        from msgspec.data import Array, ArrayHandle, UInt8

        arr = msgspec.msgpack.encode(b"\x01\x02", type=Array[UInt8])
        assert isinstance(msgspec.msgpack.decode(arr), ArrayHandle)

    def test_tensor_target_still_gives_tensorhandle(self):
        # Decoding array-flagged bytes against a Tensor target follows the
        # target (a TensorHandle), not the wire flag.
        from msgspec.data import Array, Tensor, UInt8

        arr = msgspec.msgpack.encode(b"\x01\x02", type=Array[UInt8])
        out = msgspec.msgpack.decode(arr, type=Tensor[1, UInt8])
        assert isinstance(out, TensorHandle)

    def test_dtype_mismatch_errors(self):
        from msgspec.data import Array, Float32

        with pytest.raises(TypeError):
            msgspec.json.encode(b"\x00\x01", type=Array[Float32])  # not a multiple of 4

    def test_inspect(self):
        from msgspec.data import Array, Float32
        from msgspec.inspect import ArrayType, type_info

        assert type_info(Array[Float32, 8]) == ArrayType(dtype="float32", size=8)
        assert type_info(Array) == ArrayType(dtype=None, size=None)


class TestJsonSchema:
    """Tensor/Array types render as their self-describing JSON object form."""

    def test_tensor_schema(self):
        from msgspec.data import Float32, Tensor

        class Model(msgspec.Struct):
            weights: Tensor[2, Float32]

        schema = msgspec.json.schema(Model)
        prop = schema["$defs"]["Model"]["properties"]["weights"]
        assert prop["type"] == "object"
        assert prop["properties"]["type"] == {"enum": ["tensor"]}
        assert prop["properties"]["dtype"] == {"enum": ["float32"]}
        assert prop["properties"]["shape"] == {
            "type": "array",
            "prefixItems": [{"type": "integer"}, {"type": "integer"}],
            "minItems": 2,
            "maxItems": 2,
        }
        assert prop["properties"]["data"] == {
            "type": "string",
            "contentEncoding": "base64",
        }
        assert prop["required"] == ["type", "shape", "dtype", "data"]

    def test_tensor_schema_untyped(self):
        from msgspec.data import Tensor

        schema = msgspec.json.schema(Tensor)
        assert schema["properties"]["dtype"] == {"type": "string"}
        assert "anyOf" in schema["properties"]["shape"]

    def test_tensor_schema_sized(self):
        from msgspec.data import Tensor, UInt8

        schema = msgspec.json.schema(Tensor[(2, 3), UInt8])
        assert schema["properties"]["shape"]["prefixItems"] == [
            {"enum": [2]},
            {"enum": [3]},
        ]

    def test_array_schema(self):
        from msgspec.data import Array, UInt8

        schema = msgspec.json.schema(Array[UInt8, 4])
        assert schema["properties"]["dtype"] == {"enum": ["uint8"]}
        assert schema["properties"]["shape"]["prefixItems"] == [{"enum": [4]}]

    def test_array_schema_unsized(self):
        from msgspec.data import Array, UInt8

        schema = msgspec.json.schema(Array[UInt8])
        assert "anyOf" in schema["properties"]["shape"]
