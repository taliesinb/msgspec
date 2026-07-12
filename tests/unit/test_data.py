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
        out = Decoder(Tensor[3, UInt8]).decode(encode(h))
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
            weights: Tensor[3, Float32]
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
