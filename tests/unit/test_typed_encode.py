"""Tests for type-directed encoding (`encode(..., type=...)`).

Currently covers the JSON hex-string encoding of 64-bit / generic integer
markers (Int64/UInt64/Int/UInt), so they round-trip losslessly through the JS
safe-integer range.
"""

from typing import Union

import pytest

import msgspec
from msgspec import Struct
from msgspec.data import Int, Int64, Scalar, UInt, UInt64


def enc(obj, type):
    return msgspec.json.encode(obj, type=type)


def dec(buf, type):
    return msgspec.json.decode(buf, type=type)


# --- value-sensitive hex encoding ------------------------------------------


def test_small_ints_stay_plain_numbers():
    assert enc(5, type=Int64) == b"5"
    assert enc(-5, type=Int64) == b"-5"
    assert enc(2**53 - 1, type=Int64) == str(2**53 - 1).encode()
    assert enc(0, type=UInt64) == b"0"


def test_large_signed_ints_hex_with_sign():
    assert enc(2**60, type=Int64) == b'"+0x1000000000000000"'
    assert enc(-(2**62), type=Int64) == b'"-0x4000000000000000"'
    assert enc(2**60, type=Int) == b'"+0x1000000000000000"'


def test_large_unsigned_ints_hex_no_sign():
    assert enc(2**64 - 1, type=UInt64) == b'"0xffffffffffffffff"'
    assert enc(2**70, type=UInt) == b'"0x400000000000000000"'


def test_arbitrary_precision_generic_int():
    v = 10**25
    assert enc(v, type=Int) == b'"+0x84595161401484a000000"'
    assert dec(enc(v, type=Int), type=Int) == v


def test_plain_int_never_hex():
    # A plain `int` stays a JSON number even when large (JS-precision is the
    # caller's problem for plain int).
    assert enc(2**60, type=int) == str(2**60).encode()
    # And untyped encoding is unchanged.
    assert msgspec.json.encode(2**60) == str(2**60).encode()


# --- round-trips -----------------------------------------------------------


@pytest.mark.parametrize(
    "T, v",
    [
        (Int64, 2**60),
        (Int64, -(2**62)),
        (Int64, 5),
        (UInt64, 2**64 - 1),
        (Int, 10**25),
        (UInt, 2**70),
    ],
)
def test_roundtrip(T, v):
    assert dec(enc(v, type=T), type=T) == v


def test_struct_field_roundtrip():
    class Rec(Struct):
        id: Int64
        count: int  # plain int, stays a number

    r = Rec(id=2**60, count=2**60)
    buf = enc(r, type=Rec)
    assert b'"id":"+0x1000000000000000"' in buf
    assert b'"count":%d' % (2**60) in buf
    assert dec(buf, type=Rec) == r


def test_container_roundtrip():
    assert dec(enc([2**60, 3], type=list[Int64]), type=list[Int64]) == [2**60, 3]
    assert dec(enc({"a": 2**60}, type=dict[str, Int64]), type=dict[str, Int64]) == {
        "a": 2**60
    }


# --- Scalar (Int | Float | Bool): always-hex ints for JSON safety ----------


def test_scalar_int_always_hex():
    # Inside a Scalar union an int must be distinguishable from a float, so even
    # a small int is hex-encoded.
    assert enc(5, type=Scalar) == b'"+0x5"'
    assert enc(1.5, type=Scalar) == b"1.5"
    assert enc(True, type=Scalar) == b"true"


@pytest.mark.parametrize("v", [5, 1.5, True, 2**60, -(2**70)])
def test_scalar_roundtrip_preserves_type(v):
    out = dec(enc(v, type=Scalar), type=Scalar)
    assert out == v and type(out) is type(v)


# --- union disambiguation --------------------------------------------------


def test_int_str_union_disambiguation():
    U = Union[Int64, str]
    assert dec(b'"+0x1000000000000000"', type=U) == 2**60  # hex-shaped -> int
    assert dec(b'"hello"', type=U) == "hello"  # not hex-shaped -> str
    assert dec(b"5", type=U) == 5  # a number -> int


# --- untyped / msgpack unaffected ------------------------------------------


def test_msgpack_int_type_is_byte_identical():
    # msgpack ints are already exact, so `type=` doesn't change int bytes.
    class Rec(Struct):
        id: Int64

    r = Rec(id=2**60)
    assert msgspec.msgpack.encode(r, type=Rec) == msgspec.msgpack.encode(r)


# --- msgpack Float32 narrowing ---------------------------------------------


def test_msgpack_float32_narrows():
    from msgspec.data import Float, Float32, Float64

    # Float32 -> 5-byte float32 (0xca); Float64/Float/plain -> 9-byte (0xcb).
    assert msgspec.msgpack.encode(1.5, type=Float32) == bytes.fromhex("ca3fc00000")
    assert msgspec.msgpack.encode(1.5, type=Float64)[0] == 0xCB
    assert msgspec.msgpack.encode(1.5, type=Float)[0] == 0xCB
    assert msgspec.msgpack.encode(1.5)[0] == 0xCB
    # round-trips at float32 precision
    v = msgspec.msgpack.decode(
        msgspec.msgpack.encode(0.1, type=Float32), type=Float32
    )
    assert abs(v - 0.1) < 1e-6 and v != 0.1


def test_msgpack_float32_in_struct_and_list():
    from msgspec.data import Float32

    class S(Struct):
        a: Float32
        b: float

    buf = msgspec.msgpack.encode(S(a=1.5, b=1.5), type=S)
    # a is float32 (5-byte), b is float64 (9-byte)
    assert buf.hex() == "82a161ca3fc00000a162cb3ff8000000000000"
    assert msgspec.msgpack.encode([1.5], type=list[Float32]) == bytes.fromhex(
        "91ca3fc00000"
    )


# --- Encoder class type= ---------------------------------------------------


def test_encoder_type_matches_module_level():
    from msgspec.data import Float32

    class Rec(Struct):
        id: Int64
        ratio: Float32

    r = Rec(id=2**60, ratio=1.5)
    assert msgspec.json.Encoder(type=Rec).encode(r) == msgspec.json.encode(r, type=Rec)
    assert msgspec.msgpack.Encoder(type=Rec).encode(r) == msgspec.msgpack.encode(
        r, type=Rec
    )
    # A plain Encoder is value-directed.
    assert msgspec.json.Encoder().encode(r) == msgspec.json.encode(r)


# --- elide_implied_tag ------------------------------------------------------


class _Animal(Struct, tag_field="kind", abstract=True):
    pass


class _Cat(_Animal):
    name: str = "felix"


class _Dog(_Animal):
    legs: int = 4


class _TaggedArr(Struct, tag="ta", array_like=True):
    x: int = 1


class TestElideImpliedTag:
    def test_default_keeps_tag(self):
        for encode in (msgspec.json.encode, msgspec.msgpack.encode):
            data = encode(_Cat(), type=_Cat)
            assert b"kind" in data

    def test_monomorphic_target_elides(self):
        assert (
            msgspec.json.encode(_Cat(), type=_Cat, elide_implied_tag=True)
            == b'{"name":"felix"}'
        )
        assert (
            msgspec.msgpack.encode(_Cat(), type=_Cat, elide_implied_tag=True)
            == msgspec.msgpack.encode({"name": "felix"})
        )

    def test_union_target_keeps_tag(self):
        for encode in (msgspec.json.encode, msgspec.msgpack.encode):
            data = encode([_Cat()], type=list[_Animal], elide_implied_tag=True)
            assert b"kind" in data

    def test_nested_fields_elide_by_position(self):
        class Wrap(Struct):
            mono: _Cat
            poly: _Animal

        data = msgspec.json.encode(
            Wrap(mono=_Cat(), poly=_Dog()), type=Wrap, elide_implied_tag=True
        )
        assert data == b'{"mono":{"name":"felix"},"poly":{"kind":"_Dog","legs":4}}'

    def test_array_like_never_elides(self):
        for encode in (msgspec.json.encode, msgspec.msgpack.encode):
            data = encode(_TaggedArr(), type=_TaggedArr, elide_implied_tag=True)
            assert data == encode(_TaggedArr(), type=_TaggedArr)

    def test_roundtrip(self):
        v = [_Cat(name="tom"), _Cat()]
        for mod in (msgspec.json, msgspec.msgpack):
            data = mod.encode(v, type=list[_Cat], elide_implied_tag=True)
            assert mod.decode(data, type=list[_Cat]) == v

    def test_encoder_kwarg(self):
        enc = msgspec.json.Encoder(type=_Cat, elide_implied_tag=True)
        assert enc.encode(_Cat()) == b'{"name":"felix"}'
        enc = msgspec.msgpack.Encoder(type=_Cat, elide_implied_tag=True)
        assert enc.encode(_Cat()) == msgspec.msgpack.encode({"name": "felix"})

    def test_no_effect_without_type(self):
        assert (
            msgspec.json.encode(_Cat(), elide_implied_tag=True)
            == msgspec.json.encode(_Cat())
        )
