"""Tests for the ``abstract=`` Struct kwarg and abstract-struct-as-union behavior."""

import pytest

import msgspec
from msgspec import Struct, defstruct, inspect as mi

# --- Config plumbing -------------------------------------------------------


def test_abstract_config_defaults_false():
    class Plain(Struct):
        x: int

    cfg = Plain.__struct_config__
    assert cfg.abstract is False
    assert cfg.abstract_parents is None
    assert cfg.concrete_children is None


def test_abstract_bool_marks_only_declaring_class():
    class Animal(Struct, tag_field="kind", abstract=True):
        pass

    class Cat(Animal, tag="cat"):
        name: str

    class Dog(Animal, tag="dog"):
        sound: str

    assert Animal.__struct_config__.abstract is True
    # A plain `abstract=True` bool is NOT inherited: children flip to concrete.
    assert Cat.__struct_config__.abstract is False
    assert Dog.__struct_config__.abstract is False

    assert Animal.__struct_config__.concrete_children == [Cat, Dog]
    assert Cat.__struct_config__.abstract_parents == [Animal]
    assert Dog.__struct_config__.abstract_parents == [Animal]
    # concrete leaves don't track children; abstract has no abstract parents
    assert Cat.__struct_config__.concrete_children is None
    assert Animal.__struct_config__.abstract_parents is None


def test_abstract_explicit_false_on_child():
    class Base(Struct, abstract=True, tag_field="t"):
        pass

    class Mid(Base, abstract=True):
        pass

    class Leaf(Mid, tag="leaf"):
        x: int

    assert Mid.__struct_config__.abstract is True
    assert Leaf.__struct_config__.abstract is False
    # Leaf registered with every abstract ancestor
    assert Base.__struct_config__.concrete_children == [Leaf]
    assert Mid.__struct_config__.concrete_children == [Leaf]
    assert Leaf.__struct_config__.abstract_parents == [Base, Mid]


def test_abstract_callable_inherited_and_reevaluated():
    class Base(Struct, tag_field="t", abstract=lambda n: n.startswith("Abstract")):
        pass

    class AbstractPet(Base):
        pass

    class Cat(AbstractPet, tag="cat"):
        name: str

    assert Base.__struct_config__.abstract is False
    assert AbstractPet.__struct_config__.abstract is True
    assert Cat.__struct_config__.abstract is False
    assert AbstractPet.__struct_config__.concrete_children == [Cat]
    assert Cat.__struct_config__.abstract_parents == [AbstractPet]


def test_abstract_callable_must_return_bool():
    with pytest.raises(TypeError, match="callable must return a bool"):

        class Bad(Struct, abstract=lambda n: "yes"):
            pass


def test_abstract_bad_type():
    with pytest.raises(TypeError, match="must be a bool or a callable"):

        class Bad(Struct, abstract=123):
            pass


def test_abstract_defstruct():
    Animal = defstruct("Animal", [], tag_field="kind", abstract=True)
    Cat = defstruct("Cat", [("name", str)], bases=(Animal,), tag="cat")
    assert Animal.__struct_config__.abstract is True
    assert Cat.__struct_config__.abstract is False
    assert Animal.__struct_config__.concrete_children == [Cat]


# --- Instantiation ---------------------------------------------------------


def test_abstract_cannot_instantiate():
    class Animal(Struct, tag_field="kind", abstract=True):
        pass

    class Cat(Animal, tag="cat"):
        name: str

    with pytest.raises(TypeError, match="Can't instantiate abstract Struct type"):
        Animal()

    # concrete subclass instantiates fine
    assert Cat(name="x").name == "x"


# --- Decode as the union of concretes --------------------------------------


@pytest.fixture
def zoo():
    class Animal(Struct, tag_field="kind", abstract=True):
        pass

    class Cat(Animal, tag="cat"):
        name: str

    class Dog(Animal, tag="dog"):
        sound: str

    return Animal, Cat, Dog


@pytest.mark.parametrize("proto", [msgspec.json, msgspec.msgpack])
def test_abstract_decode_roundtrip(zoo, proto):
    Animal, Cat, Dog = zoo
    for obj in [Cat(name="Whiskers"), Dog(sound="woof")]:
        out = proto.decode(proto.encode(obj), type=Animal)
        assert out == obj
        assert type(out) is type(obj)


def test_abstract_decode_bad_tag(zoo):
    Animal, Cat, Dog = zoo
    with pytest.raises(msgspec.ValidationError):
        msgspec.json.decode(b'{"kind":"fish"}', type=Animal)


def test_abstract_nested_field(zoo):
    Animal, Cat, Dog = zoo

    class Owner(Struct):
        pet: Animal
        others: list[Animal] = []

    src = Owner(pet=Cat(name="Leo"), others=[Dog(sound="r")])
    out = msgspec.json.decode(msgspec.json.encode(src), type=Owner)
    assert out == src
    assert type(out.pet) is Cat
    assert type(out.others[0]) is Dog


def test_abstract_decoder_object(zoo):
    Animal, Cat, Dog = zoo
    dec = msgspec.json.Decoder(Animal)
    assert dec.decode(msgspec.json.encode(Cat(name="x"))) == Cat(name="x")


def test_abstract_convert(zoo):
    Animal, Cat, Dog = zoo
    out = msgspec.convert({"kind": "cat", "name": "x"}, type=Animal)
    assert out == Cat(name="x")


def test_abstract_single_concrete_collapses():
    class Base(Struct, abstract=True):
        pass

    class Only(Base):
        x: int

    # A single concrete descendant needs no tag - it decodes as that struct.
    assert msgspec.json.decode(b'{"x":5}', type=Base) == Only(x=5)


def test_abstract_zero_concretes_errors():
    class Empty(Struct, abstract=True):
        pass

    with pytest.raises(TypeError, match="no concrete subclasses"):
        msgspec.json.decode(b"{}", type=Empty)


def test_abstract_untagged_multiple_errors():
    class Base(Struct, abstract=True):
        pass

    class A(Base):
        a: int

    class B(Base):
        b: int

    with pytest.raises(TypeError, match="must be tagged"):
        msgspec.json.decode(b"{}", type=Base)


# --- inspect ---------------------------------------------------------------


def test_inspect_abstract_struct_type(zoo):
    Animal, Cat, Dog = zoo
    ti = mi.type_info(Animal)
    assert isinstance(ti, mi.AbstractStructType)
    assert ti.abstract is True
    assert ti.concrete_classes == (Cat, Dog)
    union = ti.concrete_union_type
    assert isinstance(union, mi.UnionType)
    assert {t.cls for t in union.types} == {Cat, Dog}
    # cached on first access
    assert ti.concrete_union_type is union


def test_inspect_concrete_is_plain_struct_type(zoo):
    Animal, Cat, Dog = zoo
    ti = mi.type_info(Cat)
    assert type(ti) is mi.StructType
    assert ti.abstract is False


# --- JSON schema -----------------------------------------------------------


def test_abstract_json_schema(zoo):
    Animal, Cat, Dog = zoo
    schema = msgspec.json.schema(Animal)
    assert "anyOf" in schema
    assert schema["discriminator"]["propertyName"] == "kind"
    assert set(schema["discriminator"]["mapping"]) == {"cat", "dog"}
    assert set(schema["$defs"]) == {"Cat", "Dog"}


# --- TypeScript ------------------------------------------------------------


def test_abstract_typescript_schema(zoo):
    Animal, Cat, Dog = zoo
    out = msgspec.typescript.schema(Animal)
    # The abstract struct keeps its name as a union alias.
    assert "export type Animal = Cat | Dog;" in out
    assert "export class Cat" in out
    assert "export class Dog" in out
    # ...but is not emitted as a class.
    assert "export class Animal" not in out


def test_abstract_typescript_named_field(zoo):
    Animal, Cat, Dog = zoo

    class Zoo(Struct):
        star: Animal

    out = msgspec.typescript.schema(Zoo)
    # A field of an abstract type references it by name.
    assert "star: Animal;" in out
    assert "export type Animal = Cat | Dog;" in out


def test_abstract_typescript_codec_dispatch(zoo):
    Animal, Cat, Dog = zoo
    out = msgspec.typescript.codec(Animal)
    assert "export type Animal = Cat | Dog;" in out
    assert "encode(value: Animal): Uint8Array" in out
    assert "export const msgpack = {" in out
    assert "case \"cat\": return decodeCat(r);" in out
    assert "case \"dog\": return decodeDog(r);" in out
