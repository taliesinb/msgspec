TypeScript
==========

``msgspec`` can generate `TypeScript`_ definitions - and optionally
MessagePack encoders/decoders - from msgspec-compatible :doc:`types
<supported-types>`. This is useful for sharing a schema between a Python
backend and a TypeScript frontend.

- `msgspec.typescript.schema`: generate TypeScript *type definitions* for a
  single type.
- `msgspec.typescript.schema_components`: like ``schema``, but for multiple
  types, returning inline references plus a mapping of named definitions.
- `msgspec.typescript.codec`: generate type definitions **plus** MessagePack
  ``encode``/``decode`` functions (assuming the `@msgpack/msgpack`_ library is
  available).


Type definitions
----------------

`msgspec.typescript.schema` turns a type into TypeScript source. `Struct`
types (and other "nameable" types like enums, dataclasses, typed-dicts, and
named-tuples) become top-level ``class``/``enum`` definitions; everything else
is rendered inline.

.. code-block:: python

    import msgspec
    from msgspec import Struct


    class Point(Struct):
        x: int
        y: int


    class Line(Struct):
        start: Point
        end: Point
        label: str | None = None


    print(msgspec.typescript.schema(Line))


.. code-block:: typescript

    export class Line {
      start: Point;
      end: Point;
      label?: string | null;
    }

    export class Point {
      x: number;
      y: number;
    }


Type mapping
~~~~~~~~~~~~

The mapping from Python types to TypeScript is necessarily lossy (TypeScript
can't express every constraint), but covers the common cases:

===================================  ==============================
Python                               TypeScript
===================================  ==============================
``bool``                             ``boolean``
``int``, ``float``                   ``number``
``str``                              ``string``
``bytes``, ``datetime``, ``UUID``    ``string``
``None``                             ``null``
``list[T]``, ``tuple[T, ...]``       ``Array<T>``
``set[T]``                           ``Set<T>``
``tuple[A, B]``                      ``[A, B]``
``dict[K, V]``                       ``Record<K, V>``
``A | B``                            ``A | B``
``Literal["a", "b"]``                ``"a" | "b"``
`Struct`                             ``class`` definition
`enum.Enum`                          ``enum`` definition
===================================  ==============================

Type aliases (a `typing.NewType` or a :pep:`695` ``type X = ...``) are
preserved as ``type`` aliases when present. Array-like structs and named-tuples
render as positional tuple types.


Encoders and decoders
---------------------

`msgspec.typescript.codec` emits everything ``schema`` does, plus a matching
pair of MessagePack functions per struct and a top-level ``encode(value)`` /
``decode(bytes)``. The generated code imports ``encode``/``decode`` from
`@msgpack/msgpack`_.

The transformation is *structural* - no runtime validation is performed.
Tagged-union structs emit their discriminant tag when **encoding**, and the
union is dispatched on that tag when **decoding**:

.. code-block:: python

    from typing import Union
    from msgspec import Struct


    class Cat(Struct, tag="cat"):
        name: str


    class Dog(Struct, tag="dog"):
        name: str


    print(msgspec.typescript.codec(Union[Cat, Dog]))


produces (abbreviated) encoders that add the tag:

.. code-block:: typescript

    export function encodeCat(value: Cat): unknown {
      return {
        type: "cat",
        name: value["name"],
      };
    }

and a top-level codec that dispatches on it when decoding (a non-nameable
top-level type like a union gets a generated ``Root`` alias):

.. code-block:: typescript

    export type Root = Cat | Dog;

    export function decode(bytes: Uint8Array): Root {
      return ((v: any) => {
        switch (v["type"]) {
          case "cat": return decodeCat(v);
          case "dog": return decodeDog(v);
        }
        throw new Error("unexpected tag for type union");
      })(_mpDecode(bytes)) as Root;
    }

The generated ``encode`` produces byte-for-byte identical MessagePack to
`msgspec.msgpack.encode`, so the two ends interoperate directly.


Tensors
~~~~~~~

To encode/decode :doc:`tensors <tensors>`, pass the names of two TypeScript
functions (assumed to be in scope) via ``tensor_encoder`` and
``tensor_decoder``:

- ``tensor_encoder(value) -> TensorHandle`` - called at tensor encode
  positions.
- ``tensor_decoder(handle: TensorHandle) -> value`` - called at tensor decode
  positions.

When any tensor type is present, the generated module also emits a
``TensorHandle`` class (``{ data: Uint8Array, dtype, shape }``) and registers an
`@msgpack/msgpack`_ ``ExtensionCodec`` for the reserved extension type ``84``,
byte-compatible with ``msgspec``.

.. code-block:: python

    from msgspec import Struct
    from msgspec.data import Tensor, Float32


    class Layer(Struct):
        weights: Tensor[1, Float32]


    src = msgspec.typescript.codec(
        Layer, tensor_encoder="wrapTensor", tensor_decoder="unwrapTensor"
    )


.. _TypeScript: https://www.typescriptlang.org/
.. _@msgpack/msgpack: https://github.com/msgpack/msgpack-javascript
