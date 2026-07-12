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
render as ordinary object ``class`` definitions - their array-on-the-wire form
is handled by the codec (see below), not the schema.


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

``wrapTensor``/``unwrapTensor`` are whatever bridges your chosen JS array type
to a ``TensorHandle``. The example below hosts tensors as `@stdlib/ndarray`_
values - a numpy-like array that is a *view* over a typed array, so the
handoff is nearly zero-copy. It is only a reference (msgspec has no dependency
on it); adapt it to your array library of choice, e.g. `TensorFlow.js`_ tensors
for GPU acceleration.

Save it alongside the generated codec (or import these two functions into it)
and pass their names: ``codec(..., tensor_encoder="ndToTensor",
tensor_decoder="tensorToNd")``.

.. code-block:: typescript

    // Reference adapter between @stdlib/ndarray and the generated TensorHandle.
    import { array } from "@stdlib/ndarray";
    import type { ndarray } from "@stdlib/ndarray";
    import { TensorHandle } from "./codec.ts"; // the generated module

    // msgspec dtype -> { typed-array view, @stdlib dtype name }.
    const DTYPES: Record<string, { View: any; stdlib: string }> = {
      uint8:   { View: Uint8Array,     stdlib: "uint8" },
      uint16:  { View: Uint16Array,    stdlib: "uint16" },
      uint32:  { View: Uint32Array,    stdlib: "uint32" },
      uint64:  { View: BigUint64Array, stdlib: "uint64" },
      int8:    { View: Int8Array,      stdlib: "int8" },
      int16:   { View: Int16Array,     stdlib: "int16" },
      int32:   { View: Int32Array,     stdlib: "int32" },
      int64:   { View: BigInt64Array,  stdlib: "int64" },
      float32: { View: Float32Array,   stdlib: "float32" },
      float64: { View: Float64Array,   stdlib: "float64" },
      bool:    { View: Uint8Array,     stdlib: "uint8" },
    };
    const STDLIB_TO_MSGSPEC: Record<string, string> = {
      uint8: "uint8", uint8c: "uint8", uint16: "uint16", uint32: "uint32",
      uint64: "uint64", int8: "int8", int16: "int16", int32: "int32",
      int64: "int64", float32: "float32", float64: "float64", bool: "bool",
    };

    // decode: TensorHandle -> ndarray (zero-copy view over the decoded bytes)
    export function tensorToNd(h: TensorHandle): ndarray {
      const info = DTYPES[h.dtype ?? "uint8"];
      if (!info) throw new Error(`unsupported tensor dtype: ${h.dtype}`);
      const buf = new info.View(
        h.data.buffer,
        h.data.byteOffset,
        h.data.byteLength / info.View.BYTES_PER_ELEMENT,
      );
      return array(buf, { shape: h.shape ?? [buf.length], dtype: info.stdlib });
    }

    // encode: ndarray -> TensorHandle
    export function ndToTensor(x: ndarray): TensorHandle {
      const dtype = STDLIB_TO_MSGSPEC[x.dtype];
      if (!dtype) throw new Error(`unsupported ndarray dtype: ${x.dtype}`);
      const flat = toRowMajor(x); // msgpack packs a C-contiguous buffer
      const bytes = new Uint8Array(flat.buffer, flat.byteOffset, flat.byteLength);
      return new TensorHandle(bytes, dtype, x.shape.slice());
    }

    // A C-contiguous typed array of x's elements: a view when x already owns a
    // row-major buffer, else a copy (use @stdlib/ndarray/base/assign in practice).
    function toRowMajor(x: ndarray): ArrayBufferView {
      if (x.offset === 0 && x.order === "row-major" && isContiguous(x)) {
        return x.data;
      }
      const y = array({ shape: x.shape, dtype: x.dtype, order: "row-major" });
      const ndim = x.shape.length, idx = new Array(ndim).fill(0);
      for (let i = 0; i < x.length; i++) {
        y.iset(i, x.get(...idx));
        for (let d = ndim - 1; d >= 0; d--) {
          if (++idx[d] < x.shape[d]) break;
          idx[d] = 0;
        }
      }
      return y.data;
    }

    function isContiguous(x: ndarray): boolean {
      let s = 1;
      for (let d = x.shape.length - 1; d >= 0; d--) {
        if (x.strides[d] !== s) return false;
        s *= x.shape[d];
      }
      return true;
    }

Notes: decoding is genuinely zero-copy (the ``TensorHandle`` bytes are
reinterpreted and wrapped); encoding is too whenever the ndarray is already
contiguous. ``int64``/``uint64`` map to ``BigInt64Array``/``BigUint64Array``
(matching the codec's ``bigint`` handling); ``bool`` is treated as raw
``uint8`` bytes here; and the raw buffer is native-endian, so only a big-endian
peer would need a byte-swap.


.. _TypeScript: https://www.typescriptlang.org/
.. _@msgpack/msgpack: https://github.com/msgpack/msgpack-javascript
.. _@stdlib/ndarray: https://github.com/stdlib-js/ndarray
.. _TensorFlow.js: https://github.com/tensorflow/tfjs
