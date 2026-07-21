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

`msgspec.typescript.codec` emits everything ``schema`` does, plus per-struct
encode/decode functions and namespaced ``msgpack`` and ``json`` codecs. By
default (``embed_msgpack=True``) it is **self-contained**: it inlines its own
tight MessagePack reader/writer, so there is no ``@msgpack/msgpack`` dependency.

.. code-block:: typescript

    import { msgpack, json } from "./codec.ts";

    const bytes: Uint8Array = msgpack.encode(value);
    const value2 = msgpack.decode(bytes);

    const text: string = json.encode(value);
    const value3 = json.decode(text);

The ``msgpack`` and ``json`` namespaces are the **client side of a type-directed
``msgspec`` encoder** (see :doc:`type-directed-encoding`): pass the same
``type`` to `msgspec.msgpack.encode` / `msgspec.json.encode` on the Python side
and the two are byte/format-compatible. In particular ``Int64``/``UInt64``/etc.
round-trip as native ``bigint`` (hex strings in JSON when large), ``bytes`` are
``Uint8Array`` (base64 in JSON), and ``Float32`` is a 5-byte MessagePack float.
The transformation is *structural* - no runtime validation is performed;
tagged-union structs emit their discriminant tag when encoding and are
dispatched on it when decoding. Pass ``elide_implied_tag=True`` to omit the tag
wherever the schema position is a single concrete struct (union positions keep
it), matching a Python encoder constructed with the same option.

Which namespaces to emit is controlled by the ``msgpack`` / ``json`` flags
(both ``True`` by default). Struct shapes are emitted as ``interface`` (the
codec works with plain objects), and the output type-checks under ``tsc
--strict``.

.. note::

    Passing ``embed_msgpack=False`` produces the older output that imports
    ``encode``/``decode`` from `@msgpack/msgpack`_ instead. That library can't
    serialize ``bigint`` natively, so the codec narrows to a ``number`` and
    *throws* above the JS safe-integer range unless ``force_int64=True`` enables
    its ``useBigInt64`` mode. Prefer the default embedded codec, which has
    neither limitation.


Tensors
~~~~~~~

The default embedded codec supports :doc:`tensors <tensors>` natively. A
``Tensor[...]`` field decodes to a ``TensorHandle`` - a class (re-exported by
the generated module) holding the packed ``array`` (a typed array chosen by
dtype), the ``dtype`` string, and the ``shape``, mirroring ``msgspec.data``'s
TensorHandle. Construct one to encode. Both the MessagePack ext form and the
JSON object form are byte-identical to ``msgspec``:

.. code-block:: typescript

    import { msgpack, TensorHandle } from "./codec.ts";

    const doc = msgpack.decode(bytes);
    doc.weights.array;   // Float32Array
    doc.weights.shape;   // [3]
    msgpack.encode({ weights: new TensorHandle(new Float32Array([1, 2, 3]), "float32", [3]) });

To host tensors as a third-party array type instead, use the
``@msgpack/msgpack`` path (``embed_msgpack=False``) and pass the names of two
TypeScript functions (assumed to be in scope) via ``tensor_encoder`` and
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
        Layer,
        tensor_encoder="wrapTensor",
        tensor_decoder="unwrapTensor",
        embed_msgpack=False,
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
