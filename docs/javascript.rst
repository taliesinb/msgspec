JavaScript
==========

`msgspec.javascript.codec` generates a self-contained `JavaScript`_ codec from
msgspec-compatible :doc:`types <supported-types>` - a plain-JS counterpart to
the :doc:`TypeScript codec <typescript>`. It is useful for sharing a schema
between a Python backend and a JavaScript frontend.

The output is a dependency-free ES module that inlines its own tight MessagePack
reader/writer and exports a ``msgpack`` and/or a ``json`` namespace (``{ encode,
decode }`` each). Both are the **client side of a type-directed ``msgspec``
encoder** (see :doc:`type-directed-encoding`): encode on the Python side with
the same ``type=`` and the two are byte/format-compatible.

- 64-bit / generic integers (``Int``/``UInt``/``Int64``/``UInt64``) round-trip as
  native ``bigint`` - no safe-integer narrowing; hex strings in JSON when large.
- ``bytes`` are ``Uint8Array`` (base64 in JSON); ``Float32`` is a 5-byte
  MessagePack float32.
- No runtime validation - it is a structural transform.


Usage
-----

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


    print(msgspec.javascript.codec(Line))


produces (abbreviated) a module that inlines a ``Writer``/``Reader`` plus:

.. code-block:: javascript

    export function encodePoint(w, v) {
      w.mapHeader(2);
      w.str("x");
      w.int(v["x"]);
      w.str("y");
      w.int(v["y"]);
    }

    export function decodePoint(r) {
      const n = r.mapHeader();
      const o = {};
      for (let i = 0; i < n; i++) {
        switch (r.str()) {
          case "x": o["x"] = r.int(); break;
          case "y": o["y"] = r.int(); break;
          default: r.skip();
        }
      }
      return o;
    }

    export const msgpack = {
      encode(value) { const w = new Writer(); encodeLine(w, value); return w.bytes(); },
      decode(bytes) { const r = new Reader(bytes); return decodeLine(r); },
    };

    export const json = {
      encode(value) { return JSON.stringify(encodeJsonLine(value)); },
      decode(text) { const o = JSON.parse(text); return decodeJsonLine(o); },
    };

Use it as:

.. code-block:: javascript

    import { msgpack, json } from "./codec.js";

    const bytes = msgpack.encode(value);   // -> Uint8Array
    const value2 = msgpack.decode(bytes);
    const text = json.encode(value);       // -> string
    const value3 = json.decode(text);

Pass ``msgpack=False`` or ``json=False`` to omit a namespace.


Tagged unions and abstract structs
----------------------------------

As with the TypeScript codec, tagged-union structs emit their discriminant tag
when **encoding** and are dispatched on it when **decoding**. An
:ref:`abstract struct <abstract>` behaves as the union of its concrete
descendants. The decoder writes the tag field back onto the returned object so a
value can be decoded and re-encoded unchanged.

.. code-block:: python

    from typing import Union
    from msgspec import Struct


    class Cat(Struct, tag="cat"):
        name: str


    class Dog(Struct, tag="dog"):
        legs: int


    print(msgspec.javascript.codec(Union[Cat, Dog]))


Type mapping
------------

===================================  ==============================
Python                               JavaScript
===================================  ==============================
``bool``                             ``boolean``
``int``, ``float``                   ``number``
``Int``/``UInt``/``Int64``/``UInt64``  ``bigint``
``str``                              ``string``
``bytes``                            ``Uint8Array``
``None``                             ``null``
``list[T]``, ``tuple[T, ...]``       ``Array``
``set[T]``                           ``Set``
``dict[K, V]``                       object
`Struct`                             object (``{ ... }``)
`enum.Enum`                          its underlying value
===================================  ==============================

Using the bundle from a JavaScript project
------------------------------------------

The generated codec is self-contained, but the underlying runtime is also
shipped as a standalone package so a JavaScript project can depend on it
directly. `msgspec.javascript.bundle_path` returns the on-disk directory of that
package (``package.json`` + ``src/msgpack.mjs``):

.. code-block:: python

    >>> import msgspec
    >>> msgspec.javascript.bundle_path()
    PosixPath('.../msgspec/javascript_bundle')

Point your JS build/bundle step at it - for example as a local npm dependency
(``"@msgspec/msgpack": "file:<bundle_path>"``) or by copying
``bundle_path()/"src"/"msgpack.mjs"`` into your sources. In a source checkout or
editable install this resolves to the repository's top-level ``javascript/``
directory; in an installed wheel, to the copy bundled inside the package.


.. note::

    The codec is a *structural* transform - no runtime validation is performed.
    ``datetime``/``UUID``/``Decimal`` types are not yet supported and raise
    ``NotImplementedError``; ``omit_defaults`` structs are encoded with all
    fields present. :doc:`Tensors <tensors>` **are** supported - they decode to a
    re-exported ``TensorHandle`` (typed array + ``dtype`` + ``shape``). The
    inlined runtime is authored and tested at
    ``javascript/src/msgpack.mjs`` in the repository, and is also published as a
    standalone package for direct use.


.. _JavaScript: https://developer.mozilla.org/en-US/docs/Web/JavaScript
.. _@msgpack/msgpack: https://github.com/msgpack/msgpack-javascript
