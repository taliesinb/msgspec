JavaScript
==========

`msgspec.javascript.codec` generates a self-contained `JavaScript`_ codec from
msgspec-compatible :doc:`types <supported-types>` - a plain-JS counterpart to
the :doc:`TypeScript codec <typescript>`. It is useful for sharing a schema
between a Python backend and a JavaScript frontend.
`msgspec.javascript.schema` generates matching JSDoc_-annotated type
definitions (see `JSDoc schemas`_ below).

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


JSDoc schemas
-------------

`msgspec.javascript.schema` (and `msgspec.javascript.schema_components`) is the
plain-JavaScript counterpart to `msgspec.typescript.schema`: instead of
TypeScript declarations it emits JSDoc_-annotated JavaScript - the standard way
plain JS is typed, fully understood by editors and by ``tsc --checkJs``
(the whole output type-checks under ``tsc --strict --checkJs``).

Struct-like types (structs, dataclasses, typed-dicts, named-tuples) become real
``class`` definitions: each field is declared with a ``@type`` annotation, and a
destructuring constructor documents field defaults as executable JavaScript
parameter defaults. Enums become frozen ``@enum`` objects; named aliases and
:ref:`abstract structs <abstract>` become ``@typedef`` declarations.

.. code-block:: python

    import enum
    import msgspec
    from msgspec import Struct


    class Fruit(enum.Enum):
        APPLE = "apple"
        BANANA = "banana"


    class Product(Struct, tag="product"):
        """A product in a catalog"""
        id: int
        name: str
        tags: set[str] = set()
        fruit: Fruit = Fruit.APPLE


    print(msgspec.javascript.schema(Product))

.. code-block:: javascript

    /** A product in a catalog */
    export class Product {
      /** @type {"product"} */
      type = "product";
      /** @type {number} */
      id;
      /** @type {string} */
      name;
      /** @type {Set<string>} */
      tags;
      /** @type {Fruit} */
      fruit;

      /**
       * @param {Object} fields
       * @param {number} fields.id
       * @param {string} fields.name
       * @param {Set<string>} [fields.tags]
       * @param {Fruit} [fields.fruit]
       */
      constructor({ id, name, tags = new Set(), fruit = Fruit.APPLE }) {
        this.id = id;
        this.name = name;
        this.tags = tags;
        this.fruit = fruit;
      }

      /**
       * @param {Product} fields
       * @returns {Product}
       */
      static mk(fields) {
        return Object.assign(Object.create(Product.prototype), fields);
      }
    }

    /** @enum {string} */
    export const Fruit = Object.freeze({
      APPLE: "apple",
      BANANA: "banana",
    });

The type expressions inside ``{...}`` use TypeScript type syntax (which JSDoc
accepts), so the two schema generators stay in lockstep. If the top-level type
is not itself nameable (e.g. ``list[Product]``), a ``@typedef ... Root`` names
it. Every class also gets a ``static mk(fields)`` factory - the JS analog of
Python's ``__new__`` - building an instance from a plain all-fields object
without going through the constructor signature.

By default the codec is independent of the schema: decoders return plain
objects, and the constructors are a convenience for user code (a constructed
instance has exactly the codec's expected shape, tag field included). But a
struct whose resolved constructor spec is non-``None`` **decodes into a real
class instance** (via ``mk``): structs with their own ``js_constructor``
class kwarg get this automatically, and passing ``constructors=<spec>`` or
``classes=True`` to `msgspec.javascript.codec` extends it to all structs
(the module then also exports the schema definitions). ``classes=False``
forces plain structural output.

The ``constructors`` keyword sets the default constructor signature as a
JS-style parameter-list spec - ``'{**}'`` (default, shown above),
``'(*, **)'`` (required fields positional, optionals in a trailing object:
``new Product(id, name, { tags })``), ``'(*)'`` (fully positional), explicit
signatures like ``'(name, *, **)'`` or ``'(name, ...tags)'``, or ``None``
(no classes; ``@typedef``/``@property`` documentation blocks instead). A
Struct can pick its own signature via the ``js_constructor`` class kwarg,
which always wins over ``constructors``. See :doc:`typescript` for details -
the same keyword with the same semantics exists on
`msgspec.typescript.schema`.


Tagged unions and abstract structs
----------------------------------

As with the TypeScript codec, tagged-union structs emit their discriminant tag
when **encoding** and are dispatched on it when **decoding**. An
:ref:`abstract struct <abstract>` behaves as the union of its concrete
descendants. The decoder writes the tag field back onto the returned object so a
value can be decoded and re-encoded unchanged.

Pass ``elide_implied_tag=True`` to omit the tag wherever the schema position is
a single concrete struct (keeping it in union positions) - matching a Python
encoder constructed with the same option (see
:doc:`type-directed-encoding`).

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
    re-exported ``TensorHandle`` (typed array + ``dtype`` + ``shape``), and flat
    ``Array[...]`` types decode to a plain typed array. The
    inlined runtime is authored and tested at
    ``javascript/src/msgpack.mjs`` in the repository, and is also published as a
    standalone package for direct use.


.. _JavaScript: https://developer.mozilla.org/en-US/docs/Web/JavaScript
.. _JSDoc: https://jsdoc.app/
.. _@msgpack/msgpack: https://github.com/msgpack/msgpack-javascript
