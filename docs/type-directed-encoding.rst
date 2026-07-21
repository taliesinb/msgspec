.. _type-directed-encoding:

Type-Directed Encoding
======================

By default ``msgspec`` encoding is *value-directed*: `msgspec.json.encode` and
`msgspec.msgpack.encode` look only at the runtime value. Passing a ``type=``
argument (also accepted by the `Encoder <msgspec.json.Encoder>` classes) opts
into a parallel *type-directed* path that walks the type alongside the value.
This lets the :doc:`msgspec.data <tensors>` integer/float markers select a
precise wire format, so values round-trip losslessly through JavaScript.

.. code-block:: python

    import msgspec
    from msgspec import Struct
    from msgspec.data import Int64, Float32

    class Reading(Struct):
        id: Int64          # a 64-bit integer
        value: Float32     # a 32-bit float

    r = Reading(id=2**60, value=1.5)

    # Value-directed (the default): id is a plain JSON number.
    >>> msgspec.json.encode(r)
    b'{"id":1152921504606846976,"value":1.5}'

    # Type-directed: id is hex-encoded so it survives JS number precision.
    >>> msgspec.json.encode(r, type=Reading)
    b'{"id":"+0x1000000000000000","value":1.5}'

When ``type`` is omitted the fast value-directed path is used unchanged.


What the format markers do
--------------------------

The markers only affect the wire form when encoding *with* ``type=`` (or
decoding, which is always type-directed):

- **JSON, 64-bit / generic integers** (``Int64``, ``UInt64``, ``Int``, ``UInt``):
  encoded as a plain JSON number when they fit the JS safe-integer range
  (``abs(v) <= 2**53 - 1``), otherwise as a **hex string**. Signed types carry an
  explicit sign (``"+0x…"`` / ``"-0x…"``); unsigned types do not (``"0x…"``), so
  the two are unambiguous. Fixed-width ``Int8``…``Int32`` / ``UInt8``…``UInt32``
  and plain ``int`` are always numbers.

- **``Scalar``** (the ``Int | Float | Bool`` union): its integer part is *always*
  hex-encoded in JSON, since a bare JSON number can't be told apart from a float.

- **MessagePack, ``Float32``**: encoded as a 5-byte float32 rather than the
  9-byte float64. Integers are already exact in MessagePack, so the type-directed
  bytes are otherwise identical to the value-directed ones.

- **``bytes``**: base64 in JSON (as always), raw bin in MessagePack.

Decoding is symmetric: a type-directed decoder (any ``Decoder(type=T)`` or
``decode(..., type=T)``) accepts the hex-string integers, and in an ``Int64 | str``
union it routes a hex-shaped string to the integer and everything else to
``str``.


JavaScript / TypeScript
-----------------------

The output of this encoder is exactly what :doc:`msgspec.javascript <javascript>`
and :doc:`msgspec.typescript <typescript>` generate codecs for: the markers map
to JavaScript ``bigint`` (decoded from hex when large), ``bytes`` to
``Uint8Array``, and ``Float32`` to a 5-byte MessagePack float. Encode on the
Python side with ``type=`` and the generated JS/TS ``json`` / ``msgpack``
decoders read it back losslessly.


Eliding implied tags (``elide_implied_tag``)
--------------------------------------------

A :ref:`tagged struct <struct-tagged-unions>` normally emits its tag field
every time it's encoded - the tag is a property of the class, not of the
position it appears in. But when encoding against a type, the target often
already *implies* the struct's identity: encoding a ``list[Foo]`` doesn't need
a tag on each element, while a ``list[FooBar]`` union does.

Passing ``elide_implied_tag=True`` to a type-directed encoder (``Encoder`` or
module-level ``encode``, JSON and MessagePack alike) omits the tag wherever the
target position is a single concrete struct type, and keeps it wherever the
position is a union (including an abstract struct)::

    class Shape(Struct, tag_field="kind", abstract=True): ...
    class Circle(Shape): r: float
    class Square(Shape): side: float

    msgspec.json.encode([Circle(r=1.0)], type=list[Circle], elide_implied_tag=True)
    # b'[{"r":1.0}]'                       <- tag implied by the target
    msgspec.json.encode([Circle(r=1.0)], type=list[Shape], elide_implied_tag=True)
    # b'[{"kind":"Circle","r":1.0}]'       <- union still needs the tag

Notes:

- **Decoding needs no flag**: for a concrete struct target the tag is already
  optional-but-validated, so elided messages decode with any type-directed
  decoder. Union targets still dispatch on the (present) tag.
- **Positional structs are exempt**: an ``array_like=True`` struct's tag
  occupies a fixed slot, so it is always emitted.
- **The trade-off**: with tags elided the bytes are no longer self-describing -
  a message encoded against ``list[Circle]`` can't later be decoded against
  ``list[Shape]`` or as ``Any``. Use it when both ends share the exact schema
  (e.g. with a generated :doc:`JavaScript <javascript>`/:doc:`TypeScript
  <typescript>` codec, which accept the same ``elide_implied_tag=True`` option
  in ``codec()`` and stay byte-compatible).
