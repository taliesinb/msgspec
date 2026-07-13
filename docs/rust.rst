Rust
====

``msgspec`` can generate `Rust`_ (`serde`_) type definitions from
msgspec-compatible :doc:`types <supported-types>`. This is useful for sharing a
schema between a Python service and a Rust one: the generated types carry
``#[derive(Serialize, Deserialize)]`` and the serde attributes needed to match
msgspec's wire format, so `serde_json`_ (JSON) and `rmp-serde`_ (MessagePack)
interoperate directly with `msgspec.json` and `msgspec.msgpack`.

- `msgspec.rust.schema`: generate Rust definitions for a single type.
- `msgspec.rust.schema_components`: like ``schema``, but for multiple types,
  returning inline references plus a mapping of named definitions.

Unlike the :doc:`TypeScript <typescript>` / :doc:`JavaScript <javascript>`
codecs, there is no separate ``codec`` function: in Rust the serde derives *are*
the codec, so the generated type definitions are all you need.


Type definitions
----------------

`Struct` types become ``struct``\ s, tagged-union / :ref:`abstract <abstract>`
structs become internally-tagged ``enum``\ s, and everything else maps onto the
obvious Rust type.

.. code-block:: python

    import msgspec
    from msgspec import Struct
    from msgspec.data import UInt64, Int32


    class Point(Struct):
        x: Int32
        y: Int32
        ids: list[UInt64]
        label: str | None = None


    print(msgspec.rust.schema(Point))

.. code-block:: rust

    #[derive(Serialize, Deserialize)]
    pub struct Point {
        pub x: i32,
        pub y: i32,
        pub ids: Vec<u64>,
        pub label: Option<String>,
    }


Tagged unions and abstract structs
-----------------------------------

An :ref:`abstract <abstract>` struct (or an explicit tagged ``Union`` of
structs) becomes a records-style, *internally-tagged* serde ``enum`` whose
variants wrap the concrete structs. This mirrors msgspec's tagged-union wire
format exactly.

.. code-block:: python

    class Node(Struct, abstract=True, tag_field="op", tag=str.lower):
        pass

    class Add(Node):
        lhs: float
        rhs: float

    class Neg(Node):
        val: float

    print(msgspec.rust.schema(Node))

.. code-block:: rust

    #[derive(Serialize, Deserialize)]
    pub struct Add {
        pub lhs: f64,
        pub rhs: f64,
    }

    #[derive(Serialize, Deserialize)]
    pub struct Neg {
        pub val: f64,
    }

    #[derive(Serialize, Deserialize)]
    #[serde(tag = "op")]
    pub enum Node {
        #[serde(rename = "add")]
        Add(Add),
        #[serde(rename = "neg")]
        Neg(Neg),
    }


Type mapping
------------

========================================  ==================================
Python / ``msgspec.data``                 Rust
========================================  ==================================
``bool``                                  ``bool``
``int``, ``Int``, ``Int64``               ``i64``
``Int8`` / ``Int16`` / ``Int32``          ``i8`` / ``i16`` / ``i32``
``UInt`` / ``UInt64``                     ``u64``
``UInt8`` / ``UInt16`` / ``UInt32``       ``u8`` / ``u16`` / ``u32``
``float``, ``Float64``                    ``f64``
``Float32``                               ``f32``
``str``                                   ``String``
``bytes``                                 ``serde_bytes::ByteBuf``
``list[T]``, ``tuple[T, ...]``            ``Vec<T>``
``set[T]`` / ``frozenset[T]``             ``std::collections::HashSet<T>``
``tuple[A, B]``                           ``(A, B)``
``dict[K, V]``                            ``std::collections::HashMap<K, V>``
``T | None``                              ``Option<T>``
``Struct``                                ``struct``
tagged union / ``abstract`` struct        internally-tagged ``enum``
``enum.Enum`` (str-valued)                ``enum`` with ``#[serde(rename)]``
``TypeAliasType``                         ``pub type`` alias
========================================  ==================================

Reserved Rust keywords used as field names become raw identifiers
(``type`` → ``r#type``); names that are not valid Rust identifiers get a
``#[serde(rename = "...")]`` instead, preserving the wire name. ``bytes`` maps
to `serde_bytes`_'s ``ByteBuf`` so MessagePack uses the ``bin`` family (add
``serde_bytes`` to your ``Cargo.toml``).


.. _wire-format-contract:

Wire-format contract
--------------------

The generated types are byte-compatible with msgspec, with one asymmetry driven
by integer width:

- **MessagePack** -- encode with ``msgspec.msgpack.encode(value, type=T)``. The
  ``type=`` argument narrows ``Float32`` fields to a 5-byte float32 (matching
  Rust's ``f32``); 64-bit integers are native on both sides.
- **JSON** -- encode with plain ``msgspec.json.encode(value)`` (no ``type=``).
  The JavaScript/TypeScript codecs encode overflow-prone 64-bit integers as
  hex *strings* to survive JavaScript's ``2**53`` safe-integer limit, but Rust
  has native ``i64``/``u64``/``i128`` and reads plain JSON numbers -- so the
  JS-safe hex encoding is neither needed nor wanted here.


.. _Rust: https://www.rust-lang.org/
.. _serde: https://serde.rs/
.. _serde_json: https://docs.rs/serde_json/
.. _serde_bytes: https://docs.rs/serde_bytes/
.. _rmp-serde: https://docs.rs/rmp-serde/
