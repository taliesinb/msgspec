JSON Schema
===========

``msgspec`` provides a few utilities for generating `JSON Schema`_
specifications from msgspec-compatible :doc:`types <supported-types>` and
:doc:`constraints <constraints>`.

- `msgspec.json.schema`: generates a complete JSON Schema for a single type.
- `msgspec.json.schema_components`: generates JSON schemas for multiple types,
  along with a corresponding ``components`` mapping. This is mainly useful when
  generating multiple schemas to include in a larger specification like OpenAPI_.


The generated schemas are compatible with `JSON Schema`_ 2020-12 and OpenAPI_
3.1.


Example
-------


.. code-block:: python

    import msgspec
    from msgspec import Struct, Meta
    from typing import Annotated


    # A float constrained to values > 0
    PositiveFloat = Annotated[float, Meta(gt=0)]


    class Dimensions(Struct):
        """Dimensions for a product, all measurements in centimeters"""
        length: PositiveFloat
        width: PositiveFloat
        height: PositiveFloat


    class Product(Struct):
        """A product in a catalog"""
        id: int
        name: str
        price: PositiveFloat
        tags: set[str] = set()
        dimensions: Dimensions | None = None


    # Generate a schema for a list of products
    schema = msgspec.json.schema(list[Product])

    # Print out that schema as JSON
    print(msgspec.json.encode(schema))


.. code-block:: json

    {
      "type": "array",
      "items": {"$ref": "#/$defs/Product"},
      "$defs": {
        "Dimensions": {
          "title": "Dimensions",
          "description": "Dimensions for a product, all measurements in centimeters",
          "type": "object",
          "properties": {
            "length": {"type": "number", "exclusiveMinimum": 0},
            "width": {"type": "number", "exclusiveMinimum": 0},
            "height": {"type": "number", "exclusiveMinimum": 0}
          },
          "required": ["length", "width", "height"]
        },
        "Product": {
          "title": "Product",
          "description": "A product in a catalog",
          "type": "object",
          "properties": {
            "id": {"type": "integer"},
            "name": {"type": "string"},
            "price": {"type": "number", "exclusiveMinimum": 0},
            "tags": {
              "type": "array",
              "items": {"type": "string"},
              "default": [],
            },
            "dimensions": {
              "anyOf": [{"type": "null"}, {"$ref": "#/$defs/Dimensions"}],
              "default": null,
            }
          },
          "required": ["id", "name", "price"]
        }
      }
    }


Simplifying unions
------------------

By default every union renders as an ``anyOf`` of its member schemas, so
``int | None`` becomes ``{"anyOf": [{"type": "integer"}, {"type": "null"}]}``.
Passing ``simplify_unions=True`` to `msgspec.json.schema` or
`msgspec.json.schema_components` collapses such unions into the equivalent
type-array form wherever it is safe to do so:

.. code-block:: python

    >>> msgspec.json.schema(int | None, simplify_unions=True)
    {'type': ['integer', 'null']}

A union collapses when every member is a plain *scalar* type-keyword schema
(``null``/``boolean``/``integer``/``number``/``string``) and at most one member
carries extra constraint keys (which are per-type in JSON Schema, so they
transfer unambiguously). Members with substructure - arrays and objects - are
never folded in, so keywords like ``items`` can't end up attached to a merged
type-array:

.. code-block:: python

    >>> msgspec.json.schema(Annotated[int, Meta(ge=0)] | None, simplify_unions=True)
    {'type': ['integer', 'null'], 'minimum': 0}

Unions containing ``$ref`` members (structs, enums), literals, or multiple
constrained members fall back to ``anyOf`` unchanged. The two forms validate
identically; the type-array form is simply more compact and is handled more
gracefully by some schema consumers.


Const tags
----------

Struct tag fields (and other fixed single values, like the ``"tensor"``
marker in tensor schemas) render as ``{"const": <tag>}`` - the idiomatic
modern form, equivalent to upstream msgspec's ``{"enum": [<tag>]}``. The
output already relies on draft 2019-09+ constructs (``$defs``,
``discriminator``), so ``const`` (draft 6) imposes no additional
compatibility requirement.


Named type aliases
------------------

By default named type aliases (a `typing.NewType` or a :pep:`695` ``type X =
...``) are transparently resolved and inlined at each use site. Passing
``aliases=True`` names them as components instead, mirroring
``msgspec.inspect.type_info(..., aliases=True)``:

.. code-block:: python

    >>> type Lit = Literal["a", "b", "c"]
    >>> msgspec.json.schema(Lit, aliases=True)
    {'$ref': '#/$defs/Lit', '$defs': {'Lit': {'enum': ['a', 'b', 'c']}}}

Generic alias specializations get distinct names (``Vec[int]`` →
``Vec_int_``), and an alias of an already-named type (e.g. a Struct) becomes a
``$ref`` to that type's component.


.. _JSON Schema: https://json-schema.org/
.. _OpenAPI: https://www.openapis.org/
