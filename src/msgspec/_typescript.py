from __future__ import annotations

import re
import textwrap
from collections.abc import Iterable
from typing import Any

from . import inspect as mi

__all__ = ("schema", "schema_components", "codec")

# Matches a valid (unquoted) TypeScript identifier.
_IDENT_RE = re.compile(r"^[A-Za-z_$][A-Za-z0-9_$]*$")

_INDENT = "  "

# TypeScript representation of a `msgspec.data` scalar dtype. 64-bit integers
# exceed JS number precision so map to `bigint`.
_DTYPE_TS_SCALAR = {
    "int8": "number",
    "int16": "number",
    "int32": "number",
    "uint8": "number",
    "uint16": "number",
    "uint32": "number",
    "float32": "number",
    "float64": "number",
    "int64": "bigint",
    "uint64": "bigint",
    "bool": "boolean",
}

# TypeScript typed-array for a packed tensor of a given dtype. TS has no bit-
# packed boolean array, so `bool` uses `Uint8Array`.
_DTYPE_TS_TENSOR = {
    "int8": "Int8Array",
    "int16": "Int16Array",
    "int32": "Int32Array",
    "uint8": "Uint8Array",
    "uint16": "Uint16Array",
    "uint32": "Uint32Array",
    "int64": "BigInt64Array",
    "uint64": "BigUint64Array",
    "float32": "Float32Array",
    "float64": "Float64Array",
    "bool": "Uint8Array",
}


def schema(type: Any) -> str:
    """Generate TypeScript type definitions for a given type.

    Struct types (and other "nameable" types like enums, dataclasses,
    typed-dicts, and named-tuples) are emitted as top-level ``class``/``enum``
    definitions. If the top-level ``type`` is not itself a nameable type (for
    example ``list[Point]``), an exported ``type Root = ...`` alias is emitted
    to name it.

    Parameters
    ----------
    type : type
        The type to generate TypeScript definitions for.

    Returns
    -------
    schema : str
        The generated TypeScript source.

    See Also
    --------
    schema_components
    """
    (root,), components = schema_components((type,))

    parts = list(components.values())
    # If the root type isn't a nameable component it won't already appear in
    # `components`; emit an alias so the returned source names it.
    if root not in components:
        parts.append(f"export type Root = {root};")
    return "\n\n".join(parts) + "\n"


def schema_components(
    types: Iterable[Any],
) -> tuple[tuple[str, ...], dict[str, str]]:
    """Generate TypeScript definitions for one or more types.

    Parameters
    ----------
    types : Iterable[type]
        An iterable of one or more types to generate definitions for.

    Returns
    -------
    refs : tuple[str, ...]
        A tuple of TypeScript type *references* (e.g. ``"Point"`` or
        ``"Array<Point>"``), one for each type in ``types``.
    components : dict[str, str]
        A mapping of name to the TypeScript source defining each nameable
        component referenced by ``refs``.

    See Also
    --------
    schema
    """
    # `aliases=True` so `type Pixels = int` is preserved as an `AliasType` and
    # emitted as a TypeScript `type` alias rather than being inlined.
    type_infos = mi.multi_type_info(types, aliases=True)

    component_types = _collect_component_types(type_infos)

    name_map = _build_name_map(component_types)

    gen = _SchemaGenerator(name_map)

    refs = tuple(gen.to_ref(t) for t in type_infos)

    components = {
        name_map[cls]: gen.to_def(name_map[cls], t)
        for cls, t in component_types.items()
    }
    return refs, components


def _collect_component_types(type_infos: Iterable[mi.Type]) -> dict[Any, mi.Type]:
    """Find all "nameable" types in the type tree worthy of a top-level
    definition (Struct, Dataclass, NamedTuple, TypedDict, Enum, and alias
    types). All are keyed by their ``.cls`` (the alias' ``.cls`` being the
    possibly-subscripted alias itself)."""
    components: dict[Any, mi.Type] = {}

    def collect(t):
        if isinstance(t, mi.AliasType):
            if t.cls not in components:
                components[t.cls] = t
                collect(t.value)
        elif isinstance(
            t, (mi.StructType, mi.TypedDictType, mi.DataclassType, mi.NamedTupleType)
        ):
            if t.cls not in components:
                components[t.cls] = t
                for f in t.fields:
                    collect(f.type)
        elif isinstance(t, mi.EnumType):
            components[t.cls] = t
        elif isinstance(t, mi.Metadata):
            collect(t.type)
        elif isinstance(t, mi.CollectionType):
            collect(t.item_type)
        elif isinstance(t, mi.TupleType):
            for st in t.item_types:
                collect(st)
        elif isinstance(t, (mi.DictType, mi.FrozenDictType)):
            collect(t.key_type)
            collect(t.value_type)
        elif isinstance(t, mi.UnionType):
            for st in t.types:
                collect(st)

    for t in type_infos:
        collect(t)

    return components


def _type_repr(obj):
    return obj.__name__ if isinstance(obj, type) else repr(obj)


def _get_class_name(cls: Any) -> str:
    if hasattr(cls, "__origin__"):
        name = cls.__origin__.__name__
        args = "_".join(_type_repr(a) for a in cls.__args__)
        return f"{name}_{args}"
    return cls.__name__


def _get_doc(t: mi.Type) -> str:
    assert hasattr(t, "cls")
    cls = getattr(t.cls, "__origin__", t.cls)
    doc = getattr(cls, "__doc__", "")
    if not doc:
        return ""
    doc = textwrap.dedent(doc).strip("\r\n")
    if isinstance(t, mi.EnumType):
        if doc == "An enumeration.":
            return ""
    elif isinstance(t, (mi.NamedTupleType, mi.DataclassType)):
        if doc.startswith(f"{cls.__name__}(") and doc.endswith(")"):
            return ""
    return doc


def _build_name_map(component_types: dict[Any, mi.Type]) -> dict[Any, str]:
    """A mapping from nameable subcomponents to a generated TypeScript
    identifier, disambiguating conflicts by import path."""

    def normalize(name):
        name = re.sub(r"[^a-zA-Z0-9_$]", "_", name)
        if name and name[0].isdigit():
            name = "_" + name
        return name

    def fullname(cls):
        return normalize(f"{cls.__module__}.{cls.__qualname__}")

    conflicts = set()
    names: dict[str, Any] = {}

    for cls in component_types:
        name = normalize(_get_class_name(cls))
        if name in names:
            old = names.pop(name)
            conflicts.add(name)
            names[fullname(old)] = old
        if name in conflicts:
            names[fullname(cls)] = cls
        else:
            names[name] = cls
    return {v: k for k, v in names.items()}


def _prop_name(name: str) -> str:
    """Render a property name, quoting it if it's not a valid identifier."""
    if _IDENT_RE.match(name):
        return name
    escaped = name.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _literal_value(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    return repr(value)


def _jsdoc(doc: str) -> str:
    lines = doc.splitlines()
    if len(lines) == 1:
        return f"/** {lines[0]} */\n"
    body = "\n".join(f" * {line}".rstrip() for line in lines)
    return f"/**\n{body}\n */\n"


class _SchemaGenerator:
    def __init__(self, name_map: dict[Any, str]):
        self.name_map = name_map

    def to_ref(self, t: mi.Type) -> str:
        """Render a Type as an inline TypeScript type reference."""
        while isinstance(t, mi.Metadata):
            t = t.type

        # Nameable components (structs, enums, aliases, ...) are referenced by
        # name via their `.cls`.
        if hasattr(t, "cls"):
            if name := self.name_map.get(t.cls):
                return name

        if isinstance(t, (mi.AnyType, mi.RawType)):
            return "any"
        elif isinstance(t, mi.NoneType):
            return "null"
        elif isinstance(t, mi.BoolType):
            return "boolean"
        elif isinstance(t, (mi.IntType, mi.FloatType)):
            return "number"
        elif isinstance(
            t,
            (
                mi.StrType,
                mi.BytesType,
                mi.ByteArrayType,
                mi.MemoryViewType,
                mi.DateTimeType,
                mi.TimeType,
                mi.DateType,
                mi.TimeDeltaType,
                mi.UUIDType,
                mi.DecimalType,
            ),
        ):
            return "string"
        elif isinstance(t, mi.ListType):
            return f"Array<{self.to_ref(t.item_type)}>"
        elif isinstance(t, mi.VarTupleType):
            return f"Array<{self.to_ref(t.item_type)}>"
        elif isinstance(t, (mi.SetType, mi.FrozenSetType)):
            return f"Set<{self.to_ref(t.item_type)}>"
        elif isinstance(t, mi.TupleType):
            if not t.item_types:
                return "[]"
            inner = ", ".join(self.to_ref(i) for i in t.item_types)
            return f"[{inner}]"
        elif isinstance(t, (mi.DictType, mi.FrozenDictType)):
            key = self.to_ref(t.key_type)
            # Record keys must be string-ish/number in TypeScript.
            if key not in ("string", "number"):
                key = "string"
            return f"Record<{key}, {self.to_ref(t.value_type)}>"
        elif isinstance(t, mi.ScalarType):
            # A `msgspec.data` scalar; dtype `None` (`Scalar`) is any scalar.
            if t.dtype is None:
                return "number | boolean"
            return _DTYPE_TS_SCALAR[t.dtype]
        elif isinstance(t, mi.TensorType):
            # A packed tensor -> a typed array, dispatched on dtype. TypeScript
            # can't express rank/shape, so ndims/sizes are dropped. dtype `None`
            # (any) falls back to the generic typed-array view.
            if t.dtype is None:
                return "ArrayBufferView"
            return _DTYPE_TS_TENSOR[t.dtype]
        elif isinstance(t, mi.UnionType):
            return " | ".join(self.to_ref(a) for a in t.types)
        elif isinstance(t, mi.LiteralType):
            return " | ".join(_literal_value(v) for v in t.values)
        elif isinstance(t, mi.ExtType):
            raise TypeError("TypeScript schema doesn't support msgpack Ext types")
        elif isinstance(t, mi.CustomType):
            raise TypeError(
                f"TypeScript schema doesn't support custom type {t.cls!r}"
            )
        else:
            raise TypeError(f"TypeScript schema doesn't support type {t!r}")

    def to_def(self, name: str, t: mi.Type) -> str:
        """Render a nameable Type as a top-level TypeScript definition."""
        if isinstance(t, mi.AliasType):
            return f"export type {name} = {self.to_ref(t.value)};"
        elif isinstance(t, mi.EnumType):
            return self._enum_def(name, t)
        elif isinstance(t, mi.StructType) and t.array_like:
            return self._array_struct_def(name, t)
        elif isinstance(t, mi.NamedTupleType):
            return self._namedtuple_def(name, t)
        else:
            # Struct (object-like), TypedDict, Dataclass
            return self._object_def(name, t)

    def _enum_def(self, name: str, t: mi.EnumType) -> str:
        lines = []
        if doc := _get_doc(t):
            lines.append(_jsdoc(doc).rstrip("\n"))
        lines.append(f"export enum {name} {{")
        for member in t.cls:
            lines.append(f"{_INDENT}{_prop_name(member.name)} = {_literal_value(member.value)},")
        lines.append("}")
        return "\n".join(lines)

    def _field_line(self, field: mi.Field) -> str:
        optional = "" if field.required else "?"
        return (
            f"{_INDENT}{_prop_name(field.encode_name)}{optional}: "
            f"{self.to_ref(field.type)};"
        )

    def _object_def(self, name: str, t: mi.Type) -> str:
        lines = []
        if doc := _get_doc(t):
            lines.append(_jsdoc(doc).rstrip("\n"))
        lines.append(f"export class {name} {{")
        if isinstance(t, mi.StructType) and t.tag_field is not None:
            lines.append(
                f"{_INDENT}{_prop_name(t.tag_field)}: {_literal_value(t.tag)};"
            )
        for field in t.fields:
            lines.append(self._field_line(field))
        lines.append("}")
        return "\n".join(lines)

    def _array_struct_def(self, name: str, t: mi.StructType) -> str:
        # array_like structs encode as fixed tuples.
        lines = []
        if doc := _get_doc(t):
            lines.append(_jsdoc(doc).rstrip("\n"))
        elems = []
        if t.tag_field is not None:
            elems.append(_literal_value(t.tag))
        elems.extend(self.to_ref(f.type) for f in t.fields)
        lines.append(f"export type {name} = [{', '.join(elems)}];")
        return "\n".join(lines)

    def _namedtuple_def(self, name: str, t: mi.NamedTupleType) -> str:
        # NamedTuples encode as arrays.
        lines = []
        if doc := _get_doc(t):
            lines.append(_jsdoc(doc).rstrip("\n"))
        elems = ", ".join(self.to_ref(f.type) for f in t.fields)
        lines.append(f"export type {name} = [{elems}];")
        return "\n".join(lines)


# Types whose TypeScript value is already exactly the MessagePack wire value, so
# encoding/decoding is the identity (no transform needed).
_SCALAR_IDENTITY = (
    mi.AnyType,
    mi.RawType,
    mi.NoneType,
    mi.BoolType,
    mi.IntType,
    mi.FloatType,
    mi.StrType,
    mi.BytesType,
    mi.ByteArrayType,
    mi.MemoryViewType,
    mi.DateTimeType,
    mi.TimeType,
    mi.DateType,
    mi.TimeDeltaType,
    mi.UUIDType,
    mi.DecimalType,
    mi.EnumType,
    mi.LiteralType,
    mi.ScalarType,
)

# Object/tuple-like components that get a generated encode/decode function pair.
_CODEC_FUNC_TYPES = (
    mi.StructType,
    mi.TypedDictType,
    mi.DataclassType,
    mi.NamedTupleType,
)


def _is_identity(t: mi.Type) -> bool:
    """Whether a value of this type round-trips through MessagePack unchanged
    (so its codec is the identity function)."""
    while isinstance(t, mi.Metadata):
        t = t.type
    if isinstance(t, _SCALAR_IDENTITY):
        return True
    if isinstance(t, (mi.ListType, mi.VarTupleType)):
        return _is_identity(t.item_type)
    if isinstance(t, mi.TupleType):
        return all(_is_identity(i) for i in t.item_types)
    if isinstance(t, (mi.DictType, mi.FrozenDictType)):
        return _is_identity(t.value_type)
    if isinstance(t, mi.AliasType):
        return _is_identity(t.value)
    if isinstance(t, mi.UnionType):
        # A union with a struct member needs tag dispatch; otherwise (scalars,
        # optionals of scalars) it passes through.
        return all(_is_identity(a) for a in t.types)
    # Structs, dataclasses, typed-dicts, named-tuples, sets, tensors: not identity.
    return False


def _unwrap(t: mi.Type) -> mi.Type:
    while isinstance(t, mi.Metadata):
        t = t.type
    return t


class _CodecGenerator:
    """Emits TypeScript encode/decode functions via structural recursion.

    Encoders emit the discriminant tag for tagged-union structs; decoders
    dispatch on that tag. Validation is intentionally not performed.
    """

    def __init__(self, name_map: dict[Any, str]):
        self.name_map = name_map
        self.schema = _SchemaGenerator(name_map)

    # -- expression helpers -------------------------------------------------
    def enc(self, t: mi.Type, expr: str) -> str:
        """A TS expression encoding the typed value ``expr`` to wire form."""
        t = _unwrap(t)
        if _is_identity(t):
            return expr
        if isinstance(t, _CODEC_FUNC_TYPES):
            return f"encode{self.name_map[t.cls]}({expr})"
        if isinstance(t, mi.AliasType):
            return self.enc(t.value, expr)
        if isinstance(t, (mi.ListType, mi.VarTupleType)):
            return f"{expr}.map((v) => {self.enc(t.item_type, 'v')})"
        if isinstance(t, (mi.SetType, mi.FrozenSetType)):
            inner = self.enc(t.item_type, "v")
            return f"[...{expr}]" if inner == "v" else f"[...{expr}].map((v) => {inner})"
        if isinstance(t, mi.TupleType):
            elems = ", ".join(
                self.enc(it, f"{expr}[{i}]") for i, it in enumerate(t.item_types)
            )
            return f"[{elems}]"
        if isinstance(t, (mi.DictType, mi.FrozenDictType)):
            return (
                f"Object.fromEntries(Object.entries({expr})"
                f".map(([k, v]) => [k, {self.enc(t.value_type, 'v')}]))"
            )
        if isinstance(t, mi.UnionType):
            return self._enc_union(t, expr)
        raise NotImplementedError(
            f"TypeScript codec doesn't support type {t!r} yet"
        )

    def dec(self, t: mi.Type, expr: str) -> str:
        """A TS expression decoding the wire value ``expr`` to typed form."""
        t = _unwrap(t)
        if _is_identity(t):
            return f"({expr} as {self.schema.to_ref(t)})"
        if isinstance(t, _CODEC_FUNC_TYPES):
            return f"decode{self.name_map[t.cls]}({expr})"
        if isinstance(t, mi.AliasType):
            return self.dec(t.value, expr)
        if isinstance(t, (mi.ListType, mi.VarTupleType)):
            return f"({expr} as unknown[]).map((v) => {self.dec(t.item_type, 'v')})"
        if isinstance(t, (mi.SetType, mi.FrozenSetType)):
            return (
                f"new Set(({expr} as unknown[])"
                f".map((v) => {self.dec(t.item_type, 'v')}))"
            )
        if isinstance(t, mi.TupleType):
            elems = ", ".join(
                self.dec(it, f"({expr} as unknown[])[{i}]")
                for i, it in enumerate(t.item_types)
            )
            return f"[{elems}]"
        if isinstance(t, (mi.DictType, mi.FrozenDictType)):
            return (
                f"Object.fromEntries(Object.entries({expr} as Record<string, unknown>)"
                f".map(([k, v]) => [k, {self.dec(t.value_type, 'v')}]))"
            )
        if isinstance(t, mi.UnionType):
            return self._dec_union(t, expr)
        raise NotImplementedError(
            f"TypeScript codec doesn't support type {t!r} yet"
        )

    # -- unions -------------------------------------------------------------
    def _partition_union(self, t: mi.UnionType):
        members = [_unwrap(m) for m in t.types]
        has_none = any(isinstance(m, mi.NoneType) for m in members)
        tagged = [
            m
            for m in members
            if isinstance(m, mi.StructType)
            and not m.array_like
            and m.tag_field is not None
        ]
        others = [
            m
            for m in members
            if not isinstance(m, mi.NoneType) and m not in tagged
        ]
        return has_none, tagged, others

    def _enc_union(self, t: mi.UnionType, expr: str) -> str:
        has_none, tagged, others = self._partition_union(t)
        if not tagged:
            if has_none and len(others) == 1:
                return f"({expr} === null ? null : {self.enc(others[0], expr)})"
            # scalar-only unions are identity; anything else passes through.
            return expr
        return self._union_iife(tagged, others, has_none, expr, "encode")

    def _dec_union(self, t: mi.UnionType, expr: str) -> str:
        has_none, tagged, others = self._partition_union(t)
        if not tagged:
            if has_none and len(others) == 1:
                return f"({expr} === null ? null : {self.dec(others[0], expr)})"
            return f"({expr} as {self.schema.to_ref(t)})"
        return self._union_iife(tagged, others, has_none, expr, "decode")

    def _union_iife(self, tagged, others, has_none, expr, kind: str) -> str:
        tag_field = tagged[0].tag_field
        body = []
        if has_none:
            body.append(f"{_INDENT}if (v === null) return null;")
        body.append(f"{_INDENT}switch (v[{_literal_value(tag_field)}]) {{")
        for m in tagged:
            body.append(
                f"{_INDENT}{_INDENT}case {_literal_value(m.tag)}: "
                f"return {kind}{self.name_map[m.cls]}(v);"
            )
        body.append(f"{_INDENT}}}")
        if others:
            body.append(f"{_INDENT}return v;")
        else:
            body.append(
                f"{_INDENT}throw new Error("
                f'"unexpected tag for {_str_inner(tag_field)} union");'
            )
        joined = "\n".join(body)
        return f"((v: any) => {{\n{joined}\n}})({expr})"

    # -- per-component function definitions ---------------------------------
    def enc_def(self, name: str, t: mi.Type) -> str:
        if isinstance(t, mi.NamedTupleType) or (
            isinstance(t, mi.StructType) and t.array_like
        ):
            return self._enc_array_def(name, t)
        return self._enc_object_def(name, t)

    def dec_def(self, name: str, t: mi.Type) -> str:
        if isinstance(t, mi.NamedTupleType) or (
            isinstance(t, mi.StructType) and t.array_like
        ):
            return self._dec_array_def(name, t)
        return self._dec_object_def(name, t)

    def _enc_object_def(self, name: str, t: mi.Type) -> str:
        lines = [f"export function encode{name}(value: {name}): unknown {{"]
        lines.append(f"{_INDENT}return {{")
        if isinstance(t, mi.StructType) and t.tag_field is not None:
            lines.append(
                f"{_INDENT}{_INDENT}{_prop_name(t.tag_field)}: "
                f"{_literal_value(t.tag)},"
            )
        for f in t.fields:
            key = _prop_name(f.encode_name)
            lines.append(
                f"{_INDENT}{_INDENT}{key}: {self.enc(f.type, f'value[{_str(f.encode_name)}]')},"
            )
        lines.append(f"{_INDENT}}};")
        lines.append("}")
        return "\n".join(lines)

    def _dec_object_def(self, name: str, t: mi.Type) -> str:
        lines = [f"export function decode{name}(data: unknown): {name} {{"]
        lines.append(f"{_INDENT}const o = data as Record<string, unknown>;")
        lines.append(f"{_INDENT}return {{")
        if isinstance(t, mi.StructType) and t.tag_field is not None:
            lines.append(
                f"{_INDENT}{_INDENT}{_prop_name(t.tag_field)}: "
                f"{_literal_value(t.tag)},"
            )
        for f in t.fields:
            key = _prop_name(f.encode_name)
            lines.append(
                f"{_INDENT}{_INDENT}{key}: {self.dec(f.type, f'o[{_str(f.encode_name)}]')},"
            )
        lines.append(f"{_INDENT}}} as {name};")
        lines.append("}")
        return "\n".join(lines)

    def _enc_array_def(self, name: str, t: mi.Type) -> str:
        # NamedTuple / array_like struct: encodes to a positional array.
        offset = 0
        head = []
        if isinstance(t, mi.StructType) and t.tag_field is not None:
            head.append(_literal_value(t.tag))
            offset = 1
        elems = head + [
            self.enc(f.type, f"value[{i + offset}]") for i, f in enumerate(t.fields)
        ]
        body = ", ".join(elems)
        return (
            f"export function encode{name}(value: {name}): unknown {{\n"
            f"{_INDENT}return [{body}];\n}}"
        )

    def _dec_array_def(self, name: str, t: mi.Type) -> str:
        offset = 0
        head = []
        if isinstance(t, mi.StructType) and t.tag_field is not None:
            head.append(_literal_value(t.tag))
            offset = 1
        elems = head + [
            self.dec(f.type, f"a[{i + offset}]") for i, f in enumerate(t.fields)
        ]
        body = ", ".join(elems)
        return (
            f"export function decode{name}(data: unknown): {name} {{\n"
            f"{_INDENT}const a = data as unknown[];\n"
            f"{_INDENT}return [{body}] as {name};\n}}"
        )


def _str_inner(s: str) -> str:
    """Escape a string for use inside a double-quoted TS string literal."""
    return s.replace("\\", "\\\\").replace('"', '\\"')


def _str(s: str) -> str:
    """A double-quoted TS string literal for a property access key."""
    return f'"{_str_inner(s)}"'


def codec(type: Any) -> str:
    """Generate TypeScript type definitions plus MessagePack encoders/decoders
    for a given type.

    The output imports ``encode``/``decode`` from ``@msgpack/msgpack`` and adds
    an ``encode(value)`` / ``decode(bytes)`` pair for the top-level type, along
    with per-struct ``encodeX``/``decodeX`` functions. Encoders emit the
    discriminant tag for tagged-union structs; decoders dispatch on it. No
    runtime validation is performed - this is a structural transform.

    Parameters
    ----------
    type : type
        The type to generate a codec for.

    Returns
    -------
    str
        The generated TypeScript source.

    See Also
    --------
    schema
    """
    type_infos = mi.multi_type_info([type], aliases=True)
    (root,) = type_infos
    component_types = _collect_component_types(type_infos)
    name_map = _build_name_map(component_types)

    schema_gen = _SchemaGenerator(name_map)
    codec_gen = _CodecGenerator(name_map)

    parts = [
        'import { encode as _mpEncode, decode as _mpDecode } '
        'from "@msgpack/msgpack";'
    ]

    # Type definitions (same as `schema`).
    for cls, t in component_types.items():
        parts.append(schema_gen.to_def(name_map[cls], t))

    root_ref = schema_gen.to_ref(root)
    if getattr(root, "cls", None) not in name_map:
        parts.append(f"export type Root = {root_ref};")
        root_ref = "Root"

    # Per-component encode/decode functions.
    for cls, t in component_types.items():
        if isinstance(t, _CODEC_FUNC_TYPES):
            parts.append(codec_gen.enc_def(name_map[cls], t))
            parts.append(codec_gen.dec_def(name_map[cls], t))

    # Top-level entry points.
    parts.append(
        f"export function encode(value: {root_ref}): Uint8Array {{\n"
        f"{_INDENT}return _mpEncode({codec_gen.enc(root, 'value')});\n}}"
    )
    parts.append(
        f"export function decode(bytes: Uint8Array): {root_ref} {{\n"
        f"{_INDENT}return {codec_gen.dec(root, '_mpDecode(bytes)')} as {root_ref};\n}}"
    )

    return "\n\n".join(parts) + "\n"
