from __future__ import annotations

import re
import textwrap
from collections.abc import Iterable
from typing import Any

from . import inspect as mi
from ._msgpack_runtime_ts import RUNTIME as _MSGPACK_RUNTIME_TS

__all__ = ("schema", "schema_components", "codec")


def _ts_sig_swap(src: str, js_sig: str, ts_sig: str) -> str:
    """Retype a JS function definition's signature line (its body is valid TS)."""
    return src.replace(
        f"export function {js_sig} {{", f"export function {ts_sig} {{", 1
    )

# Matches a valid (unquoted) TypeScript identifier.
_IDENT_RE = re.compile(r"^[A-Za-z_$][A-Za-z0-9_$]*$")

_INDENT = "  "

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
        if isinstance(t, mi.AbstractStructType):
            # An abstract struct is emitted as a named union alias
            # (`type Foo = Bar | Baz`), so keep its name as a component and
            # also collect its concrete descendants.
            if t.cls not in components:
                components[t.cls] = t
                collect(t.concrete_union_type)
        elif isinstance(t, mi.AliasType):
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
    def __init__(self, name_map: dict[Any, str], embed: bool = False):
        self.name_map = name_map
        # In the embedded codec the in-memory value for `bytes` is a
        # `Uint8Array` (base64 only appears on the JSON wire), so the type must
        # reflect that rather than the `string` used by the schema.
        self.embed = embed

    def to_ref(self, t: mi.Type) -> str:
        """Render a Type as an inline TypeScript type reference."""
        while isinstance(t, mi.Metadata):
            t = t.type

        # Nameable components (structs, enums, aliases, ...) are referenced by
        # name via their `.cls`.
        if hasattr(t, "cls"):
            if name := self.name_map.get(t.cls):
                return name

        if isinstance(t, mi.AbstractStructType):
            # An abstract struct behaves as the union of its concretes.
            return self.to_ref(t.concrete_union_type)
        elif isinstance(t, (mi.AnyType, mi.RawType)):
            return "any"
        elif isinstance(t, mi.NoneType):
            return "null"
        elif isinstance(t, mi.BoolType):
            return "boolean"
        elif isinstance(t, mi.IntType):
            return "bigint" if _is_bigint(t) else "number"
        elif isinstance(t, mi.FloatType):
            return "number"
        elif self.embed and isinstance(
            t, (mi.BytesType, mi.ByteArrayType, mi.MemoryViewType)
        ):
            return "Uint8Array"
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
        elif isinstance(t, mi.TensorType):
            # In the embedded codec a tensor decodes to a `TensorHandle` (a typed
            # array + dtype + shape), so the field type is `TensorHandle`. In the
            # bare schema / external codec it's just a typed array, dispatched on
            # dtype (TypeScript can't express rank/shape). dtype `None` (any)
            # falls back to the generic typed-array view.
            if self.embed:
                return "TensorHandle"
            if t.dtype is None:
                return "ArrayBufferView"
            return _DTYPE_TS_TENSOR[t.dtype]
        elif isinstance(t, mi.ArrayType):
            # A flat `Array` is a plain typed array (no handle) in both the bare
            # schema and the embedded codec, dispatched on dtype.
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
        """Render a nameable Type as a top-level TypeScript definition.

        Structs (object- and array-like), dataclasses, typed-dicts, and
        named-tuples all become object `class` definitions - the array-on-wire
        shape of array-like structs and named-tuples is handled by the codec,
        not the schema.
        """
        if isinstance(t, mi.AbstractStructType):
            # A named alias for the tagged union of its concrete descendants.
            return f"export type {name} = {self.to_ref(t.concrete_union_type)};"
        elif isinstance(t, mi.AliasType):
            return f"export type {name} = {self.to_ref(t.value)};"
        elif isinstance(t, mi.EnumType):
            return self._enum_def(name, t)
        else:
            # Struct, TypedDict, Dataclass, NamedTuple
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


# Types whose TypeScript value is already exactly the MessagePack wire value, so
# encoding/decoding is the identity (no transform needed). `IntType`/`FloatType`
# are handled separately in `_is_identity`, since a 64-bit `IntType` maps to
# `bigint` and needs a conversion.
_SCALAR_IDENTITY = (
    mi.AnyType,
    mi.RawType,
    mi.NoneType,
    mi.BoolType,
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
)

# The `msgspec.data` integer formats that map to a JavaScript `bigint`.
_BIGINT_ITYPES = frozenset({"int64", "uint64", "int", "uint"})


def _is_bigint(t: mi.Type) -> bool:
    """Whether an `IntType` maps to a JavaScript `bigint` (a 64-bit or generic
    integer marker) rather than a `number`."""
    return isinstance(t, mi.IntType) and t.itype in _BIGINT_ITYPES


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
    if isinstance(t, mi.IntType):
        # `bigint` ints (int64/uint64/Int/UInt) need a `BigInt(...)` /
        # `_toSafeInt(...)` conversion; `number` ints pass through.
        return not _is_bigint(t)
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
    if isinstance(t, mi.AbstractStructType):
        # An abstract struct behaves as the union of its concrete descendants.
        return _unwrap(t.concrete_union_type)
    return t


class _CodecGenerator:
    """Emits TypeScript encode/decode functions via structural recursion.

    Encoders emit the discriminant tag for tagged-union structs; decoders
    dispatch on that tag. Validation is intentionally not performed.
    """

    def __init__(
        self,
        name_map: dict[Any, str],
        tensor_encoder: str | None = None,
        tensor_decoder: str | None = None,
        force_int64: bool = False,
    ):
        self.name_map = name_map
        self.schema = _SchemaGenerator(name_map)
        self.tensor_encoder = tensor_encoder
        self.tensor_decoder = tensor_decoder
        self.force_int64 = force_int64

    # -- expression helpers -------------------------------------------------
    def enc(self, t: mi.Type, expr: str) -> str:
        """A TS expression encoding the typed value ``expr`` to wire form."""
        t = _unwrap(t)
        if _is_identity(t):
            return expr
        if isinstance(t, mi.TensorType):
            if self.tensor_encoder is None:
                raise TypeError(
                    "codec() requires `tensor_encoder` to encode tensor types"
                )
            return f"{self.tensor_encoder}({expr})"
        if _is_bigint(t):
            # a `bigint` int. With force_int64 the msgpack lib encodes the
            # bigint directly (via useBigInt64); otherwise narrow to a number for
            # the wire, throwing if that would lose precision.
            if self.force_int64:
                return expr
            return f"_toSafeInt({expr})"
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
        if isinstance(t, mi.TensorType):
            if self.tensor_decoder is None:
                raise TypeError(
                    "codec() requires `tensor_decoder` to decode tensor types"
                )
            return f"{self.tensor_decoder}({expr} as TensorHandle)"
        if _is_bigint(t):
            # a `bigint` int; the wire value is a number (or, with
            # useBigInt64, already a bigint), so normalize into a real bigint.
            return f"BigInt({expr} as number | bigint)"
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
        # An abstract-struct member unwraps to a nested union; flatten so its
        # concretes participate in tag dispatch directly.
        flat: list = []
        for m in members:
            if isinstance(m, mi.UnionType):
                flat.extend(_unwrap(x) for x in m.types)
            else:
                flat.append(m)
        members = flat
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
        # NamedTuple / array_like struct: the TS value is an object (named
        # fields), but the wire form is a positional array (tag first if tagged).
        head = []
        if isinstance(t, mi.StructType) and t.tag_field is not None:
            head.append(_literal_value(t.tag))
        elems = head + [
            self.enc(f.type, f"value[{_str(f.encode_name)}]") for f in t.fields
        ]
        body = ", ".join(elems)
        return (
            f"export function encode{name}(value: {name}): unknown {{\n"
            f"{_INDENT}return [{body}];\n}}"
        )

    def _dec_array_def(self, name: str, t: mi.Type) -> str:
        # Inverse of `_enc_array_def`: a positional wire array back into an
        # object with named fields.
        lines = [f"export function decode{name}(data: unknown): {name} {{"]
        lines.append(f"{_INDENT}const a = data as unknown[];")
        lines.append(f"{_INDENT}return {{")
        offset = 0
        if isinstance(t, mi.StructType) and t.tag_field is not None:
            lines.append(
                f"{_INDENT}{_INDENT}{_prop_name(t.tag_field)}: "
                f"{_literal_value(t.tag)},"
            )
            offset = 1
        for i, f in enumerate(t.fields):
            key = _prop_name(f.encode_name)
            lines.append(
                f"{_INDENT}{_INDENT}{key}: {self.dec(f.type, f'a[{i + offset}]')},"
            )
        lines.append(f"{_INDENT}}} as {name};")
        lines.append("}")
        return "\n".join(lines)


def _contains_tensor(t: mi.Type, _seen: set | None = None) -> bool:
    """Whether a `TensorType` appears anywhere in the type tree."""
    if _seen is None:
        _seen = set()
    t = _unwrap(t)
    if isinstance(t, mi.TensorType):
        return True
    if hasattr(t, "cls"):
        if t.cls in _seen:
            return False
        _seen.add(t.cls)
    if isinstance(t, (mi.StructType, mi.TypedDictType, mi.DataclassType, mi.NamedTupleType)):
        return any(_contains_tensor(f.type, _seen) for f in t.fields)
    if isinstance(t, mi.AliasType):
        return _contains_tensor(t.value, _seen)
    if isinstance(t, (mi.ListType, mi.VarTupleType, mi.SetType, mi.FrozenSetType)):
        return _contains_tensor(t.item_type, _seen)
    if isinstance(t, mi.TupleType):
        return any(_contains_tensor(i, _seen) for i in t.item_types)
    if isinstance(t, (mi.DictType, mi.FrozenDictType)):
        return _contains_tensor(t.key_type, _seen) or _contains_tensor(t.value_type, _seen)
    if isinstance(t, mi.UnionType):
        return any(_contains_tensor(m, _seen) for m in t.types)
    return False


# TypeScript runtime for tensors: a `TensorHandle` class plus an
# `@msgpack/msgpack` ExtensionCodec that serializes it to/from the reserved
# ext type 84, byte-compatible with msgspec's payload
# (version + dtype code + shape[u64 BE] + raw bytes).
_TENSOR_RUNTIME = '''\
const _TENSOR_DTYPES = ["uint8", "uint16", "uint32", "uint64", "int8", "int16", "int32", "int64", "float32", "float64", "bool"];
const _TENSOR_EXT_TYPE = 84;

export class TensorHandle {
  data: Uint8Array;
  dtype: string | null;
  shape: number[] | null;
  constructor(data: Uint8Array, dtype: string | null = null, shape: number[] | null = null) {
    this.data = data;
    this.dtype = dtype;
    this.shape = shape;
  }
}

function _encodeTensorHandle(h: TensorHandle): Uint8Array {
  const ndim = h.shape === null ? 0 : h.shape.length;
  const headerLen = 3 + (h.shape === null ? 0 : 8 * ndim);
  const out = new Uint8Array(headerLen + h.data.length);
  const view = new DataView(out.buffer);
  out[0] = 1;
  out[1] = h.dtype === null ? 0xff : _TENSOR_DTYPES.indexOf(h.dtype);
  out[2] = h.shape === null ? 0xff : ndim;
  let off = 3;
  if (h.shape !== null) {
    for (const s of h.shape) {
      view.setBigUint64(off, BigInt(s), false);
      off += 8;
    }
  }
  out.set(h.data, off);
  return out;
}

function _decodeTensorHandle(data: Uint8Array): TensorHandle {
  const view = new DataView(data.buffer, data.byteOffset, data.byteLength);
  const dcode = data[1];
  const ndimByte = data[2];
  let off = 3;
  let shape: number[] | null = null;
  if (ndimByte !== 0xff) {
    shape = [];
    for (let i = 0; i < ndimByte; i++) {
      shape.push(Number(view.getBigUint64(off, false)));
      off += 8;
    }
  }
  const dtype = dcode === 0xff ? null : _TENSOR_DTYPES[dcode];
  return new TensorHandle(data.slice(off), dtype, shape);
}

const _extensionCodec = new ExtensionCodec();
_extensionCodec.register({
  type: _TENSOR_EXT_TYPE,
  encode: (input: unknown): Uint8Array | null =>
    input instanceof TensorHandle ? _encodeTensorHandle(input) : null,
  decode: (data: Uint8Array): TensorHandle => _decodeTensorHandle(data),
});'''

# Emitted (when `force_int64=False` and the type has a bigint scalar) so encode
# fails loudly instead of silently narrowing a too-large integer to a lossy
# `number`.
_TOSAFEINT_RUNTIME = '''\
function _toSafeInt(v: bigint | number): number {
  const n = Number(v);
  if (!Number.isSafeInteger(n)) {
    throw new Error(
      "integer " + v + " is outside the JS safe-integer range; " +
      "regenerate this codec with force_int64=True"
    );
  }
  return n;
}'''


def _contains_bigint(t: mi.Type, _seen: set | None = None) -> bool:
    """Whether a `bigint` int (int64/uint64/Int/UInt) appears anywhere in the tree."""
    if _seen is None:
        _seen = set()
    t = _unwrap(t)
    if isinstance(t, mi.IntType):
        return _is_bigint(t)
    if hasattr(t, "cls"):
        if t.cls in _seen:
            return False
        _seen.add(t.cls)
    if isinstance(t, (mi.StructType, mi.TypedDictType, mi.DataclassType, mi.NamedTupleType)):
        return any(_contains_bigint(f.type, _seen) for f in t.fields)
    if isinstance(t, mi.AliasType):
        return _contains_bigint(t.value, _seen)
    if isinstance(t, (mi.ListType, mi.VarTupleType, mi.SetType, mi.FrozenSetType)):
        return _contains_bigint(t.item_type, _seen)
    if isinstance(t, mi.TupleType):
        return any(_contains_bigint(i, _seen) for i in t.item_types)
    if isinstance(t, (mi.DictType, mi.FrozenDictType)):
        return _contains_bigint(t.key_type, _seen) or _contains_bigint(t.value_type, _seen)
    if isinstance(t, mi.UnionType):
        return any(_contains_bigint(m, _seen) for m in t.types)
    return False


def _str_inner(s: str) -> str:
    """Escape a string for use inside a double-quoted TS string literal."""
    return s.replace("\\", "\\\\").replace('"', '\\"')


def _str(s: str) -> str:
    """A double-quoted TS string literal for a property access key."""
    return f'"{_str_inner(s)}"'


def codec(
    type: Any,
    *,
    msgpack: bool = True,
    json: bool = True,
    embed_msgpack: bool = True,
    tensor_encoder: str | None = None,
    tensor_decoder: str | None = None,
    force_int64: bool = False,
) -> str:
    """Generate TypeScript type definitions plus a MessagePack/JSON codec.

    By default (``embed_msgpack=True``) the output is a self-contained module
    that inlines its own tight MessagePack reader/writer and exports a
    ``msgpack`` and/or ``json`` namespace (``{ encode, decode }`` each) - the
    TypeScript counterpart of `msgspec.javascript.codec`, and the client side of
    a **type-directed** ``msgspec`` encoder. 64-bit / generic integer types
    round-trip as native ``bigint`` (hex strings in JSON when large); ``bytes``
    are base64 in JSON; ``Float32`` narrows to a 5-byte msgpack float32.
    ``msgspec.data.Tensor`` fields decode to a re-exported ``TensorHandle``
    (typed array + ``dtype`` + ``shape``).

    With ``embed_msgpack=False`` the older `@msgpack/msgpack`-based output is
    produced instead; in that mode tensors require the ``tensor_encoder`` /
    ``tensor_decoder`` params (see below).

    Parameters
    ----------
    type : type
        The type to generate a codec for.
    msgpack, json : bool, optional
        Which format namespaces to emit (default both). Only used when
        ``embed_msgpack=True``.
    embed_msgpack : bool, optional
        Inline our own msgpack runtime instead of importing ``@msgpack/msgpack``
        (default ``True``).
    tensor_encoder, tensor_decoder, force_int64
        Only used when ``embed_msgpack=False`` (the ``@msgpack/msgpack`` path).

    Returns
    -------
    str
        The generated TypeScript source.
    """
    if not (msgpack or json):
        raise ValueError("at least one of `msgpack` / `json` must be enabled")
    if embed_msgpack:
        return _codec_embed(type, msgpack, json)
    return _codec_external(type, tensor_encoder, tensor_decoder, force_int64)


def _codec_embed(type: Any, msgpack: bool, json: bool) -> str:
    from . import _javascript as _js

    type_infos = mi.multi_type_info([type], aliases=True)
    (root,) = type_infos
    component_types = _collect_component_types(type_infos)
    name_map = _build_name_map(component_types)

    schema_gen = _SchemaGenerator(name_map, embed=True)
    jsgen = _js._CodecGenerator(name_map)

    parts = [_MSGPACK_RUNTIME_TS.strip()]
    if _contains_tensor(root):
        # Re-export the runtime's TensorHandle (as both a type and a value).
        parts.append("export { TensorHandle };")

    # Type definitions. The codec works with plain objects, so struct shapes are
    # emitted as `interface` (not `class`) - accurate and `strict`-clean. Enums
    # and union aliases are unaffected.
    for cls, t in component_types.items():
        parts.append(
            schema_gen.to_def(name_map[cls], t).replace(
                "export class ", "export interface ", 1
            )
        )
    root_ref = schema_gen.to_ref(root)
    if getattr(root, "cls", None) not in name_map:
        parts.append(f"export type Root = {root_ref};")
        root_ref = "Root"

    def struct_components():
        for cls, t in component_types.items():
            if isinstance(t, _CODEC_FUNC_TYPES) and not isinstance(
                t, mi.AbstractStructType
            ):
                yield name_map[cls], t

    if msgpack:
        for nm, t in struct_components():
            parts.append(
                _ts_sig_swap(
                    jsgen.enc_def(nm, t),
                    f"encode{nm}(w, v)",
                    f"encode{nm}(w: Writer, v: {nm}): void",
                )
            )
            dec = _ts_sig_swap(
                jsgen.dec_def(nm, t),
                f"decode{nm}(r)",
                f"decode{nm}(r: Reader): {nm}",
            )
            # the decode accumulator is built untyped then returned as the struct
            dec = dec.replace("const o = {", "const o: any = {")
            parts.append(dec)
        renc = jsgen.enc(root, "value")
        parts.append(
            "export const msgpack = {\n"
            f"{_INDENT}encode(value: {root_ref}): Uint8Array {{\n"
            f"{_INDENT}{_INDENT}const w = new Writer();\n"
            f"{textwrap.indent(renc, _INDENT * 2)}\n"
            f"{_INDENT}{_INDENT}return w.bytes();\n"
            f"{_INDENT}}},\n"
            f"{_INDENT}decode(bytes: Uint8Array): {root_ref} {{\n"
            f"{_INDENT}{_INDENT}const r = new Reader(bytes);\n"
            f"{_INDENT}{_INDENT}return {jsgen.dec(root)};\n"
            f"{_INDENT}}},\n"
            "};"
        )

    if json:
        for nm, t in struct_components():
            parts.append(
                _ts_sig_swap(
                    jsgen.json_enc_def(nm, t),
                    f"encodeJson{nm}(v)",
                    f"encodeJson{nm}(v: {nm}): any",
                )
            )
            parts.append(
                _ts_sig_swap(
                    jsgen.json_dec_def(nm, t),
                    f"decodeJson{nm}(o)",
                    f"decodeJson{nm}(o: any): {nm}",
                )
            )
        parts.append(
            "export const json = {\n"
            f"{_INDENT}encode(value: {root_ref}): string {{ return JSON.stringify({jsgen.json_enc(root, 'value')}); }},\n"
            f"{_INDENT}decode(text: string): {root_ref} {{ const o = JSON.parse(text); return {jsgen.json_dec(root, 'o')}; }},\n"
            "};"
        )

    return "\n\n".join(parts) + "\n"


def _codec_external(
    type: Any,
    tensor_encoder: str | None = None,
    tensor_decoder: str | None = None,
    force_int64: bool = False,
) -> str:
    """The `@msgpack/msgpack`-based codec (``embed_msgpack=False``).

    The output imports ``encode``/``decode`` from ``@msgpack/msgpack`` and adds
    an ``encode(value)`` / ``decode(bytes)`` pair for the top-level type, along
    with per-struct ``encodeX``/``decodeX`` functions. Encoders emit the
    discriminant tag for tagged-union structs; decoders dispatch on it. No
    runtime validation is performed - this is a structural transform.

    Parameters
    ----------
    type : type
        The type to generate a codec for.
    tensor_encoder : str, optional
        The name of a TypeScript function ``(value) => TensorHandle`` to call
        when encoding a tensor. Required if ``type`` contains any tensor types.
    tensor_decoder : str, optional
        The name of a TypeScript function ``(handle: TensorHandle) => value`` to
        call when decoding a tensor. Required if ``type`` contains any tensor
        types. Both functions are assumed to be in scope in the emitted module;
        a ``TensorHandle`` class and the ext-84 ``ExtensionCodec`` are generated.
    force_int64 : bool, optional
        How 64-bit integer scalars (``Int``/``Int64``/``UInt64``, which are
        ``bigint`` in TypeScript) cross the wire. `@msgpack/msgpack` does not
        encode ``bigint`` by default. If ``False`` (the default), the codec
        narrows each ``bigint`` to a ``number`` for encoding, throwing if the
        value exceeds the JS safe-integer range (``2**53``) - so the compact,
        byte-identical output is only ever produced when it is lossless. If
        ``True``, the msgpack library's ``useBigInt64`` option is enabled and
        bigints are encoded directly as 64-bit ints (full precision, but larger,
        non-compact output).

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

    has_tensor = _contains_tensor(root)
    if has_tensor and (tensor_encoder is None or tensor_decoder is None):
        raise TypeError(
            "codec() requires both `tensor_encoder` and `tensor_decoder` when "
            "the type contains tensor types"
        )

    has_bigint = _contains_bigint(root)

    schema_gen = _SchemaGenerator(name_map)
    codec_gen = _CodecGenerator(
        name_map, tensor_encoder, tensor_decoder, force_int64
    )

    imports = "encode as _mpEncode, decode as _mpDecode"
    if has_tensor:
        imports += ", ExtensionCodec"
    parts = [f'import {{ {imports} }} from "@msgpack/msgpack";']

    if has_tensor:
        parts.append(_TENSOR_RUNTIME)
    if has_bigint and not force_int64:
        parts.append(_TOSAFEINT_RUNTIME)

    # A shared options object for the msgpack calls, when any is needed.
    opt_fields = []
    if has_tensor:
        opt_fields.append("extensionCodec: _extensionCodec")
    if force_int64:
        opt_fields.append("useBigInt64: true")
    if opt_fields:
        parts.append(f"const _codecOptions = {{ {', '.join(opt_fields)} }};")

    # Type definitions (same as `schema`).
    for cls, t in component_types.items():
        parts.append(schema_gen.to_def(name_map[cls], t))

    root_ref = schema_gen.to_ref(root)
    if getattr(root, "cls", None) not in name_map:
        parts.append(f"export type Root = {root_ref};")
        root_ref = "Root"

    # Per-component encode/decode functions. An abstract struct is a union
    # alias, not an object - its codec dispatch is inlined at use sites.
    for cls, t in component_types.items():
        if isinstance(t, _CODEC_FUNC_TYPES) and not isinstance(
            t, mi.AbstractStructType
        ):
            parts.append(codec_gen.enc_def(name_map[cls], t))
            parts.append(codec_gen.dec_def(name_map[cls], t))

    # Top-level entry points.
    opts = ", _codecOptions" if opt_fields else ""
    decode_call = f"_mpDecode(bytes{opts})"
    parts.append(
        f"export function encode(value: {root_ref}): Uint8Array {{\n"
        f"{_INDENT}return _mpEncode({codec_gen.enc(root, 'value')}{opts});\n}}"
    )
    parts.append(
        f"export function decode(bytes: Uint8Array): {root_ref} {{\n"
        f"{_INDENT}return {codec_gen.dec(root, decode_call)} as {root_ref};\n}}"
    )

    return "\n\n".join(parts) + "\n"
