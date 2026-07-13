"""Generate Rust (serde) type definitions from msgspec-compatible types.

`msgspec.rust.schema` turns a type into Rust source: `Struct` types become
``struct``\\ s, tagged-union / :ref:`abstract <abstract>` structs become
internally-tagged ``enum``\\ s, and everything else maps onto the obvious Rust
type (``list[T]`` → ``Vec<T>``, ``UInt64`` → ``u64``, …). The output carries
``#[derive(Serialize, Deserialize)]`` and the serde attributes needed to match
msgspec's wire format, so `serde_json`_ / `rmp-serde`_ interoperate directly with
`msgspec.json` / `msgspec.msgpack`.

Wire-format contract (verified byte-for-byte against serde):

* **MessagePack** <-> ``msgspec.msgpack.encode(value, type=T)``. Use ``type=`` so
  ``Float32`` fields narrow to a 5-byte float32 (matching Rust ``f32``); 64-bit
  ints are native on both sides.
* **JSON** <-> ``msgspec.json.encode(value)`` (plain / value-directed). Unlike the
  JavaScript/TypeScript codecs, Rust has native ``i64``/``u64``/``i128`` and needs
  no JS-safe-integer hex encoding, so do **not** pass ``type=`` when encoding JSON
  for a Rust consumer -- serde reads plain JSON numbers.

.. _serde_json: https://docs.rs/serde_json/
.. _rmp-serde: https://docs.rs/rmp-serde/
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

from . import inspect as mi
from ._typescript import _get_class_name, _get_doc

__all__ = ("schema", "schema_components")

# `msgspec.data` integer/float format -> Rust primitive.
_ITYPE_RUST = {
    "int8": "i8", "int16": "i16", "int32": "i32", "int64": "i64",
    "uint8": "u8", "uint16": "u16", "uint32": "u32", "uint64": "u64",
    "int": "i64", "uint": "u64",
}
_FTYPE_RUST = {"float32": "f32", "float64": "f64", "float": "f64"}

# Rust keywords. Most can be used as raw identifiers (`r#name`); a few can't and
# are sanitized + given a `#[serde(rename)]` instead.
_RUST_KEYWORDS = {
    "as", "break", "const", "continue", "crate", "dyn", "else", "enum", "extern",
    "false", "fn", "for", "if", "impl", "in", "let", "loop", "match", "mod",
    "move", "mut", "pub", "ref", "return", "self", "Self", "static", "struct",
    "super", "trait", "true", "type", "unsafe", "use", "where", "while", "async",
    "await", "abstract", "become", "box", "do", "final", "macro", "override",
    "priv", "typeof", "unsized", "virtual", "yield", "try", "gen",
}
_RUST_NON_RAW = {"crate", "self", "Self", "super"}

_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_INDENT = "    "


def _field_ident(name: str) -> tuple[str, str | None]:
    """A Rust field identifier for a wire name, plus a serde rename (or None)."""
    if _IDENT_RE.match(name):
        if name in _RUST_KEYWORDS:
            if name in _RUST_NON_RAW:
                return f"{name}_", name
            return f"r#{name}", None  # raw identifier serializes as `name`
        return name, None
    sanitized = re.sub(r"[^A-Za-z0-9_]", "_", name)
    if not sanitized or sanitized[0].isdigit():
        sanitized = "_" + sanitized
    return sanitized, name


class _Generator:
    def __init__(self, name_map: dict[Any, str], unions: dict[Any, Any]):
        self.name_map = name_map
        self.unions = unions

    # -- inline type references ---------------------------------------------
    def to_ref(self, t: mi.Type) -> str:
        while isinstance(t, mi.Metadata):
            t = t.type

        if hasattr(t, "cls") and t.cls in self.name_map:
            return self.name_map[t.cls]

        if isinstance(t, mi.AbstractStructType):
            return self.name_map[t.cls]
        if isinstance(t, mi.NoneType):
            return "()"
        if isinstance(t, mi.BoolType):
            return "bool"
        if isinstance(t, mi.IntType):
            return _ITYPE_RUST.get(t.itype or "", "i64")
        if isinstance(t, mi.FloatType):
            return _FTYPE_RUST.get(t.ftype or "", "f64")
        if isinstance(t, (mi.StrType, mi.DateTimeType, mi.TimeType, mi.DateType,
                          mi.TimeDeltaType, mi.UUIDType, mi.DecimalType)):
            return "String"
        if isinstance(t, (mi.BytesType, mi.ByteArrayType, mi.MemoryViewType)):
            return "serde_bytes::ByteBuf"
        if isinstance(t, (mi.ListType, mi.VarTupleType)):
            return f"Vec<{self.to_ref(t.item_type)}>"
        if isinstance(t, (mi.SetType, mi.FrozenSetType)):
            return f"std::collections::HashSet<{self.to_ref(t.item_type)}>"
        if isinstance(t, mi.TupleType):
            if not t.item_types:
                return "()"
            inner = ", ".join(self.to_ref(i) for i in t.item_types)
            return f"({inner},)" if len(t.item_types) == 1 else f"({inner})"
        if isinstance(t, (mi.DictType, mi.FrozenDictType)):
            return (
                f"std::collections::HashMap<{self.to_ref(t.key_type)}, "
                f"{self.to_ref(t.value_type)}>"
            )
        if isinstance(t, (mi.AnyType, mi.RawType)):
            return "serde_json::Value"
        if isinstance(t, mi.UnionType):
            return self._union_ref(t)
        if isinstance(t, mi.LiteralType):
            # Rust has no literal types; fall back to the underlying scalar.
            v = t.values[0]
            return "String" if isinstance(v, str) else "i64"
        raise NotImplementedError(
            f"msgspec.rust doesn't support type {t!r} yet"
        )

    def _union_ref(self, t: mi.UnionType) -> str:
        has_none, tagged, others = _partition_union(t)
        if not tagged and has_none and len(others) == 1:
            return f"Option<{self.to_ref(others[0])}>"
        key = _union_key(t)
        if key in self.unions:
            inner = self.name_map[key]
            return f"Option<{inner}>" if has_none else inner
        raise NotImplementedError(
            "msgspec.rust supports optional (`T | None`) and tagged-struct "
            f"unions only, got {t!r}"
        )

    # -- top-level definitions ----------------------------------------------
    def struct_def(self, name: str, t: mi.Type) -> str:
        lines = []
        doc = _get_doc(t)
        if doc:
            lines.append(_docblock(doc))
        lines.append("#[derive(Serialize, Deserialize)]")
        lines.append(f"pub struct {name} {{")
        for f in t.fields:
            ident, rename = _field_ident(f.encode_name)
            ftype = self.to_ref(f.type)
            if not f.required:
                ftype = ftype if ftype.startswith("Option<") else f"Option<{ftype}>"
            if rename is not None:
                lines.append(f'{_INDENT}#[serde(rename = "{rename}")]')
            lines.append(f"{_INDENT}pub {ident}: {ftype},")
        lines.append("}")
        return "\n".join(lines)

    def enum_def(self, name: str, t: mi.EnumType) -> str:
        lines = []
        doc = _get_doc(t)
        if doc:
            lines.append(_docblock(doc))
        lines.append("#[derive(Serialize, Deserialize)]")
        lines.append(f"pub enum {name} {{")
        for member in t.cls:
            vname = _to_pascal(str(member.name))
            lines.append(f'{_INDENT}#[serde(rename = "{member.value}")]')
            lines.append(f"{_INDENT}{vname},")
        lines.append("}")
        return "\n".join(lines)

    def union_def(self, name: str, info: Any) -> str:
        tag_field, variants = info
        lines = ["#[derive(Serialize, Deserialize)]", f'#[serde(tag = "{tag_field}")]']
        lines.append(f"pub enum {name} {{")
        for variant_name, member_name, tag in variants:
            if tag != variant_name:
                lines.append(f'{_INDENT}#[serde(rename = "{tag}")]')
            lines.append(f"{_INDENT}{variant_name}({member_name}),")
        lines.append("}")
        return "\n".join(lines)

    def alias_def(self, name: str, t: mi.AliasType) -> str:
        return f"pub type {name} = {self.to_ref(t.value)};"


def _to_pascal(name: str) -> str:
    """A Rust-idiomatic PascalCase enum-variant identifier, normalizing case
    (``RED`` -> ``Red``, ``dark_red`` -> ``DarkRed``)."""
    parts = re.split(r"[^A-Za-z0-9]+", name)
    out = "".join(p[:1].upper() + p[1:].lower() for p in parts if p)
    if not out:
        out = "Unnamed"
    if out[0].isdigit():
        out = "_" + out
    return out


def _docblock(doc: str) -> str:
    return "\n".join(f"/// {line}".rstrip() for line in doc.splitlines())


def _partition_union(t: mi.UnionType):
    members = []
    for m in t.types:
        while isinstance(m, mi.Metadata):
            m = m.type
        if isinstance(m, mi.AbstractStructType):
            members.extend(m.concrete_union_type.types)
        else:
            members.append(m)
    has_none = any(isinstance(m, mi.NoneType) for m in members)
    tagged = [
        m for m in members
        if isinstance(m, mi.StructType) and m.tag_field is not None
    ]
    others = [m for m in members if not isinstance(m, mi.NoneType) and m not in tagged]
    return has_none, tagged, others


def _union_key(t: mi.UnionType):
    _, tagged, _ = _partition_union(t)
    return frozenset(m.cls for m in tagged)


def _variants_of(tagged: list) -> tuple[str, list]:
    """(tag_field, [(variant_name, struct_name, tag_value), ...]) for a union."""
    tag_field = tagged[0].tag_field
    variants = []
    for m in tagged:
        name = _get_class_name(m.cls)
        variants.append((name, name, m.tag))
    return tag_field, variants


def _collect(type_infos: Iterable[mi.Type]):
    """Gather the nameable components: struct-likes, enums, aliases, and the
    (abstract or anonymous) tagged unions that become Rust enums."""
    components: dict[Any, mi.Type] = {}
    unions: dict[Any, Any] = {}  # key -> (tag_field, variants); also -> name via cls

    def collect(t):
        while isinstance(t, mi.Metadata):
            t = t.type
        if isinstance(t, mi.AbstractStructType):
            if t.cls not in unions:
                _, tagged, _ = _partition_union(t.concrete_union_type)
                unions[t.cls] = _variants_of(tagged)
                for m in tagged:
                    collect(m)
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
        elif isinstance(t, (mi.CollectionType,)):
            collect(t.item_type)
        elif isinstance(t, mi.TupleType):
            for st in t.item_types:
                collect(st)
        elif isinstance(t, (mi.DictType, mi.FrozenDictType)):
            collect(t.key_type)
            collect(t.value_type)
        elif isinstance(t, mi.UnionType):
            has_none, tagged, others = _partition_union(t)
            if tagged:
                key = _union_key(t)
                if key not in unions:
                    unions[key] = _variants_of(tagged)
                for m in tagged:
                    collect(m)
            for m in others:
                collect(m)

    for t in type_infos:
        collect(t)
    return components, unions


def _build_name_map(components: dict[Any, mi.Type], unions: dict[Any, Any]) -> dict[Any, str]:
    def to_pascal(name):
        name = re.sub(r"[^A-Za-z0-9_]", "_", name)
        if name and name[0].islower():
            name = name[0].upper() + name[1:]
        if name and name[0].isdigit():
            name = "_" + name
        return name

    name_map: dict[Any, str] = {}
    for cls in components:
        name_map[cls] = to_pascal(_get_class_name(cls))
    for key, (_, variants) in unions.items():
        if not isinstance(key, frozenset):
            # abstract struct: name after the class
            name_map[key] = to_pascal(_get_class_name(key))
        else:
            # anonymous tagged union: join the variant names
            name_map[key] = "Or".join(v[0] for v in variants) or "Union"
    return name_map


def schema_components(
    types: Iterable[Any],
) -> tuple[tuple[str, ...], dict[str, str]]:
    """Generate Rust definitions for one or more types.

    Returns ``(refs, components)`` - a tuple of inline Rust type references (one
    per input type) and a mapping of name to the Rust source defining each
    nameable component.
    """
    type_infos = mi.multi_type_info(types, aliases=True)
    components, unions = _collect(type_infos)
    name_map = _build_name_map(components, unions)
    gen = _Generator(name_map, unions)

    refs = tuple(gen.to_ref(t) for t in type_infos)

    defs: dict[str, str] = {}
    for cls, t in components.items():
        name = name_map[cls]
        if isinstance(t, mi.EnumType):
            defs[name] = gen.enum_def(name, t)
        elif isinstance(t, mi.AliasType):
            defs[name] = gen.alias_def(name, t)
        else:
            defs[name] = gen.struct_def(name, t)
    for key, info in unions.items():
        name = name_map[key]
        defs[name] = gen.union_def(name, info)
    return refs, defs


def schema(type: Any) -> str:
    """Generate Rust type definitions for a given type.

    ``Struct`` types become ``struct``\\ s, tagged-union / abstract structs
    become internally-tagged ``enum``\\ s, and everything else maps onto the
    obvious Rust type. If the top-level ``type`` is not itself a nameable
    component (e.g. ``list[Point]``) a ``pub type Root = …;`` alias is emitted to
    name it.
    """
    (root,), components = schema_components((type,))
    parts = list(components.values())
    if root not in components:
        parts.append(f"pub type Root = {root};")
    return "\n\n".join(parts) + "\n"
