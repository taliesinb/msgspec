from __future__ import annotations

import re
import textwrap
from collections.abc import Iterable
from typing import Any

from . import inspect as mi

__all__ = ("schema", "schema_components")

# Matches a valid (unquoted) TypeScript identifier.
_IDENT_RE = re.compile(r"^[A-Za-z_$][A-Za-z0-9_$]*$")

_INDENT = "  "


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
    type_infos = mi.multi_type_info(types)

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
    definition (Struct, Dataclass, NamedTuple, TypedDict, and Enum types)."""
    components: dict[Any, mi.Type] = {}

    def collect(t):
        if isinstance(
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

        # Nameable components are referenced by name.
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
        if isinstance(t, mi.EnumType):
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
