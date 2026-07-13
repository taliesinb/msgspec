"""Generate self-contained JavaScript MessagePack codecs from msgspec types.

`msgspec.javascript.codec` emits a dependency-free ES module: per-struct
``encodeX``/``decodeX`` functions plus a top-level ``encode(value)`` /
``decode(bytes)``, backed by a small MessagePack reader/writer inlined into the
module (no ``@msgpack/msgpack`` dependency). The output is byte-for-byte
identical to `msgspec.msgpack`.

Unlike the TypeScript codec, this targets plain JavaScript: there are no type
annotations, and 64-bit integer types (``Int``/``Int64``/``UInt64``) round-trip
as native ``bigint`` with no safe-integer narrowing.
"""

from __future__ import annotations

import textwrap
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from . import inspect as mi
from ._msgpack_runtime_js import RUNTIME as _MSGPACK_RUNTIME
from ._typescript import (
    _build_name_map,
    _literal_value,
    _prop_name,
    _str,
    _str_inner,
)

__all__ = ("codec", "bundle_path")

_INDENT = "  "


def bundle_path() -> Path:
    """The on-disk directory of the ``@msgspec/msgpack`` JavaScript bundle.

    This is the standalone JS package (``package.json``, ``src/msgpack.mjs``,
    ...) that `codec` inlines - useful for wiring the runtime into a
    JavaScript project's build/bundle step (e.g. as a ``file:`` npm dependency,
    or by copying ``src/msgpack.mjs``).

    In a source checkout or editable install it resolves to the repository's
    top-level ``javascript/`` directory (the files you have checked out). In an
    installed wheel it resolves to the copy bundled alongside the package.
    """
    here = Path(__file__).resolve().parent
    checkout = here.parents[1] / "javascript"
    if (checkout / "src" / "msgpack.mjs").is_file():
        return checkout
    return here / "javascript_bundle"

# The `msgspec.data` integer formats that round-trip as a JS `bigint` (via the
# runtime's `int64` reader/writer) rather than a `number`.
_BIGINT_ITYPES = frozenset({"int64", "uint64", "int", "uint"})


def _int_method(t: mi.IntType) -> str:
    """The runtime method suffix for an `IntType`: `int64` for the bigint
    formats, else `int`."""
    return "int64" if t.itype in _BIGINT_ITYPES else "int"


def _is_bigint(t: mi.Type) -> bool:
    return isinstance(t, mi.IntType) and t.itype in _BIGINT_ITYPES


def _float_method(t: mi.FloatType) -> str:
    """`float32` narrows to a 5-byte msgpack float32; everything else is float64."""
    return "float32" if t.ftype == "float32" else "float"


def _js_bool(b: bool) -> str:
    return "true" if b else "false"


def _json_identity(t: mi.Type) -> bool:
    """Whether a value of this type is already JSON-ready (no transform needed
    on encode, none on decode) - so it can pass straight through
    ``JSON.stringify`` / ``JSON.parse``."""
    t = _unwrap(t)
    if isinstance(t, mi.IntType):
        return not _is_bigint(t)  # number ints are identity; bigints are hex
    if isinstance(
        t,
        (
            mi.NoneType,
            mi.BoolType,
            mi.StrType,
            mi.FloatType,
            mi.AnyType,
            mi.RawType,
            mi.EnumType,
            mi.LiteralType,
        ),
    ):
        return True
    if isinstance(t, (mi.ListType, mi.VarTupleType)):
        return _json_identity(t.item_type)
    if isinstance(t, (mi.DictType, mi.FrozenDictType)):
        return _json_identity(t.value_type)
    if isinstance(t, mi.TupleType):
        return all(_json_identity(i) for i in t.item_types)
    # bytes (base64), sets (Set<->array), structs, unions, tensors: transformed.
    return False

# Struct-like components that get a generated encode/decode function pair.
_FUNC_TYPES = (
    mi.StructType,
    mi.TypedDictType,
    mi.DataclassType,
    mi.NamedTupleType,
)


def _ind(s: str, n: int = 1) -> str:
    return textwrap.indent(s, _INDENT * n)


def _unwrap(t: mi.Type) -> mi.Type:
    """Strip metadata and expand an abstract struct to its concrete union."""
    while isinstance(t, mi.Metadata):
        t = t.type
    if isinstance(t, mi.AbstractStructType):
        return _unwrap(t.concrete_union_type)
    return t


def _enum_kind(t: mi.EnumType) -> str | None:
    values = [m.value for m in t.cls]
    if values and all(isinstance(v, str) for v in values):
        return "str"
    if values and all(isinstance(v, int) and not isinstance(v, bool) for v in values):
        return "int"
    return None


def _collect_component_types(type_infos: Iterable[mi.Type]) -> dict[Any, mi.Type]:
    """Find every struct-like type (which needs an encode/decode function) in the
    type tree. Abstract structs are transparent - their concrete descendants are
    collected instead."""
    components: dict[Any, mi.Type] = {}

    def collect(t):
        t = _unwrap(t)  # note: also flattens abstract structs to their union
        if isinstance(t, _FUNC_TYPES):
            if t.cls not in components:
                components[t.cls] = t
                for f in t.fields:
                    collect(f.type)
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


class _CodecGenerator:
    def __init__(self, name_map: dict[Any, str]):
        self.name_map = name_map
        self._counter = 0

    def _fresh(self) -> str:
        self._counter += 1
        return f"_e{self._counter}"

    # -- encode -------------------------------------------------------------
    def enc(self, t: mi.Type, v: str) -> str:
        """A statement (possibly a block) writing JS expression ``v`` into ``w``."""
        t = _unwrap(t)
        if isinstance(t, mi.NoneType):
            return "w.nil();"
        if isinstance(t, mi.BoolType):
            return f"w.bool({v});"
        if isinstance(t, mi.IntType):
            return f"w.{_int_method(t)}({v});"
        if isinstance(t, mi.FloatType):
            return f"w.{_float_method(t)}({v});"
        if isinstance(t, mi.StrType):
            return f"w.str({v});"
        if isinstance(t, (mi.BytesType, mi.ByteArrayType, mi.MemoryViewType)):
            return f"w.bin({v});"
        if isinstance(t, (mi.AnyType, mi.RawType)):
            return f"w.value({v});"
        if isinstance(t, mi.EnumType):
            method = {"str": "str", "int": "int"}.get(_enum_kind(t) or "", "value")
            return f"w.{method}({v});"
        if isinstance(t, mi.LiteralType):
            return f"w.value({v});"
        if isinstance(t, (mi.ListType, mi.VarTupleType)):
            e = self._fresh()
            inner = self.enc(t.item_type, e)
            return (
                f"w.arrayHeader({v}.length);\n"
                f"for (const {e} of {v}) {{\n{_ind(inner)}\n}}"
            )
        if isinstance(t, (mi.SetType, mi.FrozenSetType)):
            e = self._fresh()
            inner = self.enc(t.item_type, e)
            return (
                f"w.arrayHeader({v}.size);\n"
                f"for (const {e} of {v}) {{\n{_ind(inner)}\n}}"
            )
        if isinstance(t, mi.TupleType):
            lines = [f"w.arrayHeader({len(t.item_types)});"]
            for i, it in enumerate(t.item_types):
                lines.append(self.enc(it, f"{v}[{i}]"))
            return "\n".join(lines)
        if isinstance(t, (mi.DictType, mi.FrozenDictType)):
            k, ks = self._fresh(), self._fresh()
            kenc = self._enc_key(t.key_type, k)
            venc = self.enc(t.value_type, f"{v}[{k}]")
            return (
                "{\n"
                f"{_INDENT}const {ks} = Object.keys({v});\n"
                f"{_INDENT}w.mapHeader({ks}.length);\n"
                f"{_INDENT}for (const {k} of {ks}) {{\n"
                f"{_ind(kenc, 2)}\n{_ind(venc, 2)}\n"
                f"{_INDENT}}}\n"
                "}"
            )
        if isinstance(t, _FUNC_TYPES):
            return f"encode{self.name_map[t.cls]}(w, {v});"
        if isinstance(t, mi.UnionType):
            return self._enc_union(t, v)
        raise NotImplementedError(
            f"msgspec.javascript codec doesn't support type {t!r} yet"
        )

    def _enc_key(self, t: mi.Type, k: str) -> str:
        """Encode a dict key. Object keys are JS strings, so int keys are
        converted back to a number/bigint to match msgspec's on-wire form."""
        t = _unwrap(t)
        if isinstance(t, mi.IntType):
            if _int_method(t) == "int64":
                return f"w.int64(BigInt({k}));"
            return f"w.int(Number({k}));"
        return f"w.str({k});"

    def _enc_union(self, t: mi.UnionType, v: str) -> str:
        has_none, tagged, others = self._partition_union(t)
        if not tagged:
            if has_none and len(others) == 1:
                inner = self.enc(others[0], v)
                return (
                    f"if ({v} === null || {v} === undefined) {{\n"
                    f"{_ind('w.nil();')}\n}} else {{\n{_ind(inner)}\n}}"
                )
            return self._enc_scalar_union(v, has_none, others)
        tag_field = tagged[0].tag_field
        cases = "\n".join(
            f"{_INDENT}case {_literal_value(m.tag)}: encode{self.name_map[m.cls]}(w, {v}); break;"
            for m in tagged
        )
        switch = (
            f"switch ({v}[{_str(tag_field)}]) {{\n{cases}\n"
            f'{_INDENT}default: throw new Error("unexpected tag for {_str_inner(tag_field)} union");\n}}'
        )
        if has_none:
            return (
                f"if ({v} === null || {v} === undefined) {{\n{_ind('w.nil();')}\n}} "
                f"else {{\n{_ind(switch)}\n}}"
            )
        return switch

    def _enc_scalar_union(self, v: str, has_none: bool, others: list) -> str:
        """Encode a non-struct union (e.g. `Scalar` = Int | Float | Bool) by
        dispatching on the JavaScript runtime type of the value."""
        has_bigint = any(_is_bigint(m) for m in others)
        has_float = any(isinstance(m, mi.FloatType) for m in others)
        has_int = any(
            isinstance(m, mi.IntType) and not _is_bigint(m) for m in others
        )
        has_bool = any(isinstance(m, mi.BoolType) for m in others)
        has_str = any(isinstance(m, mi.StrType) for m in others)
        has_bytes = any(
            isinstance(m, (mi.BytesType, mi.ByteArrayType, mi.MemoryViewType))
            for m in others
        )
        branches = []
        if has_none:
            branches.append((f"{v} === null || {v} === undefined", "w.nil();"))
        if has_bigint:
            branches.append((f'typeof {v} === "bigint"', f"w.int64({v});"))
        if has_int and has_float:
            branches.append(
                (f'typeof {v} === "number"',
                 f"if (Number.isInteger({v})) w.int({v}); else w.float({v});")
            )
        elif has_float:
            branches.append((f'typeof {v} === "number"', f"w.float({v});"))
        elif has_int:
            branches.append((f'typeof {v} === "number"', f"w.int({v});"))
        if has_bool:
            branches.append((f'typeof {v} === "boolean"', f"w.bool({v});"))
        if has_str:
            branches.append((f'typeof {v} === "string"', f"w.str({v});"))
        if has_bytes:
            branches.append((f"{v} instanceof Uint8Array", f"w.bin({v});"))
        if not branches:
            raise NotImplementedError(
                f"msgspec.javascript codec can't encode union with members {others!r}"
            )
        out = ""
        for i, (cond, body) in enumerate(branches):
            kw = "if" if i == 0 else " else if"
            out += f"{kw} ({cond}) {{\n{_ind(body)}\n}}"
        throw = _ind('throw new Error("unencodable scalar union value");')
        out += " else {\n" + throw + "\n}"
        return out

    # -- decode -------------------------------------------------------------
    def dec(self, t: mi.Type) -> str:
        """An expression reading one value of type ``t`` from reader ``r``."""
        t = _unwrap(t)
        if isinstance(t, mi.NoneType):
            return "(r.skip(), null)"
        if isinstance(t, mi.BoolType):
            return "r.bool()"
        if isinstance(t, mi.IntType):
            return f"r.{_int_method(t)}()"
        if isinstance(t, mi.FloatType):
            return "r.float()"
        if isinstance(t, mi.StrType):
            return "r.str()"
        if isinstance(t, (mi.BytesType, mi.ByteArrayType, mi.MemoryViewType)):
            return "r.bin()"
        if isinstance(t, (mi.AnyType, mi.RawType)):
            return "r.value()"
        if isinstance(t, mi.EnumType):
            method = {"str": "str", "int": "int"}.get(_enum_kind(t), "value")
            return f"r.{method}()"
        if isinstance(t, mi.LiteralType):
            return "r.value()"
        if isinstance(t, (mi.ListType, mi.VarTupleType)):
            return f"decArray(r, (r) => {self.dec(t.item_type)})"
        if isinstance(t, (mi.SetType, mi.FrozenSetType)):
            return f"decSet(r, (r) => {self.dec(t.item_type)})"
        if isinstance(t, (mi.DictType, mi.FrozenDictType)):
            return (
                f"decMap(r, (r) => {self._dec_key(t.key_type)}, "
                f"(r) => {self.dec(t.value_type)})"
            )
        if isinstance(t, mi.TupleType):
            elems = ", ".join(self.dec(it) for it in t.item_types)
            return f"(r.arrayHeader(), [{elems}])"
        if isinstance(t, _FUNC_TYPES):
            return f"decode{self.name_map[t.cls]}(r)"
        if isinstance(t, mi.UnionType):
            return self._dec_union(t)
        raise NotImplementedError(
            f"msgspec.javascript codec doesn't support type {t!r} yet"
        )

    def _dec_key(self, t: mi.Type) -> str:
        """Decode a dict key back into a JS value (object keys stringify anyway)."""
        t = _unwrap(t)
        if isinstance(t, mi.IntType):
            return f"r.{_int_method(t)}()"
        return "r.str()"

    def _dec_scalar_union(self, has_none: bool, others: list) -> str:
        """Decode a non-struct union by peeking the next MessagePack tag byte."""
        lines = []
        if has_none:
            lines.append("if (r.tryNil()) return null;")
        lines.append("const _t = r.b[r.p];")
        if any(isinstance(m, mi.FloatType) for m in others):
            lines.append("if (_t === 0xca || _t === 0xcb) return r.float();")
        if any(isinstance(m, mi.BoolType) for m in others):
            lines.append("if (_t === 0xc2 || _t === 0xc3) return r.bool();")
        if any(isinstance(m, mi.StrType) for m in others):
            lines.append(
                "if ((_t >= 0xa0 && _t <= 0xbf) || _t === 0xd9 || _t === 0xda || _t === 0xdb) return r.str();"
            )
        if any(
            isinstance(m, (mi.BytesType, mi.ByteArrayType, mi.MemoryViewType))
            for m in others
        ):
            lines.append("if (_t === 0xc4 || _t === 0xc5 || _t === 0xc6) return r.bin();")
        if any(_is_bigint(m) for m in others):
            lines.append("return r.int64();")
        elif any(isinstance(m, mi.IntType) for m in others):
            lines.append("return r.int();")
        else:
            lines.append('throw new Error("undecodable scalar union");')
        body = "\n".join(lines)
        return f"(() => {{\n{_ind(body)}\n}})()"

    def _dec_union(self, t: mi.UnionType) -> str:
        has_none, tagged, others = self._partition_union(t)
        if not tagged:
            if has_none and len(others) == 1:
                return f"(r.tryNil() ? null : {self.dec(others[0])})"
            return self._dec_scalar_union(has_none, others)
        tag_field = tagged[0].tag_field
        array_like = tagged[0].array_like
        read_tag = "r.str()" if isinstance(tagged[0].tag, str) else "r.int()"
        cases = "\n".join(
            f"{_INDENT}case {_literal_value(m.tag)}: return decode{self.name_map[m.cls]}(r);"
            for m in tagged
        )
        none_guard = "if (r.tryNil()) return null;\n" if has_none else ""
        if array_like:
            scan = f"const _s = r.p;\nr.arrayHeader();\nconst _t = {read_tag};\nr.p = _s;"
        else:
            scan = (
                "const _s = r.p;\n"
                "const _n = r.mapHeader();\n"
                "let _t;\n"
                "for (let _i = 0; _i < _n; _i++) {\n"
                f"{_INDENT}if (r.str() === {_str(tag_field)}) {{ _t = {read_tag}; break; }}\n"
                f"{_INDENT}r.skip();\n"
                "}\n"
                "r.p = _s;"
            )
        body = (
            f"{none_guard}{scan}\n"
            f"switch (_t) {{\n{cases}\n"
            f'{_INDENT}default: throw new Error("unexpected tag for {_str_inner(tag_field)} union");\n}}'
        )
        return f"(() => {{\n{_ind(body)}\n}})()"

    def _partition_union(self, t: mi.UnionType):
        members = []
        for m in t.types:
            m = _unwrap(m)
            if isinstance(m, mi.UnionType):
                members.extend(_unwrap(x) for x in m.types)
            else:
                members.append(m)
        has_none = any(isinstance(m, mi.NoneType) for m in members)
        tagged = [
            m
            for m in members
            if isinstance(m, mi.StructType) and m.tag_field is not None
        ]
        others = [m for m in members if not isinstance(m, mi.NoneType) and m not in tagged]
        return has_none, tagged, others

    # -- per-component definitions -----------------------------------------
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

    def _tag_write(self, t: mi.StructType) -> str:
        key = f"w.str({_str(t.tag_field)});"
        if isinstance(t.tag, str):
            val = f"w.str({_literal_value(t.tag)});"
        else:
            val = f"w.int({t.tag});"
        return f"{key}\n{val}"

    def _tagged(self, t: mi.Type) -> bool:
        return isinstance(t, mi.StructType) and t.tag_field is not None

    def _enc_object_def(self, name: str, t: mi.Type) -> str:
        n = len(t.fields) + (1 if self._tagged(t) else 0)
        lines = [f"w.mapHeader({n});"]
        if self._tagged(t):
            lines.append(self._tag_write(t))
        for f in t.fields:
            lines.append(f"w.str({_str(f.encode_name)});")
            lines.append(self.enc(f.type, f"v[{_str(f.encode_name)}]"))
        body = "\n".join(lines)
        return f"export function encode{name}(w, v) {{\n{_ind(body)}\n}}"

    def _enc_array_def(self, name: str, t: mi.Type) -> str:
        n = len(t.fields) + (1 if self._tagged(t) else 0)
        lines = [f"w.arrayHeader({n});"]
        if self._tagged(t):
            if isinstance(t.tag, str):
                lines.append(f"w.str({_literal_value(t.tag)});")
            else:
                lines.append(f"w.int({t.tag});")
        for f in t.fields:
            lines.append(self.enc(f.type, f"v[{_str(f.encode_name)}]"))
        body = "\n".join(lines)
        return f"export function encode{name}(w, v) {{\n{_ind(body)}\n}}"

    def _dec_object_def(self, name: str, t: mi.Type) -> str:
        cases = []
        for f in t.fields:
            cases.append(
                f"case {_str(f.encode_name)}: o[{_str(f.encode_name)}] = {self.dec(f.type)}; break;"
            )
        if self._tagged(t):
            cases.append(f"case {_str(t.tag_field)}: r.skip(); break;")
        cases.append("default: r.skip();")
        switch = "switch (r.str()) {\n" + _ind("\n".join(cases)) + "\n}"
        seed = (
            f"const o = {{ {_prop_name(t.tag_field)}: {_literal_value(t.tag)} }};\n"
            if self._tagged(t)
            else "const o = {};\n"
        )
        body = (
            "const n = r.mapHeader();\n"
            f"{seed}"
            "for (let i = 0; i < n; i++) {\n"
            f"{_ind(switch)}\n"
            "}\n"
            "return o;"
        )
        return f"export function decode{name}(r) {{\n{_ind(body)}\n}}"

    def _dec_array_def(self, name: str, t: mi.Type) -> str:
        if self._tagged(t):
            seed = f"const o = {{ {_prop_name(t.tag_field)}: {_literal_value(t.tag)} }};"
        else:
            seed = "const o = {};"
        lines = ["const n = r.arrayHeader();", "let i = 0;", seed]
        if self._tagged(t):
            lines.append("if (i < n) { r.skip(); i++; }")
        for f in t.fields:
            lines.append(f"o[{_str(f.encode_name)}] = {self.dec(f.type)}; i++;")
        lines.append("while (i < n) { r.skip(); i++; }")
        lines.append("return o;")
        body = "\n".join(lines)
        return f"export function decode{name}(r) {{\n{_ind(body)}\n}}"

    # -- JSON codec (structural transform over JSON.stringify/parse) ---------
    def json_enc(self, t: mi.Type, v: str) -> str:
        """A JSON-ready expression for typed value ``v`` (before stringify)."""
        t = _unwrap(t)
        if _json_identity(t):
            return v
        if _is_bigint(t):
            signed = t.itype in ("int64", "int")
            return f"encHexInt({v}, {_js_bool(signed)}, false)"
        if isinstance(t, (mi.BytesType, mi.ByteArrayType, mi.MemoryViewType)):
            return f"b64encode({v})"
        if isinstance(t, (mi.ListType, mi.VarTupleType)):
            return f"{v}.map((_e) => {self.json_enc(t.item_type, '_e')})"
        if isinstance(t, (mi.SetType, mi.FrozenSetType)):
            inner = self.json_enc(t.item_type, "_e")
            return f"[...{v}]" if inner == "_e" else f"[...{v}].map((_e) => {inner})"
        if isinstance(t, (mi.DictType, mi.FrozenDictType)):
            vx = self.json_enc(t.value_type, "_x")
            return (
                f"Object.fromEntries(Object.entries({v})"
                f".map(([_k, _x]) => [_k, {vx}]))"
            )
        if isinstance(t, mi.TupleType):
            return "[" + ", ".join(
                self.json_enc(it, f"{v}[{i}]") for i, it in enumerate(t.item_types)
            ) + "]"
        if isinstance(t, _FUNC_TYPES):
            return f"encodeJson{self.name_map[t.cls]}({v})"
        if isinstance(t, mi.UnionType):
            return self._json_union(t, v, encode=True)
        raise NotImplementedError(
            f"msgspec.javascript json codec doesn't support type {t!r} yet"
        )

    def json_dec(self, t: mi.Type, o: str) -> str:
        """An expression producing the typed value from parsed JSON ``o``."""
        t = _unwrap(t)
        if _json_identity(t):
            return o
        if _is_bigint(t):
            return f"decHexInt({o})"
        if isinstance(t, (mi.BytesType, mi.ByteArrayType, mi.MemoryViewType)):
            return f"b64decode({o})"
        if isinstance(t, (mi.ListType, mi.VarTupleType)):
            return f"{o}.map((_e) => {self.json_dec(t.item_type, '_e')})"
        if isinstance(t, (mi.SetType, mi.FrozenSetType)):
            inner = self.json_dec(t.item_type, "_e")
            return f"new Set({o})" if inner == "_e" else f"new Set({o}.map((_e) => {inner}))"
        if isinstance(t, (mi.DictType, mi.FrozenDictType)):
            vx = self.json_dec(t.value_type, "_x")
            return (
                f"Object.fromEntries(Object.entries({o})"
                f".map(([_k, _x]) => [_k, {vx}]))"
            )
        if isinstance(t, mi.TupleType):
            return "[" + ", ".join(
                self.json_dec(it, f"{o}[{i}]") for i, it in enumerate(t.item_types)
            ) + "]"
        if isinstance(t, _FUNC_TYPES):
            return f"decodeJson{self.name_map[t.cls]}({o})"
        if isinstance(t, mi.UnionType):
            return self._json_union(t, o, encode=False)
        raise NotImplementedError(
            f"msgspec.javascript json codec doesn't support type {t!r} yet"
        )

    def _json_union(self, t: mi.UnionType, expr: str, encode: bool) -> str:
        has_none, tagged, others = self._partition_union(t)

        def wrap_none(inner: str) -> str:
            if not has_none:
                return inner
            return f"({expr} === null || {expr} === undefined ? null : {inner})"

        if tagged:
            verb = "encodeJson" if encode else "decodeJson"
            tag_field = tagged[0].tag_field
            cases = "\n".join(
                f"{_INDENT}case {_literal_value(m.tag)}: return {verb}{self.name_map[m.cls]}(_u);"
                for m in tagged
            )
            iife = (
                "((_u) => {\n"
                f"{_INDENT}switch (_u[{_str(tag_field)}]) {{\n{cases}\n{_INDENT}}}\n"
                f'{_INDENT}throw new Error("unexpected tag for {_str_inner(tag_field)} union");\n'
                f"}})({expr})"
            )
            return wrap_none(iife)

        if has_none and len(others) == 1 and not _is_bigint(others[0]):
            inner = (
                self.json_enc(others[0], expr)
                if encode
                else self.json_dec(others[0], expr)
            )
            return wrap_none(inner)

        bigints = [m for m in others if _is_bigint(m)]
        if bigints:
            signed = bigints[0].itype in ("int64", "int")
            force = any(isinstance(m, mi.FloatType) for m in others)  # Int|Float
            has_str = any(isinstance(m, mi.StrType) for m in others)
            if encode:
                inner = (
                    f'(typeof {expr} === "bigint" '
                    f"? encHexInt({expr}, {_js_bool(signed)}, {_js_bool(force)}) : {expr})"
                )
            elif has_str:
                inner = (
                    f'(typeof {expr} === "string" '
                    f"? (/^[+-]?0x/.test({expr}) ? decHexInt({expr}) : {expr}) : {expr})"
                )
            else:
                inner = f'(typeof {expr} === "string" ? decHexInt({expr}) : {expr})'
            return wrap_none(inner)

        # A plain scalar union (str | number | bool) is JSON-native.
        return wrap_none(expr)

    def json_enc_def(self, name: str, t: mi.Type) -> str:
        if isinstance(t, mi.NamedTupleType) or (
            isinstance(t, mi.StructType) and t.array_like
        ):
            return self._json_enc_array_def(name, t)
        return self._json_enc_object_def(name, t)

    def json_dec_def(self, name: str, t: mi.Type) -> str:
        if isinstance(t, mi.NamedTupleType) or (
            isinstance(t, mi.StructType) and t.array_like
        ):
            return self._json_dec_array_def(name, t)
        return self._json_dec_object_def(name, t)

    def _json_enc_object_def(self, name: str, t: mi.Type) -> str:
        lines = []
        if self._tagged(t):
            lines.append(f"{_prop_name(t.tag_field)}: {_literal_value(t.tag)},")
        for f in t.fields:
            lines.append(
                f"{_prop_name(f.encode_name)}: {self.json_enc(f.type, f'v[{_str(f.encode_name)}]')},"
            )
        body = "return {\n" + _ind("\n".join(lines)) + "\n};"
        return f"export function encodeJson{name}(v) {{\n{_ind(body)}\n}}"

    def _json_dec_object_def(self, name: str, t: mi.Type) -> str:
        lines = []
        if self._tagged(t):
            lines.append(f"{_prop_name(t.tag_field)}: {_literal_value(t.tag)},")
        for f in t.fields:
            lines.append(
                f"{_prop_name(f.encode_name)}: {self.json_dec(f.type, f'o[{_str(f.encode_name)}]')},"
            )
        body = "return {\n" + _ind("\n".join(lines)) + "\n};"
        return f"export function decodeJson{name}(o) {{\n{_ind(body)}\n}}"

    def _json_enc_array_def(self, name: str, t: mi.Type) -> str:
        elems = []
        if self._tagged(t):
            elems.append(_literal_value(t.tag))
        for f in t.fields:
            elems.append(self.json_enc(f.type, f"v[{_str(f.encode_name)}]"))
        body = "return [" + ", ".join(elems) + "];"
        return f"export function encodeJson{name}(v) {{\n{_ind(body)}\n}}"

    def _json_dec_array_def(self, name: str, t: mi.Type) -> str:
        off = 1 if self._tagged(t) else 0
        lines = []
        if self._tagged(t):
            lines.append(f"{_prop_name(t.tag_field)}: {_literal_value(t.tag)},")
        for i, f in enumerate(t.fields):
            lines.append(
                f"{_prop_name(f.encode_name)}: {self.json_dec(f.type, f'o[{i + off}]')},"
            )
        body = "return {\n" + _ind("\n".join(lines)) + "\n};"
        return f"export function decodeJson{name}(o) {{\n{_ind(body)}\n}}"


def codec(type: Any, *, msgpack: bool = True, json: bool = True) -> str:
    """Generate a self-contained JavaScript codec for ``type``.

    The output is a dependency-free ES module. Depending on the flags it exports
    a ``msgpack`` and/or a ``json`` namespace, each with ``encode``/``decode``::

        import { msgpack, json } from "./codec.js";
        const bytes = msgpack.encode(value);   // -> Uint8Array
        const text  = json.encode(value);      // -> string

    Both are byte/format-compatible with a **type-directed** ``msgspec`` encoder
    (``msgspec.msgpack.encode(v, type=T)`` / ``msgspec.json.encode(v, type=T)``).
    64-bit / generic integer types (``Int``/``Int64``/``UInt``/``UInt64``)
    round-trip as native ``bigint`` - as hex strings in JSON when they'd exceed
    the JS safe-integer range; ``bytes`` are base64 in JSON; ``Float32`` narrows
    to a 5-byte msgpack float32.

    Parameters
    ----------
    type : type
        The type to generate a codec for.
    msgpack : bool, optional
        Emit the ``msgpack`` namespace (default ``True``).
    json : bool, optional
        Emit the ``json`` namespace (default ``True``).

    Returns
    -------
    str
        The generated JavaScript source.
    """
    if not (msgpack or json):
        raise ValueError("at least one of `msgpack` / `json` must be enabled")

    (root,) = mi.multi_type_info([type])
    component_types = _collect_component_types([root])
    name_map = _build_name_map(component_types)
    gen = _CodecGenerator(name_map)

    parts = [_MSGPACK_RUNTIME.strip()]

    if msgpack:
        for cls, t in component_types.items():
            parts.append(gen.enc_def(name_map[cls], t))
            parts.append(gen.dec_def(name_map[cls], t))
        root_enc = gen.enc(root, "value")
        parts.append(
            "export const msgpack = {\n"
            f"{_INDENT}encode(value) {{\n"
            f"{_INDENT}{_INDENT}const w = new Writer();\n"
            f"{_ind(root_enc, 2)}\n"
            f"{_INDENT}{_INDENT}return w.bytes();\n"
            f"{_INDENT}}},\n"
            f"{_INDENT}decode(bytes) {{\n"
            f"{_INDENT}{_INDENT}const r = new Reader(bytes);\n"
            f"{_INDENT}{_INDENT}return {gen.dec(root)};\n"
            f"{_INDENT}}},\n"
            "};"
        )

    if json:
        for cls, t in component_types.items():
            parts.append(gen.json_enc_def(name_map[cls], t))
            parts.append(gen.json_dec_def(name_map[cls], t))
        parts.append(
            "export const json = {\n"
            f"{_INDENT}encode(value) {{ return JSON.stringify({gen.json_enc(root, 'value')}); }},\n"
            f"{_INDENT}decode(text) {{ const o = JSON.parse(text); return {gen.json_dec(root, 'o')}; }},\n"
            "};"
        )

    return "\n\n".join(parts) + "\n"
