"""Tokenizer for the JS/TS-style parameter-list destructuring grammar.

These describe the class constructor argument structure of the class definition
emitted by `msgspec.{java,type}script.schema` for a given msgspec.Struct.

# SEMANTICS

Basic signatures:
    '(*, **)`
        use positional args for required fields, and trailing object for the
        optional fields (if any)
    `{**}`
        use an object argument for all fields
    `(*)`
        use positional args for all fields
    `(foo, baz)`
        explicit argument order (must exhaust all required fields)
    `(foo, bar, *)`
        non-mentioned fields are expanded where `*` occurs

If both '*' and '**' occur in a signature, they mean 'required' and 'optional'
respectively, but if only one occurs, it means 'all'. `**` is permitted in
`( )`, where it means `{**}`. A empty {**} is elided.

Each INDIVIDUAL signature argument, or element of a `[ ]`, specification can be:
    `my_field`
        obtains Struct field `my_field` from this argument
    `[ XXX ]` or `{ XXX }`
        a nested iterable or object pattern that will bind Struct fields
    `XXX = my_default`
        ... with a default value if not provided.
    `*`
        expands to (unbound) Struct fields in the order they occur.
    `...my_seq_field`
        binds remaining user-provided elems as Struct field `my_seq_field`

Each INDIVIDUAL element of `{ }` can be:
    `my_field`
        obtains Struct field `my_field` from object key `my_field`
    `key: my_field`
        obtains Struct field `my_field` from object key `key`
    `key: [ XXX ]` or `key: { XXX }`
        nested pattern that will bind Struct fields
    `**`
        expands to (unbound) Struct fields in the order they occur.
    `...my_dict_field`
        binds remaining user-provided items as Struct field `my_dict_field`

# USAGE

The user provides these signature strings via a new  `js_constructor` kwarg
to a `msgspec.Struct` class definition.
"""

from __future__ import annotations

import re
from typing import Literal, NoReturn, Sequence

from msgspec import Struct

__all__ = [
    'parse_constructor_spec',
    'expand_signature',
]

# ------

ObjKey = str
MsgKey = str
JsLiteral = str

# `Pat`/`Comp` are defined below the classes they union over.

# ---

class Signature(Struct):
    args: Seq
    keys: list[MsgKey]

# ---

class Atomic(Struct): ...


class Symbol(Struct):
    name:    MsgKey
    default: JsLiteral | None = None


class Item(Struct):
    key:     ObjKey
    val:     Symbol | Comp


class Splice(Struct):
    kind: Literal['all', 'req', 'opt']


class Compound(Struct):
    elems:   list[Pat]
    splat:   MsgKey | None = None
    default: JsLiteral | None = None

class Seq(Compound): ...
class Obj(Compound): ...

Pat = Seq | Obj | Symbol | Splice | Item
Comp = Seq | Obj

DELIM = {Seq: '[]', Obj: '{}'}

# ----

comp = re.compile
SPLAT   = comp(r"(?=\.)\.\.")
IDENT   = comp(r"[A-Za-z_$][A-Za-z0-9_$]*")
LIT     = comp(r"-?\d+\.\d+|-?\d+|[a-z]+|'(?:[^'\\]|\\.)*'|\[\]|\{\}")
SPACE   = comp(r' +')
CHAR    = comp(r' *(.)')

# ------------------------------

Regex = re.Pattern[str]

def parse_constructor_spec(spec: str) -> Signature:

    pos = 0
    s_opt = Splice('opt')
    s_req = Splice('req')
    s_used: list[Splice] = []
    keys: list[MsgKey] = []

    def add_key(k: MsgKey):
        if k in keys:
            raise error(f"Field {k!r} bound more than once")
        keys.append(k)

    def add_splice(k: Literal['opt', 'req']):
        s = s_opt if k == 'opt' else s_req
        if s in s_used:
            raise error(f"{k!r} spliced more than once")
        s_used.append(s)
        return s

    def match(regex: Regex) -> str | None:
        nonlocal pos
        if jump := SPACE.match(spec, pos):
            pos = jump.end()
        if m := regex.match(spec, pos):
            pos = m.end()
            return m.group()
        return None

    def letter(chars: str) -> str | None:
        nonlocal pos
        if m := CHAR.match(spec, pos):
            c = m.group(1)
            if c.isalpha():
                if 'x' in chars:
                    return 'x'
            elif c in chars:
                pos = m.end()
                return c
        return None

    def error(msg: str) -> NoReturn:
        l = max(pos-6,0)
        r = max(pos+6,0)
        bef = '...' + spec[l:pos] if l > 0 else spec[:pos]
        mid = spec[pos]
        aft = spec[pos+1:r] + '...' if r < len(spec) else spec[pos+1:]
        raise SyntaxError(f"{msg}: '{bef}\x1b[4m{mid}\x1b[24m{aft}'")

    def splat(right: str) -> MsgKey:
        if not match(SPLAT): error('. can only be part of splat `...`')
        name = match(IDENT) or error('splat `...` must have a name')
        if not letter(right): error(f"splat not followed by closing {right!r}")
        add_key(name)
        return name

    def default() -> JsLiteral | None:
        if letter('='):
            return match(LIT) or error('invalid default value')
        return None

    def symbol() -> Symbol:
        key = match(IDENT) or error('invalid identifier')
        add_key(key)
        return Symbol(key, default())

    def seq(right: str) -> Seq:
        res = Seq([])
        add = res.elems.append
        chars = '*[{,.x' + right
        while True:
            match letter(chars):
                case 'x':
                    add(symbol())
                case '*' if letter('*'):
                    add(Obj([add_splice('opt')]))
                case '*':
                    add(add_splice('req'))
                case '[':
                    add(seq(']'))
                case '{':
                    add(obj())
                case ',':
                    continue
                case '.':
                    res.splat = splat(right)
                    break
                case ']' | ')':
                    break
                case _:
                    error('invalid sequence element')
        res.default = default()
        return res

    def item() -> Item:
        key = match(IDENT) or error('invalid object key')
        val: Symbol | Comp
        if letter(':'):
            match letter('x[{'):
                case 'x': val = symbol()
                case '[': val = seq(']')
                case '{': val = obj()
                case _: error('invalid object key binding')
        else:
            val = Symbol(key, default())
            add_key(key)
        return Item(key, val)

    def obj() -> Obj:
        res = Obj([])
        add = res.elems.append
        while True:
            match letter('*,.x}'):
                case 'x':
                    add(item())
                case '*':
                    if not letter('*'):
                        error('only ** allowed in objects')
                    add(add_splice('opt'))
                case ',':
                    continue
                case '.':
                    res.splat = splat('}')
                    break
                case '}':
                    break
                case _:
                    error('invalid object element')
        res.default = default()
        return res

    def root() -> Seq:
        match letter('({['):
            case '(': return seq(')')
            case '[': return seq(']')
            case '{': return Seq([obj()])
            case _: error('invalid root element')

    sig = Signature(root(), keys)
    if len(s_used) == 1:
        s_used.pop().kind = 'all'
    return sig


# ---------------------------------------------------------------------------
# Expansion: resolving a parsed Signature against a struct's fields
# ---------------------------------------------------------------------------
#
# `expand_signature` is a pure symbolic pass: it takes the parsed Signature
# plus a pre-rendered `FieldArg` per struct field (the caller supplies the
# rendered type reference, default literal, and binding name - everything
# language- or generator-specific) and produces a concrete pattern tree with
# every `*`/`**` splice expanded and every symbol bound to its field. The
# TS/JSDoc emitters then render that tree to text in their own separate
# passes.


class FieldArg(Struct):
    """A struct field as seen by the constructor expander, with all
    generator-specific rendering (type refs, default literals, binding-name
    sanitization) already performed by the caller."""

    name:    MsgKey            # Python-level field name (what specs reference)
    prop:    str               # class property name (the encoded name)
    binding: str               # sanitized, unique JS binding name
    type:    str               # rendered type reference
    default: JsLiteral | None  # rendered default value, if it has one
    required: bool


class Bound(Struct):
    """A single binding in an expanded pattern: one field, bound."""

    arg:     FieldArg
    key:     ObjKey | None = None      # object key, when inside an object pattern
    default: JsLiteral | None = None   # spec-level default, else the field's
    rest:    bool = False              # bound via a `...name` splat

    @property
    def optional(self) -> bool:
        return self.default is not None or not self.arg.required


class XSeq(Struct):
    """An expanded sequence pattern (or the parameter list itself)."""

    elems:   list[XPat]
    rest:    Bound | None = None
    default: JsLiteral | None = None


class XObj(Struct):
    """An expanded object pattern."""

    items:   list[tuple[ObjKey, XPat]]
    rest:    Bound | None = None
    default: JsLiteral | None = None


XPat = XSeq | XObj | Bound


class Expanded(Struct):
    """A Signature resolved against a concrete field list."""

    params: XSeq             # the constructor parameter list
    bound:  list[Bound]      # every binding, in appearance order
    extras: list[FieldArg]   # optional fields the signature leaves unbound


def expand_signature(sig: Signature, args: Sequence[FieldArg]) -> Expanded:
    """Expand a parsed constructor Signature against a struct's fields.

    Splices take the fields not explicitly mentioned in the signature, in
    field order; an empty `{**}` is elided. Raises ValueError if the
    signature references an unknown field or leaves a required field
    unbound.
    """
    index = {a.name: a for a in args}
    known = set(sig.keys)
    required = [a for a in args if a.required and a.name not in known]
    optional = [a for a in args if not a.required and a.name not in known]
    bound: list[Bound] = []

    def bind(arg: FieldArg, key: ObjKey | None = None,
             default: JsLiteral | None = None, rest: bool = False) -> Bound:
        if default is None and not rest:
            default = arg.default
        b = Bound(arg, key, default, rest)
        bound.append(b)
        return b

    def lookup(name: MsgKey) -> FieldArg:
        try:
            return index[name]
        except KeyError:
            raise ValueError(
                f"constructor signature references unknown field {name!r}"
            ) from None

    def taken(kind: str) -> list[FieldArg]:
        out: list[FieldArg] = []
        if kind in ('req', 'all'):
            out += required
            required.clear()
        if kind in ('opt', 'all'):
            out += optional
            optional.clear()
        return out

    def conv_seq(s: Seq) -> XSeq:
        out = XSeq([])
        for e in s.elems:
            match e:
                case Splice(kind):
                    out.elems += [bind(a) for a in taken(kind)]
                case Symbol(name, default):
                    out.elems.append(bind(lookup(name), default=default))
                case Seq():
                    out.elems.append(conv_seq(e))
                case Obj():
                    x = conv_obj(e)
                    if x.items or x.rest:  # an empty `{**}` is elided
                        out.elems.append(x)
                case _:  # Item can't appear in a sequence pattern
                    raise TypeError(f"unexpected sequence element {e!r}")
        if s.splat is not None:
            out.rest = bind(lookup(s.splat), rest=True)
        out.default = s.default
        return out

    def conv_obj(o: Obj) -> XObj:
        out = XObj([])
        for e in o.elems:
            match e:
                case Splice(kind):
                    out.items += [
                        (a.prop, bind(a, key=a.prop)) for a in taken(kind)
                    ]
                case Item(key, Symbol(name, default)):
                    out.items.append(
                        (key, bind(lookup(name), key=key, default=default))
                    )
                case Item(key, Seq() as s):
                    out.items.append((key, conv_seq(s)))
                case Item(key, Obj() as o2):
                    out.items.append((key, conv_obj(o2)))
                case _:  # only Splice/Item appear in an object pattern
                    raise TypeError(f"unexpected object element {e!r}")
        if o.splat is not None:
            out.rest = bind(lookup(o.splat), rest=True)
        out.default = o.default
        return out

    params = conv_seq(sig.args)
    if required:
        names = ', '.join(repr(a.name) for a in required)
        raise ValueError(
            f"constructor signature leaves required field(s) {names} unbound"
        )
    return Expanded(params, bound, optional)
