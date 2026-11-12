"""Fast Expression parser and validator (Phase 3, used by the Alpha Generator).

Method: a small hand-written lexer and recursive-descent parser turn an expression into
a syntax tree. A walk over the tree then checks it against ``config/fastexpr.yaml``:
every called operator must be listed with a matching argument count, every bare name
must be a variable defined earlier in the same expression, a group, a constant, a base
field or (when a catalog is supplied) a datafield from the catalog.

Two rules follow BRAIN's own simulation errors (run rs20261104163055, 5 of 19 simulations
lost): an optional argument (``name=default`` in the signature) must be written by name,
since ``winsorize(x, 4)`` gives "Invalid number of inputs : 2, should be exactly 1"; and an
event (VECTOR) datafield may appear only as the argument of a ``vec_*`` operator, since
``ts_delta(<vector field>, 30)`` gives "Operator ts_delta does not support event inputs".
Field types come from the catalog (``catalog(ids, meta)``); without types the second rule
is not checked.

Supported syntax: calls with positional and keyword arguments, ``+ - * /``, unary minus,
comparisons (``< > <= >= == !=``), ``&&`` and ``||``, numbers, quoted strings, and
``name = expr;`` statements before the final expression. Anything else is reported as an
error, never guessed at.

Generated text is untrusted. ``analyze`` therefore never raises on bad input; it returns
the problems in ``Analysis.errors`` so the generator can feed them back to the model.
Only a malformed spec file raises (``ConfigError``).

The operator list in config/fastexpr.yaml was checked against our account's operators
with ``cli brain operators`` (OQ-10).
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Collection, Mapping

import yaml

from .config import ConfigError


class FastExprError(ValueError):
    """Syntax problem found while parsing one expression."""


# ---------------------------------------------------------------------------
# lexer
# ---------------------------------------------------------------------------

_LEX_RE = re.compile(
    r"""
    (?P<ws>\s+)
  | (?P<sci>(?:\d+\.\d*|\.\d+|\d+)[eE][+-]?\d+)
  | (?P<num>\d+\.\d*|\.\d+|\d+)
  | (?P<name>[A-Za-z_][A-Za-z0-9_]*)
  | (?P<str>"[^"\n]*"|'[^'\n]*')
  | (?P<op>&&|\|\||<=|>=|==|!=|[-+*/(),;=<>])
    """,
    re.X,
)


@dataclass(frozen=True)
class Lexeme:
    kind: str  # num, name, str, op, end
    text: str
    pos: int


def _lex(source: str) -> list[Lexeme]:
    out: list[Lexeme] = []
    pos = 0
    while pos < len(source):
        m = _LEX_RE.match(source, pos)
        if not m:
            raise FastExprError(f"unexpected character {source[pos]!r} at position {pos}")
        if m.lastgroup == "sci":
            # BRAIN's parser stops at the "e" (run rs20261111162604: 'Unexpected character')
            plain = f"{float(m.group()):.12f}".rstrip("0").rstrip(".")
            raise FastExprError(f"scientific notation {m.group()!r} at position {pos} is not "
                                f"accepted by BRAIN; write it as a plain decimal ({plain})")
        pos = m.end()
        if m.lastgroup != "ws":
            out.append(Lexeme(m.lastgroup, m.group(), m.start()))
    out.append(Lexeme("end", "", len(source)))
    return out


# ---------------------------------------------------------------------------
# syntax tree
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Num:
    text: str


@dataclass(frozen=True)
class Str:
    value: str


@dataclass(frozen=True)
class Name:
    name: str


@dataclass(frozen=True)
class Call:
    name: str
    args: tuple
    kwargs: tuple  # ((keyword, node), ...)


@dataclass(frozen=True)
class Unary:
    op: str
    operand: object


@dataclass(frozen=True)
class Binary:
    op: str
    left: object
    right: object


@dataclass(frozen=True)
class Assign:
    name: str
    value: object


class _Parser:
    def __init__(self, lexemes: list[Lexeme]):
        self.lx = lexemes
        self.i = 0

    def peek(self, k: int = 0) -> Lexeme:
        return self.lx[min(self.i + k, len(self.lx) - 1)]

    def take(self) -> Lexeme:
        t = self.peek()
        self.i += 1
        return t

    def at_op(self, text: str, k: int = 0) -> bool:
        t = self.peek(k)
        return t.kind == "op" and t.text == text

    def accept(self, text: str) -> bool:
        if self.at_op(text):
            self.i += 1
            return True
        return False

    def expect(self, text: str) -> None:
        if not self.accept(text):
            t = self.peek()
            found = t.text or "end of expression"
            raise FastExprError(f"expected {text!r} at position {t.pos}, found {found!r}")

    def program(self) -> tuple:
        stmts = []
        while self.peek().kind != "end":
            if self.accept(";"):
                continue
            stmts.append(self.statement())
            if self.at_op(")"):
                raise FastExprError(f"unmatched ')' at position {self.peek().pos}")
            if self.peek().kind != "end":
                self.expect(";")
        if not stmts:
            raise FastExprError("empty expression")
        return tuple(stmts)

    def statement(self):
        t = self.peek()
        if t.kind == "name" and self.at_op("=", 1):
            self.i += 2
            return Assign(t.text, self.expr())
        return self.expr()

    def expr(self):
        node = self.and_expr()
        while self.at_op("||"):
            self.i += 1
            node = Binary("||", node, self.and_expr())
        return node

    def and_expr(self):
        node = self.cmp_expr()
        while self.at_op("&&"):
            self.i += 1
            node = Binary("&&", node, self.cmp_expr())
        return node

    def cmp_expr(self):
        node = self.add_expr()
        while any(self.at_op(o) for o in ("<", ">", "<=", ">=", "==", "!=")):
            op = self.take().text
            node = Binary(op, node, self.add_expr())
        return node

    def add_expr(self):
        node = self.mul_expr()
        while self.at_op("+") or self.at_op("-"):
            op = self.take().text
            node = Binary(op, node, self.mul_expr())
        return node

    def mul_expr(self):
        node = self.unary()
        while self.at_op("*") or self.at_op("/"):
            op = self.take().text
            node = Binary(op, node, self.unary())
        return node

    def unary(self):
        if self.at_op("-") or self.at_op("+"):
            op = self.take().text
            return Unary(op, self.unary())
        return self.primary()

    def primary(self):
        t = self.take()
        if t.kind == "num":
            return Num(t.text)
        if t.kind == "str":
            return Str(t.text[1:-1])
        if t.kind == "name":
            if self.accept("("):
                return self.call_rest(t.text)
            return Name(t.text)
        if t.kind == "op" and t.text == "(":
            node = self.expr()
            self.expect(")")
            return node
        raise FastExprError(f"unexpected {t.text or 'end of expression'!r} at position {t.pos}")

    def call_rest(self, name: str) -> Call:
        args: list = []
        kwargs: list = []
        if not self.accept(")"):
            while True:
                if self.peek().kind == "name" and self.at_op("=", 1):
                    key = self.take().text
                    self.i += 1
                    kwargs.append((key, self.expr()))
                else:
                    if kwargs:
                        raise FastExprError(
                            f"positional argument after keyword argument in {name}(...)")
                    args.append(self.expr())
                if self.accept(","):
                    continue
                self.expect(")")
                break
        return Call(name, tuple(args), tuple(kwargs))


def parse(expression: str) -> tuple:
    """Parse to a tuple of statements. Raises ``FastExprError`` on any syntax problem."""
    if not isinstance(expression, str):
        raise FastExprError(f"expression must be a string, got {type(expression).__name__}")
    try:
        return _Parser(_lex(expression)).program()
    except RecursionError as exc:
        raise FastExprError("expression is nested too deeply to parse") from exc


_PREC = {"||": 1, "&&": 2, "<": 3, ">": 3, "<=": 3, ">=": 3, "==": 3, "!=": 3,
         "+": 4, "-": 4, "*": 5, "/": 5}
_UNARY_PREC = 6


def render_node(node, parent_prec: int = 0, right_side: bool = False) -> str:
    """Turn a syntax-tree node back into text, adding parentheses only where precedence
    needs them (operators are left-associative, so an equal-precedence right operand is
    wrapped). ``parse(render(...))`` gives back the same tree."""
    if isinstance(node, Num):
        return node.text
    if isinstance(node, Str):
        return f'"{node.value}"'
    if isinstance(node, Name):
        return node.name
    if isinstance(node, Call):
        parts = [render_node(a) for a in node.args] + [f"{k}={render_node(v)}" for k, v in node.kwargs]
        return f"{node.name}({', '.join(parts)})"
    if isinstance(node, Unary):
        text = node.op + render_node(node.operand, _UNARY_PREC)
        return f"({text})" if parent_prec > _UNARY_PREC else text
    if isinstance(node, Binary):
        prec = _PREC[node.op]
        text = (f"{render_node(node.left, prec)} {node.op} "
                f"{render_node(node.right, prec, right_side=True)}")
        wrap = prec < parent_prec or (right_side and prec == parent_prec)
        return f"({text})" if wrap else text
    if isinstance(node, Assign):
        return f"{node.name} = {render_node(node.value)}"
    raise TypeError(f"cannot render {type(node).__name__}")


def render(stmts: tuple) -> str:
    """Render a parsed program: ``a = x; b = y; final``."""
    return "; ".join(render_node(s) for s in stmts)


# ---------------------------------------------------------------------------
# spec (config/fastexpr.yaml)
# ---------------------------------------------------------------------------

_SIG_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*\((.*)\)\s*$")
_SIG_ARG_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(\s*=\s*[^,()]+)?$")


@dataclass(frozen=True)
class OperatorSpec:
    name: str
    signature: str
    min_args: int
    max_args: int | None  # None = any number of further arguments
    optional: tuple[str, ...] = ()  # names of the name=default arguments


def _parse_signature(sig: str) -> OperatorSpec:
    m = _SIG_RE.match(sig) if isinstance(sig, str) else None
    if not m:
        raise ConfigError(f"fastexpr: bad operator signature {sig!r}")
    name, inner = m.group(1), m.group(2).strip()
    items = [p.strip() for p in inner.split(",")] if inner else []
    if any(not p for p in items):
        raise ConfigError(f"fastexpr: empty argument in signature {sig!r}")
    variadic = bool(items) and items[-1] == "..."
    if variadic:
        items = items[:-1]
    if any("..." in p for p in items):
        raise ConfigError(f"fastexpr: '...' must be last in signature {sig!r}")
    if not all(_SIG_ARG_RE.match(p) for p in items):
        raise ConfigError(f"fastexpr: bad argument list in signature {sig!r}")
    required = [p for p in items if "=" not in p]
    optional_seen = False
    for p in items:  # required arguments may not follow optional ones
        if "=" in p:
            optional_seen = True
        elif optional_seen:
            raise ConfigError(f"fastexpr: required argument after optional one in {sig!r}")
    optional = tuple(p.split("=")[0].strip() for p in items if "=" in p)
    return OperatorSpec(name, sig.strip(), len(required), None if variadic else len(items),
                        optional)


@dataclass(frozen=True)
class FastExprSpec:
    operators: dict[str, OperatorSpec]
    groups: frozenset[str]
    base_fields: frozenset[str]
    constants: frozenset[str]

    @classmethod
    def load(cls, path: str | Path) -> "FastExprSpec":
        try:
            raw = yaml.safe_load(Path(path).read_text())
        except (OSError, yaml.YAMLError) as exc:
            raise ConfigError(f"cannot read fastexpr spec {path}: {exc}") from exc
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw) -> "FastExprSpec":
        if not isinstance(raw, dict):
            raise ConfigError("fastexpr spec must be a mapping")
        for key in ("operators", "groups", "base_fields", "constants"):
            if key not in raw:
                raise ConfigError(f"fastexpr spec: missing key {key!r}")
            if not isinstance(raw[key], list):
                raise ConfigError(f"fastexpr spec: {key!r} must be a list")
        ops: dict[str, OperatorSpec] = {}
        for sig in raw["operators"]:
            spec = _parse_signature(sig)
            if spec.name in ops:
                raise ConfigError(f"fastexpr spec: operator {spec.name!r} listed twice")
            ops[spec.name] = spec
        if not ops:
            raise ConfigError("fastexpr spec: no operators listed")
        sets = {}
        for key in ("groups", "base_fields", "constants"):
            if not all(isinstance(x, str) and x for x in raw[key]):
                raise ConfigError(f"fastexpr spec: {key!r} must hold non-empty strings")
            sets[key] = frozenset(raw[key])
        return cls(ops, sets["groups"], sets["base_fields"], sets["constants"])

    def signatures(self) -> list[str]:
        return [op.signature for op in self.operators.values()]


@dataclass(frozen=True)
class ComplexityLimits:
    """Policy limits. ``None`` means not configured, so the check is skipped."""
    max_depth: int | None = None
    max_calls: int | None = None
    max_chars: int | None = None


# ---------------------------------------------------------------------------
# analysis
# ---------------------------------------------------------------------------

@dataclass
class Analysis:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    operators_used: tuple[str, ...] = ()
    fields_used: tuple[str, ...] = ()
    depth: int = 0
    n_calls: int = 0

    @property
    def ok(self) -> bool:
        return not self.errors


def _depth(node) -> int:
    if isinstance(node, (Num, Str, Name)):
        return 0
    if isinstance(node, Assign):
        return _depth(node.value)
    if isinstance(node, Unary):
        return 1 + _depth(node.operand)
    if isinstance(node, Binary):
        return 1 + max(_depth(node.left), _depth(node.right))
    kids = list(node.args) + [v for _, v in node.kwargs]
    return 1 + max((_depth(k) for k in kids), default=0)


def _suggest(name: str, known: Collection[str]) -> str:
    lowered = {k.lower(): k for k in known}
    if name.lower() in lowered and lowered[name.lower()] != name:
        return f" (operator names are lowercase; did you mean {lowered[name.lower()]!r}?)"
    close = difflib.get_close_matches(name, list(known), n=2, cutoff=0.75)
    return f" (did you mean {' or '.join(repr(c) for c in close)}?)" if close else ""


EVENT_TYPE = "VECTOR"  # BRAIN's type for event datafields
VEC_PREFIX = "vec_"    # operators that turn an event field into one value per day


def catalog(datafields: Collection[str] | Mapping[str, str | None] | None,
            meta: Mapping[str, dict] | None = None) -> dict[str, str | None] | None:
    """Datafield catalog as {id: BRAIN type or None}. A mapping is taken as it is; a plain
    collection of ids gets its types from ``meta`` (``generator.load_datafield_meta``)."""
    if datafields is None:
        return None
    if isinstance(datafields, Mapping):
        return dict(datafields)
    meta = meta or {}
    return {f: (meta.get(f) or {}).get("type") for f in datafields}


def catalog_for(markets: Mapping[tuple, Mapping] | None, default: Mapping | None,
                region: str, delay: int) -> Mapping | None:
    """The catalog to validate an expression for ``region``/``delay`` against: that
    market's own (a multi-region plan has one per region and delay), else ``default``."""
    if markets and (region, delay) in markets:
        return markets[(region, delay)]
    return default


class _Walker:
    def __init__(self, spec: FastExprSpec,
                 datafields: Collection[str] | Mapping[str, str | None] | None):
        self.spec = spec
        self.datafields = None if datafields is None else set(datafields)
        self.events = ({f for f, t in datafields.items() if t == EVENT_TYPE}
                       if isinstance(datafields, Mapping) else set())
        self.variables: set[str] = set()
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.operators: set[str] = set()
        self.fields: set[str] = set()
        self.n_calls = 0

    def err(self, msg: str) -> None:
        if msg not in self.errors:
            self.errors.append(msg)

    def visit(self, node, in_vec: bool = False) -> None:
        if isinstance(node, (Num, Str)):
            return
        if isinstance(node, Name):
            self.name(node.name, in_vec)
        elif isinstance(node, Unary):
            self.visit(node.operand)
        elif isinstance(node, Binary):
            self.visit(node.left)
            self.visit(node.right)
        elif isinstance(node, Call):
            self.call(node)

    def name(self, n: str, in_vec: bool = False) -> None:
        s = self.spec
        if n in self.variables or n in s.groups or n in s.constants:
            return
        if n in self.events and not in_vec:
            self.err(f"datafield {n!r} is an event (vector) field: use it only inside "
                     f"vec_avg({n}) or vec_sum({n}), then apply other operators to that")
        if n in s.base_fields:
            self.fields.add(n)
        elif self.datafields is not None:
            if n in self.datafields:
                self.fields.add(n)
            else:
                self.err(f"unknown datafield {n!r}{_suggest(n, self.datafields | set(s.base_fields))}")
        else:
            self.fields.add(n)
            msg = f"datafield {n!r} not verified (no datafield catalog supplied)"
            if msg not in self.warnings:
                self.warnings.append(msg)

    def call(self, node: Call) -> None:
        self.n_calls += 1
        op = self.spec.operators.get(node.name)
        if op is None:
            self.err(f"unknown operator {node.name!r}{_suggest(node.name, self.spec.operators)}")
        else:
            self.operators.add(node.name)
            n = len(node.args) + len(node.kwargs)
            if n < op.min_args or (op.max_args is not None and n > op.max_args):
                want = (f"{op.min_args}" if op.max_args == op.min_args else
                        f"at least {op.min_args}" if op.max_args is None else
                        f"{op.min_args} to {op.max_args}")
                self.err(f"{node.name} takes {want} arguments, got {n} (signature: {op.signature})")
            elif op.max_args is not None and len(node.args) > op.min_args:
                named = ", ".join(f"{k}=..." for k in op.optional[:len(node.args) - op.min_args])
                self.err(f"{node.name}: optional arguments must be written by name "
                         f"({named}); signature: {op.signature}")
        vec = node.name.startswith(VEC_PREFIX)
        for a in node.args:
            self.visit(a, vec)
        for _, v in node.kwargs:
            self.visit(v)


def analyze(expression: str, spec: FastExprSpec, *,
            datafields: Collection[str] | Mapping[str, str | None] | None = None,
            limits: ComplexityLimits | None = None) -> Analysis:
    """Check one expression. Never raises on bad input; problems go in ``errors``.
    ``datafields`` with types (``catalog``) also checks that event fields sit in vec_*."""
    try:
        stmts = parse(expression)
    except FastExprError as exc:
        return Analysis(errors=[str(exc)])

    w = _Walker(spec, datafields)
    for i, stmt in enumerate(stmts):
        last = i == len(stmts) - 1
        if isinstance(stmt, Assign):
            if last:
                w.err("the last statement must be the alpha expression, not an assignment")
            if stmt.name in spec.operators:
                w.err(f"variable {stmt.name!r} has the same name as an operator")
            w.visit(stmt.value)
            w.variables.add(stmt.name)  # defined only for later statements
        else:
            w.visit(stmt)
    if not w.fields and not w.errors:
        w.err("expression references no datafield")

    depth = max(_depth(s) for s in stmts)
    if limits is not None:
        if limits.max_depth is not None and depth > limits.max_depth:
            w.err(f"nesting depth {depth} exceeds the limit of {limits.max_depth}")
        if limits.max_calls is not None and w.n_calls > limits.max_calls:
            w.err(f"{w.n_calls} operator calls exceed the limit of {limits.max_calls}")
        if limits.max_chars is not None and len(expression) > limits.max_chars:
            w.err(f"{len(expression)} characters exceed the limit of {limits.max_chars}")

    return Analysis(errors=w.errors, warnings=w.warnings,
                    operators_used=tuple(sorted(w.operators)),
                    fields_used=tuple(sorted(w.fields)),
                    depth=depth, n_calls=w.n_calls)
