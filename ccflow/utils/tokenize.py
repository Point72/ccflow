"""Tokenization engine for ccflow models.

Provides deterministic content hashing for pydantic BaseModel instances,
with configurable hash algorithms, Merkle tree optimization for frozen models,
and extensibility via ``__ccflow_tokenize__`` hooks.

Usage::

    from ccflow import BaseModel
    from ccflow.utils.tokenize import DefaultTokenizer

    class MyModel(BaseModel):
        x: int = 1

    model = MyModel()
    model.model_token  # hex digest string

    # With behavior hashing
    tokenizer = DefaultTokenizer.with_bytecode()
"""

import ast
import enum
import hashlib
import inspect
import logging
import textwrap
from abc import ABC, abstractmethod
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from functools import partial, singledispatch
from pathlib import PurePath
from types import MappingProxyType
from typing import Any, Callable, List, Optional, Set, Tuple
from uuid import UUID

from pydantic import BaseModel as PydanticBaseModel

log = logging.getLogger(__name__)

__all__ = [
    "SourceTokenizer",
    "ASTSourceTokenizer",
    "BytecodeSourceTokenizer",
    "FunctionCollector",
    "OwnMethodCollector",
    "Tokenizer",
    "DefaultTokenizer",
    "normalize_token",
    "compute_behavior_token",
]


# ---------------------------------------------------------------------------
# SourceTokenizer — how to hash a single function
# ---------------------------------------------------------------------------


class SourceTokenizer(ABC):
    """Tokenizes a single callable into a digest string.

    Subclass to provide different code hashing strategies (AST, bytecode, etc.).
    """

    @abstractmethod
    def tokenize(self, func: Callable) -> Optional[str]:
        """Return a hex digest of *func*'s source/bytecode, or None if unavailable."""
        ...


def _normalize_source_ast(source: str) -> str:
    """Parse source, strip docstrings, return normalized form via ast.unparse."""
    source = textwrap.dedent(source)
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, (ast.Constant, ast.Str)):
                node.body.pop(0)
    return ast.unparse(tree)


class ASTSourceTokenizer(SourceTokenizer):
    """Hash function source via AST normalization (strips docstrings, normalizes whitespace)."""

    def tokenize(self, func: Callable) -> Optional[str]:
        try:
            source = inspect.getsource(func)
            normalized = _normalize_source_ast(source)
            return hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        except (OSError, TypeError):
            # Source not available — fall back to bytecode
            code = getattr(func, "__code__", None)
            if code is not None:
                return hashlib.sha256(code.co_code).hexdigest()
            return None


class BytecodeSourceTokenizer(SourceTokenizer):
    """Hash function bytecode (co_code + co_consts, docstring stripped)."""

    def tokenize(self, func: Callable) -> Optional[str]:
        code = getattr(func, "__code__", None)
        if code is None:
            return None
        consts = code.co_consts
        # Strip the docstring slot (co_consts[0]): a str when present, None when absent
        if consts and isinstance(consts[0], (str, type(None))):
            consts = consts[1:]
        payload = repr((code.co_code, consts)).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


# ---------------------------------------------------------------------------
# FunctionCollector — which functions to hash for a class
# ---------------------------------------------------------------------------


class FunctionCollector(ABC):
    """Discovers functions relevant to a class for behavior hashing.

    Subclass to control which methods/functions participate in the behavior token.
    """

    @abstractmethod
    def collect(self, cls: type) -> List[Tuple[str, Callable]]:
        """Return a sorted list of ``(name, callable)`` pairs for *cls*."""
        ...


class OwnMethodCollector(FunctionCollector):
    """Collects all callable methods defined directly on *cls* (via ``cls.__dict__``),
    plus any standalone functions listed in ``cls.__ccflow_tokenizer_deps__``."""

    def collect(self, cls: type) -> List[Tuple[str, Callable]]:
        methods = []
        for name, value in cls.__dict__.items():
            if isinstance(value, (classmethod, staticmethod)):
                methods.append((name, value.__func__))
            elif callable(value):
                methods.append((name, value))
        methods.sort(key=lambda pair: pair[0])

        # Add standalone functions from __ccflow_tokenizer_deps__
        deps = []
        extra_deps = cls.__dict__.get("__ccflow_tokenizer_deps__")
        if extra_deps is not None:
            for func in extra_deps:
                if isinstance(func, (classmethod, staticmethod)):
                    func = func.__func__
                if callable(func):
                    func_id = getattr(func, "__qualname__", getattr(func, "__name__", repr(func)))
                    deps.append((f"__dep__:{func_id}", func))
            deps.sort(key=lambda pair: pair[0])
            methods.extend(deps)

        return methods


# ---------------------------------------------------------------------------
# compute_behavior_token — composes collector + source tokenizer
# ---------------------------------------------------------------------------

_BEHAVIOR_CACHE_ATTR = "__ccflow_behavior_token__"


def _get_behavior_cache(cls: type, cache_key: tuple) -> Optional[str]:
    """Read from the per-class behavior token cache (returns None on miss)."""
    cached = cls.__dict__.get(_BEHAVIOR_CACHE_ATTR)
    if isinstance(cached, dict) and cache_key in cached:
        return cached[cache_key]
    return None


def _set_behavior_cache(cls: type, cache_key: tuple, token: str) -> None:
    """Write to the per-class behavior token cache."""
    try:
        if not isinstance(cls.__dict__.get(_BEHAVIOR_CACHE_ATTR), dict):
            setattr(cls, _BEHAVIOR_CACHE_ATTR, {})
        getattr(cls, _BEHAVIOR_CACHE_ATTR)[cache_key] = token
    except (TypeError, AttributeError):
        pass


def compute_behavior_token(
    cls: type,
    collector: Optional["FunctionCollector"] = None,
    source_tokenizer: Optional["SourceTokenizer"] = None,
) -> Optional[str]:
    """Compute a behavior token for a class by hashing collected functions.

    Results are cached on the class keyed by ``(collector_type, source_type)``.
    """
    if collector is None:
        collector = OwnMethodCollector()
    if source_tokenizer is None:
        source_tokenizer = BytecodeSourceTokenizer()

    cache_key = (id(collector), id(source_tokenizer))
    cached = _get_behavior_cache(cls, cache_key)
    if cached is not None:
        return cached

    method_hashes = [(name, h) for name, method in collector.collect(cls) if (h := source_tokenizer.tokenize(method)) is not None]
    if not method_hashes:
        return None

    token = hashlib.sha256(repr(method_hashes).encode("utf-8")).hexdigest()
    _set_behavior_cache(cls, cache_key, token)
    return token


# ---------------------------------------------------------------------------
# normalize_token — singledispatch-based canonical normalization
# ---------------------------------------------------------------------------


@singledispatch
def normalize_token(obj: Any) -> Any:
    """Produce a canonical, hashable representation of an object.

    This is a singledispatch function — register handlers for new types via::

        @normalize_token.register(MyType)
        def _(obj):
            return ...

    Objects with a ``__ccflow_tokenize__`` method use it automatically.
    """
    # Check for custom hook
    method = getattr(obj, "__ccflow_tokenize__", None)
    if method is not None:
        return method()

    # Try cloudpickle as last resort
    try:
        import cloudpickle

        pickled = cloudpickle.dumps(obj)
        return ("__cloudpickle__", hashlib.sha256(pickled).hexdigest())
    except Exception:
        raise TypeError(
            f"Cannot tokenize object of type {type(obj).__qualname__}. Implement __ccflow_tokenize__() or register a normalize_token handler."
        )


# --- Primitives ---


@normalize_token.register(type(None))
def _normalize_none(obj):
    return None


@normalize_token.register(bool)
@normalize_token.register(int)
@normalize_token.register(float)
@normalize_token.register(str)
@normalize_token.register(bytes)
def _normalize_primitive(obj):
    return obj


# --- Date/time ---


@normalize_token.register(date)
def _normalize_date(obj):
    return ("date", obj.isoformat())


@normalize_token.register(datetime)
def _normalize_datetime(obj):
    return ("datetime", obj.isoformat())


@normalize_token.register(time)
def _normalize_time(obj):
    return ("time", obj.isoformat())


@normalize_token.register(timedelta)
def _normalize_timedelta(obj):
    return ("timedelta", obj.total_seconds())


# --- UUID ---


@normalize_token.register(UUID)
def _normalize_uuid(obj):
    return ("uuid", str(obj))


# --- Path ---


@normalize_token.register(PurePath)
def _normalize_path(obj):
    return ("path", str(obj))


# --- Enum ---


@normalize_token.register(enum.Enum)
def _normalize_enum(obj):
    return ("enum", type(obj).__qualname__, obj.name)


# --- Collections ---


@normalize_token.register(tuple)
def _normalize_tuple(obj):
    return ("tuple", tuple(normalize_token(item) for item in obj))


@normalize_token.register(list)
def _normalize_list(obj):
    return ("list", tuple(normalize_token(item) for item in obj))


@normalize_token.register(set)
def _normalize_set(obj):
    return ("set", tuple(sorted((normalize_token(item) for item in obj), key=repr)))


@normalize_token.register(frozenset)
def _normalize_frozenset(obj):
    return ("frozenset", tuple(sorted((normalize_token(item) for item in obj), key=repr)))


@normalize_token.register(dict)
def _normalize_dict(obj):
    return (
        "dict",
        tuple(
            sorted(
                ((normalize_token(k), normalize_token(v)) for k, v in obj.items()),
                key=repr,
            )
        ),
    )


# --- Additional builtins ---


@normalize_token.register(complex)
def _normalize_complex(obj):
    return ("complex", obj.real, obj.imag)


@normalize_token.register(type(Ellipsis))
def _normalize_ellipsis(obj):
    return ("ellipsis",)


@normalize_token.register(slice)
def _normalize_slice(obj):
    return ("slice", obj.start, obj.stop, obj.step)


@normalize_token.register(type(len))  # builtin_function_or_method
def _normalize_builtin(obj):
    return ("builtin", obj.__qualname__)


@normalize_token.register(Decimal)
def _normalize_decimal(obj):
    return ("decimal", str(obj))


@normalize_token.register(partial)
def _normalize_partial(obj):
    return (
        "partial",
        normalize_token(obj.func),
        normalize_token(obj.args),
        normalize_token(sorted(obj.keywords.items())),
    )


@normalize_token.register(MappingProxyType)
def _normalize_mappingproxy(obj):
    return _normalize_dict(dict(obj))


# --- Numpy ---


def _register_numpy():
    """Register numpy normalize_token handlers."""
    try:
        import numpy as np
    except ImportError:
        return

    @normalize_token.register(np.ndarray)
    def _normalize_ndarray(obj):
        return ("ndarray", str(obj.dtype), obj.shape, hashlib.sha256(obj.tobytes()).hexdigest())

    @normalize_token.register(np.generic)
    def _normalize_np_scalar(obj):
        return ("np_scalar", str(type(obj).__name__), obj.item())


def _register_pandas():
    """Register pandas normalize_token handlers."""
    try:
        import pandas as pd
    except ImportError:
        return

    @normalize_token.register(pd.Timestamp)
    def _normalize_pd_timestamp(obj):
        return ("pd_timestamp", obj.isoformat())


# --- Functions ---


@normalize_token.register(type(lambda: None))  # FunctionType
def _normalize_function(obj):
    method = getattr(obj, "__ccflow_tokenize__", None)
    if method is not None:
        return method()
    # Try AST-normalized source
    try:
        source = inspect.getsource(obj)
        normalized = _normalize_source_ast(source)
        return ("func", obj.__qualname__, hashlib.sha256(normalized.encode()).hexdigest())
    except (OSError, TypeError):
        pass
    # Fallback to bytecode
    code = getattr(obj, "__code__", None)
    if code is not None:
        return ("func", obj.__qualname__, hashlib.sha256(code.co_code).hexdigest())
    # Last resort: qualified name only
    return ("func", obj.__qualname__)


@normalize_token.register(type)
def _normalize_type(obj):
    return ("type", f"{obj.__module__}.{obj.__qualname__}")


# --- Pydantic BaseModel ---
# NOTE: The ccflow BaseModel handler is registered in ccflow/base.py
# to avoid circular imports. This handles plain pydantic BaseModel.


@normalize_token.register(PydanticBaseModel)
def _normalize_pydantic_basemodel(obj):
    type_path = f"{type(obj).__module__}.{type(obj).__qualname__}"
    model_fields = type(obj).model_fields
    fields = tuple((k, normalize_token(v)) for k, v in obj if k in model_fields and not model_fields[k].exclude)
    return ("pydantic", type_path, fields)


# Register numpy/pandas handlers at import time
_register_numpy()
_register_pandas()


# ---------------------------------------------------------------------------
# Tokenizer ABC and DefaultTokenizer
# ---------------------------------------------------------------------------


class Tokenizer(ABC):
    """Abstract tokenization engine.

    Subclass and override methods to customize tokenization behavior.
    Set ``__ccflow_tokenizer__`` on a BaseModel class to swap engines.
    """

    def hash_canonical(self, canonical: Any) -> str:
        """Hash an arbitrary canonical form to a hex digest."""
        return hashlib.sha256(repr(canonical).encode("utf-8")).hexdigest()

    @abstractmethod
    def normalize(self, model: PydanticBaseModel, *, _visited: Optional[Set[int]] = None) -> Any:
        """Produce a canonical structured representation of a model."""
        ...

    @abstractmethod
    def tokenize(self, model: PydanticBaseModel) -> str:
        """Produce a hex digest token for a model."""
        ...

    def normalize_value(self, value: Any, *, _visited: Optional[Set[int]] = None) -> Any:
        """Normalize an arbitrary value. Override for custom dispatch."""
        return normalize_token(value)


def _normalize_model_fields(tokenizer: "Tokenizer", model: PydanticBaseModel, _visited: Set[int]) -> List[Tuple[str, Any]]:
    """Normalize a model's non-excluded fields via the tokenizer's normalize_value."""
    fields = []
    model_fields = type(model).model_fields
    for field_name, field_info in model_fields.items():
        if field_info.exclude:
            continue
        value = getattr(model, field_name)
        fields.append((field_name, tokenizer.normalize_value(value, _visited=_visited)))
    return fields


class DefaultTokenizer(Tokenizer):
    """Default tokenization engine using singledispatch-based normalization.

    Composes a ``FunctionCollector`` and ``SourceTokenizer`` for optional
    behavior hashing.  When both are ``None`` (the default), only field
    data is hashed.
    """

    def __init__(
        self,
        collector: Optional[FunctionCollector] = None,
        source_tokenizer: Optional[SourceTokenizer] = None,
    ):
        self.collector = collector
        self.source_tokenizer = source_tokenizer

    @classmethod
    def with_ast(cls) -> "DefaultTokenizer":
        """Convenience constructor: own methods hashed via AST normalization."""
        return cls(collector=OwnMethodCollector(), source_tokenizer=ASTSourceTokenizer())

    @classmethod
    def with_bytecode(cls) -> "DefaultTokenizer":
        """Convenience constructor: own methods hashed via bytecode."""
        return cls(collector=OwnMethodCollector(), source_tokenizer=BytecodeSourceTokenizer())

    def normalize_value(self, value: Any, *, _visited: Optional[Set[int]] = None) -> Any:
        """Normalize an arbitrary value, routing containers and models through the tokenizer.

        Re-implements container handling (rather than delegating to normalize_token)
        so that nested models participate in cycle detection via _visited.
        """
        # Fast path for common primitives — avoids singledispatch overhead
        if type(value) in (int, str, float, bool, type(None), bytes):
            return value

        if isinstance(value, PydanticBaseModel):
            is_frozen = value.model_config.get("frozen", False)
            if is_frozen and hasattr(value, "model_token"):
                return ("__child__", value.model_token)
            return self.normalize(value, _visited=_visited)

        # Cycle detection for mutable containers
        val_id = id(value)
        if isinstance(value, (list, dict, set)):
            if _visited is None:
                _visited = set()
            if val_id in _visited:
                return ("__cycle__", type(value).__name__)
            _visited.add(val_id)

        if isinstance(value, dict):
            result = (
                "dict",
                tuple(
                    sorted(
                        ((self.normalize_value(k, _visited=_visited), self.normalize_value(v, _visited=_visited)) for k, v in value.items()),
                        key=repr,
                    )
                ),
            )
            _visited.discard(val_id)
            return result
        if isinstance(value, (list, tuple)):
            tag = "list" if isinstance(value, list) else "tuple"
            result = (tag, tuple(self.normalize_value(v, _visited=_visited) for v in value))
            _visited.discard(val_id)
            return result
        if isinstance(value, (set, frozenset)):
            tag = "set" if isinstance(value, set) else "frozenset"
            result = (
                tag,
                tuple(
                    sorted(
                        (self.normalize_value(v, _visited=_visited) for v in value),
                        key=repr,
                    )
                ),
            )
            _visited.discard(val_id)
            return result

        return normalize_token(value)

    def normalize(self, model: PydanticBaseModel, *, _visited: Optional[Set[int]] = None) -> Any:
        """Produce a canonical structured representation."""
        model_id = id(model)

        if _visited is not None and model_id in _visited:
            type_path = f"{type(model).__module__}.{type(model).__qualname__}"
            return ("__cycle__", type_path)

        if _visited is None:
            _visited = set()
        _visited.add(model_id)

        type_path = f"{type(model).__module__}.{type(model).__qualname__}"

        behavior = None
        if self.collector is not None and self.source_tokenizer is not None:
            behavior = compute_behavior_token(
                type(model),
                collector=self.collector,
                source_tokenizer=self.source_tokenizer,
            )

        fields = _normalize_model_fields(self, model, _visited)

        # Backtrack so sibling fields of a parent model don't false-positive as cycles
        _visited.discard(model_id)
        return (type_path, behavior, tuple(fields))

    def tokenize(self, model: PydanticBaseModel) -> str:
        """Produce a hex digest token."""
        return self.hash_canonical(self.normalize(model))


class DaskTokenizer(Tokenizer):
    """Tokenizer that delegates to ``dask.base.tokenize`` for backward compatibility.

    Hashes ``model.model_dump(mode="python")`` using dask's tokenization,
    matching the legacy ``cache_key()`` behavior.  Requires ``dask`` to be
    installed (imported lazily).
    """

    def normalize(self, model: PydanticBaseModel, *, _visited: Optional[Set[int]] = None) -> Any:
        return model.model_dump(mode="python")

    def tokenize(self, model: PydanticBaseModel) -> str:
        import dask.base

        return dask.base.tokenize(model.model_dump(mode="python"))
