"""Tokenization utilities for ccflow models.

Provides a native ``normalize_token`` ``singledispatch`` registry that
canonicalizes Python objects to deterministically hashable forms, plus
helpers for ccflow-specific behavior hashing and combined cache-token
hashing used by ``cache_key()`` for cache-key invalidation when callable
logic changes.

The native engine replaces the previous ``dask.base.tokenize`` dependency.
Type handlers cover stdlib primitives, datetime, Decimal, UUID, pathlib,
``functools.partial``, ``MappingProxyType``, numpy, pandas, and plain
pydantic ``BaseModel``. Unknown objects fall back to a cloudpickle-based
digest; objects whose pickling raises an exception produce a clear
``TypeError``.
"""

import contextvars
import enum
import hashlib
import inspect
from collections import OrderedDict
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from functools import partial, singledispatch
from pathlib import PurePath
from types import CodeType, MappingProxyType, MethodType, MethodWrapperType, ModuleType
from typing import Any, Callable, Iterable, List, Optional, Tuple
from uuid import UUID

from pydantic import BaseModel as _PydanticBaseModel

__all__ = (
    "compute_behavior_token",
    "compute_cache_token",
    "compute_data_token",
    "normalize_token",
    "tokenize",
)


def _sha256_hexdigest(*parts: bytes | str) -> str:
    """Return a SHA-256 hex digest for one or more byte/string parts."""
    hasher = hashlib.sha256()
    for part in parts:
        if isinstance(part, str):
            part = part.encode("utf-8")
        hasher.update(part)
    return hasher.hexdigest()


# ContextVar (rather than threading.local) so the visited set auto-isolates across asyncio tasks.
_visited: contextvars.ContextVar[Optional[set]] = contextvars.ContextVar("_ccflow_normalize_visited", default=None)

# Tracks functions currently being hashed to detect circular closure references.
_hashing_functions: contextvars.ContextVar[Optional[set]] = contextvars.ContextVar("_ccflow_hashing_functions", default=None)


def _with_cycle_check(obj: Any, build: Callable[[], Any]) -> Any:
    """Invoke ``build`` with object-identity cycle detection, returning ``("__cycle__", type_name)`` on re-entry."""
    visited = _visited.get()
    created = visited is None
    if created:
        visited = set()
        token = _visited.set(visited)
    elif id(obj) in visited:
        return ("__cycle__", type(obj).__name__)
    visited.add(id(obj))
    try:
        return build()
    finally:
        visited.discard(id(obj))
        if created:
            _visited.reset(token)


@singledispatch
def normalize_token(obj: Any) -> Any:
    """Produce a canonical, deterministically hashable representation of ``obj``.

    This is a ``singledispatch`` function — register handlers for new types via::

        @normalize_token.register(MyType)
        def _(obj):
            return ("mytype", ...)

    Unknown types fall back to a ``cloudpickle``-based digest, raising ``TypeError`` on pickling failure.
    """
    # Python 3.14 exposes internal objects (e.g. _abc._abc_data, pydantic._internal.*,
    # pydantic_core._pydantic_core.*) in function closures of ABC-derived classes.
    # These are not behavior-relevant; produce a stable token keyed by module + qualname
    # so they don't affect hashing.
    obj_module = getattr(type(obj), "__module__", "") or ""
    if obj_module.startswith("_") or "._" in obj_module:
        return ("__internal__", obj_module, type(obj).__qualname__)

    try:
        import cloudpickle
    except ImportError as exc:  # pragma: no cover - defensive
        raise TypeError(f"Cannot tokenize object of type {type(obj).__qualname__}: cloudpickle is not installed.") from exc

    try:
        pickled = cloudpickle.dumps(obj)
    except Exception as exc:
        raise TypeError(f"Cannot tokenize object of type {type(obj).__qualname__}. Register a normalize_token handler for this type.") from exc
    return ("__cloudpickle__", hashlib.sha256(pickled).hexdigest())


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
    return ("timedelta", obj.days, obj.seconds, obj.microseconds)


@normalize_token.register(UUID)
def _normalize_uuid(obj):
    return ("uuid", str(obj))


@normalize_token.register(PurePath)
def _normalize_path(obj):
    return ("path", str(obj))


@normalize_token.register(enum.Enum)
def _normalize_enum(obj):
    return ("enum", type(obj).__module__, type(obj).__qualname__, obj.name)


@normalize_token.register(tuple)
def _normalize_tuple(obj):
    return ("tuple", tuple(normalize_token(item) for item in obj))


@normalize_token.register(list)
def _normalize_list(obj):
    return _with_cycle_check(obj, lambda: ("list", tuple(normalize_token(item) for item in obj)))


@normalize_token.register(set)
def _normalize_set(obj):
    return ("set", tuple(sorted((normalize_token(item) for item in obj), key=repr)))


@normalize_token.register(frozenset)
def _normalize_frozenset(obj):
    return ("frozenset", tuple(sorted((normalize_token(item) for item in obj), key=repr)))


@normalize_token.register(dict)
def _normalize_dict(obj):
    return _with_cycle_check(
        obj,
        lambda: ("dict", tuple(sorted(((normalize_token(k), normalize_token(v)) for k, v in obj.items()), key=repr))),
    )


@normalize_token.register(OrderedDict)
def _normalize_ordered_dict(obj):
    return _with_cycle_check(obj, lambda: ("__ordereddict__", tuple((normalize_token(k), normalize_token(v)) for k, v in obj.items())))


@normalize_token.register(complex)
def _normalize_complex(obj):
    return ("complex", obj.real, obj.imag)


@normalize_token.register(type(Ellipsis))
def _normalize_ellipsis(obj):
    return ("ellipsis",)


@normalize_token.register(slice)
def _normalize_slice(obj):
    return ("slice", normalize_token(obj.start), normalize_token(obj.stop), normalize_token(obj.step))


@normalize_token.register(type(len))  # builtin_function_or_method
def _normalize_builtin(obj):
    # __self__ is the bound instance for methods like [].append, the module for unbound builtins like
    # math.sin (where __self__ is the math module), or absent for some C-level callables. Distinguish
    # the module case (cheap by name) from the bound-instance case (recurse) so math.sin vs cmath.sin
    # don't collide and `[1,2].append` vs `[3,4].append` differ by their bound list contents.
    self_obj = getattr(obj, "__self__", None)
    module = getattr(obj, "__module__", None) or ""
    if isinstance(self_obj, ModuleType):
        return ("builtin", module, obj.__qualname__, ("module", self_obj.__name__))
    return ("builtin", module, obj.__qualname__, normalize_token(self_obj))


@normalize_token.register(Decimal)
def _normalize_decimal(obj):
    return ("decimal", str(obj))


@normalize_token.register(partial)
def _normalize_partial(obj):
    return ("partial", normalize_token(obj.func), normalize_token(obj.args), normalize_token(sorted(obj.keywords.items())))


@normalize_token.register(MappingProxyType)
def _normalize_mappingproxy(obj):
    # Preserve the proxy's iteration order rather than sorting (the underlying mapping might be an
    # OrderedDict where order is semantically meaningful). Tagged separately so a proxy is never
    # mistaken for an equivalent plain dict.
    return _with_cycle_check(obj, lambda: ("mappingproxy", tuple((normalize_token(k), normalize_token(v)) for k, v in obj.items())))


@normalize_token.register(CodeType)
def _normalize_code(obj):
    # Faithfully include identifier/signature fields so two functions that differ only in attribute
    # access, local variable names, or signature don't collide. Docstring stripping is left to
    # _hash_function_bytecode, which knows it's looking at function code (not a `compile(..., "exec")`
    # block whose first const may be real program data).
    return (
        "code",
        obj.co_code,
        tuple(normalize_token(c) for c in obj.co_consts),
        obj.co_names,
        obj.co_varnames,
        obj.co_freevars,
        obj.co_cellvars,
        obj.co_argcount,
        obj.co_kwonlyargcount,
        obj.co_posonlyargcount,
        obj.co_flags,
    )


@normalize_token.register(type(lambda: None))  # FunctionType
def _normalize_function(obj):
    seen = _hashing_functions.get()
    created = seen is None
    if created:
        seen = set()
        token = _hashing_functions.set(seen)
    elif id(obj) in seen:
        return ("__function_cycle__", getattr(obj, "__qualname__", "?"))
    seen.add(id(obj))
    try:
        result = ("__function__", _hash_function_bytecode(obj))
    finally:
        seen.discard(id(obj))
        if created:
            _hashing_functions.reset(token)
    return result


@normalize_token.register(MethodType)
def _normalize_method(obj):
    return ("__method__", normalize_token(obj.__func__), normalize_token(obj.__self__))


@normalize_token.register(MethodWrapperType)
def _normalize_method_wrapper(obj):
    # method-wrapper objects (e.g. `{}.__init__`) have no __func__; key by name and bound instance.
    return ("__method_wrapper__", obj.__name__, normalize_token(obj.__self__))


@normalize_token.register(type)
def _normalize_type(obj):
    return ("type", f"{obj.__module__}.{obj.__qualname__}")


@normalize_token.register(_PydanticBaseModel)
def _normalize_pydantic_basemodel(obj):
    """Hash a pydantic model by its non-excluded fields (and any ``extra='allow'`` extras)."""

    def build():
        type_path = f"{type(obj).__module__}.{type(obj).__qualname__}"
        model_fields = type(obj).model_fields
        # Iterate model_fields directly rather than `for k, v in obj` so models that override __iter__
        # (or otherwise hide fields) still tokenize structurally.
        fields = tuple((name, normalize_token(getattr(obj, name))) for name, info in model_fields.items() if not info.exclude)
        extras = getattr(obj, "__pydantic_extra__", None) or {}
        extras_canonical = tuple(sorted(((k, normalize_token(v)) for k, v in extras.items()), key=repr))
        return ("pydantic", type_path, fields, extras_canonical)

    return _with_cycle_check(obj, build)


def _register_numpy() -> None:
    try:
        import numpy as np
    except ImportError:  # pragma: no cover
        return

    # Perf note: dask streams a contiguous view directly into the hasher; we materialize via tobytes() and
    # have a separate fast-path for object-dtype arrays. ~10x slower than dask on large numeric arrays and
    # ~18x slower on object-dtype arrays — fine for typical config-style cache keys, room for improvement
    # if multi-MB arrays become routine cache inputs.
    @normalize_token.register(np.ndarray)
    def _normalize_ndarray(obj):
        # Object-dtype arrays store PyObject* pointers; tobytes() embeds process-local addresses, so recurse
        # element-wise instead. Cycle detection is keyed on the array because tolist() returns a fresh list.
        if obj.dtype.hasobject:
            return _with_cycle_check(obj, lambda: ("ndarray", str(obj.dtype), obj.shape, normalize_token(obj.tolist())))
        return ("ndarray", str(obj.dtype), obj.shape, hashlib.sha256(np.ascontiguousarray(obj).tobytes()).hexdigest())

    @normalize_token.register(np.ma.MaskedArray)
    def _normalize_masked_array(obj):
        # MaskedArray is a subclass of ndarray, so without an explicit handler the mask would be silently
        # dropped. Normalize the underlying data as a plain ndarray and include the mask + fill_value.
        data = normalize_token(np.asarray(obj.data))
        mask = obj.mask
        mask_canonical = bool(mask) if mask is np.ma.nomask else normalize_token(np.asarray(mask))
        return ("masked_array", data, mask_canonical, normalize_token(obj.fill_value))

    @normalize_token.register(np.generic)
    def _normalize_np_scalar(obj):
        return ("np_scalar", str(type(obj).__name__), obj.item())


def _register_pandas() -> None:
    try:
        import pandas as pd
    except ImportError:  # pragma: no cover
        return

    # Only Timestamp has a structural handler; DataFrame/Series/Index fall through to the cloudpickle
    # fallback. That works today but is fragile across pandas version upgrades — a follow-up could add
    # structural handlers (matching dask) for stability and a perf win on large frames.
    @normalize_token.register(pd.Timestamp)
    def _normalize_pd_timestamp(obj):
        return ("pd_timestamp", obj.isoformat())


_register_numpy()
_register_pandas()


def tokenize(*args: Any, **kwargs: Any) -> str:
    """Return a deterministic SHA-256 hex digest of the given args/kwargs (variadic to match ``dask.base.tokenize``)."""
    payload = (args, kwargs) if kwargs else args
    visited_token = _visited.set(set())
    try:
        return _sha256_hexdigest(repr(normalize_token(payload)))
    finally:
        _visited.reset(visited_token)


def compute_data_token(value: Any) -> str:
    """Compute a deterministic data token for a single value."""
    visited_token = _visited.set(set())
    try:
        return _sha256_hexdigest(repr(normalize_token(value)))
    finally:
        _visited.reset(visited_token)


def compute_cache_token(*, data_values: Iterable[Any] = (), behavior_classes: Iterable[type] = ()) -> str:
    """Compute a cache token by combining data and behavior tokens.

    Args:
        data_values: Values whose serialized data should affect the cache key.
        behavior_classes: Classes whose behavior tokens should affect the cache key.
    """

    tokens = [compute_data_token(value) for value in data_values]
    for cls in behavior_classes:
        behavior = compute_behavior_token(cls)
        if behavior is not None:
            tokens.append(behavior)
    return compute_data_token(tuple(tokens))


# Methods that are never behavior-relevant (pydantic/python internals)
_SKIPPED_METHODS = frozenset(
    {
        "__eq__",
        "__hash__",
        "__repr__",
        "__str__",
        "__init__",
        "__init_subclass__",
        "__class_getitem__",
        "__get_pydantic_core_schema__",
        "__get_pydantic_json_schema__",
        "model_post_init",
    }
)


# ---------------------------------------------------------------------------
# Behavior hashing — bytecode-based fingerprinting of class methods
# ---------------------------------------------------------------------------


def _unwrap_function(func: object) -> Optional[Callable]:
    """Unwrap descriptors and decorator chains to get the underlying function.

    Handles ``classmethod``, ``staticmethod``, ``property`` (fget), and
    ``functools.wraps``-style ``__wrapped__`` chains (e.g. ``@Flow.call``).
    Returns ``None`` if the underlying object has no ``__code__``.
    """
    if isinstance(func, classmethod):
        func = func.__func__
    elif isinstance(func, staticmethod):
        func = func.__func__
    elif isinstance(func, property):
        func = func.fget
        if func is None:
            return None

    # Unwrap decorator chains (e.g. @Flow.call sets __wrapped__)
    try:
        func = inspect.unwrap(func)
    except (TypeError, ValueError):
        pass

    if not callable(func) or not hasattr(func, "__code__"):
        return None
    return func


def _function_state(func: Callable) -> Tuple[Any, Any, Tuple[Tuple[str, bool, Any], ...] | None]:
    """Return defaults/kwdefaults/closure state that affects runtime behavior."""

    kwdefaults = getattr(func, "__kwdefaults__", None)
    if kwdefaults is not None:
        kwdefaults = tuple(sorted(kwdefaults.items()))

    closure = getattr(func, "__closure__", None)
    closure_state = None
    if closure:
        closure_state = []
        for name, cell in zip(func.__code__.co_freevars, closure):
            try:
                closure_state.append((name, True, cell.cell_contents))
            except ValueError:
                closure_state.append((name, False, None))
        closure_state = tuple(closure_state)

    return getattr(func, "__defaults__", None), kwdefaults, closure_state


def _hash_function_bytecode(func: Callable) -> Optional[str]:
    """Return a SHA-256 hex digest of a function's behavior-relevant state.

    Unwraps decorator chains (``inspect.unwrap``) so that wrappers like
    ``@Flow.call`` do not mask the implementation. Includes the recursively
    normalized code object (with the leading docstring const stripped here,
    where we know we have a function body), positional and keyword-only
    defaults, and closure cell contents.

    Returns ``None`` for objects without ``__code__``.
    """
    unwrapped = _unwrap_function(func)
    if unwrapped is None:
        return None
    code = unwrapped.__code__
    consts = code.co_consts
    # Strip the docstring constant if present. In Python < 3.14, a None sentinel occupies
    # co_consts[0] when there is no docstring; in Python >= 3.14 that slot is absent.
    # Only strip when the first constant actually matches the function's __doc__.
    doc = getattr(unwrapped, "__doc__", None)
    if doc is not None and consts and consts[0] == doc:
        consts = consts[1:]
    elif consts and consts[0] is None:
        consts = consts[1:]
    code_canonical = (
        "code",
        code.co_code,
        tuple(normalize_token(c) for c in consts),
        code.co_names,
        code.co_varnames,
        code.co_freevars,
        code.co_cellvars,
        code.co_argcount,
        code.co_kwonlyargcount,
        code.co_posonlyargcount,
        code.co_flags,
    )
    return _sha256_hexdigest(repr(code_canonical), compute_data_token(_function_state(unwrapped)))


def _dependency_info(dep: object, *, _visited: Tuple[type, ...]) -> Optional[Tuple[Tuple[str, str, str, str], str, str]]:
    """Return deterministic identity, name, and token for a dependency entry."""

    if isinstance(dep, type):
        module = getattr(dep, "__module__", "")
        qualname = getattr(dep, "__qualname__", getattr(dep, "__name__", repr(dep)))
        behavior = compute_behavior_token(dep, _visited=_visited)
        if behavior is None:
            return None
        return ("class", module, qualname, behavior), f"__dep_class__:{qualname}", behavior

    unwrapped = _unwrap_function(dep)
    if unwrapped is None:
        return None
    module = getattr(unwrapped, "__module__", "")
    qualname = getattr(unwrapped, "__qualname__", getattr(unwrapped, "__name__", repr(unwrapped)))
    behavior = _hash_function_bytecode(unwrapped)
    if behavior is None:
        return None
    return ("callable", module, qualname, behavior), f"__dep__:{qualname}", behavior


def _collect_methods(cls: type) -> List[Tuple[str, Callable]]:
    """Collect callable methods from *cls* (walking MRO).

    Methods are collected with MRO override semantics: for each method name,
    the first definition found in the MRO wins. This means a subclass's
    ``__call__`` overrides the base class's even if the subclass doesn't
    redefine every method.

    Own methods are sorted alphabetically.

    Internal framework attributes (``__ccflow_*``) and pydantic/python
    boilerplate methods are skipped.
    """
    # Walk MRO to collect methods with override semantics
    seen_names = set()
    methods = []
    for klass in cls.__mro__:
        if klass is object or klass is _PydanticBaseModel:
            break
        for name, value in klass.__dict__.items():
            if name in seen_names:
                continue
            seen_names.add(name)
            if name.startswith("__ccflow_"):
                continue
            if name in _SKIPPED_METHODS:
                continue
            if isinstance(value, (classmethod, staticmethod, property)) or callable(value):
                methods.append((name, value))

    methods.sort(key=lambda pair: pair[0])

    return methods


def _collect_dependency_hashes(cls: type, *, _visited: Tuple[type, ...]) -> List[Tuple[str, str]]:
    """Collect hashed ``__ccflow_tokenizer_deps__`` entries from the full MRO.

    Dependencies are merged across the MRO, deduplicated, and sorted
    deterministically so that declaration order does not affect the hash.

    Dependency entries may be either:
    - function-like objects hashable via ``_hash_function_bytecode()``
    - classes, in which case ``compute_behavior_token(dep_class)`` is included
    """
    deps = []
    seen_dep_keys = set()
    for klass in cls.__mro__:
        extra_deps = klass.__dict__.get("__ccflow_tokenizer_deps__")
        if extra_deps is None:
            continue
        for dep in extra_deps:
            dep_info = _dependency_info(dep, _visited=_visited)
            if dep_info is None:
                continue
            dep_key, dep_name, dep_token = dep_info
            if dep_key in seen_dep_keys:
                continue
            seen_dep_keys.add(dep_key)
            deps.append((dep_key, dep_name, dep_token))

    deps.sort(key=lambda item: item[0])
    return [(dep_name, dep_token) for _, dep_name, dep_token in deps]


def compute_behavior_token(cls: type, *, _visited: Tuple[type, ...] = ()) -> Optional[str]:
    """Compute a SHA-256 behavior token for *cls* based on its method bytecode.

    The token captures behavior-relevant state for every method in *cls*'s MRO
    (with standard override semantics): bytecode, constants (minus docstrings),
    defaults, keyword-only defaults, and closure cell contents. It also
    includes any functions or classes listed in ``cls.__ccflow_tokenizer_deps__``.

    Decorator chains (e.g. ``@Flow.call``) are automatically unwrapped so
    that the hash reflects the user's implementation, not the wrapper.

    ``__ccflow_tokenizer_deps__`` values are merged across the full MRO, so
    subclasses can add dependencies without dropping inherited ones. Class
    entries contribute their own ``compute_behavior_token()`` recursively.

    Results are cached on the class in ``cls.__ccflow_tokenizer_cache__``.
    The cache is stored directly on the class (not inherited), so subclass
    tokens are independent of parent tokens.

    Returns ``None`` if the class has no hashable methods.

    .. note::

        Monkey-patching methods on an existing class after its behavior token
        has been computed will **not** invalidate the cached token. Redefining
        the class (e.g. in Jupyter) creates a new class object and works fine.
    """
    if cls in _visited:
        raise TypeError(f"Recursive __ccflow_tokenizer_deps__ class dependency detected for {cls.__module__}.{cls.__qualname__}")

    # Check cache on cls itself (not inherited)
    cache = cls.__dict__.get("__ccflow_tokenizer_cache__")
    if cache is not None:
        return cache

    visited = _visited + (cls,)
    methods = _collect_methods(cls)
    method_hashes = [(name, h) for name, func in methods if (h := _hash_function_bytecode(func)) is not None]
    method_hashes.extend(_collect_dependency_hashes(cls, _visited=visited))

    if not method_hashes:
        return None

    token = _sha256_hexdigest(repr(method_hashes))

    try:
        cls.__ccflow_tokenizer_cache__ = token
    except (TypeError, AttributeError):
        pass  # e.g. C extension types that don't allow attribute setting

    return token
