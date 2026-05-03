# ruff: noqa: F401
"""Tokenization utilities for ccflow models.

Re-exports ``normalize_token`` and ``tokenize`` from dask for data hashing.
Adds helpers for dask-based data hashing, ccflow-specific behavior hashing,
and combined cache-token hashing, useful for cache-key invalidation when
callable logic changes.
"""

import hashlib
import inspect
from typing import Any, Callable, Iterable, List, Optional, Tuple

from dask.base import normalize_token, tokenize

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


def compute_data_token(value: Any) -> str:
    """Compute a deterministic data token using dask's tokenization."""

    return tokenize(value)


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

    The function is first unwrapped through any decorator chains, so that
    e.g. ``@Flow.call`` wrappers do not mask the real implementation.

    In addition to ``co_code`` and ``co_consts``, this includes:
    - positional defaults (``__defaults__``)
    - keyword-only defaults (``__kwdefaults__``)
    - closure cell contents
    so that behavior changes that do not affect bytecode alone still change
    the token.

    Returns ``None`` for objects without ``__code__`` (C builtins, etc.).
    """
    unwrapped = _unwrap_function(func)
    if unwrapped is None:
        return None
    code = unwrapped.__code__
    # Include constants (skip first if it's the docstring)
    consts = code.co_consts
    if consts and isinstance(consts[0], str):
        consts = consts[1:]
    return _sha256_hexdigest(
        code.co_code,
        repr(consts),
        compute_data_token(_function_state(unwrapped)),
    )


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
        if klass is object:
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
