# ruff: noqa: F401
"""Tokenization utilities for ccflow models.

Re-exports ``normalize_token`` and ``tokenize`` from dask for data hashing.
Adds thin wrappers around dask-based data hashing and ccflow-specific
behavior hashing, useful for cache-key invalidation when callable logic
changes.
"""

import hashlib
import inspect
import logging
from typing import Any, Callable, List, Optional, Tuple

from dask.base import normalize_token, tokenize

__all__ = [
    "compute_data_token",
    "normalize_token",
    "tokenize",
    "compute_behavior_token",
]

logger = logging.getLogger(__name__)


def compute_data_token(value: Any) -> str:
    """Compute a deterministic data token using dask's tokenization."""

    return tokenize(value)


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
    h = hashlib.sha256(code.co_code)
    # Include constants (skip first if it's the docstring)
    consts = code.co_consts
    if consts and isinstance(consts[0], str):
        consts = consts[1:]
    h.update(repr(consts).encode("utf-8"))
    h.update(compute_data_token(_function_state(unwrapped)).encode("utf-8"))
    return h.hexdigest()


def _dependency_sort_key(func: Callable) -> Tuple[str, str, str]:
    """Return a deterministic identity for dependency sorting/deduping."""

    unwrapped = _unwrap_function(func) or func
    module = getattr(unwrapped, "__module__", "")
    qualname = getattr(unwrapped, "__qualname__", getattr(unwrapped, "__name__", repr(unwrapped)))
    behavior = _hash_function_bytecode(unwrapped) or ""
    return module, qualname, behavior


def _collect_methods(cls: type) -> List[Tuple[str, Callable]]:
    """Collect callable methods from *cls* (walking MRO) plus ``__ccflow_tokenizer_deps__``.

    Methods are collected with MRO override semantics: for each method name,
    the first definition found in the MRO wins. This means a subclass's
    ``__call__`` overrides the base class's even if the subclass doesn't
    redefine every method.

    Own methods are sorted alphabetically. Dependencies are merged across the
    MRO, deduplicated, and sorted deterministically so that declaration order
    does not affect the hash.

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

    # Collect __ccflow_tokenizer_deps__ from the full MRO. Subclasses may add
    # deps without losing inherited ones.
    deps = []
    seen_dep_keys = set()
    for klass in cls.__mro__:
        extra_deps = klass.__dict__.get("__ccflow_tokenizer_deps__")
        if extra_deps is None:
            continue
        for func in extra_deps:
            unwrapped = _unwrap_function(func) or func
            if not callable(unwrapped):
                continue
            dep_key = _dependency_sort_key(func)
            if dep_key in seen_dep_keys:
                continue
            seen_dep_keys.add(dep_key)
            deps.append((dep_key, func))

    deps.sort(key=lambda pair: pair[0])
    methods.extend((f"__dep__:{dep_key[1]}", func) for dep_key, func in deps)

    return methods


def compute_behavior_token(cls: type) -> Optional[str]:
    """Compute a SHA-256 behavior token for *cls* based on its method bytecode.

    The token captures behavior-relevant state for every method in *cls*'s MRO
    (with standard override semantics): bytecode, constants (minus docstrings),
    defaults, keyword-only defaults, and closure cell contents. It also
    includes any standalone functions listed in ``cls.__ccflow_tokenizer_deps__``.

    Decorator chains (e.g. ``@Flow.call``) are automatically unwrapped so
    that the hash reflects the user's implementation, not the wrapper.

    ``__ccflow_tokenizer_deps__`` values are merged across the full MRO, so
    subclasses can add dependencies without dropping inherited ones.

    Results are cached on the class in ``cls.__behavior_token_cache__``.
    The cache is stored directly on the class (not inherited), so subclass
    tokens are independent of parent tokens.

    Returns ``None`` if the class has no hashable methods.

    .. note::

        Monkey-patching methods on an existing class after its behavior token
        has been computed will **not** invalidate the cached token. Redefining
        the class (e.g. in Jupyter) creates a new class object and works fine.
    """
    # Check cache on cls itself (not inherited)
    cache = cls.__dict__.get("__behavior_token_cache__")
    if cache is not None:
        return cache

    methods = _collect_methods(cls)
    method_hashes = [(name, h) for name, func in methods if (h := _hash_function_bytecode(func)) is not None]

    if not method_hashes:
        return None

    token = hashlib.sha256(repr(method_hashes).encode("utf-8")).hexdigest()

    try:
        cls.__behavior_token_cache__ = token
    except (TypeError, AttributeError):
        pass  # e.g. C extension types that don't allow attribute setting

    return token
