# ruff: noqa: F401
"""Tokenization utilities for ccflow models.

Re-exports ``normalize_token`` and ``tokenize`` from dask for data hashing.
Adds behavior hashing: deterministic SHA-256 fingerprints of class method
bytecode, useful for cache-key invalidation when callable logic changes.
"""

import hashlib
import inspect
import logging
from typing import Callable, List, Optional, Tuple

from dask.base import normalize_token, tokenize

__all__ = [
    "normalize_token",
    "tokenize",
    "compute_behavior_token",
]

logger = logging.getLogger(__name__)

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


def _hash_function_bytecode(func: Callable) -> Optional[str]:
    """Return a SHA-256 hex digest of a function's bytecode and constants.

    The function is first unwrapped through any decorator chains, so that
    e.g. ``@Flow.call`` wrappers do not mask the real implementation.

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
    return h.hexdigest()


def _collect_methods(cls: type) -> List[Tuple[str, Callable]]:
    """Collect callable methods from *cls* (walking MRO) plus ``__ccflow_tokenizer_deps__``.

    Methods are collected with MRO override semantics: for each method name,
    the first definition found in the MRO wins. This means a subclass's
    ``__call__`` overrides the base class's even if the subclass doesn't
    redefine every method.

    Own methods are sorted alphabetically. Dependencies are sorted by
    qualified name so that declaration order does not affect the hash.

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

    # Collect __ccflow_tokenizer_deps__ (also walk MRO, first definition wins)
    extra_deps = None
    for klass in cls.__mro__:
        if "__ccflow_tokenizer_deps__" in klass.__dict__:
            extra_deps = klass.__dict__["__ccflow_tokenizer_deps__"]
            break

    if extra_deps is not None:
        deps = []
        for func in extra_deps:
            unwrapped = _unwrap_function(func) or func
            if callable(unwrapped):
                func_id = getattr(unwrapped, "__qualname__", getattr(unwrapped, "__name__", repr(unwrapped)))
                deps.append((f"__dep__:{func_id}", func))
        deps.sort(key=lambda pair: pair[0])
        methods.extend(deps)

    return methods


def compute_behavior_token(cls: type) -> Optional[str]:
    """Compute a SHA-256 behavior token for *cls* based on its method bytecode.

    The token captures bytecode (``co_code``) and constants (``co_consts``,
    minus docstrings) of every method in *cls*'s MRO (with standard override
    semantics), plus any standalone functions listed in
    ``cls.__ccflow_tokenizer_deps__``.

    Decorator chains (e.g. ``@Flow.call``) are automatically unwrapped so
    that the hash reflects the user's implementation, not the wrapper.

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
