"""Helpers for persisting BaseModel-derived classes defined inside local scopes."""

from __future__ import annotations

import importlib.abc
import importlib.util
import re
import sys
from collections import defaultdict
from itertools import count
from typing import TYPE_CHECKING, Any, DefaultDict, Type

if TYPE_CHECKING:
    from importlib.machinery import ModuleSpec
    from types import ModuleType

__all__ = ("LOCAL_ARTIFACTS_MODULE_NAME",)

LOCAL_ARTIFACTS_MODULE_NAME = "ccflow._local_artifacts"
_LOCAL_MODULE_DOC = "Auto-generated BaseModel subclasses created outside importable modules."

_SANITIZE_PATTERN = re.compile(r"[^0-9A-Za-z_]")
_LOCAL_KIND_COUNTERS: DefaultDict[str, count] = defaultdict(lambda: count())
_LOCAL_ARTIFACTS_MODULE: "ModuleType | None" = None


class _LocalArtifactsLoader(importlib.abc.Loader):
    """Minimal loader so importlib can reload our synthetic module if needed."""

    def __init__(self, *, doc: str) -> None:
        self._doc = doc

    def create_module(self, spec: "ModuleSpec") -> "ModuleType | None":
        """Defer to default module creation (keeping importlib from recursing)."""
        return None

    def exec_module(self, module: "ModuleType") -> None:
        module.__doc__ = module.__doc__ or self._doc


class _LocalArtifactsFinder(importlib.abc.MetaPathFinder):
    """Ensures importlib can rediscover the synthetic module when reloading."""

    def find_spec(
        self,
        fullname: str,
        path: Any,
        target: "ModuleType | None" = None,
    ) -> "ModuleSpec | None":
        if fullname != LOCAL_ARTIFACTS_MODULE_NAME:
            return None
        return _build_module_spec(fullname, _LOCAL_MODULE_DOC)


def _build_module_spec(name: str, doc: str) -> "ModuleSpec":
    loader = _LocalArtifactsLoader(doc=doc)
    spec = importlib.util.spec_from_loader(
        name,
        loader=loader,
        origin="ccflow.local_persistence:_ensure_module",
    )
    if spec is None:
        raise ImportError(f"Unable to create spec for dynamic module {name!r}.")
    spec.has_location = False
    return spec


def _ensure_finder_installed() -> None:
    for finder in sys.meta_path:
        if isinstance(finder, _LocalArtifactsFinder):
            return
    sys.meta_path.insert(0, _LocalArtifactsFinder())


def _ensure_module(name: str, doc: str) -> "ModuleType":
    """Materialize the synthetic module with a real spec/loader so importlib treats it like disk-backed code.

    We only do this on demand, but once built the finder/spec/loader plumbing
    keeps reload, pickling, and other importlib consumers happy. The Python docs recommend this approach instead of creating modules directly via the constructor."""
    _ensure_finder_installed()
    module = sys.modules.get(name)
    if module is None:
        # Create a proper ModuleSpec + loader so importlib reloads and introspection behave
        # the same way they would for filesystem-backed modules
        spec = _build_module_spec(name, doc)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)  # type: ignore[union-attr]
        sys.modules[name] = module
        parent_name, _, attr = name.rpartition(".")
        if parent_name:
            parent_module = sys.modules.get(parent_name)
            if parent_module and not hasattr(parent_module, attr):
                setattr(parent_module, attr, module)
    return module


def _get_local_artifacts_module() -> "ModuleType":
    """Lazily materialize the synthetic module to avoid errors during creation until needed."""
    global _LOCAL_ARTIFACTS_MODULE
    if _LOCAL_ARTIFACTS_MODULE is None:
        _LOCAL_ARTIFACTS_MODULE = _ensure_module(LOCAL_ARTIFACTS_MODULE_NAME, _LOCAL_MODULE_DOC)
    return _LOCAL_ARTIFACTS_MODULE


def _needs_registration(cls: Type[Any]) -> bool:
    qualname = getattr(cls, "__qualname__", "")
    return "<locals>" in qualname


def _sanitize_identifier(value: str, fallback: str) -> str:
    sanitized = _SANITIZE_PATTERN.sub("_", value or "")
    sanitized = sanitized.strip("_") or fallback
    if sanitized[0].isdigit():
        sanitized = f"_{sanitized}"
    return sanitized


def _build_unique_name(*, kind_slug: str, name_hint: str) -> str:
    sanitized_hint = _sanitize_identifier(name_hint, "BaseModel")
    counter = _LOCAL_KIND_COUNTERS[kind_slug]
    return f"{kind_slug}__{sanitized_hint}__{next(counter)}"


def _register_local_subclass(cls: Type[Any], *, kind: str = "model") -> None:
    """Register BaseModel subclasses created in local scopes."""
    if getattr(cls, "__module__", "").startswith(LOCAL_ARTIFACTS_MODULE_NAME):
        return
    if not _needs_registration(cls):
        return

    name_hint = f"{getattr(cls, '__module__', '')}.{getattr(cls, '__qualname__', '')}"
    kind_slug = _sanitize_identifier(kind, "model").lower()
    unique_name = _build_unique_name(kind_slug=kind_slug, name_hint=name_hint)
    artifacts_module = _get_local_artifacts_module()
    setattr(artifacts_module, unique_name, cls)
    cls.__module__ = artifacts_module.__name__
    cls.__qualname__ = unique_name
    setattr(cls, "__ccflow_dynamic_origin__", name_hint)
