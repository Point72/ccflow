"""Helpers for persisting BaseModel-derived classes defined inside local scopes."""

from __future__ import annotations

import re
import sys
from collections import defaultdict
from itertools import count
from types import ModuleType
from typing import Any, DefaultDict, Type

__all__ = ("LOCAL_ARTIFACTS_MODULE_NAME", "register_local_subclass")


LOCAL_ARTIFACTS_MODULE_NAME = "ccflow._local_artifacts"
_LOCAL_MODULE_DOC = "Auto-generated BaseModel subclasses created outside importable modules."

_SANITIZE_PATTERN = re.compile(r"[^0-9A-Za-z_]")
_LOCAL_KIND_COUNTERS: DefaultDict[str, count] = defaultdict(lambda: count())


def _ensure_module(name: str, doc: str) -> ModuleType:
    """Ensure the dynamic module exists so import paths remain stable."""
    module = sys.modules.get(name)
    if module is None:
        module = ModuleType(name, doc)
        sys.modules[name] = module
        parent_name, _, attr = name.rpartition(".")
        if parent_name:
            parent_module = sys.modules.get(parent_name)
            if parent_module and not hasattr(parent_module, attr):
                setattr(parent_module, attr, module)
    return module


_LOCAL_ARTIFACTS_MODULE = _ensure_module(LOCAL_ARTIFACTS_MODULE_NAME, _LOCAL_MODULE_DOC)


def _needs_registration(cls: Type[Any]) -> bool:
    module = getattr(cls, "__module__", "")
    qualname = getattr(cls, "__qualname__", "")
    return "<locals>" in qualname or module == "__main__"


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


def register_local_subclass(cls: Type[Any], *, kind: str = "model") -> None:
    """Register BaseModel subclasses created in local scopes."""
    if getattr(cls, "__module__", "").startswith(LOCAL_ARTIFACTS_MODULE_NAME):
        return
    if not _needs_registration(cls):
        return

    name_hint = f"{getattr(cls, '__module__', '')}.{getattr(cls, '__qualname__', '')}"
    kind_slug = _sanitize_identifier(kind, "model").lower()
    unique_name = _build_unique_name(kind_slug=kind_slug, name_hint=name_hint)
    setattr(_LOCAL_ARTIFACTS_MODULE, unique_name, cls)
    cls.__module__ = _LOCAL_ARTIFACTS_MODULE.__name__
    cls.__qualname__ = unique_name
    setattr(cls, "__ccflow_dynamic_origin__", name_hint)
