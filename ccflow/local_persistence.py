"""Register local-scope classes on ccflow.base so PyObjectPath can import them.

Classes defined in functions or via create_model aren't normally importable.
We give them a unique name and put them on ccflow.base. We keep __module__ and
__qualname__ unchanged so cloudpickle can still serialize the class definition.
"""

import re
import sys
import uuid
from typing import Any, Type

__all__ = ("LOCAL_ARTIFACTS_MODULE_NAME", "_register_local_subclass_if_needed", "_register", "_sync_to_module")

LOCAL_ARTIFACTS_MODULE_NAME = "ccflow.base"


def _is_importable(cls: Type[Any]) -> bool:
    """Can cls be imported via its __module__.__qualname__ path?"""
    qualname = getattr(cls, "__qualname__", "")
    module_name = getattr(cls, "__module__", "")

    if "<locals>" in qualname or module_name == "__main__":
        return False

    module = sys.modules.get(module_name)
    if not module:
        return False

    obj = module
    for part in qualname.split("."):
        obj = getattr(obj, part, None)
        if obj is None:
            return False
    return obj is cls


def _register(cls: Type[Any]) -> None:
    """Give cls a unique name and put it on ccflow.base."""
    name = re.sub(r"[^0-9A-Za-z_]", "_", cls.__name__ or "Model").strip("_") or "Model"
    if name[0].isdigit():
        name = f"_{name}"
    unique = f"_Local_{name}_{uuid.uuid4().hex[:12]}"

    setattr(sys.modules["ccflow.base"], unique, cls)
    cls.__ccflow_import_path__ = f"{LOCAL_ARTIFACTS_MODULE_NAME}.{unique}"


def _sync_to_module(cls: Type[Any]) -> None:
    """Ensure cls is on ccflow.base in this process (for cross-process unpickle)."""
    path = getattr(cls, "__ccflow_import_path__", "")
    if path.startswith(LOCAL_ARTIFACTS_MODULE_NAME + "."):
        name = path.rsplit(".", 1)[-1]
        base = sys.modules["ccflow.base"]
        if getattr(base, name, None) is not cls:
            setattr(base, name, cls)


def _register_local_subclass_if_needed(cls: Type[Any]) -> None:
    """Register cls if not importable. Called from PyObjectPath and __reduce_ex__."""
    if "__ccflow_import_path__" in cls.__dict__:
        _sync_to_module(cls)
        return
    if "__ccflow_importable__" in cls.__dict__:
        return

    from ccflow.base import BaseModel

    if not (isinstance(cls, type) and issubclass(cls, BaseModel)):
        return

    if _is_importable(cls):
        cls.__ccflow_importable__ = True
    else:
        _register(cls)
