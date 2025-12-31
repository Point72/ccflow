"""Register local-scope classes on a module so PyObjectPath can import them.

Classes defined in functions (with '<locals>' in __qualname__) aren't normally importable.
We give them a unique name and register them on this module (ccflow.local_persistence).
We keep __module__ and __qualname__ unchanged so cloudpickle can still serialize the
class definition.

This module provides:
- _register(cls): Register a local class with a unique import path
- _sync_to_module(cls): Ensure a class with __ccflow_import_path__ is on the module
  (used for cross-process unpickle scenarios)
- create_ccflow_model: Wrapper around pydantic.create_model that registers the created model
"""

import re
import sys
import uuid
from typing import Any, Type

__all__ = ("LOCAL_ARTIFACTS_MODULE_NAME", "create_ccflow_model")

LOCAL_ARTIFACTS_MODULE_NAME = "ccflow.local_persistence"


def _register(cls: Type[Any]) -> None:
    """Give cls a unique name and register it on the artifacts module.

    This sets __ccflow_import_path__ on the class without modifying __module__ or
    __qualname__, preserving cloudpickle's ability to serialize the class definition.
    """
    # Sanitize the class name to be a valid Python identifier
    name = re.sub(r"[^0-9A-Za-z_]", "_", cls.__name__ or "Model").strip("_") or "Model"
    if name[0].isdigit():
        name = f"_{name}"
    unique = f"_Local_{name}_{uuid.uuid4().hex[:12]}"

    setattr(sys.modules[LOCAL_ARTIFACTS_MODULE_NAME], unique, cls)
    cls.__ccflow_import_path__ = f"{LOCAL_ARTIFACTS_MODULE_NAME}.{unique}"


def _sync_to_module(cls: Type[Any]) -> None:
    """Ensure cls is registered on the artifacts module in this process.

    This handles cross-process unpickle scenarios where cloudpickle recreates the class
    with __ccflow_import_path__ already set (from the original process), but the class
    isn't yet registered on ccflow.local_persistence in the new process.
    """
    path = getattr(cls, "__ccflow_import_path__", "")
    if path.startswith(LOCAL_ARTIFACTS_MODULE_NAME + "."):
        name = path.rsplit(".", 1)[-1]
        base = sys.modules[LOCAL_ARTIFACTS_MODULE_NAME]
        if getattr(base, name, None) is not cls:
            setattr(base, name, cls)


def create_ccflow_model(__model_name: str, *, __base__: Any = None, **field_definitions: Any) -> Type[Any]:
    """Create a dynamic ccflow model and register it for PyObjectPath serialization.

    Wraps pydantic's create_model and registers the model so it can be serialized
    via PyObjectPath, including across processes (e.g., with Ray).

    Example:
        >>> from ccflow import ContextBase, create_ccflow_model
        >>> MyContext = create_ccflow_model(
        ...     "MyContext",
        ...     __base__=ContextBase,
        ...     name=(str, ...),
        ...     value=(int, 0),
        ... )
        >>> ctx = MyContext(name="test", value=42)
    """
    from pydantic import create_model as pydantic_create_model

    model = pydantic_create_model(__model_name, __base__=__base__, **field_definitions)

    # Register if it's a ccflow BaseModel subclass
    from ccflow.base import BaseModel

    if isinstance(model, type) and issubclass(model, BaseModel):
        _register(model)

    return model
