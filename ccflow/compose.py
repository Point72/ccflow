from typing import Any, Dict, Optional

from .base import BaseModel
from .exttypes.pyobjectpath import _TYPE_ADAPTER as PyObjectPathTA

__all__ = (
    "model_alias",
    "model_copy_update",
    "from_python",
)


def model_alias(model_name: str) -> BaseModel:
    """Return a model by alias from the registry.

    Hydra-friendly: `_target_: ccflow.compose.model_alias` with `model_name`.
    """
    return BaseModel.model_validate(model_name)


def from_python(py_object_path: str) -> Any:
    """Hydra-friendly: resolve and return any Python object by import path.

    Example YAML usage:
      some_value:
        _target_: ccflow.compose.from_python
        py_object_path: mypkg.module.OBJECT
    """
    return PyObjectPathTA.validate_python(py_object_path).object


def model_copy_update(model_name: str, update: Optional[Dict[str, Any]] = None) -> BaseModel:
    """Return a model by alias from the registry, applying updates.

    Uses a shallow dict copy to preserve nested object identity. This will create a new object, and not update the original.
    """
    model = model_alias(model_name)
    # Shallow dict to avoid recursively converting nested models; preserves identity
    model_dict = dict(model)
    if update:
        model_dict.update(update)
    return model.__class__(**model_dict)
