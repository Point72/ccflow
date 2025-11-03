from typing import TYPE_CHECKING, Any, Dict, Optional, Type, Union

from .base import BaseModel
from .exttypes.pyobjectpath import _TYPE_ADAPTER as PyObjectPathTA

if TYPE_CHECKING:
    from ccflow.base import BaseModel

__all__ = (
    "model_alias",
    "from_python",
    "update_from_template",
)


def model_alias(model_name: str) -> "BaseModel":
    """Return a model by alias from the registry.

    Hydra-friendly: `_target_: ccflow.compose.model_alias` with `model_name`.

    Args:
        model_name: Alias string registered in the model registry. Typically a
            short name that maps to a configured BaseModel.

    Returns:
        A ``BaseModel`` instance resolved from the registry by ``model_name``.
    """
    return BaseModel.model_validate(model_name)


def from_python(py_object_path: str, indexer: Optional[list] = None) -> Any:
    """Hydra-friendly: resolve and return any Python object by import path.

    Optionally accepts ``indexer``, a list of keys that will be applied in
    order to index into the resolved object. No safety checks are performed;
    indexing errors will propagate.

    Args:
        py_object_path: Dotted import path to a Python object, e.g.
            ``mypkg.module.OBJECT`` or ``mypkg.module.ClassName``.
        indexer: Optional list of keys to apply in order to index into the
            resolved object (e.g., strings for dict keys or integers for list
            indexes).

    Returns:
        The resolved Python object, or the value obtained after applying all
        ``indexer`` keys to the resolved object.

    Example YAML usage:
      some_value:
        _target_: ccflow.compose.from_python
        py_object_path: mypkg.module.OBJECT

      nested_value:
        _target_: ccflow.compose.from_python
        py_object_path: mypkg.module.NESTED
        indexer: ["a", "b"]
    """
    obj = PyObjectPathTA.validate_python(py_object_path).object
    if indexer:
        for key in indexer:
            obj = obj[key]
    return obj


def update_from_template(
    base: Optional[Union[str, Dict[str, Any], "BaseModel"]] = None,
    *,
    target_class: Optional[Union[str, Type]] = None,
    update: Optional[Dict[str, Any]] = None,
) -> Any:
    """Generic update helper that constructs an instance from a base and updates.

    Args:
        base: Either a registry alias string, a dict, or a Pydantic BaseModel. If BaseModel, it is converted
              to a shallow dict via ``dict(base)`` to preserve nested object identity.
        target_class: Optional path to the target class to construct. May be a
              string import path or the type itself. If None and ``base`` is a
              BaseModel, returns an instance of ``base.__class__``. If None and
              ``base`` is a dict, returns the updated dict.
        update: Optional dict of updates to apply.

    Returns:
        Instance of ``target_class`` if provided; otherwise an instance of the same
        class as ``base`` when base is a BaseModel; or the updated dict when base
        is a dict.
    """
    # Determine base dict and default target
    default_target = None
    if isinstance(base, str):
        # Allow passing alias name directly; resolve from registry
        base = model_alias(base)
    if isinstance(base, BaseModel):
        base_dict = dict(base)
        default_target = base.__class__
    elif isinstance(base, dict):
        base_dict = dict(base)
    elif base is None:
        base_dict = {}
    else:
        raise TypeError("base must be a dict, BaseModel, or None")

    # Merge updates: explicit dict first, then kwargs
    if update:
        base_dict.update(update)

    # Resolve target class if provided as string path
    target = None
    if target_class is not None:
        if isinstance(target_class, str):
            target = PyObjectPathTA.validate_python(target_class).object
        else:
            target = target_class
    else:
        target = default_target

    if target is None:
        # No target: return dict update for dict base
        return base_dict

    # Construct instance of target with updated fields
    return target(**base_dict)
