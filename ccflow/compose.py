from typing import Any, Dict, List, Optional, Union

from .base import BaseModel
from .exttypes.pyobjectpath import PyObjectPath
from .utils.path_resolve_spec import PathResolveSpec

__all__ = ("model_alias", "model_alias_update", "model_compose")


def model_alias(model_name: str) -> BaseModel:
    """Return a model by alias from the registry.

    Hydra-friendly: `_target_: ccflow.compose.model_alias` with `model_name`.
    """
    return BaseModel.model_validate(model_name)


def model_alias_update(model_name: str, /, **updates) -> BaseModel:
    """Alias helper that returns a copy with updates applied.

    Preserves `model_alias` signature by resolving, then applying `model_copy(update=...)`.
    """
    model = model_alias(model_name)
    return model if not updates else model.model_copy(update=updates)


def model_compose(
    model: Union[str, BaseModel],
    *,
    # Python object source (optional)
    path: Optional[Union[str, PyObjectPath]] = None,
    keys: Optional[Union[str, int, List[Union[str, int]]]] = None,
    merge: str = "resolved_wins",
    filter_extras: bool = True,
    allowed_prefixes: Optional[List[str]] = None,
    # Explicit updates (optional)
    update: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> BaseModel:
    """Compose a model from a registry alias or instance, plus updates and optional python-backed defaults.

    Hydra-friendly usage examples:
      - _target_: ccflow.compose.model_compose
        model: foo
        param: 1

      - _target_: ccflow.compose.model_compose
        model: foo
        path: mypkg.defaults.CONFIG
        keys: "sub.section"
        merge: resolved_wins
        param: 2

    Behavior:
      1) Resolve `model` (string alias or BaseModel instance)
      2) Optionally load mapping from `path`/`keys` (python object) and merge
         with explicit updates based on `merge` semantics using PathResolveSpec
      3) Apply final updates via `model.model_copy(update=...)`

    Args:
        model: String alias into the registry or a BaseModel instance.
        path: Python import path (or `PyObjectPath`) to a mapping-like object.
        keys: Nested traversal into the resolved object (supports dotted string, ints for lists).
        merge: One of {"resolved_wins","explicit_wins","raise_on_conflict"} controlling conflict resolution
               between python-resolved mapping and explicit updates.
        filter_extras: If True, drop keys from the python-resolved mapping that are not fields of the model.
        allowed_prefixes: Optional list of allowed import path prefixes for safety.
        update: Explicit update mapping (merged with `**kwargs`).
        **kwargs: Additional explicit updates.
    """

    base_model = BaseModel.model_validate(model)

    # Collect explicit updates (kwargs take precedence over `update` mapping keys if present)
    explicit_values: Dict[str, Any] = {}
    if update:
        explicit_values.update(update)
    if kwargs:
        explicit_values.update(kwargs)

    # Optionally resolve mapping from python object and merge into explicit values
    if path is not None:
        spec = PathResolveSpec.model_validate(
            {
                "path": path,
                "keys": keys,
                "merge": merge,
                "filter_extras": filter_extras,
            }
        )
        obj = spec.resolve_object(allowed_prefixes=allowed_prefixes)
        node = spec.traverse(obj)
        mapping = spec.as_mapping(node)
        # Only drop extras from the resolved mapping; explicit updates should be validated by Pydantic normally
        mapping = spec.filter_extras_map(mapping, getattr(base_model.__class__, "model_fields", {}))
        explicit_values = spec.merge_into(explicit_values, mapping)

    # Apply final updates using pydantic's copy-with-update
    return base_model if not explicit_values else base_model.model_copy(update=explicit_values)
