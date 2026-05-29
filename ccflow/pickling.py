"""Pickle helpers for ccflow model classes.

This module works around Pydantic generic specializations sometimes pickling by
process-local generated class names, such as ``GenericResult[int]``. Fresh
processes may not have materialized those names yet, so unpickling can fail
before ccflow code gets control. See the upstream Pydantic issue for the
underlying behavior:

https://github.com/pydantic/pydantic/issues/9390

If Pydantic or cloudpickle eventually make generated generic model
specializations portable across fresh processes without relying on module-global
side effects, this module can likely be deleted and ``BaseModel.__reduce_ex__``
can fall back to Pydantic's default reducer.
"""

import types
from functools import singledispatch
from typing import Any, NamedTuple, Optional, Tuple, Type, get_args, get_origin

from pydantic import BaseModel as PydanticBaseModel

__all__ = ("reduce_generic_model_instance",)


class _GenericTypeSpec(NamedTuple):
    # Pickle must not rely on process-local names like ``GenericResult[int]``.
    # Store generic args as data so the receiver can rebuild ``origin[args]``.
    origin: Any
    args: Tuple[Any, ...]


def _is_pydantic_generic_specialization(value: Any) -> bool:
    if not isinstance(value, type):
        return False
    metadata = getattr(value, "__pydantic_generic_metadata__", None)
    return bool(metadata and metadata.get("origin") is not None)


@singledispatch
def _portable_generic_type_arg_with_changed(value: Any) -> tuple[Any, bool]:
    if _is_pydantic_generic_specialization(value):
        metadata = value.__pydantic_generic_metadata__
        portable_args, _ = _portable_generic_type_args(metadata["args"])
        return _GenericTypeSpec(metadata["origin"], portable_args), True

    origin = get_origin(value)
    args = get_args(value)
    if origin is not None and args:
        portable_args, changed = _portable_generic_type_args(args)
        if changed:
            return _GenericTypeSpec(origin, portable_args), True
    return value, False


@_portable_generic_type_arg_with_changed.register(list)
def _(value: list) -> tuple[Any, bool]:
    portable_items = []
    changed = False
    for item in value:
        portable_item, item_changed = _portable_generic_type_arg_with_changed(item)
        portable_items.append(portable_item)
        changed = changed or item_changed
    return (portable_items, True) if changed else (value, False)


@_portable_generic_type_arg_with_changed.register(tuple)
def _(value: tuple) -> tuple[Any, bool]:
    portable_items, changed = _portable_generic_type_args(value)
    return (portable_items, True) if changed else (value, False)


def _portable_generic_type_args(args: tuple[Any, ...]) -> tuple[tuple[Any, ...], bool]:
    portable_args = []
    changed = False
    for arg in args:
        portable_arg, arg_changed = _portable_generic_type_arg_with_changed(arg)
        portable_args.append(portable_arg)
        changed = changed or arg_changed
    return tuple(portable_args), changed


def _portable_generic_type_arg(value: Any) -> Any:
    return _portable_generic_type_arg_with_changed(value)[0]


@singledispatch
def _restore_generic_type_arg(value: Any) -> Any:
    return value


@_restore_generic_type_arg.register(_GenericTypeSpec)
def _(value: _GenericTypeSpec) -> Any:
    origin, args = value
    restored_args = tuple(_restore_generic_type_arg(arg) for arg in args)
    try:
        return origin[restored_args]
    except TypeError as exc:
        if len(restored_args) == 1:
            try:
                return origin[restored_args[0]]
            except TypeError:
                pass
        # ``types.UnionType`` is not itself subscriptable; rebuild PEP 604
        # unions from their members if one of those members needed restoration.
        union_type = getattr(types, "UnionType", None)
        if union_type is not None and origin is union_type:
            result = restored_args[0]
            for arg in restored_args[1:]:
                result = result | arg
            return result
        raise exc


@_restore_generic_type_arg.register(list)
def _(value: list) -> list:
    return [_restore_generic_type_arg(item) for item in value]


@_restore_generic_type_arg.register(tuple)
def _(value: tuple) -> tuple:
    return tuple(_restore_generic_type_arg(item) for item in value)


def _new_ccflow_generic_model(origin: Type[PydanticBaseModel], args: Tuple[Any, ...]) -> PydanticBaseModel:
    """Restore a Pydantic generic specialization without a process-local global."""

    cls = origin[tuple(_restore_generic_type_arg(arg) for arg in args)]
    return cls.__new__(cls)


def reduce_generic_model_instance(model: PydanticBaseModel) -> Optional[tuple[Any, tuple[Any, ...], dict[str, Any]]]:
    """Return a portable reducer for Pydantic generic specializations."""

    if not _is_pydantic_generic_specialization(type(model)):
        return None

    metadata = type(model).__pydantic_generic_metadata__
    args, _ = _portable_generic_type_args(metadata["args"])
    return (_new_ccflow_generic_model, (metadata["origin"], args), model.__getstate__())
