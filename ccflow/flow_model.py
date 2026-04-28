"""Flow.model decorator implementation built around ``FromContext``."""

import inspect
import logging
from base64 import b64decode, b64encode
from collections import OrderedDict
from functools import lru_cache, wraps
from typing import (
    Annotated,
    Any,
    Callable,
    ClassVar,
    Dict,
    List,
    Literal,
    Mapping,
    NamedTuple,
    Optional,
    Set,
    Tuple,
    Type,
    cast,
    get_args,
    get_origin,
    get_type_hints,
)

from pydantic import BaseModel as PydanticModel, Field, TypeAdapter, ValidationError, model_validator
from pydantic.errors import PydanticSchemaGenerationError, PydanticUndefinedAnnotation

from ._flow_model_binding import (
    _REMOVED_CONTEXT_ARGS,
    _UNION_ORIGINS,
    _UNSET,
    FromContext,
    Lazy,
    _analyze_flow_context_transform,
    _analyze_flow_model,
    _callable_name,
    _FlowModelConfig,
    _FlowModelParam,
    _resolved_flow_signature,
    _strip_annotated,
)
from .base import BaseModel, ContextBase, ResultBase
from .callable import CallableModel, Flow, FlowOptions, GraphDepList, WrapperModel
from .context import FlowContext
from .exttypes import PyObjectPath
from .local_persistence import register_ccflow_import_path
from .result import GenericResult

__all__ = (
    "FlowAPI",
    "BoundModel",
    "FromContext",
    "Lazy",
    "ContextTransform",
    "clear_flow_model_caches",
    "flow_context_transform",
)

_AnyCallable = Callable[..., Any]
_OptionalContextBase = ContextBase | None
log = logging.getLogger(__name__)


class _UnsetFlowInput:
    def __repr__(self) -> str:
        return "<unset>"

    def __reduce__(self):
        return (_unset_flow_input_factory, ())


_UNSET_FLOW_INPUT = _UnsetFlowInput()
_TYPE_ADAPTER_CACHE_MAXSIZE = 256
_HASHABLE_TYPE_ADAPTER_CACHE: "OrderedDict[Any, TypeAdapter]" = OrderedDict()
_UNHASHABLE_TYPE_ADAPTER_CACHE: "OrderedDict[int, Tuple[Any, TypeAdapter]]" = OrderedDict()


def _unset_flow_input_factory() -> _UnsetFlowInput:
    return _UNSET_FLOW_INPUT


def _is_unset_flow_input(value: Any) -> bool:
    return value is _UNSET_FLOW_INPUT


_ModelContextContract = NamedTuple(
    "_ModelContextContract",
    [
        ("runtime_context_type", Type[ContextBase]),
        ("input_types", Optional[Dict[str, Any]]),
        ("required_names", Tuple[str, ...]),
        ("generated_model", Optional["_GeneratedFlowModelBase"]),
    ],
)


class ContextTransform(PydanticModel):
    kind: Literal["context_transform"] = "context_transform"
    path: Optional[PyObjectPath] = None
    serialized_config: Optional[str] = None
    bound_args: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_location(self):
        if (self.path is None) == (self.serialized_config is None):
            raise ValueError("ContextTransform must define exactly one of path or serialized_config.")
        return self


class StaticValueSpec(PydanticModel):
    kind: Literal["static_value"] = "static_value"
    value: Any


class FieldContextSpec(PydanticModel):
    kind: Literal["context_value"] = "context_value"
    binding: ContextTransform


class PatchContextSpec(PydanticModel):
    kind: Literal["context_patch"] = "context_patch"
    binding: ContextTransform


_FieldOverrideSpec = StaticValueSpec | FieldContextSpec


class _BoundContextSpec(PydanticModel):
    patches: List[PatchContextSpec] = Field(default_factory=list)
    field_overrides: Dict[str, _FieldOverrideSpec] = Field(default_factory=dict)


def _context_values(context: ContextBase) -> Dict[str, Any]:
    return dict(context)


def _context_transform_repr(transform: Any) -> str:
    if isinstance(transform, ContextTransform):
        name = _callable_name(_load_context_transform_config_from_binding(transform).func)
        if not transform.bound_args:
            return f"{name}()"
        args = ", ".join(f"{key}={value!r}" for key, value in sorted(transform.bound_args.items()))
        return f"{name}({args})"
    return repr(transform)


def _context_transform_identifier(binding: ContextTransform) -> str:
    if binding.path is not None:
        return str(binding.path)
    return _callable_name(_load_context_transform_config_from_binding(binding).func)


def _is_model_dependency(value: Any) -> bool:
    return isinstance(value, CallableModel)


def _bound_field_names(model: Any) -> set[str]:
    fields_set = getattr(model, "model_fields_set", None)
    if fields_set is not None:
        return set(fields_set)
    return set()


def _concrete_context_type(context_type: Any) -> Optional[Type[ContextBase]]:
    if isinstance(context_type, type) and issubclass(context_type, ContextBase):
        return context_type

    if get_origin(context_type) in _UNION_ORIGINS:
        for arg in get_args(context_type):
            if arg is type(None):
                continue
            if isinstance(arg, type) and issubclass(arg, ContextBase):
                return arg

    return None


def _remember_type_adapter(cache: "OrderedDict[Any, Any]", key: Any, value: Any) -> Any:
    cache[key] = value
    cache.move_to_end(key)
    if len(cache) > _TYPE_ADAPTER_CACHE_MAXSIZE:
        cache.popitem(last=False)
    return value


def _type_adapter(annotation: Any) -> TypeAdapter:
    try:
        adapter = _HASHABLE_TYPE_ADAPTER_CACHE.pop(annotation)
    except TypeError:
        key = id(annotation)
        cached = _UNHASHABLE_TYPE_ADAPTER_CACHE.pop(key, None)
        if cached is not None and cached[0] is annotation:
            _UNHASHABLE_TYPE_ADAPTER_CACHE[key] = cached
            return cached[1]
        adapter = TypeAdapter(annotation)
        return _remember_type_adapter(_UNHASHABLE_TYPE_ADAPTER_CACHE, key, (annotation, adapter))[1]
    except KeyError:
        adapter = TypeAdapter(annotation)
        return _remember_type_adapter(_HASHABLE_TYPE_ADAPTER_CACHE, annotation, adapter)
    _HASHABLE_TYPE_ADAPTER_CACHE[annotation] = adapter
    return adapter


def _can_validate_type(annotation: Any) -> bool:
    try:
        _type_adapter(annotation)
    except (PydanticSchemaGenerationError, PydanticUndefinedAnnotation, TypeError, ValueError):
        return False
    return True


def _expected_type_repr(annotation: Any) -> str:
    try:
        return annotation.__name__
    except AttributeError:
        return repr(annotation)


def _coerce_value(name: str, value: Any, annotation: Any, source: str) -> Any:
    if not _can_validate_type(annotation):
        return value
    try:
        return _type_adapter(annotation).validate_python(value)
    except (ValidationError, ValueError, TypeError) as exc:
        expected = _expected_type_repr(annotation)
        raise TypeError(f"{source} '{name}': expected {expected}, got {type(value).__name__} ({value!r})") from exc


def _unwrap_model_result(value: Any) -> Any:
    if isinstance(value, GenericResult):
        return value.value
    return value


def _make_lazy_thunk(value: CallableModel, context: ContextBase) -> Callable[[], Any]:
    cache: Dict[str, Any] = {}

    def thunk():
        if "result" not in cache:
            dependency_model, dependency_context = _resolved_dependency_invocation(value, context)
            cache["result"] = _unwrap_model_result(dependency_model(dependency_context))
        return cache["result"]

    return thunk


def _make_coercing_lazy_thunk(inner_thunk: Callable[[], Any], name: str, annotation: Any) -> Callable[[], Any]:
    cache: Dict[str, Any] = {}

    def thunk():
        if "result" not in cache:
            cache["result"] = _coerce_value(name, inner_thunk(), annotation, "Regular parameter")
        return cache["result"]

    return thunk


def _maybe_auto_unwrap_external_result(target: CallableModel, result: Any) -> Any:
    generated = _generated_model_instance(target)
    if generated is None:
        return result

    config = type(generated).__flow_model_config__
    if config.auto_wrap_result and config.auto_unwrap:
        return _unwrap_model_result(result)
    return result


def _type_accepts_str(annotation: Any) -> bool:
    if annotation is Any or annotation is inspect.Parameter.empty:
        return True
    if annotation is str:
        return True
    origin = get_origin(annotation)
    if origin is Annotated:
        return _type_accepts_str(get_args(annotation)[0])
    if origin in _UNION_ORIGINS:
        return any(_type_accepts_str(arg) for arg in get_args(annotation) if arg is not type(None))
    return False


def _resolve_registry_candidate(value: str) -> Any:
    try:
        candidate = BaseModel.model_validate(value)
    except ValidationError:
        return None
    return candidate if isinstance(candidate, BaseModel) else None


def _registry_candidate_allowed(expected_type: Any, candidate: Any) -> bool:
    if _is_model_dependency(candidate):
        return True
    if not _can_validate_type(expected_type):
        return True
    try:
        _type_adapter(expected_type).validate_python(candidate)
    except ValidationError:
        return False
    return True


def _ensure_top_level_named_function(fn: _AnyCallable, *, decorator_name: str) -> None:
    if not inspect.isfunction(fn):
        raise TypeError(f"{decorator_name} only supports Python functions.")

    name = getattr(fn, "__name__", "")
    if name == "<lambda>":
        raise TypeError(f"{decorator_name} only supports named Python functions.")


@lru_cache(maxsize=None)
def _load_context_transform_factory(path: str) -> _AnyCallable:
    return PyObjectPath(path).object


@lru_cache(maxsize=None)
def _load_context_transform_config(path: str) -> _FlowModelConfig:
    factory = _load_context_transform_factory(path)
    config = getattr(factory, "__flow_context_transform_config__", None)
    if not isinstance(config, _FlowModelConfig):
        raise TypeError(f"Stored context transform path '{path}' does not resolve to a Flow.context_transform binding.")
    return config


def _serialize_context_transform_config(config: _FlowModelConfig) -> str:
    import cloudpickle

    payload = cloudpickle.dumps(config, protocol=5)
    return b64encode(payload).decode("ascii")


@lru_cache(maxsize=None)
def _load_serialized_context_transform_config(serialized_config: str) -> _FlowModelConfig:
    import cloudpickle

    config = cloudpickle.loads(b64decode(serialized_config.encode("ascii")))
    if not isinstance(config, _FlowModelConfig):
        raise TypeError("Stored context transform payload does not contain a Flow.context_transform binding.")
    return config


def _load_context_transform_config_from_binding(binding: ContextTransform) -> _FlowModelConfig:
    if binding.path is not None:
        return _load_context_transform_config(str(binding.path))
    if binding.serialized_config is None:
        raise TypeError("ContextTransform has neither path nor serialized_config.")
    return _load_serialized_context_transform_config(binding.serialized_config)


def clear_flow_model_caches() -> None:
    """Clear module-level caches used by Flow.model internals."""

    _HASHABLE_TYPE_ADAPTER_CACHE.clear()
    _UNHASHABLE_TYPE_ADAPTER_CACHE.clear()
    _load_context_transform_factory.cache_clear()
    _load_context_transform_config.cache_clear()
    _load_serialized_context_transform_config.cache_clear()


def _is_mapping_annotation(annotation: Any) -> bool:
    if annotation is inspect.Signature.empty:
        return False
    annotation = _strip_annotated(annotation)
    origin = get_origin(annotation)
    if origin in _UNION_ORIGINS:
        variants = [arg for arg in get_args(annotation) if arg is not type(None)]
        return bool(variants) and all(_is_mapping_annotation(arg) for arg in variants)
    origin = origin or annotation
    try:
        return issubclass(origin, Mapping)
    except TypeError:
        return False


def _restore_pickled_flow_model(type_path: str, state: Dict[str, Any]) -> BaseModel:
    cls = cast(type[BaseModel], PyObjectPath(type_path).object)
    instance = cls.__new__(cls)
    instance.__setstate__(state)
    return instance


def _restore_pickled_local_flow_model(serialized_factory_payload: bytes, state: Dict[str, Any]) -> BaseModel:
    import cloudpickle

    fn, factory_kwargs = cloudpickle.loads(serialized_factory_payload)
    factory = flow_model(fn, **factory_kwargs)
    cls = cast(type[BaseModel], getattr(factory, "_generated_model"))
    instance = cls.__new__(cls)
    instance.__setstate__(state)
    return instance


def _restore_generated_flow_model(factory_path: str, state: Dict[str, Any]) -> BaseModel:
    """Restore a generated flow model by importing its factory function.

    This is the cross-process-safe restore path: importing the factory's module
    triggers the ``@Flow.model`` decorator, which re-creates the GeneratedModel
    class.  We then reconstruct the instance from the pickled state.
    """
    factory = PyObjectPath(factory_path).object
    generated_cls = getattr(factory, "_generated_model", None)
    if generated_cls is None:
        raise ImportError(f"Cannot restore generated flow model: '{factory_path}' does not have a _generated_model attribute.")
    instance = generated_cls.__new__(generated_cls)
    instance.__setstate__(state)
    return instance


def _is_importable_function(func: _AnyCallable) -> bool:
    """Return True if *func* is a top-level, importable named function."""
    module = getattr(func, "__module__", None)
    name = getattr(func, "__name__", None)
    qualname = getattr(func, "__qualname__", None)
    return bool(module and module != "__main__" and name and qualname and qualname == name and "<locals>" not in qualname)


def _importable_function_path(func: _AnyCallable) -> Optional[str]:
    if not _is_importable_function(func):
        return None
    return f"{func.__module__}.{func.__name__}"


def _generated_model_factory_path_for_pickle(config: _FlowModelConfig, generated_cls: type) -> Optional[str]:
    path = _importable_function_path(config.func)
    if path is None:
        return None
    try:
        factory = PyObjectPath(path).object
    except ImportError:
        return None
    if getattr(factory, "_generated_model", None) is generated_cls:
        return path
    return None


def _context_transform_should_use_import_path(config: _FlowModelConfig) -> bool:
    path = config.path
    if path is None or not _is_importable_function(config.func):
        return False
    try:
        resolved = PyObjectPath(str(path)).object
    except ImportError:
        return True
    return isinstance(getattr(resolved, "__flow_context_transform_config__", None), _FlowModelConfig)


def _runtime_context_for_model(model: CallableModel, values: Dict[str, Any]) -> ContextBase:
    contract = _model_context_contract(model)
    if contract.runtime_context_type is FlowContext:
        return FlowContext(**values)
    return contract.runtime_context_type.model_validate(values)


def _project_context_values_for_model(model: CallableModel, values: Dict[str, Any]) -> Dict[str, Any]:
    contract = _model_context_contract(model)
    if contract.runtime_context_type is FlowContext or contract.input_types is None:
        return values
    return {name: values[name] for name in contract.input_types if name in values}


def _dependency_context_values(model: CallableModel, context: ContextBase) -> Dict[str, Any]:
    return _project_context_values_for_model(model, _context_values(context))


def _dependency_context_for_model(model: CallableModel, context: ContextBase) -> ContextBase:
    return _runtime_context_for_model(model, _dependency_context_values(model, context))


def _resolved_dependency_invocation(value: CallableModel, context: ContextBase) -> Tuple[CallableModel, ContextBase]:
    if isinstance(value, BoundModel):
        return value, FlowContext(**_context_values(context))
    return value, _dependency_context_for_model(value, context)


def _merge_context_specs(
    existing: _BoundContextSpec, patches: List[PatchContextSpec], field_overrides: Dict[str, _FieldOverrideSpec]
) -> _BoundContextSpec:
    return _BoundContextSpec(
        patches=[*existing.patches, *patches],
        field_overrides={**existing.field_overrides, **field_overrides},
    )


def _generated_model_instance(stage: Any) -> Optional["_GeneratedFlowModelBase"]:
    model = stage.model if isinstance(stage, BoundModel) else stage
    if isinstance(model, _GeneratedFlowModelBase):
        return model
    return None


def _model_context_contract(
    model: CallableModel,
) -> _ModelContextContract:
    generated = _generated_model_instance(model)
    if generated is not None:
        config = type(generated).__flow_model_config__
        return _ModelContextContract(FlowContext, dict(config.context_input_types), config.context_required_names, generated)

    context_cls = _concrete_context_type(model.context_type)
    if context_cls is None:
        return _ModelContextContract(FlowContext, None, (), None)
    if context_cls is FlowContext or not hasattr(context_cls, "model_fields"):
        return _ModelContextContract(context_cls, None, (), None)
    return _ModelContextContract(
        context_cls,
        {name: info.annotation for name, info in context_cls.model_fields.items()},
        tuple(name for name, info in context_cls.model_fields.items() if info.is_required()),
        None,
    )


def _model_base_field_names(generated: "_GeneratedFlowModelBase") -> set[str]:
    """Return field names from model_base that aren't function parameters or internal fields."""
    config = type(generated).__flow_model_config__
    param_names = {param.name for param in config.parameters}
    return {name for name in type(generated).model_fields if name not in param_names and name != "meta"}


def _missing_regular_param_names(model: "_GeneratedFlowModelBase", config: _FlowModelConfig) -> List[str]:
    missing = []
    for param in config.regular_params:
        value = getattr(model, param.name, _UNSET_FLOW_INPUT)
        if _is_unset_flow_input(value):
            missing.append(param.name)
    return missing


def _resolve_regular_param_value(model: "_GeneratedFlowModelBase", param: _FlowModelParam, context: ContextBase) -> Any:
    value = getattr(model, param.name, _UNSET_FLOW_INPUT)
    if _is_unset_flow_input(value):
        raise TypeError(
            f"Regular parameter '{param.name}' for {_callable_name(type(model).__flow_model_config__.func)} is still unbound. "
            "Bind it at construction time."
        )
    if _is_model_dependency(value):
        if param.is_lazy:
            return _make_lazy_thunk(value, context)
        dependency_model, dependency_context = _resolved_dependency_invocation(value, context)
        return _unwrap_model_result(dependency_model(dependency_context))
    if param.is_lazy:
        raise TypeError(f"Parameter '{param.name}' is marked Lazy[...] and must be bound to a CallableModel dependency.")
    return value


def _collect_contextual_values(
    model: "_GeneratedFlowModelBase",
    config: _FlowModelConfig,
    explicit_values: Dict[str, Any],
) -> Tuple[Dict[str, Any], List[str]]:
    resolved: Dict[str, Any] = {}
    missing: List[str] = []

    for param in config.contextual_params:
        if param.name in explicit_values:
            resolved[param.name] = explicit_values[param.name]
            continue

        value = getattr(model, param.name, _UNSET_FLOW_INPUT)
        if not _is_unset_flow_input(value):
            resolved[param.name] = value
            continue

        if param.has_function_default:
            resolved[param.name] = param.function_default
            continue

        missing.append(param.name)

    return resolved, missing


def _resolved_contextual_inputs(model: "_GeneratedFlowModelBase", config: _FlowModelConfig, context: ContextBase) -> Dict[str, Any]:
    resolved, missing_contextual = _collect_contextual_values(model, config, _context_values(context))

    if missing_contextual:
        missing = ", ".join(sorted(missing_contextual))
        raise TypeError(
            f"Missing contextual input(s) for {_callable_name(config.func)}: {missing}. "
            "Supply them via the runtime context, compute(), with_context(), or construction-time contextual defaults."
        )

    if config.declared_context_type is not None:
        return _validate_declared_context_values(config, resolved)

    return {
        param.name: _coerce_value(param.name, resolved[param.name], param.validation_annotation, "Context field")
        for param in config.contextual_params
    }


def _validate_declared_context_values(config: _FlowModelConfig, values: Dict[str, Any]) -> Dict[str, Any]:
    if config.declared_context_type is None:
        return values

    validated = config.declared_context_type.model_validate(values)
    return {param.name: getattr(validated, param.name) for param in config.contextual_params}


def _validate_declared_context_field(config: _FlowModelConfig, name: str, value: Any) -> Any:
    if config.declared_context_type is None:
        return _UNSET

    try:
        validated = config.declared_context_type.model_validate({name: value})
    except ValidationError as exc:
        field_errors = [error for error in exc.errors() if error.get("loc") and error["loc"][0] == name]
        if field_errors:
            raise
        return _UNSET
    return getattr(validated, name)


def _coerce_contextual_value(config: _FlowModelConfig, param: _FlowModelParam, value: Any, source: str) -> Any:
    declared_value = _validate_declared_context_field(config, param.name, value)
    if declared_value is not _UNSET:
        return declared_value
    return _coerce_value(param.name, value, param.validation_annotation, source)


def _coerce_model_context_value(model: CallableModel, field_name: str, value: Any, source: str) -> Any:
    generated = _generated_model_instance(model)
    if generated is not None:
        config = type(generated).__flow_model_config__
        if field_name in config.contextual_param_names:
            return _coerce_contextual_value(config, config.param(field_name), value, source)

    contract = _model_context_contract(model)
    if contract.input_types is None or field_name not in contract.input_types:
        return value
    return _coerce_value(field_name, value, contract.input_types[field_name], source)


def _identity_context_values_for_model(model: CallableModel, context: ContextBase) -> Dict[str, Any]:
    return _identity_context_values_for_model_values(model, _context_values(context))


def _identity_context_values_for_model_values(model: CallableModel, values: Dict[str, Any]) -> Dict[str, Any]:
    contract = _model_context_contract(model)
    if contract.input_types is None:
        return values
    return {name: values[name] for name in contract.input_types if name in values}


def _identity_context_values_and_missing_for_model(model: CallableModel, values: Dict[str, Any]) -> Tuple[Dict[str, Any], Tuple[str, ...]]:
    generated = _generated_model_instance(model)
    if generated is not None:
        config = type(generated).__flow_model_config__
        resolved, missing = _collect_contextual_values(generated, config, values)
        return (
            {param.name: resolved[param.name] for param in config.contextual_params if param.name in resolved},
            tuple(missing),
        )

    context_values = _identity_context_values_for_model_values(model, values)
    missing = tuple(name for name in _model_context_contract(model).required_names if name not in context_values)
    return context_values, missing


def _context_transform_missing_context_names(binding: ContextTransform, values: Dict[str, Any]) -> Tuple[str, ...]:
    config = _load_context_transform_config_from_binding(binding)
    return tuple(param.name for param in config.contextual_params if param.name not in values and not param.has_function_default)


def _evaluate_context_transform_from_values(binding: ContextTransform, values: Dict[str, Any]) -> Any:
    config = _load_context_transform_config_from_binding(binding)
    kwargs = _bound_context_transform_regular_kwargs(config, binding)

    for param in config.contextual_params:
        if param.name in values:
            kwargs[param.name] = _coerce_value(param.name, values[param.name], param.annotation, "Context transform field")
        elif param.has_function_default:
            kwargs[param.name] = param.function_default
        else:
            raise TypeError(
                f"Missing contextual input(s) for context transform {_callable_name(config.func)}: {param.name}. "
                "Supply them via the runtime context or with_context() ordering."
            )

    return config.func(**kwargs)


def _apply_context_spec_values_for_identity(
    model: CallableModel, context_spec: "_BoundContextSpec", context: ContextBase
) -> Tuple[Dict[str, Any], Tuple[Tuple[str, Tuple[str, ...]], ...]]:
    current_values = _context_values(context)
    missing_transforms: List[Tuple[str, Tuple[str, ...]]] = []

    for patch in context_spec.patches:
        missing = _context_transform_missing_context_names(patch.binding, _context_values(context))
        if missing:
            missing_transforms.append((_context_transform_identifier(patch.binding), missing))
            continue
        result = _evaluate_context_transform_from_values(patch.binding, _context_values(context))
        current_values.update(_validate_patch_result(model, result))

    for name, spec in context_spec.field_overrides.items():
        if isinstance(spec, StaticValueSpec):
            current_values[name] = spec.value
            continue

        missing = _context_transform_missing_context_names(spec.binding, _context_values(context))
        if missing:
            missing_transforms.append((name, missing))
            current_values.pop(name, None)
            continue
        result = _evaluate_context_transform_from_values(spec.binding, _context_values(context))
        current_values[name] = _coerce_model_context_value(model, name, result, "with_context()")

    return current_values, tuple(missing_transforms)


def _unresolved_lazy_dependency_descriptor(
    value: CallableModel,
    context_values: Dict[str, Any],
    missing_context: Tuple[str, ...],
    missing_transform_context: Tuple[Tuple[str, Tuple[str, ...]], ...] = (),
) -> Dict[str, Any]:
    return {
        "kind": "unresolved_lazy_dependency",
        "model_type": str(PyObjectPath.validate(type(value))),
        "model": value.model_dump(mode="python"),
        "context_type": str(PyObjectPath.validate(FlowContext)),
        "context": context_values,
        "missing_context": missing_context,
        "missing_transform_context": missing_transform_context,
    }


def _lazy_dependency_identity(
    value: CallableModel,
    context: ContextBase,
) -> Tuple[Optional[Dict[str, Any]], Optional[CallableModel], Optional[ContextBase]]:
    if isinstance(value, BoundModel):
        dependency_model = value.model
        values, missing_transform_context = _apply_context_spec_values_for_identity(dependency_model, value.context_spec, context)
        if missing_transform_context:
            context_values: Dict[str, Any] = {}
            missing_context = _model_context_contract(dependency_model).required_names
        else:
            context_values, missing_context = _identity_context_values_and_missing_for_model(dependency_model, values)
        if missing_context or missing_transform_context:
            return _unresolved_lazy_dependency_descriptor(value, context_values, missing_context, missing_transform_context), None, None
        return None, *_resolved_dependency_invocation(value, context)

    dependency_model = value
    context_values, missing_context = _identity_context_values_and_missing_for_model(dependency_model, _context_values(context))
    if missing_context:
        return _unresolved_lazy_dependency_descriptor(value, context_values, missing_context), None, None
    return None, *_resolved_dependency_invocation(value, context)


def _validate_bound_param_value(
    config: _FlowModelConfig,
    param: _FlowModelParam,
    value: Any,
    source: str,
) -> Any:
    if param.is_contextual:
        if _is_model_dependency(value):
            raise TypeError(
                f"Parameter '{param.name}' is marked FromContext[...] and cannot be bound to a CallableModel. "
                "Bind a literal contextual default or supply it via compute()/with_context()."
            )
        return _coerce_contextual_value(config, param, value, source)

    if param.is_lazy and not _is_model_dependency(value):
        raise TypeError(f"Parameter '{param.name}' is marked Lazy[...] and must be bound to a CallableModel dependency.")
    if _is_model_dependency(value):
        return value
    return _coerce_value(param.name, value, param.annotation, source)


def _generated_model_identity_payload(
    model: "_GeneratedFlowModelBase",
    context: ContextBase,
    child_evaluation_key: Callable[[CallableModel, ContextBase], bytes],
) -> Optional[Dict[str, Any]]:
    """Describe the generated model's effective invocation for cache keys.

    Contract:
    - contextual identity is projected to the ``FromContext[...]`` fields the
      generated model consumes;
    - unused ambient ``FlowContext`` fields are ignored;
    - regular literal inputs are included directly;
    - regular ``CallableModel`` inputs recurse through their own effective
      evaluation key; and
    - unresolved lazy dependency context is recorded explicitly instead of
      forcing eager dependency resolution.

    Returning ``None`` asks the evaluator to use the structural key.
    """

    config = type(model).__flow_model_config__
    regular_inputs = []
    for param in config.regular_params:
        value = getattr(model, param.name, _UNSET_FLOW_INPUT)
        if _is_unset_flow_input(value):
            return None

        descriptor = {"name": param.name, "lazy": param.is_lazy}
        if _is_model_dependency(value):
            if param.is_lazy:
                unresolved, dependency_model, dependency_context = _lazy_dependency_identity(value, context)
                if unresolved is not None:
                    descriptor.update(unresolved)
                else:
                    assert dependency_model is not None
                    assert dependency_context is not None
                    descriptor.update({"kind": "dependency", "key": child_evaluation_key(dependency_model, dependency_context)})
            else:
                dependency_model, dependency_context = _resolved_dependency_invocation(value, context)
                descriptor.update({"kind": "dependency", "key": child_evaluation_key(dependency_model, dependency_context)})
        else:
            descriptor.update({"kind": "literal", "value": value})
        regular_inputs.append(descriptor)

    model_base_fields = {name: getattr(model, name) for name in sorted(_model_base_field_names(model))}

    return {
        "kind": "generated_flow_model_v1",
        "model_type": str(PyObjectPath.validate(type(model))),
        "contextual_inputs": _resolved_contextual_inputs(model, config, context),
        "regular_inputs": regular_inputs,
        "model_base_fields": model_base_fields,
    }


def _resolved_static_contextual_values(
    model: "_GeneratedFlowModelBase",
    config: _FlowModelConfig,
    static_overrides: Optional[Dict[str, StaticValueSpec]] = None,
) -> Optional[Dict[str, Any]]:
    override_values = {name: spec.value for name, spec in (static_overrides or {}).items()}
    resolved, missing = _collect_contextual_values(model, config, override_values)
    return None if missing else resolved


def _validate_bound_declared_context_defaults(model: "_GeneratedFlowModelBase", config: _FlowModelConfig) -> None:
    resolved = _resolved_static_contextual_values(model, config)
    if resolved is None:
        return

    validated = _validate_declared_context_values(config, resolved)
    for param in config.contextual_params:
        value = getattr(model, param.name, _UNSET_FLOW_INPUT)
        if _is_unset_flow_input(value):
            continue
        object.__setattr__(model, param.name, validated[param.name])


def _bound_context_transform_regular_kwargs(config: _FlowModelConfig, binding: ContextTransform) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {}
    for param in config.regular_params:
        if param.name in binding.bound_args:
            kwargs[param.name] = binding.bound_args[param.name]
        elif param.has_function_default:
            kwargs[param.name] = param.function_default
        else:
            raise TypeError(f"Context transform '{_callable_name(config.func)}' is missing required regular parameter '{param.name}'.")
    return kwargs


def _evaluate_static_context_transform(binding: ContextTransform) -> Any:
    config = _load_context_transform_config_from_binding(binding)
    kwargs = _bound_context_transform_regular_kwargs(config, binding)

    for param in config.contextual_params:
        if param.has_function_default:
            kwargs[param.name] = param.function_default
            continue
        return _UNSET

    return config.func(**kwargs)


def _static_field_override_value(model: CallableModel, field_name: str, spec: _FieldOverrideSpec) -> Any:
    if isinstance(spec, StaticValueSpec):
        return spec.value

    value = _evaluate_static_context_transform(spec.binding)
    if value is _UNSET:
        return _UNSET

    contract = _model_context_contract(model)
    if contract.input_types is None or field_name not in contract.input_types:
        return value
    return _coerce_model_context_value(model, field_name, value, "with_context()")


def _statically_resolved_context_values(model: CallableModel, context_spec: _BoundContextSpec) -> Optional[Dict[str, Any]]:
    values: Dict[str, Any] = {}

    for patch in context_spec.patches:
        result = _evaluate_static_context_transform(patch.binding)
        if result is _UNSET:
            return None
        values.update(_validate_patch_result(model, result))

    for name, spec in context_spec.field_overrides.items():
        value = _static_field_override_value(model, name, spec)
        if value is _UNSET:
            return None
        values[name] = value

    return values


def _statically_resolved_context_field_names(model: CallableModel, context_spec: _BoundContextSpec) -> Set[str]:
    names: Set[str] = set()

    for patch in context_spec.patches:
        result = _evaluate_static_context_transform(patch.binding)
        if result is _UNSET:
            continue
        names.update(_validate_patch_result(model, result))

    for name, spec in context_spec.field_overrides.items():
        if _static_field_override_value(model, name, spec) is not _UNSET:
            names.add(name)

    return names


def _context_transform_input_types(binding: ContextTransform, *, required_only: bool) -> Dict[str, Any]:
    config = _load_context_transform_config_from_binding(binding)
    names = config.context_required_names if required_only else config.contextual_param_names
    return {name: config.context_input_types[name] for name in names}


def _validate_static_context_spec_declared_context(model: CallableModel, context_spec: _BoundContextSpec) -> _BoundContextSpec:
    generated = _generated_model_instance(model)
    if generated is None:
        return context_spec

    config = type(generated).__flow_model_config__
    if config.declared_context_type is None:
        return context_spec

    static_context_values = _statically_resolved_context_values(model, context_spec)
    if static_context_values is None:
        return context_spec

    static_overrides = {name: StaticValueSpec(value=value) for name, value in static_context_values.items()}
    resolved = _resolved_static_contextual_values(generated, config, static_overrides)
    if resolved is None:
        return context_spec

    _validate_declared_context_values(config, resolved)
    return context_spec


def _validate_with_context_field_names(model: CallableModel, names: List[str]) -> None:
    contract = _model_context_contract(model)
    if contract.input_types is not None:
        invalid = sorted(set(names) - set(contract.input_types))
        if invalid:
            names = ", ".join(invalid)
            raise TypeError(f"with_context() only accepts contextual fields. Invalid field(s): {names}.")


def _binding_uses_patch_shape(binding: ContextTransform) -> bool:
    return _is_mapping_annotation(_load_context_transform_config_from_binding(binding).return_annotation)


def _validate_context_transform_factory_kwargs(config: _FlowModelConfig, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    unknown = sorted(set(kwargs) - {param.name for param in config.parameters})
    if unknown:
        raise TypeError(f"{_callable_name(config.func)}() got unexpected keyword argument(s): {', '.join(unknown)}")

    contextual = sorted(name for name in kwargs if config.param(name).is_contextual)
    if contextual:
        raise TypeError(
            f"{_callable_name(config.func)}() only binds regular parameters. Do not pass contextual parameter(s): {', '.join(contextual)}."
        )

    missing = [param.name for param in config.regular_params if param.name not in kwargs and not param.has_function_default]
    if missing:
        raise TypeError(f"{_callable_name(config.func)}() is missing required regular parameter(s): {', '.join(missing)}")

    validated: Dict[str, Any] = {}
    for param in config.regular_params:
        if param.name not in kwargs:
            continue
        value = kwargs[param.name]
        validated[param.name] = _coerce_value(param.name, value, param.annotation, "Context transform argument")
    return validated


def _validate_patch_result(model: CallableModel, result: Any) -> Dict[str, Any]:
    if not isinstance(result, Mapping):
        raise TypeError(
            f"Patch context transform for {model!r} must return a mapping of contextual field names to values, got {type(result).__name__}."
        )

    patch = dict(result)
    if not all(isinstance(name, str) for name in patch):
        raise TypeError("Patch context transforms must return a mapping with string field names.")

    _validate_with_context_field_names(model, list(patch))
    contract = _model_context_contract(model)
    if contract.input_types is None:
        return patch

    return {name: _coerce_model_context_value(model, name, value, "with_context() patch") for name, value in patch.items()}


def _normalize_with_context(model: CallableModel, patches: Tuple[Any, ...], field_overrides: Dict[str, Any]) -> _BoundContextSpec:
    normalized_patches = []
    for patch in patches:
        if callable(patch):
            raise TypeError("with_context() no longer accepts raw callables. Replace the callable with a top-level @Flow.context_transform binding.")
        if not isinstance(patch, ContextTransform):
            raise TypeError("Positional with_context() arguments must be @Flow.context_transform bindings that return a mapping.")
        if not _binding_uses_patch_shape(patch):
            raise TypeError(
                "Field context transforms must be passed by keyword to with_context(...). Patch transforms belong in positional arguments."
            )
        normalized_patches.append(PatchContextSpec(binding=patch))

    _validate_with_context_field_names(model, list(field_overrides))
    contract = _model_context_contract(model)
    normalized_field_overrides: Dict[str, _FieldOverrideSpec] = {}
    for name, value in field_overrides.items():
        if callable(value):
            raise TypeError("with_context() no longer accepts raw callables. Replace the callable with a top-level @Flow.context_transform binding.")
        if isinstance(value, ContextTransform):
            if _binding_uses_patch_shape(value):
                raise TypeError("Patch transforms must be passed positionally to with_context(...), not as keyword field overrides.")
            normalized_field_overrides[name] = FieldContextSpec(binding=value)
            continue
        normalized_field_overrides[name] = StaticValueSpec(
            value=value
            if contract.input_types is None or name not in contract.input_types
            else _coerce_model_context_value(model, name, value, "with_context()")
        )

    context_spec = _BoundContextSpec(patches=normalized_patches, field_overrides=normalized_field_overrides)
    return _validate_static_context_spec_declared_context(model, context_spec)


def _apply_context_spec_values(model: CallableModel, context_spec: _BoundContextSpec, context: ContextBase) -> Dict[str, Any]:
    current_values = _context_values(context)

    for patch in context_spec.patches:
        result = _evaluate_context_transform_from_values(patch.binding, _context_values(context))
        current_values.update(_validate_patch_result(model, result))

    for name, spec in context_spec.field_overrides.items():
        if isinstance(spec, StaticValueSpec):
            current_values[name] = spec.value
            continue
        result = _evaluate_context_transform_from_values(spec.binding, _context_values(context))
        current_values[name] = _coerce_model_context_value(model, name, result, "with_context()")

    return current_values


def _apply_context_spec(model: CallableModel, context_spec: _BoundContextSpec, context: ContextBase) -> ContextBase:
    if not context_spec.patches and not context_spec.field_overrides:
        return _dependency_context_for_model(model, context)

    values = _apply_context_spec_values(model, context_spec, context)
    return _runtime_context_for_model(model, _project_context_values_for_model(model, values))


def _build_compute_context(model: CallableModel, context: Any, kwargs: Dict[str, Any]) -> Optional[ContextBase]:
    if context is not _UNSET and kwargs:
        raise TypeError("compute() accepts either one context object or contextual keyword arguments, but not both.")

    ctx_type = model.context_type
    _ctx_is_optional = get_origin(ctx_type) in _UNION_ORIGINS and type(None) in get_args(ctx_type)

    contract = _model_context_contract(model)

    if context is not _UNSET:
        if context is None and _ctx_is_optional:
            return None
        if isinstance(context, FlowContext):
            return context
        if isinstance(context, ContextBase):
            return _runtime_context_for_model(model, _context_values(context))
        return contract.runtime_context_type.model_validate(context)

    if contract.generated_model is None:
        if not kwargs and _ctx_is_optional:
            return None
        return contract.runtime_context_type.model_validate(kwargs)

    generated = contract.generated_model
    config = type(generated).__flow_model_config__
    regular_kwargs = sorted(name for name in config.regular_param_names if name in kwargs)
    unresolved_regular = sorted(name for name in regular_kwargs if _is_unset_flow_input(getattr(generated, name, _UNSET_FLOW_INPUT)))
    if unresolved_regular:
        names = ", ".join(unresolved_regular)
        raise TypeError(
            f"compute() cannot satisfy unbound regular parameter(s): {names}. "
            "Bind them at construction time; compute() only supplies runtime context."
        )

    already_bound_regular = sorted(name for name in regular_kwargs if name not in unresolved_regular)
    if already_bound_regular:
        names = ", ".join(already_bound_regular)
        raise TypeError(
            f"compute() does not accept regular parameter override(s): {names}. "
            "Those parameters are already bound on the model. Pass a context object if you need ambient fields with the same names."
        )

    base_kwargs = sorted(name for name in _model_base_field_names(generated) if name in kwargs)
    if base_kwargs:
        names = ", ".join(base_kwargs)
        raise TypeError(
            f"compute() does not accept model configuration override(s): {names}. Those fields are bound on the model at construction time."
        )

    ambient = dict(kwargs)
    for param in config.contextual_params:
        if param.name not in kwargs:
            continue
        ambient[param.name] = _coerce_contextual_value(config, param, kwargs[param.name], "compute() input")
    return FlowContext(**ambient)


def _is_optional_context_type(context_type: Any) -> bool:
    return get_origin(context_type) in _UNION_ORIGINS and type(None) in get_args(context_type)


def _bound_model_preserves_none_context(bound_model: "BoundModel") -> bool:
    return (
        not bound_model.context_spec.patches
        and not bound_model.context_spec.field_overrides
        and _is_optional_context_type(bound_model.model.context_type)
    )


def _build_bound_compute_context(bound_model: "BoundModel", context: Any, kwargs: Dict[str, Any]) -> Optional[ContextBase]:
    if context is not _UNSET and kwargs:
        raise TypeError("compute() accepts either one context object or contextual keyword arguments, but not both.")
    if context is not _UNSET:
        return context
    if not kwargs and _bound_model_preserves_none_context(bound_model):
        return None
    return FlowContext(**kwargs)


class FlowAPI:
    """API namespace for contextual execution and context binding."""

    def __init__(self, model: CallableModel):
        self._model = model

    @property
    def _compute_target(self) -> CallableModel:
        return self._model

    def compute(self, context: Any = _UNSET, /, _options: Optional[FlowOptions] = None, **kwargs) -> Any:
        target = self._compute_target
        built_context = _build_compute_context(target, context, kwargs)
        return _maybe_auto_unwrap_external_result(target, target(built_context, _options=_options))

    @property
    def context_inputs(self) -> Dict[str, Any]:
        contract = _model_context_contract(self._model)
        return dict(contract.input_types or {})

    @property
    def unbound_inputs(self) -> Dict[str, Any]:
        contract = _model_context_contract(self._model)
        if contract.generated_model is None and _is_optional_context_type(self._model.context_type):
            return {}
        if contract.generated_model is None:
            return {} if contract.input_types is None else {name: contract.input_types[name] for name in contract.required_names}

        generated = contract.generated_model
        config = type(generated).__flow_model_config__
        result = {}
        for param in config.contextual_params:
            if not _is_unset_flow_input(getattr(generated, param.name, _UNSET_FLOW_INPUT)):
                continue
            if param.has_function_default:
                continue
            result[param.name] = param.annotation if contract.input_types is None else contract.input_types[param.name]
        return result

    @property
    def bound_inputs(self) -> Dict[str, Any]:
        generated = _model_context_contract(self._model).generated_model
        if generated is not None:
            config = type(generated).__flow_model_config__
            result: Dict[str, Any] = {}
            explicit_fields = _bound_field_names(generated)
            for param in config.regular_params:
                value = getattr(generated, param.name, _UNSET_FLOW_INPUT)
                if _is_unset_flow_input(value):
                    continue
                result[param.name] = value
            for param in config.contextual_params:
                if param.name not in explicit_fields:
                    continue
                value = getattr(generated, param.name, _UNSET_FLOW_INPUT)
                if _is_unset_flow_input(value):
                    continue
                result[param.name] = value
            for name in _model_base_field_names(generated):
                if name in explicit_fields:
                    result[name] = getattr(generated, name)
            return result

        result: Dict[str, Any] = {}
        model_fields = getattr(self._model.__class__, "model_fields", {})
        for name in model_fields:
            if name == "meta":
                continue
            result[name] = getattr(self._model, name)
        return result

    def with_context(self, *patches, **field_overrides) -> "BoundModel":
        context_spec = _normalize_with_context(self._model, patches, field_overrides)
        return BoundModel(model=self._model, context_spec=context_spec)


class BoundModel(WrapperModel):
    """A model with contextual input transforms applied locally."""

    context_spec: _BoundContextSpec = Field(default_factory=_BoundContextSpec, repr=False)

    def __reduce__(self):
        return (_restore_pickled_flow_model, (str(PyObjectPath.validate(type(self))), self.__getstate__()))

    def _rewrite_context(self, context: ContextBase) -> ContextBase:
        return _apply_context_spec(self.model, self.context_spec, context)

    @property
    def context_type(self) -> Any:
        if _bound_model_preserves_none_context(self):
            return FlowContext | None
        return FlowContext

    @Flow.call
    def __call__(self, context: _OptionalContextBase) -> ResultBase:
        if context is None and _bound_model_preserves_none_context(self):
            return self.model(None)
        return self.model(self._rewrite_context(context))

    @Flow.deps
    def __deps__(self, context: _OptionalContextBase) -> GraphDepList:
        if context is None and _bound_model_preserves_none_context(self):
            return [(self.model, [None])]
        return [(self.model, [self._rewrite_context(context)])]

    def __repr__(self) -> str:
        args = [_context_transform_repr(patch.binding) for patch in self.context_spec.patches]
        args.extend(
            f"{name}={_context_transform_repr(spec.binding if isinstance(spec, FieldContextSpec) else spec.value)}"
            for name, spec in self.context_spec.field_overrides.items()
        )
        return f"{self.model!r}.flow.with_context({', '.join(args)})"

    def _evaluation_identity_payload(
        self,
        context: ContextBase,
        child_evaluation_key: Callable[[CallableModel, ContextBase], bytes],
    ) -> Optional[Any]:
        return {
            "kind": "bound_model_v1",
            "model": child_evaluation_key(self.model, self._rewrite_context(context)),
        }

    @property
    def flow(self) -> "FlowAPI":
        return _BoundFlowAPI(self)


class _BoundFlowAPI(FlowAPI):
    def __init__(self, bound_model: BoundModel):
        self._bound = bound_model
        super().__init__(bound_model.model)

    @property
    def _compute_target(self) -> CallableModel:
        return self._bound

    def compute(self, context: Any = _UNSET, /, _options: Optional[FlowOptions] = None, **kwargs) -> Any:
        built_context = _build_bound_compute_context(self._bound, context, kwargs)
        return _maybe_auto_unwrap_external_result(self._bound, self._bound(built_context, _options=_options))

    @property
    def bound_inputs(self) -> Dict[str, Any]:
        result = super().bound_inputs
        for patch in self._bound.context_spec.patches:
            patch_result = _evaluate_static_context_transform(patch.binding)
            if patch_result is not _UNSET:
                result.update(_validate_patch_result(self._bound.model, patch_result))
        for name, spec in self._bound.context_spec.field_overrides.items():
            value = _static_field_override_value(self._bound.model, name, spec)
            if value is not _UNSET:
                result[name] = value
            else:
                result.pop(name, None)
        return result

    @property
    def context_inputs(self) -> Dict[str, Any]:
        result = super().context_inputs
        for name in _statically_resolved_context_field_names(self._bound.model, self._bound.context_spec):
            result.pop(name, None)
        for patch in self._bound.context_spec.patches:
            if _evaluate_static_context_transform(patch.binding) is _UNSET:
                result.update(_context_transform_input_types(patch.binding, required_only=False))
        for name, spec in self._bound.context_spec.field_overrides.items():
            if isinstance(spec, FieldContextSpec) and _static_field_override_value(self._bound.model, name, spec) is _UNSET:
                result.pop(name, None)
                result.update(_context_transform_input_types(spec.binding, required_only=False))
        return result

    @property
    def unbound_inputs(self) -> Dict[str, Any]:
        result = super().unbound_inputs
        for name in _statically_resolved_context_field_names(self._bound.model, self._bound.context_spec):
            result.pop(name, None)
        for patch in self._bound.context_spec.patches:
            if _evaluate_static_context_transform(patch.binding) is _UNSET:
                result.update(_context_transform_input_types(patch.binding, required_only=True))
        for name, spec in self._bound.context_spec.field_overrides.items():
            if isinstance(spec, FieldContextSpec) and _static_field_override_value(self._bound.model, name, spec) is _UNSET:
                result.pop(name, None)
                result.update(_context_transform_input_types(spec.binding, required_only=True))
        return result

    def with_context(self, *patches, **field_overrides) -> BoundModel:
        context_spec = _normalize_with_context(self._bound.model, patches, field_overrides)
        merged = _merge_context_specs(self._bound.context_spec, context_spec.patches, context_spec.field_overrides)
        return BoundModel(
            model=self._bound.model,
            context_spec=_validate_static_context_spec_declared_context(self._bound.model, merged),
        )


class _GeneratedFlowModelBase(CallableModel):
    __flow_model_config__: ClassVar[_FlowModelConfig]

    def __reduce__(self):
        config = type(self).__flow_model_config__
        factory_path = _generated_model_factory_path_for_pickle(config, type(self))
        if factory_path is not None:
            return (_restore_generated_flow_model, (factory_path, self.__getstate__()))
        import cloudpickle

        payload = (config.func, type(self).__flow_model_factory_kwargs__)
        return (_restore_pickled_local_flow_model, (cloudpickle.dumps(payload, protocol=5), self.__getstate__()))

    @model_validator(mode="before")
    @classmethod
    def _resolve_registry_refs(cls, values):
        if not isinstance(values, dict):
            return values

        config = getattr(cls, "__flow_model_config__", None)
        if config is None:
            return values

        resolved = dict(values)
        for param in config.regular_params:
            if param.name not in resolved:
                continue
            value = resolved[param.name]
            if not isinstance(value, str):
                continue
            if _type_accepts_str(param.annotation):
                continue
            candidate = _resolve_registry_candidate(value)
            if candidate is None:
                continue
            if _registry_candidate_allowed(param.annotation, candidate):
                resolved[param.name] = candidate
        return resolved

    @model_validator(mode="after")
    def _validate_flow_model_fields(self):
        config = self.__class__.__flow_model_config__

        for param in config.parameters:
            value = getattr(self, param.name, _UNSET_FLOW_INPUT)
            if _is_unset_flow_input(value):
                continue
            object.__setattr__(
                self,
                param.name,
                _validate_bound_param_value(config, param, value, "Contextual default" if param.is_contextual else "Field"),
            )

        _validate_bound_declared_context_defaults(self, config)
        return self

    @property
    def context_type(self) -> Type[ContextBase]:
        return self.__class__.__flow_model_config__.context_type

    @property
    def result_type(self) -> Type[ResultBase]:
        return self.__class__.__flow_model_config__.result_type

    @property
    def flow(self) -> FlowAPI:
        return FlowAPI(self)

    def _evaluation_identity_payload(
        self,
        context: ContextBase,
        child_evaluation_key: Callable[[CallableModel, ContextBase], bytes],
    ) -> Optional[Any]:
        return _generated_model_identity_payload(self, context, child_evaluation_key)


def _make_call_impl(config: _FlowModelConfig) -> _AnyCallable:
    def __call__(self, context):
        missing_regular = _missing_regular_param_names(self, config)
        if missing_regular:
            missing = ", ".join(sorted(missing_regular))
            raise TypeError(
                f"Missing regular parameter(s) for {_callable_name(config.func)}: {missing}. "
                "Bind them at construction time; compute() only supplies contextual inputs."
            )

        fn_kwargs: Dict[str, Any] = {}
        for param in config.regular_params:
            value = _resolve_regular_param_value(self, param, context)
            if param.is_lazy:
                fn_kwargs[param.name] = _make_coercing_lazy_thunk(value, param.name, param.annotation)
            else:
                fn_kwargs[param.name] = _coerce_value(param.name, value, param.annotation, "Regular parameter")

        fn_kwargs.update(_resolved_contextual_inputs(self, config, context))

        raw_result = config.func(**fn_kwargs)
        if config.auto_wrap_result:
            return config.result_type.model_validate(raw_result)
        return raw_result

    cast(Any, __call__).__signature__ = inspect.Signature(
        parameters=[
            inspect.Parameter("self", inspect.Parameter.POSITIONAL_OR_KEYWORD),
            inspect.Parameter("context", inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=config.context_type),
        ],
        return_annotation=config.result_type,
    )
    return __call__


def _make_deps_impl(config: _FlowModelConfig) -> _AnyCallable:
    def __deps__(self, context):
        missing_regular = _missing_regular_param_names(self, config)
        if missing_regular:
            missing = ", ".join(sorted(missing_regular))
            raise TypeError(f"Missing regular parameter(s) for {_callable_name(config.func)}: {missing}. Bind them before dependency evaluation.")

        deps = []
        for param in config.regular_params:
            if param.is_lazy:
                continue
            value = getattr(self, param.name, _UNSET_FLOW_INPUT)
            if isinstance(value, CallableModel):
                dependency_model, dependency_context = _resolved_dependency_invocation(value, context)
                deps.append((dependency_model, [dependency_context]))
        return deps

    cast(Any, __deps__).__signature__ = inspect.Signature(
        parameters=[
            inspect.Parameter("self", inspect.Parameter.POSITIONAL_OR_KEYWORD),
            inspect.Parameter("context", inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=config.context_type),
        ],
        return_annotation=GraphDepList,
    )
    return __deps__


def _resolve_generated_model_bases(model_base: Type[CallableModel]) -> Tuple[type, ...]:
    if not isinstance(model_base, type) or not issubclass(model_base, CallableModel):
        raise TypeError(f"model_base must be a CallableModel subclass, got {model_base!r}")

    if issubclass(model_base, _GeneratedFlowModelBase):
        return (model_base,)
    if model_base is CallableModel:
        return (_GeneratedFlowModelBase,)
    return (_GeneratedFlowModelBase, model_base)


def flow_context_transform(func: Optional[_AnyCallable] = None) -> _AnyCallable:
    """Decorator that turns a top-level function into a serializable with_context() transform factory."""

    def decorator(fn: _AnyCallable) -> _AnyCallable:
        _ensure_top_level_named_function(fn, decorator_name="@Flow.context_transform")
        try:
            resolved_hints = get_type_hints(fn, include_extras=True)
        except AttributeError:
            resolved_hints = {}
        sig = _resolved_flow_signature(
            fn,
            resolved_hints=resolved_hints,
            require_return_annotation=True,
            function_name=_callable_name(fn),
        )
        config = _analyze_flow_context_transform(fn, sig, is_model_dependency=_is_model_dependency)
        serialized_config = None if _context_transform_should_use_import_path(config) else _serialize_context_transform_config(config)

        @wraps(fn)
        def factory(**kwargs) -> ContextTransform:
            return ContextTransform(
                path=config.path if serialized_config is None else None,
                serialized_config=serialized_config,
                bound_args=_validate_context_transform_factory_kwargs(config, kwargs),
            )

        cast(Any, factory).__flow_context_transform_config__ = config
        return factory

    if func is not None:
        return decorator(func)
    return decorator


def flow_model(
    func: Optional[_AnyCallable] = None,
    *,
    context_args: Any = _REMOVED_CONTEXT_ARGS,
    context_type: Optional[Type[ContextBase]] = None,
    auto_unwrap: bool = False,
    model_base: Type[CallableModel] = CallableModel,
    cacheable: Any = _UNSET,
    volatile: Any = _UNSET,
    log_level: Any = _UNSET,
    validate_result: Any = _UNSET,
    verbose: Any = _UNSET,
    evaluator: Any = _UNSET,
) -> _AnyCallable:
    """Decorator that generates a CallableModel class from a plain Python function."""

    if context_args is not _REMOVED_CONTEXT_ARGS:
        raise TypeError("context_args=... has been removed. Mark runtime/contextual parameters with FromContext[...] instead.")

    def decorator(fn: _AnyCallable) -> _AnyCallable:
        try:
            resolved_hints = get_type_hints(fn, include_extras=True)
        except AttributeError:
            resolved_hints = {}
        sig = _resolved_flow_signature(
            fn,
            resolved_hints=resolved_hints,
            require_return_annotation=True,
            function_name=_callable_name(fn),
        )
        config = _analyze_flow_model(
            fn,
            sig,
            context_type=context_type,
            auto_unwrap=auto_unwrap,
            is_model_dependency=_is_model_dependency,
        )

        annotations: Dict[str, Any] = {}
        namespace: Dict[str, Any] = {
            "__module__": getattr(fn, "__module__", __name__),
            "__qualname__": f"_{_callable_name(fn)}_Model",
            "__call__": Flow.call(
                **{
                    name: value
                    for name, value in [
                        ("cacheable", cacheable),
                        ("volatile", volatile),
                        ("log_level", log_level),
                        ("validate_result", validate_result),
                        ("verbose", verbose),
                        ("evaluator", evaluator),
                    ]
                    if value is not _UNSET
                }
            )(_make_call_impl(config)),
            "__deps__": Flow.deps(_make_deps_impl(config)),
        }

        for param in config.parameters:
            annotations[param.name] = Any
            if param.is_contextual:
                namespace[param.name] = Field(default_factory=_unset_flow_input_factory, exclude_if=_is_unset_flow_input)
            elif param.has_function_default:
                namespace[param.name] = param.function_default
            else:
                namespace[param.name] = Field(default_factory=_unset_flow_input_factory, exclude_if=_is_unset_flow_input)

        namespace["__annotations__"] = annotations

        GeneratedModel = cast(
            type[_GeneratedFlowModelBase],
            type(f"_{_callable_name(fn)}_Model", _resolve_generated_model_bases(model_base), namespace),
        )
        GeneratedModel.__flow_model_config__ = config
        GeneratedModel.__flow_model_factory_kwargs__ = {
            "context_type": context_type,
            "auto_unwrap": auto_unwrap,
            "model_base": model_base,
            "cacheable": cacheable,
            "volatile": volatile,
            "log_level": log_level,
            "validate_result": validate_result,
            "verbose": verbose,
            "evaluator": evaluator,
        }
        register_ccflow_import_path(GeneratedModel)
        GeneratedModel.model_rebuild()

        @wraps(fn)
        def factory(**kwargs) -> _GeneratedFlowModelBase:
            return GeneratedModel(**kwargs)

        cast(Any, factory)._generated_model = GeneratedModel
        factory.__doc__ = fn.__doc__
        return factory

    if func is not None:
        return decorator(func)
    return decorator
