"""Flow.model decorator implementation built around ``FromContext``."""

import inspect
import logging
from collections import OrderedDict
from dataclasses import dataclass, field
from functools import lru_cache, wraps
from types import UnionType
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
    Tuple,
    Type,
    Union,
    cast,
    get_args,
    get_origin,
    get_type_hints,
)

from pydantic import BaseModel as PydanticModel, Field, TypeAdapter, ValidationError, model_validator
from pydantic.errors import PydanticSchemaGenerationError, PydanticUndefinedAnnotation

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
    "StaticValueSpec",
    "FieldContextSpec",
    "PatchContextSpec",
    "clear_flow_model_caches",
    "flow_context_transform",
)

_AnyCallable = Callable[..., Any]
log = logging.getLogger(__name__)


class _UnsetFlowInput:
    def __repr__(self) -> str:
        return "<unset>"

    def __reduce__(self):
        return (_unset_flow_input_factory, ())


class _InternalSentinel:
    def __init__(self, name: str):
        self._name = name

    def __repr__(self) -> str:
        return self._name

    def __reduce__(self):
        return (_get_internal_sentinel, (self._name,))


_UNSET_FLOW_INPUT = _UnsetFlowInput()
_UNION_ORIGINS = (Union, UnionType)
_TYPE_ADAPTER_CACHE_MAXSIZE = 256
_HASHABLE_TYPE_ADAPTER_CACHE: "OrderedDict[Any, TypeAdapter]" = OrderedDict()
_UNHASHABLE_TYPE_ADAPTER_CACHE: "OrderedDict[int, Tuple[Any, TypeAdapter]]" = OrderedDict()


def _get_internal_sentinel(name: str) -> _InternalSentinel:
    return _INTERNAL_SENTINELS[name]


_INTERNAL_SENTINELS = {
    "_UNSET": _InternalSentinel("_UNSET"),
    "_REMOVED_CONTEXT_ARGS": _InternalSentinel("_REMOVED_CONTEXT_ARGS"),
}
_UNSET = _INTERNAL_SENTINELS["_UNSET"]
_REMOVED_CONTEXT_ARGS = _INTERNAL_SENTINELS["_REMOVED_CONTEXT_ARGS"]


def _unset_flow_input_factory() -> _UnsetFlowInput:
    return _UNSET_FLOW_INPUT


def _is_unset_flow_input(value: Any) -> bool:
    return value is _UNSET_FLOW_INPUT


class _LazyMarker:
    pass


class _FromContextMarker:
    pass


class FromContext:
    """Marker used in ``@Flow.model`` signatures for runtime/contextual inputs."""

    def __class_getitem__(cls, item):
        return Annotated[item, _FromContextMarker()]


class Lazy:
    """Lazy dependency marker used only as ``Lazy[T]`` in type annotations."""

    def __new__(cls, *args, **kwargs):
        raise TypeError("Lazy(model)(...) has been removed. Use model.flow.with_context(...) for contextual rewrites.")

    def __class_getitem__(cls, item):
        return Annotated[item, _LazyMarker()]


@dataclass(frozen=True)
class _ParsedAnnotation:
    base: Any
    is_lazy: bool
    is_from_context: bool


def _callable_name(func: _AnyCallable) -> str:
    return getattr(func, "__name__", type(func).__name__)


def _callable_qualname(func: _AnyCallable) -> str:
    return getattr(func, "__qualname__", type(func).__qualname__)


def _resolved_flow_signature(
    fn: _AnyCallable,
    *,
    resolved_hints: Optional[Dict[str, Any]] = None,
    skip_self: bool = False,
    require_return_annotation: bool = False,
    annotation_error_suffix: str = "",
    return_error_suffix: str = "",
    function_name: Optional[str] = None,
) -> inspect.Signature:
    sig = inspect.signature(fn)
    resolved_hints = resolved_hints or {}
    function_name = function_name or _callable_name(fn)
    parameters = []

    for name, param in sig.parameters.items():
        if skip_self and name == "self":
            continue
        if param.kind is inspect.Parameter.POSITIONAL_ONLY:
            raise TypeError(f"Function {function_name} does not support positional-only parameter '{name}'.")
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            raise TypeError(f"Function {function_name} does not support {param.kind.description} parameter '{name}'.")

        annotation = resolved_hints.get(name, param.annotation)
        if annotation is inspect.Parameter.empty:
            raise TypeError(f"Parameter '{name}' must have a type annotation{annotation_error_suffix}")

        parameters.append(param.replace(annotation=annotation))

    return_annotation = resolved_hints.get("return", sig.return_annotation)
    if require_return_annotation and return_annotation is inspect.Signature.empty:
        raise TypeError(f"Function {function_name} must have a return type annotation{return_error_suffix}")

    return sig.replace(parameters=parameters, return_annotation=return_annotation)


@dataclass(frozen=True)
class _FlowModelParam:
    name: str
    annotation: Any
    kind: str
    is_lazy: bool
    has_function_default: bool
    function_default: Any = _UNSET
    context_validation_annotation: Any = _UNSET

    @property
    def is_contextual(self) -> bool:
        return self.kind == "contextual"

    @property
    def validation_annotation(self) -> Any:
        if self.context_validation_annotation is not _UNSET:
            return self.context_validation_annotation
        return self.annotation


@dataclass(frozen=True)
class _FlowModelConfig:
    func: _AnyCallable
    return_annotation: Any
    context_type: Type[ContextBase]
    result_type: Type[ResultBase]
    auto_wrap_result: bool
    auto_unwrap: bool
    parameters: Tuple[_FlowModelParam, ...]
    context_input_types: Dict[str, Any]
    context_required_names: Tuple[str, ...]
    declared_context_type: Optional[Type[ContextBase]] = None
    path: Optional[PyObjectPath] = None
    _regular_params: Tuple[_FlowModelParam, ...] = field(init=False, repr=False)
    _contextual_params: Tuple[_FlowModelParam, ...] = field(init=False, repr=False)
    _regular_param_names: Tuple[str, ...] = field(init=False, repr=False)
    _contextual_param_names: Tuple[str, ...] = field(init=False, repr=False)
    _params_by_name: Dict[str, _FlowModelParam] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        regular = tuple(param for param in self.parameters if not param.is_contextual)
        contextual = tuple(param for param in self.parameters if param.is_contextual)
        object.__setattr__(self, "_regular_params", regular)
        object.__setattr__(self, "_contextual_params", contextual)
        object.__setattr__(self, "_regular_param_names", tuple(param.name for param in regular))
        object.__setattr__(self, "_contextual_param_names", tuple(param.name for param in contextual))
        object.__setattr__(self, "_params_by_name", {param.name: param for param in self.parameters})

    @property
    def regular_params(self) -> Tuple[_FlowModelParam, ...]:
        return self._regular_params

    @property
    def contextual_params(self) -> Tuple[_FlowModelParam, ...]:
        return self._contextual_params

    @property
    def regular_param_names(self) -> Tuple[str, ...]:
        return self._regular_param_names

    @property
    def contextual_param_names(self) -> Tuple[str, ...]:
        return self._contextual_param_names

    def param(self, name: str) -> _FlowModelParam:
        return self._params_by_name[name]


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
    path: PyObjectPath
    bound_args: Dict[str, Any] = Field(default_factory=dict)


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
        name = str(transform.path).rsplit(".", 1)[-1]
        if not transform.bound_args:
            return f"{name}()"
        args = ", ".join(f"{key}={value!r}" for key, value in sorted(transform.bound_args.items()))
        return f"{name}({args})"
    return repr(transform)


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


def _make_lazy_thunk(model: CallableModel, context: ContextBase) -> Callable[[], Any]:
    cache: Dict[str, Any] = {}

    def thunk():
        if "result" not in cache:
            cache["result"] = _unwrap_model_result(model(context))
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


def _parse_annotation(annotation: Any) -> _ParsedAnnotation:
    is_lazy = False
    is_from_context = False

    while get_origin(annotation) is Annotated:
        args = get_args(annotation)
        annotation = args[0]
        for metadata in args[1:]:
            if isinstance(metadata, _LazyMarker):
                is_lazy = True
            elif isinstance(metadata, _FromContextMarker):
                is_from_context = True

    return _ParsedAnnotation(base=annotation, is_lazy=is_lazy, is_from_context=is_from_context)


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
        raise TypeError(f"{decorator_name} only supports top-level named Python functions.")

    name = getattr(fn, "__name__", "")
    qualname = getattr(fn, "__qualname__", "")
    if name == "<lambda>" or qualname != name or "<locals>" in qualname:
        raise TypeError(f"{decorator_name} only supports top-level named Python functions.")


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


def clear_flow_model_caches() -> None:
    """Clear module-level caches used by Flow.model internals."""

    _HASHABLE_TYPE_ADAPTER_CACHE.clear()
    _UNHASHABLE_TYPE_ADAPTER_CACHE.clear()
    _load_context_transform_factory.cache_clear()
    _load_context_transform_config.cache_clear()


def _strip_annotated(annotation: Any) -> Any:
    while get_origin(annotation) is Annotated:
        annotation = get_args(annotation)[0]
    return annotation


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
    return bool(module and name and qualname and qualname == name and "<locals>" not in qualname)


def _runtime_context_for_model(model: CallableModel, values: Dict[str, Any]) -> ContextBase:
    contract = _model_context_contract(model)
    if contract.runtime_context_type is FlowContext:
        return FlowContext(**values)
    return contract.runtime_context_type.model_validate(values)


def _dependency_context_values(model: CallableModel, context: ContextBase) -> Dict[str, Any]:
    values = _context_values(context)
    contract = _model_context_contract(model)
    if contract.runtime_context_type is FlowContext or contract.input_types is None:
        return values
    return {name: values[name] for name in contract.input_types if name in values}


def _dependency_context_for_model(model: CallableModel, context: ContextBase) -> ContextBase:
    return _runtime_context_for_model(model, _dependency_context_values(model, context))


def _resolved_dependency_invocation(value: CallableModel, context: ContextBase) -> Tuple[CallableModel, ContextBase]:
    if isinstance(value, BoundModel):
        return value.model, value._rewrite_context(context)
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


def _generated_model_class(stage: Any) -> Optional[type["_GeneratedFlowModelBase"]]:
    model = _generated_model_instance(stage)
    if model is not None:
        return type(model)

    generated_model = getattr(stage, "_generated_model", None)
    if isinstance(generated_model, type) and issubclass(generated_model, _GeneratedFlowModelBase):
        return generated_model
    return None


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
        dependency_model, dependency_context = _resolved_dependency_invocation(value, context)
        if param.is_lazy:
            return _make_lazy_thunk(dependency_model, dependency_context)
        return _unwrap_model_result(dependency_model(dependency_context))
    if param.is_lazy:
        return lambda v=value: v
    return value


def _resolved_contextual_inputs(model: "_GeneratedFlowModelBase", config: _FlowModelConfig, context: ContextBase) -> Dict[str, Any]:
    context_values = _context_values(context)
    resolved: Dict[str, Any] = {}
    missing_contextual = []

    for param in config.contextual_params:
        if param.name in context_values:
            resolved[param.name] = context_values[param.name]
            continue
        value = getattr(model, param.name, _UNSET_FLOW_INPUT)
        if not _is_unset_flow_input(value):
            resolved[param.name] = value
            continue
        if param.has_function_default:
            resolved[param.name] = param.function_default
            continue
        missing_contextual.append(param.name)

    if missing_contextual:
        missing = ", ".join(sorted(missing_contextual))
        raise TypeError(
            f"Missing contextual input(s) for {_callable_name(config.func)}: {missing}. "
            "Supply them via the runtime context, compute(), with_context(), or construction-time contextual defaults."
        )

    if config.declared_context_type is not None:
        validated = config.declared_context_type.model_validate(resolved)
        return {param.name: getattr(validated, param.name) for param in config.contextual_params}

    return {
        param.name: _coerce_value(param.name, resolved[param.name], param.validation_annotation, "Context field")
        for param in config.contextual_params
    }


def _validate_declared_context_values(config: _FlowModelConfig, values: Dict[str, Any]) -> Dict[str, Any]:
    if config.declared_context_type is None:
        return values

    validated = config.declared_context_type.model_validate(values)
    return {param.name: getattr(validated, param.name) for param in config.contextual_params}


def _generated_model_identity_payload(
    model: "_GeneratedFlowModelBase",
    context: ContextBase,
    child_evaluation_key: Callable[[CallableModel, ContextBase], bytes],
) -> Optional[Dict[str, Any]]:
    config = type(model).__flow_model_config__
    regular_inputs = []
    for param in config.regular_params:
        value = getattr(model, param.name, _UNSET_FLOW_INPUT)
        if _is_unset_flow_input(value):
            return None

        descriptor = {"name": param.name, "lazy": param.is_lazy}
        if _is_model_dependency(value):
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
    resolved: Dict[str, Any] = {}
    static_overrides = static_overrides or {}

    for param in config.contextual_params:
        if param.name in static_overrides:
            resolved[param.name] = static_overrides[param.name].value
            continue

        value = getattr(model, param.name, _UNSET_FLOW_INPUT)
        if not _is_unset_flow_input(value):
            resolved[param.name] = value
            continue

        if param.has_function_default:
            resolved[param.name] = param.function_default
            continue

        return None

    return resolved


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


def _evaluate_static_context_transform(binding: ContextTransform) -> Any:
    config = _load_context_transform_config(str(binding.path))
    kwargs: Dict[str, Any] = {}
    for param in config.regular_params:
        if param.name in binding.bound_args:
            kwargs[param.name] = binding.bound_args[param.name]
        elif param.has_function_default:
            kwargs[param.name] = param.function_default
        else:
            raise TypeError(f"Context transform '{config.path}' is missing required regular parameter '{param.name}'.")

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
    return _coerce_value(field_name, value, contract.input_types[field_name], "with_context()")


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
    return _is_mapping_annotation(_load_context_transform_config(str(binding.path)).return_annotation)


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


def _evaluate_context_transform(binding: ContextTransform, context: ContextBase) -> Any:
    config = _load_context_transform_config(str(binding.path))
    kwargs: Dict[str, Any] = {}
    for param in config.regular_params:
        if param.name in binding.bound_args:
            kwargs[param.name] = binding.bound_args[param.name]
        elif param.has_function_default:
            kwargs[param.name] = param.function_default
        else:
            raise TypeError(f"Context transform '{config.path}' is missing required regular parameter '{param.name}'.")

    context_values = _context_values(context)
    missing = []
    for param in config.contextual_params:
        if param.name in context_values:
            kwargs[param.name] = _coerce_value(param.name, context_values[param.name], param.annotation, "Context transform field")
            continue
        if param.has_function_default:
            kwargs[param.name] = param.function_default
            continue
        missing.append(param.name)

    if missing:
        raise TypeError(
            f"Missing contextual input(s) for context transform {_callable_name(config.func)}: {', '.join(sorted(missing))}. "
            "Supply them via the runtime context or with_context() ordering."
        )
    return config.func(**kwargs)


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

    return {name: _coerce_value(name, value, contract.input_types[name], "with_context() patch") for name, value in patch.items()}


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
            else _coerce_value(name, value, contract.input_types[name], "with_context()")
        )

    context_spec = _BoundContextSpec(patches=normalized_patches, field_overrides=normalized_field_overrides)
    return _validate_static_context_spec_declared_context(model, context_spec)


def _apply_context_spec(model: CallableModel, context_spec: _BoundContextSpec, context: ContextBase) -> ContextBase:
    if not context_spec.patches and not context_spec.field_overrides:
        return context

    current_values = _context_values(context)
    # All transforms evaluate against the original runtime context. `current_values`
    # only accumulates their outputs before we rebuild the final context object.
    original_context = _runtime_context_for_model(model, current_values)

    for patch in context_spec.patches:
        current_values.update(_validate_patch_result(model, _evaluate_context_transform(patch.binding, original_context)))

    for name, spec in context_spec.field_overrides.items():
        if isinstance(spec, StaticValueSpec):
            current_values[name] = spec.value
            continue
        value = _evaluate_context_transform(spec.binding, original_context)
        contract = _model_context_contract(model)
        current_values[name] = (
            value
            if contract.input_types is None or name not in contract.input_types
            else _coerce_value(name, value, contract.input_types[name], "with_context()")
        )

    return _runtime_context_for_model(model, current_values)


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
        ambient[param.name] = _coerce_value(param.name, kwargs[param.name], param.validation_annotation, "compute() input")
    return FlowContext(**ambient)


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

    @Flow.call
    def __call__(self, context: ContextBase) -> ResultBase:
        return self.model(self._rewrite_context(context))

    @Flow.deps
    def __deps__(self, context: ContextBase) -> GraphDepList:
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
        return self.model._evaluation_identity_payload(self._rewrite_context(context), child_evaluation_key)

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

    def with_context(self, *patches, **field_overrides) -> BoundModel:
        context_spec = _normalize_with_context(self._bound.model, patches, field_overrides)
        return BoundModel(
            model=self._bound.model,
            context_spec=_merge_context_specs(self._bound.context_spec, context_spec.patches, context_spec.field_overrides),
        )


class _GeneratedFlowModelBase(CallableModel):
    __flow_model_config__: ClassVar[_FlowModelConfig]

    def __reduce__(self):
        config = type(self).__flow_model_config__
        if _is_importable_function(config.func):
            factory_path = f"{config.func.__module__}.{config.func.__name__}"
            return (_restore_generated_flow_model, (factory_path, self.__getstate__()))
        return (_restore_pickled_flow_model, (str(PyObjectPath.validate(type(self))), self.__getstate__()))

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

            if param.is_contextual:
                if _is_model_dependency(value):
                    raise TypeError(
                        f"Parameter '{param.name}' is marked FromContext[...] and cannot be bound to a CallableModel. "
                        "Bind a literal contextual default or supply it via compute()/with_context()."
                    )
                object.__setattr__(
                    self,
                    param.name,
                    _coerce_value(param.name, value, param.validation_annotation, "Contextual default"),
                )
                continue

            if _is_model_dependency(value):
                continue

            object.__setattr__(self, param.name, _coerce_value(param.name, value, param.annotation, "Field"))

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


def _context_type_annotations_compatible(func_annotation: Any, context_annotation: Any) -> bool:
    func_annotation = _strip_annotated(func_annotation)
    context_annotation = _strip_annotated(context_annotation)

    if func_annotation is Any:
        return True
    if context_annotation is Any:
        return True
    if func_annotation is context_annotation or func_annotation == context_annotation:
        return True

    func_origin = get_origin(func_annotation)
    context_origin = get_origin(context_annotation)

    if func_origin in _UNION_ORIGINS:
        func_args = tuple(arg for arg in get_args(func_annotation) if arg is not type(None))
        if context_origin in _UNION_ORIGINS:
            context_args = tuple(arg for arg in get_args(context_annotation) if arg is not type(None))
            return bool(context_args) and all(
                any(_context_type_annotations_compatible(func_arg, context_arg) for func_arg in func_args) for context_arg in context_args
            )
        return any(_context_type_annotations_compatible(func_arg, context_annotation) for func_arg in func_args)

    if context_origin in _UNION_ORIGINS:
        context_args = tuple(arg for arg in get_args(context_annotation) if arg is not type(None))
        return bool(context_args) and all(_context_type_annotations_compatible(func_annotation, context_arg) for context_arg in context_args)

    if func_origin is Literal or context_origin is Literal:
        return func_annotation == context_annotation

    func_base = func_origin or func_annotation
    context_base = context_origin or context_annotation
    if isinstance(func_base, type) and isinstance(context_base, type):
        if not issubclass(context_base, func_base):
            return False
    elif func_base != context_base:
        return False

    func_args = get_args(func_annotation)
    context_args = get_args(context_annotation)
    if bool(func_args) != bool(context_args):
        return False
    if len(func_args) != len(context_args):
        return False

    for func_arg, context_arg in zip(func_args, context_args):
        if get_origin(func_arg) is not None or get_origin(context_arg) is not None or isinstance(func_arg, type) or isinstance(context_arg, type):
            if not _context_type_annotations_compatible(func_arg, context_arg):
                return False
        elif func_arg != context_arg:
            return False

    return True


def _validate_declared_context_type(context_type: Any, contextual_params: Tuple[_FlowModelParam, ...]) -> Type[ContextBase]:
    if not isinstance(context_type, type) or not issubclass(context_type, ContextBase):
        raise TypeError(f"context_type must be a ContextBase subclass, got {context_type!r}")

    context_fields = getattr(context_type, "model_fields", {})
    contextual_names = {param.name for param in contextual_params}

    missing = sorted(name for name in contextual_names if name not in context_fields)
    if missing:
        raise TypeError(f"context_type {context_type.__name__} must define fields for all FromContext parameters: {', '.join(missing)}")

    required_extra_fields = sorted(
        name for name, info in context_fields.items() if name not in ContextBase.model_fields and name not in contextual_names and info.is_required()
    )
    if required_extra_fields:
        raise TypeError(
            f"context_type {context_type.__name__} has required fields that are not declared as FromContext parameters: "
            f"{', '.join(required_extra_fields)}"
        )

    for param in contextual_params:
        ctx_field = context_fields[param.name]
        if not _context_type_annotations_compatible(param.annotation, ctx_field.annotation):
            raise TypeError(
                f"FromContext parameter '{param.name}' annotates {param.annotation!r}, but "
                f"context_type {context_type.__name__} declares {ctx_field.annotation!r}."
            )

    return context_type


def _analyze_flow_function(fn: _AnyCallable, sig: inspect.Signature) -> Tuple[Tuple[_FlowModelParam, ...], Dict[str, Any]]:
    analyzed_params: List[_FlowModelParam] = []

    for param in sig.parameters.values():
        parsed = _parse_annotation(param.annotation)
        if parsed.is_lazy and parsed.is_from_context:
            raise TypeError(f"Parameter '{param.name}' cannot combine Lazy[...] and FromContext[...].")
        has_default = param.default is not inspect.Parameter.empty
        if parsed.is_from_context and has_default and _is_model_dependency(param.default):
            raise TypeError(f"Parameter '{param.name}' is marked FromContext[...] and cannot default to a CallableModel.")

        analyzed_params.append(
            _FlowModelParam(
                name=param.name,
                annotation=parsed.base,
                kind="contextual" if parsed.is_from_context else "regular",
                is_lazy=parsed.is_lazy,
                has_function_default=has_default,
                function_default=param.default if has_default else _UNSET,
            )
        )

    analyzed_params_tuple = tuple(analyzed_params)
    return analyzed_params_tuple, {param.name: param.annotation for param in analyzed_params_tuple if param.is_contextual}


def _analyze_flow_model(
    fn: _AnyCallable,
    sig: inspect.Signature,
    *,
    context_type: Optional[Type[ContextBase]],
    auto_unwrap: bool,
) -> _FlowModelConfig:
    parameters, context_input_types = _analyze_flow_function(fn, sig)
    contextual_params = tuple(param for param in parameters if param.is_contextual)
    declared_context_type = None
    if context_type is not None and not contextual_params:
        raise TypeError("context_type=... requires FromContext[...] parameters.")
    if context_type is not None:
        declared_context_type = _validate_declared_context_type(context_type, contextual_params)

    context_required_names = tuple(param.name for param in contextual_params if not param.has_function_default)

    if declared_context_type is not None:
        updated_params = []
        context_fields = declared_context_type.model_fields
        for param in parameters:
            if not param.is_contextual:
                updated_params.append(param)
                continue
            updated_params.append(
                _FlowModelParam(
                    name=param.name,
                    annotation=param.annotation,
                    kind=param.kind,
                    is_lazy=param.is_lazy,
                    has_function_default=param.has_function_default,
                    function_default=param.function_default,
                    context_validation_annotation=context_fields[param.name].annotation,
                )
            )
        parameters = tuple(updated_params)

    return_origin = get_origin(sig.return_annotation) or sig.return_annotation
    auto_wrap_result = not (isinstance(return_origin, type) and issubclass(return_origin, ResultBase))
    result_type = GenericResult[sig.return_annotation] if auto_wrap_result else sig.return_annotation

    return _FlowModelConfig(
        func=fn,
        return_annotation=sig.return_annotation,
        context_type=FlowContext,
        result_type=result_type,
        auto_wrap_result=auto_wrap_result,
        auto_unwrap=auto_unwrap,
        parameters=parameters,
        context_input_types=context_input_types,
        context_required_names=context_required_names,
        declared_context_type=declared_context_type,
    )


def _validate_factory_kwargs(config: _FlowModelConfig, kwargs: Dict[str, Any]) -> None:
    for param in config.parameters:
        if param.name not in kwargs:
            continue
        value = kwargs[param.name]
        if param.is_contextual:
            if _is_model_dependency(value):
                raise TypeError(
                    f"Parameter '{param.name}' is marked FromContext[...] and cannot be bound to a CallableModel. "
                    "Use a literal contextual default or supply it at runtime."
                )
            _coerce_value(param.name, value, param.validation_annotation, "Field")
            continue

        if value is None or _is_model_dependency(value):
            continue
        if isinstance(value, str) and not _type_accepts_str(param.annotation):
            candidate = _resolve_registry_candidate(value)
            if candidate is not None and _registry_candidate_allowed(param.annotation, candidate):
                continue
        _coerce_value(param.name, value, param.annotation, "Field")


def _resolve_generated_model_bases(model_base: Type[CallableModel]) -> Tuple[type, ...]:
    if not isinstance(model_base, type) or not issubclass(model_base, CallableModel):
        raise TypeError(f"model_base must be a CallableModel subclass, got {model_base!r}")

    if issubclass(model_base, _GeneratedFlowModelBase):
        return (model_base,)
    if model_base is CallableModel:
        return (_GeneratedFlowModelBase,)
    return (_GeneratedFlowModelBase, model_base)


def _analyze_flow_context_transform(fn: _AnyCallable, sig: inspect.Signature) -> _FlowModelConfig:
    parameters, context_input_types = _analyze_flow_function(fn, sig)
    return _FlowModelConfig(
        func=fn,
        return_annotation=sig.return_annotation,
        context_type=FlowContext,
        result_type=GenericResult,
        auto_wrap_result=False,
        auto_unwrap=False,
        parameters=parameters,
        context_input_types=context_input_types,
        context_required_names=tuple(param.name for param in parameters if param.is_contextual and not param.has_function_default),
        path=PyObjectPath(f"{getattr(fn, '__module__', __name__)}.{_callable_name(fn)}"),
    )


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
        config = _analyze_flow_context_transform(fn, sig)

        @wraps(fn)
        def factory(**kwargs) -> ContextTransform:
            return ContextTransform(path=config.path, bound_args=_validate_context_transform_factory_kwargs(config, kwargs))

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
        config = _analyze_flow_model(fn, sig, context_type=context_type, auto_unwrap=auto_unwrap)

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
        register_ccflow_import_path(GeneratedModel)
        GeneratedModel.model_rebuild()

        @wraps(fn)
        def factory(**kwargs) -> _GeneratedFlowModelBase:
            _validate_factory_kwargs(config, kwargs)
            return GeneratedModel(**kwargs)

        cast(Any, factory)._generated_model = GeneratedModel
        factory.__doc__ = fn.__doc__
        return factory

    if func is not None:
        return decorator(func)
    return decorator
