"""Flow.model decorator implementation built around ``FromContext``."""

import hashlib
import inspect
import logging
import marshal
from dataclasses import dataclass
from functools import lru_cache, wraps
from types import UnionType
from typing import Annotated, Any, Callable, ClassVar, Dict, List, Optional, Tuple, Type, Union, cast, get_args, get_origin, get_type_hints

from pydantic import Field, PrivateAttr, TypeAdapter, ValidationError, model_serializer, model_validator
from pydantic.errors import PydanticUndefinedAnnotation

from .base import BaseModel, ContextBase, ResultBase
from .callable import CallableModel, Flow, GraphDepList, WrapperModel
from .context import FlowContext
from .local_persistence import register_ccflow_import_path
from .result import GenericResult

__all__ = ("FlowAPI", "BoundModel", "FromContext", "Lazy")

_AnyCallable = Callable[..., Any]
log = logging.getLogger(__name__)


class _UnsetFlowInput:
    def __repr__(self) -> str:
        return "<unset>"


_UNSET_FLOW_INPUT = _UnsetFlowInput()
_UNSET = object()
_REMOVED_CONTEXT_ARGS = object()
_UNION_ORIGINS = (Union, UnionType)


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
        raise TypeError("Lazy(model)(...) has been removed. Use model.flow.with_inputs(...) for contextual rewrites.")

    def __class_getitem__(cls, item):
        return Annotated[item, _LazyMarker()]


@dataclass(frozen=True)
class _ParsedAnnotation:
    base: Any
    is_lazy: bool
    is_from_context: bool


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
    context_type: Type[ContextBase]
    result_type: Type[ResultBase]
    auto_wrap_result: bool
    auto_unwrap: bool
    explicit_context_param: Optional[str]
    parameters: Tuple[_FlowModelParam, ...]
    context_input_types: Dict[str, Any]
    context_required_names: Tuple[str, ...]
    declared_context_type: Optional[Type[ContextBase]] = None

    @property
    def regular_params(self) -> Tuple[_FlowModelParam, ...]:
        return tuple(param for param in self.parameters if not param.is_contextual)

    @property
    def contextual_params(self) -> Tuple[_FlowModelParam, ...]:
        return tuple(param for param in self.parameters if param.is_contextual)

    @property
    def regular_param_names(self) -> Tuple[str, ...]:
        return tuple(param.name for param in self.regular_params)

    @property
    def contextual_param_names(self) -> Tuple[str, ...]:
        return tuple(param.name for param in self.contextual_params)

    @property
    def uses_explicit_context(self) -> bool:
        return self.explicit_context_param is not None

    def param(self, name: str) -> _FlowModelParam:
        for param in self.parameters:
            if param.name == name:
                return param
        raise KeyError(name)


def _callable_name(func: _AnyCallable) -> str:
    return getattr(func, "__name__", type(func).__name__)


def _callable_module(func: _AnyCallable) -> str:
    return getattr(func, "__module__", __name__)


def _context_values(context: ContextBase) -> Dict[str, Any]:
    return dict(context)


def _transform_repr(transform: Any) -> str:
    if callable(transform):
        name = _callable_name(transform)
        if name.startswith("<") and name.endswith(">"):
            return name
        return f"<{name}>"
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


@lru_cache(maxsize=None)
def _type_adapter(annotation: Any) -> TypeAdapter:
    return TypeAdapter(annotation)


def _can_validate_type(annotation: Any) -> bool:
    try:
        _type_adapter(annotation)
    except (PydanticUndefinedAnnotation, TypeError, ValueError):
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
    except Exception as exc:
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


def _callable_closure_repr(transform: Any) -> str:
    closure = getattr(transform, "__closure__", None)
    if not closure:
        return ""
    pieces = []
    for cell in closure:
        try:
            pieces.append(repr(cell.cell_contents))
        except Exception:
            pieces.append("<unreprable>")
    return "|".join(pieces)


def _callable_fingerprint(transform: Any) -> str:
    module = getattr(transform, "__module__", type(transform).__module__)
    qualname = getattr(transform, "__qualname__", type(transform).__qualname__)
    code = getattr(transform, "__code__", None)
    if code is None:
        return f"callable:{module}:{qualname}:{repr(transform)}"

    payload = "|".join(
        [
            module,
            qualname,
            code.co_filename,
            str(code.co_firstlineno),
            hashlib.sha256(marshal.dumps(code)).hexdigest(),
            repr(getattr(transform, "__defaults__", None)),
            _callable_closure_repr(transform),
        ]
    )
    return f"callable:{payload}"


def _fingerprint_transforms(transforms: Dict[str, Any]) -> Tuple[Tuple[str, str], ...]:
    items = []
    for name, transform in sorted(transforms.items()):
        if callable(transform):
            items.append((name, _callable_fingerprint(transform)))
        else:
            items.append((name, repr(transform)))
    return tuple(items)


def _generated_model_instance(stage: Any) -> Optional["_GeneratedFlowModelBase"]:
    model = stage.model if isinstance(stage, BoundModel) else stage
    if isinstance(model, _GeneratedFlowModelBase):
        return model
    return None


def _generated_model_class(stage: Any) -> Optional[type["_GeneratedFlowModelBase"]]:
    model = _generated_model_instance(stage)
    if model is not None:
        return type(model)

    generated_model = getattr(stage, "_generated_model", None)
    if isinstance(generated_model, type) and issubclass(generated_model, _GeneratedFlowModelBase):
        return generated_model
    return None


def _context_input_types_for_model(model: CallableModel) -> Optional[Dict[str, Any]]:
    generated = _generated_model_instance(model)
    if generated is not None:
        return dict(type(generated).__flow_model_config__.context_input_types)

    context_cls = _concrete_context_type(model.context_type)
    if context_cls is None or context_cls is FlowContext or not hasattr(context_cls, "model_fields"):
        return None
    return {name: info.annotation for name, info in context_cls.model_fields.items()}


def _context_required_names_for_model(model: CallableModel) -> Tuple[str, ...]:
    generated = _generated_model_instance(model)
    if generated is not None:
        return type(generated).__flow_model_config__.context_required_names

    context_cls = _concrete_context_type(model.context_type)
    if context_cls is None or not hasattr(context_cls, "model_fields"):
        return ()
    return tuple(name for name, info in context_cls.model_fields.items() if info.is_required())


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
    if param.is_lazy:
        if _is_model_dependency(value):
            return _make_lazy_thunk(value, context)
        return lambda v=value: v
    if _is_model_dependency(value):
        return _unwrap_model_result(value(context))
    return value


def _resolve_contextual_param_value(
    model: "_GeneratedFlowModelBase",
    param: _FlowModelParam,
    context_values: Dict[str, Any],
) -> Tuple[Any, bool]:
    if param.name in context_values:
        return context_values[param.name], True

    value = getattr(model, param.name, _UNSET_FLOW_INPUT)
    if not _is_unset_flow_input(value):
        return value, True

    if param.has_function_default:
        return param.function_default, True

    return _UNSET, False


def _resolved_contextual_inputs(model: "_GeneratedFlowModelBase", config: _FlowModelConfig, context: ContextBase) -> Dict[str, Any]:
    context_values = _context_values(context)
    resolved: Dict[str, Any] = {}
    missing_contextual = []

    for param in config.contextual_params:
        value, found = _resolve_contextual_param_value(model, param, context_values)
        if not found:
            missing_contextual.append(param.name)
            continue
        resolved[param.name] = value

    if missing_contextual:
        missing = ", ".join(sorted(missing_contextual))
        raise TypeError(
            f"Missing contextual input(s) for {_callable_name(config.func)}: {missing}. "
            "Supply them via the runtime context, compute(), with_inputs(), or construction-time contextual defaults."
        )

    if config.declared_context_type is not None:
        validated = config.declared_context_type.model_validate(resolved)
        return {param.name: getattr(validated, param.name) for param in config.contextual_params}

    return {
        param.name: _coerce_value(param.name, resolved[param.name], param.validation_annotation, "Context field")
        for param in config.contextual_params
    }


def _validate_with_inputs_transforms(model: CallableModel, transforms: Dict[str, Any]) -> Dict[str, Any]:
    context_input_types = _context_input_types_for_model(model)
    validated = dict(transforms)

    if context_input_types is not None:
        invalid = sorted(set(transforms) - set(context_input_types))
        if invalid:
            names = ", ".join(invalid)
            raise TypeError(f"with_inputs() only accepts contextual fields. Invalid field(s): {names}.")

        for name, transform in list(validated.items()):
            if callable(transform):
                continue
            validated[name] = _coerce_value(name, transform, context_input_types[name], "with_inputs()")

    return validated


def _build_generated_compute_context(model: "_GeneratedFlowModelBase", context: Any, kwargs: Dict[str, Any]) -> ContextBase:
    config = type(model).__flow_model_config__

    if context is not _UNSET and kwargs:
        raise TypeError("compute() accepts either one context object or contextual keyword arguments, but not both.")

    if config.uses_explicit_context:
        if context is _UNSET:
            return config.context_type.model_validate(kwargs)
        return context if isinstance(context, ContextBase) else config.context_type.model_validate(context)

    if context is not _UNSET:
        if isinstance(context, FlowContext):
            return context
        if isinstance(context, ContextBase):
            return FlowContext(**_context_values(context))
        return FlowContext.model_validate(context)

    invalid = sorted(set(kwargs) - set(config.context_input_types))
    if invalid:
        names = ", ".join(invalid)
        raise TypeError(f"compute() only accepts contextual inputs. Bind regular parameter(s) separately: {names}.")

    coerced = {}
    for param in config.contextual_params:
        if param.name not in kwargs:
            continue
        coerced[param.name] = _coerce_value(param.name, kwargs[param.name], param.validation_annotation, "compute() input")
    return FlowContext(**coerced)


class FlowAPI:
    """API namespace for contextual execution and rewrites."""

    def __init__(self, model: CallableModel):
        self._model = model

    @property
    def _compute_target(self) -> CallableModel:
        return self._model

    def compute(self, context: Any = _UNSET, /, **kwargs) -> Any:
        target = self._compute_target
        generated = _generated_model_instance(target)
        if generated is not None:
            built_context = _build_generated_compute_context(generated, context, kwargs)
            return _maybe_auto_unwrap_external_result(target, target(built_context))

        if context is not _UNSET and kwargs:
            raise TypeError("compute() accepts either one context object or contextual keyword arguments, but not both.")
        if context is _UNSET:
            built_context = target.context_type.model_validate(kwargs)
        else:
            built_context = context if isinstance(context, ContextBase) else target.context_type.model_validate(context)
        return _maybe_auto_unwrap_external_result(target, target(built_context))

    @property
    def context_inputs(self) -> Dict[str, Any]:
        context_input_types = _context_input_types_for_model(self._model)
        return dict(context_input_types or {})

    @property
    def unbound_inputs(self) -> Dict[str, Any]:
        generated = _generated_model_instance(self._model)
        if generated is not None:
            config = type(generated).__flow_model_config__
            if config.uses_explicit_context:
                return {name: config.context_input_types[name] for name in config.context_required_names}
            result = {}
            for param in config.contextual_params:
                if not _is_unset_flow_input(getattr(generated, param.name, _UNSET_FLOW_INPUT)):
                    continue
                if param.has_function_default:
                    continue
                result[param.name] = config.context_input_types[param.name]
            return result

        required_names = _context_required_names_for_model(self._model)
        context_input_types = _context_input_types_for_model(self._model) or {}
        return {name: context_input_types[name] for name in required_names}

    @property
    def bound_inputs(self) -> Dict[str, Any]:
        generated = _generated_model_instance(self._model)
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
            return result

        result: Dict[str, Any] = {}
        model_fields = getattr(self._model.__class__, "model_fields", {})
        for name in model_fields:
            if name == "meta":
                continue
            result[name] = getattr(self._model, name)
        return result

    def with_inputs(self, **transforms) -> "BoundModel":
        validated = _validate_with_inputs_transforms(self._model, transforms)
        return BoundModel(model=self._model, input_transforms=validated)


class BoundModel(WrapperModel):
    """A model with contextual input transforms applied locally."""

    _input_transforms: Dict[str, Any] = PrivateAttr(default_factory=dict)
    serialized_transforms: Dict[str, Any] = Field(default_factory=dict, alias="_static_transforms", repr=False, exclude=True)

    @model_validator(mode="before")
    @classmethod
    def _strip_runtime_serializer_fields(cls, values):
        if isinstance(values, dict):
            cleaned = dict(values)
            cleaned.pop("_input_transforms_fingerprint", None)
            return cleaned
        return values

    def __init__(self, *, model: CallableModel, input_transforms: Optional[Dict[str, Any]] = None, **kwargs):
        if input_transforms is not None:
            static_transforms = {name: value for name, value in input_transforms.items() if not callable(value)}
            kwargs["_static_transforms"] = static_transforms
        super().__init__(model=model, **kwargs)
        if input_transforms is not None:
            self._input_transforms = dict(input_transforms)
        else:
            self._input_transforms = dict(self.serialized_transforms)

    def model_post_init(self, __context):
        if not self._input_transforms:
            self._input_transforms = dict(self.serialized_transforms)

    def _transform_context(self, context: ContextBase) -> ContextBase:
        ctx_dict = _context_values(context)
        context_input_types = _context_input_types_for_model(self.model)

        for name, transform in self._input_transforms.items():
            value = transform(context) if callable(transform) else transform
            if context_input_types is not None and name in context_input_types:
                value = _coerce_value(name, value, context_input_types[name], "with_inputs()")
            ctx_dict[name] = value

        generated = _generated_model_instance(self.model)
        if generated is not None and not type(generated).__flow_model_config__.uses_explicit_context:
            return FlowContext(**ctx_dict)

        context_type = _concrete_context_type(self.model.context_type)
        if context_type is not None and context_type is not FlowContext:
            return context_type.model_validate(ctx_dict)
        return FlowContext(**ctx_dict)

    @Flow.call
    def __call__(self, context: ContextBase) -> ResultBase:
        return self.model(self._transform_context(context))

    @Flow.deps
    def __deps__(self, context: ContextBase) -> GraphDepList:
        return [(self.model, [self._transform_context(context)])]

    @model_serializer(mode="wrap")
    def _serialize_with_transforms(self, handler):
        data = handler(self)
        if self.serialized_transforms:
            data["_static_transforms"] = dict(self.serialized_transforms)
        data["_input_transforms_fingerprint"] = _fingerprint_transforms(self._input_transforms)
        return data

    def __repr__(self) -> str:
        transforms = ", ".join(f"{name}={_transform_repr(transform)}" for name, transform in self._input_transforms.items())
        return f"{self.model!r}.flow.with_inputs({transforms})"

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

    def with_inputs(self, **transforms) -> BoundModel:
        validated = _validate_with_inputs_transforms(self._bound.model, transforms)
        merged = {**self._bound._input_transforms, **validated}
        return BoundModel(model=self._bound.model, input_transforms=merged)


class _GeneratedFlowModelBase(CallableModel):
    __flow_model_config__: ClassVar[_FlowModelConfig]

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
                        "Bind a literal contextual default or supply it via compute()/with_inputs()."
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
            fn_kwargs[param.name] = _resolve_regular_param_value(self, param, context)

        if config.uses_explicit_context:
            fn_kwargs[cast(str, config.explicit_context_param)] = context
        else:
            fn_kwargs.update(_resolved_contextual_inputs(self, config, context))

        raw_result = config.func(**fn_kwargs)
        if config.auto_wrap_result:
            return GenericResult(value=raw_result)
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
            if isinstance(value, BoundModel):
                deps.append((value.model, [value._transform_context(context)]))
            elif isinstance(value, CallableModel):
                deps.append((value, [context]))
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
    if func_annotation is context_annotation:
        return True
    if isinstance(func_annotation, type) and isinstance(context_annotation, type):
        return issubclass(func_annotation, context_annotation) or issubclass(context_annotation, func_annotation)
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


def _analyze_flow_model(
    fn: _AnyCallable,
    sig: inspect.Signature,
    resolved_hints: Dict[str, Any],
    *,
    context_type: Optional[Type[ContextBase]],
    auto_unwrap: bool,
) -> _FlowModelConfig:
    params = sig.parameters

    explicit_context_param = None
    if "context" in params:
        explicit_context_param = "context"
    elif "_" in params:
        explicit_context_param = "_"

    analyzed_params: List[_FlowModelParam] = []
    explicit_context_type = None

    if explicit_context_param is not None:
        explicit_context = params[explicit_context_param]
        if explicit_context.kind is inspect.Parameter.POSITIONAL_ONLY:
            raise TypeError(f"Function {_callable_name(fn)} does not support positional-only parameter '{explicit_context_param}'.")
        if explicit_context.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            raise TypeError(
                f"Function {_callable_name(fn)} does not support {explicit_context.kind.description} parameter '{explicit_context_param}'."
            )
        context_annotation = resolved_hints.get(explicit_context_param, params[explicit_context_param].annotation)
        explicit_context_type = _concrete_context_type(context_annotation)
        if explicit_context_type is None:
            raise TypeError(f"Function {_callable_name(fn)}: '{explicit_context_param}' must be annotated with a ContextBase subclass.")
        if context_type is not None:
            raise TypeError("context_type=... is inferred from the explicit context parameter; remove the keyword argument.")

    for name, param in params.items():
        if name == "self" or name == explicit_context_param:
            continue

        if param.kind is inspect.Parameter.POSITIONAL_ONLY:
            raise TypeError(f"Function {_callable_name(fn)} does not support positional-only parameter '{name}'.")
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            raise TypeError(f"Function {_callable_name(fn)} does not support {param.kind.description} parameter '{name}'.")

        annotation = resolved_hints.get(name, param.annotation)
        if annotation is inspect.Parameter.empty:
            raise TypeError(f"Parameter '{name}' must have a type annotation")

        parsed = _parse_annotation(annotation)
        if parsed.is_lazy and parsed.is_from_context:
            raise TypeError(f"Parameter '{name}' cannot combine Lazy[...] and FromContext[...].")

        has_function_default = param.default is not inspect.Parameter.empty
        function_default = param.default if has_function_default else _UNSET
        if parsed.is_from_context and has_function_default and _is_model_dependency(function_default):
            raise TypeError(f"Parameter '{name}' is marked FromContext[...] and cannot default to a CallableModel.")

        analyzed_params.append(
            _FlowModelParam(
                name=name,
                annotation=parsed.base,
                kind="contextual" if parsed.is_from_context else "regular",
                is_lazy=parsed.is_lazy,
                has_function_default=has_function_default,
                function_default=function_default,
            )
        )

    contextual_params = tuple(param for param in analyzed_params if param.is_contextual)
    if explicit_context_param is not None and contextual_params:
        raise TypeError("Functions using an explicit context parameter cannot also declare FromContext[...] parameters.")

    declared_context_type = None
    if explicit_context_type is not None:
        call_context_type = explicit_context_type
        context_input_types = {name: info.annotation for name, info in explicit_context_type.model_fields.items()}
        context_required_names = tuple(name for name, info in explicit_context_type.model_fields.items() if info.is_required())
    else:
        if context_type is not None and not contextual_params:
            raise TypeError("context_type=... requires FromContext[...] parameters or an explicit context parameter.")
        if context_type is not None:
            declared_context_type = _validate_declared_context_type(context_type, contextual_params)

        call_context_type = FlowContext
        context_input_types = {param.name: param.annotation for param in contextual_params}
        context_required_names = tuple(param.name for param in contextual_params if not param.has_function_default)

    if declared_context_type is not None:
        updated_params = []
        context_fields = declared_context_type.model_fields
        for param in analyzed_params:
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
        analyzed_params = updated_params

    return_type = resolved_hints.get("return", sig.return_annotation)
    if return_type is inspect.Signature.empty:
        raise TypeError(f"Function {_callable_name(fn)} must have a return type annotation")

    return_origin = get_origin(return_type) or return_type
    auto_wrap_result = not (isinstance(return_origin, type) and issubclass(return_origin, ResultBase))
    result_type = GenericResult if auto_wrap_result else return_type

    return _FlowModelConfig(
        func=fn,
        context_type=call_context_type,
        result_type=result_type,
        auto_wrap_result=auto_wrap_result,
        auto_unwrap=auto_unwrap,
        explicit_context_param=explicit_context_param,
        parameters=tuple(analyzed_params),
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
        sig = inspect.signature(fn)

        try:
            resolved_hints = get_type_hints(fn, include_extras=True)
        except (AttributeError, NameError, TypeError):
            resolved_hints = {}

        config = _analyze_flow_model(fn, sig, resolved_hints, context_type=context_type, auto_unwrap=auto_unwrap)

        annotations: Dict[str, Any] = {}
        namespace: Dict[str, Any] = {
            "__module__": _callable_module(fn),
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
