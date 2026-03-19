"""Flow.model decorator implementation.

This module provides the Flow.model decorator that generates CallableModel classes
from plain Python functions, reducing boilerplate while maintaining full compatibility
with existing ccflow infrastructure.

Key design: Uses TypedDict + TypeAdapter for context schema validation instead of
generating dynamic ContextBase subclasses. This avoids class registration overhead
and enables clean pickling for distributed computing (e.g., Ray).
"""

import inspect
import logging
import threading
from functools import wraps
from typing import Annotated, Any, Callable, ClassVar, Dict, List, Optional, Tuple, Type, Union, cast, get_args, get_origin, get_type_hints

from pydantic import Field, PrivateAttr, TypeAdapter, model_serializer, model_validator
from typing_extensions import NotRequired, TypedDict

from .base import ContextBase, ResultBase
from .callable import CallableModel, Flow, GraphDepList, WrapperModel
from .context import FlowContext
from .local_persistence import register_ccflow_import_path
from .result import GenericResult

__all__ = ("FlowAPI", "BoundModel", "Lazy")

_AnyCallable = Callable[..., Any]


class _DeferredInput:
    """Sentinel for dynamic @Flow.model inputs left for runtime context."""

    def __repr__(self) -> str:
        return "<deferred>"


_DEFERRED_INPUT = _DeferredInput()


def _callable_name(func: _AnyCallable) -> str:
    return getattr(func, "__name__", type(func).__name__)


def _callable_module(func: _AnyCallable) -> str:
    return getattr(func, "__module__", __name__)


class _LazyMarker:
    """Sentinel that marks a parameter as lazily evaluated via Lazy[T]."""

    pass


def _extract_lazy(annotation) -> Tuple[Any, bool]:
    """Check if annotation is Lazy[T]. Returns (base_type, is_lazy).

    Handles nested Annotated types, so we need to check the outermost
    Annotated layer for _LazyMarker.
    """
    if get_origin(annotation) is Annotated:
        args = get_args(annotation)
        for metadata in args[1:]:
            if isinstance(metadata, _LazyMarker):
                return args[0], True
    return annotation, False


def _make_lazy_thunk(model, context):
    """Create a zero-arg callable that evaluates model(context) on demand.

    The thunk caches its result so repeated calls don't re-evaluate.
    """
    _cache = {}

    def thunk():
        if "result" not in _cache:
            result = model(context)
            if isinstance(result, GenericResult):
                result = result.value
            _cache["result"] = result
        return _cache["result"]

    return thunk


log = logging.getLogger(__name__)


def _context_values(context: ContextBase) -> Dict[str, Any]:
    """Return a plain mapping of all context values.

    `dict(context)` uses pydantic's public iteration behavior, which includes
    both declared fields and any allowed extra fields.
    """

    return dict(context)


def _transform_repr(transform: Any) -> str:
    """Render an input transform without noisy object addresses."""

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
    return set(getattr(model, "_bound_fields", set()))


def _has_deferred_input(value: Any) -> bool:
    return isinstance(value, _DeferredInput)


def _deferred_input_factory() -> _DeferredInput:
    return _DEFERRED_INPUT


def _effective_bound_field_names(model: Any) -> set[str]:
    fields = _bound_field_names(model)
    defaults = getattr(model.__class__, "__flow_model_default_param_names__", set())
    return fields | set(defaults)


def _runtime_input_names(model: Any) -> set[str]:
    all_param_names = set(getattr(model.__class__, "__flow_model_all_param_types__", {}))
    if not all_param_names:
        return set()
    return all_param_names - _effective_bound_field_names(model)


def _resolve_registry_candidate(value: str) -> Any:
    from .base import BaseModel as _BM

    try:
        candidate = _BM.model_validate(value)
    except Exception:
        return None
    return candidate if isinstance(candidate, _BM) else None


def _registry_candidate_allowed(expected_type: Type, candidate: Any) -> bool:
    if _is_model_dependency(candidate):
        return True
    try:
        TypeAdapter(expected_type).validate_python(candidate)
    except Exception:
        return False
    return True


def _type_accepts_str(annotation) -> bool:
    """Return True when ``str`` is a valid type for *annotation*.

    Handles ``str``, ``Union[str, ...]``, ``Optional[str]``, and
    ``Annotated[str, ...]``.
    """
    if annotation is str:
        return True
    origin = get_origin(annotation)
    if origin is Annotated:
        return _type_accepts_str(get_args(annotation)[0])
    if origin is Union:
        return any(_type_accepts_str(arg) for arg in get_args(annotation) if arg is not type(None))
    return False


def _build_typed_dict_adapter(name: str, schema: Dict[str, Type], *, total: bool = True) -> TypeAdapter:
    """Build a TypeAdapter for a runtime TypedDict schema."""

    if not schema:
        return TypeAdapter(dict)
    return TypeAdapter(TypedDict(name, schema, total=total))


def _concrete_context_type(context_type: Any) -> Optional[Type[ContextBase]]:
    """Extract a concrete ContextBase subclass from a context annotation."""

    if isinstance(context_type, type) and issubclass(context_type, ContextBase):
        return context_type

    if get_origin(context_type) in (Optional, Union):
        for arg in get_args(context_type):
            if arg is type(None):
                continue
            if isinstance(arg, type) and issubclass(arg, ContextBase):
                return arg

    return None


def _build_config_validators(all_param_types: Dict[str, Type]) -> Tuple[Dict[str, Type], Dict[str, TypeAdapter]]:
    """Precompute validators for constructor fields."""

    validatable_types: Dict[str, Type] = {}
    for name, typ in all_param_types.items():
        try:
            TypeAdapter(typ)
            validatable_types[name] = typ
        except Exception:
            pass

    validators = {name: TypeAdapter(typ) for name, typ in validatable_types.items()}
    return validatable_types, validators


def _coerce_context_value(name: str, value: Any, validators: Dict[str, TypeAdapter], validatable_types: Dict[str, Type]) -> Any:
    """Validate/coerce a single context-sourced value. Returns coerced value or raises TypeError."""
    if name not in validators:
        return value
    try:
        return validators[name].validate_python(value)
    except Exception:
        expected = validatable_types.get(name, "unknown")
        raise TypeError(f"Context field '{name}': expected {expected}, got {type(value).__name__} ({value!r})")


def _validate_config_kwargs(kwargs: Dict[str, Any], validatable_types: Dict[str, Type], validators: Dict[str, TypeAdapter]) -> None:
    """Validate plain config inputs while still allowing dependency objects."""

    if not validators:
        return

    from .base import ModelRegistry as _MR

    for field_name, validator in validators.items():
        if field_name not in kwargs:
            continue
        value = kwargs[field_name]
        if value is None or _is_model_dependency(value):
            continue
        if isinstance(value, str) and value in _MR.root():
            expected_type = validatable_types[field_name]
            if _type_accepts_str(expected_type):
                continue
            candidate = _resolve_registry_candidate(value)
            if candidate is not None and _registry_candidate_allowed(expected_type, candidate):
                continue
        try:
            validator.validate_python(value)
        except Exception:
            expected_type = validatable_types[field_name]
            raise TypeError(f"Field '{field_name}': expected {expected_type}, got {type(value).__name__} ({value!r})")


def _generated_model_instance(stage: Any) -> Optional["_GeneratedFlowModelBase"]:
    if isinstance(stage, BoundModel):
        model = stage.model
    else:
        model = stage
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


def _describe_pipe_stage(stage: Any) -> str:
    if isinstance(stage, BoundModel):
        return repr(stage)
    if isinstance(stage, _GeneratedFlowModelBase):
        return repr(stage)
    if callable(stage):
        return _callable_name(stage)
    return repr(stage)


def _generated_model_explicit_kwargs(model: "_GeneratedFlowModelBase") -> Dict[str, Any]:
    return cast(Dict[str, Any], model.model_dump(mode="python", exclude_unset=True))


def _infer_pipe_param(
    stage_name: str,
    param_names: List[str],
    default_param_names: set[str],
    occupied_names: set[str],
) -> str:
    required_candidates = [name for name in param_names if name not in occupied_names and name not in default_param_names]
    if len(required_candidates) == 1:
        return required_candidates[0]
    if len(required_candidates) > 1:
        candidates = ", ".join(required_candidates)
        raise TypeError(
            f"pipe() could not infer a target parameter for {stage_name}; unbound candidates are: {candidates}. Pass param='...' explicitly."
        )

    fallback_candidates = [name for name in param_names if name not in occupied_names]
    if len(fallback_candidates) == 1:
        return fallback_candidates[0]
    if len(fallback_candidates) > 1:
        candidates = ", ".join(fallback_candidates)
        raise TypeError(
            f"pipe() could not infer a target parameter for {stage_name}; unbound candidates are: {candidates}. Pass param='...' explicitly."
        )

    raise TypeError(f"pipe() could not find an available target parameter for {stage_name}.")


def _resolve_pipe_param(source: Any, stage: Any, param: Optional[str], bindings: Dict[str, Any]) -> Tuple[str, type["_GeneratedFlowModelBase"]]:
    del source  # Source only matters when binding, not during target resolution.

    generated_model_cls = _generated_model_class(stage)
    if generated_model_cls is None:
        raise TypeError("pipe() only supports downstream stages created by @Flow.model or bound versions of those stages.")

    stage_name = _describe_pipe_stage(stage)
    all_param_types = getattr(generated_model_cls, "__flow_model_all_param_types__", {})
    if not all_param_types:
        raise TypeError(f"pipe() could not determine bindable parameters for {stage_name}.")

    param_names = list(all_param_types.keys())
    default_param_names = set(getattr(generated_model_cls, "__flow_model_default_param_names__", set()))

    generated_model = _generated_model_instance(stage)
    occupied_names = set(bindings)
    if generated_model is not None:
        occupied_names |= _bound_field_names(generated_model)
    if isinstance(stage, BoundModel):
        occupied_names |= set(stage._input_transforms)

    if param is not None:
        if param not in all_param_types:
            valid = ", ".join(param_names)
            raise TypeError(f"pipe() target parameter '{param}' is not valid for {stage_name}. Available parameters: {valid}.")
        if param in occupied_names:
            raise TypeError(f"pipe() target parameter '{param}' is already bound for {stage_name}.")
        return param, generated_model_cls

    return _infer_pipe_param(stage_name, param_names, default_param_names, occupied_names), generated_model_cls


def pipe_model(source: Any, stage: Any, /, *, param: Optional[str] = None, **bindings: Any) -> Any:
    """Wire ``source`` into a downstream generated ``@Flow.model`` stage."""

    if not _is_model_dependency(source):
        raise TypeError(f"pipe() source must be a CallableModel, got {type(source).__name__}.")

    target_param, generated_model_cls = _resolve_pipe_param(source, stage, param, bindings)
    build_kwargs = dict(bindings)
    build_kwargs[target_param] = source

    if isinstance(stage, BoundModel):
        generated_model = _generated_model_instance(stage)
        if generated_model is None:
            raise TypeError("pipe() only supports downstream BoundModel stages created from @Flow.model.")
        explicit_kwargs = _generated_model_explicit_kwargs(generated_model)
        explicit_kwargs.update(build_kwargs)
        rebound_model = generated_model_cls(**explicit_kwargs)
        return BoundModel(model=rebound_model, input_transforms=dict(stage._input_transforms))

    generated_model = _generated_model_instance(stage)
    if generated_model is not None:
        explicit_kwargs = _generated_model_explicit_kwargs(generated_model)
        explicit_kwargs.update(build_kwargs)
        return generated_model_cls(**explicit_kwargs)

    return stage(**build_kwargs)


class FlowAPI:
    """API namespace for deferred computation operations.

    Provides methods for executing models and transforming contexts.
    Accessed via model.flow property.
    """

    def __init__(self, model: CallableModel):
        self._model = model

    def _build_context(self, kwargs: Dict[str, Any]) -> ContextBase:
        """Construct a runtime context for either generated or hand-written models."""
        get_validator = getattr(self._model, "_get_context_validator", None)
        if get_validator is not None:
            validator = get_validator()
            validated = validator.validate_python(kwargs)
            if isinstance(validated, ContextBase):
                return validated
            return FlowContext(**validated)

        validator = TypeAdapter(self._model.context_type)
        return validator.validate_python(kwargs)

    def compute(self, **kwargs) -> Any:
        """Execute the model with the provided context arguments.

        Validates kwargs against the model's context schema using TypeAdapter,
        then wraps in FlowContext and calls the model.

        Args:
            **kwargs: Context arguments (e.g., start_date, end_date)

        Returns:
            The model's result, using the same return contract as ``model(context)``.
        """
        ctx = self._build_context(kwargs)
        return self._model(ctx)

    @property
    def unbound_inputs(self) -> Dict[str, Type]:
        """Return the context schema (field name -> type).

        In deferred mode, this is everything that must still come from runtime context.
        """
        all_param_types = getattr(self._model.__class__, "__flow_model_all_param_types__", {})
        model_cls = self._model.__class__

        # If explicit context_args was provided, use _context_schema minus
        # fields that have function defaults (they aren't truly required).
        explicit_args = getattr(model_cls, "__flow_model_explicit_context_args__", None)
        if explicit_args is not None:
            context_schema = getattr(model_cls, "_context_schema", None)
            if context_schema is None:
                return {}
            ctx_arg_defaults = getattr(model_cls, "__flow_model_context_arg_defaults__", {})
            return {name: typ for name, typ in context_schema.items() if name not in ctx_arg_defaults}

        # Dynamic @Flow.model: unbound = params with no explicit value and no declared default
        if all_param_types:
            runtime_inputs = _runtime_input_names(self._model)
            return {name: typ for name, typ in all_param_types.items() if name in runtime_inputs}

        # Generic CallableModel / Mode 1: runtime inputs are the required
        # context fields (fields with defaults are not required).
        context_cls = _concrete_context_type(self._model.context_type)
        if context_cls is None or not hasattr(context_cls, "model_fields"):
            return {}
        return {name: info.annotation for name, info in context_cls.model_fields.items() if info.is_required()}

    @property
    def bound_inputs(self) -> Dict[str, Any]:
        """Return the effective config values for this model."""
        result: Dict[str, Any] = {}
        flow_param_types = getattr(self._model.__class__, "__flow_model_all_param_types__", {})
        if flow_param_types:
            for name in flow_param_types:
                value = getattr(self._model, name, _DEFERRED_INPUT)
                if _has_deferred_input(value):
                    continue
                result[name] = value
            return result

        # Generic CallableModel: configured model fields are the bound inputs.
        model_fields = getattr(self._model.__class__, "model_fields", {})
        for name in model_fields:
            if name == "meta":
                continue
            result[name] = getattr(self._model, name)
        return result

    def with_inputs(self, **transforms) -> "BoundModel":
        """Create a version of this model with transformed context inputs.

        Args:
            **transforms: Mapping of field name to either:
                - A callable (ctx) -> value for dynamic transforms
                - A static value to bind

        Returns:
            A BoundModel that applies the transforms before calling.
        """
        return BoundModel(model=self._model, input_transforms=transforms)


_bound_model_restore = threading.local()


def _fingerprint_transforms(transforms: Dict[str, Any]) -> Dict[str, str]:
    """Create a stable, hashable fingerprint of input transforms for cache key differentiation.

    Callable transforms are identified by their id() (unique per object), which is
    stable within a process lifetime. Static values are repr'd directly.
    """
    result = {}
    for name, transform in sorted(transforms.items()):
        if callable(transform):
            result[name] = f"callable:{id(transform)}"
        else:
            result[name] = repr(transform)
    return result


class BoundModel(WrapperModel):
    """A model with context transforms applied.

    Created by model.flow.with_inputs(). Applies transforms to context
    before delegating to the underlying model.

    Context propagation across dependencies:
        Each BoundModel transforms the context locally — only for the model it
        wraps. When used as a dependency inside another model, the FlowContext
        flows through the chain unchanged until it reaches this BoundModel,
        which intercepts it, applies its transforms, and passes the modified
        context to the wrapped model. Upstream models never see the transform.

    Chaining with_inputs:
        Calling ``bound.flow.with_inputs(...)`` merges the new transforms with
        the existing ones (new overrides old for the same key). All transforms
        are applied to the incoming context in one pass — they don't compose
        sequentially (each transform sees the original context, not the output
        of a previous transform).
    """

    _input_transforms: Dict[str, Any] = PrivateAttr(default_factory=dict)

    @model_validator(mode="wrap")
    @classmethod
    def _restore_serialized_transforms(cls, values, handler):
        """Strip serialization-injected keys, restore static transforms, guarantee cleanup.

        Uses thread-local storage to pass static transforms to __init__ because
        pydantic rejects unknown keys in the input dict.  The wrap validator's
        try/finally ensures the thread-local is always cleaned up, even if
        validation fails before __init__ runs.
        """
        if isinstance(values, dict):
            values = dict(values)  # Don't mutate the caller's dict
            values.pop("_input_transforms_token", None)
            static = values.pop("_static_transforms", None)
        else:
            static = None

        if static is not None:
            _bound_model_restore.pending = static
        try:
            return handler(values)
        except Exception:
            _bound_model_restore.pending = None
            raise

    def __init__(self, *, model: CallableModel, input_transforms: Optional[Dict[str, Any]] = None, **kwargs):
        super().__init__(model=model, **kwargs)
        restore = getattr(_bound_model_restore, "pending", None)
        if restore is not None:
            _bound_model_restore.pending = None
        if input_transforms is not None:
            self._input_transforms = input_transforms
        elif restore is not None:
            self._input_transforms = restore
        else:
            self._input_transforms = {}

    def _transform_context(self, context: ContextBase) -> ContextBase:
        """Return this model's preferred context type with input transforms applied."""
        ctx_dict = _context_values(context)
        for name, transform in self._input_transforms.items():
            if callable(transform):
                ctx_dict[name] = transform(context)
            else:
                ctx_dict[name] = transform
        context_type = _concrete_context_type(self.model.context_type)
        if context_type is not None and context_type is not FlowContext:
            return context_type.model_validate(ctx_dict)
        return FlowContext(**ctx_dict)

    @Flow.call
    def __call__(self, context: ContextBase) -> ResultBase:
        """Call the model with transformed context."""
        return self.model(self._transform_context(context))

    @Flow.deps
    def __deps__(self, context: ContextBase) -> GraphDepList:
        """Declare the wrapped model as an upstream dependency with transformed context."""
        return [(self.model, [self._transform_context(context)])]

    @model_serializer(mode="wrap")
    def _serialize_with_transforms(self, handler):
        """Include transforms in serialization for cache keys and faithful roundtrips.

        Static (non-callable) transforms are serialized in _static_transforms for
        faithful restoration. A fingerprint token covers all transforms (including
        callables) for cache key differentiation.
        """
        data = handler(self)
        static = {k: v for k, v in self._input_transforms.items() if not callable(v)}
        if static:
            data["_static_transforms"] = static
        data["_input_transforms_token"] = _fingerprint_transforms(self._input_transforms)
        return data

    def pipe(self, stage: Any, /, *, param: Optional[str] = None, **bindings: Any) -> Any:
        """Wire this bound model into a downstream generated ``@Flow.model`` stage."""
        return pipe_model(self, stage, param=param, **bindings)

    def __repr__(self) -> str:
        transforms = ", ".join(f"{name}={_transform_repr(transform)}" for name, transform in self._input_transforms.items())
        return f"{self.model!r}.flow.with_inputs({transforms})"

    @property
    def flow(self) -> "FlowAPI":
        """Access the flow API."""
        return _BoundFlowAPI(self)


class _BoundFlowAPI(FlowAPI):
    """FlowAPI that delegates to a BoundModel, honoring transforms."""

    def __init__(self, bound_model: BoundModel):
        self._bound = bound_model
        super().__init__(bound_model.model)

    def compute(self, **kwargs) -> Any:
        ctx = self._build_context(kwargs)
        return self._bound(ctx)  # Call through BoundModel, not inner model

    def with_inputs(self, **transforms) -> "BoundModel":
        """Chain transforms: merge new transforms with existing ones.

        New transforms override existing ones for the same key.
        """
        merged = {**self._bound._input_transforms, **transforms}
        return BoundModel(model=self._bound.model, input_transforms=merged)


class _GeneratedFlowModelBase(CallableModel):
    """Shared behavior for models generated by ``@Flow.model``."""

    __flow_model_context_type__: ClassVar[Type[ContextBase]] = FlowContext
    __flow_model_return_type__: ClassVar[Type[ResultBase]] = GenericResult
    __flow_model_func__: ClassVar[_AnyCallable | None] = None
    __flow_model_use_context_args__: ClassVar[bool] = True
    __flow_model_explicit_context_args__: ClassVar[Optional[List[str]]] = None
    __flow_model_all_param_types__: ClassVar[Dict[str, Type]] = {}
    __flow_model_default_param_names__: ClassVar[set[str]] = set()
    __flow_model_context_arg_defaults__: ClassVar[Dict[str, Any]] = {}
    __flow_model_auto_wrap__: ClassVar[bool] = False
    __flow_model_validatable_types__: ClassVar[Dict[str, Type]] = {}
    __flow_model_config_validators__: ClassVar[Dict[str, TypeAdapter]] = {}
    _context_schema: ClassVar[Dict[str, Type]] = {}
    _context_td: ClassVar[Any | None] = None
    _cached_context_validator: ClassVar[TypeAdapter | None] = None

    @model_validator(mode="before")
    def _resolve_registry_refs(cls, values, info):
        if not isinstance(values, dict):
            return values

        param_types = getattr(cls, "__flow_model_all_param_types__", {})
        resolved = dict(values)
        for field_name, expected_type in param_types.items():
            if field_name not in resolved:
                continue
            value = resolved[field_name]
            if not isinstance(value, str):
                continue
            if _type_accepts_str(expected_type):
                continue
            candidate = _resolve_registry_candidate(value)
            if candidate is None:
                continue
            if _registry_candidate_allowed(expected_type, candidate):
                resolved[field_name] = candidate
        return resolved

    @model_validator(mode="after")
    def _validate_field_types(self):
        """Validate field values against their declared types.

        This catches type mismatches in the model_validate/deserialization path,
        where fields are typed as Any and pydantic won't reject wrong types.
        """
        cls = self.__class__
        config_validators = getattr(cls, "__flow_model_config_validators__", {})
        validatable_types = getattr(cls, "__flow_model_validatable_types__", {})
        if not config_validators:
            return self

        for field_name, validator in config_validators.items():
            value = getattr(self, field_name, _DEFERRED_INPUT)
            if _has_deferred_input(value) or value is None or _is_model_dependency(value):
                continue
            try:
                validator.validate_python(value)
            except Exception:
                expected_type = validatable_types[field_name]
                raise TypeError(f"Field '{field_name}': expected {expected_type}, got {type(value).__name__} ({value!r})")
        return self

    @property
    def context_type(self) -> Type[ContextBase]:
        return self.__class__.__flow_model_context_type__

    @property
    def result_type(self) -> Type[ResultBase]:
        return self.__class__.__flow_model_return_type__

    @property
    def flow(self) -> FlowAPI:
        return FlowAPI(self)

    def _get_context_validator(self) -> TypeAdapter:
        """Get or create the context validator for this generated model."""

        cls = self.__class__
        explicit_args = getattr(cls, "__flow_model_explicit_context_args__", None)

        if explicit_args is not None or not getattr(cls, "__flow_model_use_context_args__", True):
            if cls._cached_context_validator is None:
                use_ctx_args = getattr(cls, "__flow_model_use_context_args__", True)
                ctx_type = cls.__flow_model_context_type__
                if not use_ctx_args and isinstance(ctx_type, type) and issubclass(ctx_type, ContextBase) and ctx_type is not FlowContext:
                    # Mode 1 with concrete context type — use TypeAdapter(context_type)
                    # directly so defaults on the context type are respected.
                    cls._cached_context_validator = TypeAdapter(ctx_type)
                elif cls._context_td is not None:
                    cls._cached_context_validator = TypeAdapter(cls._context_td)
                elif cls._context_schema:
                    cls._cached_context_validator = _build_typed_dict_adapter(f"{cls.__name__}Inputs", cls._context_schema)
                else:
                    cls._cached_context_validator = TypeAdapter(cls.__flow_model_context_type__)
            return cls._cached_context_validator

        if not hasattr(self, "_instance_context_validator"):
            all_param_types = getattr(cls, "__flow_model_all_param_types__", {})
            runtime_inputs = _runtime_input_names(self)
            unbound_schema = {name: typ for name, typ in all_param_types.items() if name in runtime_inputs}
            object.__setattr__(
                self,
                "_instance_context_validator",
                _build_typed_dict_adapter(f"{cls.__name__}Inputs", unbound_schema, total=False),
            )
        return cast(TypeAdapter, getattr(self, "_instance_context_validator"))


class Lazy:
    """Deferred model execution with runtime context overrides.

    Has two distinct uses:

    1. **Type annotation** — ``Lazy[T]`` marks a parameter as lazily evaluated.
       The framework will NOT pre-evaluate the dependency; instead the function
       receives a zero-arg thunk that triggers evaluation on demand::

           @Flow.model
           def smart_training(
               data: PreparedData,
               fast_metrics: Metrics,
               slow_metrics: Lazy[Metrics],  # NOT eagerly evaluated
               threshold: float = 0.9,
           ) -> Metrics:
               if fast_metrics.r2 > threshold:
                   return fast_metrics
               return slow_metrics()  # Evaluated on demand

    2. **Runtime helper** — ``Lazy(model)(overrides)`` creates a callable that
       applies context overrides before calling the model.  Used with
       ``with_inputs()`` for deferred execution::

           lookback = Lazy(model)(start_date=lambda ctx: ctx.start_date - timedelta(days=7))
    """

    def __class_getitem__(cls, item):
        """Support Lazy[T] syntax as a type annotation marker.

        Returns Annotated[T, _LazyMarker()] so the framework can detect
        lazy parameters during signature analysis.
        """
        return Annotated[item, _LazyMarker()]

    def __init__(self, model: "CallableModel"):  # noqa: F821
        """Wrap a model for deferred execution.

        Args:
            model: The CallableModel to wrap
        """
        self._model = model

    def __call__(self, **overrides) -> Callable[[ContextBase], Any]:
        """Create a callable that applies overrides to context before execution.

        Args:
            **overrides: Context field overrides. Values can be:
                - Static values (applied directly)
                - Callables (ctx) -> value (called with context at runtime)

        Returns:
            A callable (context) -> result that applies overrides and calls the model
        """
        model = self._model

        def execute_with_overrides(context: ContextBase) -> Any:
            # Build context dict from incoming context
            ctx_dict = _context_values(context)

            # Apply overrides
            for name, value in overrides.items():
                if callable(value):
                    ctx_dict[name] = value(context)
                else:
                    ctx_dict[name] = value

            # Call model with modified context
            new_ctx = FlowContext(**ctx_dict)
            return model(new_ctx)

        return execute_with_overrides

    @property
    def model(self) -> "CallableModel":  # noqa: F821
        """Access the wrapped model."""
        return self._model


def _build_context_schema(
    context_args: List[str], func: _AnyCallable, sig: inspect.Signature, resolved_hints: Dict[str, Any]
) -> Tuple[Dict[str, Type], Any]:
    """Build context schema from context_args parameter names.

    Instead of creating a dynamic ContextBase subclass, this builds:
    - A schema dict mapping field names to types
    - A TypedDict for Pydantic TypeAdapter validation
    - Optionally, a matched existing ContextBase type for compatibility

    Args:
        context_args: List of parameter names that come from context
        func: The decorated function
        sig: The function signature

    Returns:
        Tuple of (schema_dict, TypedDict type)
    """
    # Build schema dict from parameter annotations
    schema = {}
    td_schema = {}
    for name in context_args:
        if name not in sig.parameters:
            raise ValueError(f"context_arg '{name}' not found in function parameters")
        param = sig.parameters[name]
        annotation = resolved_hints.get(name, param.annotation)
        if annotation is inspect.Parameter.empty:
            raise ValueError(f"context_arg '{name}' must have a type annotation")
        schema[name] = annotation
        # Use NotRequired in the TypedDict for params that have a default in the
        # function signature, so compute() doesn't require them.
        if param.default is not inspect.Parameter.empty:
            td_schema[name] = NotRequired[annotation]
        else:
            td_schema[name] = annotation

    # Create TypedDict for validation (not registered anywhere!)
    context_td = TypedDict(f"{_callable_name(func)}Inputs", td_schema)

    return schema, context_td


def _validate_context_type_override(
    context_type: Any,
    context_args: List[str],
    func_schema: Dict[str, Type],
    func_defaults: set[str] = frozenset(),
) -> Type[ContextBase]:
    """Validate an explicit ``context_type`` override for ``context_args`` mode."""

    if not isinstance(context_type, type) or not issubclass(context_type, ContextBase):
        raise TypeError(f"context_type must be a ContextBase subclass, got {context_type!r}")

    context_fields = getattr(context_type, "model_fields", {})
    missing = sorted(name for name in context_args if name not in context_fields)
    if missing:
        raise TypeError(f"context_type {context_type.__name__} must define fields for context_args: {', '.join(missing)}")

    required_extra_fields = sorted(
        name for name, info in context_fields.items() if name not in ContextBase.model_fields and name not in context_args and info.is_required()
    )
    if required_extra_fields:
        raise TypeError(f"context_type {context_type.__name__} has required fields not listed in context_args: {', '.join(required_extra_fields)}")

    # Warn when the function's annotation for a context_arg doesn't match the
    # context_type's field annotation.  A mismatch means the function declares
    # one type but will silently receive whatever Pydantic coerces to.
    for name in context_args:
        func_ann = func_schema.get(name)
        ctx_field = context_fields.get(name)
        if func_ann is None or ctx_field is None:
            continue
        ctx_ann = ctx_field.annotation
        if func_ann is ctx_ann:
            continue
        # Both are concrete types — check subclass relationship
        if isinstance(func_ann, type) and isinstance(ctx_ann, type):
            if not (issubclass(func_ann, ctx_ann) or issubclass(ctx_ann, func_ann)):
                raise TypeError(
                    f"context_arg '{name}': function annotates {func_ann.__name__} "
                    f"but context_type {context_type.__name__} declares {ctx_ann.__name__}"
                )

    # Reject if the function has a default for a context_arg but the
    # context_type declares that field as required — this is contradictory.
    for name in context_args:
        if name in func_defaults:
            ctx_field = context_fields.get(name)
            if ctx_field is not None and ctx_field.is_required():
                raise TypeError(f"context_arg '{name}': function has a default but context_type {context_type.__name__} requires this field")

    return context_type


_UNSET = object()


def flow_model(
    func: Optional[_AnyCallable] = None,
    *,
    # Context handling
    context_args: Optional[List[str]] = None,
    context_type: Optional[Type[ContextBase]] = None,
    # Flow.call options (passed to generated __call__)
    # Default to _UNSET so FlowOptionsOverride can control these globally.
    # Only explicitly user-provided values are passed to Flow.call.
    cacheable: Any = _UNSET,
    volatile: Any = _UNSET,
    log_level: Any = _UNSET,
    validate_result: Any = _UNSET,
    verbose: Any = _UNSET,
    evaluator: Any = _UNSET,
) -> _AnyCallable:
    """Decorator that generates a CallableModel class from a plain Python function.

    This is syntactic sugar over CallableModel. The decorator generates a real
    CallableModel class with proper __call__ and __deps__ methods, so all existing
    features (caching, evaluation, registry, serialization) work unchanged.

    Args:
        func: The function to decorate
        context_args: List of parameter names that come from context (for unpacked mode)
        context_type: Explicit ContextBase subclass to use with ``context_args`` mode.
        cacheable: Enable caching of results (default: unset, inherits from FlowOptionsOverride)
        volatile: Mark as volatile (always re-execute) (default: unset, inherits from FlowOptionsOverride)
        log_level: Logging verbosity (default: unset, inherits from FlowOptionsOverride)
        validate_result: Validate return type (default: unset, inherits from FlowOptionsOverride)
        verbose: Verbose logging output (default: unset, inherits from FlowOptionsOverride)
        evaluator: Custom evaluator (default: unset, inherits from FlowOptionsOverride)

    Two Context Modes:
        1. Explicit context parameter: Function has a 'context' parameter annotated
           with a ContextBase subclass.

           @Flow.model
           def load_prices(context: DateRangeContext, source: str) -> GenericResult[pl.DataFrame]:
               ...

        2. Unpacked context_args: Context fields are unpacked into function parameters.

           @Flow.model(context_args=["start_date", "end_date"], context_type=DateRangeContext)
           def load_prices(start_date: date, end_date: date, source: str) -> GenericResult[pl.DataFrame]:
               ...

    Returns:
        A factory function that creates CallableModel instances
    """

    def decorator(fn: _AnyCallable) -> _AnyCallable:
        sig = inspect.signature(fn)
        params = sig.parameters

        # Resolve string annotations (PEP 563 / from __future__ import annotations)
        # into real type objects. include_extras=True preserves Annotated metadata.
        try:
            _resolved_hints = get_type_hints(fn, include_extras=True)
        except Exception:
            _resolved_hints = {}

        # Validate return type
        return_type = _resolved_hints.get("return", sig.return_annotation)
        if return_type is inspect.Signature.empty:
            raise TypeError(f"Function {_callable_name(fn)} must have a return type annotation")
        # Check if return type is a ResultBase subclass; if not, auto-wrap in GenericResult
        return_origin = get_origin(return_type) or return_type
        auto_wrap_result = False
        if not (isinstance(return_origin, type) and issubclass(return_origin, ResultBase)):
            auto_wrap_result = True
            internal_return_type = GenericResult  # unparameterized for safety
        else:
            internal_return_type = return_type

        # Determine context mode
        context_schema_early: Dict[str, Type] = {}
        context_td_early = None
        if "context" in params or "_" in params:
            # Mode 1: Explicit context parameter (named 'context' or '_' for unused)
            if context_type is not None:
                raise TypeError("context_type=... is only supported when using context_args=[...]")
            context_param_name = "context" if "context" in params else "_"
            context_param = params[context_param_name]
            context_annotation = _resolved_hints.get(context_param_name, context_param.annotation)
            if context_annotation is inspect.Parameter.empty:
                raise TypeError(f"Function {_callable_name(fn)}: '{context_param_name}' parameter must have a type annotation")
            resolved_context_type = context_annotation
            if not (isinstance(resolved_context_type, type) and issubclass(resolved_context_type, ContextBase)):
                raise TypeError(f"Function {_callable_name(fn)}: '{context_param_name}' must be annotated with a ContextBase subclass")
            model_field_params = {name: param for name, param in params.items() if name not in (context_param_name, "self")}
            use_context_args = False
            explicit_context_args = None
        elif context_args is not None:
            # Mode 2: Explicit context_args - specified params come from context
            context_param_name = "context"
            context_schema_early, context_td_early = _build_context_schema(context_args, fn, sig, _resolved_hints)
            _func_defaults_set = {name for name in context_args if sig.parameters[name].default is not inspect.Parameter.empty}
            explicit_context_type = (
                _validate_context_type_override(context_type, context_args, context_schema_early, _func_defaults_set)
                if context_type is not None
                else None
            )
            resolved_context_type = explicit_context_type if explicit_context_type is not None else FlowContext
            # Exclude context_args from model fields
            model_field_params = {name: param for name, param in params.items() if name not in context_args and name != "self"}
            use_context_args = True
            explicit_context_args = context_args
        else:
            # Mode 3: Dynamic deferred mode - every param can be configured on the model,
            # but only params without Python defaults remain runtime inputs when omitted.
            if context_type is not None:
                raise TypeError("context_type=... is only supported when using context_args=[...]")
            context_param_name = "context"
            resolved_context_type = FlowContext
            model_field_params = {name: param for name, param in params.items() if name != "self"}
            use_context_args = True
            explicit_context_args = None  # Dynamic - determined at construction

        # Analyze parameters to find lazy fields and regular fields.
        model_fields: Dict[str, Tuple[Type, Any]] = {}  # name -> (type, default)
        lazy_fields: set[str] = set()  # Names of parameters marked with Lazy[T]
        default_param_names: set[str] = set()

        # In dynamic deferred mode (no explicit context_args), fields without Python defaults
        # are internally represented by a deferred sentinel until runtime context supplies them.
        dynamic_deferred_mode = use_context_args and explicit_context_args is None

        for name, param in model_field_params.items():
            # Use resolved hint (handles PEP 563 string annotations)
            annotation = _resolved_hints.get(name, param.annotation)
            if annotation is inspect.Parameter.empty:
                raise TypeError(f"Parameter '{name}' must have a type annotation")

            # Check for Lazy[T] annotation first
            unwrapped_annotation, is_lazy = _extract_lazy(annotation)
            if is_lazy:
                lazy_fields.add(name)

            if param.default is not inspect.Parameter.empty:
                default_param_names.add(name)
                default = param.default
            elif dynamic_deferred_mode:
                # In dynamic mode, params without defaults remain deferred to runtime context.
                default = Field(default_factory=_deferred_input_factory, exclude_if=_has_deferred_input)
            else:
                # In explicit mode, params without defaults are required
                default = ...

            model_fields[name] = (Any, default)

        # Capture variables for closures
        ctx_param_name = context_param_name if not use_context_args else "context"
        all_param_names = list(model_fields.keys())  # All non-context params (model fields)
        all_param_types = {name: _resolved_hints.get(name, param.annotation) for name, param in model_field_params.items()}
        # For explicit context_args mode, we also need the list of context arg names
        ctx_args_for_closure = context_args if context_args is not None else []
        is_dynamic_mode = use_context_args and explicit_context_args is None

        # Compute context_arg defaults and validators for Mode 2 (context_args)
        context_arg_defaults: Dict[str, Any] = {}
        _ctx_validatable_types: Dict[str, Type] = {}
        _ctx_validators: Dict[str, TypeAdapter] = {}
        if context_args is not None:
            for name in context_args:
                p = sig.parameters[name]
                if p.default is not inspect.Parameter.empty:
                    context_arg_defaults[name] = p.default
            _ctx_validatable_types, _ctx_validators = _build_config_validators(context_schema_early)

        # Create the __call__ method
        def make_call_impl():
            def __call__(self, context):
                def resolve_callable_model(value):
                    """Resolve a CallableModel field."""
                    resolved = value(context)
                    if isinstance(resolved, GenericResult):
                        return resolved.value
                    return resolved

                # Build kwargs for the original function
                fn_kwargs = {}

                def _resolve_field(name, value):
                    """Resolve a single field value, handling lazy wrapping."""
                    is_dep = isinstance(value, CallableModel)
                    if name in lazy_fields:
                        # Lazy field: wrap in a thunk regardless of type
                        if is_dep:
                            return _make_lazy_thunk(value, context)
                        else:
                            # Non-dep value: wrap in trivial thunk
                            return lambda v=value: v
                    elif is_dep:
                        return resolve_callable_model(value)
                    else:
                        return value

                if not use_context_args:
                    # Mode 1: Explicit context param - pass context directly
                    fn_kwargs[ctx_param_name] = context
                    # Add model fields
                    for name in all_param_names:
                        value = getattr(self, name)
                        fn_kwargs[name] = _resolve_field(name, value)
                elif not is_dynamic_mode:
                    # Mode 2: Explicit context_args - get those from context, rest from self
                    for name in ctx_args_for_closure:
                        value = getattr(context, name, _UNSET)
                        if value is _UNSET:
                            if name in context_arg_defaults:
                                fn_kwargs[name] = context_arg_defaults[name]
                            else:
                                raise TypeError(f"Missing context field '{name}'")
                        else:
                            fn_kwargs[name] = _coerce_context_value(name, value, _ctx_validators, _ctx_validatable_types)
                    # Add model fields
                    for name in all_param_names:
                        value = getattr(self, name)
                        fn_kwargs[name] = _resolve_field(name, value)
                else:
                    # Mode 3: Dynamic deferred mode - explicit values or Python defaults from self,
                    # otherwise values come from runtime context.
                    explicit_fields = _bound_field_names(self)
                    missing_fields = []

                    for name in all_param_names:
                        value = getattr(self, name, _DEFERRED_INPUT)
                        if name in explicit_fields or name in default_param_names:
                            # Explicitly provided or implicitly bound via Python default.
                            value = getattr(self, name)
                            fn_kwargs[name] = _resolve_field(name, value)
                            continue

                        if _has_deferred_input(value):
                            value = getattr(context, name, _UNSET)
                            if value is _UNSET:
                                missing_fields.append(name)
                                continue
                            # Validate/coerce context-sourced value, skip CallableModel deps
                            if not _is_model_dependency(value):
                                value = _coerce_context_value(name, value, _config_validators, _validatable_types)
                        fn_kwargs[name] = _resolve_field(name, value)

                    if missing_fields:
                        missing = ", ".join(sorted(missing_fields))
                        raise TypeError(
                            f"Missing runtime input(s) for {_callable_name(fn)}: {missing}. "
                            "Provide them in the call context or bind them at construction time."
                        )

                raw_result = fn(**fn_kwargs)
                if auto_wrap_result:
                    return GenericResult(value=raw_result)
                return raw_result

            # Set proper signature for CallableModel validation
            cast(Any, __call__).__signature__ = inspect.Signature(
                parameters=[
                    inspect.Parameter("self", inspect.Parameter.POSITIONAL_OR_KEYWORD),
                    inspect.Parameter("context", inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=resolved_context_type),
                ],
                return_annotation=internal_return_type,
            )
            return __call__

        call_impl = make_call_impl()

        # Apply Flow.call decorator — only include options the user explicitly set
        flow_options = {}
        for opt_name, opt_val in [
            ("cacheable", cacheable),
            ("volatile", volatile),
            ("log_level", log_level),
            ("validate_result", validate_result),
            ("verbose", verbose),
            ("evaluator", evaluator),
        ]:
            if opt_val is not _UNSET:
                flow_options[opt_name] = opt_val

        decorated_call = Flow.call(**flow_options)(call_impl)

        # Create the __deps__ method
        def make_deps_impl():
            def __deps__(self, context) -> GraphDepList:
                deps = []
                # Check ALL fields for CallableModel dependencies (auto-detection)
                for name in model_fields:
                    if name in lazy_fields:
                        continue  # Lazy deps are NOT pre-evaluated
                    value = getattr(self, name)
                    if isinstance(value, BoundModel):
                        deps.append((value.model, [value._transform_context(context)]))
                    elif isinstance(value, CallableModel):
                        deps.append((value, [context]))
                return deps

            # Set proper signature
            cast(Any, __deps__).__signature__ = inspect.Signature(
                parameters=[
                    inspect.Parameter("self", inspect.Parameter.POSITIONAL_OR_KEYWORD),
                    inspect.Parameter("context", inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=resolved_context_type),
                ],
                return_annotation=GraphDepList,
            )
            return __deps__

        deps_impl = make_deps_impl()
        decorated_deps = Flow.deps(deps_impl)

        # Build pydantic field annotations for the class
        annotations = {}

        namespace = {
            "__module__": _callable_module(fn),
            "__qualname__": f"_{_callable_name(fn)}_Model",
            "__call__": decorated_call,
            "__deps__": decorated_deps,
        }

        for name, (typ, default) in model_fields.items():
            annotations[name] = typ
            if default is not ...:
                namespace[name] = default
            else:
                # For required fields, use Field(...)
                namespace[name] = Field(...)

        namespace["__annotations__"] = annotations

        _validatable_types, _config_validators = _build_config_validators(all_param_types)

        # Create the class using type()
        GeneratedModel = cast(type[_GeneratedFlowModelBase], type(f"_{_callable_name(fn)}_Model", (_GeneratedFlowModelBase,), namespace))

        # Set class-level attributes after class creation (to avoid pydantic processing)
        GeneratedModel.__flow_model_context_type__ = resolved_context_type
        GeneratedModel.__flow_model_return_type__ = internal_return_type
        setattr(GeneratedModel, "__flow_model_func__", fn)
        GeneratedModel.__flow_model_use_context_args__ = use_context_args
        GeneratedModel.__flow_model_explicit_context_args__ = explicit_context_args
        GeneratedModel.__flow_model_all_param_types__ = all_param_types  # All param name -> type
        GeneratedModel.__flow_model_default_param_names__ = default_param_names
        GeneratedModel.__flow_model_context_arg_defaults__ = context_arg_defaults
        GeneratedModel.__flow_model_auto_wrap__ = auto_wrap_result
        GeneratedModel.__flow_model_validatable_types__ = _validatable_types
        GeneratedModel.__flow_model_config_validators__ = _config_validators

        # Build context_schema
        context_schema: Dict[str, Type] = {}
        context_td = None

        if explicit_context_args is not None:
            # Explicit context_args provided - use early-computed schema
            context_schema, context_td = context_schema_early, context_td_early
        elif not use_context_args:
            # Explicit context mode - schema comes from the context type's fields
            if hasattr(resolved_context_type, "model_fields"):
                context_schema = {name: info.annotation for name, info in resolved_context_type.model_fields.items()}
        # For dynamic mode (is_dynamic_mode), _context_schema remains empty
        # and schema is built dynamically from the instance's unresolved runtime inputs.

        # Store context schema for TypedDict-based validation (picklable!)
        GeneratedModel._context_schema = context_schema
        GeneratedModel._context_td = context_td
        # Validator is created lazily to survive pickling
        GeneratedModel._cached_context_validator = None

        # Register the MODEL class for serialization (needed for model_dump/_target_).
        # Note: We do NOT register dynamic context classes anymore - context handling
        # uses FlowContext + TypedDict instead, which don't need registration.
        register_ccflow_import_path(GeneratedModel)

        # Rebuild the model to process annotations properly
        GeneratedModel.model_rebuild()

        # Create factory function that returns model instances
        @wraps(fn)
        def factory(**kwargs) -> _GeneratedFlowModelBase:
            _validate_config_kwargs(kwargs, _validatable_types, _config_validators)
            return GeneratedModel(**kwargs)

        # Preserve useful attributes on factory
        cast(Any, factory)._generated_model = GeneratedModel
        factory.__doc__ = fn.__doc__

        return factory

    # Handle both @Flow.model and @Flow.model(...) syntax
    if func is not None:
        return decorator(func)
    return decorator
