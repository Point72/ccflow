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
from functools import wraps
from typing import Annotated, Any, Callable, ClassVar, Dict, List, Optional, Tuple, Type, Union, cast, get_args, get_origin

from pydantic import Field, TypeAdapter, model_validator
from typing_extensions import TypedDict

from .base import ContextBase, ResultBase
from .callable import CallableModel, Flow, GraphDepList, _CallableModel
from .context import FlowContext
from .local_persistence import register_ccflow_import_path
from .result import GenericResult

__all__ = ("flow_model", "FlowAPI", "BoundModel", "Lazy", "FieldExtractor")

_AnyCallable = Callable[..., Any]


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


def _build_typed_dict_adapter(name: str, schema: Dict[str, Type]) -> TypeAdapter:
    """Build a TypeAdapter for a runtime TypedDict schema."""

    if not schema:
        return TypeAdapter(dict)
    return TypeAdapter(TypedDict(name, schema))


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


def _validate_config_kwargs(kwargs: Dict[str, Any], validatable_types: Dict[str, Type], validators: Dict[str, TypeAdapter]) -> None:
    """Validate plain config inputs while still allowing dependency objects."""

    if not validators:
        return

    from .base import ModelRegistry as _MR
    from .callable import CallableModel as _CM

    for field_name, validator in validators.items():
        if field_name not in kwargs:
            continue
        value = kwargs[field_name]
        if value is None or isinstance(value, (_CM, BoundModel)):
            continue
        if isinstance(value, str) and value in _MR.root():
            continue
        try:
            validator.validate_python(value)
        except Exception:
            expected_type = validatable_types[field_name]
            raise TypeError(f"Field '{field_name}': expected {expected_type}, got {type(value).__name__} ({value!r})")


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
            The model's result, unwrapped from GenericResult if applicable.
        """
        ctx = self._build_context(kwargs)

        # Call the model
        result = self._model(ctx)

        # Unwrap GenericResult if present
        if isinstance(result, GenericResult):
            return result.value
        return result

    @property
    def unbound_inputs(self) -> Dict[str, Type]:
        """Return the context schema (field name -> type).

        In deferred mode, this is everything NOT provided at construction.
        """
        all_param_types = getattr(self._model.__class__, "__flow_model_all_param_types__", {})
        bound_fields = getattr(self._model, "_bound_fields", set())
        model_cls = self._model.__class__

        # If explicit context_args was provided, use _context_schema
        explicit_args = getattr(model_cls, "__flow_model_explicit_context_args__", None)
        if explicit_args is not None:
            context_schema = getattr(model_cls, "_context_schema", None)
            return context_schema.copy() if context_schema is not None else {}

        # Dynamic @Flow.model: unbound = all params - bound
        if all_param_types:
            return {name: typ for name, typ in all_param_types.items() if name not in bound_fields}

        # Generic CallableModel: runtime inputs are the context schema.
        context_cls = _concrete_context_type(self._model.context_type)
        if context_cls is None or not hasattr(context_cls, "model_fields"):
            return {}
        return {name: info.annotation for name, info in context_cls.model_fields.items()}

    @property
    def bound_inputs(self) -> Dict[str, Any]:
        """Return the config values bound at construction time."""
        bound_fields = getattr(self._model, "_bound_fields", set())
        result: Dict[str, Any] = {}
        for name in bound_fields:
            if hasattr(self._model, name):
                result[name] = getattr(self._model, name)
        if result:
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


class BoundModel:
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

    def __init__(self, model: CallableModel, input_transforms: Dict[str, Any]):
        self._model = model
        self._input_transforms = input_transforms

    def _transform_context(self, context: ContextBase) -> FlowContext:
        """Return a FlowContext with this model's input transforms applied."""
        ctx_dict = _context_values(context)
        for name, transform in self._input_transforms.items():
            if callable(transform):
                ctx_dict[name] = transform(context)
            else:
                ctx_dict[name] = transform
        return FlowContext(**ctx_dict)

    def __call__(self, context: ContextBase) -> Any:
        """Call the model with transformed context."""
        return self._model(self._transform_context(context))

    def __repr__(self) -> str:
        transforms = ", ".join(f"{name}={_transform_repr(transform)}" for name, transform in self._input_transforms.items())
        return f"{self._model!r}.flow.with_inputs({transforms})"

    @property
    def flow(self) -> "FlowAPI":
        """Access the flow API."""
        return _BoundFlowAPI(self)


class _BoundFlowAPI(FlowAPI):
    """FlowAPI that delegates to a BoundModel, honoring transforms."""

    def __init__(self, bound_model: BoundModel):
        self._bound = bound_model
        super().__init__(bound_model._model)

    def compute(self, **kwargs) -> Any:
        ctx = self._build_context(kwargs)
        result = self._bound(ctx)  # Call through BoundModel, not _model
        if isinstance(result, GenericResult):
            return result.value
        return result

    def with_inputs(self, **transforms) -> "BoundModel":
        """Chain transforms: merge new transforms with existing ones.

        New transforms override existing ones for the same key.
        """
        merged = {**self._bound._input_transforms, **transforms}
        return BoundModel(model=self._bound._model, input_transforms=merged)


class _FieldExtractorMixin:
    """Turn unknown public attributes into FieldExtractors.

    Real model attributes are still resolved by the normal pydantic/base-model
    attribute path via ``super().__getattr__``.
    """

    def __getattr__(self, name):
        try:
            super_getattr = getattr(super(), "__getattr__", None)
            if super_getattr is None:
                raise AttributeError(name)
            return super_getattr(name)
        except AttributeError:
            if name.startswith("_"):
                raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'") from None
            return FieldExtractor(source=self, field_name=name)


class _GeneratedFlowModelBase(_FieldExtractorMixin, CallableModel):
    """Shared behavior for models generated by ``@Flow.model``."""

    __flow_model_context_type__: ClassVar[Type[ContextBase]] = FlowContext
    __flow_model_return_type__: ClassVar[Type[ResultBase]] = GenericResult
    __flow_model_func__: ClassVar[_AnyCallable | None] = None
    __flow_model_use_context_args__: ClassVar[bool] = True
    __flow_model_explicit_context_args__: ClassVar[Optional[List[str]]] = None
    __flow_model_all_param_types__: ClassVar[Dict[str, Type]] = {}
    __flow_model_auto_wrap__: ClassVar[bool] = False
    _context_schema: ClassVar[Dict[str, Type]] = {}
    _context_td: ClassVar[Any | None] = None
    _matched_context_type: ClassVar[Optional[Type[ContextBase]]] = None
    _cached_context_validator: ClassVar[TypeAdapter | None] = None

    @model_validator(mode="before")
    def _resolve_registry_refs(cls, values, info):
        if not isinstance(values, dict):
            return values

        from .base import BaseModel as _BM

        param_types = getattr(cls, "__flow_model_all_param_types__", {})
        resolved = dict(values)
        for field_name, expected_type in param_types.items():
            if field_name not in resolved:
                continue
            value = resolved[field_name]
            if not isinstance(value, str):
                continue
            if expected_type is str:
                continue
            try:
                candidate = _BM.model_validate(value)
            except Exception:
                continue
            if isinstance(candidate, _BM):
                resolved[field_name] = candidate
        return resolved

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
                if cls._context_td is not None:
                    cls._cached_context_validator = TypeAdapter(cls._context_td)
                elif cls._context_schema:
                    cls._cached_context_validator = _build_typed_dict_adapter(f"{cls.__name__}Inputs", cls._context_schema)
                else:
                    cls._cached_context_validator = TypeAdapter(cls.__flow_model_context_type__)
            return cls._cached_context_validator

        if not hasattr(self, "_instance_context_validator"):
            all_param_types = getattr(cls, "__flow_model_all_param_types__", {})
            bound_fields = getattr(self, "_bound_fields", set())
            unbound_schema = {name: typ for name, typ in all_param_types.items() if name not in bound_fields}
            object.__setattr__(self, "_instance_context_validator", _build_typed_dict_adapter(f"{cls.__name__}Inputs", unbound_schema))
        return self._instance_context_validator


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
    context_args: List[str], func: _AnyCallable, sig: inspect.Signature
) -> Tuple[Dict[str, Type], Any, Optional[Type[ContextBase]]]:
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
        Tuple of (schema_dict, TypedDict type, optional matched ContextBase type)
    """
    # Build schema dict from parameter annotations
    schema = {}
    for name in context_args:
        if name not in sig.parameters:
            raise ValueError(f"context_arg '{name}' not found in function parameters")
        param = sig.parameters[name]
        if param.annotation is inspect.Parameter.empty:
            raise ValueError(f"context_arg '{name}' must have a type annotation")
        schema[name] = param.annotation

    # Try to match common context types for compatibility
    matched_context_type = None
    from .context import DateRangeContext

    if set(context_args) == {"start_date", "end_date"}:
        from datetime import date

        if all(
            sig.parameters[name].annotation in (date, "date")
            or (isinstance(sig.parameters[name].annotation, type) and sig.parameters[name].annotation is date)
            for name in context_args
        ):
            matched_context_type = DateRangeContext

    # Create TypedDict for validation (not registered anywhere!)
    context_td = TypedDict(f"{_callable_name(func)}Inputs", schema)

    return schema, context_td, matched_context_type


_UNSET = object()


def flow_model(
    func: Optional[_AnyCallable] = None,
    *,
    # Context handling
    context_args: Optional[List[str]] = None,
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

           @Flow.model(context_args=["start_date", "end_date"])
           def load_prices(start_date: date, end_date: date, source: str) -> GenericResult[pl.DataFrame]:
               ...

    Returns:
        A factory function that creates CallableModel instances
    """

    def decorator(fn: _AnyCallable) -> _AnyCallable:
        import typing as _typing

        sig = inspect.signature(fn)
        params = sig.parameters

        # Resolve string annotations (PEP 563 / from __future__ import annotations)
        # into real type objects. include_extras=True preserves Annotated metadata.
        try:
            _resolved_hints = _typing.get_type_hints(fn, include_extras=True)
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
        if "context" in params or "_" in params:
            # Mode 1: Explicit context parameter (named 'context' or '_' for unused)
            context_param_name = "context" if "context" in params else "_"
            context_param = params[context_param_name]
            context_annotation = _resolved_hints.get(context_param_name, context_param.annotation)
            if context_annotation is inspect.Parameter.empty:
                raise TypeError(f"Function {_callable_name(fn)}: '{context_param_name}' parameter must have a type annotation")
            context_type = context_annotation
            if not (isinstance(context_type, type) and issubclass(context_type, ContextBase)):
                raise TypeError(f"Function {_callable_name(fn)}: '{context_param_name}' must be annotated with a ContextBase subclass")
            model_field_params = {name: param for name, param in params.items() if name not in (context_param_name, "self")}
            use_context_args = False
            explicit_context_args = None
        elif context_args is not None:
            # Mode 2: Explicit context_args - specified params come from context
            context_param_name = "context"
            # Build context schema early to determine matched_context_type
            context_schema_early, _, matched_type = _build_context_schema(context_args, fn, sig)
            # Use matched type if available (e.g., DateRangeContext), else FlowContext
            context_type = matched_type if matched_type is not None else FlowContext
            # Exclude context_args from model fields
            model_field_params = {name: param for name, param in params.items() if name not in context_args and name != "self"}
            use_context_args = True
            explicit_context_args = context_args
        else:
            # Mode 3: Dynamic deferred mode - ALL params are potential context or config
            # What's provided at construction = config/deps
            # What's NOT provided = comes from context at runtime
            context_param_name = "context"
            context_type = FlowContext
            model_field_params = {name: param for name, param in params.items() if name != "self"}
            use_context_args = True
            explicit_context_args = None  # Dynamic - determined at construction

        # Analyze parameters to find lazy fields and regular fields.
        model_fields: Dict[str, Tuple[Type, Any]] = {}  # name -> (type, default)
        lazy_fields: set[str] = set()  # Names of parameters marked with Lazy[T]

        # In dynamic deferred mode (no explicit context_args), all fields are optional
        # because values not provided at construction come from context at runtime
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
                default = param.default
            elif dynamic_deferred_mode:
                # In dynamic mode, params without defaults are optional (come from context)
                default = None
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
                    is_dep = isinstance(value, (CallableModel, BoundModel))
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
                        fn_kwargs[name] = getattr(context, name)
                    # Add model fields
                    for name in all_param_names:
                        value = getattr(self, name)
                        fn_kwargs[name] = _resolve_field(name, value)
                else:
                    # Mode 3: Dynamic deferred mode - unbound from context, bound from self
                    bound_fields = getattr(self, "_bound_fields", set())

                    for name in all_param_names:
                        if name in bound_fields:
                            # Bound at construction - get from self
                            value = getattr(self, name)
                            fn_kwargs[name] = _resolve_field(name, value)
                        else:
                            # Unbound - get from context
                            fn_kwargs[name] = getattr(context, name)

                raw_result = fn(**fn_kwargs)
                if auto_wrap_result:
                    return GenericResult(value=raw_result)
                return raw_result

            # Set proper signature for CallableModel validation
            cast(Any, __call__).__signature__ = inspect.Signature(
                parameters=[
                    inspect.Parameter("self", inspect.Parameter.POSITIONAL_OR_KEYWORD),
                    inspect.Parameter("context", inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=context_type),
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
                # Check ALL fields for CallableModels/BoundModels (auto-detection)
                for name in model_fields:
                    if name in lazy_fields:
                        continue  # Lazy deps are NOT pre-evaluated
                    value = getattr(self, name)
                    if isinstance(value, BoundModel):
                        deps.append((value._model, [value._transform_context(context)]))
                    elif isinstance(value, CallableModel):
                        deps.append((value, [context]))
                return deps

            # Set proper signature
            cast(Any, __deps__).__signature__ = inspect.Signature(
                parameters=[
                    inspect.Parameter("self", inspect.Parameter.POSITIONAL_OR_KEYWORD),
                    inspect.Parameter("context", inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=context_type),
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
        GeneratedModel.__flow_model_context_type__ = context_type
        GeneratedModel.__flow_model_return_type__ = internal_return_type
        setattr(GeneratedModel, "__flow_model_func__", fn)
        GeneratedModel.__flow_model_use_context_args__ = use_context_args
        GeneratedModel.__flow_model_explicit_context_args__ = explicit_context_args
        GeneratedModel.__flow_model_all_param_types__ = all_param_types  # All param name -> type
        GeneratedModel.__flow_model_auto_wrap__ = auto_wrap_result

        # Build context_schema and matched_context_type
        context_schema: Dict[str, Type] = {}
        context_td = None
        matched_context_type: Optional[Type[ContextBase]] = None

        if explicit_context_args is not None:
            # Explicit context_args provided - use early-computed schema
            # (matched_context_type was already used to set context_type above)
            context_schema, context_td, matched_context_type = _build_context_schema(explicit_context_args, fn, sig)
        elif not use_context_args:
            # Explicit context mode - schema comes from the context type's fields
            if hasattr(context_type, "model_fields"):
                context_schema = {name: info.annotation for name, info in context_type.model_fields.items()}
        # For dynamic mode (is_dynamic_mode), _context_schema remains empty
        # and schema is built dynamically from _bound_fields at runtime

        # Store context schema for TypedDict-based validation (picklable!)
        GeneratedModel._context_schema = context_schema
        GeneratedModel._context_td = context_td
        GeneratedModel._matched_context_type = matched_context_type
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

            instance = GeneratedModel(**kwargs)
            # Track which fields were explicitly provided at construction
            # These are "bound" - everything else comes from context at runtime
            object.__setattr__(instance, "_bound_fields", set(kwargs.keys()))
            return instance

        # Preserve useful attributes on factory
        cast(Any, factory)._generated_model = GeneratedModel
        factory.__doc__ = fn.__doc__

        return factory

    # Handle both @Flow.model and @Flow.model(...) syntax
    if func is not None:
        return decorator(func)
    return decorator


# =============================================================================
# FieldExtractor — structured output field access
# =============================================================================


class FieldExtractor(_FieldExtractorMixin, CallableModel):
    """Extracts a named field from a source model's result.

    Created automatically by accessing an unknown attribute on a @Flow.model
    instance (e.g., ``prepared.X_train``). The extractor is itself a
    CallableModel, so it can be wired as a dependency to downstream models.

    When evaluated, it runs the source model and returns
    ``GenericResult(value=getattr(source_result, field_name))``.

    Multiple extractors from the same source share the source model instance.
    If caching is enabled on the evaluator, the source is evaluated only once.
    """

    source: Any  # The source CallableModel
    field_name: str  # The attribute to extract

    @property
    def context_type(self):
        if isinstance(self.source, _CallableModel):
            return self.source.context_type
        return ContextBase

    @property
    def result_type(self):
        return GenericResult

    @Flow.call
    def __call__(self, context: ContextBase) -> GenericResult:
        result = self.source(context)
        if isinstance(result, GenericResult):
            result = result.value
        # Support both attribute access and dict key access
        if isinstance(result, dict):
            return GenericResult(value=result[self.field_name])
        return GenericResult(value=getattr(result, self.field_name))

    @Flow.deps
    def __deps__(self, context: ContextBase) -> GraphDepList:
        if isinstance(self.source, _CallableModel):
            return [(self.source, [context])]
        return []


register_ccflow_import_path(FieldExtractor)
