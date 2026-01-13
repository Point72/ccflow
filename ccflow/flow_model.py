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
from typing import Annotated, Any, Callable, Dict, List, Optional, Tuple, Type, Union, get_origin

from pydantic import Field, TypeAdapter
from typing_extensions import TypedDict

from .base import ContextBase, ResultBase
from .context import FlowContext
from .dep import Dep, extract_dep

__all__ = ("flow_model", "FlowAPI", "BoundModel", "Lazy")

log = logging.getLogger(__name__)


class FlowAPI:
    """API namespace for deferred computation operations.

    Provides methods for executing models and transforming contexts.
    Accessed via model.flow property.
    """

    def __init__(self, model: "CallableModel"):  # noqa: F821
        self._model = model

    def compute(self, **kwargs) -> Any:
        """Execute the model with the provided context arguments.

        Validates kwargs against the model's context schema using TypeAdapter,
        then wraps in FlowContext and calls the model.

        Args:
            **kwargs: Context arguments (e.g., start_date, end_date)

        Returns:
            The model's result, unwrapped from GenericResult if applicable.
        """
        # Get validator from model (lazily created if needed after unpickling)
        validator = self._model._get_context_validator()

        # Validate and coerce kwargs via TypeAdapter
        validated = validator.validate_python(kwargs)

        # Wrap in FlowContext (single class, always)
        ctx = FlowContext(**validated)

        # Call the model
        result = self._model(ctx)

        # Unwrap GenericResult if present
        if hasattr(result, "value"):
            return result.value
        return result

    @property
    def unbound_inputs(self) -> Dict[str, Type]:
        """Return the context schema (field name -> type).

        In deferred mode, this is everything NOT provided at construction.
        """
        all_param_types = getattr(self._model.__class__, "__flow_model_all_param_types__", {})
        bound_fields = getattr(self._model, "_bound_fields", set())

        # If explicit context_args was provided, use _context_schema
        explicit_args = getattr(self._model.__class__, "__flow_model_explicit_context_args__", None)
        if explicit_args is not None:
            return self._model._context_schema.copy()

        # Otherwise, unbound = all params - bound
        return {name: typ for name, typ in all_param_types.items() if name not in bound_fields}

    @property
    def bound_inputs(self) -> Dict[str, Any]:
        """Return the config values bound at construction time."""
        bound_fields = getattr(self._model, "_bound_fields", set())
        result = {}
        for name in bound_fields:
            if hasattr(self._model, name):
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
    """

    def __init__(self, model: "CallableModel", input_transforms: Dict[str, Any]):  # noqa: F821
        self._model = model
        self._input_transforms = input_transforms

    def __call__(self, context: ContextBase) -> Any:
        """Call the model with transformed context."""
        # Build new context dict with transforms applied
        ctx_dict = {}

        # Get fields from context
        if hasattr(context, "__pydantic_extra__") and context.__pydantic_extra__:
            ctx_dict.update(context.__pydantic_extra__)
        for field in context.__class__.model_fields:
            ctx_dict[field] = getattr(context, field)

        # Apply transforms
        for name, transform in self._input_transforms.items():
            if callable(transform):
                ctx_dict[name] = transform(context)
            else:
                ctx_dict[name] = transform

        # Create new context and call model
        new_ctx = FlowContext(**ctx_dict)
        return self._model(new_ctx)

    @property
    def flow(self) -> FlowAPI:
        """Access the flow API."""
        return FlowAPI(self._model)


class Lazy:
    """Deferred model execution with runtime context overrides.

    Wraps a CallableModel to allow context fields to be determined at
    runtime rather than at construction time. Use in with_inputs() when
    you need values that aren't available until execution.

    Example:
        # Create a model that needs runtime-determined context
        market_data = load_market_data(symbols=["AAPL"])

        # Use Lazy to defer the start_date calculation
        lookback_data = market_data.flow.with_inputs(
            start_date=Lazy(market_data)(start_date=lambda ctx: ctx.start_date - timedelta(days=7))
        )

        # More commonly, use Lazy for self-referential transforms:
        adjusted_model = model.flow.with_inputs(
            value=Lazy(other_model)(multiplier=2)  # Call other_model with multiplier=2
        )

    The __call__ method returns a callable that, when invoked with a context,
    calls the wrapped model with the specified overrides applied.
    """

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
            ctx_dict = {}
            if hasattr(context, "__pydantic_extra__") and context.__pydantic_extra__:
                ctx_dict.update(context.__pydantic_extra__)
            for field in context.__class__.model_fields:
                ctx_dict[field] = getattr(context, field)

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
    context_args: List[str], func: Callable, sig: inspect.Signature
) -> Tuple[Dict[str, Type], Type, Optional[Type[ContextBase]]]:
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
    context_td = TypedDict(f"{func.__name__}Inputs", schema)

    return schema, context_td, matched_context_type


def _get_dep_info(annotation) -> Tuple[Type, Optional[Dep]]:
    """Extract dependency info from an annotation.

    Returns:
        Tuple of (base_type, Dep instance or None)
    """
    return extract_dep(annotation)


def flow_model(
    func: Callable = None,
    *,
    # Context handling
    context_args: Optional[List[str]] = None,
    # Flow.call options (passed to generated __call__)
    cacheable: bool = False,
    volatile: bool = False,
    log_level: int = logging.DEBUG,
    validate_result: bool = True,
    verbose: bool = True,
    evaluator: Optional[Any] = None,
) -> Callable:
    """Decorator that generates a CallableModel class from a plain Python function.

    This is syntactic sugar over CallableModel. The decorator generates a real
    CallableModel class with proper __call__ and __deps__ methods, so all existing
    features (caching, evaluation, registry, serialization) work unchanged.

    Args:
        func: The function to decorate
        context_args: List of parameter names that come from context (for unpacked mode)
        cacheable: Enable caching of results
        volatile: Mark as volatile (always re-execute)
        log_level: Logging verbosity
        validate_result: Validate return type
        verbose: Verbose logging output
        evaluator: Custom evaluator

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

    def decorator(fn: Callable) -> Callable:
        # Import here to avoid circular imports
        from .callable import CallableModel, Flow, GraphDepList

        sig = inspect.signature(fn)
        params = sig.parameters

        # Validate return type
        return_type = sig.return_annotation
        if return_type is inspect.Signature.empty:
            raise TypeError(f"Function {fn.__name__} must have a return type annotation")
        # Check that return type is a ResultBase subclass
        return_origin = get_origin(return_type) or return_type
        if not (isinstance(return_origin, type) and issubclass(return_origin, ResultBase)):
            raise TypeError(f"Function {fn.__name__} must return a ResultBase subclass, got {return_type}")

        # Determine context mode
        if "context" in params or "_" in params:
            # Mode 1: Explicit context parameter (named 'context' or '_' for unused)
            context_param_name = "context" if "context" in params else "_"
            context_param = params[context_param_name]
            if context_param.annotation is inspect.Parameter.empty:
                raise TypeError(f"Function {fn.__name__}: '{context_param_name}' parameter must have a type annotation")
            context_type = context_param.annotation
            if not (isinstance(context_type, type) and issubclass(context_type, ContextBase)):
                raise TypeError(f"Function {fn.__name__}: '{context_param_name}' must be annotated with a ContextBase subclass")
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

        # Analyze parameters to find dependencies and regular fields
        dep_fields: Dict[str, Tuple[Type, Dep]] = {}  # name -> (base_type, Dep)
        model_fields: Dict[str, Tuple[Type, Any]] = {}  # name -> (type, default)

        # In dynamic deferred mode (no explicit context_args), all fields are optional
        # because values not provided at construction come from context at runtime
        dynamic_deferred_mode = use_context_args and explicit_context_args is None

        for name, param in model_field_params.items():
            if param.annotation is inspect.Parameter.empty:
                raise TypeError(f"Parameter '{name}' must have a type annotation")

            base_type, dep = _get_dep_info(param.annotation)
            if param.default is not inspect.Parameter.empty:
                default = param.default
            elif dynamic_deferred_mode:
                # In dynamic mode, params without defaults are optional (come from context)
                default = None
            else:
                # In explicit mode, params without defaults are required
                default = ...

            if dep is not None:
                # This is an explicit dependency parameter (DepOf annotation)
                dep_fields[name] = (base_type, dep)
                # Use Annotated so _resolve_deps_and_call in callable.py can find the Dep
                model_fields[name] = (Annotated[Union[base_type, CallableModel], dep], default)
            else:
                # Regular model field - use Any for auto-detection of CallableModels.
                # We can't use Union[T, CallableModel] because Pydantic tries to generate
                # schema for T, which fails for arbitrary types like pl.DataFrame.
                # Using Any allows any value; we do runtime isinstance checks in __call__.
                model_fields[name] = (Any, default)

        # Capture variables for closures
        ctx_param_name = context_param_name if not use_context_args else "context"
        all_param_names = list(model_fields.keys())  # All non-context params (model fields)
        all_param_types = {name: param.annotation for name, param in model_field_params.items()}
        # For explicit context_args mode, we also need the list of context arg names
        ctx_args_for_closure = context_args if context_args is not None else []
        is_dynamic_mode = use_context_args and explicit_context_args is None

        # Create the __call__ method
        def make_call_impl():
            def __call__(self, context):
                # Import here (inside function) to avoid pickling issues with ContextVar
                from .callable import _resolved_deps

                # Check if this model has custom deps (from @func.deps decorator)
                has_custom_deps = getattr(self.__class__, "__has_custom_deps__", False)

                def resolve_callable_model(name, value, store):
                    """Resolve a CallableModel field.

                    When has_custom_deps is True and the value is NOT in the store,
                    it means the custom deps function chose not to include this dep.
                    In that case, we return None (the field's default) instead of
                    calling the CallableModel directly.
                    """
                    if id(value) in store:
                        return store[id(value)]
                    elif has_custom_deps:
                        # Custom deps excluded this field - use None
                        return None
                    else:
                        # Auto-detection fallback: call directly
                        resolved = value(context)
                        if hasattr(resolved, 'value'):
                            return resolved.value
                        return resolved

                # Build kwargs for the original function
                fn_kwargs = {}
                store = _resolved_deps.get()

                if not use_context_args:
                    # Mode 1: Explicit context param - pass context directly
                    fn_kwargs[ctx_param_name] = context
                    # Add model fields
                    for name in all_param_names:
                        value = getattr(self, name)
                        if isinstance(value, CallableModel):
                            fn_kwargs[name] = resolve_callable_model(name, value, store)
                        else:
                            fn_kwargs[name] = value
                elif not is_dynamic_mode:
                    # Mode 2: Explicit context_args - get those from context, rest from self
                    for name in ctx_args_for_closure:
                        fn_kwargs[name] = getattr(context, name)
                    # Add model fields
                    for name in all_param_names:
                        value = getattr(self, name)
                        if isinstance(value, CallableModel):
                            fn_kwargs[name] = resolve_callable_model(name, value, store)
                        else:
                            fn_kwargs[name] = value
                else:
                    # Mode 3: Dynamic deferred mode - unbound from context, bound from self
                    bound_fields = getattr(self, "_bound_fields", set())

                    for name in all_param_names:
                        if name in bound_fields:
                            # Bound at construction - get from self
                            value = getattr(self, name)
                            if isinstance(value, CallableModel):
                                fn_kwargs[name] = resolve_callable_model(name, value, store)
                            else:
                                fn_kwargs[name] = value
                        else:
                            # Unbound - get from context
                            fn_kwargs[name] = getattr(context, name)

                return fn(**fn_kwargs)

            # Set proper signature for CallableModel validation
            __call__.__signature__ = inspect.Signature(
                parameters=[
                    inspect.Parameter("self", inspect.Parameter.POSITIONAL_OR_KEYWORD),
                    inspect.Parameter("context", inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=context_type),
                ],
                return_annotation=return_type,
            )
            return __call__

        call_impl = make_call_impl()

        # Apply Flow.call decorator
        flow_options = {
            "cacheable": cacheable,
            "volatile": volatile,
            "log_level": log_level,
            "validate_result": validate_result,
            "verbose": verbose,
        }
        if evaluator is not None:
            flow_options["evaluator"] = evaluator

        decorated_call = Flow.call(**flow_options)(call_impl)

        # Create the __deps__ method
        def make_deps_impl():
            def __deps__(self, context) -> GraphDepList:
                deps = []
                # Check ALL fields for CallableModels (auto-detection)
                for name in model_fields:
                    value = getattr(self, name)
                    if isinstance(value, CallableModel):
                        if name in dep_fields:
                            # Explicit DepOf with transform (backwards compat)
                            _, dep_obj = dep_fields[name]
                            transformed_ctx = dep_obj.apply(context)
                            deps.append((value, [transformed_ctx]))
                        else:
                            # Auto-detected dependency - use context as-is
                            deps.append((value, [context]))
                return deps

            # Set proper signature
            __deps__.__signature__ = inspect.Signature(
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
            "__module__": fn.__module__,
            "__qualname__": f"_{fn.__name__}_Model",
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

        # Add model validator for dependency validation if we have dep fields
        if dep_fields:
            from pydantic import model_validator

            # Create validator function that captures dep_fields and context_type
            def make_dep_validator(d_fields, ctx_type):
                @model_validator(mode="after")
                def __validate_deps__(self):
                    from .callable import CallableModel

                    for dep_name, (base_type, dep_obj) in d_fields.items():
                        value = getattr(self, dep_name)
                        if isinstance(value, CallableModel):
                            dep_obj.validate_dependency(value, base_type, ctx_type, dep_name)
                    return self

                return __validate_deps__

            namespace["__validate_deps__"] = make_dep_validator(dep_fields, context_type)

        # Create the class using type()
        GeneratedModel = type(f"_{fn.__name__}_Model", (CallableModel,), namespace)

        # Set class-level attributes after class creation (to avoid pydantic processing)
        GeneratedModel.__flow_model_context_type__ = context_type
        GeneratedModel.__flow_model_return_type__ = return_type
        GeneratedModel.__flow_model_func__ = fn
        GeneratedModel.__flow_model_dep_fields__ = dep_fields
        GeneratedModel.__flow_model_use_context_args__ = use_context_args
        GeneratedModel.__flow_model_explicit_context_args__ = explicit_context_args
        GeneratedModel.__flow_model_all_param_types__ = all_param_types  # All param name -> type

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

        # Method to get/create context validator (lazy for pickling support)
        def _get_context_validator(self) -> TypeAdapter:
            """Get or create the context validator.

            For dynamic deferred mode, builds schema from unbound fields.
            For explicit context_args or explicit context mode, uses cached schema.
            """
            cls = self.__class__
            explicit_args = getattr(cls, "__flow_model_explicit_context_args__", None)

            # For explicit context_args or explicit context mode, use cached validator
            if explicit_args is not None or not getattr(cls, "__flow_model_use_context_args__", True):
                if cls._cached_context_validator is None:
                    if cls._context_td is not None:
                        cls._cached_context_validator = TypeAdapter(cls._context_td)
                    elif cls._context_schema:
                        td = TypedDict(f"{cls.__name__}Inputs", cls._context_schema)
                        cls._cached_context_validator = TypeAdapter(td)
                    else:
                        cls._cached_context_validator = TypeAdapter(cls.__flow_model_context_type__)
                return cls._cached_context_validator

            # Dynamic mode: build schema from unbound fields (instance-specific)
            # Cache on instance since bound_fields varies per instance
            if not hasattr(self, "_instance_context_validator"):
                all_param_types = getattr(cls, "__flow_model_all_param_types__", {})
                bound_fields = getattr(self, "_bound_fields", set())
                unbound_schema = {name: typ for name, typ in all_param_types.items() if name not in bound_fields}
                if unbound_schema:
                    td = TypedDict(f"{cls.__name__}Inputs", unbound_schema)
                    object.__setattr__(self, "_instance_context_validator", TypeAdapter(td))
                else:
                    # No unbound fields - empty validator
                    object.__setattr__(self, "_instance_context_validator", TypeAdapter(dict))
            return self._instance_context_validator

        GeneratedModel._get_context_validator = _get_context_validator

        # Override context_type property after class creation
        @property
        def context_type_getter(self) -> Type[ContextBase]:
            return self.__class__.__flow_model_context_type__

        # Override result_type property after class creation
        @property
        def result_type_getter(self) -> Type[ResultBase]:
            return self.__class__.__flow_model_return_type__

        # Add .flow property for the new API
        @property
        def flow_getter(self) -> FlowAPI:
            return FlowAPI(self)

        GeneratedModel.context_type = context_type_getter
        GeneratedModel.result_type = result_type_getter
        GeneratedModel.flow = flow_getter

        # Register the MODEL class for serialization (needed for model_dump/_target_).
        # Note: We do NOT register dynamic context classes anymore - context handling
        # uses FlowContext + TypedDict instead, which don't need registration.
        from .local_persistence import register_ccflow_import_path

        register_ccflow_import_path(GeneratedModel)

        # Rebuild the model to process annotations properly
        GeneratedModel.model_rebuild()

        # Create factory function that returns model instances
        @wraps(fn)
        def factory(**kwargs) -> GeneratedModel:
            instance = GeneratedModel(**kwargs)
            # Track which fields were explicitly provided at construction
            # These are "bound" - everything else comes from context at runtime
            object.__setattr__(instance, "_bound_fields", set(kwargs.keys()))
            return instance

        # Preserve useful attributes on factory
        factory._generated_model = GeneratedModel
        factory.__doc__ = fn.__doc__

        # Add .deps decorator for customizing __deps__
        def deps_decorator(deps_fn):
            """Decorator to customize the __deps__ method.

            Usage:
                @Flow.model
                def my_func(start_date: date, prices: dict) -> GenericResult[...]:
                    ...

                @my_func.deps
                def _(self, context):
                    # Custom context transform
                    lookback_ctx = FlowContext(
                        start_date=context.start_date - timedelta(days=30),
                        end_date=context.end_date,
                    )
                    return [(self.prices, [lookback_ctx])]
            """
            from .callable import GraphDepList

            # Rename the function to __deps__ so Flow.deps accepts it
            deps_fn.__name__ = "__deps__"
            deps_fn.__qualname__ = f"{GeneratedModel.__qualname__}.__deps__"
            # Set proper signature to match __call__'s context type
            deps_fn.__signature__ = inspect.Signature(
                parameters=[
                    inspect.Parameter("self", inspect.Parameter.POSITIONAL_OR_KEYWORD),
                    inspect.Parameter("context", inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=context_type),
                ],
                return_annotation=GraphDepList,
            )
            # Wrap with Flow.deps and replace on the class
            decorated = Flow.deps(deps_fn)
            GeneratedModel.__deps__ = decorated
            # Mark that this model has custom deps (so _resolve_deps_and_call will call it)
            GeneratedModel.__has_custom_deps__ = True
            return factory  # Return factory for chaining

        factory.deps = deps_decorator

        return factory

    # Handle both @Flow.model and @Flow.model(...) syntax
    if func is not None:
        return decorator(func)
    return decorator
