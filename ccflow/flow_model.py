"""Flow.model decorator implementation.

This module provides the Flow.model decorator that generates CallableModel classes
from plain Python functions, reducing boilerplate while maintaining full compatibility
with existing ccflow infrastructure.
"""

import inspect
import logging
from functools import wraps
from typing import Annotated, Any, Callable, Dict, List, Optional, Tuple, Type, Union, get_origin

from pydantic import Field

from .base import ContextBase, ResultBase
from .dep import Dep, extract_dep
from .local_persistence import register_ccflow_import_path

__all__ = ("flow_model",)

log = logging.getLogger(__name__)


def _infer_context_type_from_args(context_args: List[str], func: Callable, sig: inspect.Signature) -> Type[ContextBase]:
    """Infer or create a context type from context_args parameter names.

    This attempts to match existing context types or creates a new one.

    Args:
        context_args: List of parameter names that come from context
        func: The decorated function
        sig: The function signature

    Returns:
        A ContextBase subclass
    """
    from .local_persistence import create_ccflow_model

    # Build field definitions for the context from parameter annotations
    fields = {}
    for name in context_args:
        if name not in sig.parameters:
            raise ValueError(f"context_arg '{name}' not found in function parameters")
        param = sig.parameters[name]
        if param.annotation is inspect.Parameter.empty:
            raise ValueError(f"context_arg '{name}' must have a type annotation")
        default = ... if param.default is inspect.Parameter.empty else param.default
        fields[name] = (param.annotation, default)

    # Try to match common context types
    from .context import DateRangeContext

    # Check for DateRangeContext pattern
    if set(context_args) == {"start_date", "end_date"}:
        from datetime import date

        if all(
            sig.parameters[name].annotation in (date, "date")
            or (isinstance(sig.parameters[name].annotation, type) and sig.parameters[name].annotation is date)
            for name in context_args
        ):
            return DateRangeContext

    # Create a new context type dynamically
    context_class = create_ccflow_model(
        f"_{func.__name__}_Context",
        __base__=ContextBase,
        **fields,
    )
    return context_class


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

        # Determine context mode and extract info
        if context_args is not None:
            # Mode 2: Unpacked context args
            context_type = _infer_context_type_from_args(context_args, fn, sig)
            model_field_params = {name: param for name, param in params.items() if name not in context_args and name != "self"}
            use_context_args = True
        elif "context" in params or "_" in params:
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
        else:
            raise TypeError(f"Function {fn.__name__} must either have a 'context' (or '_') parameter or specify context_args in the decorator")

        # Analyze parameters to find dependencies and regular fields
        dep_fields: Dict[str, Tuple[Type, Dep]] = {}  # name -> (base_type, Dep)
        model_fields: Dict[str, Tuple[Type, Any]] = {}  # name -> (type, default)

        for name, param in model_field_params.items():
            if param.annotation is inspect.Parameter.empty:
                raise TypeError(f"Parameter '{name}' must have a type annotation")

            base_type, dep = _get_dep_info(param.annotation)
            default = ... if param.default is inspect.Parameter.empty else param.default

            if dep is not None:
                # This is a dependency parameter
                dep_fields[name] = (base_type, dep)
                # Use Annotated so _resolve_deps_and_call in callable.py can find the Dep
                # This consolidates resolution logic into one place
                model_fields[name] = (Annotated[Union[base_type, CallableModel], dep], default)
            else:
                # Regular model field
                model_fields[name] = (param.annotation, default)

        # Capture context_args in local variable for closures
        ctx_args_list = context_args or []
        # Capture context parameter name for closures (only used in mode 1)
        ctx_param_name = context_param_name if not use_context_args else "context"

        # Create the __call__ method
        def make_call_impl():
            # Import resolve here to avoid circular import at module level
            from .callable import resolve

            def __call__(self, context):
                # Build kwargs for the original function
                if use_context_args:
                    # Unpack context into args
                    fn_kwargs = {name: getattr(context, name) for name in ctx_args_list}
                else:
                    # Pass context directly (using actual parameter name: 'context' or '_')
                    fn_kwargs = {ctx_param_name: context}

                # Add model fields - use resolve() for dep fields to get resolved values
                for name in model_fields:
                    value = getattr(self, name)
                    if name in dep_fields:
                        # Use resolve() to get the resolved value from context var
                        fn_kwargs[name] = resolve(value)
                    else:
                        fn_kwargs[name] = value

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
                for dep_name, (base_type, dep_obj) in dep_fields.items():
                    value = getattr(self, dep_name)
                    if isinstance(value, CallableModel):
                        transformed_ctx = dep_obj.apply(context)
                        deps.append((value, [transformed_ctx]))
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
        GeneratedModel.__flow_model_context_args__ = ctx_args_list

        # Override context_type property after class creation
        @property
        def context_type_getter(self) -> Type[ContextBase]:
            return self.__class__.__flow_model_context_type__

        # Override result_type property after class creation
        @property
        def result_type_getter(self) -> Type[ResultBase]:
            return self.__class__.__flow_model_return_type__

        GeneratedModel.context_type = context_type_getter
        GeneratedModel.result_type = result_type_getter

        # Register for serialization (local classes need this)
        register_ccflow_import_path(GeneratedModel)

        # Rebuild the model to process annotations properly
        GeneratedModel.model_rebuild()

        # Create factory function that returns model instances
        @wraps(fn)
        def factory(**kwargs) -> GeneratedModel:
            return GeneratedModel(**kwargs)

        # Preserve useful attributes on factory
        factory._generated_model = GeneratedModel
        factory.__doc__ = fn.__doc__

        return factory

    # Handle both @Flow.model and @Flow.model(...) syntax
    if func is not None:
        return decorator(func)
    return decorator
