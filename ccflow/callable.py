"""This module defines the "Callable Model" framework.

This framework helps define fairly generic calculations with strongly typed,
validated components that can be combined together (as well as registered and
configured using the tools in ccflow module)

In addition to the CallableModelBase class, we define the Flow decorator, which allows us to inject additional
functionality, and the Evaluator interface, which lets us control how the models are evaluated.

This module is particularly large, but hard to break up due to the interdependence between the components,
which all need to be defined together so that pydantic (especially V1) can resolve all the forward references.
"""

import abc
import logging
from functools import wraps
from inspect import Signature, isclass, signature
from typing import Any, ClassVar, Dict, Generic, List, Optional, Tuple, Type, TypeVar

from pydantic import BaseModel as PydanticBaseModel, Field, PrivateAttr, TypeAdapter, field_validator, model_validator, root_validator
from typing_extensions import override

from .base import (
    BaseModel,
    ContextBase,
    ContextType,  # noqa: F401
    ResultBase,
    ResultType,
)
from .validators import str_to_log_level

__all__ = (
    "GraphDepType",
    "GraphDepList",
    "MetaData",
    "CallableModel",
    "CallableModelType",
    "CallableModelGenericType",
    "Flow",
    "FlowOptions",
    "FlowOptionsDeps",
    "FlowOptionsOverride",
    "ModelEvaluationContext",
    "EvaluatorBase",
    "Evaluator",
    "WrapperModel",
)

log = logging.getLogger(__name__)


# *****************************************************************************
# Base CallableModel definitions, before introducing the Flow decorator or
# any evaluators
# *****************************************************************************

# TODO: Revisit these types, align with ModelAndContext and CallableModelGraph
GraphDepType = Tuple["CallableModelType", List[ContextType]]  # noqa: F405
GraphDepList = List[GraphDepType]


class MetaData(BaseModel):
    """Class to represent metadata for all callable models"""

    name: str = ""
    description: str = Field("", repr=False)
    options: Optional["FlowOptions"] = Field(None, exclude=True, repr=False)  # noqa F405


class _CallableModel(BaseModel, abc.ABC):
    """Generic base class for Callable Models.

    The purpose of this class is to provide type definitions of context_type and return_type.
    """

    meta: MetaData = Field(default_factory=MetaData)

    class Config(BaseModel.Config):
        ignored_types = (property,)
        # Whether to validate that the decorator has been applied to __call__
        validate_decorator: bool = True

    @root_validator(skip_on_failure=True)
    def _check_decorator(cls, values):
        call = cls.__call__
        if cls.Config.validate_decorator and not hasattr(call, "__wrapped__") and getattr(call, "__name__", "") != "Flow.call":
            raise ValueError("__call__ function of CallableModel must be wrapped with the Flow.call decorator")

        if cls.Config.validate_decorator and not hasattr(cls.__deps__, "__wrapped__") and getattr(cls.__deps__, "__name__", "") != "Flow.deps":
            raise ValueError("__deps__ function of CallableModel must be wrapped with the Flow.deps decorator")
        return values

    @root_validator(skip_on_failure=True)
    def _check_signature(cls, values):
        sig_call = signature(cls.__call__)
        if len(sig_call.parameters) != 2 or "context" not in sig_call.parameters:  # ("self", "context")
            raise ValueError("__call__ method must take a single argument, named 'context'")

        sig_deps = signature(cls.__deps__)
        if len(sig_deps.parameters) != 2 or "context" not in sig_deps.parameters:
            raise ValueError("__deps__ method must take a single argument, named 'context'")

        if cls.__deps__ is not CallableModel.__deps__:
            type_call_arg = signature(cls.__call__).parameters["context"].annotation
            type_deps_arg = signature(cls.__deps__).parameters["context"].annotation
            err_msg_type_mismatch = (
                f"The type of the context accepted by __deps__ {type_deps_arg} " f"must match that accepted by __call__ {type_call_arg}!"
            )
            if type_call_arg is not type_deps_arg:
                raise ValueError(err_msg_type_mismatch)

        return values

    @property
    def context_type(self) -> Type[ContextType]:
        """This property returns the context type for the model.

        By default, it reads the value from the function signature (if a concrete value is provided),
        otherwise the implementation needs to be overridden.
        """
        typ = signature(self).parameters["context"].annotation
        if typ is Signature.empty:
            raise TypeError("Must either define a type annotation for context on __call__ or implement 'context_type'")
        if not issubclass(typ, ContextBase):
            raise TypeError("Context type declared in signature of __call__ must be a subclass of ContextBase")

        return typ

    @property
    def result_type(self) -> Type[ResultType]:
        """This property returns the result type for the model.

        By default, it reads the value from the function signature (if a concrete value is provided),
        otherwise the implementation needs to be overridden.
        """
        typ = signature(self).return_annotation
        if typ is Signature.empty:
            raise TypeError("Must either define a return type annotation on __call__ or implement 'result_type'")
        if not issubclass(typ, ResultBase):
            raise TypeError("Return type declared in signature of __call__ must be a subclass of ResultBase (i.e. GenericResult)")
        return typ

    @abc.abstractmethod
    def __call__(self, context: ContextType) -> ResultType:
        """This method produces the result. Implementations should be decorated with Flow.call."""

    @abc.abstractmethod
    def __deps__(
        self,
        context: ContextType,
    ) -> GraphDepList:
        """
        Overwrite this method to specify dependencies of this `CallableModel` that can then be used for parallelization
        of the implicit `CallableModel` graph. The 'call graph' of a `CallableModel` is implicitly defined by the calls
        made in the `__call__` function of a `CallableModel` to other `CallableModel`s. Since these dependencies can
        only be discovered at runtime, it is given as an option to the user that they specify a `CallableModel`s
        upstream dependencies in this function.

        Implementations should be decorated with Flow.call.
        """


CallableModelType = TypeVar("CallableModelType", bound=_CallableModel)

# *****************************************************************************
# Define the "Flow" framework, including the decorator and its options
# *****************************************************************************


class FlowOptions(BaseModel):
    """Options for Flow evaluation.

    This class is typically used by exporting it to a dict with exclude_unset=True, such that only fields that have been
    explicitly passed by the user will be used for overriding. This allows default behavior to be separately defined
    (i.e. by an evaluator) if the user has not explicitly specified a field.
    """

    log_level: int = Field(
        logging.DEBUG,
        description="If no 'evaluator' is set, will use a LoggingEvaluator with this log level",
    )
    verbose: bool = Field(
        True,
        description="Whether to use verbose logging",
    )
    validate_result: bool = Field(
        True,
        description="Whether to validate the result to the model's result_type before returning",
    )
    volatile: bool = Field(
        False,
        description="Whether this function is volatile (i.e. always returns a different value), and hence should always be excluded from caching",
    )
    cacheable: bool = Field(
        False, description="Whether the model results should be cached if possible. This is False by default so that caching is opt-in"
    )
    evaluator: Optional["EvaluatorBase"] = Field(None, description="A hook to set a custom evaluator")
    _deps: bool = PrivateAttr(False)
    _parse_log_level = field_validator("log_level", mode="before")(str_to_log_level)

    def get_options(self, model: CallableModelType):
        """Gets the options with overrides applied."""
        return FlowOptionsOverride.get_options(model, self)

    def get_evaluator(self, model: CallableModelType) -> "EvaluatorBase":
        """Gets the implementation of the evaluator."""
        # We need to make sure this gets called from inside each wrapper,
        # otherwise, global changes to Flow.options will not be picked up.
        options = FlowOptionsOverride.get_options(model, self)
        if options.evaluator is not None:
            return options.evaluator
        from .evaluators import LoggingEvaluator

        return LoggingEvaluator(log_level=options.log_level)

    def __call__(self, fn):
        def wrapper(model, context=Signature.empty):
            from .callable import CallableModel
            from .evaluator import ModelEvaluationContext

            # TODO: Let ModelEvaluationContext handle this type checking
            if not isinstance(model, CallableModel):
                raise TypeError("Can only decorate methods on CallableModels with the flow decorator")
            if not isclass(model.context_type) or not issubclass(model.context_type, ContextBase):
                raise TypeError(f"Context type {model.context_type} must be a subclass of ContextBase")
            if not isclass(model.result_type) or not issubclass(model.result_type, ResultBase):
                raise TypeError(f"Result type {model.result_type} must be a subclass of ResultBase")
            if context is Signature.empty:
                context = signature(fn).parameters["context"].default
                if context is Signature.empty:
                    raise TypeError(f"{fn.__name__}() missing 1 required positional argument: 'context'")
            # Type coercion on input  TODO: Move to ModelEvaluationContext
            context = model.context_type.model_validate(context)
            # Create the evaluation context.
            # Record the options that are used, in case the evaluators want to use it,
            # but exclude the evaluator itself to avoid potential circular dependencies
            # or other difficulties with serialization/caching of the options
            options = FlowOptionsOverride.get_options(model, self).model_dump(mode="python", exclude={"evaluator"}, exclude_unset=True)
            if fn != getattr(model.__class__, fn.__name__).__wrapped__:
                # This happens when super().__call__ is used when implementing a CallableModel that derives from another one.
                # In this case, we don't apply the decorator again, we just call the function on the model and context.
                return fn(model, context)
            evaluation_context = ModelEvaluationContext(model=model, context=context, fn=fn.__name__, options=options)
            evaluator = self.get_evaluator(model)
            result = evaluator(evaluation_context)

            # TODO: Move into the __call__ function of ModelEvaluationContext
            if options.get("validate_result", True):
                if self._deps:
                    if fn.__name__ != "__deps__":
                        raise ValueError("Can only apply Flow.deps decorator to __deps__")
                    result = TypeAdapter(GraphDepList).validate_python(result)
                # If we validate a delayed result, we will force evaluation.
                # Instead, we can flag that validation is requested, and have it done after evaluation
                elif hasattr(result, "_lazy_is_delayed"):
                    object.__setattr__(result, "_lazy_validation_requested", True)
                else:
                    result = model.result_type.model_validate(result)
            return result

        wrap = wraps(fn)(wrapper)
        wrap.get_evaluator = self.get_evaluator
        wrap.get_options = self.get_options

        # Used for building a graph of model evaluation contexts without evaluating
        def get_evaluation_context(model: CallableModelType, context: ContextType):
            # TODO: This logic is duplicative of the logic in wrapper - combine into a single place
            options = FlowOptionsOverride.get_options(model, self).model_dump(mode="python", exclude={"evaluator"}, exclude_unset=True)
            evaluation_context = ModelEvaluationContext(model=model, context=context, fn=fn.__name__, options=options)
            evaluator = self.get_evaluator(model)
            return ModelEvaluationContext(model=evaluator, context=evaluation_context)

        wrap.get_evaluation_context = get_evaluation_context
        return wrap


class FlowOptionsDeps(FlowOptions):
    """Flow options for dependency evaluation"""

    _deps: bool = PrivateAttr(True)


class FlowOptionsOverride(BaseModel):
    """This python context helps the registry track dependencies of underlying calls to the registry.

    Do not confuse the name with "Context" from callable.py.
    """

    model_config = {"protected_namespaces": ()}  # Because of model_types field

    _OPEN_OVERRIDES: ClassVar[Dict] = {}
    options: FlowOptions = Field(description="The options that represent the overrides to apply in this context")
    models: Tuple[CallableModelType, ...] = Field((), description="Which specific model instances to apply the overrides to")
    model_types: Tuple[Type[CallableModelType], ...] = Field((), description="Which specific model types to apply the overrides to")

    @classmethod
    def _apply_options(cls, old: FlowOptions, new: FlowOptions) -> FlowOptions:
        return old.model_copy(update={f: v for f, v in new if f in new.model_fields_set})

    @classmethod
    def get_options(cls, model: CallableModelType, model_options: Optional[FlowOptions] = None) -> FlowOptions:
        """Return a set of options with overrides applied."""
        current_options = FlowOptions()
        for override in cls._OPEN_OVERRIDES.values():  # noqa: F402
            # Apply global options first
            if not override.models and not override.model_types:
                current_options = cls._apply_options(current_options, override.options)
        # Then apply the decorator-provided model-level options (because they always take precedence over global)
        # The order has to be this way so that if a user flags a model explicitly as i.e. cacheable=False in either
        # the decorator or the MetaData, then that will always be obeyed, even if the "global" setting is to set
        # cacheable=True for all models.
        # However, because _apply_options uses exclude_unset=True, any model that's not explicitly set to True/False
        # at the decorator MetaData level will then pick up the global setting.
        if model_options:
            current_options = cls._apply_options(current_options, model_options)
        # Then apply the model meta-provided options
        if model.meta.options is not None:
            current_options = model.meta.options
        # Then apply all model-specific overrides
        for override in cls._OPEN_OVERRIDES.values():
            if any(model is m for m in override.models) or isinstance(model, override.model_types):
                current_options = cls._apply_options(current_options, override.options)
        return current_options

    def __enter__(self):
        override_id = id(self)
        if override_id in FlowOptionsOverride._OPEN_OVERRIDES:
            raise ValueError(f"{self} has already been entered.")
        FlowOptionsOverride._OPEN_OVERRIDES[override_id] = self
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        override_id = id(self)
        del FlowOptionsOverride._OPEN_OVERRIDES[override_id]


class Flow(PydanticBaseModel):
    @staticmethod
    def call(*args, **kwargs):
        """Decorator for methods on callable models"""
        if len(args) == 1 and callable(args[0]):
            # No arguments to decorator, this is the decorator
            fn = args[0]
            impl = FlowOptions()
            return wraps(fn)(impl(fn))
        else:
            # Arguments to decorator, this is just returning the decorator
            # Note that the code below is executed only once
            return FlowOptions(**kwargs)

    @staticmethod
    def deps(*args, **kwargs):
        """Decorator for the __deps__ method on callable models"""
        if len(args) == 1 and callable(args[0]):
            # No arguments to decorator, this is the decorator
            fn = args[0]
            if fn.__name__ != "__deps__":
                raise ValueError("Can only apply Flow.deps decorator to __deps__")
            impl = FlowOptionsDeps()
            return wraps(fn)(impl(fn))
        else:
            # Arguments to decorator, this is just returning the decorator
            # Note that the code below is executed only once
            return FlowOptionsDeps(**kwargs)


# *****************************************************************************
# Define "Evaluators" and associated types
# Evaluators are basically Callable Models that operate on a context made up of
# the underlying model and context
# ******************************************************************************


class ModelAndContext(ContextBase, Generic[CallableModelType, ContextType]):
    """A context that holds both a model and an underlying context, for higher-order models."""

    model: CallableModelType
    context: ContextType


class ModelEvaluationContext(
    ModelAndContext[CallableModelType, ContextType],
    Generic[CallableModelType, ContextType],
):
    """An extension of ModelAndContext which also takes a function "f" to apply to both the model and the context.

    This is used for decorator construction.
    """

    fn: str = Field("__call__", strict=True)
    options: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="wrap")
    def _context_validator(cls, values, handler, info):
        # Override context validator from parent - no funky coercion stuff here,
        # and no deep copy.
        context = handler(values)
        if not hasattr(context.model, context.fn):
            raise ValueError(f"Class {type(context.model)} does not have a function {context.fn} to call")
        return context

    def __call__(self) -> ResultType:
        fn = getattr(self.model, self.fn)
        if hasattr(fn, "__wrapped__"):
            return fn.__wrapped__(self.model, self.context)
        else:
            return fn(self.context)


class EvaluatorBase(_CallableModel, abc.ABC):
    """Base class for evaluators, which are higher-order models that evaluate ModelAndContext.

    Note that evaluators don't use the Flow decorator on __call__ and __deps__ by design.
    """

    @abc.abstractmethod
    def __call__(self, context: ModelEvaluationContext) -> ResultType:
        pass

    def __deps__(self, context: ModelEvaluationContext) -> GraphDepList:
        """The __deps__ method on an evaluator will just evaluate the __deps__ function on the underlying context.model
        in the same way that the evaluator evaluates __call__
        """
        deps_context = ModelEvaluationContext(model=context.model, context=context.context, fn="__deps__", options=context.options)
        return self(deps_context)

    def __exit__(self):
        pass

    class Config(_CallableModel.Config):
        validate_decorator = False  # Evaluators don't use the decorator by design.


class Evaluator(EvaluatorBase):
    """A higher-order model that evaluates a function on a CallableModel and a Context."""

    @override
    def __call__(self, context: ModelEvaluationContext) -> ResultType:
        return context()


# Sort out all forward ref issues
FlowOptions.model_rebuild()
FlowOptionsDeps.model_rebuild()
MetaData.model_rebuild()


# *****************************************************************************
# Define actual CallableModel and associated types
# *****************************************************************************


class CallableModel(_CallableModel):
    """Generic base class for Callable Models, with a default implementation for __deps__."""

    @Flow.deps
    def __deps__(
        self,
        context: ContextType,
    ) -> GraphDepList:
        """
        Overwrite this method to specify dependencies of this `CallableModel` that can then be used for parallelization
        of the implicit `CallableModel` graph. The 'call graph' of a `CallableModel` is implicitly defined by the calls
        made in the `__call__` function of a `CallableModel` to other `CallableModel`s. Since these dependencies can
        only be discovered at runtime, it is given as an option to the user that they specify a `CallableModel`s
        upstream dependencies in this function.
        """
        return []


class WrapperModel(CallableModel, Generic[CallableModelType], abc.ABC):
    """Abstract class that represents a wrapper around an underlying model,
    with the same context and return types.

    It reduces the amount of boilerplate required.
    Multi-model composites require their own implementation.
    """

    model: CallableModelType

    @property
    def context_type(self) -> Type[ContextType]:
        return self.model.context_type

    @property
    def result_type(self) -> Type[ResultType]:
        return self.model.result_type


class CallableModelGenericType(CallableModel, Generic[ContextType, ResultType]):
    """Special type of callable model to use for type declarations, such that the
    context and result type will be validated.
    """

    @model_validator(mode="wrap")
    def _validate_callable_model_generic_type(cls, m, handler, info):
        from ccflow.base import resolve_str

        if isinstance(m, str):
            m = resolve_str(m)
        # Raise ValueError (not TypeError) as per https://docs.pydantic.dev/latest/errors/errors/
        if not isinstance(m, CallableModel):
            raise ValueError(f"{m} is not a CallableModel: {type(m)}")
        subtypes = cls.__pydantic_generic_metadata__["args"]
        if subtypes:
            TypeAdapter(Type[subtypes[0]]).validate_python(m.context_type)
            TypeAdapter(Type[subtypes[1]]).validate_python(m.result_type)
        return m
