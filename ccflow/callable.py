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
from functools import lru_cache, wraps
from inspect import Signature, isclass, signature
from typing import Any, ClassVar, Dict, Generic, List, Optional, Tuple, Type, TypeVar, Union, get_args, get_origin

from pydantic import BaseModel as PydanticBaseModel, ConfigDict, Field, InstanceOf, PrivateAttr, TypeAdapter, field_validator, model_validator
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


@lru_cache
def _cached_signature(fn):
    return signature(fn)


class MetaData(BaseModel):
    """Class to represent metadata for all callable models"""

    name: str = ""
    description: str = Field("", repr=False)
    options: Optional["FlowOptions"] = Field(None, exclude=True, repr=False)  # noqa F405


class _CallableModel(BaseModel, abc.ABC):
    """Generic base class for Callable Models.

    The purpose of this class is to provide type definitions of context_type and return_type.
    """

    model_config = ConfigDict(
        ignored_types=(property,),
    )
    meta: MetaData = Field(default_factory=MetaData)

    @model_validator(mode="after")
    def _check_signature(self):
        sig_call = _cached_signature(self.__class__.__call__)
        if len(sig_call.parameters) != 2 or "context" not in sig_call.parameters:  # ("self", "context")
            raise ValueError("__call__ method must take a single argument, named 'context'")

        sig_deps = _cached_signature(self.__class__.__deps__)
        if len(sig_deps.parameters) != 2 or "context" not in sig_deps.parameters:
            raise ValueError("__deps__ method must take a single argument, named 'context'")

        if self.__class__.__deps__ is not CallableModel.__deps__:
            type_call_arg = _cached_signature(self.__class__.__call__).parameters["context"].annotation
            type_deps_arg = _cached_signature(self.__class__.__deps__).parameters["context"].annotation
            if type_call_arg is not type_deps_arg:
                err_msg_type_mismatch = (
                    f"The type of the context accepted by __deps__ {type_deps_arg} must match that accepted by __call__ {type_call_arg}"
                )
                raise ValueError(err_msg_type_mismatch)

        # If context_type or result_type are overridden, ensure they match the signature
        if hasattr(self, "context_type"):
            type_call_arg = _cached_signature(self.__class__.__call__).parameters["context"].annotation

            # If optional type, extract inner type
            if get_origin(type_call_arg) is Optional or (get_origin(type_call_arg) is Union and type(None) in get_args(type_call_arg)):
                type_call_arg = [t for t in get_args(type_call_arg) if t is not type(None)][0]

            if (
                not isinstance(type_call_arg, TypeVar)
                and type_call_arg is not Signature.empty
                and (not isclass(type_call_arg) or not issubclass(type_call_arg, self.context_type))
            ):
                err_msg_type_mismatch = (
                    f"The context_type {self.context_type} must match the type of the context accepted by __call__ {type_call_arg}"
                )
                raise ValueError(err_msg_type_mismatch)

        if hasattr(self, "result_type"):
            type_call_return = _cached_signature(self.__class__.__call__).return_annotation
            if (
                not isinstance(type_call_return, TypeVar)
                and type_call_return is not Signature.empty
                and (not isclass(type_call_return) or not issubclass(type_call_return, self.result_type))
            ):
                err_msg_type_mismatch = f"The result_type {self.result_type} must match the return type of __call__ {type_call_return}"
                raise ValueError(err_msg_type_mismatch)

        return self

    @abc.abstractmethod
    def __call__(self, context: ContextType) -> ResultType:
        """This method produces the result for the given context.

        Instead of passing the context, one can pass an object that pydantic will try to validate as the context.
        Additionally, if kwargs are passed instead of the context, it will use these to construct the context.
        """

    @abc.abstractmethod
    def __deps__(
        self,
        context: ContextType,
    ) -> "GraphDepList":
        """
        Overwrite this method to specify dependencies of this `CallableModel` that can then be used for parallelization
        of the implicit `CallableModel` graph. The 'call graph' of a `CallableModel` is implicitly defined by the calls
        made in the `__call__` function of a `CallableModel` to other `CallableModel`s. Since these dependencies can
        only be discovered at runtime, it is given as an option to the user that they specify a `CallableModel`s
        upstream dependencies in this function.

        Implementations should be decorated with Flow.call.
        """


CallableModelType = TypeVar("CallableModelType", bound=_CallableModel)

# Since we only want to check the types, not revalidate the models, we use InstanceOf here
GraphDepType = Tuple[InstanceOf[_CallableModel], List[InstanceOf[ContextBase]]]  # noqa: F405
GraphDepList = List[GraphDepType]

# *****************************************************************************
# Define the "Flow" framework, including the decorator and its options
# *****************************************************************************


@lru_cache
def _get_logging_evaluator(log_level):
    from .evaluators import LoggingEvaluator  # Import locally to prevent circular dependency

    return LoggingEvaluator(log_level=log_level)


class FlowOptions(BaseModel):
    """Options for Flow evaluation.

    This class is typically used by exporting it to a dict with exclude_unset=True, such that only fields that have been
    explicitly passed by the user will be used for overriding. This allows default behavior to be separately defined
    (i.e. by an evaluator) if the user has not explicitly specified a field.
    """

    model_config: ConfigDict = {"frozen": True}

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
    evaluator: Optional[InstanceOf["EvaluatorBase"]] = Field(None, description="A hook to set a custom evaluator")
    _deps: bool = PrivateAttr(False)
    _parse_log_level = field_validator("log_level", mode="before")(str_to_log_level)

    def get_options(self, model: CallableModelType):
        """Gets the options with overrides applied."""
        return FlowOptionsOverride.get_options(model, self)

    def _get_evaluator_from_options(self, options: "FlowOptions") -> "EvaluatorBase":
        if options.evaluator:
            return options.evaluator

        return _get_logging_evaluator(log_level=options.log_level)

    def get_evaluator(self, model: CallableModelType) -> "EvaluatorBase":
        """Gets the implementation of the evaluator."""
        # We need to make sure this gets called from inside each wrapper,
        # otherwise, global changes to Flow.options will not be picked up.
        options = FlowOptionsOverride.get_options(model, self)
        return self._get_evaluator_from_options(options)

    def __call__(self, fn):
        # Used for building a graph of model evaluation contexts without evaluating
        def get_evaluation_context(model: CallableModelType, context: ContextType, as_dict: bool = False, *, _options: Optional[FlowOptions] = None):
            # Create the evaluation context.
            # Record the options that are used, in case the evaluators want to use it,
            # but exclude the evaluator itself to avoid potential circular dependencies
            # or other difficulties with serialization/caching of the options
            if _options:
                if not isinstance(_options, FlowOptions):
                    _options = FlowOptions.model_validate(_options)
                options = _options
            else:
                options = FlowOptionsOverride.get_options(model, self)
            evaluator = self._get_evaluator_from_options(options)
            options_dict = options.model_dump(mode="python", exclude={"evaluator"}, exclude_unset=True)
            evaluation_context = ModelEvaluationContext(model=model, context=context, fn=fn.__name__, options=options_dict)
            if as_dict:
                return dict(model=evaluator, context=evaluation_context)
            else:
                return ModelEvaluationContext(model=evaluator, context=evaluation_context)

        # The decorator implementation
        def wrapper(model, context=Signature.empty, *, _options: Optional[FlowOptions] = None, **kwargs):
            if not isinstance(model, CallableModel):
                raise TypeError(f"Can only decorate methods on CallableModels (not {type(model)}) with the flow decorator.")
            if (not isclass(model.context_type) or not issubclass(model.context_type, ContextBase)) and not (
                get_origin(model.context_type) is Union and type(None) in get_args(model.context_type)
            ):
                raise TypeError(f"Context type {model.context_type} must be a subclass of ContextBase")
            if not isclass(model.result_type) or not issubclass(model.result_type, ResultBase):
                raise TypeError(f"Result type {model.result_type} must be a subclass of ResultBase")
            if self._deps and fn.__name__ != "__deps__":
                raise ValueError("Can only apply Flow.deps decorator to __deps__")
            if context is Signature.empty:
                context = _cached_signature(fn).parameters["context"].default
                if context is Signature.empty:
                    if kwargs:
                        context = kwargs
                    else:
                        raise TypeError(
                            f"{fn.__name__}() missing 1 required positional argument: 'context' of type {model.context_type}, or kwargs to construct it"
                        )
            elif kwargs:  # Kwargs passed in as well as context. Not allowed
                raise TypeError(f"{fn.__name__}() was passed a context and got an unexpected keyword argument '{next(iter(kwargs.keys()))}'")

            # Type coercion on input. We do this here (rather than relying on ModelEvaluationContext) as it produces a nicer traceback/error message
            if not isinstance(context, model.context_type):
                if get_origin(model.context_type) is Union and type(None) in get_args(model.context_type):
                    model_context_type = [t for t in get_args(model.context_type) if t is not type(None)][0]
                else:
                    model_context_type = model.context_type
                context = model_context_type.model_validate(context)

            if fn != getattr(model.__class__, fn.__name__).__wrapped__:
                # This happens when super().__call__ is used when implementing a CallableModel that derives from another one.
                # In this case, we don't apply the decorator again, we just call the function on the model and context.
                return fn(model, context)

            evaluation_context = get_evaluation_context(model, context, as_dict=True, _options=_options)
            # Here we call the evaluator directly on the context, instead of calling evaluation_context()
            # to eliminate one level in the call stack when things go wrong/when debugging.
            result = evaluation_context["model"](evaluation_context["context"])
            return result

        wrap = wraps(fn)(wrapper)
        wrap.get_evaluator = self.get_evaluator
        wrap.get_options = self.get_options
        wrap.get_evaluation_context = get_evaluation_context
        return wrap


class FlowOptionsDeps(FlowOptions):
    """Flow options for dependency evaluation"""

    _deps: bool = PrivateAttr(True)


class FlowOptionsOverride(BaseModel):
    """This python context helps the registry track dependencies of underlying calls to the registry.

    Do not confuse the name with "Context" from callable.py.
    """

    model_config = ConfigDict(protected_namespaces=())  # Because of model_types field

    _OPEN_OVERRIDES: ClassVar[Dict] = {}
    options: FlowOptions = Field(description="The options that represent the overrides to apply in this context")
    models: Tuple[CallableModelType, ...] = Field((), description="Which specific model instances to apply the overrides to")
    model_types: Tuple[Type[CallableModelType], ...] = Field((), description="Which specific model types to apply the overrides to")

    @classmethod
    def _apply_options(cls, old: FlowOptions, new: FlowOptions) -> FlowOptions:
        return old.model_copy(update={f: getattr(new, f) for f in new.model_fields_set})

    @classmethod
    def get_options(cls, model: CallableModelType, model_options: Optional[FlowOptions] = None) -> FlowOptions:
        """Return a set of options with overrides applied.

        Options are applied in the following order:
            Defaults of FlowOptions
            Globally-scoped option overrides from FlowOptionsOverride
            Overrides from 'model_options' input
            Overrides specified on model MetaData (i.e. model.meta)
            Model and model type scopes option overrides from FlowOptionsOverride

        Args:
            model: The model to apply options from/to
            model_options: Additional options to inject in the overrides
        """
        current_options = _FLOW_OPTIONS.model_copy()
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
        if model.meta.options:
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
    model: InstanceOf[_CallableModel]
    context: Union[InstanceOf[ContextBase], None]

    # Using InstanceOf instead of the actual type will limit Pydantic's validation of the field to instance checking
    # Otherwise, the validation will re-run fully despite the models already being validated on construction
    # TODO: Make the instance check compatible with the generic types instead of the base type

    @model_validator(mode="wrap")
    def _context_validator(cls, values, handler, info):
        """Override _context_validator from parent"""

        # Validate the context with the model, if possible
        model = values.get("model")
        if model and isinstance(model, CallableModel) and not isinstance(values.get("context"), model.context_type):
            values["context"] = model.context_type.model_validate(values.get("context"))

        # Apply standard pydantic validation
        context = handler(values)

        # Check that the function is correctly specified
        if not hasattr(context.model, context.fn):
            raise ValueError(f"Class {type(context.model)} does not have a function {context.fn} to call")
        return context

    def __call__(self) -> ResultType:
        fn = getattr(self.model, self.fn)
        if hasattr(fn, "__wrapped__"):
            result = fn.__wrapped__(self.model, self.context)
            # If it's a callable model, then we can validate the result
            if self.options.get("validate_result", True):
                if fn.__name__ == "__deps__":
                    result = _GraphDepListAdapter.validate_python(result)
                # If we validate a delayed result, we will force evaluation.
                # Instead, we can flag that validation is requested, and have it done after evaluation
                elif hasattr(result, "_lazy_is_delayed"):
                    object.__setattr__(result, "_lazy_validation_requested", True)
                elif hasattr(self.model, "result_type"):
                    result = self.model.result_type.model_validate(result)

            return result
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


class Evaluator(EvaluatorBase):
    """A higher-order model that evaluates a function on a CallableModel and a Context."""

    @override
    def __call__(self, context: ModelEvaluationContext) -> ResultType:
        return context()


_GraphDepListAdapter = TypeAdapter(GraphDepList)
# Sort out all forward ref issues
FlowOptions.model_rebuild()
FlowOptionsDeps.model_rebuild()
MetaData.model_rebuild()
# Define default FlowOptions prototype
_FLOW_OPTIONS = FlowOptions()

# *****************************************************************************
# Define actual CallableModel and associated types
# *****************************************************************************


class CallableModel(_CallableModel):
    """Generic base class for Callable Models, with a default implementation for __deps__."""

    @model_validator(mode="after")
    def _check_decorator(self):
        call = self.__call__
        if not hasattr(call, "__wrapped__") and getattr(call, "__name__", "") != "Flow.call":
            raise ValueError("__call__ function of CallableModel must be wrapped with the Flow.call decorator")

        if not hasattr(self.__deps__, "__wrapped__") and getattr(self.__deps__, "__name__", "") != "Flow.deps":
            raise ValueError("__deps__ function of CallableModel must be wrapped with the Flow.deps decorator")
        return self

    @property
    def context_type(self) -> Type[ContextType]:
        """Return the context type for the model.

        By default, it reads the value from the function signature (if a concrete value is provided),
        otherwise the implementation needs to be overridden.
        """
        typ = _cached_signature(self.__class__.__call__).parameters["context"].annotation
        if typ is Signature.empty:
            raise TypeError("Must either define a type annotation for context on __call__ or implement 'context_type'")

        # If optional type, extract inner type
        if get_origin(typ) is Optional or (get_origin(typ) is Union and type(None) in get_args(typ)):
            typ_to_check = [t for t in get_args(typ) if t is not type(None)][0]
        else:
            typ_to_check = typ

        # Ensure subclass of ContextBase
        if not isclass(typ_to_check) or not issubclass(typ_to_check, ContextBase):
            raise TypeError(f"Context type declared in signature of __call__ must be a subclass of ContextBase. Received {typ_to_check}.")
        return typ

    @property
    def result_type(self) -> Type[ResultType]:
        """Return the result type for the model.

        By default, it reads the value from the function signature (if a concrete value is provided),
        otherwise the implementation needs to be overridden.
        """
        typ = _cached_signature(self.__class__.__call__).return_annotation
        if typ is Signature.empty:
            raise TypeError("Must either define a return type annotation on __call__ or implement 'result_type'")
        if not isclass(typ) or not issubclass(typ, ResultBase):
            raise TypeError(f"Return type declared in signature of __call__ must be a subclass of ResultBase (i.e. GenericResult). Received {typ}.")
        return typ

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
    """Abstract class that represents a wrapper around an underlying model, with the same context and return types.

    It reduces the amount of boilerplate required. Multi-model composites require their own implementation.
    """

    model: CallableModelType

    @property
    def context_type(self) -> Type[ContextType]:
        """Return the context type of the underlying model."""
        return self.model.context_type

    @property
    def result_type(self) -> Type[ResultType]:
        """Return the result type of the underlying model."""
        return self.model.result_type


class CallableModelGenericType(CallableModel, Generic[ContextType, ResultType]):
    """Special type of callable model that provides context and result via
    a generic type instead of annotations on __call__.
    """

    _context_type: ClassVar[Type[ContextType]]
    _result_type: ClassVar[Type[ResultType]]

    @property
    def context_type(self) -> Type[ContextType]:
        return self._context_type

    @property
    def result_type(self) -> Type[ResultType]:
        return self._result_type

    def __setstate__(self, state):
        super().__setstate__(state)
        self._determine_context_result()

    @classmethod
    def _determine_context_result(cls):
        # Extract the generic types from the class definition
        if not hasattr(cls, "_context_type") or not hasattr(cls, "_result_type"):
            new_context_type = None
            new_result_type = None
            for base in cls.__mro__:
                if issubclass(base, CallableModelGenericType):
                    # Found the generic base class, it should
                    # have either generic parameters or context/result
                    if new_context_type is None and hasattr(base, "_context_type") and issubclass(base._context_type, ContextBase):
                        new_context_type = base._context_type
                    if new_result_type is None and hasattr(base, "_result_type") and issubclass(base._result_type, ResultBase):
                        new_result_type = base._result_type
                    if base.__pydantic_generic_metadata__["args"]:
                        if len(base.__pydantic_generic_metadata__["args"]) >= 2:
                            # Assume order is ContextType, ResultType
                            arg0, arg1 = base.__pydantic_generic_metadata__["args"][:2]
                            if new_context_type is None and isinstance(arg0, type) and issubclass(arg0, ContextBase):
                                new_context_type = arg0
                            if new_result_type is None and isinstance(arg1, type) and issubclass(arg1, ResultBase):
                                # NOTE: ContextBase inherits from ResultBase, so order matters here!
                                new_result_type = arg1
                        else:
                            for arg in base.__pydantic_generic_metadata__["args"]:
                                if new_context_type is None and isinstance(arg, type) and issubclass(arg, ContextBase):
                                    new_context_type = arg
                                elif new_result_type is None and isinstance(arg, type) and issubclass(arg, ResultBase):
                                    # NOTE: ContextBase inherits from ResultBase, so order matters here!
                                    new_result_type = arg
                    if new_context_type and new_result_type:
                        break

            if new_context_type is not None:
                # Validate that the model's context_type match
                annotation_context_type = _cached_signature(cls.__call__).parameters["context"].annotation
                if get_origin(annotation_context_type) is Optional or (
                    get_origin(annotation_context_type) is Union and type(None) in get_args(annotation_context_type)
                ):
                    annotation_context_type = [t for t in get_args(annotation_context_type) if t is not type(None)][0]
                if (
                    annotation_context_type is not Signature.empty
                    and not isinstance(annotation_context_type, TypeVar)
                    and not issubclass(annotation_context_type, new_context_type)
                ):
                    raise TypeError(
                        f"Context type annotation {annotation_context_type} on __call__ does not match context_type {new_context_type} defined by CallableModelGenericType"
                    )
                elif isclass(annotation_context_type) and issubclass(annotation_context_type, new_context_type):
                    new_context_type = annotation_context_type

                # Set on class
                cls._context_type = new_context_type

            if new_result_type is not None:
                # Validate that the model's result_type match
                annotation_result_type = _cached_signature(cls.__call__).return_annotation
                if (
                    annotation_result_type is not Signature.empty
                    and not isinstance(annotation_result_type, TypeVar)
                    and not issubclass(annotation_result_type, new_result_type)
                ):
                    raise TypeError(
                        f"Return type annotation {annotation_result_type} on __call__ does not match result_type {new_result_type} defined by CallableModelGenericType"
                    )
                elif isclass(annotation_result_type) and issubclass(annotation_result_type, new_result_type):
                    new_result_type = annotation_result_type

                # Set on class
                cls._result_type = new_result_type

    @model_validator(mode="wrap")
    def _validate_callable_model_generic_type(cls, m, handler, info):
        from ccflow.base import resolve_str

        if isinstance(m, str):
            m = resolve_str(m)

        if isinstance(m, dict):
            m = handler(m)
            cls._determine_context_result()

        # Raise ValueError (not TypeError) as per https://docs.pydantic.dev/latest/errors/errors/
        if not isinstance(m, CallableModel):
            raise ValueError(f"{m} is not a CallableModel: {type(m)}")

        subtypes = cls.__pydantic_generic_metadata__["args"]
        if subtypes:
            TypeAdapter(Type[subtypes[0]]).validate_python(m.context_type)
            TypeAdapter(Type[subtypes[1]]).validate_python(m.result_type)

        return m
