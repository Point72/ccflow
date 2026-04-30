"""Shared signature and context-contract analysis for Flow authoring APIs."""

import inspect
from dataclasses import dataclass, field
from functools import wraps
from types import UnionType
from typing import Annotated, Any, Callable, Dict, Literal, Optional, Tuple, Type, Union, get_args, get_origin, get_type_hints

from .base import ContextBase, ResultBase
from .context import FlowContext
from .exttypes import PyObjectPath
from .local_persistence import create_ccflow_model
from .result import GenericResult

_AnyCallable = Callable[..., Any]
_UNION_ORIGINS = (Union, UnionType)


class _InternalSentinel:
    def __init__(self, name: str):
        self._name = name

    def __repr__(self) -> str:
        return self._name

    def __reduce__(self):
        return (_get_internal_sentinel, (self._name,))


def _get_internal_sentinel(name: str) -> _InternalSentinel:
    return _INTERNAL_SENTINELS[name]


_INTERNAL_SENTINELS = {
    "_UNSET": _InternalSentinel("_UNSET"),
    "_REMOVED_CONTEXT_ARGS": _InternalSentinel("_REMOVED_CONTEXT_ARGS"),
}
_UNSET = _INTERNAL_SENTINELS["_UNSET"]
_REMOVED_CONTEXT_ARGS = _INTERNAL_SENTINELS["_REMOVED_CONTEXT_ARGS"]


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

    @property
    def context_input_types(self) -> Dict[str, Any]:
        return {param.name: param.validation_annotation for param in self.contextual_params}

    @property
    def context_required_names(self) -> Tuple[str, ...]:
        return tuple(param.name for param in self.contextual_params if not param.has_function_default)

    def param(self, name: str) -> _FlowModelParam:
        return self._params_by_name[name]


@dataclass(frozen=True)
class _AutoContextSpec:
    signature: inspect.Signature
    base_class: Type[ContextBase]
    class_name: str
    fields: Dict[str, Tuple[Any, Any]]


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


def _strip_annotated(annotation: Any) -> Any:
    while get_origin(annotation) is Annotated:
        annotation = get_args(annotation)[0]
    return annotation


def _is_result_annotation(annotation: Any) -> bool:
    origin = get_origin(annotation) or annotation
    if isinstance(origin, type) and issubclass(origin, ResultBase):
        return True

    if get_origin(annotation) in _UNION_ORIGINS:
        args = tuple(arg for arg in get_args(annotation) if arg is not type(None))
        return bool(args) and all(_is_result_annotation(arg) for arg in args)

    return False


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
        raw_func_args = get_args(func_annotation)
        func_accepts_none = type(None) in raw_func_args
        func_args = tuple(arg for arg in raw_func_args if arg is not type(None))
        if context_origin in _UNION_ORIGINS:
            raw_context_args = get_args(context_annotation)
            if type(None) in raw_context_args and not func_accepts_none:
                return False
            context_args = tuple(arg for arg in raw_context_args if arg is not type(None))
            if not context_args:
                return func_accepts_none
            return bool(context_args) and all(
                any(_context_type_annotations_compatible(func_arg, context_arg) for func_arg in func_args) for context_arg in context_args
            )
        if context_annotation is type(None):
            return func_accepts_none
        return any(_context_type_annotations_compatible(func_arg, context_annotation) for func_arg in func_args)

    if context_origin in _UNION_ORIGINS:
        raw_context_args = get_args(context_annotation)
        if type(None) in raw_context_args:
            return False
        context_args = tuple(arg for arg in raw_context_args if arg is not type(None))
        return bool(context_args) and all(_context_type_annotations_compatible(func_annotation, context_arg) for context_arg in context_args)

    if func_origin is Literal and context_origin is Literal:
        return set(get_args(context_annotation)).issubset(set(get_args(func_annotation)))
    if func_origin is Literal:
        return False
    if context_origin is Literal:
        literal_values = get_args(context_annotation)
        func_base = func_origin or func_annotation
        if isinstance(func_base, type):
            return all(isinstance(value, func_base) for value in literal_values)
        return False

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


def _analyze_flow_function(
    fn: _AnyCallable,
    sig: inspect.Signature,
    *,
    is_model_dependency: Callable[[Any], bool],
) -> Tuple[_FlowModelParam, ...]:
    analyzed_params = []

    for param in sig.parameters.values():
        parsed = _parse_annotation(param.annotation)
        if parsed.is_lazy and parsed.is_from_context:
            raise TypeError(f"Parameter '{param.name}' cannot combine Lazy[...] and FromContext[...].")
        has_default = param.default is not inspect.Parameter.empty
        if parsed.is_lazy and has_default and not is_model_dependency(param.default):
            raise TypeError(f"Parameter '{param.name}' is marked Lazy[...] and must default to a CallableModel dependency.")
        if parsed.is_from_context and has_default and is_model_dependency(param.default):
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

    return tuple(analyzed_params)


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
    *,
    context_type: Optional[Type[ContextBase]],
    auto_unwrap: bool,
    is_model_dependency: Callable[[Any], bool],
) -> _FlowModelConfig:
    parameters = _analyze_flow_function(fn, sig, is_model_dependency=is_model_dependency)
    contextual_params = tuple(param for param in parameters if param.is_contextual)
    declared_context_type = None
    if context_type is not None and not contextual_params:
        raise TypeError("context_type=... requires FromContext[...] parameters.")
    if context_type is not None:
        declared_context_type = _validate_declared_context_type(context_type, contextual_params)

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

    auto_wrap_result = not _is_result_annotation(sig.return_annotation)
    result_type = GenericResult[sig.return_annotation] if auto_wrap_result else sig.return_annotation

    return _FlowModelConfig(
        func=fn,
        return_annotation=sig.return_annotation,
        context_type=FlowContext,
        result_type=result_type,
        auto_wrap_result=auto_wrap_result,
        auto_unwrap=auto_unwrap,
        parameters=parameters,
        declared_context_type=declared_context_type,
    )


def _analyze_flow_context_transform(
    fn: _AnyCallable,
    sig: inspect.Signature,
    *,
    is_model_dependency: Callable[[Any], bool],
) -> _FlowModelConfig:
    parameters = _analyze_flow_function(fn, sig, is_model_dependency=is_model_dependency)
    lazy_params = [param.name for param in parameters if param.is_lazy]
    if lazy_params:
        raise TypeError(f"Flow.context_transform does not support Lazy[...] parameter(s): {', '.join(lazy_params)}")
    return _FlowModelConfig(
        func=fn,
        return_annotation=sig.return_annotation,
        context_type=FlowContext,
        result_type=GenericResult,
        auto_wrap_result=False,
        auto_unwrap=False,
        parameters=parameters,
        path=PyObjectPath(f"{getattr(fn, '__module__', __name__)}.{_callable_name(fn)}"),
    )


def _analyze_auto_context_function(
    func: _AnyCallable,
    *,
    parent: Optional[Type[ContextBase]],
    resolved_hints: Dict[str, Any],
    is_model_dependency: Callable[[Any], bool],
) -> _AutoContextSpec:
    sig = _resolved_flow_signature(
        func,
        resolved_hints=resolved_hints,
        skip_self=True,
        require_return_annotation=True,
        annotation_error_suffix=" when auto_context=True",
        return_error_suffix=" when auto_context=True",
        function_name=_callable_qualname(func),
    )
    base_class = parent or ContextBase

    if parent is not None:
        parent_fields = set(parent.model_fields.keys()) - set(ContextBase.model_fields.keys())
        sig_params = set(sig.parameters)
        missing = parent_fields - sig_params
        if missing:
            raise TypeError(f"Parent context fields {missing} must be included in function signature")

        for fname in parent_fields:
            parent_annotation = parent.model_fields[fname].annotation
            func_annotation = sig.parameters[fname].annotation
            if func_annotation is inspect.Parameter.empty:
                continue
            if not _context_type_annotations_compatible(func_annotation, parent_annotation):
                raise TypeError(
                    f"auto_context field '{fname}' has annotation {func_annotation!r} which is incompatible "
                    f"with parent field annotation {parent_annotation!r}"
                )

    lazy_params = [name for name, param in sig.parameters.items() if _parse_annotation(param.annotation).is_lazy]
    if lazy_params:
        raise TypeError(f"Flow.call(auto_context=...) does not support Lazy[...] parameter(s): {', '.join(lazy_params)}")

    model_defaults = [
        name for name, param in sig.parameters.items() if param.default is not inspect.Parameter.empty and is_model_dependency(param.default)
    ]
    if model_defaults:
        raise TypeError(f"Flow.call(auto_context=...) parameters cannot default to CallableModel dependencies: {', '.join(model_defaults)}")

    fields = {}
    parent_model_fields = {} if parent is None else parent.model_fields
    for name, param in sig.parameters.items():
        if name in parent_model_fields:
            field_info = parent_model_fields[name]
            fields[name] = (field_info.annotation, field_info)
            continue
        fields[name] = (param.annotation, ... if param.default is inspect.Parameter.empty else param.default)
    return _AutoContextSpec(
        signature=sig,
        base_class=base_class,
        class_name=f"{_callable_qualname(func)}_AutoContext",
        fields=fields,
    )


def _normalize_auto_context_parent(auto_context: Any) -> Type[ContextBase]:
    if auto_context is True:
        return ContextBase
    if inspect.isclass(auto_context) and issubclass(auto_context, ContextBase):
        return auto_context
    raise TypeError(f"auto_context must be False, True, or a ContextBase subclass, got {auto_context!r}")


def _wrap_auto_context_call(
    func: _AnyCallable,
    *,
    parent: Type[ContextBase],
    is_model_dependency: Callable[[Any], bool],
) -> _AnyCallable:
    resolved_hints = get_type_hints(func, include_extras=True)
    spec = _analyze_auto_context_function(
        func,
        parent=parent,
        resolved_hints=resolved_hints,
        is_model_dependency=is_model_dependency,
    )

    auto_context_class = create_ccflow_model(spec.class_name, __base__=spec.base_class, **spec.fields)

    @wraps(func)
    def wrapper(self, context):
        fn_kwargs = {name: getattr(context, name) for name in spec.fields}
        return func(self, **fn_kwargs)

    context_default = inspect.Signature.empty
    if all(not field.is_required() for field in auto_context_class.model_fields.values()):
        context_default = auto_context_class()

    wrapper.__signature__ = inspect.Signature(
        parameters=[
            inspect.Parameter("self", inspect.Parameter.POSITIONAL_OR_KEYWORD),
            inspect.Parameter("context", inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=auto_context_class, default=context_default),
        ],
        return_annotation=spec.signature.return_annotation,
    )
    wrapper.__auto_context__ = auto_context_class
    return wrapper
