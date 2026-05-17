"""Shared signature and context-contract analysis for Flow authoring APIs.

This module also owns the portable serialization format for an already-analyzed
``@Flow.model`` contract.  That payload is used by generated-model pickle/Ray
restore and serialized context transforms so workers do not re-resolve function
annotations from the original defining scope.
"""

import inspect
from dataclasses import dataclass, field
from functools import wraps
from types import FunctionType, UnionType
from typing import Annotated, Any, Callable, Dict, Literal, NamedTuple, Optional, Tuple, Type, Union, get_args, get_origin, get_type_hints

from .base import ContextBase, ResultBase
from .context import FlowContext
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
}
_UNSET = _INTERNAL_SENTINELS["_UNSET"]
_RESERVED_FLOW_MODEL_PARAM_NAMES = frozenset({"flow", "meta", "context_type", "result_type", "type_"})


class _LazyMarker:
    def __repr__(self) -> str:
        return "Lazy"


class _FromContextMarker:
    def __repr__(self) -> str:
        return "FromContext"


class _DepMarker:
    def __repr__(self) -> str:
        return "Dep"


class FromContext:
    """Marker used in ``@Flow.model`` signatures for runtime/contextual inputs."""

    def __class_getitem__(cls, item):
        return Annotated[item, _FromContextMarker()]


class Lazy:
    """Lazy dependency marker used only as ``Lazy[T]`` in type annotations."""

    def __new__(cls, *args, **kwargs):
        raise TypeError("Lazy is an annotation marker; use Lazy[T] in @Flow.model signatures.")

    def __class_getitem__(cls, item):
        return Annotated[item, _LazyMarker()]


class Dep:
    """Marker used in ``@Flow.model`` signatures for explicit dependency leaves."""

    def __new__(cls, *args, **kwargs):
        raise TypeError("Dep is an annotation marker; use Dep[T] in @Flow.model signatures.")

    def __class_getitem__(cls, item):
        return Annotated[item, _DepMarker()]


@dataclass(frozen=True)
class _ParsedAnnotation:
    base: Any
    is_lazy: bool
    is_from_context: bool
    is_dep: bool


@dataclass(frozen=True)
class _FlowModelParam:
    name: str
    annotation: Any
    is_contextual: bool
    is_lazy: bool
    has_function_default: bool
    function_default: Any = _UNSET
    context_validation_annotation: Any = _UNSET
    has_dep_slots: bool = False

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


class _SerializedAnnotation(NamedTuple):
    kind: str
    value: Any
    args: Tuple[Any, ...] = ()


class _SerializedFlowModelParam(NamedTuple):
    name: str
    annotation: _SerializedAnnotation
    is_contextual: bool
    is_lazy: bool
    has_function_default: bool
    function_default: Any
    context_validation_annotation: _SerializedAnnotation
    has_dep_slots: bool = False


class _SerializedFlowModelConfig(NamedTuple):
    func: _AnyCallable
    return_annotation: _SerializedAnnotation
    context_type: _SerializedAnnotation
    result_type: _SerializedAnnotation
    auto_wrap_result: bool
    auto_unwrap: bool
    parameters: Tuple[_SerializedFlowModelParam, ...]
    declared_context_type: _SerializedAnnotation


def _callable_name(func: _AnyCallable) -> str:
    return getattr(func, "__name__", type(func).__name__)


def _callable_qualname(func: _AnyCallable) -> str:
    return getattr(func, "__qualname__", type(func).__qualname__)


def _clone_function_without_annotations(fn: _AnyCallable) -> _AnyCallable:
    """Return a behavior-equivalent function whose annotations are not a pickle dependency.

    ``@Flow.model`` analyzes annotations eagerly when the decorator runs and
    stores the resolved contract in ``_FlowModelConfig``.  During local/generated
    model pickling we want workers to execute the original function body, but we
    do not want them to re-evaluate or unpickle the original annotations.  Some
    runtime annotations, notably Pydantic generic specializations such as
    ``GenericResult[int]``, are valid Python objects in-process but do not have a
    durable import path for fresh-process cloudpickle restore.
    """

    if not isinstance(fn, FunctionType):
        return fn

    clone = FunctionType(fn.__code__, fn.__globals__, fn.__name__, fn.__defaults__, fn.__closure__)
    clone.__kwdefaults__ = fn.__kwdefaults__
    clone.__module__ = fn.__module__
    clone.__qualname__ = fn.__qualname__
    clone.__doc__ = fn.__doc__
    clone.__dict__.update(getattr(fn, "__dict__", {}))
    clone.__annotations__ = {}
    return clone


def _is_pydantic_generic_type(annotation: Any) -> bool:
    metadata = getattr(annotation, "__pydantic_generic_metadata__", None)
    return isinstance(annotation, type) and bool(metadata) and metadata.get("origin") is not None


def _serialize_annotation(annotation: Any) -> Any:
    """Serialize the annotation shapes that are part of a Flow.model contract.

    The goal is not to serialize arbitrary Python typing objects perfectly.  It
    is narrower and deliberate: keep the already-analyzed ``@Flow.model``
    contract portable across subprocess/Ray restore without asking workers to
    resolve annotations from the original defining scope again.

    Raw annotations are still allowed as a fallback for ordinary importable
    objects.  The explicit cases below cover annotation containers that commonly
    wrap ccflow/Pydantic model types and would otherwise embed fragile runtime
    objects directly in the cloudpickle payload.
    """

    if annotation is _UNSET or annotation is inspect.Signature.empty:
        return _SerializedAnnotation(kind="raw", value=annotation)

    if _is_pydantic_generic_type(annotation):
        metadata = annotation.__pydantic_generic_metadata__
        return _SerializedAnnotation(
            kind="pydantic_generic",
            value=_serialize_annotation(metadata["origin"]),
            args=tuple(_serialize_annotation(arg) for arg in metadata["args"]),
        )

    origin = get_origin(annotation)
    if origin is Annotated:
        args = get_args(annotation)
        return _SerializedAnnotation(kind="annotated", value=_serialize_annotation(args[0]), args=args[1:])
    if origin in _UNION_ORIGINS:
        return _SerializedAnnotation(kind="union", value=tuple(_serialize_annotation(arg) for arg in get_args(annotation)))
    if origin is Literal:
        return _SerializedAnnotation(kind="literal", value=get_args(annotation))
    if origin is not None:
        return _SerializedAnnotation(
            kind="generic_alias",
            value=_serialize_annotation(origin),
            args=tuple(_serialize_annotation(arg) for arg in get_args(annotation)),
        )

    return _SerializedAnnotation(kind="raw", value=annotation)


def _restore_annotation(payload: Any) -> Any:
    """Restore an annotation payload produced by ``_serialize_annotation``."""

    if not isinstance(payload, _SerializedAnnotation):
        raise TypeError(f"Unknown serialized annotation payload: {payload!r}")
    value = payload.value
    if payload.kind == "raw":
        return value
    if payload.kind == "pydantic_generic":
        origin = _restore_annotation(value)
        args = tuple(_restore_annotation(arg) for arg in payload.args)
        return origin[args[0] if len(args) == 1 else args]
    if payload.kind == "annotated":
        return Annotated[(_restore_annotation(value), *payload.args)]
    if payload.kind == "union":
        return Union[tuple(_restore_annotation(arg) for arg in value)]
    if payload.kind == "literal":
        return Literal[value]
    if payload.kind == "generic_alias":
        origin = _restore_annotation(value)
        args = tuple(_restore_annotation(arg) for arg in payload.args)
        return origin[args[0] if len(args) == 1 else args]
    raise TypeError(f"Unknown serialized annotation payload kind: {payload.kind!r}")


def _serialize_flow_model_param(param: _FlowModelParam) -> _SerializedFlowModelParam:
    return _SerializedFlowModelParam(
        name=param.name,
        annotation=_serialize_annotation(param.annotation),
        is_contextual=param.is_contextual,
        is_lazy=param.is_lazy,
        has_function_default=param.has_function_default,
        function_default=param.function_default,
        context_validation_annotation=_serialize_annotation(param.context_validation_annotation),
        has_dep_slots=param.has_dep_slots,
    )


def _restore_flow_model_param(payload: _SerializedFlowModelParam) -> _FlowModelParam:
    if not isinstance(payload, _SerializedFlowModelParam):
        raise TypeError(f"Unknown Flow.model parameter payload: {payload!r}")
    annotation = _restore_annotation(payload.annotation)
    context_validation_annotation = _restore_annotation(payload.context_validation_annotation)
    return _FlowModelParam(
        name=payload.name,
        annotation=annotation,
        is_contextual=payload.is_contextual,
        is_lazy=payload.is_lazy,
        has_function_default=payload.has_function_default,
        function_default=payload.function_default,
        context_validation_annotation=context_validation_annotation,
        has_dep_slots=getattr(payload, "has_dep_slots", _annotation_contains_dep(annotation)),
    )


def _serialize_flow_model_config(config: _FlowModelConfig) -> _SerializedFlowModelConfig:
    """Return a tagged, portable description of an analyzed Flow.model config.

    This is intentionally explicit instead of relying on ``_FlowModelConfig`` or
    ``_FlowModelParam`` pickle hooks.  The payload is the persistence boundary
    for local generated models and serialized context transforms: function
    behavior plus the resolved contract needed to rebuild the generated
    ``CallableModel`` class.  It is not a second signature-analysis path.
    """

    return _SerializedFlowModelConfig(
        func=_clone_function_without_annotations(config.func),
        return_annotation=_serialize_annotation(config.return_annotation),
        context_type=_serialize_annotation(config.context_type),
        result_type=_serialize_annotation(config.result_type),
        auto_wrap_result=config.auto_wrap_result,
        auto_unwrap=config.auto_unwrap,
        parameters=tuple(_serialize_flow_model_param(param) for param in config.parameters),
        declared_context_type=_serialize_annotation(config.declared_context_type),
    )


def _restore_flow_model_config(payload: _SerializedFlowModelConfig) -> _FlowModelConfig:
    if not isinstance(payload, _SerializedFlowModelConfig):
        raise TypeError(f"Unknown Flow.model config payload: {payload!r}")
    return _FlowModelConfig(
        func=payload.func,
        return_annotation=_restore_annotation(payload.return_annotation),
        context_type=_restore_annotation(payload.context_type),
        result_type=_restore_annotation(payload.result_type),
        auto_wrap_result=payload.auto_wrap_result,
        auto_unwrap=payload.auto_unwrap,
        parameters=tuple(_restore_flow_model_param(param) for param in payload.parameters),
        declared_context_type=_restore_annotation(payload.declared_context_type),
    )


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
    is_dep = False

    while get_origin(annotation) is Annotated:
        args = get_args(annotation)
        annotation = args[0]
        for metadata in args[1:]:
            if isinstance(metadata, _LazyMarker):
                is_lazy = True
            elif isinstance(metadata, _FromContextMarker):
                is_from_context = True
            elif isinstance(metadata, _DepMarker):
                is_dep = True

    if annotation is FromContext:
        raise TypeError("FromContext is an annotation marker; use FromContext[T] in @Flow.model signatures.")
    if annotation is Lazy:
        raise TypeError("Lazy is an annotation marker; use Lazy[T] in @Flow.model signatures.")
    if annotation is Dep:
        raise TypeError("Dep is an annotation marker; use Dep[T] in @Flow.model signatures.")

    return _ParsedAnnotation(base=annotation, is_lazy=is_lazy, is_from_context=is_from_context, is_dep=is_dep)


def _strip_annotated(annotation: Any) -> Any:
    while get_origin(annotation) is Annotated:
        annotation = get_args(annotation)[0]
    return annotation


def _pop_dep_marker(annotation: Any) -> Tuple[Any, bool]:
    """Remove only the outer Dep marker while preserving other Annotated metadata."""

    if get_origin(annotation) is not Annotated:
        return annotation, False

    args = get_args(annotation)
    metadata = tuple(item for item in args[1:] if not isinstance(item, _DepMarker))
    has_dep = len(metadata) != len(args[1:])
    base = args[0]
    if not metadata:
        return base, has_dep
    # Python 3.10 cannot spell this as ``Annotated[base, *metadata]``. Keep
    # non-Dep metadata, such as pydantic Field constraints, on the annotation
    # used to validate literals and resolved dependency results.
    return Annotated.__class_getitem__((base, *metadata)), has_dep


def _annotation_contains_dep(annotation: Any) -> bool:
    annotation, has_dep = _pop_dep_marker(annotation)
    if has_dep:
        return True
    return any(_annotation_contains_dep(arg) for arg in get_args(annotation))


def _validate_dep_annotation(annotation: Any, *, in_dep: bool = False, dep_allowed: bool = False) -> None:
    """Validate the deliberately small Dep marker language.

    Dep marks exact substitution slots. It is allowed inside container values,
    but not inside another Dep and not in dict keys.
    """

    annotation, has_dep = _pop_dep_marker(annotation)
    if has_dep:
        if not dep_allowed:
            raise TypeError("Dep[...] is only supported in regular parameter container values.")
        if in_dep:
            raise TypeError("Dep[...] cannot contain another Dep[...] marker.")
        _validate_dep_annotation(annotation, in_dep=True, dep_allowed=False)
        return

    origin = get_origin(annotation)
    args = get_args(annotation)
    if origin is list and len(args) == 1:
        _validate_dep_annotation(args[0], in_dep=in_dep, dep_allowed=True)
        return
    if origin is tuple and args:
        item_args = args[:1] if len(args) == 2 and args[1] is Ellipsis else args
        for arg in item_args:
            _validate_dep_annotation(arg, in_dep=in_dep, dep_allowed=True)
        return
    if origin is dict and len(args) == 2:
        key_annotation, value_annotation = args
        if _annotation_contains_dep(key_annotation):
            raise TypeError("Dep[...] is not supported in dict keys.")
        _validate_dep_annotation(value_annotation, in_dep=in_dep, dep_allowed=True)
        return

    for arg in args:
        _validate_dep_annotation(arg, in_dep=in_dep, dep_allowed=False)


def _is_result_annotation(annotation: Any) -> bool:
    annotation = _strip_annotated(annotation)
    origin = get_origin(annotation) or annotation
    return isinstance(origin, type) and issubclass(origin, ResultBase)


def _result_union_members(annotation: Any) -> Tuple[Any, ...]:
    annotation = _strip_annotated(annotation)
    if get_origin(annotation) not in _UNION_ORIGINS:
        return ()
    return tuple(arg for arg in get_args(annotation) if arg is not type(None))


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
        if (parsed.is_dep or _annotation_contains_dep(parsed.base)) and (parsed.is_lazy or parsed.is_from_context):
            marker = "Lazy" if parsed.is_lazy else "FromContext"
            raise TypeError(f"Parameter '{param.name}' cannot combine Dep[...] and {marker}[...].")
        if parsed.is_dep:
            raise TypeError("Dep[...] is only supported in regular parameter container values.")
        _validate_dep_annotation(parsed.base)
        has_dep_slots = _annotation_contains_dep(parsed.base)
        has_default = param.default is not inspect.Parameter.empty
        if parsed.is_lazy and has_default and not is_model_dependency(param.default):
            raise TypeError(f"Parameter '{param.name}' is marked Lazy[...] and must default to a CallableModel dependency.")
        if parsed.is_from_context and has_default and is_model_dependency(param.default):
            raise TypeError(f"Parameter '{param.name}' is marked FromContext[...] and cannot default to a CallableModel.")

        analyzed_params.append(
            _FlowModelParam(
                name=param.name,
                annotation=parsed.base,
                is_contextual=parsed.is_from_context,
                is_lazy=parsed.is_lazy,
                has_function_default=has_default,
                function_default=param.default if has_default else _UNSET,
                has_dep_slots=has_dep_slots,
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
    reserved = sorted(param.name for param in parameters if param.name in _RESERVED_FLOW_MODEL_PARAM_NAMES)
    if reserved:
        names = ", ".join(repr(name) for name in reserved)
        raise TypeError(f"Parameter name(s) {names} are reserved for generated model framework attributes.")

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
                    is_contextual=param.is_contextual,
                    is_lazy=param.is_lazy,
                    has_function_default=param.has_function_default,
                    function_default=param.function_default,
                    context_validation_annotation=context_fields[param.name].annotation,
                )
            )
        parameters = tuple(updated_params)

    return_annotation = _strip_annotated(sig.return_annotation)
    union_result_members = _result_union_members(return_annotation)
    if union_result_members and any(_is_result_annotation(arg) for arg in union_result_members):
        raise TypeError(
            "@Flow.model does not support Union or Optional ResultBase return annotations. "
            "Return one concrete ResultBase subclass, or return an ordinary value and let @Flow.model wrap it."
        )

    auto_wrap_result = not _is_result_annotation(return_annotation)
    result_type = GenericResult[return_annotation] if auto_wrap_result else return_annotation

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
