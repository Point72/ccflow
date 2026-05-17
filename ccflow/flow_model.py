"""Generated ``@Flow.model`` implementation.

This module is intentionally the owner of the generated-model API.  The public
surface is small:

* ``@Flow.model`` turns a typed Python function into a ``CallableModel`` factory.
* ``FromContext[T]`` marks function parameters that should come from runtime
  context instead of model construction.
* ``Lazy[T]`` marks a dependency that should be passed as a thunk and evaluated
  only if user code calls it.
* ``Dep[T]`` marks a nested regular-parameter slot that can be a literal ``T``
  or a ``CallableModel`` dependency returning ``T``.
* ``model.flow.compute(...)`` and ``model.flow.with_context(...)`` provide the
  ergonomic execution and contextual binding API.

The implementation has four moving parts that should stay conceptually
separate:

* Signature analysis lives in ``_flow_model_binding.py`` so ``callable.py`` does
  not become a generated-model implementation module.
* Runtime context construction turns arbitrary context objects/kwargs into the
  narrow context shape required by a target model.
* Context bindings are represented as serializable specs, then applied at
  execution time before the wrapped model validates its context.
* Effective identity describes the parts of generated and bound model
  invocations that are known before evaluation so cache/graph keys can ignore
  unused ambient ``FlowContext`` fields.

Two invariants matter more than cleverness here:

* Existing ``CallableModel`` behavior must remain structural unless a generated
  or bound model explicitly opts into effective identity.
* Public ``cache_key(...)`` stays structural by default; evaluators and graph
  construction can request effective identity explicitly for opt-in models.

File layout:

1. Internal data structures and small value helpers.
2. Type coercion, result unwrapping, and registry-reference helpers.
3. Context-transform and generated-model serialization helpers.
4. Runtime context contracts and dependency context projection.
5. Contextual value resolution and effective identity.
6. ``with_context`` binding validation/application.
7. ``model.flow`` APIs and ``BoundModel``.
8. Generated model class construction and decorators.
"""

import importlib
import inspect
import sys
from abc import update_abstractmethods
from base64 import b64decode, b64encode
from collections import OrderedDict
from functools import lru_cache, wraps
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
    Set,
    Tuple,
    Type,
    Union,
    cast,
    get_args,
    get_origin,
    get_type_hints,
)

import cloudpickle
from pydantic import (
    BaseModel as PydanticModel,
    Field,
    PrivateAttr,
    SkipValidation,
    TypeAdapter,
    ValidationError,
    create_model,
    model_serializer,
    model_validator,
)
from pydantic.errors import PydanticSchemaGenerationError, PydanticUndefinedAnnotation

from ._flow_model_binding import (
    _UNION_ORIGINS,
    _UNSET,
    Dep,
    FromContext,
    Lazy,
    _analyze_flow_context_transform,
    _analyze_flow_model,
    _callable_name,
    _FlowModelConfig,
    _FlowModelParam,
    _pop_dep_marker,
    _resolved_flow_signature,
    _restore_flow_model_config,
    _serialize_flow_model_config,
    _strip_annotated,
)
from .base import BaseModel, ContextBase, ContextType, ResultBase
from .callable import CallableModel, EvaluationDependency, Flow, FlowOptions, GraphDepList, WrapperModel
from .context import FlowContext
from .exttypes import PyObjectPath
from .local_persistence import register_ccflow_import_path
from .result import GenericResult
from .utils.tokenize import compute_behavior_token, compute_data_token

__all__ = (
    "FlowAPI",
    "BoundModel",
    "FlowInspection",
    "InputSpec",
    "Dep",
    "FromContext",
    "Lazy",
)

_AnyCallable = Callable[..., Any]


# ---------------------------------------------------------------------------
# Internal data structures
# ---------------------------------------------------------------------------


class _UnsetFlowInput:
    def __repr__(self) -> str:
        return "<unset>"

    def __reduce__(self):
        return (_unset_flow_input_factory, ())


_UNSET_FLOW_INPUT = _UnsetFlowInput()
_FIELD_EXCLUDE_IF_SUPPORTED = "exclude_if" in inspect.signature(Field).parameters
_TYPE_ADAPTER_CACHE_MAXSIZE = 256
_HASHABLE_TYPE_ADAPTER_CACHE: "OrderedDict[Any, TypeAdapter]" = OrderedDict()
_UNHASHABLE_TYPE_ADAPTER_CACHE: "OrderedDict[int, Tuple[Any, TypeAdapter]]" = OrderedDict()


class InputSpec(NamedTuple):
    """Richer description of one direct function input in ``FlowInspection``."""

    name: str
    type: Any
    required: bool
    default: Any
    value: Any
    source: str

    @property
    def default_repr(self) -> str:
        """Compact representation of the declared function default."""

        return _flow_debug_value_repr(self.default)

    @property
    def value_repr(self) -> str:
        """Compact representation of the effective direct value."""

        return _flow_debug_value_repr(self.value)

    def __repr__(self) -> str:
        return (
            f"InputSpec(name={self.name!r}, type={_expected_type_repr(self.type)}, "
            f"required={self.required!r}, default={self.default_repr}, "
            f"value={self.value_repr}, source={self.source!r})"
        )


class DependencySpec(NamedTuple):
    """User-facing provenance for one direct dependency edge."""

    path: str
    model: CallableModel
    context: Optional[ContextBase] = None
    lazy: bool = False

    def __repr__(self) -> str:
        return (
            f"DependencySpec(path={self.path!r}, model={_flow_debug_value_repr(self.model)}, "
            f"context={_flow_debug_value_repr(self.context)}, lazy={self.lazy!r})"
        )


class FlowInspection(NamedTuple):
    """Structured current-level debugging summary returned by ``model.flow.inspect()``."""

    model: CallableModel
    context_inputs: Dict[str, Any]
    runtime_inputs: Dict[str, Any]
    required_inputs: Dict[str, Any]
    bound_inputs: Dict[str, Any]
    inputs: Dict[str, InputSpec]
    dependencies: Tuple[DependencySpec, ...]

    def __str__(self) -> str:
        lines = [f"FlowInspection(model={_flow_debug_model_name(self.model)})"]
        if self.inputs:
            lines.append("  inputs:")
            for spec in self.inputs.values():
                required = " required" if spec.required else ""
                lines.append(f"    {spec.name}: {_expected_type_repr(spec.type)} = {spec.value_repr} [{spec.source}{required}]")
        else:
            lines.append("  inputs: none")
        if self.context_inputs:
            context_inputs = ", ".join(f"{name}: {_expected_type_repr(annotation)}" for name, annotation in self.context_inputs.items())
        else:
            context_inputs = "none"
        lines.append(f"  contextual inputs: {context_inputs}")
        if self.runtime_inputs:
            runtime_inputs = ", ".join(f"{name}: {_expected_type_repr(annotation)}" for name, annotation in self.runtime_inputs.items())
        else:
            runtime_inputs = "none"
        lines.append(f"  runtime inputs: {runtime_inputs}")
        if self.required_inputs:
            required = ", ".join(f"{name}: {_expected_type_repr(annotation)}" for name, annotation in self.required_inputs.items())
        else:
            required = "none"
        lines.append(f"  required runtime inputs: {required}")
        lines.append(f"  bound inputs: {', '.join(self.bound_inputs) if self.bound_inputs else 'none'}")
        if self.dependencies:
            lines.append("  dependencies:")
            for dependency in self.dependencies:
                target = _flow_debug_model_name(dependency.model)
                suffix = " lazy" if dependency.lazy else ""
                context = f" context={_flow_debug_value_repr(dependency.context)}" if dependency.context is not None else ""
                lines.append(f"    {dependency.path} -> {target}{suffix}{context}")
        else:
            lines.append("  dependencies: none")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return str(self)

    def _repr_pretty_(self, printer, cycle: bool) -> None:
        printer.text("..." if cycle else str(self))


def _unset_flow_input_factory() -> _UnsetFlowInput:
    return _UNSET_FLOW_INPUT


def _is_unset_flow_input(value: Any) -> bool:
    return value is _UNSET_FLOW_INPUT


def _unset_flow_input_field_default() -> Any:
    kwargs = {"default_factory": _unset_flow_input_factory}
    if _FIELD_EXCLUDE_IF_SUPPORTED:
        kwargs["exclude_if"] = _is_unset_flow_input
    return Field(**kwargs)


def _flow_debug_value_repr(value: Any) -> str:
    if _is_unset_flow_input(value):
        return repr(value)

    bound_context_type = globals().get("_BoundModelContext")
    if bound_context_type is not None and isinstance(value, bound_context_type):
        return repr(FlowContext(**_context_values(value)))

    callable_model_type = globals().get("CallableModel")
    if callable_model_type is not None and isinstance(value, callable_model_type):
        return f"<dependency {_flow_debug_model_name(value)}>"

    return repr(value)


def _flow_debug_model_name(model: CallableModel) -> str:
    if isinstance(model, BoundModel):
        return f"{_flow_debug_model_name(model.model)}.flow.with_context(...)"
    name = model.meta.name
    cls_name = type(model).__name__
    return f"{name} ({cls_name})" if name else cls_name


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
    """Serializable binding produced by ``@Flow.context_transform``.

    Transform bindings store the analyzed transform contract directly. We avoid
    import-path restore here because decorators usually run before the module
    global points at the returned transform factory.
    """

    kind: Literal["context_transform"] = "context_transform"
    serialized_config: str
    bound_args: Dict[str, Any] = Field(default_factory=dict)


class StaticValueSpec(PydanticModel):
    """A ``with_context(field=value)`` static contextual override."""

    kind: Literal["static_value"] = "static_value"
    value: Any


_FieldOverrideSpec = Annotated[StaticValueSpec | ContextTransform, Field(discriminator="kind")]


class PatchContextOperation(PydanticModel):
    """One ordered positional context patch in a ``with_context`` chain."""

    kind: Literal["patch"] = "patch"
    binding: ContextTransform


class FieldContextOperation(PydanticModel):
    """One ordered field override in a ``with_context`` chain."""

    kind: Literal["field"] = "field"
    name: str
    spec: _FieldOverrideSpec


_ContextOperation = Annotated[PatchContextOperation | FieldContextOperation, Field(discriminator="kind")]


class _BoundContextSpec(PydanticModel):
    """Normalized, serializable representation of all context bindings."""

    operations: List[_ContextOperation] = Field(default_factory=list)


class _BoundModelContext(FlowContext):
    """Flow.call carrier for BoundModel that preserves existing context objects."""

    _base_context: Optional[ContextBase] = PrivateAttr(default=None)

    @model_validator(mode="wrap")
    @classmethod
    def _preserve_context_base(cls, value, handler, info):
        if isinstance(value, ContextBase):
            return value
        return handler(value)

    @classmethod
    def from_values(cls, values: Dict[str, Any], base_context: Optional[ContextBase] = None) -> "_BoundModelContext":
        context = cls(**values)
        context._base_context = base_context
        return context


class _DependencyIdentity(NamedTuple):
    kind: Literal["dependency"]
    evaluation: EvaluationDependency


class _LiteralIdentity(NamedTuple):
    kind: Literal["literal"]
    value: Any


class _DepMarkedIdentity(NamedTuple):
    kind: Literal["dep_marked"]
    value: Any


class _UnresolvedLazyDependencyIdentity(NamedTuple):
    kind: Literal["unresolved_lazy_dependency"]
    model_type: str
    model: Dict[str, Any]
    context_type: str
    context: Dict[str, Any]
    missing_context: Tuple[str, ...]
    missing_transform_context: Tuple[Tuple[str, Tuple[str, ...]], ...]


class _RegularInputIdentity(NamedTuple):
    kind: Literal["regular_input"]
    name: str
    lazy: bool
    payload: Any


class _GeneratedModelIdentity(NamedTuple):
    kind: Literal["generated_flow_model_v1"]
    model_type: str
    contextual_inputs: Dict[str, Any]
    regular_inputs: Tuple[_RegularInputIdentity, ...]
    model_base_fields: Dict[str, Any]


class _LocalFlowModelPicklePayload(NamedTuple):
    serialized_config: Any
    factory_kwargs: Dict[str, Any]


# ---------------------------------------------------------------------------
# Small value helpers
# ---------------------------------------------------------------------------


def _context_values(context: ContextBase) -> Dict[str, Any]:
    return dict(context)


def _context_transform_repr(transform: Any) -> str:
    if isinstance(transform, ContextTransform):
        name = _callable_name(_load_context_transform_config_from_binding(transform).func)
        if not transform.bound_args:
            return f"{name}()"
        args = ", ".join(f"{key}={value!r}" for key, value in sorted(transform.bound_args.items()))
        return f"{name}({args})"
    return repr(transform)


def _context_transform_identifier(binding: ContextTransform) -> str:
    return _callable_name(_load_context_transform_config_from_binding(binding).func)


def _is_model_dependency(value: Any) -> bool:
    # Keep this predicate in the module that can import CallableModel. The
    # binding analyzers also need it, but _flow_model_binding.py is imported by
    # callable.py and cannot import CallableModel directly without a cycle.
    return isinstance(value, CallableModel)


def _strip_outer_non_dep_annotated(annotation: Any) -> Any:
    """Strip outer Annotated layers only until an explicit Dep marker is found."""

    while get_origin(annotation) is Annotated:
        _, has_dep = _pop_dep_marker(annotation)
        if has_dep:
            return annotation
        annotation = get_args(annotation)[0]
    return annotation


def _annotation_origin_args(annotation: Any) -> Tuple[Any, Tuple[Any, ...]]:
    # Container walking cares about list/tuple/dict origins, but unrelated
    # Annotated metadata should not hide those origins.
    annotation = _strip_outer_non_dep_annotated(annotation)
    return get_origin(annotation), get_args(annotation)


def _path_name(name: str, path: Tuple[Any, ...]) -> str:
    # Use a field-like path in nested validation errors, e.g. values[0] or
    # rows['left'][2], instead of reporting every failure at the root field.
    suffix = "".join(f"[{item!r}]" if not isinstance(item, int) else f"[{item}]" for item in path)
    return f"{name}{suffix}"


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


# ---------------------------------------------------------------------------
# Type coercion, lazy thunks, and registry references
# ---------------------------------------------------------------------------


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
    except (PydanticSchemaGenerationError, PydanticUndefinedAnnotation):
        return False
    return True


def _expected_type_repr(annotation: Any) -> str:
    if get_origin(annotation) in _UNION_ORIGINS:
        return " | ".join(_expected_type_repr(arg) for arg in get_args(annotation))
    try:
        return annotation.__name__
    except AttributeError:
        return repr(annotation)


def _coerce_value(name: str, value: Any, annotation: Any, source: str) -> Any:
    if not _can_validate_type(annotation):
        return value
    try:
        return _type_adapter(annotation).validate_python(value)
    except ValidationError as exc:
        expected = _expected_type_repr(annotation)
        raise TypeError(f"{source} '{name}': expected {expected}, got {type(value).__name__} ({value!r})") from exc


def _unwrap_model_result(value: Any) -> Any:
    if isinstance(value, GenericResult):
        return value.value
    return value


def _make_lazy_thunk(value: CallableModel, context: ContextBase) -> Callable[[], Any]:
    cache: Dict[str, Any] = {}

    def thunk():
        if "result" not in cache:
            dependency_model, dependency_context = _resolved_dependency_invocation(value, context)
            cache["result"] = _unwrap_model_result(dependency_model(dependency_context))
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


def _resolve_bound_param_registry_ref(param: _FlowModelParam, value: Any) -> Any:
    """Resolve registry references for bound regular parameters.

    Generated fields use SkipValidation, so registry lookup no longer needs to
    run as a Pydantic before-validator just to beat field validation. Keeping it
    here lets the generated model's after-validator own the construction
    contract. Literal string validation gets first refusal for non-lazy
    parameters; registry aliases are fallback dependency syntax.
    """

    if not isinstance(value, str):
        return value
    if not param.is_lazy and _can_validate_type(param.annotation):
        try:
            _type_adapter(param.annotation).validate_python(value)
        except ValidationError:
            pass
        else:
            return value

    candidate = _resolve_registry_candidate(value)
    if candidate is None:
        return value
    if _registry_candidate_allowed(param.annotation, candidate):
        return candidate
    return value


def _resolve_serialized_dependency_ref(
    value: Any,
    *,
    include_target_alias: bool = False,
) -> Any:
    """Restore a direct serialized CallableModel reference."""

    def serialized_model_type(item: Dict[str, Any]) -> Optional[type]:
        marker = item.get("type_", _UNSET)
        if marker is _UNSET and include_target_alias:
            marker = item.get("_target_", _UNSET)
        if marker is _UNSET:
            return None
        try:
            candidate = marker.object if isinstance(marker, PyObjectPath) else PyObjectPath(marker).object
        except (ImportError, AttributeError, TypeError, ValueError):
            return None
        return candidate if inspect.isclass(candidate) and issubclass(candidate, CallableModel) else None

    if type(value) is not dict or serialized_model_type(value) is None:
        return value
    # ``type_`` is ccflow's default serialized-model marker. ``_target_`` is
    # also a ccflow alias, but it is Hydra's config language too, so callers
    # only enable that spelling after normal literal validation has failed.
    try:
        restored = BaseModel.model_validate(value)
    except (ValidationError, ImportError, AttributeError, TypeError, ValueError):
        return value
    return restored if _is_model_dependency(restored) else value


def _ensure_named_python_function(fn: _AnyCallable, *, decorator_name: str) -> None:
    if not inspect.isfunction(fn):
        raise TypeError(f"{decorator_name} only supports Python functions.")

    name = getattr(fn, "__name__", "")
    if name == "<lambda>":
        raise TypeError(f"{decorator_name} only supports named Python functions.")


# ---------------------------------------------------------------------------
# Context-transform serialization and generated-model persistence
# ---------------------------------------------------------------------------


def _serialize_context_transform_config(config: _FlowModelConfig) -> str:
    payload = cloudpickle.dumps(_serialize_flow_model_config(config), protocol=5)
    return b64encode(payload).decode("ascii")


@lru_cache(maxsize=None)
def _load_serialized_context_transform_config(serialized_config: str) -> _FlowModelConfig:
    try:
        payload = cloudpickle.loads(b64decode(serialized_config.encode("ascii")))
        config = _restore_flow_model_config(payload)
    except Exception as exc:
        raise TypeError("Stored context transform payload does not contain a Flow.context_transform binding.") from exc
    return config


def _load_context_transform_config_from_binding(binding: ContextTransform) -> _FlowModelConfig:
    return _load_serialized_context_transform_config(binding.serialized_config)


def clear_flow_model_caches() -> None:
    """Clear module-level caches used by Flow.model internals."""

    _HASHABLE_TYPE_ADAPTER_CACHE.clear()
    _UNHASHABLE_TYPE_ADAPTER_CACHE.clear()
    _load_serialized_context_transform_config.cache_clear()


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


def _new_local_flow_model_for_pickle(serialized_factory_payload: bytes) -> BaseModel:
    payload = cloudpickle.loads(serialized_factory_payload)
    config = _restore_flow_model_config(payload.serialized_config)
    # Do not call ``flow_model(config.func, **factory_kwargs)`` here.  That would
    # re-run worker-side type-hint resolution for local/postponed annotations.
    # The serialized config is the resolved contract from the defining process;
    # rebuild the generated class from it, then let pickle apply the third
    # reducer element through ``__setstate__``.
    factory = _build_flow_model_factory_from_config(config, payload.factory_kwargs)
    cls = cast(type[BaseModel], getattr(factory, "_generated_model"))
    return cls.__new__(cls)


def _new_generated_flow_model_for_pickle(factory_path: str) -> BaseModel:
    """Allocate a generated flow model by importing its factory function.

    This is the cross-process-safe restore path: importing the factory's module
    triggers the ``@Flow.model`` decorator, which re-creates the GeneratedModel
    class.  The reducer returns Pydantic state separately so pickle applies
    ``__setstate__`` in the outer pickle stream, preserving normal memo
    semantics for shared references, cycles, and protocol-5 buffers.
    """
    factory = _load_module_attribute_uncached(factory_path)
    generated_cls = getattr(factory, "_generated_model", None)
    if generated_cls is None:
        raise ImportError(f"Cannot restore generated flow model: '{factory_path}' does not have a _generated_model attribute.")
    return generated_cls.__new__(generated_cls)


def _is_importable_function(func: _AnyCallable) -> bool:
    """Return True if *func* is a top-level, importable named function."""
    module = getattr(func, "__module__", None)
    name = getattr(func, "__name__", None)
    qualname = getattr(func, "__qualname__", None)
    return bool(module and module != "__main__" and name and qualname and qualname == name and "<locals>" not in qualname)


def _importable_function_path(func: _AnyCallable) -> Optional[str]:
    if not _is_importable_function(func):
        return None
    return f"{func.__module__}.{func.__name__}"


def _load_module_attribute_uncached(path: str) -> Any:
    module_name, attribute_name = path.rsplit(".", 1)
    return getattr(importlib.import_module(module_name), attribute_name)


def _generated_model_factory_path_for_pickle(config: _FlowModelConfig, generated_cls: type) -> Optional[str]:
    path = _importable_function_path(config.func)
    if path is None:
        return None
    try:
        factory = _load_module_attribute_uncached(path)
    except (ImportError, AttributeError, ValueError):
        return None
    if getattr(factory, "_generated_model", None) is generated_cls:
        return path
    return None


def _module_has_factory_for_generated_class(module: Any, generated_cls: type, *, excluding: str) -> bool:
    """Return whether a module-level factory still owns ``generated_cls``.

    During ``importlib.reload()``, generated class attributes from the previous
    import can remain on the module while the new decorator is running.  Those
    stale classes should be replaced, not force a suffixed path that a clean
    process will never recreate.  If a live factory still points at the class,
    the slot is occupied by a real duplicate and should not be reclaimed.
    """

    return any(name != excluding and getattr(value, "_generated_model", None) is generated_cls for name, value in vars(module).items())


def _same_generated_function_source(existing_cls: type, config: _FlowModelConfig) -> bool:
    existing_config = getattr(existing_cls, "__flow_model_config__", None)
    if existing_config is None:
        return False
    existing_func = existing_config.func
    current_func = config.func
    existing_code = getattr(existing_func, "__code__", None)
    current_code = getattr(current_func, "__code__", None)
    filename = getattr(current_code, "co_filename", "")
    return (
        getattr(existing_func, "__module__", None) == getattr(current_func, "__module__", None)
        and getattr(existing_func, "__qualname__", None) == getattr(current_func, "__qualname__", None)
        and existing_code is not None
        and current_code is not None
        and not filename.startswith("<")
        and existing_code.co_filename == current_code.co_filename
        and existing_code.co_firstlineno == current_code.co_firstlineno
    )


def _register_generated_model_class(config: _FlowModelConfig, generated_cls: type) -> None:
    """Make generated classes importable when their factory function is importable.

    Importable module-level ``@Flow.model`` functions should serialize by a
    stable module path.  Local, nested, and ``__main__`` definitions still use
    local-persistence registration because there is no durable import path for
    their generated class.  Duplicate importable generated names are rejected
    instead of suffixed because suffixed paths are not reliably reproducible
    across ``importlib.reload()`` and clean-process config/Hydra round trips.
    """

    if _importable_function_path(config.func) is None:
        register_ccflow_import_path(generated_cls)
        return

    module_name = getattr(config.func, "__module__", None)
    module = sys.modules.get(module_name or "")
    qualname = getattr(generated_cls, "__qualname__", "")
    if module is None or not qualname or "<locals>" in qualname:
        register_ccflow_import_path(generated_cls)
        return

    obj = module
    parts = qualname.split(".")
    for part in parts[:-1]:
        obj = getattr(obj, part, None)
        if obj is None:
            register_ccflow_import_path(generated_cls)
            return

    name = parts[-1]
    existing = getattr(obj, name, None)
    if existing is None or existing is generated_cls:
        setattr(obj, name, generated_cls)
        return
    if getattr(existing, "__flow_model_config__", None) is not None and _same_generated_function_source(existing, config):
        # Reloaded modules can keep aliases to the previous factory until the
        # assignment currently being evaluated completes.  Matching the previous
        # generated class by source location lets those stale aliases be replaced
        # while still rejecting true duplicate function definitions.
        setattr(obj, name, generated_cls)
        return
    if getattr(existing, "__flow_model_config__", None) is not None and not _module_has_factory_for_generated_class(
        module, existing, excluding=_callable_name(config.func)
    ):
        # ``importlib.reload()`` can leave the previous generated class on the
        # module while the factory function is being rebound. Replacing that
        # stale class keeps the advertised path reproducible in a clean process.
        setattr(obj, name, generated_cls)
        return
    raise ValueError(
        f"Cannot register generated Flow model class at {module_name}.{'.'.join(parts)} because that path is already occupied. "
        "Use a unique function name for each importable @Flow.model factory."
    )


# ---------------------------------------------------------------------------
# Runtime context contracts and dependency projection
# ---------------------------------------------------------------------------


def _runtime_context_for_model(model: CallableModel, values: Dict[str, Any]) -> ContextBase:
    """Build the runtime context object expected by ``model`` from raw values."""

    contract = _model_context_contract(model)
    if contract.runtime_context_type is FlowContext:
        return FlowContext(**values)
    return contract.runtime_context_type.model_validate(values)


def _project_context_values_for_model(model: CallableModel, values: Dict[str, Any]) -> Dict[str, Any]:
    """Keep only the context fields a target model knows how to consume.

    Generated models and ordinary ``ContextBase`` subclasses have declared input
    fields.  ``FlowContext`` and opaque context types do not, so their context is
    forwarded unchanged.
    """

    contract = _model_context_contract(model)
    if contract.runtime_context_type is FlowContext or contract.input_types is None:
        return values
    return {name: values[name] for name in contract.input_types if name in values}


def _dependency_context_for_model(model: CallableModel, context: ContextBase) -> ContextBase:
    return _runtime_context_for_model(model, _project_context_values_for_model(model, _context_values(context)))


def _bound_model_default_base_context(bound_model: "BoundModel") -> Optional[ContextBase]:
    """Return the wrapped plain model's default context object when one exists."""

    contract = _model_context_contract(bound_model.model)
    if contract.generated_model is not None:
        return None
    default_context = _plain_model_default_context(bound_model.model)
    if default_context is _UNSET or default_context is None:
        return None
    if isinstance(default_context, ContextBase) and _context_matches_type(default_context, bound_model.model.context_type):
        return default_context
    return contract.runtime_context_type.model_validate(default_context)


def _bound_model_ambient_context(bound_model: "BoundModel", values: Dict[str, Any]) -> _BoundModelContext:
    """Return the ambient carrier a bound wrapper should rewrite."""

    base_context = _bound_model_default_base_context(bound_model)
    if base_context is not None:
        ambient = _context_values(base_context)
        ambient.update(values)
        return _BoundModelContext.from_values(ambient, base_context=base_context)
    return _BoundModelContext.from_values(values)


def _resolved_dependency_invocation(value: CallableModel, context: ContextBase) -> Tuple[CallableModel, ContextBase]:
    """Return the concrete ``(model, context)`` pair for a dependency call.

    Bound models must receive the full ambient ``FlowContext`` so their binding
    transforms can read source fields before narrowing to the wrapped model's.
    If the wrapper targets a handwritten model with a decorated default context,
    dependency execution uses the same default baseline as ``bound.flow.compute``.
    Unbound dependencies can be projected immediately.
    """

    if isinstance(value, BoundModel):
        values = {} if context is None else _context_values(context)
        return value, _bound_model_ambient_context(value, values)
    return value, _dependency_context_for_model(value, context)


def _effective_context_operations(context_spec: _BoundContextSpec) -> Tuple[_ContextOperation, ...]:
    """Drop field operations overwritten by later field operations.

    Patches are kept conservative because their write keys may be dynamic.  Field
    operations have explicit targets, so an earlier field transform for ``a``
    should not run or require inputs if a later field binding overwrites ``a``.
    """

    seen_fields: Set[str] = set()
    operations: List[_ContextOperation] = []
    for operation in reversed(context_spec.operations):
        if isinstance(operation, FieldContextOperation):
            if operation.name in seen_fields:
                continue
            seen_fields.add(operation.name)
        operations.append(operation)
    operations.reverse()
    return tuple(operations)


def _generated_model_instance(stage: Any) -> Optional["_GeneratedFlowModelBase"]:
    model = stage.model if isinstance(stage, BoundModel) else stage
    if isinstance(model, _GeneratedFlowModelBase):
        return model
    return None


def _model_context_contract(
    model: CallableModel,
) -> _ModelContextContract:
    """Describe how ``model`` consumes runtime context.

    This is the central adapter between generated models, plain CallableModels,
    optional/opaque context annotations, and ``FlowContext``.  Callers use it to
    decide which fields are contextual inputs, which are required, and which
    concrete ``ContextBase`` subclass should validate runtime values.
    """

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


# ---------------------------------------------------------------------------
# Generated model input resolution
# ---------------------------------------------------------------------------


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
        raise TypeError(f"Parameter '{param.name}' is marked Lazy[...] and must be bound to a CallableModel dependency.")

    if _is_model_dependency(value):
        dependency_model, dependency_context = _resolved_dependency_invocation(value, context)
        try:
            return _unwrap_model_result(dependency_model(dependency_context))
        except Exception as exc:
            parent = _callable_name(type(model).__flow_model_config__.func)
            child = dependency_model.meta.name or type(dependency_model).__name__
            add_note = getattr(exc, "add_note", None)
            if callable(add_note):
                add_note(f"Error while evaluating dependency {parent}.{param.name} -> {child}.")
            raise
    if param.has_dep_slots:
        return _resolve_dep_marked_value(param.name, value, param.annotation, context)
    return value


def _walk_dep_marked_value(
    value: Any,
    annotation: Any,
    *,
    on_dep_slot: Callable[[Any, Any, Tuple[Any, ...]], Any],
    on_literal: Callable[[Any, Any, Tuple[Any, ...]], Any],
    on_list: Callable[[Any, List[Any], Any, Tuple[Any, ...]], Any],
    on_tuple: Callable[[Any, Tuple[Any, ...], Any, Tuple[Any, ...]], Any],
    on_dict: Callable[[Any, List[Tuple[Any, Any]], Any, Any, Any, Tuple[Any, ...]], Any],
    path: Tuple[Any, ...] = (),
) -> Any:
    """Walk only the container grammar where Dep markers are valid."""

    annotation = _strip_outer_non_dep_annotated(annotation)
    dep_base, is_dep_slot = _pop_dep_marker(annotation)
    if is_dep_slot:
        return on_dep_slot(value, dep_base, path)

    origin, args = _annotation_origin_args(annotation)
    if origin is list and isinstance(value, (list, tuple)) and args:
        items = [
            _walk_dep_marked_value(
                item,
                args[0],
                on_dep_slot=on_dep_slot,
                on_literal=on_literal,
                on_list=on_list,
                on_tuple=on_tuple,
                on_dict=on_dict,
                path=path + (index,),
            )
            for index, item in enumerate(value)
        ]
        return on_list(value, items, annotation, path)

    if origin is tuple and isinstance(value, (list, tuple)) and args:
        if len(args) == 2 and args[1] is Ellipsis:
            items = tuple(
                _walk_dep_marked_value(
                    item,
                    args[0],
                    on_dep_slot=on_dep_slot,
                    on_literal=on_literal,
                    on_list=on_list,
                    on_tuple=on_tuple,
                    on_dict=on_dict,
                    path=path + (index,),
                )
                for index, item in enumerate(value)
            )
            return on_tuple(value, items, annotation, path)
        if len(args) == len(value):
            items = tuple(
                _walk_dep_marked_value(
                    item,
                    item_annotation,
                    on_dep_slot=on_dep_slot,
                    on_literal=on_literal,
                    on_list=on_list,
                    on_tuple=on_tuple,
                    on_dict=on_dict,
                    path=path + (index,),
                )
                for index, (item, item_annotation) in enumerate(zip(value, args))
            )
            return on_tuple(value, items, annotation, path)
        return on_literal(value, annotation, path)

    if origin is dict and isinstance(value, Mapping) and len(args) == 2:
        key_annotation, value_annotation = args
        items = [
            (
                key,
                _walk_dep_marked_value(
                    item,
                    value_annotation,
                    on_dep_slot=on_dep_slot,
                    on_literal=on_literal,
                    on_list=on_list,
                    on_tuple=on_tuple,
                    on_dict=on_dict,
                    path=path + (key,),
                ),
            )
            for key, item in value.items()
        ]
        return on_dict(value, items, key_annotation, value_annotation, annotation, path)

    return on_literal(value, annotation, path)


def _resolve_dep_slot_registry_ref(value: Any, annotation: Any) -> Any:
    if not isinstance(value, str):
        return value
    candidate = _resolve_registry_candidate(value)
    if candidate is not None and _registry_candidate_allowed(annotation, candidate):
        return candidate
    return value


def _validate_dep_marked_value(name: str, value: Any, annotation: Any, source: str, path: Tuple[Any, ...] = ()) -> Any:
    """Validate construction values that use explicit nested Dep markers."""

    def validate_dep_slot(item: Any, dep_base: Any, item_path: Tuple[Any, ...]) -> Any:
        if not _is_model_dependency(item):
            item = _resolve_serialized_dependency_ref(item, include_target_alias=True)
        if _is_model_dependency(item):
            return item
        try:
            return _coerce_value(_path_name(name, item_path), item, dep_base, source)
        except TypeError:
            item_with_alias_dep = _resolve_dep_slot_registry_ref(item, dep_base)
            if item_with_alias_dep is not item and _is_model_dependency(item_with_alias_dep):
                return item_with_alias_dep
            raise

    def validate_literal(item: Any, item_annotation: Any, item_path: Tuple[Any, ...]) -> Any:
        return _coerce_value(_path_name(name, item_path), item, item_annotation, source)

    def validate_dict(
        _value: Any,
        items: List[Tuple[Any, Any]],
        key_annotation: Any,
        _value_annotation: Any,
        _dict_annotation: Any,
        dict_path: Tuple[Any, ...],
    ) -> Any:
        return {_coerce_value(_path_name(name, dict_path + (key, "key")), key, key_annotation, source): item for key, item in items}

    return _walk_dep_marked_value(
        value,
        annotation,
        on_dep_slot=validate_dep_slot,
        on_literal=validate_literal,
        on_list=lambda _value, items, _annotation, _path: list(items),
        on_tuple=lambda _value, items, _annotation, _path: tuple(items),
        on_dict=validate_dict,
        path=path,
    )


def _resolve_dep_marked_value(name: str, value: Any, annotation: Any, context: ContextBase, path: Tuple[Any, ...] = ()) -> Any:
    """Resolve CallableModel leaves that appear at explicit Dep marker slots."""

    def resolve_dep_slot(item: Any, dep_base: Any, item_path: Tuple[Any, ...]) -> Any:
        if _is_model_dependency(item):
            dependency_model, dependency_context = _resolved_dependency_invocation(item, context)
            resolved = _unwrap_model_result(dependency_model(dependency_context))
            return _coerce_value(_path_name(name, item_path), resolved, dep_base, "Regular parameter")
        return item

    return _walk_dep_marked_value(
        value,
        annotation,
        on_dep_slot=resolve_dep_slot,
        on_literal=lambda item, _annotation, _path: item,
        on_list=lambda _value, items, _annotation, _path: list(items),
        on_tuple=lambda _value, items, _annotation, _path: tuple(items),
        on_dict=lambda _value, items, _key_annotation, _value_annotation, _dict_annotation, _path: dict(items),
        path=path,
    )


def _dep_marked_dependency_entries(value: Any, annotation: Any, context: ContextBase) -> GraphDepList:
    """Collect dependency graph edges from explicit Dep marker slots."""

    def dep_slot_edges(item: Any, _dep_base: Any, _path: Tuple[Any, ...]) -> GraphDepList:
        if not _is_model_dependency(item):
            return []
        dependency_model, dependency_context = _resolved_dependency_invocation(item, context)
        return [(dependency_model, [dependency_context])]

    def merge_items(items: Any) -> GraphDepList:
        deps: GraphDepList = []
        for item in items:
            deps.extend(item)
        return deps

    return _walk_dep_marked_value(
        value,
        annotation,
        on_dep_slot=dep_slot_edges,
        on_literal=lambda _item, _annotation, _path: [],
        on_list=lambda _value, items, _annotation, _path: merge_items(items),
        on_tuple=lambda _value, items, _annotation, _path: merge_items(items),
        on_dict=lambda _value, items, _key_annotation, _value_annotation, _dict_annotation, _path: merge_items(item for _, item in items),
    )


def _dep_marked_identity_value(value: Any, annotation: Any, context: ContextBase) -> Any:
    """Return an identity payload with explicit Dep leaves replaced by dependencies."""

    def dep_slot_identity(item: Any, _dep_base: Any, _path: Tuple[Any, ...]) -> Any:
        if _is_model_dependency(item):
            return _regular_dependency_identity(item, context)
        return item

    return _walk_dep_marked_value(
        value,
        annotation,
        on_dep_slot=dep_slot_identity,
        on_literal=lambda item, _annotation, _path: item,
        on_list=lambda _value, items, _annotation, _path: list(items),
        on_tuple=lambda _value, items, _annotation, _path: tuple(items),
        on_dict=lambda _value, items, _key_annotation, _value_annotation, _dict_annotation, _path: dict(items),
    )


def _regular_dependency_identity(value: CallableModel, context: ContextBase) -> _DependencyIdentity:
    dependency_model, dependency_context = _resolved_dependency_invocation(value, context)
    return _DependencyIdentity(
        kind="dependency",
        evaluation=EvaluationDependency(dependency_model, dependency_context),
    )


def _lazy_regular_dependency_identity(value: CallableModel, context: ContextBase) -> Any:
    unresolved, dependency_model, dependency_context = _lazy_dependency_identity(value, context)
    if unresolved is not None:
        return unresolved
    assert dependency_model is not None
    assert dependency_context is not None
    return _DependencyIdentity(
        kind="dependency",
        evaluation=EvaluationDependency(dependency_model, dependency_context),
    )


def _regular_input_identity(param: _FlowModelParam, value: Any, context: ContextBase) -> _RegularInputIdentity:
    if _is_model_dependency(value):
        payload = _lazy_regular_dependency_identity(value, context) if param.is_lazy else _regular_dependency_identity(value, context)
    elif param.has_dep_slots:
        payload = _DepMarkedIdentity(kind="dep_marked", value=_dep_marked_identity_value(value, param.annotation, context))
    else:
        payload = _LiteralIdentity(kind="literal", value=value)
    return _RegularInputIdentity(kind="regular_input", name=param.name, lazy=param.is_lazy, payload=payload)


def _collect_contextual_values(
    model: "_GeneratedFlowModelBase",
    config: _FlowModelConfig,
    explicit_values: Dict[str, Any],
) -> Tuple[Dict[str, Any], List[str]]:
    """Resolve ``FromContext`` values from runtime values, model defaults, and function defaults."""

    resolved: Dict[str, Any] = {}
    missing: List[str] = []

    for param in config.contextual_params:
        if param.name in explicit_values:
            resolved[param.name] = explicit_values[param.name]
            continue

        value = getattr(model, param.name, _UNSET_FLOW_INPUT)
        if not _is_unset_flow_input(value):
            resolved[param.name] = value
            continue

        if param.has_function_default:
            resolved[param.name] = param.function_default
            continue

        missing.append(param.name)

    return resolved, missing


def _resolved_contextual_inputs(model: "_GeneratedFlowModelBase", config: _FlowModelConfig, context: ContextBase) -> Dict[str, Any]:
    """Validate and return the contextual kwargs passed to the user function."""

    resolved, missing_contextual = _collect_contextual_values(model, config, _context_values(context))

    if missing_contextual:
        missing = ", ".join(sorted(missing_contextual))
        raise TypeError(
            f"Missing contextual input(s) for {_callable_name(config.func)}: {missing}. "
            "Supply them via the runtime context, compute(), with_context(), or construction-time contextual defaults."
        )

    if config.declared_context_type is not None:
        return _validate_declared_context_values(config, resolved)

    return {
        param.name: _coerce_value(param.name, resolved[param.name], param.validation_annotation, "Context field")
        for param in config.contextual_params
    }


def _validate_declared_context_values(config: _FlowModelConfig, values: Dict[str, Any]) -> Dict[str, Any]:
    if config.declared_context_type is None:
        return values

    validated = config.declared_context_type.model_validate(values)
    return {param.name: getattr(validated, param.name) for param in config.contextual_params}


def _declared_context_field_annotation(config: _FlowModelConfig, name: str) -> Any:
    """Return a field-level annotation preserving declared context constraints."""

    assert config.declared_context_type is not None
    field_info = config.declared_context_type.model_fields[name]
    if field_info.metadata:
        return Annotated[(field_info.annotation, *field_info.metadata)]
    return field_info.annotation


def _coerce_declared_context_field(config: _FlowModelConfig, name: str, value: Any) -> Any:
    if config.declared_context_type is None:
        return _UNSET
    annotation = _declared_context_field_annotation(config, name)
    if not _can_validate_type(annotation):
        return value
    return _type_adapter(annotation).validate_python(value)


def _coerce_contextual_value(config: _FlowModelConfig, param: _FlowModelParam, value: Any, source: str) -> Any:
    declared_value = _coerce_declared_context_field(config, param.name, value)
    if declared_value is not _UNSET:
        return declared_value
    return _coerce_value(param.name, value, param.validation_annotation, source)


def _coerce_model_context_value(model: CallableModel, field_name: str, value: Any, source: str) -> Any:
    """Coerce a value for a contextual field when the target field type is known."""

    generated = _generated_model_instance(model)
    if generated is not None:
        config = type(generated).__flow_model_config__
        if field_name in config.contextual_param_names:
            return _coerce_contextual_value(config, config.param(field_name), value, source)

    contract = _model_context_contract(model)
    if contract.input_types is None or field_name not in contract.input_types:
        return value
    return _coerce_value(field_name, value, contract.input_types[field_name], source)


# ---------------------------------------------------------------------------
# Effective identity helpers
# ---------------------------------------------------------------------------

# Identity terms used below:
# - config identity: stable hash of the analyzed Flow.model contract, fixed at
#   generated-class construction time and carried through local restore.
# - behavior token: tokenizer hook for invalidating cache keys when generated
#   model behavior changes.
# - model type identity: importable factory path when available, otherwise the
#   fixed local config identity; used inside effective cache-key payloads.
# - effective invocation identity: the full context/input payload for one model
#   evaluation, with unused ambient FlowContext fields removed.


def _identity_context_values_for_model_values(model: CallableModel, values: Dict[str, Any]) -> Dict[str, Any]:
    """Project context values for identity, not execution.

    This intentionally mirrors context projection but is kept separate because
    identity cares about "what affects the result", while execution cares about
    "what context object can validate for the target model".
    """

    contract = _model_context_contract(model)
    if contract.input_types is None:
        return values
    return {name: values[name] for name in contract.input_types if name in values}


def _identity_context_values_and_missing_for_model(model: CallableModel, values: Dict[str, Any]) -> Tuple[Dict[str, Any], Tuple[str, ...]]:
    """Return identity-relevant context values and required fields still missing."""

    generated = _generated_model_instance(model)
    if generated is not None:
        config = type(generated).__flow_model_config__
        resolved, missing = _collect_contextual_values(generated, config, values)
        return (
            {param.name: resolved[param.name] for param in config.contextual_params if param.name in resolved},
            tuple(missing),
        )

    context_values = _identity_context_values_for_model_values(model, values)
    missing = tuple(name for name in _model_context_contract(model).required_names if name not in context_values)
    return context_values, missing


def _context_transform_missing_context_names(binding: ContextTransform, values: Dict[str, Any]) -> Tuple[str, ...]:
    config = _load_context_transform_config_from_binding(binding)
    return tuple(param.name for param in config.contextual_params if param.name not in values and not param.has_function_default)


def _evaluate_context_transform_from_values(binding: ContextTransform, values: Dict[str, Any]) -> Any:
    """Run a context transform against a raw value mapping.

    ``with_context`` transforms read from the original ambient context.  Chained
    bindings preserve write order, but they are not an implicit transform
    pipeline; dependent rewrites should be expressed inside one transform.
    """

    config = _load_context_transform_config_from_binding(binding)
    kwargs = _bound_context_transform_regular_kwargs(config, binding)

    for param in config.contextual_params:
        if param.name in values:
            kwargs[param.name] = _coerce_value(param.name, values[param.name], param.annotation, "Context transform field")
        elif param.has_function_default:
            kwargs[param.name] = param.function_default
        else:
            raise TypeError(
                f"Missing contextual input(s) for context transform {_callable_name(config.func)}: {param.name}. "
                "Supply them via the runtime context, a transform default, or combine dependent rewrites into one patch transform."
            )

    return config.func(**kwargs)


def _apply_context_spec_values_for_identity(
    model: CallableModel, context_spec: "_BoundContextSpec", context: ContextBase
) -> Tuple[Dict[str, Any], Tuple[Tuple[str, Tuple[str, ...]], ...]]:
    """Apply a binding spec for identity derivation.

    Unlike execution, identity must be able to describe a binding even when a
    transform cannot yet run because some source context fields are missing.  In
    that case the missing transform inputs are recorded so two unresolved lazy
    dependencies do not collapse accidentally.
    """

    original_values = _context_values(context)
    current_values = dict(original_values)
    missing_transforms: List[Tuple[str, Tuple[str, ...]]] = []

    for operation in _effective_context_operations(context_spec):
        if isinstance(operation, PatchContextOperation):
            missing = _context_transform_missing_context_names(operation.binding, original_values)
            if missing:
                missing_transforms.append((_context_transform_identifier(operation.binding), missing))
                continue
            result = _evaluate_context_transform_from_values(operation.binding, original_values)
            current_values.update(_validate_patch_result(model, result))
            continue

        if isinstance(operation.spec, StaticValueSpec):
            current_values[operation.name] = operation.spec.value
            continue

        missing = _context_transform_missing_context_names(operation.spec, original_values)
        if missing:
            missing_transforms.append((operation.name, missing))
            current_values.pop(operation.name, None)
            continue
        result = _evaluate_context_transform_from_values(operation.spec, original_values)
        current_values[operation.name] = _coerce_model_context_value(model, operation.name, result, "with_context()")

    return current_values, tuple(missing_transforms)


def _unresolved_lazy_model_identity(value: CallableModel) -> Dict[str, Any]:
    """Return a stable structural model payload for unresolved lazy identity."""

    if isinstance(value, BoundModel):
        return {
            "kind": "bound_model",
            "model": _unresolved_lazy_model_identity(value.model),
            "context_spec": value.context_spec.model_dump(mode="python"),
        }

    dump = _stable_model_identity_dump(value)
    return dump


def _contains_model_dependency(value: Any) -> bool:
    if _is_model_dependency(value):
        return True
    if isinstance(value, (tuple, list, frozenset, set)):
        return any(_contains_model_dependency(item) for item in value)
    if isinstance(value, dict):
        return any(_contains_model_dependency(item) for item in value.values())
    return False


def _stable_model_identity_dump(value: Any) -> Any:
    if _is_model_dependency(value):
        dump = value.model_dump(mode="python")
        if isinstance(dump, dict):
            dump = _stable_model_identity_dump(dump)
            for name in type(value).model_fields:
                field_value = getattr(value, name, _UNSET_FLOW_INPUT)
                if name in dump and _contains_model_dependency(field_value):
                    dump[name] = _stable_model_identity_dump(field_value)
            dump["type_"] = _model_type_identity(value)
        return dump
    if isinstance(value, tuple):
        return tuple(_stable_model_identity_dump(item) for item in value)
    if isinstance(value, list):
        return [_stable_model_identity_dump(item) for item in value]
    if isinstance(value, OrderedDict):
        return OrderedDict((key, _stable_model_identity_dump(item)) for key, item in value.items())
    if isinstance(value, dict):
        return {key: _stable_model_identity_dump(item) for key, item in value.items()}
    if isinstance(value, frozenset):
        return frozenset(_stable_model_identity_dump(item) for item in value)
    if isinstance(value, set):
        return {_stable_model_identity_dump(item) for item in value}
    return value


def _unresolved_lazy_dependency_descriptor(
    value: CallableModel,
    context_values: Dict[str, Any],
    missing_context: Tuple[str, ...],
    missing_transform_context: Tuple[Tuple[str, Tuple[str, ...]], ...] = (),
) -> _UnresolvedLazyDependencyIdentity:
    """Describe a lazy dependency whose runtime context cannot be resolved yet."""

    return _UnresolvedLazyDependencyIdentity(
        kind="unresolved_lazy_dependency",
        model_type=_model_type_identity(value),
        model=_unresolved_lazy_model_identity(value),
        context_type=str(PyObjectPath.validate(FlowContext)),
        context=context_values,
        missing_context=missing_context,
        missing_transform_context=missing_transform_context,
    )


def _flow_model_config_identity(config: _FlowModelConfig) -> str:
    """Return a stable identity for a generated model's analyzed behavior.

    This is for local generated classes whose ``PyObjectPath`` points at a
    random ``ccflow.local_persistence._Local_*`` name.  The identity is computed
    once when the generated class is created and then stored in factory kwargs
    so cache keys survive pickle/cloudpickle restore.  Do not recompute it on
    every cache-key request: function closures can contain mutable state such as
    call counters, and those values are runtime state, not model identity.
    """

    return compute_data_token(
        (
            "local_generated_flow_model",
            config.func,
            config.return_annotation,
            config.context_type,
            config.result_type,
            config.auto_wrap_result,
            config.auto_unwrap,
            tuple(
                (
                    param.name,
                    param.annotation,
                    param.is_contextual,
                    param.is_lazy,
                    param.has_function_default,
                    param.function_default,
                    param.context_validation_annotation,
                )
                for param in config.parameters
            ),
            config.declared_context_type,
        )
    )


def _generated_model_behavior_token(config_identity: str, model_base: Type[CallableModel]) -> str:
    return compute_data_token(
        (
            "generated_flow_model_behavior",
            config_identity,
            f"{model_base.__module__}.{model_base.__qualname__}",
            compute_behavior_token(model_base),
        )
    )


def _model_type_identity(model: CallableModel) -> str:
    """Return a stable model-type identity for effective generated-model keys."""

    generated = _generated_model_instance(model)
    if generated is None:
        return str(PyObjectPath.validate(type(model)))

    config = type(generated).__flow_model_config__
    factory_path = _generated_model_factory_path_for_pickle(config, type(generated))
    if factory_path is not None:
        return factory_path
    identity = getattr(type(generated), "__flow_model_identity__", None)
    if identity is None:
        identity = _flow_model_config_identity(config)
    return f"local:{identity}"


def _lazy_dependency_identity(
    value: CallableModel,
    context: ContextBase,
) -> Tuple[Optional[Any], Optional[CallableModel], Optional[ContextBase]]:
    """Resolve or describe a lazy dependency for effective identity.

    If all required context is available, return the concrete dependency
    invocation so evaluator/common can recursively derive its effective key.  If
    not, return a stable unresolved descriptor instead of evaluating or raising.

    This deliberately does not try to prove whether the lazy thunk will be used
    by the downstream function.  That would require executing user logic before
    key construction.  The current policy is conservative: resolvable lazy
    dependencies participate in identity; lazy dependencies with unresolved
    runtime context are represented by descriptors.
    """

    if isinstance(value, BoundModel):
        dependency_model = value.model
        values, missing_transform_context = _apply_context_spec_values_for_identity(dependency_model, value.context_spec, context)
        if missing_transform_context:
            context_values: Dict[str, Any] = {}
            missing_context = _model_context_contract(dependency_model).required_names
        else:
            context_values, missing_context = _identity_context_values_and_missing_for_model(dependency_model, values)
        if missing_context or missing_transform_context:
            return _unresolved_lazy_dependency_descriptor(value, context_values, missing_context, missing_transform_context), None, None
        return None, *_resolved_dependency_invocation(value, context)

    dependency_model = value
    context_values, missing_context = _identity_context_values_and_missing_for_model(dependency_model, _context_values(context))
    if missing_context:
        return _unresolved_lazy_dependency_descriptor(value, context_values, missing_context), None, None
    return None, *_resolved_dependency_invocation(value, context)


def _validate_bound_param_value(
    config: _FlowModelConfig,
    param: _FlowModelParam,
    value: Any,
    source: str,
) -> Any:
    """Validate a construction-time bound parameter for a generated model."""

    if param.is_contextual:
        if _is_model_dependency(value):
            raise TypeError(
                f"Parameter '{param.name}' is marked FromContext[...] and cannot be bound to a CallableModel. "
                "Bind a literal contextual default or supply it via compute()/with_context()."
            )
        return _coerce_contextual_value(config, param, value, source)

    if param.is_lazy:
        if not _is_model_dependency(value):
            value = _resolve_serialized_dependency_ref(value, include_target_alias=True)
        if not _is_model_dependency(value):
            raise TypeError(f"Parameter '{param.name}' is marked Lazy[...] and must be bound to a CallableModel dependency.")
        return value
    if _is_model_dependency(value):
        return value
    if param.has_dep_slots:
        return _validate_dep_marked_value(param.name, value, param.annotation, source)
    try:
        return _coerce_value(param.name, value, param.annotation, source)
    except TypeError:
        value_with_alias_deps = _resolve_serialized_dependency_ref(value, include_target_alias=True)
        if value_with_alias_deps is not value and _is_model_dependency(value_with_alias_deps):
            return value_with_alias_deps
        raise


def _generated_model_identity_payload(
    model: "_GeneratedFlowModelBase",
    context: ContextBase,
) -> Optional[_GeneratedModelIdentity]:
    """Describe the generated model's effective invocation for cache keys.

    Contract:
    - contextual identity is projected to the ``FromContext[...]`` fields the
      generated model consumes;
    - unused ambient ``FlowContext`` fields are ignored;
    - regular literal inputs are included directly;
    - regular ``CallableModel`` inputs are recorded as dependency invocations;
    - lazy ``CallableModel`` inputs are recorded conservatively when their
      context can be resolved, even if the lazy thunk is not called later; and
    - unresolved lazy dependency runtime context is recorded explicitly instead of
      forcing eager dependency resolution.

    Returning ``None`` asks the evaluator to use the structural key.
    """

    config = type(model).__flow_model_config__
    regular_inputs = []
    for param in config.regular_params:
        value = getattr(model, param.name, _UNSET_FLOW_INPUT)
        if _is_unset_flow_input(value):
            return None
        regular_inputs.append(_regular_input_identity(param, value, context))

    model_base_fields = {name: getattr(model, name) for name in sorted(_model_base_field_names(model))}

    return _GeneratedModelIdentity(
        kind="generated_flow_model_v1",
        model_type=_model_type_identity(model),
        contextual_inputs=_resolved_contextual_inputs(model, config, context),
        regular_inputs=tuple(regular_inputs),
        model_base_fields=model_base_fields,
    )


# ---------------------------------------------------------------------------
# Static binding resolution and with_context normalization
# ---------------------------------------------------------------------------


def _resolved_static_contextual_values(
    model: "_GeneratedFlowModelBase",
    config: _FlowModelConfig,
    static_overrides: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    resolved, missing = _collect_contextual_values(model, config, static_overrides or {})
    return None if missing else resolved


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


def _bound_context_transform_regular_kwargs(config: _FlowModelConfig, binding: ContextTransform) -> Dict[str, Any]:
    """Return already-bound regular kwargs for a context transform invocation."""

    kwargs: Dict[str, Any] = {}
    for param in config.regular_params:
        if param.name in binding.bound_args:
            kwargs[param.name] = _coerce_value(param.name, binding.bound_args[param.name], param.annotation, "Context transform argument")
        elif param.has_function_default:
            kwargs[param.name] = param.function_default
        else:
            raise TypeError(f"Context transform '{_callable_name(config.func)}' is missing required regular parameter '{param.name}'.")
    return kwargs


def _evaluate_static_context_transform(binding: ContextTransform) -> Any:
    """Evaluate a transform only when it has no contextual inputs at all."""

    config = _load_context_transform_config_from_binding(binding)
    if config.contextual_params:
        return _UNSET

    kwargs = _bound_context_transform_regular_kwargs(config, binding)
    return config.func(**kwargs)


def _statically_resolved_context_values(model: CallableModel, context_spec: _BoundContextSpec) -> Optional[Dict[str, Any]]:
    """Return static binding values when the whole spec can be resolved without runtime context."""

    values: Dict[str, Any] = {}

    for operation in _effective_context_operations(context_spec):
        if isinstance(operation, PatchContextOperation):
            result = _evaluate_static_context_transform(operation.binding)
            if result is _UNSET:
                return None
            values.update(_validate_patch_result(model, result))
            continue

        if isinstance(operation.spec, StaticValueSpec):
            value = operation.spec.value
        else:
            value = _evaluate_static_context_transform(operation.spec)
            if value is not _UNSET:
                value = _coerce_model_context_value(model, operation.name, value, "with_context()")
        if value is _UNSET:
            return None
        values[operation.name] = value

    return values


def _statically_resolved_context_field_values(model: CallableModel, context_spec: _BoundContextSpec) -> Dict[str, Any]:
    values: Dict[str, Any] = {}

    for operation in _effective_context_operations(context_spec):
        if isinstance(operation, PatchContextOperation):
            result = _evaluate_static_context_transform(operation.binding)
            if result is _UNSET:
                values.clear()
                continue
            values.update(_validate_patch_result(model, result))
            continue

        if isinstance(operation.spec, StaticValueSpec):
            values[operation.name] = operation.spec.value
            continue

        value = _evaluate_static_context_transform(operation.spec)
        if value is _UNSET:
            values.pop(operation.name, None)
            continue
        values[operation.name] = _coerce_model_context_value(model, operation.name, value, "with_context()")

    return values


def _statically_resolved_context_field_names(model: CallableModel, context_spec: _BoundContextSpec) -> Set[str]:
    return set(_statically_resolved_context_field_values(model, context_spec))


def _context_transform_input_types(binding: ContextTransform, *, required_only: bool) -> Dict[str, Any]:
    config = _load_context_transform_config_from_binding(binding)
    names = config.context_required_names if required_only else config.contextual_param_names
    return {name: config.context_input_types[name] for name in names}


def _merge_context_input_types(target: Dict[str, Any], updates: Dict[str, Any]) -> None:
    """Merge context input annotations without silently hiding conflicts."""

    for name, annotation in updates.items():
        if name in target and target[name] != annotation:
            raise TypeError(f"Conflicting runtime context annotations for {name!r}: {target[name]!r} and {annotation!r}.")
        target[name] = annotation


def _dynamic_context_operation_effects(context_spec: _BoundContextSpec, *, required_only: bool) -> Tuple[Set[str], Dict[str, Any]]:
    supplied_fields: Set[str] = set()
    input_types: Dict[str, Any] = {}

    for operation in _effective_context_operations(context_spec):
        if isinstance(operation, PatchContextOperation):
            patch_result = _evaluate_static_context_transform(operation.binding)
            if patch_result is _UNSET:
                _merge_context_input_types(input_types, _context_transform_input_types(operation.binding, required_only=required_only))
                continue
            continue

        if isinstance(operation.spec, StaticValueSpec):
            continue

        value = _evaluate_static_context_transform(operation.spec)
        if value is _UNSET:
            supplied_fields.add(operation.name)
            _merge_context_input_types(input_types, _context_transform_input_types(operation.spec, required_only=required_only))
            continue

    return supplied_fields, input_types


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

    resolved = _resolved_static_contextual_values(generated, config, static_context_values)
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
    return _is_mapping_annotation(_load_context_transform_config_from_binding(binding).return_annotation)


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

    return {name: _coerce_model_context_value(model, name, value, "with_context() patch") for name, value in patch.items()}


def _normalize_with_context(model: CallableModel, patches: Tuple[Any, ...], field_overrides: Dict[str, Any]) -> _BoundContextSpec:
    """Validate and normalize user-facing ``with_context(...)`` arguments."""

    operations: List[_ContextOperation] = []
    for patch in patches:
        if callable(patch):
            raise TypeError("Positional with_context() arguments must be bound @Flow.context_transform results that return a mapping.")
        if not isinstance(patch, ContextTransform):
            raise TypeError("Positional with_context() arguments must be @Flow.context_transform bindings that return a mapping.")
        if not _binding_uses_patch_shape(patch):
            raise TypeError(
                "Field context transforms must be passed by keyword to with_context(...). Patch transforms belong in positional arguments."
            )
        operations.append(PatchContextOperation(binding=patch))

    _validate_with_context_field_names(model, list(field_overrides))
    contract = _model_context_contract(model)
    for name, value in field_overrides.items():
        if isinstance(value, ContextTransform):
            if _binding_uses_patch_shape(value):
                raise TypeError("Patch transforms must be passed positionally to with_context(...), not as keyword field overrides.")
            operations.append(FieldContextOperation(name=name, spec=value))
            continue
        if callable(value) and (contract.input_types is None or name not in contract.input_types):
            raise TypeError(
                "Callable keyword values in with_context() must either be bound @Flow.context_transform results "
                "or validate against a declared contextual field type."
            )
        spec = StaticValueSpec(
            value=value
            if contract.input_types is None or name not in contract.input_types
            else _coerce_model_context_value(model, name, value, "with_context()")
        )
        operations.append(FieldContextOperation(name=name, spec=spec))

    context_spec = _BoundContextSpec(operations=operations)
    return _validate_static_context_spec_declared_context(model, context_spec)


# ---------------------------------------------------------------------------
# Bound context application and compute context construction
# ---------------------------------------------------------------------------


def _context_from_values_preserving_private_state(context: ContextBase, values: Dict[str, Any]) -> ContextBase:
    """Validate updated public values while preserving private context state."""

    if values == _context_values(context):
        return context
    validated = type(context).model_validate(values)
    private = getattr(context, "__pydantic_private__", None)
    if private is not None:
        object.__setattr__(validated, "__pydantic_private__", dict(private))
    return validated


def _apply_context_spec_values(model: CallableModel, context_spec: _BoundContextSpec, context: ContextBase) -> Dict[str, Any]:
    """Apply a binding spec at execution time and return rewritten context values."""

    original_values = _context_values(context)
    current_values = dict(original_values)

    for operation in _effective_context_operations(context_spec):
        if isinstance(operation, PatchContextOperation):
            result = _evaluate_context_transform_from_values(operation.binding, original_values)
            current_values.update(_validate_patch_result(model, result))
            continue

        if isinstance(operation.spec, StaticValueSpec):
            current_values[operation.name] = operation.spec.value
            continue
        result = _evaluate_context_transform_from_values(operation.spec, original_values)
        current_values[operation.name] = _coerce_model_context_value(model, operation.name, result, "with_context()")

    return current_values


def _apply_context_spec(model: CallableModel, context_spec: _BoundContextSpec, context: ContextBase) -> ContextBase:
    """Apply bindings, project to the wrapped model, and build its runtime context."""

    if not context_spec.operations:
        if isinstance(context, _BoundModelContext):
            if context._base_context is not None:
                values = _project_context_values_for_model(model, _context_values(context))
                return _context_from_values_preserving_private_state(context._base_context, values)
            return _dependency_context_for_model(model, context)
        if _context_matches_type(context, model.context_type):
            return context
        return _dependency_context_for_model(model, context)

    values = _apply_context_spec_values(model, context_spec, context)
    if isinstance(context, _BoundModelContext):
        values = _project_context_values_for_model(model, values)
        if context._base_context is not None:
            return _context_from_values_preserving_private_state(context._base_context, values)
        return _runtime_context_for_model(model, values)
    if _context_matches_type(context, model.context_type):
        return _context_from_values_preserving_private_state(context, values)
    return _runtime_context_for_model(model, _project_context_values_for_model(model, values))


def _plain_model_default_context(model: CallableModel) -> Any:
    call = getattr(type(model), "__call__", None)
    wrapped = getattr(call, "__wrapped__", None)
    if wrapped is None:
        return _UNSET
    try:
        parameter = inspect.signature(wrapped).parameters.get("context")
    except (TypeError, ValueError):
        return _UNSET
    if parameter is None or parameter.default is inspect.Signature.empty:
        return _UNSET
    return parameter.default


def _plain_model_default_context_values(
    model: CallableModel,
    runtime_context_type: Type[ContextBase],
) -> Optional[Dict[str, Any]]:
    default_context = _plain_model_default_context(model)
    if default_context is _UNSET:
        return None
    if default_context is None:
        if _is_optional_context_type(model.context_type):
            return {}
        return _context_values(runtime_context_type.model_validate(default_context))
    if isinstance(default_context, ContextBase):
        return _context_values(default_context)
    return _context_values(runtime_context_type.model_validate(default_context))


def _plain_model_compute_context_from_default(
    model: CallableModel,
    default_context: Any,
    default_values: Dict[str, Any],
    kwargs: Dict[str, Any],
    runtime_context_type: Type[ContextBase],
) -> Optional[ContextBase]:
    if default_context is None and not kwargs and _is_optional_context_type(model.context_type):
        return None

    if isinstance(default_context, ContextBase) and _context_matches_type(default_context, model.context_type):
        values = dict(default_values)
        values.update(kwargs)
        return _context_from_values_preserving_private_state(default_context, values)

    values = dict(default_values)
    values.update(kwargs)
    return runtime_context_type.model_validate(values)


def _build_compute_context(model: CallableModel, context: Any, kwargs: Dict[str, Any]) -> Optional[ContextBase]:
    """Construct the context used by ``FlowAPI.compute`` for a target model.

    ``compute`` is intentionally not a second constructor.  For generated models
    it only supplies contextual inputs; regular parameters and model_base fields
    must already be bound on the model instance.
    """

    if context is not _UNSET and kwargs:
        raise TypeError("compute() accepts either one context object or contextual keyword inputs, but not both.")

    ctx_type = model.context_type
    _ctx_is_optional = _is_optional_context_type(ctx_type)

    contract = _model_context_contract(model)

    if context is not _UNSET:
        if context is None and _ctx_is_optional:
            return None
        if isinstance(context, FlowContext):
            return context
        if isinstance(context, ContextBase):
            if _context_matches_type(context, model.context_type):
                return context
            return _runtime_context_for_model(model, _context_values(context))
        return contract.runtime_context_type.model_validate(context)

    if contract.generated_model is None:
        default_context = _plain_model_default_context(model)
        if default_context is not _UNSET:
            default_values = _plain_model_default_context_values(model, contract.runtime_context_type)
            return _plain_model_compute_context_from_default(model, default_context, default_values, kwargs, contract.runtime_context_type)
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
        ambient[param.name] = _coerce_contextual_value(config, param, kwargs[param.name], "compute() input")
    return FlowContext(**ambient)


def _is_optional_context_type(context_type: Any) -> bool:
    return get_origin(context_type) in _UNION_ORIGINS and type(None) in get_args(context_type)


def _context_matches_type(context: Any, context_type: Any) -> bool:
    """Return whether an existing context object is accepted by a context annotation."""

    if context is None:
        return _is_optional_context_type(context_type)
    if get_origin(context_type) in _UNION_ORIGINS:
        return any(_context_matches_type(context, arg) for arg in get_args(context_type) if arg is not type(None))
    return isinstance(context_type, type) and isinstance(context, context_type)


def _bound_model_preserves_none_context(bound_model: "BoundModel") -> bool:
    return not bound_model.context_spec.operations and _is_optional_context_type(bound_model.model.context_type)


def _build_bound_compute_context(bound_model: "BoundModel", context: Any, kwargs: Dict[str, Any]) -> Optional[ContextBase]:
    """Construct the ambient context passed into a ``BoundModel`` by ``compute``."""

    if context is not _UNSET and kwargs:
        raise TypeError("compute() accepts either one context object or contextual keyword inputs, but not both.")
    if context is not _UNSET:
        return context
    if not kwargs and _bound_model_preserves_none_context(bound_model):
        return None
    return _bound_model_ambient_context(bound_model, kwargs)


def _raw_input_values_for_debug(model: CallableModel, context: Any, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Return caller-supplied context values without executing the model."""

    if context is not _UNSET and kwargs:
        raise TypeError("compute() accepts either one context object or contextual keyword inputs, but not both.")
    if context is _UNSET:
        return dict(kwargs)
    if context is None:
        return {}
    if isinstance(context, ContextBase):
        return _context_values(context)
    if isinstance(context, Mapping):
        return dict(context)
    return _context_values(_model_context_contract(model).runtime_context_type.model_validate(context))


def _partial_context_for_inspect(model: CallableModel, values: Dict[str, Any]) -> ContextBase:
    contract = _model_context_contract(model)
    if contract.input_types is None:
        return FlowContext(**values)
    return FlowContext(**{name: values[name] for name in contract.input_types if name in values})


def _partial_dependency_context_for_inspect(model: CallableModel, context: ContextBase) -> ContextBase:
    if isinstance(model, BoundModel):
        return FlowContext(**_context_values(context))
    return _partial_context_for_inspect(model, _context_values(context))


def _project_bound_dependency_context_for_inspect(model: "BoundModel", context: ContextBase) -> ContextBase:
    values = _context_values(context)
    projected = {name: values[name] for name in model.flow._runtime_inputs if name in values}
    if isinstance(context, _BoundModelContext) and context._base_context is not None:
        return _BoundModelContext.from_values(projected, base_context=context._base_context)
    return FlowContext(**projected)


def _generated_context_argument_specs(generated: "_GeneratedFlowModelBase", input_types: Optional[Dict[str, Any]]) -> Dict[str, InputSpec]:
    config = type(generated).__flow_model_config__
    explicit_fields = _bound_field_names(generated)
    specs: Dict[str, InputSpec] = {}
    for param in config.contextual_params:
        annotation = param.annotation if input_types is None else input_types[param.name]
        value = getattr(generated, param.name, _UNSET_FLOW_INPUT)
        if param.name in explicit_fields and not _is_unset_flow_input(value):
            specs[param.name] = InputSpec(param.name, annotation, False, _UNSET_FLOW_INPUT, value, "construction")
        elif param.has_function_default:
            specs[param.name] = InputSpec(param.name, annotation, False, param.function_default, param.function_default, "function_default")
        else:
            specs[param.name] = InputSpec(param.name, annotation, True, _UNSET_FLOW_INPUT, _UNSET_FLOW_INPUT, "runtime")
    return specs


def _plain_context_argument_specs(model: CallableModel, contract: _ModelContextContract) -> Dict[str, InputSpec]:
    if contract.input_types is None:
        return {}
    default_values = _plain_model_default_context_values(model, contract.runtime_context_type)
    required_inputs = set(contract.required_names)
    if default_values is not None:
        required_inputs -= set(default_values)
    specs = {}
    for name, annotation in contract.input_types.items():
        if default_values is not None and name in default_values:
            specs[name] = InputSpec(name, annotation, False, default_values[name], default_values[name], "context_default")
        else:
            specs[name] = InputSpec(name, annotation, name in required_inputs, _UNSET_FLOW_INPUT, _UNSET_FLOW_INPUT, "runtime")
    return specs


def _direct_dependency_specs(
    model: CallableModel,
    context: Optional[ContextBase] = None,
    *,
    trim_context: bool = True,
) -> Tuple[DependencySpec, ...]:
    if isinstance(model, BoundModel):
        rewritten_context = None if context is None else model._rewrite_context(context)
        return _direct_dependency_specs(model.model, rewritten_context, trim_context=trim_context)

    generated = _generated_model_instance(model)
    if generated is None:
        return ()

    specs = []
    config = type(generated).__flow_model_config__
    for param in config.regular_params:
        value = getattr(generated, param.name, _UNSET_FLOW_INPUT)
        if _is_unset_flow_input(value) or not _is_model_dependency(value):
            continue
        dependency_model = value
        dependency_context = None
        if context is not None:
            try:
                dependency_model, dependency_context = _resolved_dependency_invocation(value, context)
            except (TypeError, ValueError, ValidationError):
                dependency_model = value
                dependency_context = _partial_dependency_context_for_inspect(value, context)
            else:
                contract = _model_context_contract(dependency_model)
                if trim_context and dependency_context is not None:
                    if isinstance(dependency_model, BoundModel):
                        dependency_context = _project_bound_dependency_context_for_inspect(dependency_model, dependency_context)
                    elif contract.input_types is not None:
                        values = _context_values(dependency_context)
                        dependency_context = _runtime_context_for_model(
                            dependency_model,
                            {name: values[name] for name in contract.input_types if name in values},
                        )
        specs.append(DependencySpec(param.name, dependency_model, dependency_context, lazy=param.is_lazy))
    return tuple(specs)


def _normalize_inspect_dependencies(dependencies: str) -> Literal["none", "direct", "recursive"]:
    if dependencies == "none":
        return "none"
    if dependencies == "direct":
        return "direct"
    if dependencies == "recursive":
        return "recursive"
    raise ValueError("dependencies must be one of: direct, none, recursive")


def _is_missing_contextual_input_error(error: TypeError) -> bool:
    return str(error).startswith("Missing contextual input(s)")


def _recursive_dependency_specs_for_flow(
    flow: "FlowAPI",
    context: Optional[ContextBase],
    *,
    prefix: str = "",
    lazy_parent: bool = False,
    active: Optional[Set[int]] = None,
) -> Tuple[DependencySpec, ...]:
    """Return inspect-visible dependency specs below ``flow`` with prefixed paths."""

    active = set() if active is None else active
    model_id = id(flow._compute_target)
    if model_id in active:
        return ()

    active.add(model_id)
    try:
        try:
            argument_context = flow._argument_context(context)
            direct_dependencies = flow._dependency_specs_for_inspect(
                context,
                argument_context,
                preserve_ambient_context=True,
            )
        except TypeError as exc:
            if not _is_missing_contextual_input_error(exc):
                raise
            return ()
        result = []
        for dependency in direct_dependencies:
            path = f"{prefix}.{dependency.path}" if prefix else dependency.path
            lazy = lazy_parent or dependency.lazy
            prefixed = DependencySpec(path, dependency.model, dependency.context, lazy)
            result.append(prefixed)
            result.extend(
                _recursive_dependency_specs_for_flow(
                    dependency.model.flow,
                    dependency.context,
                    prefix=path,
                    lazy_parent=lazy,
                    active=active,
                )
            )
        return tuple(result)
    finally:
        active.remove(model_id)


# ---------------------------------------------------------------------------
# model.flow API and BoundModel wrapper
# ---------------------------------------------------------------------------


class FlowAPI:
    """API namespace exposed as ``model.flow``.

    ``FlowAPI`` works for both generated models and ordinary ``CallableModel``
    instances.  Generated models get richer introspection because their function
    signature declares regular and contextual inputs.  Plain models only expose
    what can be inferred from their ``context_type`` and pydantic fields.
    """

    _PUBLIC_HELP: ClassVar[Dict[str, str]] = {
        "compute": "Evaluate the model from a context object or runtime keyword context.",
        "with_context": "Return a new wrapper that binds or rewrites runtime context before evaluation; it does not mutate this model.",
        "inspect": "Return a readable debugging summary. Options: dependencies='direct|recursive|none'.",
    }
    _PUBLIC_NAMES: ClassVar[Tuple[str, ...]] = tuple(_PUBLIC_HELP)

    def __init__(self, model: CallableModel):
        self._model = model

    def __dir__(self) -> List[str]:
        """Return a focused list of public helpers for interactive autocomplete."""

        return sorted(self._PUBLIC_NAMES)

    def __repr__(self) -> str:
        helpers = ", ".join(self._PUBLIC_NAMES)
        return f"{type(self).__name__}(model={self._compute_target!r}, helpers=[{helpers}])"

    @property
    def _compute_target(self) -> CallableModel:
        return self._model

    def compute(self, context: Any = _UNSET, /, _options: Optional[FlowOptions] = None, **kwargs) -> Any:
        """Evaluate the model after building a runtime context from ``context`` or kwargs."""

        target = self._compute_target
        built_context = _build_compute_context(target, context, kwargs)
        return _maybe_auto_unwrap_external_result(target, target(built_context, _options=_options))

    @property
    def _context_inputs(self) -> Dict[str, Any]:
        """Declared contextual input names and expected types for this model."""

        contract = _model_context_contract(self._model)
        return dict(contract.input_types or {})

    @property
    def _runtime_inputs(self) -> Dict[str, Any]:
        """Direct runtime context fields this model may read from the caller."""

        return self._context_inputs

    @property
    def _required_inputs(self) -> Dict[str, Any]:
        """Required direct runtime context fields still needed from the caller."""

        contract = _model_context_contract(self._model)
        if contract.generated_model is None and _is_optional_context_type(self._model.context_type):
            return {}
        if contract.generated_model is None:
            if contract.input_types is None:
                return {}
            result = {name: contract.input_types[name] for name in contract.required_names}
            default_values = _plain_model_default_context_values(self._model, contract.runtime_context_type)
            if default_values is not None:
                for name in default_values:
                    result.pop(name, None)
            return result

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
    def _context_argument_specs(self) -> Dict[str, InputSpec]:
        """Rich descriptions of declared direct contextual inputs."""

        contract = _model_context_contract(self._model)
        if contract.generated_model is not None:
            return _generated_context_argument_specs(contract.generated_model, contract.input_types)
        return _plain_context_argument_specs(self._model, contract)

    @property
    def _runtime_argument_specs(self) -> Dict[str, InputSpec]:
        """Rich descriptions of direct runtime inputs read from the caller."""

        specs = self._context_argument_specs
        runtime_names = set(self._runtime_inputs)
        required_names = set(self._required_inputs)
        result = {}
        for name in runtime_names:
            spec = specs.get(name)
            result[name] = InputSpec(
                name,
                self._runtime_inputs[name],
                name in required_names,
                spec.default if spec is not None else _UNSET_FLOW_INPUT,
                spec.value if spec is not None else _UNSET_FLOW_INPUT,
                spec.source if spec is not None else "runtime",
            )
        return result

    @property
    def _argument_specs(self) -> Dict[str, InputSpec]:
        """Rich descriptions of direct construction and contextual inputs."""

        generated = _model_context_contract(self._model).generated_model
        if generated is None:
            return self._context_argument_specs

        config = type(generated).__flow_model_config__
        specs: Dict[str, InputSpec] = {}
        for param in config.regular_params:
            value = getattr(generated, param.name, _UNSET_FLOW_INPUT)
            if not _is_unset_flow_input(value):
                specs[param.name] = InputSpec(param.name, param.annotation, False, _UNSET_FLOW_INPUT, value, "construction")
            elif param.has_function_default:
                specs[param.name] = InputSpec(param.name, param.annotation, False, param.function_default, param.function_default, "function_default")
            else:
                specs[param.name] = InputSpec(param.name, param.annotation, True, _UNSET_FLOW_INPUT, _UNSET_FLOW_INPUT, "construction")
        specs.update(self._context_argument_specs)
        return specs

    def inspect(
        self,
        context: Any = _UNSET,
        /,
        *,
        dependencies: Literal["direct", "recursive", "none"] = "direct",
        **kwargs,
    ) -> FlowInspection:
        """Return a readable direct debugging summary for this model.

        Args:
            context: Optional runtime context object.
            dependencies: Dependency inspection depth. ``"direct"`` includes
                only immediate dependencies; ``"recursive"`` follows
                inspect-visible dependencies recursively; ``"none"`` skips
                dependency inspection.
            **kwargs: Runtime context field values.
        """

        dependency_depth = _normalize_inspect_dependencies(dependencies)
        dependency_context = None
        argument_context = None
        dependency_specs: Tuple[DependencySpec, ...]
        if context is not _UNSET or kwargs:
            raw_values = _raw_input_values_for_debug(self._compute_target, context, kwargs)
            try:
                dependency_context = _build_compute_context(self._compute_target, context, kwargs)
            except (TypeError, ValidationError):
                dependency_context = _partial_context_for_inspect(self._compute_target, raw_values)
            try:
                argument_context = self._argument_context(dependency_context)
            except (TypeError, ValidationError) as exc:
                if isinstance(exc, TypeError) and not _is_missing_contextual_input_error(exc):
                    raise
                argument_context = None
            dependency_specs = self._dependency_specs_for_inspect(dependency_context, argument_context) if dependency_depth == "direct" else ()
            if dependency_depth == "recursive":
                dependency_specs = _recursive_dependency_specs_for_flow(self, dependency_context)
        else:
            try:
                argument_context = self._argument_context(None)
            except (TypeError, ValidationError) as exc:
                if isinstance(exc, TypeError) and not _is_missing_contextual_input_error(exc):
                    raise
                argument_context = None
            dependency_specs = self._dependency_specs_for_inspect(None, argument_context) if dependency_depth == "direct" else ()
            if dependency_depth == "recursive":
                dependency_specs = _recursive_dependency_specs_for_flow(self, None)
        return FlowInspection(
            model=self._compute_target,
            context_inputs=self._context_inputs,
            runtime_inputs=self._runtime_inputs,
            required_inputs=self._required_inputs,
            bound_inputs=self._bound_inputs,
            inputs=self._argument_specs_for_context(argument_context),
            dependencies=dependency_specs,
        )

    def _argument_context(self, context: Optional[ContextBase]) -> Optional[ContextBase]:
        return context

    def _dependency_specs_for_inspect(
        self,
        dependency_context: Optional[ContextBase],
        _argument_context: Optional[ContextBase],
        *,
        preserve_ambient_context: bool = False,
    ) -> Tuple[DependencySpec, ...]:
        return _direct_dependency_specs(self._compute_target, dependency_context, trim_context=not preserve_ambient_context)

    def _context_argument_specs_for_context(self, context: Optional[ContextBase]) -> Dict[str, InputSpec]:
        result = dict(self._context_argument_specs)
        if context is None:
            return result
        values = _context_values(context)
        for name, spec in list(result.items()):
            if name in values:
                result[name] = InputSpec(name, spec.type, False, spec.default, values[name], "runtime")
        return result

    def _runtime_argument_specs_for_context(self, context: Optional[ContextBase]) -> Dict[str, InputSpec]:
        specs = self._context_argument_specs_for_context(context)
        runtime_names = set(self._runtime_inputs)
        required_names = set(self._required_inputs)
        result = {}
        for name in runtime_names:
            spec = specs.get(name)
            result[name] = InputSpec(
                name,
                self._runtime_inputs[name],
                name in required_names,
                spec.default if spec is not None else _UNSET_FLOW_INPUT,
                spec.value if spec is not None else _UNSET_FLOW_INPUT,
                spec.source if spec is not None else "runtime",
            )
        return result

    def _argument_specs_for_context(self, context: Optional[ContextBase]) -> Dict[str, InputSpec]:
        generated = _model_context_contract(self._model).generated_model
        if generated is None:
            return self._context_argument_specs_for_context(context)

        config = type(generated).__flow_model_config__
        specs: Dict[str, InputSpec] = {}
        for param in config.regular_params:
            value = getattr(generated, param.name, _UNSET_FLOW_INPUT)
            if not _is_unset_flow_input(value):
                specs[param.name] = InputSpec(param.name, param.annotation, False, _UNSET_FLOW_INPUT, value, "construction")
            elif param.has_function_default:
                specs[param.name] = InputSpec(param.name, param.annotation, False, param.function_default, param.function_default, "function_default")
            else:
                specs[param.name] = InputSpec(param.name, param.annotation, True, _UNSET_FLOW_INPUT, _UNSET_FLOW_INPUT, "construction")
        specs.update(self._context_argument_specs_for_context(context))
        return specs

    @property
    def _bound_inputs(self) -> Dict[str, Any]:
        """Inputs already fixed by construction-time values or static context bindings."""

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
        """Return a wrapper that rewrites runtime context before evaluating this model."""

        context_spec = _normalize_with_context(self._model, patches, field_overrides)
        return BoundModel(model=self._model, context_spec=context_spec)


class BoundModel(WrapperModel):
    """A wrapper that rewrites context for exactly one wrapped model.

    ``BoundModel`` is deliberately a ``WrapperModel`` rather than mutating the
    wrapped model.  This keeps context bindings scoped to the edge where they
    are used, lets dependency graphs show the wrapped model, and derives
    effective identity from the rewritten wrapped invocation.
    """

    context_spec: _BoundContextSpec = Field(default_factory=_BoundContextSpec, repr=False)

    def _rewrite_context(self, context: ContextBase) -> ContextBase:
        """Apply this wrapper's context bindings to an ambient runtime context."""

        return _apply_context_spec(self.model, self.context_spec, context)

    @property
    def context_type(self) -> Any:
        if _bound_model_preserves_none_context(self):
            return self.model.context_type
        return _BoundModelContext

    @Flow.call
    def __call__(self, context: ContextType) -> ResultBase:
        """Evaluate the wrapped model after rewriting context."""

        if context is None and _bound_model_preserves_none_context(self):
            return self.model(None)
        return self.model(self._rewrite_context(context))

    @Flow.deps
    def __deps__(self, context: ContextType) -> GraphDepList:
        """Expose the wrapped model as the single dependency of this binding wrapper."""

        if context is None and _bound_model_preserves_none_context(self):
            return self.model.__deps__(None)
        return [(self.model, [self._rewrite_context(context)])]

    def __repr__(self) -> str:
        args = []
        for operation in _effective_context_operations(self.context_spec):
            if isinstance(operation, PatchContextOperation):
                args.append(_context_transform_repr(operation.binding))
                continue
            value = operation.spec if isinstance(operation.spec, ContextTransform) else operation.spec.value
            args.append(f"{operation.name}={_context_transform_repr(value)}")
        return f"{self.model!r}.flow.with_context({', '.join(args)})"

    def _evaluation_identity_payload(
        self,
        context: ContextBase,
    ) -> Optional[Any]:
        """Describe this binding in terms of the rewritten wrapped call."""

        return {
            "kind": "bound_model_v1",
            "model": EvaluationDependency(self.model, self._rewrite_context(context)),
        }

    @property
    def flow(self) -> "FlowAPI":
        """Access bound flow helpers for execution, context transforms, and introspection."""

        return _BoundFlowAPI(self)


class _BoundFlowAPI(FlowAPI):
    """``model.flow`` implementation for ``BoundModel`` wrappers."""

    def __init__(self, bound_model: BoundModel):
        self._bound = bound_model
        super().__init__(bound_model.model)

    @property
    def _compute_target(self) -> CallableModel:
        return self._bound

    def compute(self, context: Any = _UNSET, /, _options: Optional[FlowOptions] = None, **kwargs) -> Any:
        """Evaluate the bound wrapper after building its ambient context."""

        built_context = _build_bound_compute_context(self._bound, context, kwargs)
        return _maybe_auto_unwrap_external_result(self._bound, self._bound(built_context, _options=_options))

    def _argument_context(self, context: Optional[ContextBase]) -> Optional[ContextBase]:
        if context is None:
            if _bound_model_preserves_none_context(self._bound):
                return None
            _supplied_fields, required_dynamic_inputs = _dynamic_context_operation_effects(self._bound.context_spec, required_only=True)
            if required_dynamic_inputs:
                return None
            return self._bound._rewrite_context(_bound_model_ambient_context(self._bound, {}))
        return self._bound._rewrite_context(context)

    def _dependency_specs_for_inspect(
        self,
        _dependency_context: Optional[ContextBase],
        argument_context: Optional[ContextBase],
        *,
        preserve_ambient_context: bool = False,
    ) -> Tuple[DependencySpec, ...]:
        return _direct_dependency_specs(self._bound.model, argument_context, trim_context=not preserve_ambient_context)

    @property
    def _bound_inputs(self) -> Dict[str, Any]:
        """Concrete values already fixed, including statically resolved context bindings."""

        result = super()._bound_inputs
        for name in self._context_inputs:
            result.pop(name, None)
        result.update(_statically_resolved_context_field_values(self._bound.model, self._bound.context_spec))
        return result

    @property
    def _context_inputs(self) -> Dict[str, Any]:
        """Declared contextual inputs of the wrapped model."""

        return super()._context_inputs

    @property
    def _context_argument_specs(self) -> Dict[str, InputSpec]:
        """Rich descriptions of wrapped-model contextual inputs after bindings."""

        result = dict(super()._context_argument_specs)
        for name, value in _statically_resolved_context_field_values(self._bound.model, self._bound.context_spec).items():
            if name in result:
                spec = result[name]
                result[name] = InputSpec(name, spec.type, False, spec.default, value, "with_context")

        supplied_fields, _dynamic_inputs = _dynamic_context_operation_effects(self._bound.context_spec, required_only=False)
        for name in supplied_fields:
            if name in result:
                spec = result[name]
                result[name] = InputSpec(name, spec.type, False, spec.default, _UNSET_FLOW_INPUT, "context_transform")
        return result

    def _context_argument_specs_for_context(self, context: Optional[ContextBase]) -> Dict[str, InputSpec]:
        result = dict(self._context_argument_specs)
        if context is None:
            return result
        values = _context_values(context)
        for name, spec in list(result.items()):
            if name not in values:
                continue
            source = spec.source if spec.source in {"context_transform", "with_context"} else "runtime"
            result[name] = InputSpec(name, spec.type, False, spec.default, values[name], source)
        return result

    @property
    def _runtime_inputs(self) -> Dict[str, Any]:
        """Direct runtime context inputs after applying this wrapper's bindings.

        Static context transforms may be evaluated to identify resolved fields.
        Dynamic transforms contribute their own runtime context inputs.
        """

        result = super()._context_inputs
        for name in _statically_resolved_context_field_names(self._bound.model, self._bound.context_spec):
            result.pop(name, None)
        supplied_fields, dynamic_inputs = _dynamic_context_operation_effects(self._bound.context_spec, required_only=False)
        for name in supplied_fields:
            result.pop(name, None)
        _merge_context_input_types(result, dynamic_inputs)
        return result

    @property
    def _runtime_argument_specs(self) -> Dict[str, InputSpec]:
        """Rich descriptions of runtime inputs after applying this wrapper's bindings."""

        result = {}
        base_specs = self._context_argument_specs
        for name, annotation in self._runtime_inputs.items():
            base = base_specs.get(name)
            result[name] = InputSpec(
                name,
                annotation,
                name in self._required_inputs,
                base.default if base is not None else _UNSET_FLOW_INPUT,
                base.value if base is not None else _UNSET_FLOW_INPUT,
                base.source if base is not None else "runtime",
            )
        _supplied_fields, dynamic_inputs = _dynamic_context_operation_effects(self._bound.context_spec, required_only=False)
        required_dynamic = set(_dynamic_context_operation_effects(self._bound.context_spec, required_only=True)[1])
        for name, annotation in dynamic_inputs.items():
            result[name] = InputSpec(name, annotation, name in required_dynamic, _UNSET_FLOW_INPUT, _UNSET_FLOW_INPUT, "context_transform")
        return result

    @property
    def _required_inputs(self) -> Dict[str, Any]:
        """Required direct runtime context inputs still missing after static bindings.

        Static context transforms may be evaluated to identify resolved fields.
        Dynamic transforms contribute their own required runtime context inputs.
        """

        result = super()._required_inputs
        for name in _statically_resolved_context_field_names(self._bound.model, self._bound.context_spec):
            result.pop(name, None)
        supplied_fields, dynamic_inputs = _dynamic_context_operation_effects(self._bound.context_spec, required_only=True)
        for name in supplied_fields:
            result.pop(name, None)
        _merge_context_input_types(result, dynamic_inputs)
        contract = _model_context_contract(self._bound.model)
        if contract.generated_model is None:
            default_values = _plain_model_default_context_values(self._bound.model, contract.runtime_context_type)
            if default_values is not None:
                for name in default_values:
                    result.pop(name, None)
        return result

    def with_context(self, *patches, **field_overrides) -> BoundModel:
        context_spec = _normalize_with_context(self._bound.model, patches, field_overrides)
        merged = _BoundContextSpec(operations=[*self._bound.context_spec.operations, *context_spec.operations])
        return BoundModel(
            model=self._bound.model,
            context_spec=_validate_static_context_spec_declared_context(self._bound.model, merged),
        )


class _GeneratedFlowModelBase(CallableModel):
    """Base class for all classes created by ``@Flow.model``."""

    __flow_model_config__: ClassVar[_FlowModelConfig]
    __flow_model_identity__: ClassVar[str]

    if not _FIELD_EXCLUDE_IF_SUPPORTED:

        @model_serializer(mode="plain")
        def _serialize_generated_flow_model_compat(self, info):
            """Drop unbound sentinels on pydantic versions before Field(exclude_if)."""

            include = getattr(info, "include", None)
            exclude = getattr(info, "exclude", None)
            by_alias = bool(getattr(info, "by_alias", False))

            def selected(name: str, key: str) -> bool:
                if include is not None and name not in include and key not in include:
                    return False
                if exclude is not None and (name in exclude or key in exclude):
                    return False
                return True

            data: Dict[str, Any] = {}
            for name, field_info in type(self).model_fields.items():
                key = name
                if by_alias:
                    key = field_info.serialization_alias or field_info.alias or name
                if not selected(name, key):
                    continue
                if getattr(info, "exclude_unset", False) and name not in self.__pydantic_fields_set__:
                    continue
                value = getattr(self, name, _UNSET_FLOW_INPUT)
                if _is_unset_flow_input(value):
                    continue
                if getattr(info, "exclude_none", False) and value is None:
                    continue
                data[key] = value

            type_key = "_target_" if by_alias else "type_"
            if selected("type_", type_key):
                data[type_key] = self.type_
            return data

    def __reduce__(self):
        """Prefer import-path restoration, falling back to serialized local factories."""

        config = type(self).__flow_model_config__

        state = self.__getstate__()
        factory_path = _generated_model_factory_path_for_pickle(config, type(self))
        if factory_path is not None:
            return (_new_generated_flow_model_for_pickle, (factory_path,), state)

        # Local generated classes are not normal importable class definitions:
        # plain pickle cannot reconstruct them, and Ray workers should not
        # re-run fragile annotation resolution.  Carry the analyzed contract
        # separately, but leave instance state in the outer pickle stream so
        # normal pickle memo semantics remain intact.
        payload = _LocalFlowModelPicklePayload(
            serialized_config=_serialize_flow_model_config(config),
            factory_kwargs=type(self).__flow_model_factory_kwargs__,
        )
        return (_new_local_flow_model_for_pickle, (cloudpickle.dumps(payload, protocol=5),), state)

    @model_validator(mode="after")
    def _validate_flow_model_fields(self):
        """Validate all bound regular and contextual defaults after pydantic construction."""

        config = self.__class__.__flow_model_config__

        for param in config.parameters:
            value = getattr(self, param.name, _UNSET_FLOW_INPUT)
            if _is_unset_flow_input(value):
                continue
            if not param.is_contextual:
                value = _resolve_bound_param_registry_ref(param, value)
            object.__setattr__(
                self,
                param.name,
                _validate_bound_param_value(config, param, value, "Contextual default" if param.is_contextual else "Field"),
            )

        _validate_bound_declared_context_defaults(self, config)
        return self

    @property
    def context_type(self) -> Type[ContextBase]:
        return self.__class__.__flow_model_config__.context_type

    @property
    def result_type(self) -> Type[ResultBase]:
        return self.__class__.__flow_model_config__.result_type

    def _evaluation_identity_payload(
        self,
        context: ContextBase,
    ) -> Optional[Any]:
        return _generated_model_identity_payload(self, context)


# ---------------------------------------------------------------------------
# Generated model method builders and decorators
# ---------------------------------------------------------------------------


def _make_call_impl(config: _FlowModelConfig) -> _AnyCallable:
    """Create the ``__call__`` implementation for one generated model class."""

    def __call__(self, context):
        """Resolve bound inputs, dependency inputs, and context inputs, then call the user function."""

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
    """Create the ``__deps__`` implementation for one generated model class."""

    def __deps__(self, context):
        """Declare non-lazy regular ``CallableModel`` inputs as graph dependencies."""

        missing_regular = _missing_regular_param_names(self, config)
        if missing_regular:
            missing = ", ".join(sorted(missing_regular))
            raise TypeError(f"Missing regular parameter(s) for {_callable_name(config.func)}: {missing}. Bind them before dependency evaluation.")

        deps = []
        for param in config.regular_params:
            if param.is_lazy:
                continue
            value = getattr(self, param.name, _UNSET_FLOW_INPUT)
            if _is_model_dependency(value):
                dependency_model, dependency_context = _resolved_dependency_invocation(value, context)
                deps.append((dependency_model, [dependency_context]))
            elif param.has_dep_slots:
                deps.extend(_dep_marked_dependency_entries(value, param.annotation, context))
        return deps

    cast(Any, __deps__).__signature__ = inspect.Signature(
        parameters=[
            inspect.Parameter("self", inspect.Parameter.POSITIONAL_OR_KEYWORD),
            inspect.Parameter("context", inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=config.context_type),
        ],
        return_annotation=GraphDepList,
    )
    return __deps__


def _factory_param_annotation(param: _FlowModelParam) -> Any:
    """Return the public factory signature annotation for a user function parameter.

    The factory signature keeps the user's surface syntax visible: contextual
    params show as FromContext[T], lazy params show as Lazy[T], and regular
    params keep T. The generated Pydantic field annotation below has a different
    job: describing the stored construction value after binding.
    """

    if param.is_contextual:
        return FromContext[param.annotation]
    if param.is_lazy:
        return Lazy[param.annotation]
    return param.annotation


def _pydantic_schema_safe_annotation(annotation: Any) -> Any:
    # Only the generated Pydantic field declaration falls back to Any for known
    # Pydantic schema-build failures. Runtime coercion still builds the real
    # TypeAdapter and propagates unexpected errors.
    try:
        _type_adapter(annotation)
    except (PydanticSchemaGenerationError, PydanticUndefinedAnnotation):
        return Any
    return annotation


def _generated_field_annotation(param: _FlowModelParam) -> Any:
    """Return the generated model field annotation used for schema only.

    This differs from _factory_param_annotation: the factory signature describes
    user-facing inputs (FromContext[T], Lazy[T]), while this describes the value
    stored on the generated Pydantic model after binding. SkipValidation keeps
    this schema visible without letting Pydantic enforce it before ccflow can
    distinguish literals, dependencies, lazy deps, and contextual defaults.
    """

    if param.is_contextual:
        annotation = param.validation_annotation
    elif param.is_lazy:
        annotation = CallableModel
    elif param.annotation is Any or param.annotation is inspect.Parameter.empty:
        annotation = Any
    else:
        annotation = Union[param.annotation, CallableModel]
    return SkipValidation[_pydantic_schema_safe_annotation(annotation)]


def _factory_signature(config: _FlowModelConfig, generated_cls: Type[BaseModel]) -> inspect.Signature:
    """Return the public construction signature for a generated factory."""

    parameters = []
    for param in config.parameters:
        parameters.append(
            inspect.Parameter(
                param.name,
                inspect.Parameter.KEYWORD_ONLY,
                annotation=_factory_param_annotation(param),
                default=param.function_default if param.has_function_default else _UNSET_FLOW_INPUT,
            )
        )

    param_names = {param.name for param in config.parameters}
    for name, field_info in generated_cls.model_fields.items():
        if name == "meta" or name in param_names:
            continue
        default = _UNSET_FLOW_INPUT if field_info.is_required() or getattr(field_info, "default_factory", None) is not None else field_info.default
        parameters.append(
            inspect.Parameter(
                name,
                inspect.Parameter.KEYWORD_ONLY,
                annotation=field_info.annotation,
                default=default,
            )
        )

    return inspect.Signature(parameters=parameters, return_annotation=generated_cls)


def _context_transform_factory_signature(config: _FlowModelConfig) -> inspect.Signature:
    """Return the public binding signature for a context transform factory."""

    parameters = []
    for param in config.regular_params:
        parameters.append(
            inspect.Parameter(
                param.name,
                inspect.Parameter.KEYWORD_ONLY,
                annotation=param.annotation,
                default=param.function_default if param.has_function_default else inspect.Parameter.empty,
            )
        )
    return inspect.Signature(parameters=parameters)


def _resolve_generated_model_bases(model_base: Type[CallableModel]) -> Tuple[type, ...]:
    """Return the class bases for a generated model, preserving custom model bases."""

    if not isinstance(model_base, type) or not issubclass(model_base, CallableModel):
        raise TypeError(f"model_base must be a CallableModel subclass, got {model_base!r}")

    if issubclass(model_base, _GeneratedFlowModelBase):
        return (model_base,)
    if model_base is CallableModel:
        return (_GeneratedFlowModelBase,)
    return (_GeneratedFlowModelBase, model_base)


def _build_flow_model_factory_from_config(config: _FlowModelConfig, factory_kwargs: Dict[str, Any]) -> _AnyCallable:
    """Build the generated model class/factory from an analyzed flow-model config."""

    fn = config.func
    factory_kwargs = dict(factory_kwargs)
    model_base = factory_kwargs["model_base"]
    # Preserve one generated-model identity across local pickle/cloudpickle
    # restore.  See ``_flow_model_config_identity`` for why this must be fixed at
    # class-construction time instead of recalculated on every cache-key build.
    config_identity = factory_kwargs.setdefault("_flow_model_identity", _flow_model_config_identity(config))
    field_definitions: Dict[str, Any] = {}

    for param in config.parameters:
        annotation = _generated_field_annotation(param)
        if param.is_contextual:
            default = _unset_flow_input_field_default()
        elif param.has_function_default:
            default = param.function_default
        else:
            default = _unset_flow_input_field_default()
        field_definitions[param.name] = (annotation, default)

    GeneratedModel = cast(
        type[_GeneratedFlowModelBase],
        create_model(
            f"_{_callable_name(fn)}_Model",
            __base__=_resolve_generated_model_bases(model_base),
            __module__=getattr(fn, "__module__", __name__),
            __qualname__=f"_{_callable_name(fn)}_Model",
            **field_definitions,
        ),
    )
    GeneratedModel.__call__ = Flow.call(
        **{
            name: value
            for name in ("cacheable", "volatile", "log_level", "validate_result", "verbose", "evaluator")
            if (value := factory_kwargs.get(name, _UNSET)) is not _UNSET
        }
    )(_make_call_impl(config))
    GeneratedModel.__deps__ = Flow.deps(_make_deps_impl(config))
    update_abstractmethods(GeneratedModel)
    GeneratedModel.__flow_model_config__ = config
    GeneratedModel.__flow_model_factory_kwargs__ = factory_kwargs
    GeneratedModel.__flow_model_identity__ = config_identity
    GeneratedModel.__ccflow_tokenizer_cache__ = _generated_model_behavior_token(config_identity, model_base)
    _register_generated_model_class(config, GeneratedModel)
    GeneratedModel.model_rebuild()

    @wraps(fn)
    def factory(**kwargs) -> _GeneratedFlowModelBase:
        """Create a generated model instance with regular/contextual defaults bound."""

        return GeneratedModel(**kwargs)

    cast(Any, factory)._generated_model = GeneratedModel
    cast(Any, factory).__signature__ = _factory_signature(config, GeneratedModel)
    factory.__doc__ = fn.__doc__
    return factory


def _flow_context_transform(func: Optional[_AnyCallable] = None) -> _AnyCallable:
    """Decorator that turns a function into a serializable ``with_context`` transform factory.

    Regular parameters are bound when the transform factory is called.
    ``FromContext`` parameters are read from the runtime context when the bound
    model executes.  Transform functions returning mappings are positional patch
    transforms; transforms returning scalar values are field transforms.
    """

    def decorator(fn: _AnyCallable) -> _AnyCallable:
        """Analyze one transform function and return its binding factory."""

        _ensure_named_python_function(fn, decorator_name="@Flow.context_transform")
        resolved_hints = get_type_hints(fn, include_extras=True)
        sig = _resolved_flow_signature(
            fn,
            resolved_hints=resolved_hints,
            require_return_annotation=True,
            function_name=_callable_name(fn),
        )
        config = _analyze_flow_context_transform(fn, sig, is_model_dependency=_is_model_dependency)
        # Store the analyzed transform contract directly. Import-path detection
        # during decoration is brittle because module globals usually still point
        # at the undecorated function until the decorator returns.
        serialized_config = _serialize_context_transform_config(config)

        @wraps(fn)
        def factory(**kwargs) -> ContextTransform:
            """Bind regular transform arguments into a serializable spec."""

            return ContextTransform(
                serialized_config=serialized_config,
                bound_args=_validate_context_transform_factory_kwargs(config, kwargs),
            )

        cast(Any, factory).__flow_context_transform_config__ = config
        cast(Any, factory).__signature__ = _context_transform_factory_signature(config)
        return factory

    if func is not None:
        return decorator(func)
    return decorator


def flow_model(
    func: Optional[_AnyCallable] = None,
    *,
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
    """Decorator that generates a ``CallableModel`` class from a plain function.

    Unmarked parameters become construction-time model fields.  Parameters
    annotated as ``FromContext[T]`` are contextual inputs supplied by
    ``FlowContext``, a declared ``context_type``, ``compute(...)`` kwargs, or
    ``with_context(...)`` bindings.  The returned object is a factory that
    creates instances of the generated model class.

    Args:
        func: The function being decorated. This is passed automatically in
            bare decorator form, for example ``@Flow.model``. When using
            options, for example ``@Flow.model(auto_unwrap=True)``, Python first
            calls ``Flow.model(...)`` without a function and then applies the
            returned decorator.
        context_type: Optional ``ContextBase`` subclass used to validate all
            contextual inputs together after individual ``FromContext[...]``
            fields are resolved.
        auto_unwrap: When ``True`` and ccflow auto-wraps a plain return
            annotation in ``GenericResult[T]``, external
            ``model.flow.compute(...)`` calls return the raw ``T`` value instead
            of ``GenericResult[T]``. Dependency evaluation and direct model calls
            keep the normal ccflow result contract.
        model_base: Custom ``CallableModel`` subclass to use as the base class
            for the generated model.
        cacheable: Optional generated-model default for ``FlowOptions.cacheable``.
        volatile: Optional generated-model default for ``FlowOptions.volatile``.
        log_level: Optional generated-model default for ``FlowOptions.log_level``.
        validate_result: Optional generated-model default for
            ``FlowOptions.validate_result``.
        verbose: Optional generated-model default for ``FlowOptions.verbose``.
        evaluator: Optional generated-model default evaluator.
    """

    def decorator(fn: _AnyCallable) -> _AnyCallable:
        """Analyze one user function and synthesize its generated model class."""

        resolved_hints = get_type_hints(fn, include_extras=True)
        sig = _resolved_flow_signature(
            fn,
            resolved_hints=resolved_hints,
            require_return_annotation=True,
            function_name=_callable_name(fn),
        )
        config = _analyze_flow_model(
            fn,
            sig,
            context_type=context_type,
            auto_unwrap=auto_unwrap,
            is_model_dependency=_is_model_dependency,
        )
        factory_kwargs = {
            "context_type": context_type,
            "auto_unwrap": auto_unwrap,
            "model_base": model_base,
            "cacheable": cacheable,
            "volatile": volatile,
            "log_level": log_level,
            "validate_result": validate_result,
            "verbose": verbose,
            "evaluator": evaluator,
        }
        factory_kwargs["_flow_model_identity"] = _flow_model_config_identity(config)
        return _build_flow_model_factory_from_config(config, factory_kwargs)

    if func is not None:
        return decorator(func)
    return decorator
