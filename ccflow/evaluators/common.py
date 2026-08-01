import logging
from collections.abc import Callable
from types import MappingProxyType
from typing import Any

from pydantic import Field, PrivateAttr
from typing_extensions import override

from ..base import BaseModel, make_lazy_result
from ..callable import (
    CallableModel,
    ContextBase,
    EvaluationDependency,
    EvaluatorBase,
    ModelEvaluationContext,
    ResultType,
    TransparentModelEvaluationContext,
    WrapperModel,
)
from ..utils.reporting import FormatConfig, LoggingPolicy
from ..utils.tokenize import compute_cache_token
from .reporting import ReportingEvaluator, _descriptor

__all__ = [
    "CallableModelGraph",
    "FallbackEvaluator",
    "FormatConfig",
    "GraphEvaluator",
    "LazyEvaluator",
    "LoggingEvaluator",
    "MemoryCacheEvaluator",
    "MultiEvaluator",
    "cache_key",
    "combine_evaluators",
    "get_dependency_graph",
]

log = logging.getLogger(__name__)


class _EffectiveEvaluationKeyUnavailable(Exception):
    """Internal signal to use the existing structural evaluation key."""


_EFFECTIVE_EVALUATION_KEY_VERSION = "ccflow_effective_evaluation_key_v1"
_RECURSIVE_EFFECTIVE_IDENTITY_SENTINEL = "recursive_effective_identity"


class _IdentityMemoKey:
    """Identity-based key that keeps objects alive while effective keys recurse.

    Raw ``id(...)`` tuples are unsafe here because rewritten context objects can
    be short-lived and Python may reuse their ids during a single graph build.
    Holding references keeps the ids stable for the lifetime of the memo key;
    equality still checks object identity, so hash collisions are harmless.
    """

    __slots__ = ("_hash", "context", "model")

    def __init__(self, model: CallableModel, context: Any):
        self.model = model
        self.context = context
        self._hash = hash((id(model), id(context)))

    def __hash__(self) -> int:
        return self._hash

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _IdentityMemoKey) and self.model is other.model and self.context is other.context


def combine_evaluators(first: EvaluatorBase | None, second: EvaluatorBase | None) -> EvaluatorBase:
    """Helper function to combine evaluators into a new evaluator.

    Args:
        first: The first evaluator to combine.
        second: The second evaluator to combine.
    """
    if not first:
        return second
    elif not second:
        return first
    elif isinstance(first, MultiEvaluator):
        if isinstance(second, MultiEvaluator):
            return MultiEvaluator(evaluators=first.evaluators + second.evaluators)
        else:
            return MultiEvaluator(evaluators=first.evaluators + [second])
    elif isinstance(second, MultiEvaluator):
        return MultiEvaluator(evaluators=[first] + second.evaluators)
    else:
        return MultiEvaluator(evaluators=[first, second])


def _flatten_cache_key_context(flow_obj: ModelEvaluationContext) -> tuple[ModelEvaluationContext, str, list[EvaluatorBase]]:
    fn = flow_obj.fn
    non_transparent: list[EvaluatorBase] = []
    while isinstance(flow_obj.context, ModelEvaluationContext):
        fn = flow_obj.fn if flow_obj.fn != "__call__" else fn
        if not isinstance(flow_obj, TransparentModelEvaluationContext):
            non_transparent.append(flow_obj.model)
        flow_obj = flow_obj.context
    return flow_obj, fn if fn != "__call__" else flow_obj.fn, non_transparent


class MultiEvaluator(EvaluatorBase):
    """An evaluator that combines multiple evaluators.

    Each child evaluator is wrapped in a ModelEvaluationContext using its own
    ``make_evaluation_context()`` method, so transparent children produce
    ``TransparentModelEvaluationContext`` layers that can be skipped during
    cache key computation.
    """

    evaluators: list[EvaluatorBase] = Field(
        description="The list of evaluators to combine. The first evaluator in the list will be called first during evaluation."
    )

    def is_transparent(self, context: ModelEvaluationContext) -> bool:
        return all(e.is_transparent(context) for e in self.evaluators)

    @override
    def __call__(self, context: ModelEvaluationContext) -> ResultType:
        for evaluator in self.evaluators:
            context = evaluator.make_evaluation_context(context, options=context.options)
        return context()


class FallbackEvaluator(EvaluatorBase):
    """An evaluator that tries a list of evaluators in turn until one succeeds."""

    evaluators: list[EvaluatorBase] = Field(description="The list of evaluators to try (in order).")

    def is_transparent(self, context: ModelEvaluationContext) -> bool:
        return all(e.is_transparent(context) for e in self.evaluators)

    @override
    def __call__(self, context: ModelEvaluationContext) -> ResultType:
        for evaluator in self.evaluators:
            try:
                return evaluator(context)
            except Exception:
                log.exception("Evaluator %s failed", evaluator)
        raise RuntimeError("All evaluators failed.")


class LazyEvaluator(EvaluatorBase):
    """Evaluator that only actually runs the callable once an attribute of the result is queried (by hooking into __getattribute__)"""

    additional_callback: Callable = Field(lambda: None, description="An additional callback that will be invoked before the evaluation takes place.")

    @override
    def __call__(self, context: ModelEvaluationContext) -> ResultType:
        def make_result():
            self.additional_callback()
            return context()

        return make_lazy_result(context.model.result_type, make_result)


class LoggingEvaluator(ReportingEvaluator, LoggingPolicy):
    """Evaluator that logs information about evaluating the callable.

    It logs start and end times, the model name, and the context. This is the *default* evaluator
    when no other is configured. It is now a thin combination of :class:`ReportingEvaluator` (span /
    contextvar correlation, optional structured events when a ``reporter`` is set) and
    :class:`~ccflow.utils.reporting.LoggingPolicy` (the actual log output, preserved exactly), so it
    also participates in the reporting span tree.
    """

    @override
    def __call__(self, context: ModelEvaluationContext) -> ResultType:
        return self._run_with_reporting(
            context,
            extra={"model": context.model, "raw_context": context.context, "options": dict(context.options)},
            **_descriptor(context),
        )


def _effective_model_key(
    model: CallableModel,
    context: Any,
    memo: dict[_IdentityMemoKey, bytes],
    active: set[int],
) -> bytes | None:
    """Return a model's opt-in effective key, or ``None`` for normal opt-out.

    Plain ``CallableModel`` instances opt out by returning ``None`` from
    ``_evaluation_identity_payload()``. Dependency invocations inside the
    payload are resolved by ``_resolve_effective_identity_payload()`` so models
    declare what matters without constructing recursive keys themselves.
    """
    token = _IdentityMemoKey(model, context)
    if token in memo:
        return memo[token]
    model_id = id(model)
    if model_id in active:
        raise _EffectiveEvaluationKeyUnavailable("recursive effective identity")

    active.add(model_id)
    try:
        payload = model._evaluation_identity_payload(context)
        # For normal CallableModels, `_evaluation_identity_payload` defaults to
        # None, so we should hit this path
        if payload is None:
            return None
        payload = _resolve_effective_identity_payload(payload, memo, active)
        key = compute_cache_token(
            data_values=[(_EFFECTIVE_EVALUATION_KEY_VERSION, payload)],
            behavior_classes=[type(model)],
        ).encode("utf-8")
        memo[token] = key
        return key
    finally:
        active.discard(model_id)


def _resolve_effective_identity_payload(
    value: Any,
    memo: dict[_IdentityMemoKey, bytes],
    active: set[int],
) -> Any:
    """Replace dependency invocation markers with recursive effective keys."""
    if isinstance(value, EvaluationDependency):
        evaluation = value.model.__call__.get_evaluation_context(value.model, value.context)
        try:
            return _effective_evaluation_key(evaluation, memo=memo, active=active, fallback=False)
        except _EffectiveEvaluationKeyUnavailable:
            return (_RECURSIVE_EFFECTIVE_IDENTITY_SENTINEL, type(value.model).__module__, type(value.model).__qualname__)
    if isinstance(value, dict):
        return {key: _resolve_effective_identity_payload(item, memo, active) for key, item in value.items()}
    if isinstance(value, list):
        return [_resolve_effective_identity_payload(item, memo, active) for item in value]
    if isinstance(value, tuple):
        return tuple(_resolve_effective_identity_payload(item, memo, active) for item in value)
    return value


def _effective_evaluation_key(
    evaluation_context: ModelEvaluationContext,
    memo: dict[_IdentityMemoKey, bytes] | None = None,
    active: set[int] | None = None,
    fallback: bool = True,
) -> bytes:
    """Use opt-in effective identity for ``__call__``; otherwise preserve ``cache_key()``."""
    memo = {} if memo is None else memo
    active = set() if active is None else active
    inner, fn, outer_to_inner_evaluators = _flatten_cache_key_context(evaluation_context)
    if fn != "__call__":
        # Keep non-call evaluations, especially ``__deps__``, on the exact
        # public structural key path. Effective identity is only meant to
        # narrow normal model execution where generated ``@Flow.model`` code
        # knows which ambient ``FlowContext`` fields affect the result.
        return cache_key(evaluation_context)
    if outer_to_inner_evaluators:
        # Non-transparent evaluators can inspect the full ModelEvaluationContext
        # and change the returned value based on ambient context fields that an
        # opt-in model would otherwise ignore. Use the structural key whenever
        # such an evaluator is part of the call chain; missing an optimization is
        # preferable to returning a value cached under a narrower model identity.
        return cache_key(evaluation_context)

    try:
        key = _effective_model_key(inner.model, inner.context, memo, active)
    except _EffectiveEvaluationKeyUnavailable as exc:
        if not fallback:
            raise
        # Effective identity is an optimization/semantic narrowing for opt-in
        # generated models. If deriving it is unclear, do not make cache/graph
        # key construction a failure mode; fall back to the structural key.
        log.debug("Falling back to structural evaluation key for %s.__call__: %s", type(inner.model).__name__, exc)
        return cache_key(evaluation_context)
    if key is None:
        # This is the ordinary path for existing CallableModel classes. The
        # base implementation returns None, so their cache and graph identities
        # remain byte-for-byte equivalent to ``cache_key(evaluation_context)``.
        return cache_key(evaluation_context)

    # Preserve the existing evaluation-context semantics around the narrowed
    # model key: options still distinguish evaluations and transparent
    # evaluators are ignored.
    return compute_cache_token(
        data_values=[
            (
                _EFFECTIVE_EVALUATION_KEY_VERSION,
                "evaluation_context",
                {
                    "fn": fn,
                    "options": inner.options,
                },
                key,
            )
        ],
    ).encode("utf-8")


def cache_key(flow_obj: ModelEvaluationContext | ContextBase | CallableModel, *, effective: bool = False) -> bytes:
    """Returns a key suitable for caching and dependency graph deduplication.

    For ``ModelEvaluationContext`` inputs, strips ``TransparentModelEvaluationContext``
    layers (evaluators that don't modify the return value) so that the key depends
    only on the underlying model, context, fn, options, and any non-transparent
    evaluators in the chain.

    By default, this key is structural.  Passing ``effective=True`` for a
    ``ModelEvaluationContext`` enables generated-model effective identity, which
    lets opt-in models ignore unused ambient context fields.  Non-opt-in models
    and non-evaluation inputs still use the structural key.

    When the underlying model has callable methods, a behavior token (SHA-256 of
    method bytecode) is included so that code changes invalidate the cache.

    Args:
        flow_obj: The object to be tokenized to form the cache key.
        effective: Whether to use generated-model effective identity for model
            evaluations that opt into it. Defaults to ``False`` to preserve the
            public structural semantics.
    """
    if effective and isinstance(flow_obj, ModelEvaluationContext):
        return _effective_evaluation_key(flow_obj)

    if isinstance(flow_obj, ModelEvaluationContext):
        flow_obj, fn, non_transparent = _flatten_cache_key_context(flow_obj)
        return compute_cache_token(
            data_values=[
                {**flow_obj.model_dump(mode="python"), "fn": fn},
                *(evaluator.model_dump(mode="python") for evaluator in non_transparent),
            ],
            behavior_classes=[type(flow_obj.model), *(type(evaluator) for evaluator in non_transparent)],
        ).encode("utf-8")
    elif isinstance(flow_obj, (ContextBase, CallableModel)):
        return compute_cache_token(
            data_values=[flow_obj.model_dump(mode="python")],
            behavior_classes=[type(flow_obj)],
        ).encode("utf-8")
    else:
        raise TypeError(f"object of type {type(flow_obj)} cannot be serialized by this function!")


class MemoryCacheEvaluator(EvaluatorBase):
    """Evaluator that caches results in memory."""

    # Note: We make the cache attributes private, so they don't affect tokenization of the MemoryCacheEvaluator itself
    _cache: dict[bytes, ResultType] = PrivateAttr({})
    _ids: dict[bytes, ModelEvaluationContext] = PrivateAttr({})

    def is_transparent(self, context: ModelEvaluationContext) -> bool:
        return True

    def key(self, context: ModelEvaluationContext):
        """Function to convert a ModelEvaluationContext to a cache key.

        Generated ``@Flow.model`` instances can use a narrower effective key
        that ignores unused ambient context fields; ordinary ``CallableModel``
        paths fall back to the structural key.
        """
        return cache_key(context, effective=True)

    @property
    def cache(self):
        """The cache values for introspection"""
        return MappingProxyType(self._cache)

    @property
    def ids(self):
        """The mapping of cache keys to ModelEvaluationContext"""
        return MappingProxyType(self._ids)

    @override
    def __call__(self, context: ModelEvaluationContext) -> ResultType:
        if context.options.get("volatile") or not context.options.get("cacheable"):
            return context()
        key = self.key(context)
        if key not in self._cache:
            self._ids[key] = context
            self._cache[key] = context()
        return self._cache[key]

    def __deepcopy__(self, memo):
        # Without this, when the framework makes deep copies of the evaluator (when used in the ModelEvaluationContext),
        # it will create new evaluators with a different cache, rather than re-using the same cache.
        return self


class CallableModelGraph(BaseModel):
    """Dependency graph of callable-model evaluation contexts.

    ``graph`` maps each node's cache key to the set of its dependency cache keys, ``ids`` maps cache
    keys back to their :class:`ModelEvaluationContext`, and ``root_id`` is the cache key of the node
    the graph was built from.
    """

    graph: dict[bytes, set[bytes]]
    ids: dict[bytes, ModelEvaluationContext]
    root_id: bytes


def _build_dependency_graph(
    evaluation_context: ModelEvaluationContext,
    graph: CallableModelGraph,
    parent_key: bytes | None = None,
    parent_model: CallableModel | None = None,
) -> bytes:
    # Generated/bound ``@Flow.model`` nodes can use effective identity so unused
    # ambient FlowContext fields do not split the graph. Normal CallableModel
    # nodes opt out and therefore still receive ``cache_key(evaluation_context)``.
    key = _effective_evaluation_key(evaluation_context)
    unwrapped_evaluation_context, _, _ = _flatten_cache_key_context(evaluation_context)
    current_model = unwrapped_evaluation_context.model
    is_same_evaluation_key = parent_key == key
    is_collapsed_wrapper_child = is_same_evaluation_key and isinstance(parent_model, WrapperModel) and parent_model.model is current_model

    # Bound/wrapper models can share an effective graph key with their wrapped
    # model after context rewriting. Adding the wrapper -> wrapped edge in that
    # case would create a fake self-loop in the public graph because both ends
    # have the same key. Suppress only that edge; real cycles between ordinary
    # models are still recorded.
    if parent_key and not is_collapsed_wrapper_child:
        graph.graph[parent_key].add(key)
    if key not in graph.ids:
        graph.ids[key] = evaluation_context
    is_new_graph_key = key not in graph.graph
    if is_new_graph_key:
        graph.graph[key] = set()

    # Effective identity can merge multiple model objects into one graph key.
    # A bound wrapper and its wrapped model may share the graph node, but the
    # wrapped model still has dependencies that must be traversed. Preserve
    # normal graph deduplication by key, and make the only exception the exact
    # same-key wrapper -> wrapped edge.
    if not is_new_graph_key and not is_collapsed_wrapper_child:
        return key

    # Note that __deps__ will be evaluated using whatever evaluator is configured for the model,
    # which could include logging, caching, etc.
    deps = evaluation_context.model.__deps__(evaluation_context.context)
    # Recursively walk dependency contexts depth-first to build the complete graph.
    for model, contexts in deps:
        for context in contexts:
            sub_evaluation_context = model.__call__.get_evaluation_context(model, context)
            _build_dependency_graph(
                sub_evaluation_context,
                graph,
                parent_key=key,
                parent_model=current_model,
            )
    return key


def get_dependency_graph(evaluation_context: ModelEvaluationContext) -> CallableModelGraph:
    """Get a dependency graph for a model and context based on recursive evaluation of __deps__.

    Args:
        evaluation_context: The model and context to build the graph for.
    """
    graph = CallableModelGraph(ids={}, graph={}, root_id=b"")
    graph.root_id = _build_dependency_graph(evaluation_context, graph)
    return graph


class GraphEvaluator(EvaluatorBase):
    """Evaluator that evaluates the dependency graph of callable models in topologically sorted order.

    It is suggested to combine it with a caching evaluator.
    """

    _is_evaluating: bool = PrivateAttr(False)

    def is_transparent(self, context: ModelEvaluationContext) -> bool:
        return True

    @override
    def __call__(self, context: ModelEvaluationContext) -> ResultType:
        import graphlib

        # If we are evaluating deps, or if we have already started using the graph evaluator further up the call tree,
        # do not apply it any further
        if self._is_evaluating:
            return context()
        self._is_evaluating = True
        root_result = None
        try:
            graph = get_dependency_graph(context)
            ts = graphlib.TopologicalSorter(graph.graph)
            for key in ts.static_order():
                evaluation_context = graph.ids[key]
                result = evaluation_context()
                if key == graph.root_id:
                    root_result = result
        finally:
            self._is_evaluating = False
        return root_result
