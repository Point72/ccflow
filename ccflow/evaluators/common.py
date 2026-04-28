import itertools
import logging
import time
from contextlib import nullcontext
from datetime import timedelta
from pprint import pformat
from types import MappingProxyType
from typing import Any, Callable, Dict, List, Optional, Set, Union

import dask.base
from pydantic import Field, PrivateAttr, ValidationError, field_validator
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

__all__ = [
    "cache_key",
    "combine_evaluators",
    "FallbackEvaluator",
    "LazyEvaluator",
    "LoggingEvaluator",
    "MemoryCacheEvaluator",
    "MultiEvaluator",
    "CallableModelGraph",
    "GraphEvaluator",
    "get_dependency_graph",
]

log = logging.getLogger(__name__)


class _EffectiveEvaluationKeyUnavailable(Exception):
    """Internal signal to use the existing structural evaluation key."""


_EFFECTIVE_IDENTITY_DECLINED_ERRORS = (TypeError, ValueError, ValidationError)
_EFFECTIVE_EVALUATION_KEY_VERSION = "ccflow_effective_evaluation_key_v1"


def combine_evaluators(first: Optional[EvaluatorBase], second: Optional[EvaluatorBase]) -> EvaluatorBase:
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


class MultiEvaluator(EvaluatorBase):
    """An evaluator that combines multiple evaluators.

    Each child evaluator is wrapped in a ModelEvaluationContext using its own
    ``make_evaluation_context()`` method, so transparent children produce
    ``TransparentModelEvaluationContext`` layers that can be skipped during
    cache key computation.
    """

    evaluators: List[EvaluatorBase] = Field(
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

    evaluators: List[EvaluatorBase] = Field(description="The list of evaluators to try (in order).")

    def is_transparent(self, context: ModelEvaluationContext) -> bool:
        return all(e.is_transparent(context) for e in self.evaluators)

    @override
    def __call__(self, context: ModelEvaluationContext) -> ResultType:
        for evaluator in self.evaluators:
            try:
                return evaluator(context)
            except Exception as e:
                log.exception("Evaluator %s failed: \n%s", evaluator, e)
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


class FormatConfig(BaseModel):
    """Configuration for formatting the result of the evaluation.

    This is used by the LoggingEvaluator to control how the result is formatted.
    """

    arrow_as_polars: bool = Field(
        False,
        description="Whether to convert pyarrow tables to polars tables for formatting, as arrow formatting does not work well with large tables or provide control over options",
    )
    pformat_config: Dict[str, Any] = Field({}, description="pformat config to use for formatting data")
    polars_config: Dict[str, Any] = Field({}, description="polars config to use for formatting polars frames")
    pandas_config: Dict[str, Any] = Field({}, description="pandas config to use for formatting pandas objects")


class LoggingEvaluator(EvaluatorBase):
    """Evaluator that logs information about evaluating the callable.

    It logs start and end times, the model name, and the context."""

    log_level: int = Field(logging.DEBUG, description="The log level for start/end of evaluation")
    verbose: bool = Field(True, description="Whether to output the model definition as part of logging")
    log_result: bool = Field(False, description="Whether to log the result of the evaluation")
    format_config: FormatConfig = Field(FormatConfig(), description="Configuration for formatting the result of the evaluation if log_result=True")

    def is_transparent(self, context: ModelEvaluationContext) -> bool:
        return True

    @field_validator("log_level", mode="before")
    @classmethod
    def _validate_log_level(cls, v: Union[int, str]) -> int:
        """Validate that the log level is a valid logging level."""
        if isinstance(v, str):
            return getattr(logging, v.upper(), "")
        return v

    @override
    def __call__(self, context: ModelEvaluationContext) -> ResultType:
        model_name = context.model.meta.name or context.model.__class__.__name__
        log_level = context.options.get("log_level", self.log_level)
        verbose = context.options.get("verbose", self.verbose)
        log.log(log_level, "[%s]: Start evaluation of %s on %s.", model_name, context.fn, context.context)
        if verbose:
            log.log(log_level, "[%s]: %s", model_name, context.model)
        start = time.time()
        result = None
        try:
            result = context()
            return result
        finally:
            end = time.time()
            if self.log_result and result is not None:
                log.log(
                    log_level,
                    self._format_result(result),
                    model_name,
                    context.fn,
                    context.context,
                )
            log.log(
                log_level,
                "[%s]: End evaluation of %s on %s (time elapsed: %s).",
                model_name,
                context.fn,
                context.context,
                timedelta(seconds=end - start),
            )

    def _format_result(self, result: ResultType) -> str:
        """Handle formatting of the result"""
        # Add special formatting for eager table/data frame types embedded in the results
        import pyarrow as pa

        result_dict = result.model_dump(by_alias=True)
        for k, v in result_dict.items():
            try:
                if self.format_config.arrow_as_polars and isinstance(v, pa.Table):
                    import polars as pl  # Only import polars if needed

                    result_dict[k] = pl.from_arrow(v)
            except TypeError:
                pass

        if self.format_config.polars_config:  # Control formatting of polars tables if set
            import polars as pl  # Only import polars if needed

            polars_context = pl.Config(**self.format_config.polars_config)
        else:
            polars_context = nullcontext()

        if self.format_config.pandas_config:  # Control formatting of pandas tables if set
            import pandas as pd

            pandas_context = pd.option_context(*itertools.chain.from_iterable(self.format_config.pandas_config.items()))
        else:
            pandas_context = nullcontext()

        with polars_context, pandas_context:
            msg_str = "[%s]: Result of %s on %s:\n"
            return f"{msg_str}{pformat(result_dict, **self.format_config.pformat_config)}"


def _unwrap_evaluation_context(evaluation_context: ModelEvaluationContext) -> tuple[ModelEvaluationContext, str, List[CallableModel]]:
    """Strip transparent evaluator wrappers and keep opaque wrappers in order.

    This preserves the existing structural cache-key behavior: transparent
    evaluators are ignored, while non-transparent evaluators remain part of the
    identity. The returned function name is the innermost non-``__call__`` name,
    so ``__deps__`` does not collapse into ``__call__`` when wrapped.
    """
    fn = evaluation_context.fn
    outer_to_inner_evaluators = []
    while isinstance(evaluation_context.context, ModelEvaluationContext):
        fn = evaluation_context.fn if evaluation_context.fn != "__call__" else fn
        if not isinstance(evaluation_context, TransparentModelEvaluationContext):
            outer_to_inner_evaluators.append(evaluation_context.model)
        evaluation_context = evaluation_context.context
    return evaluation_context, fn if fn != "__call__" else evaluation_context.fn, outer_to_inner_evaluators


def _evaluator_identity_payload(outer_to_inner_evaluators: List[CallableModel]) -> List[Dict[str, Any]]:
    return [evaluator.model_dump(mode="python") for evaluator in outer_to_inner_evaluators]


def _memo_token(model: CallableModel, context: Any) -> tuple[int, str]:
    if hasattr(context, "model_dump"):
        context_value = context.model_dump(mode="python")
    else:
        context_value = context
    return (id(model), dask.base.tokenize((type(context), context_value)))


def _effective_model_key(
    model: CallableModel,
    context: Any,
    memo: Dict[tuple[int, str], bytes],
    active: Set[tuple[int, str]],
) -> Optional[bytes]:
    """Return a model's opt-in effective key, or ``None`` for normal opt-out.

    Plain ``CallableModel`` instances opt out by returning ``None`` from
    ``_evaluation_identity_payload()``. Dependency invocations inside the
    payload are resolved by ``_resolve_effective_identity_payload()`` so models
    declare what matters without constructing recursive keys themselves.
    """
    token = _memo_token(model, context)
    if token in memo:
        return memo[token]
    if token in active:
        raise _EffectiveEvaluationKeyUnavailable("recursive effective identity")

    active.add(token)
    try:
        try:
            payload = model._evaluation_identity_payload(context)
        except _EFFECTIVE_IDENTITY_DECLINED_ERRORS as exc:
            # Identity derivation runs before the actual call and may encounter
            # the same validation failures as evaluation context construction.
            # Falling back preserves existing behavior instead of turning key
            # computation into a new failure mode for ordinary models.
            raise _EffectiveEvaluationKeyUnavailable(str(exc)) from exc
        # For normal CallableModels, `_evaluation_identity_payload` defaults to
        # None, so we should hit this path
        if payload is None:
            return None
        payload = _resolve_effective_identity_payload(payload, memo, active)
        key = dask.base.tokenize((_EFFECTIVE_EVALUATION_KEY_VERSION, payload)).encode("utf-8")
        memo[token] = key
        return key
    finally:
        active.discard(token)


def _resolve_effective_identity_payload(
    value: Any,
    memo: Dict[tuple[int, str], bytes],
    active: Set[tuple[int, str]],
) -> Any:
    """Replace dependency invocation markers with recursive effective keys."""
    if isinstance(value, EvaluationDependency):
        try:
            evaluation = value.model.__call__.get_evaluation_context(value.model, value.context)
        except _EFFECTIVE_IDENTITY_DECLINED_ERRORS as exc:
            raise _EffectiveEvaluationKeyUnavailable(f"dependency {type(value.model).__name__} could not build evaluation context: {exc}") from exc
        return _effective_evaluation_key(evaluation, memo=memo, active=active)
    if isinstance(value, dict):
        return {key: _resolve_effective_identity_payload(item, memo, active) for key, item in value.items()}
    if isinstance(value, list):
        return [_resolve_effective_identity_payload(item, memo, active) for item in value]
    if isinstance(value, tuple):
        return tuple(_resolve_effective_identity_payload(item, memo, active) for item in value)
    return value


def _effective_evaluation_key(
    evaluation_context: ModelEvaluationContext,
    memo: Optional[Dict[tuple[int, str], bytes]] = None,
    active: Optional[Set[tuple[int, str]]] = None,
) -> bytes:
    """Use opt-in effective identity for ``__call__``; otherwise preserve ``cache_key()``."""
    memo = {} if memo is None else memo
    active = set() if active is None else active
    inner, fn, outer_to_inner_evaluators = _unwrap_evaluation_context(evaluation_context)
    if fn != "__call__":
        # Keep non-call evaluations, especially ``__deps__``, on the exact
        # public structural key path. Effective identity is only meant to
        # narrow normal model execution where generated ``@Flow.model`` code
        # knows which ambient ``FlowContext`` fields affect the result.
        return cache_key(evaluation_context)

    try:
        key = _effective_model_key(inner.model, inner.context, memo, active)
    except _EffectiveEvaluationKeyUnavailable as exc:
        # Effective identity is an optimization/semantic narrowing for opt-in
        # generated models. If deriving it is unclear, do not make cache/graph
        # key construction a new failure mode; use the old structural key.
        log.debug("Falling back to structural evaluation key for %s.__call__: %s", type(inner.model).__name__, exc)
        return cache_key(evaluation_context)
    if key is None:
        # This is the ordinary path for existing CallableModel classes. The
        # base implementation returns None, so their cache and graph identities
        # remain byte-for-byte equivalent to ``cache_key(evaluation_context)``.
        return cache_key(evaluation_context)

    # Preserve the existing evaluation-context semantics around the narrowed
    # model key: options still distinguish evaluations, transparent evaluators
    # are ignored, and opaque evaluators remain part of identity.
    return dask.base.tokenize(
        (
            _EFFECTIVE_EVALUATION_KEY_VERSION,
            "evaluation_context",
            {
                "fn": fn,
                "options": inner.options,
                "_evaluators": _evaluator_identity_payload(outer_to_inner_evaluators),
            },
            key,
        )
    ).encode("utf-8")


def cache_key(flow_obj: Union[ModelEvaluationContext, ContextBase, CallableModel]) -> bytes:
    """Returns a structural key suitable for caching and dependency graph deduplication.

    For ``ModelEvaluationContext`` inputs, strips ``TransparentModelEvaluationContext``
    layers (evaluators that don't modify the return value) so that the key depends
    only on the underlying model, context, fn, options, and any non-transparent
    evaluators in the chain.

    Args:
        flow_obj: The object to be tokenized to form the cache key.
    """
    if isinstance(flow_obj, ModelEvaluationContext):
        fn = flow_obj.fn
        non_transparent = []
        while isinstance(flow_obj.context, ModelEvaluationContext):
            fn = flow_obj.fn if flow_obj.fn != "__call__" else fn
            if not isinstance(flow_obj, TransparentModelEvaluationContext):
                non_transparent.append(flow_obj.model)
            flow_obj = flow_obj.context
        d = flow_obj.model_dump(mode="python")
        d["fn"] = fn if fn != "__call__" else flow_obj.fn
        if non_transparent:
            d["_evaluators"] = [e.model_dump(mode="python") for e in non_transparent]
        return dask.base.tokenize(d).encode("utf-8")
    elif isinstance(flow_obj, (ContextBase, CallableModel)):
        return dask.base.tokenize(flow_obj.model_dump(mode="python")).encode("utf-8")
    else:
        raise TypeError(f"object of type {type(flow_obj)} cannot be serialized by this function!")


class MemoryCacheEvaluator(EvaluatorBase):
    """Evaluator that caches results in memory."""

    # Note: We make the cache attributes private, so they don't affect tokenization of the MemoryCacheEvaluator itself
    _cache: Dict[bytes, ResultType] = PrivateAttr({})
    _ids: Dict[bytes, ModelEvaluationContext] = PrivateAttr({})

    def is_transparent(self, context: ModelEvaluationContext) -> bool:
        return True

    def key(self, context: ModelEvaluationContext):
        """Function to convert a ModelEvaluationContext to a cache key.

        This is the only cache entry point that uses effective identity.
        Generated ``@Flow.model`` instances can opt in to a narrower key that
        ignores unused ambient context fields; ordinary ``CallableModel`` paths
        fall back to the same structural key returned by ``cache_key()``.
        """
        # Do not route callers of public ``cache_key()`` through this helper.
        # Keeping the effective key private to the evaluator makes the new
        # behavior additive: normal key introspection and non-opt-in models stay
        # on the structural implementation above.
        return _effective_evaluation_key(context)

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
    """Class to hold a "graph" """

    graph: Dict[bytes, Set[bytes]]
    ids: Dict[bytes, ModelEvaluationContext]
    root_id: bytes


def _is_wrapper_to_wrapped_edge(parent_model: Optional[CallableModel], current_model: CallableModel) -> bool:
    # Effective identity can intentionally collapse a wrapper model and its
    # wrapped model to the same graph/cache key. Only that wrapper-to-wrapped
    # edge should be treated as a duplicate self-edge.
    return isinstance(parent_model, WrapperModel) and parent_model.model is current_model


def _build_dependency_graph(
    evaluation_context: ModelEvaluationContext,
    graph: CallableModelGraph,
    parent_key: Optional[bytes] = None,
    parent_model: Optional[CallableModel] = None,
):
    # Generated/bound ``@Flow.model`` nodes can use effective identity so unused
    # ambient FlowContext fields do not split the graph. Normal CallableModel
    # nodes opt out and therefore still receive ``cache_key(evaluation_context)``.
    key = _effective_evaluation_key(evaluation_context)
    unwrapped_evaluation_context, _, _ = _unwrap_evaluation_context(evaluation_context)
    current_model = unwrapped_evaluation_context.model
    is_same_evaluation_key = parent_key == key
    is_collapsed_wrapper_child = is_same_evaluation_key and _is_wrapper_to_wrapped_edge(parent_model, current_model)

    # Bound/wrapper models can intentionally collapse to their wrapped model's
    # effective evaluation identity. Adding the wrapper -> wrapped edge in that
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

    # Main used ``key not in graph.graph`` as the traversal guard. That is no
    # longer enough once effective identity can collapse multiple model objects
    # to one key: a bound wrapper and its wrapped model may share the graph node,
    # but the wrapped model still has dependencies that must be traversed.
    #
    # Preserve normal graph deduplication by key, and make the only exception
    # the exact collapsed wrapper -> wrapped edge.
    if not is_new_graph_key and not is_collapsed_wrapper_child:
        return

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


def get_dependency_graph(evaluation_context: ModelEvaluationContext) -> CallableModelGraph:
    """Get a dependency graph for a model and context based on recursive evaluation of __deps__.

    Args:
        evaluation_context: The model and context to build the graph for.
    """
    # Keep the root id on the same identity function used for every graph node.
    # For existing models this is still ``cache_key(evaluation_context)``; for
    # generated flow models it is the narrowed key that ignores unused ambient
    # context fields.
    root_key = _effective_evaluation_key(evaluation_context)
    graph = CallableModelGraph(ids={}, graph={}, root_id=root_key)
    _build_dependency_graph(evaluation_context, graph)
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
        # no not apply it any further
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
