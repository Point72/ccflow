import logging
import time
from datetime import time as dt_time
from datetime import timedelta
from types import MappingProxyType
from typing import Callable, Dict, List, Optional, Set, Union

import dask.base
from dask.base import normalize_token
from pydantic import PrivateAttr
from typing_extensions import override

from ..base import BaseModel
from ..callable import CallableModel, ContextBase, EvaluatorBase, ModelEvaluationContext, ResultType

__all__ = [
    "cache_key",
    "combine_evaluators",
    "LazyEvaluator",
    "LoggingEvaluator",
    "MemoryCacheEvaluator",
    "MultiEvaluator",
    "CallableModelGraph",
    "GraphEvaluator",
    "get_dependency_graph",
]

log = logging.getLogger(__name__)


def combine_evaluators(first: Optional[EvaluatorBase], second: Optional[EvaluatorBase]) -> EvaluatorBase:
    """Helper function to combine evaluators into a new evaluator."""
    if first is None:
        return second
    elif second is None:
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
    """An evaluator that combines multiple evaluators."""

    evaluators: List[EvaluatorBase]

    @override
    def __call__(self, context: ModelEvaluationContext) -> ResultType:
        for evaluator in self.evaluators:
            context = ModelEvaluationContext(model=evaluator, context=context, options=context.options)
        return context()


class LazyEvaluator(EvaluatorBase):
    """Evaluator that only actually runs the callable once an attribute of the result is queried (by hooking __getattribute__)"""

    additional_callback: Callable = lambda: None

    @override
    def __call__(self, context: ModelEvaluationContext) -> ResultType:
        from ccflow.base import make_lazy_result

        def make_result():
            self.additional_callback()
            return context()

        return make_lazy_result(context.model.result_type, make_result)


class LoggingEvaluator(EvaluatorBase):
    """Evaluator that logs information about evaluating the callable (e.g. context the model was called with)."""

    log_level: int = logging.DEBUG
    verbose: bool = True

    @override
    def __call__(self, context: ModelEvaluationContext) -> ResultType:
        model_name = context.model.meta.name or context.model.__class__.__name__
        log_level = context.options.get("log_level", self.log_level)
        verbose = context.options.get("verbose", self.verbose)
        log.log(log_level, "[%s]: Start evaluation of %s on %s.", model_name, context.fn, context.context)
        if verbose:
            log.log(log_level, "[%s]: %s", model_name, context.model)
        start = time.time()
        try:
            return context()
        finally:
            end = time.time()
            log.log(
                log_level,
                "[%s]: End evaluation of %s on %s (time elapsed: %s).",
                model_name,
                context.fn,
                context.context,
                timedelta(seconds=end - start),
            )


@normalize_token.register(dt_time)
def tokenize_bar(t):
    """Custom tokenization of time"""
    # TODO: Remove after this is merged: https://github.com/dask/dask/pull/9528
    return hash(t)


def cache_key(flow_obj: Union[ModelEvaluationContext, ContextBase, CallableModel]) -> bytes:
    """Returns a key suitable for use in caching"""
    if isinstance(flow_obj, (ModelEvaluationContext, ContextBase, CallableModel)):
        return dask.base.tokenize(flow_obj.model_dump(mode="python")).encode("utf-8")
    else:
        raise TypeError(f"object of type {type(flow_obj)} cannot be serialized by this function!")


class MemoryCacheEvaluator(EvaluatorBase):
    """Evaluator that caches results in memory."""

    # Note: We make the cache attributes private, so they don't affect tokenization of the MemoryCacheEvaluator itself
    _cache: Dict[bytes, ResultType] = PrivateAttr({})
    _ids: Dict[bytes, ModelEvaluationContext] = PrivateAttr({})

    def key(self, context: ModelEvaluationContext):
        """Function to convert a ModelEvaluationContext to a key"""
        return cache_key(context)

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


def _build_dependency_graph(evaluation_context: ModelEvaluationContext, graph: CallableModelGraph, parent_key: Optional[bytes] = None):
    key = cache_key(evaluation_context)
    if parent_key:
        graph.graph[parent_key].add(key)
    if key not in graph.ids:
        graph.ids[key] = evaluation_context
    if key not in graph.graph:
        graph.graph[key] = set()
        # Note that __deps__ will be evaluated using whatever evaluator is configured for the model,
        # which could include logging, caching, etc.
        deps = evaluation_context.model.__deps__(evaluation_context.context)
        # Sequential evaluation of dependencies of dependencies (could have other implementations)
        for model, contexts in deps:
            for context in contexts:
                sub_evaluation_context = model.__call__.get_evaluation_context(model, context)
                _build_dependency_graph(sub_evaluation_context, graph, parent_key=key)


def get_dependency_graph(
    evaluation_context: ModelEvaluationContext,
) -> CallableModelGraph:
    """Get a dependency graph for a model and context."""
    root_key = cache_key(evaluation_context)
    graph = CallableModelGraph(ids={}, graph={}, root_id=root_key)
    _build_dependency_graph(evaluation_context, graph)
    return graph


class GraphEvaluator(EvaluatorBase):
    """Evaluator that evaluates the dependency graph of callable models first."""

    is_evaluating: bool = False

    @override
    def __call__(self, context: ModelEvaluationContext) -> ResultType:
        import graphlib

        # If we are evaluating deps, or if we have already started using the graph evaluator further up the call tree,
        # no not apply it any further
        if self.is_evaluating:
            return context()
        self.is_evaluating = True
        try:
            graph = get_dependency_graph(context)
            ts = graphlib.TopologicalSorter(graph.graph)
            for key in ts.static_order():
                evaluation_context = graph.ids[key]
                evaluation_context()
        finally:
            self.is_evaluating = False
