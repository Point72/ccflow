import logging
import time
from datetime import timedelta
from types import MappingProxyType
from typing import Callable, Dict, List, Optional, Set, Union

import dask.base
from pydantic import Field, PrivateAttr
from typing_extensions import override

from ..base import BaseModel, make_lazy_result
from ..callable import CallableModel, ContextBase, EvaluatorBase, ModelEvaluationContext, ResultType

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
    """An evaluator that combines multiple evaluators."""

    evaluators: List[EvaluatorBase] = Field(
        description="The list of evaluators to combine. The first evaluator in the list will be called first during evaluation."
    )

    @override
    def __call__(self, context: ModelEvaluationContext) -> ResultType:
        for evaluator in self.evaluators:
            context = ModelEvaluationContext(model=evaluator, context=context, options=context.options)
        return context()


class FallbackEvaluator(EvaluatorBase):
    """An evaluator that tries a list of evaluators in turn until one succeeds."""

    evaluators: List[EvaluatorBase] = Field(description="The list of evaluators to try (in order).")

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


class LoggingEvaluator(EvaluatorBase):
    """Evaluator that logs information about evaluating the callable.

    It logs start and end times, the model name, and the context."""

    log_level: int = Field(logging.DEBUG, description="The log level for start/end of evaluation")
    verbose: bool = Field(True, description="Whether to output the model defintion as part of logging")

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


def cache_key(flow_obj: Union[ModelEvaluationContext, ContextBase, CallableModel]) -> bytes:
    """Returns a key suitable for use in caching.

    Args:
        flow_obj: The object to be tokenized to form the cache key.
    """
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


def get_dependency_graph(evaluation_context: ModelEvaluationContext) -> CallableModelGraph:
    """Get a dependency graph for a model and context based on recursive evaluation of __deps__.

    Args:
        evaluation_context: The model and context to build the graph for.
    """
    root_key = cache_key(evaluation_context)
    graph = CallableModelGraph(ids={}, graph={}, root_id=root_key)
    _build_dependency_graph(evaluation_context, graph)
    return graph


class GraphEvaluator(EvaluatorBase):
    """Evaluator that evaluates the dependency graph of callable models in topologically sorted order.

    It is suggested to combine it with a caching evaluator.
    """

    _is_evaluating: bool = PrivateAttr(False)

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
