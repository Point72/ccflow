import logging
from types import MappingProxyType
from typing import Callable, Dict, List, Optional, Set, Union

from pydantic import Field, PrivateAttr
from typing_extensions import override

from ..base import BaseModel, make_lazy_result
from ..callable import (
    CallableModel,
    ContextBase,
    EvaluatorBase,
    ModelEvaluationContext,
    ResultType,
    TransparentModelEvaluationContext,
)
from ..utils.reporting import FormatConfig, LoggingPolicy
from .reporting import ReportingEvaluator, _descriptor

__all__ = [
    "cache_key",
    "combine_evaluators",
    "FallbackEvaluator",
    "FormatConfig",
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


def _flatten_cache_key_context(flow_obj: ModelEvaluationContext) -> tuple[ModelEvaluationContext, str, List[EvaluatorBase]]:
    fn = flow_obj.fn
    non_transparent: List[EvaluatorBase] = []
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


def cache_key(flow_obj: Union[ModelEvaluationContext, ContextBase, CallableModel]) -> bytes:
    """Returns a key suitable for use in caching and dependency graph deduplication.

    For ``ModelEvaluationContext`` inputs, strips ``TransparentModelEvaluationContext``
    layers (evaluators that don't modify the return value) so that the key depends
    only on the underlying model, context, fn, options, and any non-transparent
    evaluators in the chain.

    When the underlying model has callable methods, a behavior token (SHA-256 of
    method bytecode) is included so that code changes invalidate the cache.

    Args:
        flow_obj: The object to be tokenized to form the cache key.
    """
    from ..utils.tokenize import compute_cache_token

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
    _cache: Dict[bytes, ResultType] = PrivateAttr({})
    _ids: Dict[bytes, ModelEvaluationContext] = PrivateAttr({})

    def is_transparent(self, context: ModelEvaluationContext) -> bool:
        return True

    def key(self, context: ModelEvaluationContext):
        """Function to convert a ModelEvaluationContext to a cache key.

        Delegates to ``cache_key()`` which strips transparent evaluator layers.
        """
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
    """Dependency graph of callable-model evaluation contexts.

    ``graph`` maps each node's cache key to the set of its dependency cache keys, ``ids`` maps cache
    keys back to their :class:`ModelEvaluationContext`, and ``root_id`` is the cache key of the node
    the graph was built from.
    """

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
