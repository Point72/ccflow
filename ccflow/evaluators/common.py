import hashlib
import itertools
import logging
import marshal
import time
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import timedelta
from pprint import pformat
from types import MappingProxyType
from typing import Any, Callable, Dict, List, Optional, Set, Union

import dask.base
from pydantic import Field, PrivateAttr, field_validator
from typing_extensions import override

from ..base import BaseModel, make_lazy_result
from ..callable import CallableModel, ContextBase, EvaluatorBase, ModelEvaluationContext, ResultType
from ..flow_model import (
    _callable_fingerprint,
    _declared_model_dependencies,
    _generated_model_instance,
    _missing_regular_param_names,
    _resolved_contextual_inputs,
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


def cache_key(flow_obj: Union[ModelEvaluationContext, ContextBase, CallableModel]) -> bytes:
    """Returns a key suitable for use in caching.

    Args:
        flow_obj: The object to be tokenized to form the cache key.
    """
    if isinstance(flow_obj, (ModelEvaluationContext, ContextBase, CallableModel)):
        return dask.base.tokenize(flow_obj.model_dump(mode="python")).encode("utf-8")
    else:
        raise TypeError(f"object of type {type(flow_obj)} cannot be serialized by this function!")


def _callable_definition_identity(func: Any) -> str:
    module = getattr(func, "__module__", None) or type(func).__module__
    qualname = getattr(func, "__qualname__", None) or type(func).__qualname__
    code = getattr(func, "__code__", None)
    if code is None:
        return f"callable:{module}:{qualname}"

    payload = "|".join(
        [
            module,
            qualname,
            code.co_filename,
            str(code.co_firstlineno),
            hashlib.sha256(marshal.dumps(code)).hexdigest(),
            repr(getattr(func, "__defaults__", None)),
        ]
    )
    return f"callable:{payload}"


def _unwrap_generated_identity_context(evaluation_context: ModelEvaluationContext) -> Optional[ModelEvaluationContext]:
    current = evaluation_context
    seen: Set[int] = set()

    while isinstance(current.model, EvaluatorBase):
        if id(current) in seen:
            return None
        seen.add(id(current))
        if not isinstance(current.context, ModelEvaluationContext):
            return None
        current = current.context

    return current


def _literal_identity_value(value: Any) -> Any:
    if callable(value):
        return ("callable", _callable_fingerprint(value))
    return value


@dataclass(frozen=True)
class _LiteralBinding:
    name: str
    value: Any


@dataclass(frozen=True)
class _ContextBinding:
    name: str
    value: Any


@dataclass(frozen=True)
class _OpaqueChildReference:
    key: bytes


@dataclass(frozen=True)
class _GeneratedLocalRequest:
    definition_id: str
    context_type_id: str
    fn: str
    literal_inputs: tuple[_LiteralBinding, ...]
    contextual_inputs: tuple[_ContextBinding, ...]
    dependencies: tuple["_GeneratedLocalRequest | _OpaqueChildReference", ...]


def _context_type_identity(context_type: Any) -> str:
    return f"{getattr(context_type, '__module__', None)}:{getattr(context_type, '__qualname__', repr(context_type))}"


def _opaque_child_reference(model: CallableModel, context: ContextBase) -> _OpaqueChildReference:
    return _OpaqueChildReference(key=cache_key(ModelEvaluationContext(model=model, context=context)))


def _generated_local_request(evaluation_context: ModelEvaluationContext) -> Optional[_GeneratedLocalRequest]:
    memo: Dict[bytes, Optional[_GeneratedLocalRequest]] = {}
    return _generated_local_request_impl(evaluation_context, memo, set())


def _generated_local_identity_key(evaluation_context: ModelEvaluationContext) -> Optional[bytes]:
    request = _generated_local_request(evaluation_context)
    if request is None:
        return None
    return dask.base.tokenize(request).encode("utf-8")


def _generated_local_request_impl(
    evaluation_context: ModelEvaluationContext,
    memo: Dict[bytes, Optional[_GeneratedLocalRequest]],
    in_progress: Set[bytes],
) -> Optional[_GeneratedLocalRequest]:
    inner = _unwrap_generated_identity_context(evaluation_context)
    if inner is None or inner.context is None:
        return None

    try:
        memo_key = cache_key(inner)
    except Exception:
        return None

    if memo_key in memo:
        return memo[memo_key]
    if memo_key in in_progress:
        return None

    generated = _generated_model_instance(inner.model)
    if generated is None or inner.fn not in {"__call__", "__deps__"}:
        memo[memo_key] = None
        return None

    config = type(generated).__flow_model_config__
    if _missing_regular_param_names(generated, config):
        memo[memo_key] = None
        return None

    in_progress.add(memo_key)
    try:
        try:
            contextual_inputs = _resolved_contextual_inputs(generated, config, inner.context)
            dependencies = _declared_model_dependencies(generated, config, inner.context, include_lazy=True)
        except Exception:
            memo[memo_key] = None
            return None

        literal_inputs = []
        for param in config.regular_params:
            value = getattr(generated, param.name)
            if isinstance(value, CallableModel):
                continue
            literal_inputs.append(_LiteralBinding(name=param.name, value=_literal_identity_value(value)))

        dependency_requests = []
        for model, contexts in dependencies:
            for context in contexts:
                child_evaluation = ModelEvaluationContext(model=model, context=context)
                child_request = _generated_local_request_impl(child_evaluation, memo, in_progress)
                if child_request is None:
                    child_request = _opaque_child_reference(model, context)
                dependency_requests.append(child_request)

        request = _GeneratedLocalRequest(
            definition_id=_callable_definition_identity(config.func),
            context_type_id=_context_type_identity(config.context_type),
            fn=inner.fn,
            literal_inputs=tuple(literal_inputs),
            contextual_inputs=tuple(_ContextBinding(name=param.name, value=contextual_inputs[param.name]) for param in config.contextual_params),
            dependencies=tuple(dependency_requests),
        )
        memo[memo_key] = request
        return memo[memo_key]
    except Exception:
        memo[memo_key] = None
        return None
    finally:
        in_progress.discard(memo_key)


def _evaluation_cache_key(evaluation_context: ModelEvaluationContext) -> bytes:
    generated_key = _generated_local_identity_key(evaluation_context)
    if generated_key is not None:
        return generated_key
    return cache_key(evaluation_context)


class MemoryCacheEvaluator(EvaluatorBase):
    """Evaluator that caches results in memory."""

    # Note: We make the cache attributes private, so they don't affect tokenization of the MemoryCacheEvaluator itself
    _cache: Dict[bytes, ResultType] = PrivateAttr({})
    _ids: Dict[bytes, ModelEvaluationContext] = PrivateAttr({})

    def key(self, context: ModelEvaluationContext):
        """Function to convert a ModelEvaluationContext to a key"""
        return _evaluation_cache_key(context)

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
    key = _evaluation_cache_key(evaluation_context)
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
    root_key = _evaluation_cache_key(evaluation_context)
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
