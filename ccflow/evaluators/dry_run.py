"""Dependency-graph planning without model execution."""

from collections import deque
from contextvars import ContextVar

from pydantic import Field
from typing_extensions import override

from ..callable import EvaluatorBase, ModelEvaluationContext, ResultType
from ..utils.reporting import ReportContext, ReportingPolicy, ReportPhase, _new_span_id
from .reporting import _model_type

__all__ = ["DryRunEvaluator"]


_DRY_RUN_PLANNING: ContextVar[bool] = ContextVar("ccflow_dry_run_planning", default=False)


class DryRunEvaluator(EvaluatorBase):
    """Plan a dependency graph without running model bodies.

    Dependency declarations are evaluated to discover the graph. When ``reporting`` has a reporter,
    each node emits ``QUEUED`` and ``SKIPPED`` events. The default synthetic result is constructed
    without validation and must not be used in downstream computation.
    """

    reporting: ReportingPolicy = Field(default_factory=ReportingPolicy)
    synthetic_result: bool = Field(
        True,
        description="If True, return an unvalidated synthetic result. If False, run the wrapped evaluation after planning.",
    )

    def is_transparent(self, context: ModelEvaluationContext) -> bool:
        return not self.synthetic_result

    @override
    def __call__(self, context: ModelEvaluationContext) -> ResultType:
        if _DRY_RUN_PLANNING.get() or context.fn == "__deps__":
            return context()

        from .common import _flatten_cache_key_context, cache_key, get_dependency_graph

        token = _DRY_RUN_PLANNING.set(True)
        try:
            graph = get_dependency_graph(context)
            if self.reporting.reporter is not None:
                span_ids = {key: _new_span_id() for key in graph.graph}
                parent_of: dict = {}
                depth_of: dict = {graph.root_id: 0}
                order: list = []
                seen: set = set()
                queue = deque([graph.root_id])
                while queue:
                    key = queue.popleft()
                    if key in seen:
                        continue
                    seen.add(key)
                    order.append(key)
                    for child in graph.graph.get(key, ()):
                        parent_of.setdefault(child, key)
                        depth_of.setdefault(child, depth_of.get(key, 0) + 1)
                        queue.append(child)

                for key in order:
                    evaluation_context = graph.ids[key]
                    flattened, fn, _ = _flatten_cache_key_context(evaluation_context)
                    model = flattened.model
                    logical_key = cache_key(ModelEvaluationContext(model=model, context=flattened.context, fn=fn, options=flattened.options))
                    report_context = ReportContext(
                        model_name=model.meta.name or model.__class__.__name__,
                        model_type=_model_type(model),
                        fn=fn,
                        context_repr=self.reporting._context_repr(flattened.context),
                        span_id=span_ids[key],
                        parent_span_id=span_ids.get(parent_of.get(key)),
                        depth=depth_of.get(key, 0),
                        extra={"node_key": logical_key.decode("utf-8"), "dry_run": True},
                    )
                    self.reporting._emit(self.reporting._event(report_context, ReportPhase.QUEUED, extra=report_context.extra))
                    self.reporting._emit(self.reporting._event(report_context, ReportPhase.SKIPPED, extra=report_context.extra))

            if self.synthetic_result:
                return context.model.result_type.model_construct()
            return context()
        finally:
            _DRY_RUN_PLANNING.reset(token)
