"""Reporting (telemetry / tracing / metrics / alerting) evaluators.

These are the cross-cutting ("how to run") way to attach reporting to a callable graph: a
:class:`ReportingEvaluator` wraps the evaluation of every model in its scope and is configured at
runtime via ``FlowOptions`` / ``FlowOptionsOverride``. The reporting mechanics come from the shared
policies in :mod:`ccflow.utils.reporting`, mirroring how ``RetryEvaluator`` reuses ``RetryPolicy``.

A reporting evaluator is always *transparent*: a successful evaluation returns exactly the same value
as evaluating the wrapped context directly, so it does not affect cache keys or dependency-graph
deduplication. Reporting describes the *evaluation* (model identity, context, timing, graph topology,
failures) -- it never acts on the result payload, which is what distinguishes it from a publisher.

The class taxonomy is ``<Vendor><Signal>ReportingEvaluator`` where ``Signal`` is one of ``Tracing``,
``Metrics`` or ``Alerts``. OpenTelemetry tracing/metrics are implemented; the remaining vendor
backends (Datadog, Opsgenie, JSM, NewRelic, ...) are placeholders that raise ``NotImplementedError``.
"""

from collections import deque
from contextvars import ContextVar

from pydantic import Field
from typing_extensions import override

from ..callable import EvaluatorBase, ModelEvaluationContext, ResultType
from ..exttypes import PyObjectPath
from ..utils.reporting import (
    AlertsPolicy,
    MetricsPolicy,
    OpenTelemetryMetricsPolicy,
    OpenTelemetryTracingPolicy,
    ReportContext,
    ReportingPolicy,
    ReportPhase,
    TracingPolicy,
    _new_span_id,
)

__all__ = [
    "ReportingEvaluator",
    "DryRunEvaluator",
    "TracingReportingEvaluator",
    "MetricsReportingEvaluator",
    "AlertsReportingEvaluator",
    "OpenTelemetryTracingReportingEvaluator",
    "OpenTelemetryMetricsReportingEvaluator",
    "OpenTelemetryEvaluator",
    "DatadogTracingReportingEvaluator",
    "DatadogMetricsReportingEvaluator",
    "DatadogAlertsReportingEvaluator",
    "OpsgenieMetricsReportingEvaluator",
    "OpsgenieAlertsReportingEvaluator",
    "JSMAlertsReportingEvaluator",
    "NewRelicTracingReportingEvaluator",
    "NewRelicMetricsReportingEvaluator",
    "NewRelicAlertsReportingEvaluator",
]


def _model_type(model) -> str:
    """Best-effort fully-qualified type name for a model.

    Falls back to ``module.qualname`` when the type cannot be resolved as an importable
    :class:`PyObjectPath` (e.g. dynamically-generated classes such as Ray-wrapped callables), so that
    reporting never breaks evaluation.
    """
    try:
        return str(PyObjectPath.validate(type(model)))
    except Exception:
        cls = type(model)
        return f"{cls.__module__}.{cls.__qualname__}"


def _descriptor(context: ModelEvaluationContext) -> dict:
    """Build the reporting descriptor (model name/type/fn/context) for an evaluation context."""
    model = context.model
    return {
        "model_name": model.meta.name or model.__class__.__name__,
        "model_type": _model_type(model),
        "fn": context.fn,
        "context": context.context,
    }


# Re-entrancy guard for DryRunEvaluator, kept context-local (not on the instance) so a single
# evaluator instance shared across threads / concurrent evaluations cannot leak planning state from
# one evaluation into another (which would cause the second to run bodies instead of planning).
_DRY_RUN_PLANNING: ContextVar[bool] = ContextVar("ccflow_dry_run_planning", default=False)


class ReportingEvaluator(EvaluatorBase, ReportingPolicy):
    """Evaluator that reports telemetry about the evaluation of the callable models in its scope.

    This is the base of the reporting evaluator family. It is *transparent* (the result is identical
    to evaluating the wrapped context directly) and keeps no mutable per-call state on the instance --
    all per-call state lives in local variables / context vars -- so a single instance can be shared
    safely across threads and combined with parallel evaluators (Ray / Celery).

    Configure a :class:`~ccflow.utils.reporting.Reporter` to receive the emitted
    :class:`~ccflow.utils.reporting.ReportEvent` objects. To attach reporting to a single specific
    model as part of the graph itself, use :class:`~ccflow.models.reporting.ReportingModel` instead.
    """

    def is_transparent(self, context: ModelEvaluationContext) -> bool:
        return True

    @override
    def __call__(self, context: ModelEvaluationContext) -> ResultType:
        return self._run_with_reporting(context, **_descriptor(context))


class DryRunEvaluator(EvaluatorBase, ReportingPolicy):
    """Evaluator that *plans* an evaluation by walking the dependency graph without running bodies.

    It discovers the dependency graph (which does evaluate ``__deps__`` -- the cheap dependency
    declarations -- but never ``__call__``), emits a :attr:`~ccflow.utils.reporting.ReportPhase.QUEUED`
    event followed by a :attr:`~ccflow.utils.reporting.ReportPhase.SKIPPED` event for every node in
    breadth-first order with a parent/child span tree mirroring the graph, and returns a *synthetic*
    result (constructed with ``model_construct`` so no fields are validated/populated). No model body
    is executed, making it useful for previewing what a run would do and for driving a UI.

    .. warning::

        With the default ``synthetic_result=True`` the returned object is built with
        ``model_construct`` -- required fields may be **unset** and validators do **not** run. It is a
        placeholder for previewing/UIs only and must not be fed into downstream computation. Because
        this changes the return value, the evaluator is **not transparent** in that mode (see
        :meth:`is_transparent`), so it participates in cache keys and will not contaminate the real
        model/context cache entry.
    """

    synthetic_result: bool = Field(
        True,
        description="If True, return a synthetic (model_construct) result without running any body. If False, fall back to running the wrapped evaluation after planning.",
    )

    def is_transparent(self, context: ModelEvaluationContext) -> bool:
        # Transparent only when we actually run the body and return the real value. When a synthetic
        # result is returned the value differs from ``context()``, so the layer must NOT be stripped
        # from cache keys (otherwise a synthetic result could be cached under the real key).
        return not self.synthetic_result

    @override
    def __call__(self, context: ModelEvaluationContext) -> ResultType:
        # Dependency declarations (and any re-entry while already planning) must run directly rather
        # than re-planning, otherwise graph discovery -- which evaluates ``__deps__`` through the
        # active evaluator -- recurses. The guard is context-local so a shared instance is safe under
        # concurrency. This keeps the documented ``FlowOptionsOverride`` path working.
        if _DRY_RUN_PLANNING.get() or context.fn == "__deps__":
            return context()

        from .common import _flatten_cache_key_context, cache_key, get_dependency_graph  # local import to avoid an import cycle

        token = _DRY_RUN_PLANNING.set(True)
        try:
            graph = get_dependency_graph(context)
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
                # Strip any transparent evaluator layers so we report on the underlying model/context.
                flattened, fn, _ = _flatten_cache_key_context(evaluation_context)
                model = flattened.model
                # The raw graph key can include this (non-transparent) dry-run evaluator layer, so it
                # is not a stable identity for the logical node. Compute the node_key from the
                # flattened model/context/fn/options alone (mirroring ``cache_key()``) so it matches
                # across invocation styles while still distinguishing differing non-evaluator options.
                logical_key = cache_key(ModelEvaluationContext(model=model, context=flattened.context, fn=fn, options=flattened.options))
                ctx = ReportContext(
                    model_name=model.meta.name or model.__class__.__name__,
                    model_type=_model_type(model),
                    fn=fn,
                    context_repr=self._context_repr(flattened.context),
                    span_id=span_ids[key],
                    parent_span_id=span_ids.get(parent_of.get(key)),
                    depth=depth_of.get(key, 0),
                    extra={"node_key": logical_key.decode("utf-8"), "dry_run": True},
                )
                self._emit(self._event(ctx, ReportPhase.QUEUED, extra=ctx.extra))
                self._emit(self._event(ctx, ReportPhase.SKIPPED, extra=ctx.extra))

            if self.synthetic_result:
                return context.model.result_type.model_construct()
            return context()
        finally:
            _DRY_RUN_PLANNING.reset(token)


class TracingReportingEvaluator(ReportingEvaluator, TracingPolicy):
    """Reporting evaluator specialised for distributed *tracing* (spans)."""


class MetricsReportingEvaluator(ReportingEvaluator, MetricsPolicy):
    """Reporting evaluator specialised for *metrics* (counters / latency histograms)."""


class AlertsReportingEvaluator(ReportingEvaluator, AlertsPolicy):
    """Reporting evaluator specialised for *alerting*, with ``P1``-``P5`` priority tags."""


# *****************************************************************************
# OpenTelemetry (implemented; optional dependency, imported lazily)
# *****************************************************************************


class OpenTelemetryTracingReportingEvaluator(TracingReportingEvaluator, OpenTelemetryTracingPolicy):
    """Tracing evaluator backed by OpenTelemetry spans (requires ``opentelemetry-api``)."""


class OpenTelemetryMetricsReportingEvaluator(MetricsReportingEvaluator, OpenTelemetryMetricsPolicy):
    """Metrics evaluator backed by OpenTelemetry counters/histograms (requires ``opentelemetry-api``)."""


#: Convenience alias for the most common OpenTelemetry use case (tracing).
OpenTelemetryEvaluator = OpenTelemetryTracingReportingEvaluator


# *****************************************************************************
# Vendor placeholders (not yet implemented)
#
# These are intentionally defined so they can be referenced in config / type hints and tracked, but
# they raise NotImplementedError when used. Implementations should replace the placeholder body with
# a real backend policy (mirroring the OpenTelemetry classes above).
# *****************************************************************************


class _NotImplementedReportingEvaluator(ReportingEvaluator):
    """Base for reporting evaluators whose backend integration is not yet implemented."""

    @override
    def __call__(self, context: ModelEvaluationContext) -> ResultType:
        raise NotImplementedError(
            f"{type(self).__name__} is not yet implemented. "
            "Track/implement this integration or use an implemented reporting evaluator (e.g. OpenTelemetry*)."
        )


class DatadogTracingReportingEvaluator(_NotImplementedReportingEvaluator):
    """Placeholder: tracing via Datadog APM (not yet implemented)."""


class DatadogMetricsReportingEvaluator(_NotImplementedReportingEvaluator):
    """Placeholder: metrics via Datadog (not yet implemented)."""


class DatadogAlertsReportingEvaluator(_NotImplementedReportingEvaluator):
    """Placeholder: alerting via Datadog monitors (not yet implemented)."""


class OpsgenieMetricsReportingEvaluator(_NotImplementedReportingEvaluator):
    """Placeholder: metrics via Opsgenie (not yet implemented)."""


class OpsgenieAlertsReportingEvaluator(_NotImplementedReportingEvaluator):
    """Placeholder: alerting via Opsgenie (not yet implemented)."""


class JSMAlertsReportingEvaluator(_NotImplementedReportingEvaluator):
    """Placeholder: alerting via Jira Service Management (not yet implemented)."""


class NewRelicTracingReportingEvaluator(_NotImplementedReportingEvaluator):
    """Placeholder: tracing via New Relic (not yet implemented)."""


class NewRelicMetricsReportingEvaluator(_NotImplementedReportingEvaluator):
    """Placeholder: metrics via New Relic (not yet implemented)."""


class NewRelicAlertsReportingEvaluator(_NotImplementedReportingEvaluator):
    """Placeholder: alerting via New Relic (not yet implemented)."""
