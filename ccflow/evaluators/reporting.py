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
``Metrics`` or ``Alerts``. OpenTelemetry tracing and metrics are implemented.
"""

from typing_extensions import override

from ..callable import EvaluatorBase, ModelEvaluationContext, ResultType
from ..exttypes import PyObjectPath
from ..utils.reporting import (
    AlertsPolicy,
    MetricsPolicy,
    OpenTelemetryMetricsPolicy,
    OpenTelemetryTracingPolicy,
    ReportingPolicy,
    TracingPolicy,
)

__all__ = [
    "AlertsReportingEvaluator",
    "MetricsReportingEvaluator",
    "OpenTelemetryEvaluator",
    "OpenTelemetryMetricsReportingEvaluator",
    "OpenTelemetryTracingReportingEvaluator",
    "ReportingEvaluator",
    "TracingReportingEvaluator",
]


def _model_type(model) -> str:
    """Best-effort fully-qualified type name for a model.

    Falls back to ``module.qualname`` when the type cannot be resolved as an importable
    :class:`PyObjectPath` (e.g. dynamically-generated classes such as Ray-wrapped callables), so that
    reporting never breaks evaluation.
    """
    try:
        return str(PyObjectPath.validate(type(model)))
    except Exception:  # noqa: BLE001
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


class TracingReportingEvaluator(ReportingEvaluator, TracingPolicy):
    """Reporting evaluator specialised for distributed *tracing* (spans)."""


class MetricsReportingEvaluator(ReportingEvaluator, MetricsPolicy):
    """Reporting evaluator specialised for *metrics* (counters / latency histograms)."""


class AlertsReportingEvaluator(ReportingEvaluator, AlertsPolicy):
    """Reporting evaluator specialised for *alerting*, with ``P1``-``P5`` priority tags."""


class OpenTelemetryTracingReportingEvaluator(TracingReportingEvaluator, OpenTelemetryTracingPolicy):
    """Tracing evaluator backed by OpenTelemetry spans (requires ``opentelemetry-api``)."""


class OpenTelemetryMetricsReportingEvaluator(MetricsReportingEvaluator, OpenTelemetryMetricsPolicy):
    """Metrics evaluator backed by OpenTelemetry counters/histograms (requires ``opentelemetry-api``)."""


OpenTelemetryEvaluator = OpenTelemetryTracingReportingEvaluator

# Future reporting backends may include Datadog tracing, metrics, and alerts; Opsgenie metrics and
# alerts; Jira Service Management alerts; and New Relic tracing, metrics, and alerts. Add each public
# evaluator only when its backend policy is implemented.
