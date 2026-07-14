"""Reporting (telemetry / tracing / metrics / alerting) models.

These are the structural ("what the graph is") way to attach reporting to a *specific* model: a
:class:`ReportingModel` is a first-class :class:`~ccflow.callable.CallableModel` node, so a reporting
policy can be declared statically in config/registries and shows up explicitly in the graph and in
serialization. It wraps another ``CallableModel`` and reuses the shared policies from
:mod:`ccflow.utils.reporting`, exactly as :class:`~ccflow.models.retry.RetryModel` reuses
``RetryPolicy``.

Use a reporting *evaluator* (:mod:`ccflow.evaluators.reporting`) for cross-cutting, runtime reporting
across a whole scope; use a reporting *model* when reporting on one model is a declarative, visible
part of the graph itself.
"""

from typing import Generic

from typing_extensions import override

from ..callable import CallableModelType, ContextType, Flow, ResultType, WrapperModel
from ..evaluators.reporting import _model_type
from ..utils.reporting import (
    AlertsPolicy,
    LoggingPolicy,
    MetricsPolicy,
    OpenTelemetryMetricsPolicy,
    OpenTelemetryTracingPolicy,
    ReportingPolicy,
    TracingPolicy,
)

__all__ = [
    "ReportingModel",
    "LoggingModel",
    "TracingReportingModel",
    "MetricsReportingModel",
    "AlertsReportingModel",
    "OpenTelemetryTracingReportingModel",
    "OpenTelemetryMetricsReportingModel",
    "OpenTelemetryModel",
]


class ReportingModel(WrapperModel[CallableModelType], ReportingPolicy, Generic[CallableModelType]):
    """A callable model that wraps another model and reports telemetry about its evaluation.

    This is the structural counterpart to :class:`~ccflow.evaluators.reporting.ReportingEvaluator`:
    it is a first-class node (declarable in config/registries, visible in serialization and the
    dependency graph) that inherits the wrapped model's ``context_type`` / ``result_type`` (from
    ``WrapperModel``) and reuses the reporting mechanics from
    :class:`~ccflow.utils.reporting.ReportingPolicy`.

    Reporting is transparent: the wrapped model's result is returned unchanged.
    """

    @override
    @Flow.call
    def __call__(self, context: ContextType) -> ResultType:
        name = self.model.meta.name or self.model.__class__.__name__
        return self._run_with_reporting(
            lambda: self.model(context),
            model_name=name,
            model_type=_model_type(self.model),
            fn="__call__",
            context=context,
        )


class LoggingModel(ReportingModel[CallableModelType], LoggingPolicy, Generic[CallableModelType]):
    """A callable model that wraps another model and logs its evaluation.

    This is the structural counterpart to ``LoggingEvaluator``: it produces the same start/end (and
    optional result) log output via :class:`~ccflow.utils.reporting.LoggingPolicy`, but as an explicit
    graph node rather than a cross-cutting evaluator. Logging is transparent: the wrapped model's
    result is returned unchanged.
    """

    @override
    @Flow.call
    def __call__(self, context: ContextType) -> ResultType:
        name = self.model.meta.name or self.model.__class__.__name__
        return self._run_with_reporting(
            lambda: self.model(context),
            model_name=name,
            model_type=_model_type(self.model),
            fn="__call__",
            context=context,
            extra={"model": self.model, "raw_context": context, "options": {}},
        )


class TracingReportingModel(ReportingModel[CallableModelType], TracingPolicy, Generic[CallableModelType]):
    """Reporting model specialised for distributed *tracing* (spans)."""


class MetricsReportingModel(ReportingModel[CallableModelType], MetricsPolicy, Generic[CallableModelType]):
    """Reporting model specialised for *metrics* (counters / latency histograms)."""


class AlertsReportingModel(ReportingModel[CallableModelType], AlertsPolicy, Generic[CallableModelType]):
    """Reporting model specialised for *alerting*, with ``P1``-``P5`` priority tags."""


class OpenTelemetryTracingReportingModel(TracingReportingModel[CallableModelType], OpenTelemetryTracingPolicy, Generic[CallableModelType]):
    """Tracing model backed by OpenTelemetry spans (requires ``opentelemetry-api``)."""


class OpenTelemetryMetricsReportingModel(MetricsReportingModel[CallableModelType], OpenTelemetryMetricsPolicy, Generic[CallableModelType]):
    """Metrics model backed by OpenTelemetry counters/histograms (requires ``opentelemetry-api``)."""


# Convenience alias for the most common OpenTelemetry use case (tracing).
OpenTelemetryModel = OpenTelemetryTracingReportingModel
