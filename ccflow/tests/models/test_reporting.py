import logging
import pickle
from datetime import date
from unittest import TestCase

from ccflow import DateContext
from ccflow.models import (
    AlertsReportingModel,
    LoggingModel,
    MetricsReportingModel,
    OpenTelemetryModel,
    OpenTelemetryTracingReportingModel,
    ReportingModel,
    TracingReportingModel,
)
from ccflow.utils.reporting import AlertPriority, InMemoryReporter, ReportPhase

from ..evaluators.util import MyFlakyCallable, MyResult


class TestReportingModel(TestCase):
    def setUp(self):
        self.context = DateContext(date=date(2022, 1, 1))

    def test_success_reports_and_is_transparent(self):
        reporter = InMemoryReporter()
        inner = MyFlakyCallable(offset=1, fail_times=0)
        model = ReportingModel(model=inner, reporter=reporter)
        out = model(self.context)
        self.assertEqual(out, MyResult(x=2))
        self.assertEqual([e.phase for e in reporter.events], [ReportPhase.START, ReportPhase.SUCCESS, ReportPhase.END])
        self.assertEqual(reporter.events[0].model_name, "MyFlakyCallable")

    def test_error_reports_and_reraises(self):
        reporter = InMemoryReporter()
        inner = MyFlakyCallable(fail_times=5)
        model = ReportingModel(model=inner, reporter=reporter)
        with self.assertRaises(ValueError):
            model(self.context)
        self.assertEqual([e.phase for e in reporter.events], [ReportPhase.START, ReportPhase.ERROR, ReportPhase.END])

    def test_preserves_context_and_result_type(self):
        inner = MyFlakyCallable()
        model = ReportingModel(model=inner)
        self.assertEqual(model.context_type, inner.context_type)
        self.assertEqual(model.result_type, inner.result_type)

    def test_alerts_priority(self):
        reporter = InMemoryReporter()
        inner = MyFlakyCallable(fail_times=5)
        model = AlertsReportingModel(model=inner, reporter=reporter, priority=AlertPriority.P2)
        with self.assertRaises(ValueError):
            model(self.context)
        errors = [e for e in reporter.events if e.phase == ReportPhase.ERROR]
        self.assertEqual(errors[0].priority, AlertPriority.P2)

    def test_metrics_model(self):
        reporter = InMemoryReporter()
        inner = MyFlakyCallable(offset=1, fail_times=0)
        model = MetricsReportingModel(model=inner, reporter=reporter)
        model(self.context)
        self.assertTrue(any(e.extra.get("metric") for e in reporter.events))

    def test_logging_model_logs_and_is_transparent(self):
        inner = MyFlakyCallable(offset=1, fail_times=0)
        model = LoggingModel(model=inner, log_level=logging.INFO, verbose=False)
        with self.assertLogs(level=logging.INFO) as captured:
            out = model(self.context)
        self.assertEqual(out, MyResult(x=2))
        messages = [r.getMessage() for r in captured.records]
        self.assertTrue(any("Start evaluation" in m for m in messages))
        self.assertTrue(any("End evaluation" in m and "time elapsed" in m for m in messages))

    def test_serializable(self):
        model = TracingReportingModel(model=MyFlakyCallable(offset=1), reporter=InMemoryReporter())
        restored = pickle.loads(pickle.dumps(model))
        self.assertEqual(restored.model, model.model)

    def test_opentelemetry_alias(self):
        self.assertIs(OpenTelemetryModel, OpenTelemetryTracingReportingModel)

    def test_opentelemetry_runs_if_available(self):
        try:
            import opentelemetry  # noqa: F401
        except ImportError:
            self.skipTest("opentelemetry not installed")
        model = OpenTelemetryModel(model=MyFlakyCallable(offset=1, fail_times=0))
        self.assertEqual(model(self.context), MyResult(x=2))
