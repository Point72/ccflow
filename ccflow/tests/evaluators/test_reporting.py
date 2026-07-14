import pickle
from datetime import date
from unittest import TestCase

from ccflow import DateContext, FlowOptionsOverride, GenericResult, ModelEvaluationContext, TransparentModelEvaluationContext
from ccflow.evaluators import (
    AlertsReportingEvaluator,
    DryRunEvaluator,
    MemoryCacheEvaluator,
    MetricsReportingEvaluator,
    MultiEvaluator,
    OpenTelemetryEvaluator,
    OpenTelemetryMetricsReportingEvaluator,
    OpenTelemetryTracingReportingEvaluator,
    ReportingEvaluator,
    TracingReportingEvaluator,
)
from ccflow.utils.reporting import AlertPriority, InMemoryReporter, ReportingPolicy, ReportPhase

from .util import MyFlakyCallable, MyResult, NodeModel


class TestReportingEvaluator(TestCase):
    def setUp(self):
        self.context = DateContext(date=date(2022, 1, 1))

    def _eval_context(self, model):
        return ModelEvaluationContext(model=model, context=self.context)

    def test_success_reports_and_is_transparent(self):
        reporter = InMemoryReporter()
        model = MyFlakyCallable(offset=1, fail_times=0)
        evaluator = ReportingEvaluator(reporter=reporter)
        out = evaluator(self._eval_context(model))
        self.assertEqual(out, MyResult(x=2))
        self.assertEqual([e.phase for e in reporter.events], [ReportPhase.START, ReportPhase.SUCCESS, ReportPhase.END])
        self.assertEqual(reporter.events[0].model_name, "MyFlakyCallable")

    def test_is_transparent(self):
        model = MyFlakyCallable(fail_times=0)
        evaluator = ReportingEvaluator()
        context = self._eval_context(model)
        self.assertTrue(evaluator.is_transparent(context))
        wrapped = evaluator.make_evaluation_context(context, options=context.options)
        self.assertIsInstance(wrapped, TransparentModelEvaluationContext)

    def test_error_reports_and_reraises(self):
        reporter = InMemoryReporter()
        model = MyFlakyCallable(fail_times=5)
        evaluator = ReportingEvaluator(reporter=reporter)
        with self.assertRaises(ValueError):
            evaluator(self._eval_context(model))
        self.assertEqual([e.phase for e in reporter.events], [ReportPhase.START, ReportPhase.ERROR, ReportPhase.END])

    def test_flow_options_override_integration(self):
        reporter = InMemoryReporter()
        model = MyFlakyCallable(offset=1, fail_times=0)
        evaluator = ReportingEvaluator(reporter=reporter)
        with FlowOptionsOverride(options={"evaluator": evaluator}):
            out = model(self.context)
        self.assertEqual(out, MyResult(x=2))
        self.assertTrue(reporter.events)

    def test_alerts_priority(self):
        reporter = InMemoryReporter()
        model = MyFlakyCallable(fail_times=5)
        evaluator = AlertsReportingEvaluator(reporter=reporter, priority=AlertPriority.P1)
        with self.assertRaises(ValueError):
            evaluator(self._eval_context(model))
        errors = [e for e in reporter.events if e.phase == ReportPhase.ERROR]
        self.assertEqual(errors[0].priority, AlertPriority.P1)

    def test_metrics_evaluator(self):
        reporter = InMemoryReporter()
        model = MyFlakyCallable(offset=1, fail_times=0)
        evaluator = MetricsReportingEvaluator(reporter=reporter)
        evaluator(self._eval_context(model))
        self.assertTrue(any(e.extra.get("metric") for e in reporter.events))

    def test_serializable(self):
        evaluator = TracingReportingEvaluator(reporter=InMemoryReporter())
        restored = pickle.loads(pickle.dumps(evaluator))
        self.assertEqual(restored, evaluator)
        dumped = evaluator.model_dump()
        self.assertEqual(TracingReportingEvaluator.model_validate(dumped), evaluator)


class TestOpenTelemetryEvaluator(TestCase):
    def setUp(self):
        self.context = ModelEvaluationContext(model=MyFlakyCallable(offset=1, fail_times=0), context=DateContext(date=date(2022, 1, 1)))

    def test_alias(self):
        self.assertIs(OpenTelemetryEvaluator, OpenTelemetryTracingReportingEvaluator)

    def test_tracing_runs_with_otel_if_available(self):
        try:
            import opentelemetry  # noqa: F401
        except ImportError:
            self.skipTest("opentelemetry not installed")
        out = OpenTelemetryTracingReportingEvaluator()(self.context)
        self.assertEqual(out, MyResult(x=2))

    def test_metrics_runs_with_otel_if_available(self):
        try:
            import opentelemetry  # noqa: F401
        except ImportError:
            self.skipTest("opentelemetry not installed")
        out = OpenTelemetryMetricsReportingEvaluator()(self.context)
        self.assertEqual(out, MyResult(x=2))


class TestDryRunEvaluator(TestCase):
    def setUp(self):
        NodeModel._calls.clear()
        NodeModel._deps_calls.clear()
        self.context = DateContext(date=date(2022, 1, 1))
        leaf = NodeModel(meta={"name": "leaf"})
        self.root = NodeModel(meta={"name": "root"}, deps_model=[leaf])

    def _eval_context(self, model):
        return ModelEvaluationContext(model=model, context=self.context)

    def test_composes_reporting_policy(self):
        evaluator = DryRunEvaluator(reporting={"capture_context_repr": False})
        self.assertIsInstance(evaluator.reporting, ReportingPolicy)

    def test_does_not_execute_bodies(self):
        reporter = InMemoryReporter()
        evaluator = DryRunEvaluator(reporting={"reporter": reporter})
        evaluator(self._eval_context(self.root))
        # No __call__ bodies ran (NodeModel records call bodies in _calls).
        self.assertEqual(NodeModel._calls, [])

    def test_emits_queued_and_skipped_for_each_node(self):
        reporter = InMemoryReporter()
        evaluator = DryRunEvaluator(reporting={"reporter": reporter})
        evaluator(self._eval_context(self.root))
        phases = [e.phase for e in reporter.events]
        self.assertEqual(phases.count(ReportPhase.QUEUED), 2)
        self.assertEqual(phases.count(ReportPhase.SKIPPED), 2)
        names = {e.model_name for e in reporter.events}
        self.assertEqual(names, {"root", "leaf"})

    def test_parent_child_span_tree(self):
        reporter = InMemoryReporter()
        evaluator = DryRunEvaluator(reporting={"reporter": reporter})
        evaluator(self._eval_context(self.root))
        queued = {e.model_name: e for e in reporter.events if e.phase == ReportPhase.QUEUED}
        self.assertIsNone(queued["root"].parent_span_id)
        self.assertEqual(queued["leaf"].parent_span_id, queued["root"].span_id)
        self.assertEqual(queued["leaf"].depth, queued["root"].depth + 1)

    def test_returns_synthetic_result(self):
        evaluator = DryRunEvaluator()
        out = evaluator(self._eval_context(self.root))
        self.assertIsInstance(out, self.root.result_type)

    def test_synthetic_result_has_unset_fields(self):
        # Guardrail: the synthetic result is built with model_construct, so required fields are unset
        # and must not be relied upon downstream.
        out = DryRunEvaluator()(self._eval_context(self.root))
        self.assertNotIn("value", out.model_fields_set)

    def test_is_transparent(self):
        # Synthetic result changes the return value -> NOT transparent (must affect cache keys).
        self.assertFalse(DryRunEvaluator().is_transparent(self._eval_context(self.root)))
        # When it falls back to running the body, it returns the real value -> transparent.
        self.assertTrue(DryRunEvaluator(synthetic_result=False).is_transparent(self._eval_context(self.root)))

    def test_flow_options_override_does_not_recurse(self):
        # Regression: the documented FlowOptionsOverride path used to recurse while discovering deps
        # because __deps__ is evaluated through the active (dry-run) evaluator.
        reporter = InMemoryReporter()
        with FlowOptionsOverride(options={"evaluator": DryRunEvaluator(reporting={"reporter": reporter})}):
            out = self.root(self.context)
        self.assertIsInstance(out, self.root.result_type)
        self.assertEqual(NodeModel._calls, [])
        names = {e.model_name for e in reporter.events if e.phase == ReportPhase.QUEUED}
        self.assertEqual(names, {"root", "leaf"})

    def test_composed_with_cache_does_not_contaminate(self):
        # A dry run composed with caching must not store its synthetic result under the real key, so a
        # subsequent real run still computes (and caches) the genuine result.
        cache = MemoryCacheEvaluator()
        with FlowOptionsOverride(options={"evaluator": MultiEvaluator(evaluators=[DryRunEvaluator(), cache])}):
            dry = self.root(self.context)
        self.assertIsInstance(dry, self.root.result_type)
        self.assertEqual(NodeModel._calls, [])
        with FlowOptionsOverride(options={"evaluator": cache}):
            real = self.root(self.context)
        self.assertEqual(real, GenericResult(value=True))
        self.assertNotEqual(NodeModel._calls, [])

    def test_node_key_in_extra(self):
        reporter = InMemoryReporter()
        DryRunEvaluator(reporting={"reporter": reporter})(self._eval_context(self.root))
        for event in reporter.events:
            self.assertIn("node_key", event.extra)

    def test_node_key_stable_across_invocation_styles(self):
        # The emitted node_key must strip the (non-transparent) DryRunEvaluator layer that appears in
        # the FlowOptionsOverride dependency-graph keys, so a dependency has the same logical identity
        # whether the dry run is invoked directly on the evaluator or through FlowOptionsOverride.
        # (The dependency `leaf` is built by the framework identically in both styles; the root differs
        # only because a bare direct MEC carries different FlowOptions than the framework-built one.)
        direct_reporter = InMemoryReporter()
        DryRunEvaluator(reporting={"reporter": direct_reporter})(self._eval_context(self.root))
        direct_leaf = next(e.extra["node_key"] for e in direct_reporter.events if e.phase == ReportPhase.QUEUED and e.model_name == "leaf")

        NodeModel._calls.clear()
        NodeModel._deps_calls.clear()
        override_reporter = InMemoryReporter()
        with FlowOptionsOverride(options={"evaluator": DryRunEvaluator(reporting={"reporter": override_reporter})}):
            self.root(self.context)
        override_leaf = next(e.extra["node_key"] for e in override_reporter.events if e.phase == ReportPhase.QUEUED and e.model_name == "leaf")

        self.assertEqual(direct_leaf, override_leaf)

    def test_node_key_strips_evaluator_layer(self):
        # In the FlowOptionsOverride path the raw dependency-graph keys include the non-transparent
        # DryRunEvaluator layer. The emitted node_key must equal the logical cache_key() (evaluator
        # layer stripped), not that raw graph key.
        from ccflow.evaluators.common import _flatten_cache_key_context, cache_key, get_dependency_graph

        reporter = InMemoryReporter()
        with FlowOptionsOverride(options={"evaluator": DryRunEvaluator(reporting={"reporter": reporter})}):
            self.root(self.context)
        emitted = {e.model_name: e.extra["node_key"] for e in reporter.events if e.phase == ReportPhase.QUEUED}

        # Rebuild the logical graph to compute the expected cache_key() for each node.
        NodeModel._calls.clear()
        NodeModel._deps_calls.clear()
        with FlowOptionsOverride(options={"evaluator": DryRunEvaluator()}):
            inner = self.root.__call__.get_evaluation_context(self.root, self.context)
        flattened, _, _ = _flatten_cache_key_context(inner)
        graph = get_dependency_graph(flattened)
        for key in graph.graph:
            f, fn, _ = _flatten_cache_key_context(graph.ids[key])
            logical = cache_key(ModelEvaluationContext(model=f.model, context=f.context, fn=fn, options=f.options)).decode("utf-8")
            self.assertEqual(emitted[f.model.meta.name], logical)

    def test_node_key_distinguishes_non_evaluator_options(self):
        # node_key mirrors cache_key(), which includes non-evaluator FlowOptions, so two dry runs that
        # differ only in such options must not collapse to the same logical node identity.
        from ccflow.evaluators.common import cache_key

        default_reporter = InMemoryReporter()
        DryRunEvaluator(reporting={"reporter": default_reporter})(ModelEvaluationContext(model=self.root, context=self.context))
        default_key = next(e.extra["node_key"] for e in default_reporter.events if e.phase == ReportPhase.QUEUED and e.model_name == "root")

        other_reporter = InMemoryReporter()
        other_context = ModelEvaluationContext(model=self.root, context=self.context, options={"validate_result": False})
        DryRunEvaluator(reporting={"reporter": other_reporter})(other_context)
        other_key = next(e.extra["node_key"] for e in other_reporter.events if e.phase == ReportPhase.QUEUED and e.model_name == "root")

        self.assertNotEqual(default_key, other_key)
        # And the emitted key equals the real cache_key() of the logical node.
        self.assertEqual(other_key, cache_key(other_context).decode("utf-8"))

    def test_concurrent_dry_runs_share_instance_without_running_bodies(self):
        # The planning guard is context-local, so a single shared instance used by two concurrent
        # evaluations must never let one run's planning state leak into the other (which would make
        # it execute bodies instead of planning).
        import threading

        evaluator = DryRunEvaluator()
        barrier = threading.Barrier(2)
        errors = []

        def run():
            try:
                barrier.wait()
                for _ in range(20):
                    evaluator(self._eval_context(self.root))
            except Exception as exc:  # pragma: no cover - surfaced via errors list
                errors.append(exc)

        threads = [threading.Thread(target=run) for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(errors, [])
        # No model body ran in either thread.
        self.assertEqual(NodeModel._calls, [])
