import logging
import pickle
from typing import Callable
from unittest import TestCase

from ccflow.utils.reporting import (
    AlertPriority,
    AlertsPolicy,
    CompositeReporter,
    InMemoryReporter,
    LoggingPolicy,
    LoggingReporter,
    MetricsPolicy,
    NodeState,
    NoOpReporter,
    ReportEvent,
    ReportingPolicy,
    ReportingStateStore,
    ReportPhase,
    TracingPolicy,
    UIReporter,
    current_run_id,
    current_span_depth,
    current_span_id,
    run_scope,
)


def _ok() -> str:
    return "ok"


def _boom() -> str:
    raise ValueError("boom")


class TestReporters(TestCase):
    def test_in_memory_collects(self):
        reporter = InMemoryReporter()
        event = ReportEvent(phase=ReportPhase.START, model_name="m", span_id="abc")
        reporter.emit(event)
        self.assertEqual(reporter.events, [event])
        reporter.clear()
        self.assertEqual(reporter.events, [])

    def test_in_memory_deepcopy_shares_buffer(self):
        import copy

        reporter = InMemoryReporter()
        self.assertIs(copy.deepcopy(reporter), reporter)

    def test_noop_discards(self):
        reporter = NoOpReporter()
        reporter.emit(ReportEvent(phase=ReportPhase.START, model_name="m", span_id="abc"))  # no error

    def test_composite_fans_out(self):
        a, b = InMemoryReporter(), InMemoryReporter()
        composite = CompositeReporter(reporters=[a, b])
        event = ReportEvent(phase=ReportPhase.END, model_name="m", span_id="abc")
        composite.emit(event)
        self.assertEqual(a.events, [event])
        self.assertEqual(b.events, [event])

    def test_composite_isolates_failing_child(self):
        class BrokenReporter(NoOpReporter):
            def emit(self, event):
                raise RuntimeError("sink down")

        good = InMemoryReporter()
        composite = CompositeReporter(reporters=[BrokenReporter(), good])
        event = ReportEvent(phase=ReportPhase.END, model_name="m", span_id="abc")
        with self.assertLogs("ccflow.utils.reporting", level="ERROR"):
            composite.emit(event)  # must not raise
        # A broken child must not prevent the healthy sink from receiving the event.
        self.assertEqual(good.events, [event])

    def test_logging_reporter(self):
        reporter = LoggingReporter()
        with self.assertLogs("ccflow.reporting", level="INFO") as cm:
            reporter.emit(ReportEvent(phase=ReportPhase.START, model_name="m", span_id="abc"))
        self.assertTrue(any("START" in line for line in cm.output))


class TestReportingPolicy(TestCase):
    def _run(self, policy: ReportingPolicy, fn: Callable[[], str]) -> str:
        return policy._run_with_reporting(fn, model_name="M", model_type="pkg.M", fn="__call__", context="ctx")

    def test_transparent_success(self):
        reporter = InMemoryReporter()
        policy = ReportingPolicy(reporter=reporter)
        self.assertEqual(self._run(policy, _ok), "ok")
        phases = [e.phase for e in reporter.events]
        self.assertEqual(phases, [ReportPhase.START, ReportPhase.SUCCESS, ReportPhase.END])
        self.assertEqual(reporter.events[0].model_name, "M")
        self.assertEqual(reporter.events[0].model_type, "pkg.M")

    def test_no_reporter_is_passthrough(self):
        policy = ReportingPolicy()
        self.assertEqual(self._run(policy, _ok), "ok")

    def test_error_reports_and_reraises(self):
        reporter = InMemoryReporter()
        policy = ReportingPolicy(reporter=reporter)
        with self.assertRaises(ValueError):
            self._run(policy, _boom)
        phases = [e.phase for e in reporter.events]
        self.assertEqual(phases, [ReportPhase.START, ReportPhase.ERROR, ReportPhase.END])
        self.assertEqual(reporter.events[1].exception_type, "ValueError")
        self.assertEqual(reporter.events[1].exception_message, "boom")

    def test_span_correlation_nesting(self):
        reporter = InMemoryReporter()
        policy = ReportingPolicy(reporter=reporter)

        def outer() -> str:
            # The inner evaluation should see the outer span as its parent.
            self.assertIsNotNone(current_span_id())
            policy._run_with_reporting(_ok, model_name="inner", model_type="pkg.inner", fn="__call__", context="c")
            return "ok"

        self._run(policy, outer)
        starts = [e for e in reporter.events if e.phase == ReportPhase.START]
        outer_start = next(e for e in starts if e.model_name == "M")
        inner_start = next(e for e in starts if e.model_name == "inner")
        self.assertIsNone(outer_start.parent_span_id)
        self.assertEqual(inner_start.parent_span_id, outer_start.span_id)
        self.assertEqual(inner_start.depth, outer_start.depth + 1)

    def test_span_reset_after_call(self):
        policy = ReportingPolicy(reporter=InMemoryReporter())
        self.assertIsNone(current_span_id())
        self._run(policy, _ok)
        self.assertIsNone(current_span_id())

    def test_current_span_depth_tracks_nesting(self):
        policy = ReportingPolicy(reporter=InMemoryReporter())
        self.assertIsNone(current_span_depth())
        seen = []

        def outer() -> str:
            seen.append(current_span_depth())

            def inner() -> str:
                seen.append(current_span_depth())
                return "ok"

            policy._run_with_reporting(inner, model_name="inner", model_type="pkg.inner", fn="__call__", context="c")
            return "ok"

        self._run(policy, outer)
        self.assertEqual(seen, [0, 1])
        self.assertIsNone(current_span_depth())

    def test_context_repr_truncation(self):
        reporter = InMemoryReporter()
        policy = ReportingPolicy(reporter=reporter, max_context_repr=10)
        policy._run_with_reporting(_ok, model_name="M", model_type="pkg.M", fn="__call__", context="x" * 100)
        self.assertLessEqual(len(reporter.events[0].context_repr), 10)

    def test_capture_context_repr_disabled(self):
        reporter = InMemoryReporter()
        policy = ReportingPolicy(reporter=reporter, capture_context_repr=False)
        policy._run_with_reporting(_ok, model_name="M", model_type="pkg.M", fn="__call__", context="secret")
        self.assertEqual(reporter.events[0].context_repr, "")

    def test_context_repr_zero_is_empty(self):
        reporter = InMemoryReporter()
        policy = ReportingPolicy(reporter=reporter, max_context_repr=0)
        policy._run_with_reporting(_ok, model_name="M", model_type="pkg.M", fn="__call__", context="secret")
        self.assertEqual(reporter.events[0].context_repr, "")

    def test_reporter_failure_does_not_break_evaluation(self):
        class BrokenReporter(NoOpReporter):
            def emit(self, event):
                raise RuntimeError("sink down")

        policy = ReportingPolicy(reporter=BrokenReporter())
        with self.assertLogs("ccflow.utils.reporting", level="ERROR"):
            # The broken sink must not change the result nor raise.
            self.assertEqual(self._run(policy, _ok), "ok")

    def test_serializable(self):
        policy = ReportingPolicy(reporter=LoggingReporter(log_level=20))
        restored = pickle.loads(pickle.dumps(policy))
        self.assertEqual(restored, policy)
        dumped = policy.model_dump()
        self.assertEqual(ReportingPolicy.model_validate(dumped), policy)


class TestMetricsPolicy(TestCase):
    def test_metrics_extra(self):
        reporter = InMemoryReporter()
        policy = MetricsPolicy(reporter=reporter, metric_prefix="x")
        policy._run_with_reporting(_ok, model_name="M", model_type="pkg.M", fn="__call__", context="c")
        success = next(e for e in reporter.events if e.phase == ReportPhase.SUCCESS)
        self.assertEqual(success.extra["metric"], "x.success")
        self.assertIn("latency_seconds", success.extra)


class TestAlertsPolicy(TestCase):
    def test_alert_on_error_with_priority(self):
        reporter = InMemoryReporter()
        policy = AlertsPolicy(reporter=reporter, priority=AlertPriority.P1)
        with self.assertRaises(ValueError):
            policy._run_with_reporting(_boom, model_name="M", model_type="pkg.M", fn="__call__", context="c")
        errors = [e for e in reporter.events if e.phase == ReportPhase.ERROR]
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0].priority, AlertPriority.P1)

    def test_no_alert_on_success_by_default(self):
        reporter = InMemoryReporter()
        policy = AlertsPolicy(reporter=reporter, priority=AlertPriority.P2)
        policy._run_with_reporting(_ok, model_name="M", model_type="pkg.M", fn="__call__", context="c")
        self.assertEqual(reporter.events, [])

    def test_alert_on_success_opt_in(self):
        reporter = InMemoryReporter()
        policy = AlertsPolicy(reporter=reporter, alert_on_success=True, priority=AlertPriority.P5)
        policy._run_with_reporting(_ok, model_name="M", model_type="pkg.M", fn="__call__", context="c")
        self.assertEqual([e.phase for e in reporter.events], [ReportPhase.SUCCESS])
        self.assertEqual(reporter.events[0].priority, AlertPriority.P5)


class TestTracingPolicy(TestCase):
    def test_tracing_default_emits_span_events(self):
        reporter = InMemoryReporter()
        policy = TracingPolicy(reporter=reporter)
        policy._run_with_reporting(_ok, model_name="M", model_type="pkg.M", fn="__call__", context="c")
        self.assertEqual([e.phase for e in reporter.events], [ReportPhase.START, ReportPhase.SUCCESS, ReportPhase.END])


class TestRunScope(TestCase):
    def test_run_scope_sets_and_resets(self):
        self.assertIsNone(current_run_id())
        with run_scope("run-1") as rid:
            self.assertEqual(rid, "run-1")
            self.assertEqual(current_run_id(), "run-1")
        self.assertIsNone(current_run_id())

    def test_run_scope_generates_id(self):
        with run_scope() as rid:
            self.assertEqual(current_run_id(), rid)
            self.assertTrue(rid)

    def test_nested_run_scope_reuses_outer(self):
        with run_scope("outer"):
            with run_scope() as inner:
                self.assertEqual(inner, "outer")

    def test_events_tagged_with_run_id(self):
        reporter = InMemoryReporter()
        policy = ReportingPolicy(reporter=reporter)
        with run_scope("R"):
            policy._run_with_reporting(_ok, model_name="M", model_type="pkg.M", fn="__call__", context="c")
        self.assertTrue(all(e.run_id == "R" for e in reporter.events))


class TestUIReporter(TestCase):
    def test_drain_returns_and_clears(self):
        reporter = UIReporter()
        reporter.emit(ReportEvent(phase=ReportPhase.START, model_name="m", span_id="a"))
        reporter.emit(ReportEvent(phase=ReportPhase.END, model_name="m", span_id="a"))
        drained = reporter.drain()
        self.assertEqual([e.phase for e in drained], [ReportPhase.START, ReportPhase.END])
        self.assertEqual(reporter.drain(), [])

    def test_bounded_drops_oldest(self):
        reporter = UIReporter(maxlen=2)
        for i in range(5):
            reporter.emit(ReportEvent(phase=ReportPhase.START, model_name=str(i), span_id=str(i)))
        drained = reporter.drain()
        self.assertEqual([e.model_name for e in drained], ["3", "4"])

    def test_deepcopy_shares_buffer(self):
        import copy

        reporter = UIReporter()
        self.assertIs(copy.deepcopy(reporter), reporter)


class TestReportingStateStore(TestCase):
    def test_folds_terminal_phase(self):
        store = ReportingStateStore()
        store.apply(ReportEvent(phase=ReportPhase.START, model_name="m", span_id="a"))
        store.apply(ReportEvent(phase=ReportPhase.SUCCESS, model_name="m", span_id="a", duration=1.5))
        node = store.nodes["a"]
        self.assertEqual(node.phase, ReportPhase.SUCCESS)
        self.assertEqual(node.duration, 1.5)

    def test_terminal_not_overwritten_by_transient(self):
        store = ReportingStateStore()
        store.apply(ReportEvent(phase=ReportPhase.ERROR, model_name="m", span_id="a", exception_type="ValueError"))
        store.apply(ReportEvent(phase=ReportPhase.END, model_name="m", span_id="a"))
        # END marks completion but is not an outcome: the ERROR outcome and its detail must survive.
        self.assertEqual(store.nodes["a"].phase, ReportPhase.ERROR)
        self.assertEqual(store.nodes["a"].exception_type, "ValueError")

    def test_success_then_end_keeps_success(self):
        store = ReportingStateStore()
        store.apply_all(
            [
                ReportEvent(phase=ReportPhase.START, model_name="m", span_id="a"),
                ReportEvent(phase=ReportPhase.SUCCESS, model_name="m", span_id="a"),
                ReportEvent(phase=ReportPhase.END, model_name="m", span_id="a", duration=2.0),
            ]
        )
        node = store.nodes["a"]
        # The folded outcome is SUCCESS, but the duration carried on END is still merged in.
        self.assertEqual(node.phase, ReportPhase.SUCCESS)
        self.assertEqual(node.duration, 2.0)

    def test_retry_stream_error_then_retry_then_success(self):
        store = ReportingStateStore()
        store.apply_all(
            [
                ReportEvent(phase=ReportPhase.ERROR, model_name="m", span_id="a", attempt=1, exception_type="ValueError"),
                ReportEvent(phase=ReportPhase.RETRY, model_name="m", span_id="a", attempt=1),
                ReportEvent(phase=ReportPhase.SUCCESS, model_name="m", span_id="a", attempt=2),
            ]
        )
        node = store.nodes["a"]
        # The intermediate ERROR must yield to RETRY and finally SUCCESS, not stay stuck on ERROR.
        self.assertEqual(node.phase, ReportPhase.SUCCESS)
        self.assertEqual(node.attempt, 2)

    def test_late_start_does_not_clobber_outcome(self):
        store = ReportingStateStore()
        store.apply(ReportEvent(phase=ReportPhase.SUCCESS, model_name="m", span_id="a"))
        store.apply(ReportEvent(phase=ReportPhase.START, model_name="m", span_id="a"))
        self.assertEqual(store.nodes["a"].phase, ReportPhase.SUCCESS)

    def test_parent_child_tree(self):
        store = ReportingStateStore()
        store.apply(ReportEvent(phase=ReportPhase.START, model_name="root", span_id="r"))
        store.apply(ReportEvent(phase=ReportPhase.START, model_name="child", span_id="c", parent_span_id="r"))
        roots = store.roots()
        self.assertEqual([n.span_id for n in roots], ["r"])
        self.assertEqual([n.span_id for n in store.children("r")], ["c"])

    def test_run_phases_recorded(self):
        store = ReportingStateStore()
        store.apply(ReportEvent(phase=ReportPhase.RUN_STARTED, model_name="", span_id="x", run_id="R"))
        store.apply(ReportEvent(phase=ReportPhase.RUN_FINISHED, model_name="", span_id="x", run_id="R"))
        self.assertEqual(store.runs["R"], ReportPhase.RUN_FINISHED)
        # Run events should not create node state.
        self.assertEqual(store.nodes, {})

    def test_apply_all_node_state_type(self):
        store = ReportingStateStore()
        store.apply_all([ReportEvent(phase=ReportPhase.QUEUED, model_name="m", span_id="a")])
        self.assertIsInstance(store.nodes["a"], NodeState)


class TestLoggingPolicy(TestCase):
    def _ctx_extra(self):
        return {"model": "MODELREPR", "raw_context": "CTXREPR", "options": {}}

    def test_logs_start_and_end(self):
        policy = LoggingPolicy(log_level=logging.INFO)
        with self.assertLogs(level=logging.INFO) as captured:
            policy._run_with_reporting(_ok, model_name="M", model_type="pkg.M", fn="__call__", context="c", extra=self._ctx_extra())
        messages = [r.getMessage() for r in captured.records]
        self.assertTrue(any("Start evaluation of __call__ on CTXREPR" in m for m in messages))
        self.assertTrue(any("End evaluation of __call__ on CTXREPR" in m and "time elapsed" in m for m in messages))

    def test_verbose_logs_model(self):
        policy = LoggingPolicy(log_level=logging.INFO, verbose=True)
        with self.assertLogs(level=logging.INFO) as captured:
            policy._run_with_reporting(_ok, model_name="M", model_type="pkg.M", fn="__call__", context="c", extra=self._ctx_extra())
        self.assertTrue(any("MODELREPR" in r.getMessage() for r in captured.records))

    def test_options_override_log_level(self):
        policy = LoggingPolicy(log_level=logging.DEBUG)
        extra = {"model": "m", "raw_context": "c", "options": {"log_level": logging.WARNING}}
        with self.assertLogs(level=logging.WARNING) as captured:
            policy._run_with_reporting(_ok, model_name="M", model_type="pkg.M", fn="__call__", context="c", extra=extra)
        self.assertTrue(all(r.levelno == logging.WARNING for r in captured.records))

    def test_emits_structured_events_when_reporter_set(self):
        reporter = InMemoryReporter()
        policy = LoggingPolicy(log_level=logging.INFO, verbose=False, reporter=reporter)
        with self.assertLogs(level=logging.INFO):
            policy._run_with_reporting(_ok, model_name="M", model_type="pkg.M", fn="__call__", context="c", extra=self._ctx_extra())
        self.assertEqual([e.phase for e in reporter.events], [ReportPhase.START, ReportPhase.SUCCESS, ReportPhase.END])
