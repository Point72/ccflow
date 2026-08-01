import logging
import pickle
from collections.abc import Callable
from unittest import TestCase
from unittest.mock import patch

from ccflow.exttypes import PyObjectPath
from ccflow.utils.retry import RetryError, RetryPolicy


def _flaky(fail_times: int, exc: Exception) -> Callable[[], str]:
    """Return a zero-arg callable that raises ``exc`` the first ``fail_times`` calls, then returns "ok"."""
    state = {"calls": 0}

    def attempt() -> str:
        state["calls"] += 1
        if state["calls"] <= fail_times:
            raise exc
        return "ok"

    attempt.state = state  # type: ignore[attr-defined]
    return attempt


class TestRetryPolicy(TestCase):
    def _run(self, policy: RetryPolicy, attempt_fn: Callable[[], str]) -> str:
        return policy._run_with_retry(attempt_fn, name="test", detail="attempt")

    def test_success_no_retry(self):
        policy = RetryPolicy(max_attempts=3)
        attempt = _flaky(0, ValueError("boom"))
        self.assertEqual(self._run(policy, attempt), "ok")
        self.assertEqual(attempt.state["calls"], 1)

    def test_retries_then_succeeds(self):
        policy = RetryPolicy(max_attempts=3)
        attempt = _flaky(2, ValueError("boom"))
        self.assertEqual(self._run(policy, attempt), "ok")
        self.assertEqual(attempt.state["calls"], 3)

    def test_exhausts_and_reraises(self):
        policy = RetryPolicy(max_attempts=3)
        attempt = _flaky(5, ValueError("boom"))
        with self.assertRaises(ValueError):
            self._run(policy, attempt)
        self.assertEqual(attempt.state["calls"], 3)

    def test_exhausts_raises_retry_error(self):
        policy = RetryPolicy(max_attempts=2, reraise=False)
        attempt = _flaky(5, ValueError("boom"))
        with self.assertRaises(RetryError) as ctx:
            self._run(policy, attempt)
        self.assertEqual(ctx.exception.attempts, 2)
        self.assertIn("attempts exhausted", str(ctx.exception))
        self.assertIsInstance(ctx.exception.last_exception, ValueError)
        self.assertIsInstance(ctx.exception.__cause__, ValueError)

    def test_max_delay_budget_exceeded_message(self):
        # When the loop terminates because the max_delay budget would be exceeded (rather than
        # running out of attempts), the failure message distinguishes the two cases.
        policy = RetryPolicy(max_attempts=10, wait_initial=5.0, wait_multiplier=1.0, max_delay=1.0, reraise=False)
        attempt = _flaky(5, ValueError("boom"))
        with patch("ccflow.utils.retry.time.sleep"), self.assertRaises(RetryError) as ctx:
            self._run(policy, attempt)
        self.assertIn("max_delay budget exceeded", str(ctx.exception))
        self.assertEqual(ctx.exception.attempts, 1)

    def test_non_matching_exception_not_retried(self):
        policy = RetryPolicy(max_attempts=3, retry_exceptions=[ValueError])
        attempt = _flaky(5, KeyError("boom"))
        with self.assertRaises(KeyError):
            self._run(policy, attempt)
        self.assertEqual(attempt.state["calls"], 1)

    def test_no_retry_exceptions_precedence(self):
        policy = RetryPolicy(max_attempts=3, retry_exceptions=[Exception], no_retry_exceptions=[KeyError])
        attempt = _flaky(5, KeyError("boom"))
        with self.assertRaises(KeyError):
            self._run(policy, attempt)
        self.assertEqual(attempt.state["calls"], 1)

    def test_invalid_exception_type(self):
        with self.assertRaises(ValueError):
            RetryPolicy(retry_exceptions=[int])

    def test_non_exception_base_rejected(self):
        # BaseException subclasses that are not Exception subclasses are never caught by the
        # retry loop, so they are rejected at validation time.
        with self.assertRaises(ValueError):
            RetryPolicy(retry_exceptions=[KeyboardInterrupt])

    def test_log_level_string(self):
        self.assertEqual(RetryPolicy(log_level="INFO").log_level, logging.INFO)
        self.assertEqual(RetryPolicy(log_level="info").log_level, logging.INFO)
        with self.assertRaises(ValueError):
            RetryPolicy(log_level="garbage")

    def test_compute_delay_backoff(self):
        policy = RetryPolicy(wait_initial=1.0, wait_multiplier=2.0)
        self.assertEqual(policy._compute_delay(1), 1.0)
        self.assertEqual(policy._compute_delay(2), 2.0)
        self.assertEqual(policy._compute_delay(3), 4.0)

    def test_compute_delay_max(self):
        policy = RetryPolicy(wait_initial=1.0, wait_multiplier=2.0, wait_max=3.0)
        self.assertEqual(policy._compute_delay(3), 3.0)

    def test_compute_delay_jitter(self):
        policy = RetryPolicy(wait_initial=1.0, wait_multiplier=1.0, wait_jitter=0.5)
        self.assertEqual(policy._compute_delay(1, jitter_value=0.25), 1.25)

    def test_sleep_between_retries(self):
        policy = RetryPolicy(max_attempts=3, wait_initial=1.0, wait_multiplier=2.0)
        attempt = _flaky(2, ValueError("boom"))
        with patch("ccflow.utils.retry.time.sleep") as sleep_mock:
            self._run(policy, attempt)
        self.assertEqual([call.args[0] for call in sleep_mock.call_args_list], [1.0, 2.0])

    def test_max_delay_stops_retries(self):
        policy = RetryPolicy(max_attempts=10, wait_initial=5.0, wait_multiplier=1.0, max_delay=1.0)
        attempt = _flaky(5, ValueError("boom"))
        with patch("ccflow.utils.retry.time.sleep") as sleep_mock, self.assertRaises(ValueError):
            self._run(policy, attempt)
        # The first computed delay (5.0) already exceeds max_delay (1.0), so no sleep occurs.
        sleep_mock.assert_not_called()
        self.assertEqual(attempt.state["calls"], 1)

    def test_max_delay_cumulative_budget(self):
        # With a flat 1.0s delay and a 2.5s budget, two sleeps (1.0 + 1.0 = 2.0) fit but a third
        # (2.0 + 1.0 = 3.0) would exceed the budget, so retries stop after the third attempt.
        policy = RetryPolicy(max_attempts=10, wait_initial=1.0, wait_multiplier=1.0, max_delay=2.5)
        attempt = _flaky(5, ValueError("boom"))
        with patch("ccflow.utils.retry.time.sleep") as sleep_mock, self.assertRaises(ValueError):
            self._run(policy, attempt)
        self.assertEqual([call.args[0] for call in sleep_mock.call_args_list], [1.0, 1.0])
        self.assertEqual(attempt.state["calls"], 3)

    def test_logs_retry(self):
        policy = RetryPolicy(max_attempts=3, log_level=logging.WARNING)
        attempt = _flaky(1, ValueError("boom"))
        with self.assertLogs(level=logging.WARNING) as captured:
            self._run(policy, attempt)
        self.assertEqual(len(captured.records), 1)
        self.assertIn("Retrying", captured.records[0].getMessage())

    def test_serializable(self):
        policy = RetryPolicy(
            max_attempts=5,
            retry_exceptions=[ValueError, KeyError],
            wait_initial=0.1,
            wait_jitter=0.2,
        )
        # Round-trip through pickle (e.g. shipping to a Ray/Celery worker)
        restored = pickle.loads(pickle.dumps(policy))
        self.assertEqual(restored, policy)
        # Round-trip through model serialization (config friendly)
        dumped = policy.model_dump()
        self.assertEqual(dumped["retry_exceptions"], [PyObjectPath.validate(ValueError), PyObjectPath.validate(KeyError)])
        self.assertEqual(RetryPolicy.model_validate(dumped), policy)


class TestRetryReporting(TestCase):
    def _run(self, policy: RetryPolicy, attempt_fn: Callable[[], str]) -> str:
        return policy._run_with_retry(attempt_fn, name="test", detail="__call__ on ctx")

    def test_no_reporter_no_events(self):
        # Reporting is fully opt-in; with no reporter there is zero overhead and no events.
        policy = RetryPolicy(max_attempts=3)
        attempt = _flaky(1, ValueError("boom"))
        self.assertEqual(self._run(policy, attempt), "ok")  # does not raise

    def test_success_first_try_reports_success(self):
        from ccflow.utils.reporting import InMemoryReporter, ReportPhase

        reporter = InMemoryReporter()
        policy = RetryPolicy(max_attempts=3, reporter=reporter)
        self._run(policy, _flaky(0, ValueError("boom")))
        self.assertEqual([e.phase for e in reporter.events], [ReportPhase.SUCCESS])
        self.assertEqual(reporter.events[0].attempt, 1)
        self.assertEqual(reporter.events[0].max_attempts, 3)

    def test_retry_then_succeed_lifecycle(self):
        from ccflow.utils.reporting import InMemoryReporter, ReportPhase

        reporter = InMemoryReporter()
        policy = RetryPolicy(max_attempts=3, reporter=reporter)
        self._run(policy, _flaky(2, ValueError("boom")))
        # Two failures, each producing ERROR then RETRY, then a final SUCCESS.
        self.assertEqual(
            [e.phase for e in reporter.events],
            [ReportPhase.ERROR, ReportPhase.RETRY, ReportPhase.ERROR, ReportPhase.RETRY, ReportPhase.SUCCESS],
        )

    def test_events_tagged_with_run_id(self):
        from ccflow.utils.reporting import InMemoryReporter, run_scope

        reporter = InMemoryReporter()
        policy = RetryPolicy(max_attempts=3, reporter=reporter)
        with run_scope("R"):
            self._run(policy, _flaky(2, ValueError("boom")))
        self.assertTrue(reporter.events)
        self.assertTrue(all(e.run_id == "R" for e in reporter.events))

    def test_events_nested_under_reporting_span_have_parent_and_depth(self):
        from ccflow.utils.reporting import InMemoryReporter, ReportingPolicy

        reporter = InMemoryReporter()
        retry = RetryPolicy(max_attempts=3, reporter=reporter)
        outer = ReportingPolicy(reporter=reporter)

        def attempt() -> str:
            return self._run(retry, _flaky(2, ValueError("boom")))

        # Run the retry inside an active reporting span; retry events should nest one level deeper.
        outer._run_with_reporting(attempt, model_name="outer", model_type="pkg.outer", fn="__call__", context="c")
        outer_start = next(e for e in reporter.events if e.model_name == "outer" and e.phase.name == "START")
        retry_events = [e for e in reporter.events if e.model_name == "test"]
        self.assertTrue(retry_events)
        for event in retry_events:
            self.assertEqual(event.parent_span_id, outer_start.span_id)
            self.assertEqual(event.depth, outer_start.depth + 1)

    def test_reporter_failure_does_not_break_retry(self):
        from ccflow.utils.reporting import NoOpReporter

        class BrokenReporter(NoOpReporter):
            def emit(self, event):
                raise RuntimeError("sink down")

        policy = RetryPolicy(max_attempts=3, reporter=BrokenReporter())
        with self.assertLogs("ccflow.utils.retry", level="ERROR"):
            self.assertEqual(self._run(policy, _flaky(1, ValueError("boom"))), "ok")

    def test_give_up_on_exhaustion(self):
        from ccflow.utils.reporting import InMemoryReporter, ReportPhase

        reporter = InMemoryReporter()
        policy = RetryPolicy(max_attempts=2, reporter=reporter)
        with self.assertRaises(ValueError):
            self._run(policy, _flaky(5, ValueError("boom")))
        phases = [e.phase for e in reporter.events]
        self.assertEqual(phases[-1], ReportPhase.GIVE_UP)
        give_up = reporter.events[-1]
        self.assertEqual(give_up.extra["reason"], "attempts exhausted")

    def test_give_up_on_non_retryable(self):
        from ccflow.utils.reporting import InMemoryReporter, ReportPhase

        reporter = InMemoryReporter()
        policy = RetryPolicy(max_attempts=3, retry_exceptions=[ValueError], reporter=reporter)
        with self.assertRaises(KeyError):
            self._run(policy, _flaky(5, KeyError("boom")))
        self.assertEqual([e.phase for e in reporter.events], [ReportPhase.ERROR, ReportPhase.GIVE_UP])
        self.assertEqual(reporter.events[-1].extra["reason"], "non-retryable")

    def test_give_up_on_budget(self):
        from ccflow.utils.reporting import InMemoryReporter, ReportPhase

        reporter = InMemoryReporter()
        policy = RetryPolicy(max_attempts=10, wait_initial=5.0, wait_multiplier=1.0, max_delay=1.0, reporter=reporter)
        with patch("ccflow.utils.retry.time.sleep"), self.assertRaises(ValueError):
            self._run(policy, _flaky(5, ValueError("boom")))
        give_up = reporter.events[-1]
        self.assertEqual(give_up.phase, ReportPhase.GIVE_UP)
        self.assertEqual(give_up.extra["reason"], "max_delay budget exceeded")
