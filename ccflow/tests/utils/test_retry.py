import logging
import pickle
from typing import Callable
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
        with patch("ccflow.utils.retry.time.sleep"):
            with self.assertRaises(RetryError) as ctx:
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
        with patch("ccflow.utils.retry.time.sleep") as sleep_mock:
            with self.assertRaises(ValueError):
                self._run(policy, attempt)
        # The first computed delay (5.0) already exceeds max_delay (1.0), so no sleep occurs.
        sleep_mock.assert_not_called()
        self.assertEqual(attempt.state["calls"], 1)

    def test_max_delay_cumulative_budget(self):
        # With a flat 1.0s delay and a 2.5s budget, two sleeps (1.0 + 1.0 = 2.0) fit but a third
        # (2.0 + 1.0 = 3.0) would exceed the budget, so retries stop after the third attempt.
        policy = RetryPolicy(max_attempts=10, wait_initial=1.0, wait_multiplier=1.0, max_delay=2.5)
        attempt = _flaky(5, ValueError("boom"))
        with patch("ccflow.utils.retry.time.sleep") as sleep_mock:
            with self.assertRaises(ValueError):
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
