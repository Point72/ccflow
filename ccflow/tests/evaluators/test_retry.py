import logging
import pickle
from datetime import date
from unittest import TestCase
from unittest.mock import patch

from ccflow import DateContext, FlowOptionsOverride, ModelEvaluationContext, TransparentModelEvaluationContext
from ccflow.evaluators import RetryError, RetryEvaluator
from ccflow.exttypes import PyObjectPath

from .util import MyFlakyCallable, MyResult


class TestRetryEvaluator(TestCase):
    def setUp(self):
        self.context = DateContext(date=date(2022, 1, 1))

    def _eval_context(self, model):
        return ModelEvaluationContext(model=model, context=self.context)

    def test_success_no_retry(self):
        model = MyFlakyCallable(offset=1, fail_times=0)
        evaluator = RetryEvaluator(max_attempts=3)
        out = evaluator(self._eval_context(model))
        self.assertEqual(out, MyResult(x=2))
        self.assertEqual(model.calls, 1)

    def test_retries_then_succeeds(self):
        model = MyFlakyCallable(offset=1, fail_times=2)
        evaluator = RetryEvaluator(max_attempts=3)
        out = evaluator(self._eval_context(model))
        self.assertEqual(out, MyResult(x=2))
        self.assertEqual(model.calls, 3)

    def test_exhausts_and_reraises(self):
        model = MyFlakyCallable(offset=1, fail_times=5)
        evaluator = RetryEvaluator(max_attempts=3)
        with self.assertRaises(ValueError):
            evaluator(self._eval_context(model))
        self.assertEqual(model.calls, 3)

    def test_exhausts_raises_retry_error(self):
        model = MyFlakyCallable(offset=1, fail_times=5)
        evaluator = RetryEvaluator(max_attempts=2, reraise=False)
        with self.assertRaises(RetryError) as ctx:
            evaluator(self._eval_context(model))
        self.assertEqual(ctx.exception.attempts, 2)
        self.assertIsInstance(ctx.exception.last_exception, ValueError)
        self.assertIsInstance(ctx.exception.__cause__, ValueError)

    def test_non_matching_exception_not_retried(self):
        model = MyFlakyCallable(fail_times=5, exception_type=KeyError)
        evaluator = RetryEvaluator(max_attempts=3, retry_exceptions=[ValueError])
        with self.assertRaises(KeyError):
            evaluator(self._eval_context(model))
        self.assertEqual(model.calls, 1)

    def test_no_retry_exceptions_precedence(self):
        model = MyFlakyCallable(fail_times=5, exception_type=KeyError)
        evaluator = RetryEvaluator(max_attempts=3, retry_exceptions=[Exception], no_retry_exceptions=[KeyError])
        with self.assertRaises(KeyError):
            evaluator(self._eval_context(model))
        self.assertEqual(model.calls, 1)

    def test_invalid_exception_type(self):
        with self.assertRaises(ValueError):
            RetryEvaluator(retry_exceptions=[int])

    def test_non_exception_base_rejected(self):
        # BaseException subclasses that are not Exception subclasses are never caught by the
        # retry loop, so they are rejected at validation time.
        with self.assertRaises(ValueError):
            RetryEvaluator(retry_exceptions=[KeyboardInterrupt])

    def test_log_level_string(self):
        self.assertEqual(RetryEvaluator(log_level="INFO").log_level, logging.INFO)
        self.assertEqual(RetryEvaluator(log_level="info").log_level, logging.INFO)
        with self.assertRaises(ValueError):
            RetryEvaluator(log_level="garbage")

    def test_is_transparent(self):
        model = MyFlakyCallable(fail_times=0)
        evaluator = RetryEvaluator()
        context = self._eval_context(model)
        self.assertTrue(evaluator.is_transparent(context))
        wrapped = evaluator.make_evaluation_context(context, options=context.options)
        self.assertIsInstance(wrapped, TransparentModelEvaluationContext)

    def test_compute_delay_backoff(self):
        evaluator = RetryEvaluator(wait_initial=1.0, wait_multiplier=2.0)
        self.assertEqual(evaluator._compute_delay(1), 1.0)
        self.assertEqual(evaluator._compute_delay(2), 2.0)
        self.assertEqual(evaluator._compute_delay(3), 4.0)

    def test_compute_delay_max(self):
        evaluator = RetryEvaluator(wait_initial=1.0, wait_multiplier=2.0, wait_max=3.0)
        self.assertEqual(evaluator._compute_delay(3), 3.0)

    def test_compute_delay_jitter(self):
        evaluator = RetryEvaluator(wait_initial=1.0, wait_multiplier=1.0, wait_jitter=0.5)
        self.assertEqual(evaluator._compute_delay(1, jitter_value=0.25), 1.25)

    def test_sleep_between_retries(self):
        model = MyFlakyCallable(fail_times=2)
        evaluator = RetryEvaluator(max_attempts=3, wait_initial=1.0, wait_multiplier=2.0)
        with patch("ccflow.evaluators.retry.time.sleep") as sleep_mock:
            evaluator(self._eval_context(model))
        self.assertEqual([call.args[0] for call in sleep_mock.call_args_list], [1.0, 2.0])

    def test_max_delay_stops_retries(self):
        model = MyFlakyCallable(fail_times=5)
        evaluator = RetryEvaluator(max_attempts=10, wait_initial=5.0, wait_multiplier=1.0, max_delay=1.0)
        with patch("ccflow.evaluators.retry.time.sleep") as sleep_mock:
            with self.assertRaises(ValueError):
                evaluator(self._eval_context(model))
        # The first computed delay (5.0) already exceeds max_delay (1.0), so no sleep occurs.
        sleep_mock.assert_not_called()
        self.assertEqual(model.calls, 1)

    def test_max_delay_cumulative_budget(self):
        # With a flat 1.0s delay and a 2.5s budget, two sleeps (1.0 + 1.0 = 2.0) fit but a third
        # (2.0 + 1.0 = 3.0) would exceed the budget, so retries stop after the third attempt.
        model = MyFlakyCallable(fail_times=5)
        evaluator = RetryEvaluator(max_attempts=10, wait_initial=1.0, wait_multiplier=1.0, max_delay=2.5)
        with patch("ccflow.evaluators.retry.time.sleep") as sleep_mock:
            with self.assertRaises(ValueError):
                evaluator(self._eval_context(model))
        self.assertEqual([call.args[0] for call in sleep_mock.call_args_list], [1.0, 1.0])
        self.assertEqual(model.calls, 3)

    def test_logs_retry(self):
        model = MyFlakyCallable(fail_times=1)
        evaluator = RetryEvaluator(max_attempts=3, log_level=logging.WARNING)
        with self.assertLogs(level=logging.WARNING) as captured:
            evaluator(self._eval_context(model))
        self.assertEqual(len(captured.records), 1)
        self.assertIn("Retrying", captured.records[0].getMessage())

    def test_flow_options_override_integration(self):
        # Exercise the real wiring path: the evaluator is applied via FlowOptionsOverride
        # when the model is called directly.
        model = MyFlakyCallable(offset=1, fail_times=2)
        evaluator = RetryEvaluator(max_attempts=3)
        with FlowOptionsOverride(options={"evaluator": evaluator}):
            out = model(self.context)
        self.assertEqual(out, MyResult(x=2))
        self.assertEqual(model.calls, 3)

    def test_serializable(self):
        evaluator = RetryEvaluator(
            max_attempts=5,
            retry_exceptions=[ValueError, KeyError],
            wait_initial=0.1,
            wait_jitter=0.2,
        )
        # Round-trip through pickle (e.g. shipping to a Ray/Celery worker)
        restored = pickle.loads(pickle.dumps(evaluator))
        self.assertEqual(restored, evaluator)
        # Round-trip through model serialization (config friendly)
        dumped = evaluator.model_dump()
        self.assertEqual(dumped["retry_exceptions"], [PyObjectPath.validate(ValueError), PyObjectPath.validate(KeyError)])
        self.assertEqual(RetryEvaluator.model_validate(dumped), evaluator)
