import logging
import pickle
from datetime import date
from unittest import TestCase
from unittest.mock import patch

from ccflow import DateContext
from ccflow.models import RetryModel
from ccflow.utils.retry import RetryError

from ..evaluators.util import MyFlakyCallable, MyResult


class TestRetryModel(TestCase):
    def setUp(self):
        self.context = DateContext(date=date(2022, 1, 1))

    def test_success_no_retry(self):
        inner = MyFlakyCallable(offset=1, fail_times=0)
        model = RetryModel(model=inner, max_attempts=3)
        out = model(self.context)
        self.assertEqual(out, MyResult(x=2))
        self.assertEqual(inner.calls, 1)

    def test_retries_then_succeeds(self):
        inner = MyFlakyCallable(offset=1, fail_times=2)
        model = RetryModel(model=inner, max_attempts=3)
        out = model(self.context)
        self.assertEqual(out, MyResult(x=2))
        self.assertEqual(inner.calls, 3)

    def test_exhausts_and_reraises(self):
        inner = MyFlakyCallable(offset=1, fail_times=5)
        model = RetryModel(model=inner, max_attempts=3)
        with self.assertRaises(ValueError):
            model(self.context)
        self.assertEqual(inner.calls, 3)

    def test_exhausts_raises_retry_error(self):
        inner = MyFlakyCallable(offset=1, fail_times=5)
        model = RetryModel(model=inner, max_attempts=2, reraise=False)
        with self.assertRaises(RetryError) as ctx:
            model(self.context)
        self.assertEqual(ctx.exception.attempts, 2)
        self.assertIsInstance(ctx.exception.last_exception, ValueError)
        self.assertIsInstance(ctx.exception.__cause__, ValueError)

    def test_non_matching_exception_not_retried(self):
        inner = MyFlakyCallable(fail_times=5, exception_type=KeyError)
        model = RetryModel(model=inner, max_attempts=3, retry_exceptions=[ValueError])
        with self.assertRaises(KeyError):
            model(self.context)
        self.assertEqual(inner.calls, 1)

    def test_preserves_context_and_result_type(self):
        inner = MyFlakyCallable()
        model = RetryModel(model=inner, max_attempts=3)
        self.assertEqual(model.context_type, inner.context_type)
        self.assertEqual(model.result_type, inner.result_type)

    def test_sleep_between_retries(self):
        inner = MyFlakyCallable(fail_times=2)
        model = RetryModel(model=inner, max_attempts=3, wait_initial=0.5, wait_multiplier=2.0)
        with patch("ccflow.utils.retry.time.sleep") as sleep_mock:
            model(self.context)
        self.assertEqual([call.args[0] for call in sleep_mock.call_args_list], [0.5, 1.0])

    def test_logs_retry(self):
        inner = MyFlakyCallable(fail_times=1)
        model = RetryModel(model=inner, max_attempts=3, log_level=logging.WARNING)
        with self.assertLogs(level=logging.WARNING) as captured:
            model(self.context)
        self.assertEqual(len(captured.records), 1)
        self.assertIn("Retrying", captured.records[0].getMessage())

    def test_serializable(self):
        inner = MyFlakyCallable(offset=1, fail_times=0)
        model = RetryModel(model=inner, max_attempts=5, retry_exceptions=[ValueError, KeyError], wait_initial=0.1)
        restored = pickle.loads(pickle.dumps(model))
        self.assertEqual(restored, model)
        dumped = model.model_dump()
        self.assertEqual(RetryModel.model_validate(dumped), model)
