import pickle
from datetime import date
from unittest import TestCase

from ccflow import DateContext, FlowOptionsOverride, ModelEvaluationContext, TransparentModelEvaluationContext
from ccflow.evaluators import RetryEvaluator
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

    def test_is_transparent(self):
        model = MyFlakyCallable(fail_times=0)
        evaluator = RetryEvaluator()
        context = self._eval_context(model)
        self.assertTrue(evaluator.is_transparent(context))
        wrapped = evaluator.make_evaluation_context(context, options=context.options)
        self.assertIsInstance(wrapped, TransparentModelEvaluationContext)

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
        )
        # Round-trip through pickle (e.g. shipping to a Ray/Celery worker)
        restored = pickle.loads(pickle.dumps(evaluator))
        self.assertEqual(restored, evaluator)
        # Round-trip through model serialization (config friendly)
        dumped = evaluator.model_dump()
        self.assertEqual(dumped["retry_exceptions"], [PyObjectPath.validate(ValueError), PyObjectPath.validate(KeyError)])
        self.assertEqual(RetryEvaluator.model_validate(dumped), evaluator)
