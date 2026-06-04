import pickle
from datetime import date
from unittest import TestCase

from ccflow import DateContext, FlowOptionsOverride, ModelEvaluationContext, TransparentModelEvaluationContext
from ccflow.evaluators import RetryEvaluator
from ccflow.exttypes import PyObjectPath

from .util import MyFlakyCallable, MyResult


class MyOtherFlakyCallable(MyFlakyCallable):
    """A distinct CallableModel type (sibling for include/exclude selection tests)."""


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

    def test_include_model_types(self):
        # Only the included type is retried; other models pass straight through.
        included = MyOtherFlakyCallable(fail_times=2)
        excluded = MyFlakyCallable(fail_times=2)
        evaluator = RetryEvaluator(max_attempts=3, include_model_types=[MyOtherFlakyCallable])
        self.assertEqual(evaluator(self._eval_context(included)), MyResult(x=1))
        self.assertEqual(included.calls, 3)
        with self.assertRaises(ValueError):
            evaluator(self._eval_context(excluded))
        self.assertEqual(excluded.calls, 1)

    def test_exclude_model_types(self):
        # The excluded type passes straight through; everything else is retried.
        excluded = MyOtherFlakyCallable(fail_times=2)
        retried = MyFlakyCallable(fail_times=2)
        evaluator = RetryEvaluator(max_attempts=3, exclude_model_types=[MyOtherFlakyCallable])
        with self.assertRaises(ValueError):
            evaluator(self._eval_context(excluded))
        self.assertEqual(excluded.calls, 1)
        self.assertEqual(evaluator(self._eval_context(retried)), MyResult(x=1))
        self.assertEqual(retried.calls, 3)

    def test_exclude_precedence_over_include(self):
        # exclude_model_types wins even when the model also matches include_model_types.
        model = MyOtherFlakyCallable(fail_times=2)
        evaluator = RetryEvaluator(
            max_attempts=3,
            include_model_types=[MyFlakyCallable],
            exclude_model_types=[MyOtherFlakyCallable],
        )
        with self.assertRaises(ValueError):
            evaluator(self._eval_context(model))
        self.assertEqual(model.calls, 1)

    def test_invalid_model_type(self):
        with self.assertRaises(ValueError):
            RetryEvaluator(include_model_types=["math.pi"])

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
            include_model_types=[MyOtherFlakyCallable],
            exclude_model_types=[MyFlakyCallable],
        )
        # Round-trip through pickle (e.g. shipping to a Ray/Celery worker)
        restored = pickle.loads(pickle.dumps(evaluator))
        self.assertEqual(restored, evaluator)
        # Round-trip through model serialization (config friendly)
        dumped = evaluator.model_dump()
        self.assertEqual(dumped["retry_exceptions"], [PyObjectPath.validate(ValueError), PyObjectPath.validate(KeyError)])
        self.assertEqual(RetryEvaluator.model_validate(dumped), evaluator)
