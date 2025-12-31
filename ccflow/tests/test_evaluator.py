from datetime import date
from unittest import TestCase

import pytest

from ccflow import CallableModel, DateContext, Evaluator, Flow, ModelEvaluationContext

from .evaluators.util import MyDateCallable, MyResult


class MyDynamicDateCallable(CallableModel):
    """Dynamic context version of MyDateCallable for testing evaluators."""

    offset: int

    @Flow.dynamic_call(parent=DateContext)
    def __call__(self, *, date: date) -> MyResult:
        return MyResult(x=date.day + self.offset)


class TestEvaluator(TestCase):
    def test_evaluator(self):
        m1 = MyDateCallable(offset=1)
        context = DateContext(date=date(2022, 1, 1))
        model_evaluation_context = ModelEvaluationContext(model=m1, context=context)
        model_evaluation_context2 = ModelEvaluationContext(model=m1, context=context, fn="__call__")

        out = model_evaluation_context()
        self.assertEqual(out, m1(context))
        out2 = model_evaluation_context2()
        self.assertEqual(out, out2)

        evaluator = Evaluator()
        out2 = evaluator(model_evaluation_context)
        self.assertEqual(out2, out)

    def test_evaluator_deps(self):
        m1 = MyDateCallable(offset=1)
        context = DateContext(date=date(2022, 1, 1))
        model_evaluation_context = ModelEvaluationContext(model=m1, context=context, fn="__deps__")
        out = model_evaluation_context()
        self.assertEqual(out, m1.__deps__(context))

        evaluator = Evaluator()
        out2 = evaluator.__deps__(model_evaluation_context)
        self.assertEqual(out2, out)


@pytest.mark.parametrize(
    "callable_class",
    [MyDateCallable, MyDynamicDateCallable],
    ids=["standard", "dynamic"],
)
class TestEvaluatorParametrized:
    """Test evaluators work with both standard and dynamic context callables."""

    def test_evaluator_with_context_object(self, callable_class):
        """Test evaluator with a context object."""
        m1 = callable_class(offset=1)
        context = DateContext(date=date(2022, 1, 1))
        model_evaluation_context = ModelEvaluationContext(model=m1, context=context)

        out = model_evaluation_context()
        assert out == MyResult(x=2)  # day 1 + offset 1

        evaluator = Evaluator()
        out2 = evaluator(model_evaluation_context)
        assert out2 == out

    def test_evaluator_with_fn_specified(self, callable_class):
        """Test evaluator with fn='__call__' explicitly specified."""
        m1 = callable_class(offset=1)
        context = DateContext(date=date(2022, 1, 1))
        model_evaluation_context = ModelEvaluationContext(model=m1, context=context, fn="__call__")

        out = model_evaluation_context()
        assert out == MyResult(x=2)

    def test_evaluator_direct_call_matches(self, callable_class):
        """Test that evaluator result matches direct call."""
        m1 = callable_class(offset=5)
        context = DateContext(date=date(2022, 1, 15))

        # Direct call
        direct_result = m1(context)

        # Via evaluator
        model_evaluation_context = ModelEvaluationContext(model=m1, context=context)
        evaluator_result = model_evaluation_context()

        assert direct_result == evaluator_result
        assert direct_result == MyResult(x=20)  # day 15 + offset 5

    def test_evaluator_with_kwargs(self, callable_class):
        """Test that evaluator works when callable is called with kwargs."""
        m1 = callable_class(offset=1)

        # Call with kwargs
        result = m1(date=date(2022, 1, 10))
        assert result == MyResult(x=11)  # day 10 + offset 1
