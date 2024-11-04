from datetime import date
from unittest import TestCase

from ccflow import DateContext, Evaluator, ModelEvaluationContext

from .evaluators.util import MyDateCallable


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
