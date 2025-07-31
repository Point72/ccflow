import logging
from datetime import date
from unittest import TestCase

from ccflow import CallableModel, DateContext, EvaluatorBase, Flow, FlowOptions, FlowOptionsOverride, GenericResult, ModelEvaluationContext
from ccflow.evaluators import MemoryCacheEvaluator


class MyModel(CallableModel):
    @Flow.call
    def foo(self, context):
        return context.date

    @Flow.call()
    def bar(self, context):
        return context.date

    @Flow.call(validate_result=False)
    def baz(self, context) -> date:
        return context.date

    @Flow.call(log_level=logging.WARNING)
    def __call__(self, context: DateContext) -> GenericResult:
        return context.date


class SomeClass:
    @Flow.call
    def foo(self):
        return None


class BadContextModel(CallableModel):
    @property
    def context_type(self):
        return date

    @Flow.call
    def __call__(self, context) -> GenericResult:
        return context.date


class BadResultModel(CallableModel):
    @property
    def result_type(self):
        return date

    @Flow.call
    def __call__(self, context: DateContext):
        return context.date


class DefaultModel(CallableModel):
    @Flow.call
    def __call__(self, context: DateContext = "0d") -> GenericResult:
        return context.date


class TestModelEvaluationContext(TestCase):
    def test_model_evaluation_context(self):
        model = DefaultModel()
        evaluation_context = ModelEvaluationContext(model=model, context="0d")
        self.assertEqual(evaluation_context.context, DateContext(date=date.today()))
        out = evaluation_context()
        self.assertEqual(out, GenericResult(value=date.today()))


class TestFlowDecorator(TestCase):
    def test_get_options(self):
        model = DefaultModel()
        self.assertEqual(model.__call__.get_options(model), FlowOptions(log_level=logging.DEBUG))
        options = FlowOptions(log_level=logging.INFO)
        with FlowOptionsOverride(options=options):
            self.assertEqual(model.__call__.get_options(model), FlowOptions(log_level=logging.INFO))
        self.assertEqual(model.__call__.get_options(model), FlowOptions(log_level=logging.DEBUG))

        # Note that *ALL* models of this type will now have that log level
        with FlowOptionsOverride(options=options, model_types=[DefaultModel]):
            model2 = DefaultModel()
            self.assertEqual(model2.__call__.get_options(model2), FlowOptions(log_level=logging.INFO))

    def test_evaluator(self):
        # Test that you can get the (current) evaluator back from the decorated function
        model = MyModel()
        evaluator = model.__call__.get_evaluator(model)
        self.assertIsInstance(evaluator, EvaluatorBase)

    def test_evaluation_context(self):
        model = DefaultModel()
        context = DateContext(date=date.today())
        evaluation_context = model.__call__.get_evaluation_context(model, context)
        self.assertIsInstance(evaluation_context, ModelEvaluationContext)
        self.assertIsInstance(evaluation_context.model, EvaluatorBase)
        self.assertIsInstance(evaluation_context.context, ModelEvaluationContext)
        self.assertEqual(evaluation_context.context.model, model)
        self.assertEqual(evaluation_context.context.context, context)

    def test_evaluation_context_options(self):
        model = DefaultModel()
        context = DateContext(date=date.today())
        options = FlowOptions(log_level=0)
        with FlowOptionsOverride(options=options):
            self.assertEqual(model.__call__.get_options(model), options)
            evaluation_context = model.__call__.get_evaluation_context(model, context)
            self.assertEqual(evaluation_context.model.log_level, 0)

    def test_new_evaluator(self):
        """Test that we can use the decorator to make an evaluation with a new evaluator."""
        model = MyModel()
        new_evaluator = MemoryCacheEvaluator()
        wrapped_call = Flow.call(evaluator=new_evaluator)(MyModel.__call__)
        out_evaluator = wrapped_call.get_evaluator(model)
        self.assertEqual(out_evaluator, new_evaluator)

        context = DateContext(date=date.today())
        self.assertEqual(wrapped_call(model, context), model(context))

        # Now test it on foo
        wrapped_foo = Flow.call(evaluator=new_evaluator)(MyModel.foo)
        self.assertEqual(wrapped_foo(model, context), model.foo(context))

    def test_coercion(self):
        model = MyModel()
        self.assertEqual(model("2022-01-01"), GenericResult(value=date(2022, 1, 1)))
        self.assertEqual(model.foo("2022-01-01"), GenericResult(value=date(2022, 1, 1)))
        self.assertEqual(model.bar("2022-01-01"), GenericResult(value=date(2022, 1, 1)))
        self.assertEqual(model.baz("2022-01-01"), date(2022, 1, 1))

    def test_default(self):
        model = DefaultModel()
        self.assertEqual(model(), GenericResult(value=date.today()))

        model = MyModel()
        self.assertRaisesRegex(
            TypeError,
            r"__call__\(\) missing 1 required positional argument: 'context'",
            model,
        )

    def test_logging(self):
        model = MyModel()
        with self.assertLogs(level=logging.DEBUG) as captured:
            model("2022-01-01")
            model.foo("2022-01-02")
            model.bar("2022-01-02")
        # 3 calls, (start+end), 2 messages for each
        self.assertEqual(len(captured.records), 9)
        for i in range(0, 3):
            self.assertEqual(captured.records[i].levelno, logging.WARNING)
        for i in range(3, 9):
            self.assertEqual(captured.records[i].levelno, logging.DEBUG)

        with FlowOptionsOverride(options=FlowOptions(log_level=logging.INFO)):
            with self.assertLogs(level=logging.DEBUG) as captured:
                model("2022-01-01")
                model.foo("2022-01-02")
                model.bar("2022-01-02")
        # 3 calls, (start+end), 2 messages for each
        self.assertEqual(len(captured.records), 9)
        for i in range(0, 3):
            self.assertEqual(captured.records[i].levelno, logging.WARNING)
        for i in range(3, 9):
            self.assertEqual(captured.records[i].levelno, logging.INFO)

    def test_validate(self):
        model = SomeClass()
        self.assertRaises(TypeError, model.foo)

        model = BadContextModel()
        self.assertRaises(TypeError, model, DateContext(date=date(2020, 1, 1)))

        model = BadResultModel
        self.assertRaises(TypeError, model, DateContext(date=date(2020, 1, 1)))


class TestMetaData(TestCase):
    def test_default(self):
        model = DefaultModel(meta={"options": {"log_level": logging.INFO}})
        self.assertEqual(model.__call__.get_options(model), FlowOptions(log_level=logging.INFO))

    def test_decorator_precedence(self):
        # meta should take precedence over the decorator
        model = MyModel()
        get_options = MyModel.__call__.get_options
        self.assertEqual(get_options(model), FlowOptions(log_level=logging.WARNING))
        model = MyModel(meta={"options": {"log_level": logging.INFO}})
        self.assertEqual(get_options(model), FlowOptions(log_level=logging.INFO))

    def test_override_precedence(self):
        # meta should take precedence over global overrides, but not model overrides
        model = DefaultModel(meta={"options": {"log_level": logging.INFO}})
        get_options = DefaultModel.__call__.get_options
        options = get_options(model)
        new_options = FlowOptions(log_level=logging.WARNING)
        with FlowOptionsOverride(options=new_options):
            self.assertEqual(get_options(model), options)

        with FlowOptionsOverride(options=new_options, models=(model,)):
            self.assertEqual(get_options(model), new_options)
        self.assertEqual(get_options(model), options)


class TestFlowOptionsOverride(TestCase):
    def test_global(self):
        model = DefaultModel()
        get_options = DefaultModel.__call__.get_options
        options = get_options(model)
        new_options = FlowOptions(log_level=logging.INFO)
        with FlowOptionsOverride(options=new_options):
            self.assertEqual(get_options(model), new_options)
        self.assertEqual(get_options(model), options)

    def test_model_specific(self):
        model = DefaultModel()
        model2 = DefaultModel()
        model3 = DefaultModel()
        get_options = DefaultModel.__call__.get_options
        options = get_options(model)
        new_options = FlowOptions(log_level=logging.INFO)
        with FlowOptionsOverride(options=new_options, models=(model, model3)):
            self.assertEqual(get_options(model), new_options)
            # Does not pick up when model doesn't match
            self.assertEqual(get_options(model2), options)
            # Multiple models work
            self.assertEqual(get_options(model3), new_options)
        self.assertEqual(get_options(model), options)

    def test_type_specific(self):
        model = DefaultModel()
        get_options = DefaultModel.__call__.get_options
        options = get_options(model)
        new_options = FlowOptions(log_level=logging.INFO)
        with FlowOptionsOverride(options=new_options, model_types=(DefaultModel,)):
            self.assertEqual(get_options(model), new_options)
        self.assertEqual(get_options(model), options)

        # Does not pick up when type doesn't match
        with FlowOptionsOverride(options=new_options, model_types=(MyModel,)):
            self.assertEqual(get_options(model), options)

    def test_nested(self):
        model = DefaultModel()
        model2 = DefaultModel()
        model3 = DefaultModel()
        get_options = DefaultModel.__call__.get_options
        options = get_options(model)
        new_options = FlowOptions(log_level=logging.INFO)
        newer_options = FlowOptions(log_level=logging.WARNING)
        with FlowOptionsOverride(options=new_options):
            with FlowOptionsOverride(options=newer_options, models=(model, model3)):
                self.assertEqual(get_options(model), newer_options)
                self.assertEqual(get_options(model2), new_options)
                self.assertEqual(get_options(model3), newer_options)
            self.assertEqual(get_options(model), new_options)
            self.assertEqual(get_options(model3), new_options)
        self.assertEqual(get_options(model), options)
        self.assertEqual(get_options(model2), options)
        self.assertEqual(get_options(model3), options)

    def test_nested_priority(self):
        """Model and type level overrides take priority over global."""
        model = DefaultModel()
        model2 = DefaultModel()
        model3 = DefaultModel()
        get_options = DefaultModel.__call__.get_options
        options = get_options(model)
        new_options = FlowOptions(log_level=logging.INFO)
        newer_options = FlowOptions(log_level=logging.WARNING)
        with FlowOptionsOverride(options=new_options, models=(model, model3)):
            with FlowOptionsOverride(options=newer_options):
                self.assertEqual(get_options(model), new_options)
                self.assertEqual(get_options(model2), newer_options)
                self.assertEqual(get_options(model3), new_options)
            self.assertEqual(get_options(model), new_options)
            self.assertEqual(get_options(model2), options)
            self.assertEqual(get_options(model3), new_options)

    def test_existing_settings(self):
        """Test that if the decorator sets things, it is handled properly"""
        model = MyModel()
        get_options = MyModel.__call__.get_options
        options = get_options(model)
        self.assertEqual(options.log_level, logging.WARNING)
        new_options = FlowOptions(log_level=logging.INFO)
        new_baz_options = FlowOptions(validate_result=False, log_level=logging.INFO)
        with FlowOptionsOverride(options=new_options):
            # Does not override the default setting, which was set on the decorator, with the global one
            self.assertEqual(get_options(model), options)
            # For baz, check that the override does apply, and is merged with another existing param
            self.assertEqual(MyModel.baz.get_options(model), new_baz_options)

        with FlowOptionsOverride(options=new_options, model_types=(MyModel,)):
            # Does override it, as it was set at type level
            self.assertEqual(get_options(model), new_options)
            self.assertEqual(MyModel.bar.get_options(model), new_options)
            self.assertEqual(MyModel.baz.get_options(model), new_baz_options)

        with FlowOptionsOverride(options=new_options, models=(model,)):
            # Does override it, as it was set at type level
            self.assertEqual(get_options(model), new_options)
            self.assertEqual(MyModel.bar.get_options(model), new_options)
            self.assertEqual(MyModel.baz.get_options(model), new_baz_options)
