import logging
import threading
from datetime import date, datetime
from unittest import TestCase

import cloudpickle

from ccflow import CallableModel, DateContext, EvaluatorBase, Flow, FlowOptions, FlowOptionsOverride, GenericResult, ModelEvaluationContext


def _local_today() -> date:
    return datetime.now().astimezone().date()


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


class DummyEvaluator(EvaluatorBase):
    return_value: GenericResult = GenericResult(value=date(2020, 1, 1))

    def __call__(self, context: ModelEvaluationContext):
        return self.return_value


class TestModelEvaluationContext(TestCase):
    def test_model_evaluation_context(self):
        model = DefaultModel()
        evaluation_context = ModelEvaluationContext(model=model, context="0d")
        self.assertEqual(evaluation_context.context, DateContext(date=_local_today()))
        out = evaluation_context()
        self.assertEqual(out, GenericResult(value=_local_today()))


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
        context = DateContext(date=_local_today())
        evaluation_context = model.__call__.get_evaluation_context(model, context)
        self.assertIsInstance(evaluation_context, ModelEvaluationContext)
        self.assertIsInstance(evaluation_context.model, EvaluatorBase)
        self.assertIsInstance(evaluation_context.context, ModelEvaluationContext)
        self.assertEqual(evaluation_context.context.model, model)
        self.assertEqual(evaluation_context.context.context, context)

    def test_evaluation_context_options(self):
        model = DefaultModel()
        context = DateContext(date=_local_today())
        options = FlowOptions(log_level=0)

        evaluation_context = model.__call__.get_evaluation_context(model, context, _options=options)
        self.assertEqual(evaluation_context.model.log_level, 0)

        evaluation_context = model.__call__.get_evaluation_context(model, context, _options={"log_level": 0})
        self.assertEqual(evaluation_context.model.log_level, 0)

        with FlowOptionsOverride(options=options):
            self.assertEqual(model.__call__.get_options(model), options)
            evaluation_context = model.__call__.get_evaluation_context(model, context)
            self.assertEqual(evaluation_context.model.log_level, 0)

            # Make sure passing _options takes precedence over the FlowOptionsOverride context
            evaluation_context = model.__call__.get_evaluation_context(model, context, _options={"log_level": 1})
            self.assertEqual(evaluation_context.model.log_level, 1)

    def test_new_evaluator(self):
        """Test that we can call the model easily with a new evaluator."""
        model = MyModel()
        new_evaluator = DummyEvaluator()
        context = DateContext(date=_local_today())
        self.assertEqual(model(context), GenericResult(value=context.date))
        self.assertEqual(model(context, _options={"evaluator": new_evaluator}), new_evaluator.return_value)

        # Now test it on foo
        self.assertEqual(model(context), GenericResult(value=context.date))
        self.assertEqual(model.foo(context, _options={"evaluator": new_evaluator}), new_evaluator.return_value)

    def test_coercion(self):
        model = MyModel()
        self.assertEqual(model("2022-01-01"), GenericResult(value=date(2022, 1, 1)))
        self.assertEqual(model.foo("2022-01-01"), GenericResult(value=date(2022, 1, 1)))
        self.assertEqual(model.bar("2022-01-01"), GenericResult(value=date(2022, 1, 1)))
        self.assertEqual(model.baz("2022-01-01"), date(2022, 1, 1))

    def test_default(self):
        model = DefaultModel()
        self.assertEqual(model(), GenericResult(value=_local_today()))

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
        for i in range(3):
            self.assertEqual(captured.records[i].levelno, logging.WARNING)
        for i in range(3, 9):
            self.assertEqual(captured.records[i].levelno, logging.DEBUG)

        with FlowOptionsOverride(options=FlowOptions(log_level=logging.INFO)), self.assertLogs(level=logging.DEBUG) as captured:
            model("2022-01-01")
            model.foo("2022-01-02")
            model.bar("2022-01-02")
        # 3 calls, (start+end), 2 messages for each
        self.assertEqual(len(captured.records), 9)
        for i in range(3):
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
    def setUp(self):
        # A leaked override would silently change the options seen by unrelated tests.
        FlowOptionsOverride._OPEN_OVERRIDES.clear()

    def tearDown(self):
        FlowOptionsOverride._OPEN_OVERRIDES.clear()

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

    def test_reentrant_instancecheck(self):
        """get_options must tolerate a user __instancecheck__ that opens another override.

        get_options calls isinstance(model, override.model_types) from inside its own
        iteration over _OPEN_OVERRIDES, so a user-defined __instancecheck__ runs
        mid-iteration and may legitimately enter a further override. Iterating the
        dict directly raised RuntimeError: dictionary changed size during iteration.
        """
        opened = []

        class ReentrantMeta(type(CallableModel)):
            def __instancecheck__(cls, instance):
                if not opened:
                    nested = FlowOptionsOverride(options=FlowOptions(log_level=logging.ERROR))
                    nested.__enter__()
                    opened.append(nested)
                return super().__instancecheck__(instance)

        class ReentrantModel(DefaultModel, metaclass=ReentrantMeta):
            pass

        model = DefaultModel()
        get_options = DefaultModel.__call__.get_options
        new_options = FlowOptions(log_level=logging.INFO)
        try:
            with FlowOptionsOverride(options=new_options, model_types=(ReentrantModel,)):
                options = get_options(model)
        finally:
            for nested in opened:
                nested.__exit__(None, None, None)

        self.assertTrue(opened, "the __instancecheck__ hook never ran, so the test is not exercising the bug")
        self.assertIsInstance(options, FlowOptions)

    def test_concurrent_override_entry(self):
        """Entering an override on another thread must not break an in-flight get_options.

        The interleaving is pinned with a barrier rather than left to chance: the reader
        is parked inside its iteration over _OPEN_OVERRIDES until the writer has entered
        a new override.
        """
        barrier = threading.Barrier(2, timeout=30)

        class BlockingMeta(type(CallableModel)):
            def __instancecheck__(cls, instance):
                barrier.wait()  # let the writer mutate the dict
                barrier.wait()  # resume only once it has
                return super().__instancecheck__(instance)

        class BlockingModel(DefaultModel, metaclass=BlockingMeta):
            pass

        model = DefaultModel()
        get_options = DefaultModel.__call__.get_options
        reader_error = []
        writer_override = FlowOptionsOverride(options=FlowOptions(log_level=logging.ERROR))

        def reader():
            try:
                get_options(model)
            except BaseException as exc:  # noqa: BLE001 - surfaced on the main thread below
                reader_error.append(exc)

        def writer():
            try:
                barrier.wait()
                writer_override.__enter__()
            finally:
                barrier.wait()

        with FlowOptionsOverride(options=FlowOptions(log_level=logging.INFO), model_types=(BlockingModel,)):
            threads = [threading.Thread(target=reader), threading.Thread(target=writer)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=30)
            if id(writer_override) in FlowOptionsOverride._OPEN_OVERRIDES:
                writer_override.__exit__(None, None, None)

        self.assertEqual(reader_error, [], "get_options raised while another thread entered an override")

    def test_lock_is_a_module_global(self):
        """The lock guarding _OPEN_OVERRIDES must not live on the class.

        cloudpickle serialises a class by value when it cannot reference it by module
        path, copying its __dict__ wholesale. A lock held as a ClassVar would then be
        dragged into the payload and fail with "cannot pickle '_thread.RLock' object".
        Keeping it as a module global makes that impossible: there is no class attribute
        for a subclass to inherit or redefine.
        """
        import ccflow.callable as callable_module

        self.assertIsInstance(callable_module._OPEN_OVERRIDES_LOCK, type(threading.RLock()))
        for klass in FlowOptionsOverride.__mro__:
            self.assertNotIn("_OPEN_OVERRIDES_LOCK", vars(klass))

    def test_cloudpickle_instance(self):
        """An override instance must survive a cloudpickle round trip."""
        override = FlowOptionsOverride(options=FlowOptions(log_level=logging.INFO))
        restored = cloudpickle.loads(cloudpickle.dumps(override))
        self.assertEqual(restored.options.log_level, override.options.log_level)

    def test_cloudpickle_class(self):
        """The FlowOptionsOverride class itself must survive a round trip, by reference."""
        self.assertIs(cloudpickle.loads(cloudpickle.dumps(FlowOptionsOverride)), FlowOptionsOverride)

    def test_cloudpickle_subclass_by_value(self):
        """A subclass that cloudpickle must serialise by value must still round trip.

        A class defined inside a function cannot be referenced by module path, so
        cloudpickle copies its __dict__ wholesale. This is the case that fails if the
        lock is stored on the class, and it is the one that matters under ray.
        """

        class LocalOverride(FlowOptionsOverride):
            pass

        self.assertIn("<locals>", LocalOverride.__qualname__)
        restored = cloudpickle.loads(cloudpickle.dumps(LocalOverride))
        self.assertEqual(restored.__name__, "LocalOverride")
        self.assertTrue(issubclass(restored, FlowOptionsOverride))

    def test_cloudpickle_closure_with_open_override(self):
        """A closure capturing an entered override must survive a round trip.

        This mirrors how work is shipped to a ray worker: a callable is captured while
        an override is open on the submitting side.
        """
        with FlowOptionsOverride(options=FlowOptions(log_level=logging.INFO)) as override:

            def use_override():
                return override.options.log_level

            expected = override.options.log_level
            restored = cloudpickle.loads(cloudpickle.dumps(use_override))

        self.assertEqual(restored(), expected)
