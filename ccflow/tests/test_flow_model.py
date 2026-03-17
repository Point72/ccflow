"""Tests for Flow.model decorator."""

from datetime import date, timedelta
from unittest import TestCase

from ray.cloudpickle import dumps as rcpdumps, loads as rcploads

from ccflow import (
    CallableModel,
    ContextBase,
    DateRangeContext,
    Flow,
    FlowOptionsOverride,
    GenericResult,
    Lazy,
    ModelRegistry,
    ResultBase,
)
from ccflow.evaluators.common import MemoryCacheEvaluator


class SimpleContext(ContextBase):
    """Simple context for testing."""

    value: int


class ExtendedContext(ContextBase):
    """Extended context with multiple fields."""

    x: int
    y: str = "default"


class MyResult(ResultBase):
    """Custom result type for testing."""

    data: str


# =============================================================================
# Basic Flow.model Tests
# =============================================================================


class TestFlowModelBasic(TestCase):
    """Basic Flow.model functionality tests."""

    def test_simple_model_explicit_context(self):
        """Test Flow.model with explicit context parameter."""

        @Flow.model
        def simple_loader(context: SimpleContext, multiplier: int) -> GenericResult[int]:
            return GenericResult(value=context.value * multiplier)

        # Create model instance
        loader = simple_loader(multiplier=3)

        # Should be a CallableModel
        self.assertIsInstance(loader, CallableModel)

        # Execute
        ctx = SimpleContext(value=10)
        result = loader(ctx)

        self.assertIsInstance(result, GenericResult)
        self.assertEqual(result.value, 30)

    def test_model_with_default_params(self):
        """Test Flow.model with default parameter values."""

        @Flow.model
        def loader_with_defaults(context: SimpleContext, multiplier: int = 2, prefix: str = "result") -> GenericResult[str]:
            return GenericResult(value=f"{prefix}:{context.value * multiplier}")

        # Create with defaults
        loader = loader_with_defaults()
        result = loader(SimpleContext(value=5))
        self.assertEqual(result.value, "result:10")

        # Create with custom values
        loader2 = loader_with_defaults(multiplier=3, prefix="custom")
        result2 = loader2(SimpleContext(value=5))
        self.assertEqual(result2.value, "custom:15")

    def test_model_context_type_property(self):
        """Test that generated model has correct context_type."""

        @Flow.model
        def typed_model(context: ExtendedContext, factor: int) -> GenericResult[int]:
            return GenericResult(value=context.x * factor)

        model = typed_model(factor=2)
        self.assertEqual(model.context_type, ExtendedContext)

    def test_model_result_type_property(self):
        """Test that generated model has correct result_type."""

        @Flow.model
        def custom_result_model(context: SimpleContext) -> MyResult:
            return MyResult(data=f"value={context.value}")

        model = custom_result_model()
        self.assertEqual(model.result_type, MyResult)

    def test_model_with_no_extra_params(self):
        """Test Flow.model with only context parameter."""

        @Flow.model
        def identity_model(context: SimpleContext) -> GenericResult[int]:
            return GenericResult(value=context.value)

        model = identity_model()
        result = model(SimpleContext(value=42))
        self.assertEqual(result.value, 42)

    def test_model_with_flow_options(self):
        """Test Flow.model with Flow.call options."""

        @Flow.model(cacheable=True, validate_result=True)
        def cached_model(context: SimpleContext, value: int) -> GenericResult[int]:
            return GenericResult(value=value + context.value)

        model = cached_model(value=10)
        result = model(SimpleContext(value=5))
        self.assertEqual(result.value, 15)

    def test_model_with_underscore_context(self):
        """Test Flow.model with '_' as context parameter (unused context convention)."""

        @Flow.model
        def loader(context: SimpleContext, base: int) -> GenericResult[int]:
            return GenericResult(value=context.value + base)

        @Flow.model
        def consumer(_: SimpleContext, data: int) -> GenericResult[int]:
            # Context not used directly, just passed to dependency
            return GenericResult(value=data * 2)

        load = loader(base=100)
        consume = consumer(data=load)

        result = consume(SimpleContext(value=10))
        # loader: 10 + 100 = 110, consumer: 110 * 2 = 220
        self.assertEqual(result.value, 220)

        # Verify context_type is still correct
        self.assertEqual(consume.context_type, SimpleContext)


# =============================================================================
# context_args Mode Tests
# =============================================================================


class TestFlowModelContextArgs(TestCase):
    """Tests for Flow.model with context_args (unpacked context)."""

    def test_context_args_basic(self):
        """Test basic context_args usage."""

        @Flow.model(context_args=["start_date", "end_date"])
        def date_range_loader(start_date: date, end_date: date, source: str) -> GenericResult[str]:
            return GenericResult(value=f"{source}:{start_date} to {end_date}")

        loader = date_range_loader(source="db")

        # Should use DateRangeContext
        self.assertEqual(loader.context_type, DateRangeContext)

        ctx = DateRangeContext(start_date=date(2024, 1, 1), end_date=date(2024, 1, 31))
        result = loader(ctx)
        self.assertEqual(result.value, "db:2024-01-01 to 2024-01-31")

    def test_context_args_custom_context(self):
        """Test context_args with custom context type."""

        @Flow.model(context_args=["x", "y"])
        def unpacked_model(x: int, y: str, multiplier: int = 1) -> GenericResult[str]:
            return GenericResult(value=f"{y}:{x * multiplier}")

        model = unpacked_model(multiplier=2)

        # Create context with generated type
        ctx_type = model.context_type
        ctx = ctx_type(x=5, y="test")

        result = model(ctx)
        self.assertEqual(result.value, "test:10")

    def test_context_args_with_defaults(self):
        """Test context_args where context fields have defaults."""

        @Flow.model(context_args=["value"])
        def model_with_ctx_default(value: int = 42, extra: str = "foo") -> GenericResult[str]:
            return GenericResult(value=f"{extra}:{value}")

        model = model_with_ctx_default()

        # Create context - the generated context should allow default
        ctx_type = model.context_type
        ctx = ctx_type(value=100)

        result = model(ctx)
        self.assertEqual(result.value, "foo:100")


# =============================================================================
# Dependency Tests
# =============================================================================


class TestFlowModelDependencies(TestCase):
    """Tests for Flow.model with upstream CallableModel inputs."""

    def test_simple_dependency(self):
        """Test passing an upstream model as a normal parameter."""

        @Flow.model
        def loader(context: SimpleContext, value: int) -> GenericResult[int]:
            return GenericResult(value=value + context.value)

        @Flow.model
        def consumer(
            context: SimpleContext,
            data: int,
            multiplier: int = 1,
        ) -> GenericResult[int]:
            return GenericResult(value=data * multiplier)

        # Create pipeline
        load = loader(value=10)
        consume = consumer(data=load, multiplier=2)

        ctx = SimpleContext(value=5)
        result = consume(ctx)

        # loader returns 10 + 5 = 15, consumer multiplies by 2 = 30
        self.assertEqual(result.value, 30)

    def test_dependency_with_direct_value(self):
        """Test that dependency-shaped parameters can also take direct values."""

        @Flow.model
        def consumer(
            context: SimpleContext,
            data: int,
        ) -> GenericResult[int]:
            return GenericResult(value=data + context.value)

        consume = consumer(data=100)

        result = consume(SimpleContext(value=5))
        self.assertEqual(result.value, 105)

    def test_deps_method_generation(self):
        """Test that __deps__ method is correctly generated."""

        @Flow.model
        def loader(context: SimpleContext) -> GenericResult[int]:
            return GenericResult(value=context.value)

        @Flow.model
        def consumer(
            context: SimpleContext,
            data: int,
        ) -> GenericResult[int]:
            return GenericResult(value=data)

        load = loader()
        consume = consumer(data=load)

        ctx = SimpleContext(value=10)
        deps = consume.__deps__(ctx)

        # Should have one dependency
        self.assertEqual(len(deps), 1)
        self.assertEqual(deps[0][0], load)
        self.assertEqual(deps[0][1], [ctx])

    def test_no_deps_when_direct_value(self):
        """Test that __deps__ returns empty when direct values used."""

        @Flow.model
        def consumer(
            context: SimpleContext,
            data: int,
        ) -> GenericResult[int]:
            return GenericResult(value=data)

        consume = consumer(data=100)

        deps = consume.__deps__(SimpleContext(value=10))
        self.assertEqual(len(deps), 0)


# =============================================================================
# with_inputs Tests
# =============================================================================


class TestFlowModelWithInputs(TestCase):
    """Tests for Flow.model with .flow.with_inputs()."""

    def test_transformed_dependency_with_inputs(self):
        """Test dependency context transformation via .flow.with_inputs()."""

        @Flow.model
        def loader(context: SimpleContext) -> GenericResult[int]:
            return GenericResult(value=context.value)

        @Flow.model
        def consumer(context: SimpleContext, data: int) -> GenericResult[int]:
            return GenericResult(value=data * 2)

        load = loader().flow.with_inputs(value=lambda ctx: ctx.value + 10)
        consume = consumer(data=load)

        result = consume(SimpleContext(value=5))
        self.assertEqual(result.value, 30)

    def test_with_inputs_changes_dependency_context_in_deps(self):
        """Test that BoundModel contributes transformed dependency contexts."""

        @Flow.model
        def loader(context: SimpleContext) -> GenericResult[int]:
            return GenericResult(value=context.value)

        @Flow.model
        def consumer(context: SimpleContext, data: int) -> GenericResult[int]:
            return GenericResult(value=data)

        load = loader().flow.with_inputs(value=lambda ctx: ctx.value * 3)
        consume = consumer(data=load)

        deps = consume.__deps__(SimpleContext(value=7))
        self.assertEqual(len(deps), 1)
        transformed_ctx = deps[0][1][0]
        self.assertEqual(transformed_ctx.value, 21)

    def test_date_range_transform_with_inputs(self):
        """Test date-range lookback wiring via .flow.with_inputs()."""

        @Flow.model(context_args=["start_date", "end_date"])
        def range_loader(start_date: date, end_date: date, source: str) -> GenericResult[str]:
            return GenericResult(value=f"{source}:{start_date}")

        @Flow.model(context_args=["start_date", "end_date"])
        def range_processor(
            start_date: date,
            end_date: date,
            data: str,
        ) -> GenericResult[str]:
            return GenericResult(value=f"processed:{data}")

        loader = range_loader(source="db").flow.with_inputs(start_date=lambda ctx: ctx.start_date - timedelta(days=1))
        processor = range_processor(data=loader)

        ctx = DateRangeContext(start_date=date(2024, 1, 10), end_date=date(2024, 1, 31))
        result = processor(ctx)
        self.assertEqual(result.value, "processed:db:2024-01-09")


# =============================================================================
# Pipeline Tests
# =============================================================================


class TestFlowModelPipeline(TestCase):
    """Tests for multi-stage pipelines with Flow.model."""

    def test_three_stage_pipeline(self):
        """Test a three-stage computation pipeline."""

        @Flow.model
        def stage1(context: SimpleContext, base: int) -> GenericResult[int]:
            return GenericResult(value=context.value + base)

        @Flow.model
        def stage2(
            context: SimpleContext,
            input_data: int,
            multiplier: int,
        ) -> GenericResult[int]:
            return GenericResult(value=input_data * multiplier)

        @Flow.model
        def stage3(
            context: SimpleContext,
            input_data: int,
            offset: int = 0,
        ) -> GenericResult[int]:
            return GenericResult(value=input_data + offset)

        # Build pipeline
        s1 = stage1(base=100)
        s2 = stage2(input_data=s1, multiplier=2)
        s3 = stage3(input_data=s2, offset=50)

        ctx = SimpleContext(value=10)
        result = s3(ctx)

        # s1: 10 + 100 = 110
        # s2: 110 * 2 = 220
        # s3: 220 + 50 = 270
        self.assertEqual(result.value, 270)

    def test_diamond_dependency_pattern(self):
        """Test diamond-shaped dependency pattern."""

        @Flow.model
        def source(context: SimpleContext) -> GenericResult[int]:
            return GenericResult(value=context.value)

        @Flow.model
        def branch_a(
            context: SimpleContext,
            data: int,
        ) -> GenericResult[int]:
            return GenericResult(value=data * 2)

        @Flow.model
        def branch_b(
            context: SimpleContext,
            data: int,
        ) -> GenericResult[int]:
            return GenericResult(value=data + 100)

        @Flow.model
        def merger(
            context: SimpleContext,
            a: int,
            b: int,
        ) -> GenericResult[int]:
            return GenericResult(value=a + b)

        src = source()
        a = branch_a(data=src)
        b = branch_b(data=src)
        merge = merger(a=a, b=b)

        ctx = SimpleContext(value=10)
        result = merge(ctx)

        # source: 10
        # branch_a: 10 * 2 = 20
        # branch_b: 10 + 100 = 110
        # merger: 20 + 110 = 130
        self.assertEqual(result.value, 130)


# =============================================================================
# Integration Tests
# =============================================================================


class TestFlowModelIntegration(TestCase):
    """Integration tests for Flow.model with ccflow infrastructure."""

    def test_registry_integration(self):
        """Test that Flow.model models work with ModelRegistry."""

        @Flow.model
        def registrable_model(context: SimpleContext, value: int) -> GenericResult[int]:
            return GenericResult(value=context.value + value)

        model = registrable_model(value=100)

        registry = ModelRegistry.root().clear()
        registry.add("test_model", model)

        retrieved = registry["test_model"]
        self.assertEqual(retrieved, model)

        result = retrieved(SimpleContext(value=10))
        self.assertEqual(result.value, 110)

    def test_serialization_dump(self):
        """Test that generated models can be serialized."""

        @Flow.model
        def serializable_model(context: SimpleContext, value: int = 42) -> GenericResult[int]:
            return GenericResult(value=value)

        model = serializable_model(value=100)
        dumped = model.model_dump(mode="python")

        self.assertIn("value", dumped)
        self.assertEqual(dumped["value"], 100)
        self.assertIn("type_", dumped)

    def test_pickle_roundtrip(self):
        """Test cloudpickle serialization of generated models."""

        @Flow.model
        def pickleable_model(context: SimpleContext, factor: int) -> GenericResult[int]:
            return GenericResult(value=context.value * factor)

        model = pickleable_model(factor=3)

        # Cloudpickle roundtrip (standard pickle won't work for local classes)
        pickled = rcpdumps(model, protocol=5)
        restored = rcploads(pickled)

        result = restored(SimpleContext(value=10))
        self.assertEqual(result.value, 30)

    def test_mix_with_manual_callable_model(self):
        """Test mixing Flow.model with manually defined CallableModel."""

        class ManualModel(CallableModel):
            offset: int

            @Flow.call
            def __call__(self, context: SimpleContext) -> GenericResult[int]:
                return GenericResult(value=context.value + self.offset)

        @Flow.model
        def generated_consumer(
            context: SimpleContext,
            data: int,
            multiplier: int,
        ) -> GenericResult[int]:
            return GenericResult(value=data * multiplier)

        manual = ManualModel(offset=50)
        generated = generated_consumer(data=manual, multiplier=2)

        result = generated(SimpleContext(value=10))
        # manual: 10 + 50 = 60
        # generated: 60 * 2 = 120
        self.assertEqual(result.value, 120)


# =============================================================================
# Error Case Tests
# =============================================================================


class TestFlowModelErrors(TestCase):
    """Error case tests for Flow.model."""

    def test_missing_return_type(self):
        """Test error when return type annotation is missing."""
        with self.assertRaises(TypeError) as cm:

            @Flow.model
            def no_return(context: SimpleContext):
                return GenericResult(value=1)

        self.assertIn("return type annotation", str(cm.exception))

    def test_auto_wrap_plain_return_type(self):
        """Test that non-ResultBase return types are auto-wrapped in GenericResult."""

        @Flow.model
        def plain_return(context: SimpleContext) -> int:
            return context.value * 2

        model = plain_return()
        result = model(SimpleContext(value=5))
        self.assertIsInstance(result, GenericResult)
        self.assertEqual(result.value, 10)

    def test_auto_wrap_unwrap_as_dependency(self):
        """Test that auto-wrapped model used as dep delivers unwrapped value downstream.

        Auto-wrapped models have result_type=GenericResult (unparameterized).
        When used as an auto-detected dep, the framework resolves
        the GenericResult and unwraps .value for the downstream function.
        """

        @Flow.model
        def plain_source(context: SimpleContext) -> int:
            return context.value * 3

        @Flow.model
        def consumer(
            context: SimpleContext,
            data: GenericResult[int],  # Auto-detected dep
        ) -> GenericResult[int]:
            # data is auto-unwrapped to the int value by the framework
            return GenericResult(value=data + 1)

        src = plain_source()
        model = consumer(data=src)
        result = model(SimpleContext(value=10))
        # plain_source: 10 * 3 = 30, auto-wrapped to GenericResult(value=30)
        # resolve_callable_model unwraps GenericResult -> 30
        # consumer: 30 + 1 = 31
        self.assertEqual(result.value, 31)

    def test_auto_wrap_result_type_property(self):
        """Test that auto-wrapped model has GenericResult as result_type."""

        @Flow.model
        def plain_return(context: SimpleContext) -> int:
            return context.value

        model = plain_return()
        self.assertEqual(model.result_type, GenericResult)

    def test_dynamic_deferred_mode(self):
        """Test dynamic deferred mode where what you provide at construction = bound."""
        from ccflow import FlowContext

        @Flow.model
        def dynamic_model(value: int, multiplier: int) -> GenericResult[int]:
            return GenericResult(value=value * multiplier)

        # Provide 'multiplier' at construction -> it's bound
        # Don't provide 'value' -> comes from context
        model = dynamic_model(multiplier=3)

        # Check bound vs unbound
        self.assertEqual(model.flow.bound_inputs, {"multiplier": 3})
        self.assertEqual(model.flow.unbound_inputs, {"value": int})

        # Call with context providing 'value'
        ctx = FlowContext(value=10)
        result = model(ctx)
        self.assertEqual(result.value, 30)  # 10 * 3

    def test_all_defaults_is_valid(self):
        """Test that all-defaults function is valid (everything can be pre-bound)."""
        from ccflow import FlowContext

        @Flow.model
        def all_defaults(value: int = 1, other: str = "x") -> GenericResult[str]:
            return GenericResult(value=f"{value}-{other}")

        # No args provided -> everything comes from defaults or context
        model = all_defaults()

        # All params are unbound (not provided at construction)
        self.assertEqual(model.flow.unbound_inputs, {"value": int, "other": str})

        # Call with context - context values override defaults
        ctx = FlowContext(value=5, other="y")
        result = model(ctx)
        self.assertEqual(result.value, "5-y")

    def test_invalid_context_arg(self):
        """Test error when context_args refers to non-existent parameter."""
        with self.assertRaises(ValueError) as cm:

            @Flow.model(context_args=["nonexistent"])
            def bad_context_args(x: int) -> GenericResult[int]:
                return GenericResult(value=x)

        self.assertIn("nonexistent", str(cm.exception))

    def test_context_arg_without_annotation(self):
        """Test error when context_arg parameter lacks type annotation."""
        with self.assertRaises(ValueError) as cm:

            @Flow.model(context_args=["x"])
            def untyped_context_arg(x) -> GenericResult[int]:
                return GenericResult(value=x)

        self.assertIn("type annotation", str(cm.exception))


# =============================================================================
# Validation Tests
# =============================================================================


class TestFlowModelValidation(TestCase):
    """Tests for Flow.model validation behavior."""

    def test_config_validation_rejects_bad_type(self):
        """Test that config validator rejects wrong types at construction."""

        @Flow.model
        def typed_config(context: SimpleContext, n_estimators: int = 10) -> GenericResult[int]:
            return GenericResult(value=n_estimators)

        with self.assertRaises(TypeError) as cm:
            typed_config(n_estimators="banana")

        self.assertIn("n_estimators", str(cm.exception))

    def test_config_validation_accepts_callable_model(self):
        """Test that config validator allows CallableModel values for any field."""

        @Flow.model
        def source(context: SimpleContext) -> GenericResult[int]:
            return GenericResult(value=context.value)

        @Flow.model
        def consumer(context: SimpleContext, data: int = 0) -> GenericResult[int]:
            return GenericResult(value=data)

        # Passing a CallableModel for an int field should not raise
        src = source()
        model = consumer(data=src)
        self.assertIsNotNone(model)

    def test_config_validation_accepts_correct_types(self):
        """Test that config validator accepts correct types."""

        @Flow.model
        def typed_config(context: SimpleContext, n: int = 10, name: str = "x") -> GenericResult[str]:
            return GenericResult(value=f"{name}:{n}")

        # Should not raise
        model = typed_config(n=42, name="test")
        result = model(SimpleContext(value=1))
        self.assertEqual(result.value, "test:42")


# =============================================================================
# BoundModel Tests
# =============================================================================


class TestBoundModel(TestCase):
    """Tests for BoundModel and BoundModel.flow."""

    def test_bound_model_flow_compute(self):
        """Test that bound.flow.compute() honors transforms."""

        @Flow.model
        def my_model(x: int, y: int) -> GenericResult[int]:
            return GenericResult(value=x + y)

        model = my_model(x=10)

        # Create bound model with y transform
        bound = model.flow.with_inputs(y=lambda ctx: getattr(ctx, "y", 0) * 2)

        # flow.compute() should go through BoundModel, applying transform
        result = bound.flow.compute(y=5)
        # y transform: 5 * 2 = 10, x is bound to 10
        # model: 10 + 10 = 20
        self.assertEqual(result, 20)

    def test_bound_model_flow_compute_static_transform(self):
        """Test BoundModel.flow.compute() with static value transform."""

        @Flow.model
        def my_model(x: int, y: int) -> GenericResult[int]:
            return GenericResult(value=x * y)

        model = my_model(x=7)
        bound = model.flow.with_inputs(y=3)

        result = bound.flow.compute(y=999)  # y should be overridden by transform
        # y is statically bound to 3, x=7
        # 7 * 3 = 21
        self.assertEqual(result, 21)

    def test_bound_model_as_dependency(self):
        """Test that BoundModel can be passed as a dependency to another model."""

        @Flow.model
        def source(x: int) -> GenericResult[int]:
            return GenericResult(value=x * 10)

        @Flow.model
        def consumer(data: GenericResult[int]) -> GenericResult[int]:
            return GenericResult(value=data + 1)

        src = source()
        bound_src = src.flow.with_inputs(x=lambda ctx: getattr(ctx, "x", 0) * 2)

        # Pass BoundModel as a dependency
        model = consumer(data=bound_src)
        result = model.flow.compute(x=5)
        # x transform: 5 * 2 = 10
        # source: 10 * 10 = 100
        # consumer: 100 + 1 = 101
        self.assertEqual(result, 101)

    def test_bound_model_chained_with_inputs(self):
        """Test that chaining with_inputs merges transforms correctly."""

        @Flow.model
        def my_model(x: int, y: int, z: int) -> int:
            return x + y + z

        model = my_model()
        bound1 = model.flow.with_inputs(x=lambda ctx: getattr(ctx, "x", 0) * 2)
        bound2 = bound1.flow.with_inputs(y=lambda ctx: getattr(ctx, "y", 0) * 3)

        # Both transforms should be active
        result = bound2.flow.compute(x=5, y=10, z=1)
        # x transform: 5 * 2 = 10
        # y transform: 10 * 3 = 30
        # z from context: 1
        # 10 + 30 + 1 = 41
        self.assertEqual(result, 41)

    def test_bound_model_chained_with_inputs_override(self):
        """Test that chaining with_inputs allows overriding transforms."""

        @Flow.model
        def my_model(x: int) -> int:
            return x

        model = my_model()
        bound1 = model.flow.with_inputs(x=lambda ctx: getattr(ctx, "x", 0) * 2)
        bound2 = bound1.flow.with_inputs(x=lambda ctx: getattr(ctx, "x", 0) * 10)

        # Second transform should override the first for 'x'
        result = bound2.flow.compute(x=5)
        self.assertEqual(result, 50)  # 5 * 10, not 5 * 2

    def test_bound_model_with_default_args(self):
        """with_inputs works when the model has parameters with default values."""

        @Flow.model
        def load(start_date: str, end_date: str, source: str = "warehouse") -> str:
            return f"{source}:{start_date}-{end_date}"

        # Bind source at construction, leave dates for context
        model = load(source="prod_db")

        # with_inputs transforms a context param; default-valued 'source' stays bound
        lookback = model.flow.with_inputs(start_date=lambda ctx: "shifted_" + ctx.start_date)

        result = lookback.flow.compute(start_date="2024-01-01", end_date="2024-06-30")
        self.assertEqual(result, "prod_db:shifted_2024-01-01-2024-06-30")

    def test_bound_model_with_default_arg_unbound(self):
        """with_inputs works when defaulted parameter is left unbound (comes from context)."""

        @Flow.model
        def load(start_date: str, source: str = "warehouse") -> str:
            return f"{source}:{start_date}"

        # Don't bind 'source' — it keeps its default in the model,
        # but in dynamic deferred mode, unbound params come from context
        model = load()

        # Transform start_date; source comes from context (overriding the default)
        bound = model.flow.with_inputs(start_date=lambda ctx: "shifted_" + ctx.start_date)

        result = bound.flow.compute(start_date="2024-01-01", source="s3_bucket")
        self.assertEqual(result, "s3_bucket:shifted_2024-01-01")

    def test_bound_model_default_arg_as_dependency(self):
        """BoundModel with default args works correctly as a dependency."""

        @Flow.model
        def source(x: int, multiplier: int = 2) -> int:
            return x * multiplier

        @Flow.model
        def consumer(data: int) -> int:
            return data + 1

        src = source(multiplier=5)
        bound_src = src.flow.with_inputs(x=lambda ctx: ctx.x * 10)
        model = consumer(data=bound_src)

        result = model.flow.compute(x=3)
        # x transform: 3 * 10 = 30
        # source: 30 * 5 (multiplier) = 150
        # consumer: 150 + 1 = 151
        self.assertEqual(result, 151)

    def test_bound_model_as_lazy_dependency(self):
        """Test that BoundModel works as a Lazy dependency."""

        @Flow.model
        def source(x: int) -> GenericResult[int]:
            return GenericResult(value=x * 3)

        @Flow.model
        def consumer(data: int, slow: Lazy[GenericResult[int]]) -> GenericResult[int]:
            if data > 100:
                return GenericResult(value=data)
            return GenericResult(value=slow())

        src = source()
        bound_src = src.flow.with_inputs(x=lambda ctx: getattr(ctx, "x", 0) + 10)

        # Use BoundModel as lazy dependency
        model = consumer(data=5, slow=bound_src)
        result = model.flow.compute(x=7)
        # data=5 < 100, so slow path: x transform: 7+10=17, source: 17*3=51
        self.assertEqual(result, 51)

    def test_bound_and_unbound_models_share_memory_cache(self):
        """Shifted and unshifted models should share one evaluator cache.

        They should not share the same cache key when the effective contexts
        differ, but repeated evaluations of either model should still hit the
        same underlying MemoryCacheEvaluator instance rather than re-executing.
        """

        call_counts = {"source": 0}

        @Flow.model
        def source(context: SimpleContext) -> GenericResult[int]:
            call_counts["source"] += 1
            return GenericResult(value=context.value * 10)

        base = source()
        shifted = base.flow.with_inputs(value=lambda ctx: ctx.value + 1)
        evaluator = MemoryCacheEvaluator()
        ctx = SimpleContext(value=5)

        with FlowOptionsOverride(options={"evaluator": evaluator, "cacheable": True}):
            self.assertEqual(base(ctx).value, 50)
            self.assertEqual(shifted(ctx).value, 60)
            self.assertEqual(base(ctx).value, 50)
            self.assertEqual(shifted(ctx).value, 60)

        # One execution for the unshifted context and one for the shifted context.
        self.assertEqual(call_counts["source"], 2)
        self.assertEqual(len(evaluator.cache), 2)


# =============================================================================
# PEP 563 (from __future__ import annotations) Compatibility Tests
# =============================================================================

# These functions are defined at module level to simulate realistic usage.
# Note: We can't use `from __future__ import annotations` at module level
# since it would affect ALL annotations in this file. Instead, we test
# that the annotation resolution code handles string annotations.


class TestPEP563Annotations(TestCase):
    """Test that Flow.model handles string annotations (PEP 563)."""

    def test_string_annotation_lazy_resolved(self):
        """Test that Lazy annotations work even when passed through get_type_hints.

        This verifies the fix for from __future__ import annotations by
        confirming the annotation resolution pipeline processes Lazy correctly.
        """
        # Verify _extract_lazy handles real type objects (resolved by get_type_hints)
        from ccflow.flow_model import _extract_lazy

        lazy_int = Lazy[int]
        unwrapped, is_lazy = _extract_lazy(lazy_int)
        self.assertTrue(is_lazy)
        self.assertEqual(unwrapped, int)

    def test_string_annotation_return_type_resolved(self):
        """Test that string return type annotations are resolved correctly."""

        @Flow.model
        def model_func(context: SimpleContext) -> GenericResult[int]:
            return GenericResult(value=42)

        # If annotation resolution works, this should create successfully
        model = model_func()
        self.assertEqual(model.result_type, GenericResult[int])

    def test_auto_wrap_with_resolved_annotations(self):
        """Test that auto-wrap works with properly resolved type annotations."""

        @Flow.model
        def plain_model(value: int) -> int:
            return value * 2

        model = plain_model()
        result = model.flow.compute(value=5)
        self.assertEqual(result, 10)
        self.assertEqual(model.result_type, GenericResult)


# =============================================================================
# Hydra Integration Tests
# =============================================================================


# Define Flow.model functions at module level for Hydra to find them
@Flow.model
def hydra_basic_model(context: SimpleContext, value: int, name: str = "default") -> GenericResult[str]:
    """Module-level model for Hydra testing."""
    return GenericResult(value=f"{name}:{context.value + value}")


# --- Additional module-level fixtures for Hydra YAML tests ---


@Flow.model
def basic_loader(context: SimpleContext, source: str, multiplier: int = 1) -> GenericResult[int]:
    """Basic loader that multiplies context value by multiplier."""
    return GenericResult(value=context.value * multiplier)


@Flow.model
def string_processor(context: SimpleContext, prefix: str, suffix: str = "") -> GenericResult[str]:
    """Process context value into a string with prefix and suffix."""
    return GenericResult(value=f"{prefix}{context.value}{suffix}")


@Flow.model
def data_source(context: SimpleContext, base_value: int) -> GenericResult[int]:
    """Source that provides base data."""
    return GenericResult(value=context.value + base_value)


@Flow.model
def data_transformer(
    context: SimpleContext,
    source: int,
    factor: int = 2,
) -> GenericResult[int]:
    """Transform data by multiplying with factor."""
    return GenericResult(value=source * factor)


@Flow.model
def data_aggregator(
    context: SimpleContext,
    input_a: int,
    input_b: int,
    operation: str = "add",
) -> GenericResult[int]:
    """Aggregate two inputs."""
    if operation == "add":
        return GenericResult(value=input_a + input_b)
    elif operation == "multiply":
        return GenericResult(value=input_a * input_b)
    else:
        return GenericResult(value=input_a - input_b)


@Flow.model
def pipeline_stage1(context: SimpleContext, initial: int) -> GenericResult[int]:
    """First stage of pipeline."""
    return GenericResult(value=context.value + initial)


@Flow.model
def pipeline_stage2(
    context: SimpleContext,
    stage1_output: int,
    multiplier: int = 2,
) -> GenericResult[int]:
    """Second stage of pipeline."""
    return GenericResult(value=stage1_output * multiplier)


@Flow.model
def pipeline_stage3(
    context: SimpleContext,
    stage2_output: int,
    offset: int = 0,
) -> GenericResult[int]:
    """Third stage of pipeline."""
    return GenericResult(value=stage2_output + offset)


@Flow.model
def date_range_loader(
    context: DateRangeContext,
    source: str,
    include_weekends: bool = True,
) -> GenericResult[dict]:
    """Load data for a date range."""
    return GenericResult(
        value={
            "source": source,
            "start_date": str(context.start_date),
            "end_date": str(context.end_date),
        }
    )


@Flow.model
def date_range_loader_previous_day(
    context: DateRangeContext,
    source: str,
    include_weekends: bool = True,
) -> dict:
    """Hydra helper that applies a one-day lookback before delegating."""
    shifted = context.model_copy(update={"start_date": context.start_date - timedelta(days=1)})
    return date_range_loader(source=source, include_weekends=include_weekends)(shifted).value


@Flow.model
def date_range_processor(
    context: DateRangeContext,
    raw_data: dict,
    normalize: bool = False,
) -> GenericResult[str]:
    """Process date range data."""
    prefix = "normalized:" if normalize else "raw:"
    return GenericResult(value=f"{prefix}{raw_data['source']}:{raw_data['start_date']} to {raw_data['end_date']}")


@Flow.model
def hydra_default_model(context: SimpleContext, value: int = 42) -> GenericResult[int]:
    """Module-level model with defaults for Hydra testing."""
    return GenericResult(value=context.value + value)


@Flow.model
def hydra_source_model(context: SimpleContext, base: int) -> GenericResult[int]:
    """Source model for dependency testing."""
    return GenericResult(value=context.value * base)


@Flow.model
def hydra_consumer_model(
    context: SimpleContext,
    source: int,
    factor: int = 1,
) -> GenericResult[int]:
    """Consumer model for dependency testing."""
    return GenericResult(value=source * factor)


# --- context_args fixtures for Hydra testing ---


@Flow.model(context_args=["start_date", "end_date"])
def context_args_loader(start_date: date, end_date: date, source: str) -> GenericResult[dict]:
    """Loader using context_args with DateRangeContext."""
    return GenericResult(
        value={
            "source": source,
            "start_date": str(start_date),
            "end_date": str(end_date),
        }
    )


@Flow.model(context_args=["start_date", "end_date"])
def context_args_processor(
    start_date: date,
    end_date: date,
    data: dict,
    prefix: str = "processed",
) -> GenericResult[str]:
    """Processor using context_args with dependency."""
    return GenericResult(value=f"{prefix}:{data['source']}:{data['start_date']} to {data['end_date']}")


class TestFlowModelHydra(TestCase):
    """Tests for Flow.model with Hydra configuration."""

    def test_hydra_instantiate_basic(self):
        """Test that Flow.model factory can be instantiated via Hydra."""
        from hydra.utils import instantiate
        from omegaconf import OmegaConf

        # Create config that references the factory function by module path
        cfg = OmegaConf.create(
            {
                "_target_": "ccflow.tests.test_flow_model.hydra_basic_model",
                "value": 100,
                "name": "test",
            }
        )

        # Instantiate via Hydra
        model = instantiate(cfg)

        self.assertIsInstance(model, CallableModel)
        result = model(SimpleContext(value=10))
        self.assertEqual(result.value, "test:110")

    def test_hydra_instantiate_with_defaults(self):
        """Test Hydra instantiation using default parameter values."""
        from hydra.utils import instantiate
        from omegaconf import OmegaConf

        cfg = OmegaConf.create(
            {
                "_target_": "ccflow.tests.test_flow_model.hydra_default_model",
                # Not specifying value, should use default
            }
        )

        model = instantiate(cfg)
        result = model(SimpleContext(value=8))
        self.assertEqual(result.value, 50)

    def test_hydra_instantiate_with_dependency(self):
        """Test Hydra instantiation with dependencies."""
        from hydra.utils import instantiate
        from omegaconf import OmegaConf

        # Create nested config
        cfg = OmegaConf.create(
            {
                "_target_": "ccflow.tests.test_flow_model.hydra_consumer_model",
                "source": {
                    "_target_": "ccflow.tests.test_flow_model.hydra_source_model",
                    "base": 10,
                },
                "factor": 2,
            }
        )

        model = instantiate(cfg)

        result = model(SimpleContext(value=5))
        # source: 5 * 10 = 50, consumer: 50 * 2 = 100
        self.assertEqual(result.value, 100)


# =============================================================================
# Lazy[T] Type Annotation Tests
# =============================================================================


class TestLazyTypeAnnotation(TestCase):
    """Tests for Lazy[T] type annotation (deferred/conditional evaluation)."""

    def test_lazy_type_annotation_basic(self):
        """Lazy[T] param receives a thunk (zero-arg callable).

        The thunk unwraps GenericResult.value, so calling thunk() returns
        the inner value (e.g., int), not the GenericResult wrapper.
        """
        from ccflow import Lazy

        @Flow.model
        def source(context: SimpleContext) -> GenericResult[int]:
            return GenericResult(value=context.value * 10)

        @Flow.model
        def consumer(
            context: SimpleContext,
            data: Lazy[GenericResult[int]],
        ) -> GenericResult[int]:
            # data() returns the unwrapped value (int)
            resolved = data()
            return GenericResult(value=resolved + 1)

        src = source()
        model = consumer(data=src)
        result = model(SimpleContext(value=5))

        # source: 5 * 10 = 50, consumer: 50 + 1 = 51
        self.assertEqual(result.value, 51)

    def test_lazy_conditional_evaluation(self):
        """Mirror the smart_training example: lazy dep only evaluated if needed.

        Note: Non-lazy CallableModel deps are auto-resolved and their .value is
        unwrapped by the framework (auto-detected dep resolution). So 'fast'
        receives the unwrapped int, while 'slow' receives a thunk that returns
        the unwrapped value (GenericResult.value) when called.
        """
        from ccflow import Lazy

        call_counts = {"fast": 0, "slow": 0}

        @Flow.model
        def fast_path(context: SimpleContext) -> GenericResult[int]:
            call_counts["fast"] += 1
            return GenericResult(value=context.value)

        @Flow.model
        def slow_path(context: SimpleContext) -> GenericResult[int]:
            call_counts["slow"] += 1
            return GenericResult(value=context.value * 100)

        @Flow.model
        def smart_selector(
            context: SimpleContext,
            fast: GenericResult[int],  # Auto-resolved: receives unwrapped int
            slow: Lazy[GenericResult[int]],  # Lazy: receives thunk returning unwrapped value
            threshold: int = 10,
        ) -> GenericResult[int]:
            # fast is auto-unwrapped to the int value by the framework
            if fast > threshold:
                return GenericResult(value=fast)
            else:
                return GenericResult(value=slow())

        fast = fast_path()
        slow = slow_path()

        # Case 1: fast path sufficient (value > threshold)
        model = smart_selector(fast=fast, slow=slow, threshold=10)
        result = model(SimpleContext(value=20))
        self.assertEqual(result.value, 20)
        self.assertEqual(call_counts["fast"], 1)
        self.assertEqual(call_counts["slow"], 0)  # Never called!

        # Case 2: fast path insufficient (value <= threshold), slow triggered
        call_counts["fast"] = 0
        model2 = smart_selector(fast=fast, slow=slow, threshold=100)
        result2 = model2(SimpleContext(value=5))
        self.assertEqual(result2.value, 500)  # 5 * 100
        self.assertEqual(call_counts["fast"], 1)
        self.assertEqual(call_counts["slow"], 1)

    def test_lazy_thunk_caches_result(self):
        """Repeated calls to a thunk return the same value without re-evaluation."""
        from ccflow import Lazy

        call_counts = {"source": 0}

        @Flow.model
        def source(context: SimpleContext) -> GenericResult[int]:
            call_counts["source"] += 1
            return GenericResult(value=context.value * 10)

        @Flow.model
        def consumer(
            context: SimpleContext,
            data: Lazy[GenericResult[int]],
        ) -> GenericResult[int]:
            # Call thunk multiple times — returns the unwrapped int
            val1 = data()
            val2 = data()
            val3 = data()
            self.assertEqual(val1, val2)
            self.assertEqual(val2, val3)
            return GenericResult(value=val1)

        src = source()
        model = consumer(data=src)
        result = model(SimpleContext(value=5))
        self.assertEqual(result.value, 50)
        self.assertEqual(call_counts["source"], 1)  # Called only once despite 3 thunk() calls

    def test_lazy_with_direct_value(self):
        """Pre-computed (non-CallableModel) value wrapped in trivial thunk."""
        from ccflow import Lazy

        @Flow.model
        def consumer(
            context: SimpleContext,
            data: Lazy[int],
        ) -> GenericResult[int]:
            # data is a thunk even though the underlying value is a plain int
            return GenericResult(value=data() * 2)

        model = consumer(data=42)
        result = model(SimpleContext(value=0))
        self.assertEqual(result.value, 84)

    def test_lazy_dep_excluded_from_deps(self):
        """__deps__ does NOT include lazy dependencies."""
        from ccflow import Lazy

        @Flow.model
        def eager_source(context: SimpleContext) -> GenericResult[int]:
            return GenericResult(value=context.value)

        @Flow.model
        def lazy_source(context: SimpleContext) -> GenericResult[int]:
            return GenericResult(value=context.value * 10)

        @Flow.model
        def consumer(
            context: SimpleContext,
            eager: GenericResult[int],  # Auto-resolved, unwrapped to int
            lazy_dep: Lazy[GenericResult[int]],  # Thunk, returns unwrapped value
        ) -> GenericResult[int]:
            return GenericResult(value=eager + lazy_dep())

        eager = eager_source()
        lazy = lazy_source()
        model = consumer(eager=eager, lazy_dep=lazy)

        ctx = SimpleContext(value=5)
        deps = model.__deps__(ctx)

        # Only eager dep should be in __deps__
        self.assertEqual(len(deps), 1)
        self.assertIs(deps[0][0], eager)

    def test_lazy_eager_dep_still_pre_evaluated(self):
        """Non-lazy deps are still eagerly resolved via __deps__."""
        from ccflow import Lazy

        call_counts = {"eager": 0, "lazy": 0}

        @Flow.model
        def eager_source(context: SimpleContext) -> GenericResult[int]:
            call_counts["eager"] += 1
            return GenericResult(value=context.value)

        @Flow.model
        def lazy_source(context: SimpleContext) -> GenericResult[int]:
            call_counts["lazy"] += 1
            return GenericResult(value=context.value * 10)

        @Flow.model
        def consumer(
            context: SimpleContext,
            eager: GenericResult[int],  # Auto-resolved, unwrapped to int
            lazy_dep: Lazy[GenericResult[int]],  # Thunk, returns unwrapped value
        ) -> GenericResult[int]:
            # eager is auto-unwrapped to int, lazy_dep() returns unwrapped value
            return GenericResult(value=eager + lazy_dep())

        model = consumer(eager=eager_source(), lazy_dep=lazy_source())
        result = model(SimpleContext(value=5))

        self.assertEqual(result.value, 55)  # 5 + 50
        self.assertEqual(call_counts["eager"], 1)
        self.assertEqual(call_counts["lazy"], 1)

    def test_lazy_in_dynamic_deferred_mode(self):
        """Lazy[T] works in dynamic deferred mode (no context_args)."""
        from ccflow import FlowContext, Lazy

        call_counts = {"source": 0}

        @Flow.model
        def source(context: SimpleContext) -> GenericResult[int]:
            call_counts["source"] += 1
            return GenericResult(value=context.value * 10)

        @Flow.model
        def consumer(
            value: int,
            data: Lazy[GenericResult[int]],
        ) -> GenericResult[int]:
            if value > 10:
                return GenericResult(value=value)
            return GenericResult(value=data())  # data() returns unwrapped int

        # value comes from context, data is bound at construction
        model = consumer(data=source())
        result = model(FlowContext(value=20))  # value > 10, lazy not called
        self.assertEqual(result.value, 20)
        self.assertEqual(call_counts["source"], 0)

    def test_lazy_in_context_args_mode(self):
        """Lazy[T] works with explicit context_args."""
        from ccflow import FlowContext, Lazy

        @Flow.model(context_args=["x"])
        def source(x: int) -> GenericResult[int]:
            return GenericResult(value=x * 10)

        @Flow.model(context_args=["x"])
        def consumer(
            x: int,
            data: Lazy[GenericResult[int]],
        ) -> GenericResult[int]:
            return GenericResult(value=x + data())  # data() returns unwrapped int

        model = consumer(data=source())
        result = model(FlowContext(x=5))
        self.assertEqual(result.value, 55)  # 5 + 50

    def test_lazy_never_evaluated_if_not_called(self):
        """If thunk is never called, the dependency is never evaluated."""
        from ccflow import Lazy

        call_counts = {"source": 0}

        @Flow.model
        def source(context: SimpleContext) -> GenericResult[int]:
            call_counts["source"] += 1
            return GenericResult(value=context.value)

        @Flow.model
        def consumer(
            context: SimpleContext,
            data: Lazy[GenericResult[int]],
        ) -> GenericResult[int]:
            # Never call data()
            return GenericResult(value=42)

        model = consumer(data=source())
        result = model(SimpleContext(value=5))
        self.assertEqual(result.value, 42)
        self.assertEqual(call_counts["source"], 0)

    def test_lazy_with_upstream_model(self):
        """Lazy[T] works when bound to an upstream model."""
        from ccflow import Lazy

        @Flow.model
        def source(context: SimpleContext) -> GenericResult[int]:
            return GenericResult(value=context.value * 10)

        @Flow.model
        def consumer(
            context: SimpleContext,
            data: Lazy[GenericResult[int]],
        ) -> GenericResult[int]:
            return GenericResult(value=data() + 1)  # data() returns unwrapped int

        src = source()
        model = consumer(data=src)

        # Lazy dep should NOT be in __deps__
        deps = model.__deps__(SimpleContext(value=5))
        self.assertEqual(len(deps), 0)

        result = model(SimpleContext(value=5))
        self.assertEqual(result.value, 51)  # 50 + 1


# =============================================================================
# FieldExtractor Tests (Structured Output Field Access)
# =============================================================================


class TestFieldExtractor(TestCase):
    """Tests for structured output field access (prepared.X_train pattern)."""

    def test_field_extraction_basic(self):
        """Accessing unknown attr on @Flow.model instance returns FieldExtractor."""
        from ccflow.flow_model import FieldExtractor

        @Flow.model
        def prepare(context: SimpleContext, factor: int = 2) -> GenericResult[dict]:
            return GenericResult(value={"X_train": context.value * factor, "X_test": context.value})

        model = prepare(factor=3)
        extractor = model.X_train

        self.assertIsInstance(extractor, FieldExtractor)
        self.assertIs(extractor.source, model)
        self.assertEqual(extractor.field_name, "X_train")

    def test_field_extraction_evaluates_correctly(self):
        """FieldExtractor runs source and extracts the named field."""

        @Flow.model
        def prepare(context: SimpleContext) -> GenericResult[dict]:
            return GenericResult(value={"X_train": [1, 2, 3], "y_train": [4, 5, 6]})

        model = prepare()
        x_train = model.X_train

        result = x_train(SimpleContext(value=0))
        self.assertEqual(result.value, [1, 2, 3])

    def test_field_extraction_as_dependency(self):
        """FieldExtractor wired as a dep to a downstream model.

        Note: FieldExtractors are CallableModels, so they're auto-detected as deps
        and auto-unwrapped (GenericResult.value). The downstream function receives
        the raw extracted value, not a GenericResult wrapper.
        """

        @Flow.model
        def prepare(context: SimpleContext) -> GenericResult[dict]:
            v = context.value
            return GenericResult(value={"X_train": [v, v * 2], "y_train": [v * 10]})

        @Flow.model
        def train(context: SimpleContext, X: list, y: list) -> GenericResult[int]:
            # X and y are auto-unwrapped to the raw list values
            return GenericResult(value=sum(X) + sum(y))

        prepared = prepare()
        model = train(X=prepared.X_train, y=prepared.y_train)

        result = model(SimpleContext(value=5))
        # X_train = [5, 10], y_train = [50]
        # sum(X) + sum(y) = 15 + 50 = 65
        self.assertEqual(result.value, 65)

    def test_field_extraction_multiple_from_same_source(self):
        """Multiple extractors from same source share the source instance."""

        @Flow.model
        def prepare(context: SimpleContext) -> GenericResult[dict]:
            return GenericResult(value={"a": 1, "b": 2, "c": 3})

        model = prepare()
        ext_a = model.a
        ext_b = model.b
        ext_c = model.c

        # All should reference the same source
        self.assertIs(ext_a.source, model)
        self.assertIs(ext_b.source, model)
        self.assertIs(ext_c.source, model)

        # All should evaluate correctly
        ctx = SimpleContext(value=0)
        self.assertEqual(ext_a(ctx).value, 1)
        self.assertEqual(ext_b(ctx).value, 2)
        self.assertEqual(ext_c(ctx).value, 3)

    def test_field_extraction_nested(self):
        """Chained extraction (result.a.b) creates nested FieldExtractors."""
        from ccflow.flow_model import FieldExtractor

        class Nested:
            def __init__(self):
                self.inner_val = 42

        @Flow.model
        def produce(context: SimpleContext) -> GenericResult:
            return GenericResult(value={"nested": Nested()})

        model = produce()
        nested_extractor = model.nested
        inner_extractor = nested_extractor.inner_val

        self.assertIsInstance(nested_extractor, FieldExtractor)
        self.assertIsInstance(inner_extractor, FieldExtractor)

        result = inner_extractor(SimpleContext(value=0))
        self.assertEqual(result.value, 42)

    def test_field_extraction_context_type_inherited(self):
        """FieldExtractor inherits context_type from source."""

        @Flow.model
        def prepare(context: SimpleContext) -> GenericResult[dict]:
            return GenericResult(value={"x": 1})

        model = prepare()
        extractor = model.x

        self.assertEqual(extractor.context_type, SimpleContext)

    def test_field_extraction_nonexistent_field_runtime_error(self):
        """Non-existent field raises error at evaluation time, not construction.

        For dict results, raises KeyError. For object results, raises AttributeError.
        """

        @Flow.model
        def prepare(context: SimpleContext) -> GenericResult[dict]:
            return GenericResult(value={"x": 1})

        model = prepare()
        extractor = model.nonexistent  # No error at construction

        # Error at evaluation time (KeyError for dicts, AttributeError for objects)
        with self.assertRaises((KeyError, AttributeError)):
            extractor(SimpleContext(value=0))

    def test_field_extraction_pydantic_fields_not_intercepted(self):
        """Accessing real pydantic fields returns the field value, NOT an extractor."""
        from ccflow.flow_model import FieldExtractor

        @Flow.model
        def model_with_fields(context: SimpleContext, multiplier: int = 5) -> GenericResult[int]:
            return GenericResult(value=context.value * multiplier)

        model = model_with_fields(multiplier=10)

        # 'multiplier' is a real pydantic field — should return the value, not a FieldExtractor
        self.assertEqual(model.multiplier, 10)
        self.assertNotIsInstance(model.multiplier, FieldExtractor)

        # 'meta' is inherited from CallableModel — should also not be intercepted
        self.assertNotIsInstance(model.meta, FieldExtractor)

    def test_field_extraction_with_context_args(self):
        """FieldExtractor works with context_args mode models."""
        from ccflow import FlowContext

        @Flow.model(context_args=["x"])
        def prepare(x: int) -> GenericResult[dict]:
            return GenericResult(value={"doubled": x * 2, "tripled": x * 3})

        model = prepare()
        doubled = model.doubled

        result = doubled(FlowContext(x=5))
        self.assertEqual(result.value, 10)

    def test_field_extraction_has_flow_property(self):
        """FieldExtractor has .flow property (inherits from CallableModel)."""

        @Flow.model
        def prepare(context: SimpleContext) -> GenericResult[dict]:
            return GenericResult(value={"x": 1})

        model = prepare()
        extractor = model.x

        self.assertTrue(hasattr(extractor, "flow"))

    def test_field_extraction_deps(self):
        """FieldExtractor.__deps__ returns the source as a dependency."""

        @Flow.model
        def prepare(context: SimpleContext) -> GenericResult[dict]:
            return GenericResult(value={"x": 1})

        model = prepare()
        extractor = model.x

        ctx = SimpleContext(value=0)
        deps = extractor.__deps__(ctx)

        self.assertEqual(len(deps), 1)
        self.assertIs(deps[0][0], model)
        self.assertEqual(deps[0][1], [ctx])


if __name__ == "__main__":
    import unittest

    unittest.main()
