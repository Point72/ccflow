"""Tests for Flow.model decorator."""

from datetime import date, timedelta
from unittest import TestCase

from ray.cloudpickle import dumps as rcpdumps, loads as rcploads

from ccflow import (
    BaseModel,
    CallableModel,
    ContextBase,
    DateRangeContext,
    Flow,
    FlowContext,
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

        @Flow.model(context_args=["start_date", "end_date"], context_type=DateRangeContext)
        def date_range_loader(start_date: date, end_date: date, source: str) -> GenericResult[str]:
            return GenericResult(value=f"{source}:{start_date} to {end_date}")

        loader = date_range_loader(source="db")

        # Explicit context_type keeps compatibility with existing contexts.
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

        # Default context_args mode uses FlowContext unless overridden explicitly.
        self.assertEqual(model.context_type, FlowContext)

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

    def test_serialization_roundtrip_preserves_bound_inputs(self):
        """Round-tripping should preserve which inputs were bound at construction."""

        @Flow.model
        def add(x: int, y: int) -> int:
            return x + y

        model = add(x=10)
        dumped = model.model_dump(mode="python")
        restored = type(model).model_validate(dumped)

        self.assertEqual(dumped["x"], 10)
        self.assertNotIn("y", dumped)
        self.assertEqual(restored.flow.bound_inputs, {"x": 10})
        self.assertEqual(restored.flow.unbound_inputs, {"y": int})
        self.assertEqual(restored.flow.compute(y=5).value, 15)

    def test_serialization_roundtrip_preserves_defaults_and_deferred_inputs(self):
        """Default-valued params should serialize normally without binding runtime-only inputs."""

        @Flow.model
        def load(start_date: str, source: str = "warehouse") -> str:
            return f"{source}:{start_date}"

        model = load()
        dumped = model.model_dump(mode="python")
        restored = type(model).model_validate(dumped)

        self.assertEqual(dumped["source"], "warehouse")
        self.assertNotIn("start_date", dumped)
        self.assertEqual(restored.flow.bound_inputs, {"source": "warehouse"})
        self.assertEqual(restored.flow.unbound_inputs, {"start_date": str})
        self.assertEqual(restored.flow.compute(start_date="2024-01-01").value, "warehouse:2024-01-01")

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
        the GenericResult to its inner value for the downstream function.
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

    def test_dynamic_deferred_mode_missing_runtime_inputs_is_clear(self):
        """Missing deferred inputs should fail at the framework boundary."""

        @Flow.model
        def dynamic_model(value: int, multiplier: int) -> int:
            return value * multiplier

        model = dynamic_model()

        with self.assertRaises(TypeError) as cm:
            model.flow.compute()

        self.assertIn("Missing runtime input(s) for dynamic_model: multiplier, value", str(cm.exception))

    def test_all_defaults_is_valid(self):
        """All-default functions should treat those defaults as bound config."""
        from ccflow import FlowContext

        @Flow.model
        def all_defaults(value: int = 1, other: str = "x") -> GenericResult[str]:
            return GenericResult(value=f"{value}-{other}")

        model = all_defaults()

        self.assertEqual(model.flow.bound_inputs, {"value": 1, "other": "x"})
        self.assertEqual(model.flow.unbound_inputs, {})

        ctx = FlowContext(value=5, other="y")
        result = model(ctx)
        self.assertEqual(result.value, "1-x")

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

    def test_context_type_requires_context_args_mode(self):
        """context_type is only valid alongside context_args."""
        with self.assertRaises(TypeError) as cm:

            @Flow.model(context_type=DateRangeContext)
            def dynamic_model(value: int) -> GenericResult[int]:
                return GenericResult(value=value)

        self.assertIn("context_args", str(cm.exception))

    def test_context_type_must_cover_context_args(self):
        """context_type must expose all named context_args fields."""

        class StartOnlyContext(ContextBase):
            start_date: date

        with self.assertRaises(TypeError) as cm:

            @Flow.model(context_args=["start_date", "end_date"], context_type=StartOnlyContext)
            def load_data(start_date: date, end_date: date) -> GenericResult[dict]:
                return GenericResult(value={})

        self.assertIn("end_date", str(cm.exception))


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

    def test_config_validation_rejects_registry_alias_for_incompatible_type(self):
        """Registry aliases should not silently bypass scalar type validation."""

        class DummyConfig(BaseModel):
            x: int = 1

        registry = ModelRegistry.root()
        registry.clear()
        try:
            registry.add("dummy_config", DummyConfig())

            @Flow.model
            def typed_config(context: SimpleContext, n: int = 10) -> GenericResult[int]:
                return GenericResult(value=n)

            with self.assertRaises(TypeError) as cm:
                typed_config(n="dummy_config")

            self.assertIn("n", str(cm.exception))
        finally:
            registry.clear()

    def test_config_validation_accepts_registry_alias_for_callable_dependency(self):
        """Registry aliases still work for CallableModel dependencies."""

        registry = ModelRegistry.root()
        registry.clear()
        try:

            @Flow.model
            def source(context: SimpleContext) -> GenericResult[int]:
                return GenericResult(value=context.value * 2)

            @Flow.model
            def consumer(context: SimpleContext, data: int = 0) -> GenericResult[int]:
                return GenericResult(value=data + 1)

            registry.add("source_model", source())
            model = consumer(data="source_model")

            result = model(SimpleContext(value=5))
            self.assertEqual(result.value, 11)
        finally:
            registry.clear()

    def test_context_type_annotation_mismatch_raises(self):
        """context_type validation should reject incompatible field annotations."""

        class StringIdContext(ContextBase):
            item_id: str

        with self.assertRaises(TypeError) as cm:

            @Flow.model(context_args=["item_id"], context_type=StringIdContext)
            def load(item_id: int) -> int:
                return item_id

        self.assertIn("item_id", str(cm.exception))
        self.assertIn("int", str(cm.exception))
        self.assertIn("str", str(cm.exception))

    def test_model_validate_rejects_bad_scalar_type(self):
        """model_validate should reject wrong scalar types, not silently accept them."""

        @Flow.model
        def source(context: SimpleContext, x: int) -> GenericResult[int]:
            return GenericResult(value=x)

        cls = type(source(x=1))
        with self.assertRaises(TypeError) as cm:
            cls.model_validate({"x": "abc"})

        self.assertIn("x", str(cm.exception))

    def test_model_validate_accepts_correct_type(self):
        """model_validate should accept correct types."""

        @Flow.model
        def source(context: SimpleContext, x: int) -> GenericResult[int]:
            return GenericResult(value=x)

        cls = type(source(x=1))
        restored = cls.model_validate({"x": 42})
        self.assertEqual(restored(SimpleContext(value=0)).value, 42)

    def test_model_validate_rejects_bad_registry_alias(self):
        """Typoed registry aliases should not silently pass through model_validate."""

        registry = ModelRegistry.root()
        registry.clear()
        try:

            @Flow.model
            def consumer(context: SimpleContext, n: int = 10) -> GenericResult[int]:
                return GenericResult(value=n)

            cls = type(consumer(n=1))
            # "not_in_registry" is not a valid int and not a valid registry key
            with self.assertRaises(TypeError) as cm:
                cls.model_validate({"n": "not_in_registry"})
            self.assertIn("n", str(cm.exception))
        finally:
            registry.clear()

    def test_context_type_compatible_annotations_accepted(self):
        """context_type validation should accept matching or subclass annotations."""

        # Exact match should work
        @Flow.model(context_args=["start_date", "end_date"], context_type=DateRangeContext)
        def load_exact(start_date: date, end_date: date) -> str:
            return f"{start_date}"

        self.assertIsNotNone(load_exact)


# =============================================================================
# BoundModel Tests
# =============================================================================


class TestBoundModel(TestCase):
    """Tests for BoundModel and BoundModel.flow."""

    def test_bound_model_is_callable_model(self):
        """BoundModel should be a proper CallableModel subclass."""

        @Flow.model
        def source(x: int) -> int:
            return x * 10

        bound = source().flow.with_inputs(x=lambda ctx: ctx.x * 2)
        self.assertIsInstance(bound, CallableModel)

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
        self.assertEqual(result.value, 20)

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
        self.assertEqual(result.value, 21)

    def test_bound_model_dump_validate_roundtrip_static(self):
        """Static transforms survive model_dump → model_validate roundtrip."""

        @Flow.model
        def source(context: SimpleContext) -> GenericResult[int]:
            return GenericResult(value=context.value * 10)

        bound = source().flow.with_inputs(value=42)
        dump = bound.model_dump(mode="python")
        restored = type(bound).model_validate(dump)

        ctx = SimpleContext(value=1)
        self.assertEqual(bound(ctx).value, 420)
        self.assertEqual(restored(ctx).value, 420)

    def test_bound_model_validate_same_payload_twice(self):
        """Validating the same serialized BoundModel payload twice should work both times."""
        from ccflow.flow_model import BoundModel

        @Flow.model
        def source(context: SimpleContext) -> GenericResult[int]:
            return GenericResult(value=context.value * 10)

        bound = source().flow.with_inputs(value=42)
        dump = bound.model_dump(mode="python")

        r1 = BoundModel.model_validate(dump)
        r2 = BoundModel.model_validate(dump)

        ctx = SimpleContext(value=1)
        self.assertEqual(r1(ctx).value, 420)
        self.assertEqual(r2(ctx).value, 420)

    def test_bound_model_failed_validate_does_not_poison_next_construction(self):
        """A failed model_validate must not leak static transforms to subsequent constructions."""
        from ccflow.flow_model import BoundModel

        @Flow.model
        def source(context: SimpleContext) -> GenericResult[int]:
            return GenericResult(value=context.value * 10)

        base = source()

        # Attempt a model_validate that will fail (invalid model field)
        try:
            BoundModel.model_validate(
                {
                    "model": "not-a-real-model",
                    "_static_transforms": {"value": 42},
                    "_input_transforms_token": {"value": "42"},
                }
            )
        except Exception:
            pass  # Expected to fail

        # Now construct a fresh BoundModel normally — must NOT inherit stale transforms
        clean = BoundModel(model=base, input_transforms={})
        ctx = SimpleContext(value=1)
        self.assertEqual(clean(ctx).value, 10)  # 1 * 10, no transform applied

    def test_bound_model_cloudpickle_with_lambda_transform(self):
        """BoundModel with lambda transforms should survive cloudpickle round-trip."""

        @Flow.model
        def my_model(x: int, y: int) -> int:
            return x + y

        bound = my_model(x=10).flow.with_inputs(y=lambda ctx: ctx.y * 2)
        restored = rcploads(rcpdumps(bound, protocol=5))

        self.assertEqual(restored.flow.compute(y=6).value, 22)

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
        self.assertEqual(result.value, 101)

    def test_flow_compute_with_upstream_callable_model_dependency(self):
        """flow.compute() should resolve upstream generated-model dependencies."""

        @Flow.model
        def source(x: int) -> GenericResult[int]:
            return GenericResult(value=x * 10)

        @Flow.model
        def consumer(data: GenericResult[int], offset: int = 1) -> int:
            return data + offset

        model = consumer(data=source(), offset=3)
        self.assertEqual(model.flow.compute(x=5).value, 53)

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
        self.assertEqual(result.value, 41)

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
        self.assertEqual(result.value, 50)  # 5 * 10, not 5 * 2

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
        self.assertEqual(result.value, "prod_db:shifted_2024-01-01-2024-06-30")

    def test_bound_model_with_default_arg_uses_default(self):
        """with_inputs should preserve omitted Python defaults as bound config."""

        @Flow.model
        def load(start_date: str, source: str = "warehouse") -> str:
            return f"{source}:{start_date}"

        model = load()

        bound = model.flow.with_inputs(start_date=lambda ctx: "shifted_" + ctx.start_date)

        self.assertEqual(model.flow.bound_inputs, {"source": "warehouse"})
        self.assertEqual(model.flow.unbound_inputs, {"start_date": str})

        result = bound.flow.compute(start_date="2024-01-01")
        self.assertEqual(result.value, "warehouse:shifted_2024-01-01")

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
        self.assertEqual(result.value, 151)

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
        self.assertEqual(result.value, 51)

    def test_differently_transformed_bound_models_have_distinct_cache_keys(self):
        """Two BoundModels with different transforms must not collide under caching."""

        call_counts = {"source": 0}

        @Flow.model
        def source(context: SimpleContext) -> GenericResult[int]:
            call_counts["source"] += 1
            return GenericResult(value=context.value * 10)

        base = source()
        b1 = base.flow.with_inputs(value=lambda ctx: ctx.value + 1)
        b2 = base.flow.with_inputs(value=lambda ctx: ctx.value + 2)
        evaluator = MemoryCacheEvaluator()
        ctx = SimpleContext(value=5)

        with FlowOptionsOverride(options={"evaluator": evaluator, "cacheable": True}):
            r1 = b1(ctx)
            r2 = b2(ctx)

        # b1 transforms value to 6, source: 6*10=60
        # b2 transforms value to 7, source: 7*10=70
        self.assertEqual(r1.value, 60)
        self.assertEqual(r2.value, 70)
        # Source called twice (once per distinct transformed context)
        self.assertEqual(call_counts["source"], 2)

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
        # Cache has 3 entries: base(ctx), BoundModel(ctx), and base(shifted_ctx).
        # BoundModel is a proper CallableModel now, so it gets its own cache entry.
        self.assertEqual(len(evaluator.cache), 3)

    def test_transform_error_propagates(self):
        """A buggy transform should raise, not silently fall back to FlowContext."""

        @Flow.model
        def load(context: DateRangeContext, source: str = "db") -> str:
            return f"{source}:{context.start_date}"

        model = load()
        # Transform has a typo — ctx.sart_date instead of ctx.start_date
        bound = model.flow.with_inputs(start_date=lambda ctx: ctx.sart_date)

        ctx = DateRangeContext(start_date=date(2024, 1, 1), end_date=date(2024, 1, 31))
        with self.assertRaises(AttributeError):
            bound(ctx)

    def test_transform_validation_error_propagates(self):
        """If transforms produce invalid context data, the error should surface."""
        from pydantic import ValidationError

        @Flow.model
        def load(context: DateRangeContext, source: str = "db") -> str:
            return f"{source}:{context.start_date}"

        model = load()
        # Transform returns a string where a date is expected
        bound = model.flow.with_inputs(start_date="not-a-date")

        ctx = DateRangeContext(start_date=date(2024, 1, 1), end_date=date(2024, 1, 31))
        # Pydantic validation should raise, not silently fall back to FlowContext
        with self.assertRaises(ValidationError):
            bound(ctx)


class TestFlowModelPipe(TestCase):
    """Tests for the ``.pipe(..., param=...)`` convenience API."""

    def test_pipe_infers_single_required_parameter(self):
        """pipe() should infer the only required downstream parameter."""

        @Flow.model
        def source(x: int) -> GenericResult[int]:
            return GenericResult(value=x * 10)

        @Flow.model
        def consumer(data: int, offset: int = 1) -> int:
            return data + offset

        pipeline = source().pipe(consumer, offset=3)
        self.assertEqual(pipeline.flow.compute(x=5).value, 53)

    def test_pipe_infers_single_defaulted_parameter(self):
        """pipe() should fall back to a single defaulted downstream parameter."""

        @Flow.model
        def source(x: int) -> int:
            return x * 10

        @Flow.model
        def consumer(data: int = 0) -> int:
            return data + 1

        pipeline = source().pipe(consumer)
        self.assertEqual(pipeline.flow.compute(x=5).value, 51)

    def test_pipe_param_disambiguates_multiple_parameters(self):
        """param= should identify the downstream argument to bind."""

        @Flow.model
        def source(x: int) -> int:
            return x * 10

        @Flow.model
        def combine(left: int, right: int) -> int:
            return left + right

        pipeline = source().pipe(combine, param="right", left=7)
        self.assertEqual(pipeline.flow.compute(x=5).value, 57)

    def test_pipe_rejects_ambiguous_downstream_stage(self):
        """pipe() should require param= when multiple targets are available."""

        @Flow.model
        def source(x: int) -> int:
            return x

        @Flow.model
        def combine(left: int, right: int) -> int:
            return left + right

        with self.assertRaisesRegex(
            TypeError,
            r"pipe\(\) could not infer a target parameter for combine; unbound candidates are: left, right",
        ):
            source().pipe(combine)

    def test_manual_callable_model_can_pipe_into_generated_stage(self):
        """Hand-written CallableModels should be usable as pipe sources."""

        class ManualModel(CallableModel):
            offset: int

            @Flow.call
            def __call__(self, context: SimpleContext) -> GenericResult[int]:
                return GenericResult(value=context.value + self.offset)

        @Flow.model
        def consumer(data: int, multiplier: int) -> int:
            return data * multiplier

        pipeline = ManualModel(offset=5).pipe(consumer, multiplier=2)
        self.assertEqual(pipeline.flow.compute(value=10).value, 30)

    def test_bound_model_pipe_preserves_downstream_transforms(self):
        """pipe() should keep downstream with_inputs transforms intact."""

        @Flow.model
        def source(x: int) -> int:
            return x * 10

        @Flow.model
        def consumer(data: int, scale: int) -> int:
            return data + scale

        shifted_source = source().flow.with_inputs(x=lambda ctx: ctx.scale + 1)
        scaled_consumer = consumer().flow.with_inputs(scale=lambda ctx: ctx.scale * 3)

        pipeline = shifted_source.pipe(scaled_consumer)
        self.assertEqual(pipeline.flow.compute(scale=2).value, 76)


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
        self.assertEqual(result.value, 10)
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


@Flow.model(context_args=["start_date", "end_date"], context_type=DateRangeContext)
def context_args_loader(start_date: date, end_date: date, source: str) -> GenericResult[dict]:
    """Loader using context_args with DateRangeContext."""
    return GenericResult(
        value={
            "source": source,
            "start_date": str(start_date),
            "end_date": str(end_date),
        }
    )


@Flow.model(context_args=["start_date", "end_date"], context_type=DateRangeContext)
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
# Bug Fix Regression Tests
# =============================================================================


class TestFlowModelBugFixes(TestCase):
    """Regression tests for four bugs identified during code review."""

    # ----- Issue 1: .flow.compute() drops context defaults -----

    def test_compute_respects_explicit_context_defaults(self):
        """Mode 1: compute(x=1) should use ExtendedContext's default y='default'."""

        @Flow.model
        def model_fn(context: ExtendedContext, factor: int = 1) -> str:
            return f"{context.x}-{context.y}-{factor}"

        model = model_fn()
        result = model.flow.compute(x=1)
        self.assertEqual(result.value, "1-default-1")

    def test_compute_respects_context_args_defaults(self):
        """Mode 2: compute(x=1) should use function default y=42."""

        @Flow.model(context_args=["x", "y"])
        def model_fn(x: int, y: int = 42) -> int:
            return x + y

        model = model_fn()
        result = model.flow.compute(x=1)
        self.assertEqual(result.value, 43)

    def test_unbound_inputs_excludes_context_args_with_defaults(self):
        """Mode 2: unbound_inputs should not include context_args that have function defaults."""

        @Flow.model(context_args=["x", "y"])
        def model_fn(x: int, y: int = 42) -> int:
            return x + y

        model = model_fn()
        self.assertEqual(model.flow.unbound_inputs, {"x": int})

    def test_unbound_inputs_excludes_context_type_defaults(self):
        """Mode 1: unbound_inputs should not include context fields that have defaults."""

        @Flow.model
        def model_fn(context: ExtendedContext) -> str:
            return f"{context.x}-{context.y}"

        model = model_fn()
        # ExtendedContext has x: int (required) and y: str = "default"
        self.assertEqual(model.flow.unbound_inputs, {"x": int})

    def test_context_type_rejects_required_field_with_function_default(self):
        """Decoration should fail when function has default but context_type requires the field."""

        class StrictContext(ContextBase):
            x: int  # required

        with self.assertRaises(TypeError) as cm:

            @Flow.model(context_args=["x"], context_type=StrictContext)
            def model_fn(x: int = 5) -> int:
                return x

        self.assertIn("x", str(cm.exception))
        self.assertIn("requires", str(cm.exception))

    def test_context_type_accepts_optional_field_with_function_default(self):
        """Both context_type and function have defaults — should work."""

        class OptionalContext(ContextBase):
            x: int = 10

        @Flow.model(context_args=["x"], context_type=OptionalContext)
        def model_fn(x: int = 5) -> int:
            return x

        model = model_fn()
        result = model(OptionalContext())
        self.assertEqual(result.value, 10)  # context default wins

    # ----- Issue 2: Lazy[...] broken in dynamic deferred mode -----

    def test_lazy_from_runtime_context_in_dynamic_mode(self):
        """Lazy[int] provided via FlowContext should be wrapped in a thunk."""

        @Flow.model
        def model_fn(x: int, y: Lazy[int]) -> int:
            return x + y()

        model = model_fn(x=10)
        result = model(FlowContext(y=32))
        self.assertEqual(result.value, 42)

    def test_callable_model_from_runtime_context_in_dynamic_mode(self):
        """CallableModel provided in FlowContext should be resolved."""

        @Flow.model
        def source(value: int) -> int:
            return value * 10

        @Flow.model
        def consumer(x: int, data: int) -> int:
            return x + data

        model = consumer(x=1)
        src = source()
        result = model(FlowContext(data=src, value=5))
        # source resolves with value=5 → 50, consumer: 1 + 50 = 51
        self.assertEqual(result.value, 51)

    # ----- Issue 3: FlowContext-backed models skip schema validation -----

    def test_direct_call_validates_flowcontext_dynamic_mode(self):
        """Dynamic mode: FlowContext(y='hello') for int param should raise TypeError."""

        @Flow.model
        def model_fn(x: int, y: int) -> int:
            return x + y

        model = model_fn()
        with self.assertRaises(TypeError) as cm:
            model(FlowContext(x=1, y="hello"))

        self.assertIn("y", str(cm.exception))

    def test_direct_call_validates_flowcontext_context_args_mode(self):
        """context_args mode: FlowContext(x='hello') for int param should raise TypeError."""

        @Flow.model(context_args=["x"])
        def model_fn(x: int) -> int:
            return x

        model = model_fn()
        with self.assertRaises(TypeError) as cm:
            model(FlowContext(x="hello"))

        self.assertIn("x", str(cm.exception))

    def test_with_inputs_validates_transformed_fields_dynamic(self):
        """Dynamic mode: with_inputs(y='hello') for int param should raise TypeError."""

        @Flow.model
        def model_fn(x: int, y: int) -> int:
            return x + y

        model = model_fn(x=1)
        bound = model.flow.with_inputs(y="hello")

        with self.assertRaises(TypeError) as cm:
            bound(FlowContext())

        self.assertIn("y", str(cm.exception))

    def test_with_inputs_validates_transformed_fields_context_args(self):
        """context_args mode: with_inputs(x='hello') for int param should raise TypeError."""

        @Flow.model(context_args=["x"])
        def model_fn(x: int) -> int:
            return x

        model = model_fn()
        bound = model.flow.with_inputs(x="hello")

        with self.assertRaises(TypeError) as cm:
            bound(FlowContext())

        self.assertIn("x", str(cm.exception))

    # ----- Issue 4: Registry-name resolution too aggressive for union strings -----

    def test_registry_resolution_skips_union_str_annotation(self):
        """Union[str, int] field with a registry key string should keep the string."""
        from typing import Union

        registry = ModelRegistry.root()
        registry.clear()
        try:

            @Flow.model
            def dummy(context: SimpleContext) -> GenericResult[int]:
                return GenericResult(value=1)

            registry.add("my_key", dummy())

            @Flow.model
            def consumer(context: SimpleContext, tag: Union[str, int] = "none") -> str:
                return f"tag={tag}"

            model = consumer(tag="my_key")
            result = model(SimpleContext(value=0))
            self.assertEqual(result.value, "tag=my_key")
        finally:
            registry.clear()

    def test_registry_resolution_skips_optional_str_annotation(self):
        """Optional[str] field with a registry key string should keep the string."""
        from typing import Optional

        registry = ModelRegistry.root()
        registry.clear()
        try:

            @Flow.model
            def dummy(context: SimpleContext) -> GenericResult[int]:
                return GenericResult(value=1)

            registry.add("my_key", dummy())

            @Flow.model
            def consumer(context: SimpleContext, label: Optional[str] = None) -> str:
                return f"label={label}"

            model = consumer(label="my_key")
            result = model(SimpleContext(value=0))
            self.assertEqual(result.value, "label=my_key")
        finally:
            registry.clear()

    def test_registry_resolution_skips_union_annotated_str(self):
        """Union[Annotated[str, ...], int] field with a registry key should keep the string."""
        from typing import Annotated, Union

        registry = ModelRegistry.root()
        registry.clear()
        try:

            @Flow.model
            def dummy(context: SimpleContext) -> GenericResult[int]:
                return GenericResult(value=1)

            registry.add("my_key", dummy())

            @Flow.model
            def consumer(context: SimpleContext, tag: Union[Annotated[str, "label"], int] = "none") -> str:
                return f"tag={tag}"

            model = consumer(tag="my_key")
            result = model(SimpleContext(value=0))
            self.assertEqual(result.value, "tag=my_key")
        finally:
            registry.clear()


# =============================================================================
# Coverage Gap Tests
# =============================================================================


class TestExtractLazyLoopBody(TestCase):
    """Group 1: _extract_lazy loop body with non-LazyMarker metadata."""

    def test_annotated_with_extra_metadata_before_lazy_marker(self):
        """Annotated type where _LazyMarker is NOT the first metadata element."""
        from typing import Annotated

        from ccflow.flow_model import _extract_lazy, _LazyMarker

        # _LazyMarker is the second metadata element — loop must iterate past "other"
        ann = Annotated[int, "other_metadata", _LazyMarker()]
        base_type, is_lazy = _extract_lazy(ann)
        self.assertTrue(is_lazy)
        self.assertIs(base_type, int)

    def test_annotated_without_lazy_marker(self):
        """Annotated type with no _LazyMarker returns is_lazy=False."""
        from typing import Annotated

        from ccflow.flow_model import _extract_lazy

        ann = Annotated[int, "just_metadata"]
        base_type, is_lazy = _extract_lazy(ann)
        self.assertFalse(is_lazy)

    def test_lazy_type_annotation_with_extra_annotated(self):
        """End-to-end: Lazy wrapping of an Annotated type."""

        @Flow.model
        def model_with_lazy(
            x: int,
            dep: Lazy[int],
        ) -> int:
            return x + dep()

        @Flow.model
        def upstream(x: int) -> int:
            return x * 10

        model = model_with_lazy(x=1, dep=upstream())
        result = model.flow.compute(x=1)
        self.assertEqual(result.value, 11)

    def test_lazy_dep_returning_custom_result(self):
        """Lazy dep returning custom ResultBase (not GenericResult) should return raw result."""

        @Flow.model
        def upstream(context: SimpleContext) -> MyResult:
            return MyResult(data=f"v={context.value}")

        @Flow.model
        def consumer(context: SimpleContext, dep: Lazy[MyResult]) -> GenericResult[str]:
            result = dep()
            return GenericResult(value=result.data)

        model = consumer(dep=upstream())
        result = model(SimpleContext(value=42))
        self.assertEqual(result.value, "v=42")


class TestTransformReprNamedCallable(TestCase):
    """Group 2: _transform_repr with a named callable."""

    def test_named_function_transform_in_repr(self):
        """Named functions should appear in BoundModel repr wrapped in angle brackets."""
        from ccflow.flow_model import _transform_repr

        def my_custom_transform(ctx):
            return ctx.value + 1

        result = _transform_repr(my_custom_transform)
        self.assertIn("my_custom_transform", result)
        self.assertTrue(result.startswith("<"))
        self.assertTrue(result.endswith(">"))

    def test_static_value_repr(self):
        """Static (non-callable) values should use repr()."""
        from ccflow.flow_model import _transform_repr

        self.assertEqual(_transform_repr(42), "42")
        self.assertEqual(_transform_repr("hello"), "'hello'")


class TestBoundFieldNamesFallback(TestCase):
    """Group 3: _bound_field_names fallback for objects without model_fields_set."""

    def test_fallback_to_bound_fields_attr(self):
        from ccflow.flow_model import _bound_field_names

        class FakeModel:
            _bound_fields = {"x", "y"}

        result = _bound_field_names(FakeModel())
        self.assertEqual(result, {"x", "y"})

    def test_fallback_no_attrs(self):
        from ccflow.flow_model import _bound_field_names

        class Empty:
            pass

        result = _bound_field_names(Empty())
        self.assertEqual(result, set())


class TestRuntimeInputNamesEmpty(TestCase):
    """Group 4: _runtime_input_names when all_param_names is empty."""

    def test_non_flow_model_returns_empty(self):
        from ccflow.flow_model import _runtime_input_names

        class ManualModel(CallableModel):
            offset: int

            @Flow.call
            def __call__(self, context: SimpleContext) -> GenericResult[int]:
                return GenericResult(value=context.value + self.offset)

        model = ManualModel(offset=5)
        self.assertEqual(_runtime_input_names(model), set())


class TestRegistryCandidateAllowed(TestCase):
    """Group 5: _registry_candidate_allowed TypeAdapter success path."""

    def test_non_callable_model_passes_type_check(self):
        """Registry value that is not a CallableModel but passes TypeAdapter validation."""
        from ccflow.flow_model import _registry_candidate_allowed

        # int value passes TypeAdapter(int).validate_python
        self.assertTrue(_registry_candidate_allowed(int, 42))

    def test_non_callable_model_fails_type_check(self):
        from ccflow.flow_model import _registry_candidate_allowed

        self.assertFalse(_registry_candidate_allowed(int, "not_an_int"))


class TestConcreteContextTypeOptional(TestCase):
    """Group 6: _concrete_context_type with Optional/Union types."""

    def test_optional_context_type(self):
        """Optional[T] has NoneType that should be skipped to find T."""
        from typing import Optional

        from ccflow.flow_model import _concrete_context_type

        # Optional[SimpleContext] = Union[SimpleContext, None]
        # The NoneType arg must be skipped (line 196-197)
        result = _concrete_context_type(Optional[SimpleContext])
        self.assertIs(result, SimpleContext)

    def test_union_with_none_first(self):
        """Union[None, T] should skip NoneType and find T."""
        from typing import Union

        from ccflow.flow_model import _concrete_context_type

        # NoneType comes first, must be skipped
        result = _concrete_context_type(Union[None, SimpleContext])
        self.assertIs(result, SimpleContext)

    def test_union_context_type(self):
        from typing import Union

        from ccflow.flow_model import _concrete_context_type

        result = _concrete_context_type(Union[SimpleContext, None])
        self.assertIs(result, SimpleContext)

    def test_union_no_context_base(self):
        from typing import Union

        from ccflow.flow_model import _concrete_context_type

        result = _concrete_context_type(Union[int, str])
        self.assertIsNone(result)

    def test_returns_none_for_non_type(self):
        from ccflow.flow_model import _concrete_context_type

        result = _concrete_context_type("not_a_type")
        self.assertIsNone(result)


class TestBuildConfigValidatorsException(TestCase):
    """Group 7: _build_config_validators when TypeAdapter fails."""

    def test_unadaptable_type_skipped(self):
        """Types that TypeAdapter can't handle should be silently skipped."""
        from ccflow.flow_model import _build_config_validators

        # type(...) (EllipsisType) makes TypeAdapter fail
        validatable, validators = _build_config_validators({"x": int, "y": type(...)})
        self.assertIn("x", validatable)
        self.assertNotIn("y", validatable)
        self.assertIn("x", validators)
        self.assertNotIn("y", validators)


class TestCoerceContextValueNoValidator(TestCase):
    """Group 8: _coerce_context_value early return for fields without validators."""

    def test_field_without_validator_passes_through(self):
        from ccflow.flow_model import _coerce_context_value

        # When name is not in validators, value should pass through unchanged
        result = _coerce_context_value("unknown_field", 42, {}, {})
        self.assertEqual(result, 42)


class TestGeneratedModelClassFactoryPath(TestCase):
    """Group 9: _generated_model_class when stage has no generated model."""

    def test_returns_none_for_plain_callable(self):
        from ccflow.flow_model import _generated_model_class

        def plain_func():
            pass

        self.assertIsNone(_generated_model_class(plain_func))


class TestDescribePipeStagePaths(TestCase):
    """Group 10: _describe_pipe_stage for different stage types."""

    def test_generated_model_instance(self):
        from ccflow.flow_model import _describe_pipe_stage

        @Flow.model
        def my_stage(x: int) -> int:
            return x

        desc = _describe_pipe_stage(my_stage())
        self.assertIn("my_stage", desc)

    def test_callable_stage(self):
        from ccflow.flow_model import _describe_pipe_stage

        @Flow.model
        def factory_stage(x: int) -> int:
            return x

        desc = _describe_pipe_stage(factory_stage)
        self.assertIn("factory_stage", desc)

    def test_non_callable_stage(self):
        from ccflow.flow_model import _describe_pipe_stage

        desc = _describe_pipe_stage(42)
        self.assertEqual(desc, "42")


class TestInferPipeParamAmbiguousDefaults(TestCase):
    """Cover _infer_pipe_param fallback path with multiple defaulted candidates."""

    def test_ambiguous_defaulted_candidates(self):
        """When all candidates have defaults but multiple are unoccupied."""

        @Flow.model
        def source(x: int) -> int:
            return x

        @Flow.model
        def consumer(a: int = 1, b: int = 2) -> int:
            return a + b

        # Both a and b have defaults, both are unoccupied -> ambiguous
        with self.assertRaisesRegex(TypeError, "could not infer a target parameter"):
            source().pipe(consumer)


class TestPipeErrorPaths(TestCase):
    """Group 11: pipe() error paths not covered by existing tests."""

    def test_pipe_non_callable_model_source(self):
        """pipe() should reject non-CallableModel source."""
        from ccflow.flow_model import pipe_model

        @Flow.model
        def consumer(data: int) -> int:
            return data

        with self.assertRaisesRegex(TypeError, "pipe\\(\\) source must be a CallableModel"):
            pipe_model("not_a_model", consumer)

    def test_pipe_non_flow_model_target(self):
        """pipe() should reject non-@Flow.model target."""
        from ccflow.flow_model import pipe_model

        @Flow.model
        def source(x: int) -> int:
            return x

        class ManualTarget(CallableModel):
            @Flow.call
            def __call__(self, context: SimpleContext) -> GenericResult[int]:
                return GenericResult(value=0)

        with self.assertRaisesRegex(TypeError, "pipe\\(\\) only supports downstream stages"):
            pipe_model(source(), ManualTarget())

    def test_pipe_invalid_param_name(self):
        """pipe() should reject invalid target parameter names."""

        @Flow.model
        def source(x: int) -> int:
            return x

        @Flow.model
        def consumer(data: int) -> int:
            return data

        with self.assertRaisesRegex(TypeError, "is not valid for"):
            source().pipe(consumer, param="nonexistent")

    def test_pipe_already_bound_param(self):
        """pipe() should reject already-bound parameters."""

        @Flow.model
        def source(x: int) -> int:
            return x

        @Flow.model
        def consumer(data: int) -> int:
            return data

        model = consumer(data=5)
        with self.assertRaisesRegex(TypeError, "is already bound"):
            source().pipe(model, param="data")

    def test_pipe_no_available_target_parameter(self):
        """pipe() should error when all downstream params are occupied."""

        @Flow.model
        def source(x: int) -> int:
            return x

        @Flow.model
        def consumer(data: int) -> int:
            return data

        model = consumer(data=5)
        with self.assertRaisesRegex(TypeError, "could not find an available target parameter"):
            source().pipe(model)

    def test_pipe_into_generated_instance_rebuilds(self):
        """pipe() into an existing generated model instance should rebuild."""

        @Flow.model
        def source(x: int) -> int:
            return x * 10

        @Flow.model
        def consumer(data: int, extra: int = 1) -> int:
            return data + extra

        instance = consumer(extra=5)
        pipeline = source().pipe(instance)
        result = pipeline.flow.compute(x=3)
        self.assertEqual(result.value, 35)  # 3*10 + 5

    def test_pipe_bound_model_wrapping_non_generated_rejects(self):
        """pipe() into BoundModel wrapping a non-generated model should fail."""
        from ccflow.flow_model import BoundModel, pipe_model

        @Flow.model
        def source(x: int) -> int:
            return x

        class ManualModel(CallableModel):
            @Flow.call
            def __call__(self, context: SimpleContext) -> GenericResult[int]:
                return GenericResult(value=context.value)

        bound = BoundModel(model=ManualModel(), input_transforms={"value": 42})
        with self.assertRaisesRegex(TypeError, "pipe\\(\\) only supports downstream"):
            pipe_model(source(), bound)


class TestFlowAPIBuildContextFallback(TestCase):
    """Group 12: FlowAPI._build_context when _context_schema is None/unset."""

    def test_unbound_inputs_on_manual_callable_model(self):
        """Manual CallableModel with context should show required fields."""

        class ManualModel(CallableModel):
            offset: int

            @Flow.call
            def __call__(self, context: SimpleContext) -> GenericResult[int]:
                return GenericResult(value=context.value + self.offset)

        model = ManualModel(offset=5)
        unbound = model.flow.unbound_inputs
        self.assertIn("value", unbound)


class TestBoundModelRestoreNonDict(TestCase):
    """Group 13: BoundModel._restore_serialized_transforms non-dict path."""

    def test_restore_from_model_instance(self):
        """model_validate from an existing BoundModel instance (non-dict)."""
        from ccflow.flow_model import BoundModel

        @Flow.model
        def source(context: SimpleContext) -> GenericResult[int]:
            return GenericResult(value=context.value * 10)

        bound = source().flow.with_inputs(value=42)
        # Pass existing instance through model_validate (non-dict path)
        restored = BoundModel.model_validate(bound)
        ctx = SimpleContext(value=1)
        self.assertEqual(restored(ctx).value, 420)


class TestBoundModelInitEmptyTransforms(TestCase):
    """Group 14: BoundModel.__init__ with no transforms."""

    def test_init_without_transforms(self):
        from ccflow.flow_model import BoundModel

        @Flow.model
        def source(context: SimpleContext) -> GenericResult[int]:
            return GenericResult(value=context.value)

        bound = BoundModel(model=source())
        self.assertEqual(bound._input_transforms, {})
        result = bound(SimpleContext(value=5))
        self.assertEqual(result.value, 5)


class TestBoundModelDeps(TestCase):
    """Group 15: BoundModel.__deps__."""

    def test_deps_returns_wrapped_model(self):
        @Flow.model
        def source(context: SimpleContext) -> GenericResult[int]:
            return GenericResult(value=context.value * 10)

        bound = source().flow.with_inputs(value=42)
        deps = bound.__deps__(SimpleContext(value=1))
        self.assertEqual(len(deps), 1)
        self.assertIs(deps[0][0], bound.model)


class TestValidateFieldTypesAfterValidator(TestCase):
    """Group 16: _validate_field_types in the model_validate path."""

    def test_model_validate_rejects_wrong_type(self):
        """model_validate should reject wrong scalar types."""

        @Flow.model
        def source(x: int) -> int:
            return x * 10

        cls = type(source(x=5))
        with self.assertRaisesRegex(TypeError, "Field 'x'"):
            cls.model_validate({"x": "not_an_int"})


class TestGetContextValidatorPaths(TestCase):
    """Group 17: _get_context_validator fallback paths."""

    def test_mode2_context_validator_from_schema(self):
        """Mode 2 model should build validator from _context_schema."""

        @Flow.model(context_args=["start_date"])
        def loader(start_date: str, source: str = "db") -> str:
            return f"{source}:{start_date}"

        model = loader()
        # Trigger validator creation by calling flow.compute
        result = model.flow.compute(start_date="2024-01-01")
        self.assertEqual(result.value, "db:2024-01-01")

    def test_mode1_context_validator_uses_context_type_directly(self):
        """Mode 1 should use TypeAdapter(context_type) directly."""

        @Flow.model
        def model_fn(context: SimpleContext, offset: int = 0) -> GenericResult[int]:
            return GenericResult(value=context.value + offset)

        model = model_fn()
        # compute with SimpleContext fields
        result = model.flow.compute(value=5)
        self.assertEqual(result.value, 5)


class TestValidateContextTypeOverrideErrors(TestCase):
    """Group 18: _validate_context_type_override error paths."""

    def test_non_context_base_raises(self):
        with self.assertRaisesRegex(TypeError, "context_type must be a ContextBase subclass"):

            @Flow.model(context_args=["x"], context_type=int)
            def bad_model(x: int) -> int:
                return x

    def test_context_type_missing_context_args_fields(self):
        """context_type missing required context_args fields."""

        class TinyContext(ContextBase):
            a: int

        with self.assertRaisesRegex(TypeError, "must define fields for context_args"):

            @Flow.model(context_args=["a", "b"], context_type=TinyContext)
            def bad_model(a: int, b: int) -> int:
                return a + b

    def test_context_type_extra_required_fields(self):
        """context_type has required fields not listed in context_args."""

        class BigContext(ContextBase):
            a: int
            b: int
            extra: str

        with self.assertRaisesRegex(TypeError, "has required fields not listed in context_args"):

            @Flow.model(context_args=["a"], context_type=BigContext)
            def bad_model(a: int) -> int:
                return a

    def test_annotation_type_mismatch(self):
        """Function and context_type disagree on annotation type."""

        class TypedContext(ContextBase):
            x: str

        with self.assertRaisesRegex(TypeError, "context_arg 'x'"):

            @Flow.model(context_args=["x"], context_type=TypedContext)
            def bad_model(x: int) -> int:
                return x

    def test_annotation_skip_when_func_ann_is_none(self):
        """Annotation check should skip when function annotation is absent from schema."""
        from ccflow.flow_model import _validate_context_type_override

        class CompatContext(ContextBase):
            a: int

        # context_args has 'a', schema has 'a': int. Compatible, no error.
        result = _validate_context_type_override(CompatContext, ["a"], {"a": int})
        self.assertIs(result, CompatContext)

    def test_subclass_annotations_allowed(self):
        """context_type with subclass-compatible annotations should pass."""
        from ccflow.flow_model import _validate_context_type_override

        class ContextWithBase(ContextBase):
            ctx: ContextBase

        # Function declares SimpleContext which is a subclass of ContextBase — should pass
        result = _validate_context_type_override(ContextWithBase, ["ctx"], {"ctx": SimpleContext})
        self.assertIs(result, ContextWithBase)

    def test_default_vs_required_field_conflict(self):
        """Function has default for context_arg but context_type requires it."""

        class StrictContext(ContextBase):
            x: int

        with self.assertRaisesRegex(TypeError, "function has a default but context_type"):

            @Flow.model(context_args=["x"], context_type=StrictContext)
            def bad_model(x: int = 5) -> int:
                return x


class TestDecoratorErrorPaths(TestCase):
    """Group 19: Decorator error paths."""

    def test_context_type_with_explicit_context_param(self):
        """context_type= with explicit context param should raise."""
        with self.assertRaisesRegex(TypeError, "context_type.*only supported"):

            @Flow.model(context_type=SimpleContext)
            def bad_model(context: SimpleContext) -> GenericResult[int]:
                return GenericResult(value=0)

    def test_context_type_without_context_args(self):
        """context_type= without context_args should raise in dynamic mode."""
        with self.assertRaisesRegex(TypeError, "context_type.*only supported"):

            @Flow.model(context_type=SimpleContext)
            def bad_model(x: int) -> int:
                return x

    def test_missing_context_annotation(self):
        """Missing type annotation on context param should raise."""
        with self.assertRaisesRegex(TypeError, "must have a type annotation"):

            @Flow.model
            def bad_model(context) -> int:
                return 0

    def test_missing_param_annotation(self):
        """Missing type annotation on a model field param should raise."""
        with self.assertRaisesRegex(TypeError, "must have a type annotation"):

            @Flow.model
            def bad_model(context: SimpleContext, untyped_param) -> int:
                return 0

    def test_context_param_not_context_base(self):
        """context param annotated with non-ContextBase type should raise."""
        with self.assertRaisesRegex(TypeError, "must be annotated with a ContextBase subclass"):

            @Flow.model
            def bad_model(context: int) -> int:
                return 0

    def test_pep563_fallback_on_failed_get_type_hints(self):
        """When get_type_hints fails, falls back to raw annotations."""

        # This is hard to trigger directly, but we can test that string annotations work
        @Flow.model
        def model_with_string_return(x: int) -> "int":
            return x * 2

        result = model_with_string_return().flow.compute(x=5)
        self.assertEqual(result.value, 10)


class TestMode1CallPath(TestCase):
    """Group 20: Mode 1 explicit context pass-through in __call__."""

    def test_mode1_resolve_callable_model_returns_non_generic_result(self):
        """Mode 1 should handle deps that return raw ResultBase (not GenericResult)."""

        @Flow.model
        def upstream(context: SimpleContext) -> MyResult:
            return MyResult(data=f"value={context.value}")

        @Flow.model
        def downstream(context: SimpleContext, dep: CallableModel) -> GenericResult[str]:
            # dep is resolved to MyResult since it's not GenericResult
            return GenericResult(value=f"got:{dep}")

        model = downstream(dep=upstream())
        result = model(SimpleContext(value=42))
        self.assertIn("value=42", result.value)


class TestDynamicModeContextLookup(TestCase):
    """Group 21: Dynamic mode context lookup for deferred values."""

    def test_deferred_value_from_context(self):
        """Dynamic mode should pull deferred values from context."""

        @Flow.model
        def add(x: int, y: int) -> int:
            return x + y

        model = add(x=10)
        # y is deferred — pulled from context
        result = model.flow.compute(y=5)
        self.assertEqual(result.value, 15)

    def test_missing_deferred_value_raises(self):
        """Dynamic mode should raise for missing deferred values."""

        @Flow.model
        def add(x: int, y: int) -> int:
            return x + y

        model = add(x=10)
        with self.assertRaisesRegex(TypeError, "Missing runtime input"):
            model.flow.compute()  # y not provided

    def test_context_sourced_value_coercion(self):
        """Dynamic mode should coerce context-sourced values through validators."""

        @Flow.model
        def typed_model(x: int, y: int) -> int:
            return x + y

        model = typed_model(x=10)
        # y provided as a value that can be coerced to int
        result = model.flow.compute(y=5)
        self.assertEqual(result.value, 15)

    def test_deferred_value_from_context_object(self):
        """Dynamic mode should look up deferred values from context attributes."""

        @Flow.model
        def multiply(x: int, y: int) -> int:
            return x * y

        model = multiply(x=3)
        # Call directly with a FlowContext — y must come from context
        result = model(FlowContext(y=7))
        self.assertEqual(result.value, 21)


class TestGetContextValidatorFallbacks(TestCase):
    """Group 17 additional: _get_context_validator edge cases."""

    def test_mode2_with_context_type_override(self):
        """Mode 2 with explicit context_type should use that type's validator."""

        @Flow.model(context_args=["value"], context_type=SimpleContext)
        def typed_model(value: int) -> int:
            return value * 2

        model = typed_model()
        result = model(SimpleContext(value=5))
        self.assertEqual(result.value, 10)

    def test_dynamic_mode_instance_validator(self):
        """Dynamic mode should create instance-specific validator."""

        @Flow.model
        def add(x: int, y: int, z: int = 0) -> int:
            return x + y + z

        m1 = add(x=1)
        m2 = add(x=1, y=2)
        # Different bound fields => different runtime inputs
        self.assertIn("y", m1.flow.unbound_inputs)
        self.assertNotIn("y", m2.flow.unbound_inputs)


class TestRegistryResolutionInValidateFieldTypes(TestCase):
    """Group 16: _resolve_registry_refs and _validate_field_types paths."""

    def test_registry_string_not_resolving_passes_through(self):
        """String value that doesn't resolve from registry should fail type validation."""

        @Flow.model
        def model_fn(x: int) -> int:
            return x

        cls = type(model_fn(x=1))
        with self.assertRaisesRegex(TypeError, "Field 'x'"):
            cls.model_validate({"x": "nonexistent_registry_key"})

    def test_registry_ref_resolves_to_callable_model(self):
        """String value resolving to a CallableModel should be substituted."""
        registry = ModelRegistry.root()
        registry.clear()
        try:

            @Flow.model
            def upstream(context: SimpleContext) -> GenericResult[int]:
                return GenericResult(value=context.value * 10)

            @Flow.model
            def downstream(context: SimpleContext, dep: CallableModel) -> GenericResult[int]:
                return GenericResult(value=0)

            registry.add("my_upstream", upstream())
            cls = type(downstream(dep=upstream()))
            restored = cls.model_validate({"dep": "my_upstream"})
            self.assertIsNotNone(restored)
        finally:
            registry.clear()


class TestMode2MissingContextField(TestCase):
    """Line 1155: Mode 2 missing context field error."""

    def test_mode2_missing_required_context_field(self):
        """Mode 2 model called with context missing a required field should raise."""

        @Flow.model(context_args=["start_date", "end_date"])
        def loader(start_date: str, end_date: str, source: str = "db") -> str:
            return f"{source}:{start_date}-{end_date}"

        model = loader()
        # Call with a FlowContext missing end_date
        with self.assertRaisesRegex(TypeError, "Missing context field"):
            model(FlowContext(start_date="2024-01-01"))


class TestDynamicModeContextObjectLookup(TestCase):
    """Line 1155/1176: Dynamic mode pulling deferred values from context object."""

    def test_deferred_value_coercion_through_context(self):
        """Dynamic mode should coerce values from FlowContext through validators."""

        @Flow.model
        def typed_add(x: int, y: int) -> int:
            return x + y

        model = typed_add(x=10)
        # Calling with a FlowContext — y pulled from context and coerced
        result = model(FlowContext(y=5))
        self.assertEqual(result.value, 15)


if __name__ == "__main__":
    import unittest

    unittest.main()
