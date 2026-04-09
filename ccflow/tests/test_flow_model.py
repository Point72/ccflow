"""Focused tests for the FromContext-based Flow.model API."""

import graphlib
from datetime import date, timedelta
from typing import Annotated

import pytest
from pydantic import model_validator
from ray.cloudpickle import dumps as rcpdumps, loads as rcploads

import ccflow.flow_model as flow_model_module
from ccflow import (
    CallableModel,
    ContextBase,
    DateRangeContext,
    Flow,
    FlowContext,
    FlowOptionsOverride,
    FromContext,
    GenericResult,
    Lazy,
    ModelRegistry,
)
from ccflow.evaluators import GraphEvaluator


class SimpleContext(ContextBase):
    value: int


class ParentRangeContext(ContextBase):
    start_date: date
    end_date: date


class RichRangeContext(ParentRangeContext):
    label: str = "child"


class OrderedContext(ContextBase):
    a: int
    b: int

    @model_validator(mode="after")
    def _validate_order(self):
        if self.a > self.b:
            raise ValueError("a must be <= b")
        return self


@Flow.model
def basic_loader(context: SimpleContext, source: str, multiplier: int) -> GenericResult[int]:
    return GenericResult(value=context.value * multiplier)


@Flow.model
def string_processor(context: SimpleContext, prefix: str = "value=", suffix: str = "!") -> GenericResult[str]:
    return GenericResult(value=f"{prefix}{context.value}{suffix}")


@Flow.model
def data_source(context: SimpleContext, base_value: int) -> GenericResult[int]:
    return GenericResult(value=context.value + base_value)


@Flow.model
def data_transformer(context: SimpleContext, source: int, factor: int) -> GenericResult[int]:
    return GenericResult(value=source * factor)


@Flow.model
def data_aggregator(context: SimpleContext, input_a: int, input_b: int, operation: str = "add") -> GenericResult[int]:
    if operation == "add":
        return GenericResult(value=input_a + input_b)
    raise ValueError(f"unsupported operation: {operation}")


@Flow.model
def pipeline_stage1(context: SimpleContext, initial: int) -> GenericResult[int]:
    return GenericResult(value=context.value + initial)


@Flow.model
def pipeline_stage2(context: SimpleContext, stage1_output: int, multiplier: int) -> GenericResult[int]:
    return GenericResult(value=stage1_output * multiplier)


@Flow.model
def pipeline_stage3(context: SimpleContext, stage2_output: int, offset: int) -> GenericResult[int]:
    return GenericResult(value=stage2_output + offset)


@Flow.model
def date_range_loader_previous_day(
    source: str,
    start_date: FromContext[date],
    end_date: FromContext[date],
    include_weekends: bool = False,
) -> GenericResult[dict]:
    del include_weekends
    return GenericResult(
        value={
            "source": source,
            "start_date": str(start_date - timedelta(days=1)),
            "end_date": str(end_date),
        }
    )


@Flow.model
def date_range_processor(context: DateRangeContext, raw_data: dict, normalize: bool = False) -> GenericResult[str]:
    prefix = "normalized:" if normalize else "raw:"
    return GenericResult(value=f"{prefix}{raw_data['source']}:{raw_data['start_date']} to {raw_data['end_date']}")


@Flow.model
def contextual_loader(source: str, start_date: FromContext[date], end_date: FromContext[date]) -> GenericResult[dict]:
    return GenericResult(
        value={
            "source": source,
            "start_date": str(start_date),
            "end_date": str(end_date),
        }
    )


@Flow.model
def contextual_processor(
    prefix: str,
    data: dict,
    start_date: FromContext[date],
    end_date: FromContext[date],
) -> GenericResult[str]:
    del start_date, end_date
    return GenericResult(value=f"{prefix}:{data['source']}:{data['start_date']} to {data['end_date']}")


def test_from_context_anchor_behavior():
    @Flow.model
    def foo(a: int, b: FromContext[int]) -> int:
        return a + b

    assert foo(a=11).flow.compute(b=12).value == 23
    assert foo(a=11, b=12).flow.compute().value == 23

    with pytest.raises(TypeError, match="compute\\(\\) only accepts contextual inputs"):
        foo().flow.compute(a=11, b=12)


def test_regular_param_accepts_upstream_model():
    @Flow.model
    def source(value: FromContext[int], offset: int) -> int:
        return value + offset

    @Flow.model
    def foo(a: int, b: FromContext[int]) -> int:
        return a + b

    model = foo(a=source(offset=5))
    assert model.flow.compute(FlowContext(value=7, b=12)).value == 24


def test_contextual_param_rejects_callable_model():
    @Flow.model
    def source(context: SimpleContext, offset: int) -> GenericResult[int]:
        return GenericResult(value=context.value + offset)

    @Flow.model
    def foo(a: int, b: FromContext[int]) -> int:
        return a + b

    with pytest.raises(TypeError, match="cannot be bound to a CallableModel"):
        foo(a=1, b=source(offset=2))


def test_contextual_construction_defaults_and_bound_inputs():
    @Flow.model
    def foo(a: int, b: FromContext[int]) -> int:
        return a + b

    model = foo(a=11, b=12)
    assert model.flow.bound_inputs == {"a": 11, "b": 12}
    assert model.flow.context_inputs == {"b": int}
    assert model.flow.unbound_inputs == {}
    assert model.flow.compute().value == 23


def test_contextual_function_defaults_remain_contextual():
    @Flow.model
    def foo(a: int, b: FromContext[int] = 5) -> int:
        return a + b

    model = foo(a=2)
    assert model.flow.bound_inputs == {"a": 2}
    assert model.flow.context_inputs == {"b": int}
    assert model.flow.unbound_inputs == {}
    assert model.flow.compute().value == 7
    assert model.flow.compute(b=10).value == 12


def test_context_type_accepts_richer_subclass_for_from_context():
    @Flow.model(context_type=ParentRangeContext)
    def span_days(multiplier: int, start_date: FromContext[date], end_date: FromContext[date]) -> int:
        return multiplier * ((end_date - start_date).days + 1)

    model = span_days(multiplier=2)
    assert model.flow.compute(start_date="2024-01-01", end_date="2024-01-03").value == 6
    assert model(RichRangeContext(start_date=date(2024, 1, 1), end_date=date(2024, 1, 4), label="x")).value == 8


def test_context_type_validation_applies_to_resolved_contextual_values():
    @Flow.model(context_type=OrderedContext)
    def add(a: FromContext[int], b: FromContext[int]) -> int:
        return a + b

    with pytest.raises(ValueError, match="a must be <= b"):
        add().flow.compute(a=2, b=1)

    with pytest.raises(ValueError, match="a must be <= b"):
        add(a=2, b=1).flow.compute()


def test_explicit_context_interop_accepts_pep604_optional_annotation():
    @Flow.model
    def loader(context: DateRangeContext | None, source: str = "db") -> GenericResult[str]:
        assert context is not None
        return GenericResult(value=f"{source}:{context.start_date}:{context.end_date}")

    model = loader(source="api")
    assert model.flow.compute(start_date="2024-01-01", end_date="2024-01-02").value == "api:2024-01-01:2024-01-02"


def test_explicit_context_interop_still_works():
    @Flow.model
    def loader(context: DateRangeContext, source: str = "db") -> GenericResult[str]:
        return GenericResult(value=f"{source}:{context.start_date}:{context.end_date}")

    model = loader(source="api")
    assert model(DateRangeContext(start_date=date(2024, 1, 1), end_date=date(2024, 1, 2))).value == "api:2024-01-01:2024-01-02"
    assert model.flow.compute(start_date="2024-01-01", end_date="2024-01-02").value == "api:2024-01-01:2024-01-02"


def test_auto_unwrap_defaults_to_false_for_auto_wrapped_results():
    @Flow.model
    def add(a: int, b: FromContext[int]) -> int:
        return a + b

    result = add(a=10).flow.compute(b=5)
    assert isinstance(result, GenericResult)
    assert result.value == 15


def test_compute_does_not_unwrap_explicit_generic_result_returns():
    @Flow.model
    def load(value: FromContext[int]) -> GenericResult[int]:
        return GenericResult(value=value * 2)

    result = load().flow.compute(value=3)
    assert isinstance(result, GenericResult)
    assert result.value == 6


def test_auto_unwrap_can_be_enabled_for_auto_wrapped_results():
    @Flow.model(auto_unwrap=True)
    def add(a: int, b: FromContext[int]) -> int:
        return a + b

    assert add(a=10).flow.compute(b=5) == 15


def test_auto_unwrap_only_affects_external_compute_results():
    @Flow.model
    def source(value: FromContext[int]) -> int:
        return value * 2

    @Flow.model(auto_unwrap=True)
    def add(left: int, bonus: FromContext[int]) -> int:
        return left + bonus

    model = add(left=source())
    assert model.flow.compute(FlowContext(value=4, bonus=3)) == 11


def test_model_base_allows_custom_callable_model_subclass():
    class CustomFlowBase(CallableModel):
        multiplier: int = 1

        @model_validator(mode="after")
        def _validate_multiplier(self):
            if self.multiplier <= 0:
                raise ValueError("multiplier must be positive")
            return self

        def scaled(self, value: int) -> int:
            return value * self.multiplier

        @Flow.call
        def __call__(self, context: SimpleContext) -> GenericResult[int]:
            return GenericResult(value=context.value)

    @Flow.model(model_base=CustomFlowBase)
    def add(a: int, b: FromContext[int]) -> int:
        return a + b

    model = add(a=10, multiplier=3)
    assert isinstance(model, CustomFlowBase)
    assert model.multiplier == 3
    assert model.scaled(4) == 12
    assert model.flow.compute(b=5).value == 15

    with pytest.raises(ValueError, match="multiplier must be positive"):
        add(a=10, multiplier=0)


def test_model_base_must_be_callable_model_subclass():
    with pytest.raises(TypeError, match="model_base must be a CallableModel subclass"):

        @Flow.model(model_base=int)
        def add(a: int, b: FromContext[int]) -> int:
            return a + b


def test_explicit_context_and_from_context_cannot_mix():
    with pytest.raises(TypeError, match="cannot also declare FromContext"):

        @Flow.model
        def bad(context: SimpleContext, y: FromContext[int]) -> int:
            return context.value + y


def test_context_args_keyword_is_removed():
    with pytest.raises(TypeError, match="context_args=... has been removed"):

        @Flow.model(context_args=["x"])
        def bad(x: int) -> int:
            return x


def test_context_type_requires_from_context_or_explicit_context():
    with pytest.raises(TypeError, match="context_type=... requires FromContext"):

        @Flow.model(context_type=DateRangeContext)
        def bad(x: int) -> int:
            return x


def test_lazy_dependency_remains_lazy():
    calls = {"source": 0}

    @Flow.model
    def source(value: FromContext[int]) -> int:
        calls["source"] += 1
        return value * 10

    @Flow.model
    def choose(value: int, lazy_value: Lazy[int], threshold: FromContext[int]) -> int:
        if value > threshold:
            return value
        return lazy_value()

    eager = choose(value=50, lazy_value=source())
    assert eager.flow.compute(FlowContext(value=3, threshold=10)).value == 50
    assert calls["source"] == 0

    deferred = choose(value=5, lazy_value=source())
    assert deferred.flow.compute(FlowContext(value=3, threshold=10)).value == 30
    assert calls["source"] == 1


def test_lazy_runtime_helper_is_removed():
    @Flow.model
    def source(context: SimpleContext) -> GenericResult[int]:
        return GenericResult(value=context.value)

    with pytest.raises(TypeError, match="Lazy\\(model\\)\\(\\.\\.\\.\\) has been removed"):
        Lazy(source())


def test_lazy_and_from_context_combination_is_rejected():
    with pytest.raises(TypeError, match="cannot combine Lazy"):

        @Flow.model
        def bad(x: Lazy[FromContext[int]]) -> int:
            return x()


def test_auto_wrap_and_serialization_roundtrip():
    @Flow.model
    def add(a: int, b: FromContext[int]) -> int:
        return a + b

    model = add(a=10)
    dumped = model.model_dump(mode="python")
    restored = type(model).model_validate(dumped)

    assert restored.flow.bound_inputs == {"a": 10}
    assert restored.flow.unbound_inputs == {"b": int}
    assert restored.flow.compute(b=5).value == 15


def test_generated_models_cloudpickle_roundtrip():
    @Flow.model
    def multiply(a: int, b: FromContext[int]) -> int:
        return a * b

    model = multiply(a=6)
    restored = rcploads(rcpdumps(model, protocol=5))
    assert restored.flow.compute(b=7).value == 42


def test_graph_integration_fanout_fanin():
    @Flow.model
    def source(base: int, value: FromContext[int]) -> int:
        return value + base

    @Flow.model
    def scale(data: int, factor: int) -> int:
        return data * factor

    @Flow.model
    def merge(left: int, right: int, bonus: FromContext[int]) -> int:
        return left + right + bonus

    src = source(base=10)
    left = scale(data=src, factor=2)
    right = scale(data=src, factor=5)
    model = merge(left=left, right=right)

    assert model.flow.compute(FlowContext(value=3, bonus=7)).value == ((3 + 10) * 2) + ((3 + 10) * 5) + 7


def test_graph_integration_cycle_raises_cleanly():
    @Flow.model
    def increment(x: int, n: FromContext[int]) -> int:
        return x + n

    root = increment()
    branch = increment(x=root)
    object.__setattr__(root, "x", branch)

    with FlowOptionsOverride(options={"evaluator": GraphEvaluator()}):
        with pytest.raises(graphlib.CycleError):
            root.flow.compute(n=1)


def test_large_contextual_contract_stress():
    @Flow.model
    def total(
        base: int,
        x1: FromContext[int],
        x2: FromContext[int],
        x3: FromContext[int],
        x4: FromContext[int],
        x5: FromContext[int],
        x6: FromContext[int],
    ) -> int:
        return base + x1 + x2 + x3 + x4 + x5 + x6

    model = total(base=10)
    assert model.flow.context_inputs == {"x1": int, "x2": int, "x3": int, "x4": int, "x5": int, "x6": int}
    assert model.flow.compute(x1=1, x2=2, x3=3, x4=4, x5=5, x6=6).value == 31


def test_registry_integration_for_generated_models():
    registry = ModelRegistry.root().clear()
    model = basic_loader(source="warehouse", multiplier=3)
    registry.add("loader", model)

    retrieved = registry["loader"]
    assert isinstance(retrieved, CallableModel)
    assert retrieved(SimpleContext(value=4)).value == 12


def test_unexpected_type_adapter_errors_are_not_silently_swallowed():
    class BrokenSchema:
        @classmethod
        def __get_pydantic_core_schema__(cls, source, handler):
            raise RuntimeError("boom")

    @Flow.model
    def bad(x: BrokenSchema, y: FromContext[int]) -> int:
        del x, y
        return 0

    with pytest.raises(RuntimeError, match="boom"):
        bad(x=object())


def test_unexpected_type_hint_resolution_errors_propagate(monkeypatch):
    def broken_get_type_hints(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(flow_model_module, "get_type_hints", broken_get_type_hints)

    def add(x: int) -> int:
        return x

    with pytest.raises(RuntimeError, match="boom"):
        Flow.model(add)


def test_internal_generated_model_helpers_and_config_properties():
    @Flow.model
    def add(a: int, b: FromContext[int]) -> int:
        return a + b

    generated_cls = flow_model_module._generated_model_class(add)
    assert generated_cls is not None

    config = generated_cls.__flow_model_config__
    assert config.regular_param_names == ("a",)
    assert config.contextual_param_names == ("b",)
    assert config.param("a").name == "a"
    assert config.param("b").name == "b"

    with pytest.raises(KeyError):
        config.param("missing")

    model = add(a=10)
    assert flow_model_module._generated_model_class(model) is generated_cls

    class DerivedGeneratedBase(flow_model_module._GeneratedFlowModelBase):
        @Flow.call
        def __call__(self, context: SimpleContext) -> GenericResult[int]:
            return GenericResult(value=context.value)

        @Flow.deps
        def __deps__(self, context: SimpleContext):
            del context
            return []

    assert flow_model_module._resolve_generated_model_bases(DerivedGeneratedBase) == (DerivedGeneratedBase,)


def test_internal_type_helpers_and_plain_callable_flow_api_paths():
    assert flow_model_module._concrete_context_type(SimpleContext | None) is SimpleContext
    assert flow_model_module._concrete_context_type(int) is None
    assert flow_model_module._type_accepts_str(Annotated[str, flow_model_module._FromContextMarker()]) is True
    assert flow_model_module._type_accepts_str(int | None) is False
    assert flow_model_module._transform_repr(lambda value: value) == "<lambda>"
    assert flow_model_module._bound_field_names(object()) == set()

    class PlainModel(CallableModel):
        @property
        def context_type(self):
            return SimpleContext

        @property
        def result_type(self):
            return GenericResult[int]

        @Flow.call
        def __call__(self, context: SimpleContext) -> GenericResult[int]:
            return GenericResult(value=context.value)

        @Flow.deps
        def __deps__(self, context: SimpleContext):
            del context
            return []

    model = PlainModel()

    assert model.flow.context_inputs == {"value": int}
    assert model.flow.unbound_inputs == {"value": int}
    assert model.flow.bound_inputs == {}
    assert model.flow.compute({"value": 3}).value == 3

    with pytest.raises(TypeError, match="either one context object or contextual keyword arguments"):
        model.flow.compute(SimpleContext(value=1), value=2)


def test_explicit_context_paths_and_underbar_context_parameter():
    model = basic_loader(source="warehouse", multiplier=3)

    assert model.flow.context_inputs == {"value": int}
    assert model.flow.unbound_inputs == {"value": int}
    assert model.flow.compute({"value": 4}).value == 12
    assert model.flow.compute(SimpleContext(value=5)).value == 15

    with pytest.raises(TypeError, match="either one context object or contextual keyword arguments"):
        model.flow.compute(SimpleContext(value=1), value=2)

    @Flow.model
    def underbar(_: SimpleContext, a: int) -> int:
        return _.value + a

    assert underbar(a=10).flow.compute({"value": 2}).value == 12


def test_additional_validation_and_hint_fallback_paths(monkeypatch):
    class MissingFieldContext(ContextBase):
        start_date: date

    with pytest.raises(TypeError, match="must define fields for all FromContext parameters"):

        @Flow.model(context_type=MissingFieldContext)
        def bad_missing(start_date: FromContext[date], end_date: FromContext[date]) -> int:
            return 0

    class ExtraRequiredContext(ContextBase):
        start_date: date
        end_date: date
        label: str

    with pytest.raises(TypeError, match="has required fields that are not declared as FromContext parameters"):

        @Flow.model(context_type=ExtraRequiredContext)
        def bad_extra(start_date: FromContext[date], end_date: FromContext[date]) -> int:
            return 0

    class BadAnnotationContext(ContextBase):
        value: str

    with pytest.raises(TypeError, match="annotates"):

        @Flow.model(context_type=BadAnnotationContext)
        def bad_annotation(value: FromContext[int]) -> int:
            return value

    with pytest.raises(TypeError, match="context_type must be a ContextBase subclass"):

        @Flow.model(context_type=int)
        def bad_context_type(value: FromContext[int]) -> int:
            return value

    @Flow.model
    def source(context: SimpleContext) -> GenericResult[int]:
        return GenericResult(value=context.value)

    with pytest.raises(TypeError, match="cannot default to a CallableModel"):

        @Flow.model
        def bad_default(value: FromContext[int] = source()) -> int:
            return value

    with pytest.raises(TypeError, match="return type annotation"):

        @Flow.model
        def missing_return(value: int):
            return value

    with pytest.raises(TypeError, match="does not support positional-only parameter 'value'"):

        @Flow.model
        def bad_positional_only(value: int, /, bonus: FromContext[int]) -> int:
            return value + bonus

    with pytest.raises(TypeError, match="does not support variadic positional parameter 'values'"):

        @Flow.model
        def bad_varargs(*values: int) -> int:
            return sum(values)

    with pytest.raises(TypeError, match="does not support variadic keyword parameter 'values'"):

        @Flow.model
        def bad_varkw(**values: int) -> int:
            return sum(values.values())

    @Flow.model
    def keyword_only(value: int, *, bonus: FromContext[int]) -> int:
        return value + bonus

    assert keyword_only(value=2).flow.compute(bonus=3).value == 5

    @Flow.model
    def keyword_only_context(*, context: SimpleContext, offset: int) -> int:
        return context.value + offset

    assert keyword_only_context(offset=4).flow.compute({"value": 3}).value == 7

    def missing_hints(*args, **kwargs):
        raise AttributeError("missing hints")

    monkeypatch.setattr(flow_model_module, "get_type_hints", missing_hints)

    @Flow.model
    def add(x: int, y: FromContext[int]) -> int:
        return x + y

    assert add(x=1).flow.compute(y=2).value == 3
