"""Focused tests for the FromContext-based Flow.model API."""

import base64
import graphlib
import importlib
import inspect
import pickle
import subprocess
import sys
from datetime import date, timedelta
from pathlib import Path
from types import ModuleType
from typing import Annotated, Any, Callable, Literal, Optional, get_args

import cloudpickle
import pytest
from pydantic import BaseModel as PydanticBaseModel, Field, PrivateAttr, ValidationError, model_validator

import ccflow
import ccflow._flow_model_binding as binding_module
import ccflow.flow_model as flow_model_module
from ccflow import (
    BaseModel,
    CallableModel,
    ContextBase,
    DateRangeContext,
    Dep,
    EvaluatorBase,
    Flow,
    FlowContext,
    FlowOptionsOverride,
    FromContext,
    GenericResult,
    Lazy,
    ModelEvaluationContext,
    ModelRegistry,
)
from ccflow.callable import FlowOptions
from ccflow.evaluators import GraphEvaluator, LoggingEvaluator, MemoryCacheEvaluator, cache_key, combine_evaluators, get_dependency_graph
from ccflow.exttypes import PyObjectPath


class SimpleContext(ContextBase):
    value: int


class ExternalPydanticPayload(PydanticBaseModel):
    x: int
    _bonus: int = PrivateAttr(default=1)


class ExternalCcflowPayload(BaseModel):
    x: int
    _bonus: int = PrivateAttr(default=1)


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
def basic_loader(source: str, multiplier: int, value: FromContext[int]) -> GenericResult[int]:
    return GenericResult(value=value * multiplier)


@Flow.model
def string_processor(value: FromContext[int], prefix: str = "value=", suffix: str = "!") -> GenericResult[str]:
    return GenericResult(value=f"{prefix}{value}{suffix}")


@Flow.model
def data_source(base_value: int, value: FromContext[int]) -> GenericResult[int]:
    return GenericResult(value=value + base_value)


@Flow.model
def data_transformer(source: int, factor: int) -> GenericResult[int]:
    return GenericResult(value=source * factor)


@Flow.model
def data_aggregator(input_a: int, input_b: int, operation: str = "add") -> GenericResult[int]:
    if operation == "add":
        return GenericResult(value=input_a + input_b)
    raise ValueError(f"unsupported operation: {operation}")


@Flow.model
def pipeline_stage1(initial: int, value: FromContext[int]) -> GenericResult[int]:
    return GenericResult(value=value + initial)


@Flow.model
def pipeline_stage2(stage1_output: int, multiplier: int) -> GenericResult[int]:
    return GenericResult(value=stage1_output * multiplier)


@Flow.model
def pipeline_stage3(stage2_output: int, offset: int) -> GenericResult[int]:
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
def date_range_processor(raw_data: dict, normalize: bool = False) -> GenericResult[str]:
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


@Flow.context_transform
def increment_b(b: FromContext[int], amount: int) -> int:
    return b + amount


@Flow.context_transform
def shift_integer_window(start_date: FromContext[int], end_date: FromContext[int], amount: int) -> dict[str, object]:
    return {
        "start_date": start_date + amount,
        "end_date": end_date + amount,
    }


@Flow.context_transform
def bump_start_date(start_date: FromContext[int], amount: int) -> int:
    return start_date + amount


@Flow.context_transform
def annotated_start_patch(start_date: FromContext[int]) -> Annotated[dict[str, object], "meta"]:
    return {"start_date": start_date + 1}


@Flow.context_transform
def optional_start_patch(start_date: FromContext[int]) -> dict[str, object] | None:
    return {"start_date": start_date + 2}


@Flow.context_transform
def parity_bucket(raw: FromContext[int]) -> int:
    return raw % 2


@Flow.context_transform
def seed_plus_one(seed: FromContext[int]) -> int:
    return seed + 1


@Flow.context_transform
def non_idempotent_a_step(a: FromContext[int]) -> int:
    return 2 if a == 1 else 3


@Flow.context_transform
def static_bad() -> int:
    return 2


def lazy_context_transform_for_rejection(value: Lazy[int]) -> int:
    return value()


@Flow.context_transform
def static_patch() -> dict[str, object]:
    return {"a": 2}


def test_module_level_flow_model_examples_and_transforms_execute():
    assert data_aggregator(input_a=1, input_b=2, operation="add").flow.compute().value == 3
    with pytest.raises(ValueError, match="unsupported operation"):
        data_aggregator(input_a=1, input_b=2, operation="multiply").flow.compute()

    raw = date_range_loader_previous_day(source="library").flow.compute(start_date=date(2024, 1, 2), end_date=date(2024, 1, 3)).value
    assert raw == {"source": "library", "start_date": "2024-01-01", "end_date": "2024-01-03"}
    assert date_range_processor(raw_data=raw).flow.compute().value.startswith("raw:library")
    assert date_range_processor(raw_data=raw, normalize=True).flow.compute().value.startswith("normalized:library")

    contextual = contextual_loader(source="library").flow.compute(start_date=date(2024, 1, 1), end_date=date(2024, 1, 2)).value
    assert contextual_processor(prefix="p", data=contextual).flow.compute(start_date=date(2024, 1, 1), end_date=date(2024, 1, 2)).value == (
        "p:library:2024-01-01 to 2024-01-02"
    )

    assert (
        pipeline_stage3(stage2_output=pipeline_stage2(stage1_output=pipeline_stage1(initial=2), multiplier=3), offset=4).flow.compute(value=5).value
        == 25
    )

    @Flow.model
    def load(start_date: FromContext[int], end_date: FromContext[int], bucket: FromContext[int]) -> int:
        return start_date * 100 + end_date * 10 + bucket

    shifted = load().flow.with_context(
        shift_integer_window(amount=2),
        start_date=bump_start_date(amount=10),
        bucket=parity_bucket(),
    )
    assert shifted.flow.compute(start_date=1, end_date=5, raw=7).value == 1171

    @Flow.model
    def add(a: FromContext[int]) -> int:
        return a

    assert add().flow.with_context(a=non_idempotent_a_step()).flow.compute(a=1).value == 2
    assert add().flow.with_context(a=non_idempotent_a_step()).flow.compute(a=10).value == 3
    assert add().flow.with_context(static_patch()).flow.compute(a=1).value == 2


def test_flow_model_rejects_invalid_decorator_targets():
    with pytest.raises(TypeError):
        Flow.model(123)
    with pytest.raises(TypeError):
        Flow.model(lambda: None)


def test_context_transform_defaults_and_public_validation():
    @Flow.context_transform
    def default_amount(amount: int = 5) -> int:
        return amount

    @Flow.context_transform
    def default_seed(seed: FromContext[int] = 9) -> int:
        return seed + 1

    @Flow.context_transform
    def dynamic_patch(seed: FromContext[int]) -> dict[str, object]:
        return {"a": seed}

    @Flow.model
    def add(a: FromContext[int], b: FromContext[int]) -> int:
        return a + b

    assert add().flow.with_context(a=default_amount(), b=default_seed()).flow.compute().value == 15
    assert add().flow.with_context(dynamic_patch()).flow.compute(seed=1, b=2).value == 3

    with pytest.raises(TypeError, match="unexpected keyword"):
        increment_b(amount=1, extra=2)
    with pytest.raises(TypeError, match="Do not pass contextual"):
        increment_b(b=1, amount=2)
    with pytest.raises(TypeError, match="missing required regular"):
        increment_b()

    with pytest.raises(TypeError, match="Positional with_context"):
        add().flow.with_context(lambda: {"a": 1})
    with pytest.raises(TypeError, match="Positional with_context"):
        add().flow.with_context(123)


def test_plain_and_bound_optional_compute_paths():
    class OptionalContextModel(CallableModel):
        @Flow.call
        def __call__(self, context: Optional[SimpleContext] = None) -> GenericResult[int]:
            return GenericResult(value=0 if context is None else context.value)

    assert OptionalContextModel().flow.compute(None).value == 0
    assert OptionalContextModel().flow.compute().value == 0
    assert OptionalContextModel().flow.inspect().required_inputs == {}

    bound = OptionalContextModel().flow.with_context()
    assert bound.flow.compute(FlowContext(value=3)).value == 3
    with pytest.raises(TypeError, match="either one context object"):
        bound.flow.compute(FlowContext(value=3), value=4)


def test_bound_optional_none_context_preserves_wrapped_dependencies():
    class Dep(CallableModel):
        @Flow.call
        def __call__(self, context: FlowContext) -> GenericResult[int]:
            return GenericResult(value=1)

    class Root(CallableModel):
        dep: Dep

        @Flow.call
        def __call__(self, context: Optional[FlowContext] = None) -> GenericResult[int]:
            return GenericResult(value=self.dep(FlowContext()).value + (0 if context is None else context.bonus))

        @Flow.deps
        def __deps__(self, context: Optional[FlowContext]) -> list[tuple[CallableModel, list[ContextBase]]]:
            return [(self.dep, [FlowContext()])]

    root = Root(dep=Dep())
    bound = root.flow.with_context()
    graph = get_dependency_graph(bound.__call__.get_evaluation_context(bound, None))

    assert len(graph.ids) == 2
    assert bound.flow.compute().value == 1


def test_from_context_anchor_behavior():
    @Flow.model
    def foo(a: int, b: FromContext[int]) -> int:
        return a + b

    assert foo(a=11).flow.compute(b=12).value == 23
    assert foo(a=11, b=12).flow.compute().value == 23

    with pytest.raises(TypeError, match="compute\\(\\) cannot satisfy unbound regular parameter\\(s\\): a"):
        foo().flow.compute(a=11, b=12)


def test_regular_param_accepts_upstream_model():
    @Flow.model
    def source(value: FromContext[int], offset: int) -> int:
        return value + offset

    @Flow.model
    def foo(a: int, b: FromContext[int]) -> int:
        return a + b

    model = foo(a=source(offset=5))
    assert model.flow.compute(value=7, b=12).value == 24
    assert model.flow.compute(FlowContext(value=7, b=12)).value == 24


def test_regular_param_containers_are_literals():
    @Flow.model
    def source(value: FromContext[int]) -> int:
        return value

    @Flow.model
    def inspect_values(values: list[Any]) -> int:
        return sum(isinstance(value, CallableModel) for value in values)

    model = inspect_values(values=[source()])

    assert model.__deps__(FlowContext(value=10)) == []
    assert model.flow.compute(value=10).value == 1

    @Flow.model
    def total(values: list[int]) -> int:
        return sum(values)

    with pytest.raises(TypeError, match="Field 'values'"):
        total(values=[source()])


def test_regular_param_dep_marker_allows_marked_container_leaves():
    calls = {"source": 0, "row": 0, "total": 0}

    @Flow.model
    def source(value: FromContext[int], offset: int) -> int:
        calls["source"] += 1
        return value + offset

    @Flow.model
    def row_source(value: FromContext[int]) -> list[int]:
        calls["row"] += 1
        return [value, value + 1]

    @Flow.model
    def list_source(value: FromContext[int]) -> list[int]:
        return [value, value * 2]

    @Flow.model
    def total(values: list[Dep[int]], rows: list[Dep[list[int]]] = ()) -> int:
        calls["total"] += 1
        return sum(values) + sum(sum(row) for row in rows)

    model = total(values=(source(offset=1), "2", 3), rows=([4, 5], row_source()))
    deps = model.__deps__(FlowContext(value=10))

    assert len(deps) == 2
    assert model.flow.compute(value=10).value == 46
    assert calls == {"source": 1, "row": 1, "total": 1}

    whole_param_model = total(values=list_source())
    assert whole_param_model.flow.compute(value=4).value == 12

    serialized_model = total(values=[source(offset=5).model_dump(mode="python")])
    assert isinstance(serialized_model.values[0], CallableModel)
    assert serialized_model.flow.compute(value=10).value == 15

    registry = ModelRegistry.root().clear()
    registry.add("source", source(offset=4))
    try:
        registry_model = total(values=["source", "2"], rows=(row_source(),))
        assert registry_model.flow.compute(value=10).value == 37
    finally:
        registry.clear()


def test_dep_marker_allows_tuple_and_dict_value_slots():
    @Flow.model
    def source(value: FromContext[int], offset: int) -> int:
        return value + offset

    @Flow.model
    def total(pair: tuple[Dep[int], str], values: dict[str, Dep[int]], many: tuple[Dep[int], ...]) -> int:
        return pair[0] + sum(values.values()) + sum(many)

    model = total(
        pair=(source(offset=1), "ignored"),
        values={"literal": "2", "model": source(offset=3)},
        many=(source(offset=5), "7"),
    )

    assert len(model.__deps__(FlowContext(value=10))) == 3
    assert model.flow.compute(value=10).value == 48


def test_dep_marker_participates_in_graph_and_effective_cache_identity():
    calls = {"source": 0, "total": 0}

    @Flow.model
    def source(value: FromContext[int]) -> int:
        calls["source"] += 1
        return value * 10

    @Flow.model
    def total(values: list[Dep[int]], bonus: FromContext[int]) -> int:
        calls["total"] += 1
        return sum(values) + bonus

    model = total(values=[source(), 2])
    graph = get_dependency_graph(model.__call__.get_evaluation_context(model, FlowContext(value=3, bonus=7, unused="one")))
    assert len(graph.ids) == 2

    cache = MemoryCacheEvaluator()
    with FlowOptionsOverride(options={"evaluator": cache, "cacheable": True}):
        assert model.flow.compute(value=3, bonus=7, unused="one").value == 39
        assert model.flow.compute(value=3, bonus=7, unused="two").value == 39

    assert calls == {"source": 1, "total": 1}


def test_dep_marker_preserves_other_annotated_metadata():
    @Flow.model
    def source(value: FromContext[int]) -> int:
        return value

    @Flow.model
    def total(values: list[Dep[Annotated[int, Field(gt=0)]]]) -> int:
        return sum(values)

    assert total(values=["1", source()]).flow.compute(value=2).value == 3

    with pytest.raises(TypeError, match="Field 'values"):
        total(values=[-1])

    with pytest.raises(TypeError, match="Regular parameter"):
        total(values=[source()]).flow.compute(value=-1)


def test_dep_marker_rejects_unmarked_or_dynamic_dependency_shapes():
    @Flow.model
    def source(value: FromContext[int]) -> int:
        return value

    @Flow.model
    def row_source(value: FromContext[int]) -> list[int]:
        return [value]

    @Flow.model
    def row_total(rows: list[Dep[list[int]]]) -> int:
        return sum(sum(row) for row in rows)

    with pytest.raises(TypeError, match="Field 'rows"):
        row_total(rows=[[source()]])

    with pytest.raises(TypeError, match="container values"):

        @Flow.model
        def top_level(values: Dep[int]) -> int:
            return values

    with pytest.raises(TypeError, match="cannot contain another Dep"):

        @Flow.model
        def nested(values: list[Dep[list[Dep[int]]]]) -> int:
            return sum(sum(value) for value in values)

    with pytest.raises(TypeError, match="dict keys"):

        @Flow.model
        def dict_key(values: dict[Dep[str], int]) -> int:
            return sum(values.values())

    with pytest.raises(TypeError, match="container values"):

        @Flow.model
        def set_values(values: set[Dep[int]]) -> int:
            return sum(values)


def test_dep_marker_is_ordinary_annotation_for_handwritten_callable_model():
    @Flow.model
    def source(value: FromContext[int]) -> int:
        return value

    class Plain(CallableModel):
        values: list[Dep[int]]

        @Flow.call
        def __call__(self, context: FlowContext) -> GenericResult[int]:
            del context
            return GenericResult(value=sum(self.values))

    assert Plain(values=["1", 2]).flow.compute().value == 3
    with pytest.raises(ValidationError):
        Plain(values=[source()])


def test_regular_param_upstream_dependency_coerced():
    """Upstream model returning str should be coerced to downstream int annotation."""

    @Flow.model
    def str_source(tag: FromContext[str]) -> str:
        return tag

    @Flow.model
    def consumer(x: int, bonus: FromContext[int]) -> int:
        return x + bonus

    model = consumer(x=str_source())
    # str_source returns "42" (a str); consumer expects x: int; should be coerced
    result = model.flow.compute(tag="42", bonus=10)
    assert result.value == 52
    assert isinstance(result.value, int)

    # Also test that invalid coercion raises
    with pytest.raises(TypeError, match="Regular parameter"):
        model.flow.compute(tag="not_a_number", bonus=10)


def test_regular_param_lazy_upstream_dependency_coerced():
    """Lazy upstream model output should be coerced on first call."""

    @Flow.model
    def lazy_source(v: FromContext[int]) -> str:
        return str(v)

    @Flow.model
    def consumer(x: Lazy[int], bonus: FromContext[int]) -> int:
        return x() + bonus

    model = consumer(x=lazy_source())
    result = model.flow.compute(v=7, bonus=3)
    assert result.value == 10
    assert isinstance(result.value, int)


def test_regular_param_plain_callable_model_projects_dependency_context():
    class ValueContext(ContextBase):
        value: int

    class PlainSource(CallableModel):
        @property
        def context_type(self):
            return ValueContext

        @property
        def result_type(self):
            return GenericResult[int]

        @Flow.call
        def __call__(self, context: ValueContext) -> GenericResult[int]:
            return GenericResult(value=context.value * 10)

    @Flow.model
    def root(x: int, bonus: FromContext[int]) -> int:
        return x + bonus

    model = root(x=PlainSource())
    deps = model.__deps__(FlowContext(value=3, bonus=7))

    assert len(deps) == 1
    dep_model, dep_contexts = deps[0]
    assert isinstance(dep_model, PlainSource)
    assert dep_contexts == [ValueContext(value=3)]
    assert model.flow.compute(FlowContext(value=3, bonus=7)).value == 37


def test_bound_regular_param_name_can_collide_with_ambient_context():
    @Flow.model
    def source(a: FromContext[int]) -> int:
        return a

    @Flow.model
    def combine(a: int, left: int, bonus: FromContext[int]) -> int:
        return a + left + bonus

    model = combine(a=100, left=source())
    assert model.flow.compute(FlowContext(a=7, bonus=5)).value == 112

    with pytest.raises(TypeError, match="does not accept regular parameter override\\(s\\): a"):
        model.flow.compute(a=7, bonus=5)


def test_contextual_param_rejects_callable_model():
    @Flow.model
    def source(offset: int, value: FromContext[int]) -> GenericResult[int]:
        return GenericResult(value=value + offset)

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
    assert model.flow.inspect().bound_inputs == {"a": 11, "b": 12}
    assert model.flow.inspect().context_inputs == {"b": int}
    assert model.flow.inspect().required_inputs == {}
    assert model.flow.compute().value == 23


def test_contextual_function_defaults_remain_contextual():
    @Flow.model
    def foo(a: int, b: FromContext[int] = 5) -> int:
        return a + b

    model = foo(a=2)
    assert model.flow.inspect().bound_inputs == {"a": 2}
    assert model.flow.inspect().context_inputs == {"b": int}
    assert model.flow.inspect().required_inputs == {}
    assert model.flow.compute().value == 7
    assert model.flow.compute(b=10).value == 12


def test_context_type_accepts_richer_subclass_for_from_context():
    @Flow.model(context_type=ParentRangeContext)
    def span_days(multiplier: int, start_date: FromContext[date], end_date: FromContext[date]) -> int:
        return multiplier * ((end_date - start_date).days + 1)

    model = span_days(multiplier=2)
    assert model.flow.compute(start_date="2024-01-01", end_date="2024-01-03").value == 6
    assert model(RichRangeContext(start_date=date(2024, 1, 1), end_date=date(2024, 1, 4), label="x")).value == 8


def test_declared_context_type_introspection_reports_effective_field_type():
    class ModeContext(ContextBase):
        mode: Literal["a"]

    @Flow.model(context_type=ModeContext)
    def choose(mode: FromContext[str]) -> str:
        return mode

    model = choose()

    assert model.flow.inspect().context_inputs == {"mode": Literal["a"]}
    assert model.flow.inspect().required_inputs == {"mode": Literal["a"]}
    assert model.flow.compute(mode="a").value == "a"
    with pytest.raises(ValidationError):
        model.flow.compute(mode="b")


def test_context_type_validation_applies_to_resolved_contextual_values():
    @Flow.model(context_type=OrderedContext)
    def add(a: FromContext[int], b: FromContext[int]) -> int:
        return a + b

    with pytest.raises(ValueError, match="a must be <= b"):
        add().flow.compute(a=2, b=1)

    with pytest.raises(ValueError, match="a must be <= b"):
        add(a=2, b=1)


def test_context_type_validates_construction_time_contextual_defaults_early():
    class PositiveContext(ContextBase):
        x: int = Field(gt=0)

    @Flow.model(context_type=PositiveContext)
    def identity(x: FromContext[int]) -> int:
        return x

    with pytest.raises(ValueError, match="greater than 0"):
        identity(x=-1)


def test_context_type_validates_partial_contextual_defaults_early():
    class PositivePairContext(ContextBase):
        x: int = Field(gt=0)
        y: int

    @Flow.model(context_type=PositivePairContext)
    def add(x: FromContext[int], y: FromContext[int]) -> int:
        return x + y

    with pytest.raises(ValueError, match="greater than 0"):
        add(x=-1)


def test_context_type_validates_static_with_context_overrides_early():
    @Flow.model(context_type=OrderedContext)
    def add(a: FromContext[int], b: FromContext[int]) -> int:
        return a + b

    with pytest.raises(ValueError, match="a must be <= b"):
        add().flow.with_context(a=2, b=1)


def test_context_type_validates_chained_static_with_context_overrides_early():
    @Flow.model(context_type=OrderedContext)
    def add(a: FromContext[int], b: FromContext[int]) -> int:
        return a + b

    with pytest.raises(ValueError, match="a must be <= b"):
        add().flow.with_context(a=2).flow.with_context(b=1)


def test_context_type_validates_partial_static_with_context_overrides_early():
    class PositivePairContext(ContextBase):
        x: int = Field(gt=0)
        y: int

    @Flow.model(context_type=PositivePairContext)
    def add(x: FromContext[int], y: FromContext[int]) -> int:
        return x + y

    with pytest.raises(ValueError, match="greater than 0"):
        add().flow.with_context(x=-1)


def test_context_type_validates_static_field_transform_overrides_early():
    @Flow.model(context_type=OrderedContext)
    def add(a: FromContext[int], b: FromContext[int] = 1) -> int:
        return a + b

    with pytest.raises(ValueError, match="a must be <= b"):
        add().flow.with_context(a=static_bad())


def test_context_type_validates_static_patch_transform_overrides_early():
    @Flow.model(context_type=OrderedContext)
    def add(a: FromContext[int], b: FromContext[int] = 1) -> int:
        return a + b

    with pytest.raises(ValueError, match="a must be <= b"):
        add().flow.with_context(static_patch())


def test_context_named_parameters_are_just_regular_parameters():
    @Flow.model
    def loader(context: DateRangeContext, source: str = "db") -> GenericResult[str]:
        return GenericResult(value=f"{source}:{context.start_date}:{context.end_date}")

    model = loader(context=DateRangeContext(start_date=date(2024, 1, 1), end_date=date(2024, 1, 2)), source="api")
    assert model.flow.inspect().bound_inputs["context"] == DateRangeContext(start_date=date(2024, 1, 1), end_date=date(2024, 1, 2))
    assert model.flow.inspect().context_inputs == {}
    assert model.flow.compute().value == "api:2024-01-01:2024-01-02"

    with pytest.raises(TypeError, match="Missing regular parameter\\(s\\) for loader: context"):
        loader(source="api").flow.compute(start_date="2024-01-01", end_date="2024-01-02")


def test_auto_unwrap_defaults_to_false_for_auto_wrapped_results():
    @Flow.model
    def add(a: int, b: FromContext[int]) -> int:
        return a + b

    model = add(a=10)
    result = model.flow.compute(b=5)
    assert model.result_type == GenericResult[int]
    assert inspect.signature(type(model).__call__).return_annotation == GenericResult[int]
    assert type(result) is GenericResult[int]
    assert repr(result) == "GenericResult[int](value=15)"
    assert result.value == 15


def test_compute_does_not_unwrap_explicit_generic_result_returns():
    @Flow.model
    def load(value: FromContext[int]) -> GenericResult[int]:
        return GenericResult(value=value * 2)

    model = load()
    result = model.flow.compute(value=3)
    assert model.result_type == GenericResult[int]
    assert type(result) is GenericResult[int]
    assert repr(result) == "GenericResult[int](value=6)"
    assert result.value == 6


def test_flow_model_rejects_union_resultbase_return_annotations():
    with pytest.raises(TypeError, match="does not support Union or Optional ResultBase"):

        @Flow.model
        def optional_result(value: FromContext[int]) -> Optional[GenericResult[int]]:
            return GenericResult(value=value)

    with pytest.raises(TypeError, match="does not support Union or Optional ResultBase"):

        @Flow.model
        def pep604_optional_result(value: FromContext[int]) -> GenericResult[int] | None:
            return GenericResult(value=value)

    with pytest.raises(TypeError, match="does not support Union or Optional ResultBase"):

        @Flow.model
        def annotated_optional_result(value: FromContext[int]) -> Annotated[GenericResult[int] | None, "meta"]:
            return GenericResult(value=value)


def test_flow_model_allows_plain_union_return_annotations():
    @Flow.model
    def choose(flag: FromContext[bool]) -> int | str:
        return 1 if flag else "one"

    assert choose().flow.compute(flag=True).value == 1
    assert choose().flow.compute(flag=False).value == "one"


def test_flow_model_handles_annotated_result_annotations():
    @Flow.model
    def explicit_result() -> Annotated[GenericResult[int], "meta"]:
        return GenericResult(value=1)

    @Flow.model
    def plain_union(flag: FromContext[bool]) -> Annotated[int | str, "meta"]:
        return 1 if flag else "one"

    result = explicit_result().flow.compute()
    assert type(result) is GenericResult[int]
    assert result.value == 1
    assert plain_union().flow.compute(flag=True).value == 1
    assert plain_union().flow.compute(flag=False).value == "one"


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


def test_auto_wrap_validates_return_type():
    @Flow.model
    def bad(x: FromContext[int]) -> int:
        return "oops"

    with pytest.raises(ValidationError, match=r"GenericResult\[int\]"):
        bad().flow.compute(x=1)


def test_auto_wrap_respects_validate_result_false():
    @Flow.model(validate_result=False)
    def bad_decorator() -> int:
        return "oops"

    @Flow.model
    def bad_runtime() -> int:
        return "oops"

    decorator_result = bad_decorator().flow.compute()
    runtime_result = bad_runtime().flow.compute(_options=FlowOptions(validate_result=False))

    assert isinstance(decorator_result, GenericResult)
    assert decorator_result.value == "oops"
    assert isinstance(runtime_result, GenericResult)
    assert runtime_result.value == "oops"


def test_auto_wrap_coerces_compatible_return():
    @Flow.model
    def coerce(x: FromContext[int]) -> float:
        return 3

    result = coerce().flow.compute(x=1)
    assert result.value == 3.0
    assert isinstance(result.value, float)


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


def test_context_named_regular_parameter_can_coexist_with_from_context():
    @Flow.model
    def mixed(context: SimpleContext, y: FromContext[int]) -> int:
        return context.value + y

    model = mixed(context=SimpleContext(value=10))
    assert model.flow.inspect().bound_inputs == {"context": SimpleContext(value=10)}
    assert model.flow.inspect().context_inputs == {"y": int}
    assert model.flow.compute(y=5).value == 15


@pytest.mark.parametrize("reserved_name", ["meta", "context_type", "result_type", "type_"])
def test_flow_model_rejects_reserved_parameter_names(reserved_name):
    namespace = {"Flow": Flow, "FromContext": FromContext}
    exec(
        f"def bad({reserved_name}: str, value: FromContext[int]) -> str:\n    return str(value)\n",
        namespace,
    )

    with pytest.raises(TypeError, match=f"Parameter name\\(s\\) '{reserved_name}' are reserved"):
        Flow.model(namespace["bad"])


def test_flow_model_allows_flow_parameter_name():
    """``flow`` is not reserved: the field shadows the (non-data descriptor) accessor.

    ``model.flow`` returns the field value, and ``Flow.of(model)`` still reaches
    the flow API.
    """

    def add(flow: int, value: FromContext[int]) -> int:
        return flow + value

    with pytest.warns(UserWarning, match="shadows an attribute"):
        add_model = Flow.model(add)

    model = add_model(flow=10)
    assert model.flow == 10
    assert Flow.of(model).compute(value=5).value == 15


def test_context_type_requires_from_context():
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


def test_lazy_parameter_requires_dependency_binding():
    @Flow.model
    def source(value: FromContext[int]) -> int:
        return value

    @Flow.model
    def choose(lazy_value: Lazy[int]) -> int:
        return lazy_value()

    with pytest.raises(TypeError, match="Lazy"):
        choose(lazy_value=1)

    assert choose(lazy_value=source()).flow.compute(value=3).value == 3


def test_lazy_parameter_rejects_literal_function_default():
    with pytest.raises(TypeError, match="Lazy"):

        @Flow.model
        def choose(lazy_value: Lazy[int] = 1) -> int:
            return lazy_value()


def test_lazy_and_from_context_combination_is_rejected():
    with pytest.raises(TypeError, match="cannot combine Lazy"):

        @Flow.model
        def bad(x: Lazy[FromContext[int]]) -> int:
            return x()


def test_bare_flow_marker_annotations_are_rejected():
    with pytest.raises(TypeError, match=r"FromContext\[T\]"):

        @Flow.model
        def bad_context(x: FromContext) -> int:
            return 1

    with pytest.raises(TypeError, match=r"Lazy\[T\]"):

        @Flow.model
        def bad_lazy(x: Lazy) -> int:
            return x()

    with pytest.raises(TypeError, match=r"FromContext\[T\]"):

        @Flow.context_transform
        def bad_transform_context(x: FromContext) -> int:
            return 1

    with pytest.raises(TypeError, match=r"Lazy\[T\]"):

        @Flow.context_transform
        def bad_transform_lazy(x: Lazy) -> int:
            return 1


def test_auto_wrap_and_serialization_roundtrip():
    @Flow.model
    def add(a: int, b: FromContext[int]) -> int:
        return a + b

    model = add(a=10)
    dumped = model.model_dump(mode="python")
    restored = type(model).model_validate(dumped)

    assert restored.flow.inspect().bound_inputs == {"a": 10}
    assert restored.flow.inspect().required_inputs == {"b": int}
    assert restored.flow.compute(b=5).value == 15


def test_generated_models_cloudpickle_roundtrip():
    @Flow.model
    def multiply(a: int, b: FromContext[int]) -> int:
        return a * b

    model = multiply(a=6)
    restored = cloudpickle.loads(cloudpickle.dumps(model, protocol=5))
    assert restored.flow.compute(b=7).value == 42


def test_generated_models_plain_pickle_roundtrip():
    @Flow.model
    def multiply(a: int, b: FromContext[int]) -> int:
        return a * b

    model = multiply(a=6)
    restored = pickle.loads(pickle.dumps(model, protocol=5))
    assert restored.flow.compute(b=7).value == 42


def test_generated_model_dep_marker_pickle_roundtrip():
    @Flow.model
    def source(value: FromContext[int], offset: int) -> int:
        return value + offset

    @Flow.model
    def total(values: list[Dep[int]]) -> int:
        return sum(values)

    model = total(values=[source(offset=2), 3])
    context = FlowContext(value=10)

    for dumps, loads in ((pickle.dumps, pickle.loads), (rcpdumps, rcploads)):
        restored = loads(dumps(model, protocol=5))

        assert len(restored.__deps__(context)) == 1
        assert restored.flow.compute(context).value == 15


def test_generated_model_direct_call_plain_pickle_uses_serialized_factory(monkeypatch):
    module = ModuleType("ccflow_test_direct_model")

    def multiply(a: int, b: FromContext[int]) -> int:
        return a * b

    multiply.__module__ = module.__name__
    multiply.__qualname__ = multiply.__name__
    module.multiply = multiply
    monkeypatch.setitem(sys.modules, module.__name__, module)

    factory = Flow.model(multiply)
    assert not hasattr(module.multiply, "_generated_model")

    model = factory(a=6)
    restored = pickle.loads(pickle.dumps(model, protocol=5))
    assert restored.flow.compute(b=7).value == 42


def test_local_generated_model_effective_cache_key_survives_pickle_roundtrip():
    def make_model():
        @Flow.model
        def add(a: int, b: FromContext[int]) -> int:
            return a + b

        return add(a=1)

    model = make_model()
    context = FlowContext(b=2)
    before = cache_key(model.__call__.get_evaluation_context(model, context), effective=True)

    for dumps, loads in ((pickle.dumps, pickle.loads), (cloudpickle.dumps, cloudpickle.loads)):
        restored = loads(dumps(model, protocol=5))
        after = cache_key(restored.__call__.get_evaluation_context(restored, context), effective=True)
        assert after == before


def test_generated_model_plain_pickle_preserves_external_pydantic_private_state():
    @Flow.model
    def read(payload: object) -> int:
        return payload.x + payload._bonus

    payload = ExternalPydanticPayload(x=2)
    payload._bonus = 40

    restored = pickle.loads(pickle.dumps(read(payload=payload), protocol=5))

    assert restored.payload._bonus == 40
    assert restored.flow.compute().value == 42


def test_generated_model_pickle_preserves_external_ccflow_private_state():
    @Flow.model
    def read(payload: object) -> int:
        return payload.x + payload._bonus

    payload = ExternalCcflowPayload(x=2)
    payload._bonus = 40

    for dumps, loads in ((pickle.dumps, pickle.loads), (cloudpickle.dumps, cloudpickle.loads)):
        restored = loads(dumps(read(payload=payload), protocol=5))

        assert isinstance(restored.payload, ExternalCcflowPayload)
        assert restored.payload._bonus == 40
        assert restored.flow.compute().value == 42


def test_generated_model_pickle_preserves_outer_graph_identity_and_cycles():
    @Flow.model
    def read(payload: object) -> int:
        return 0

    for dumps, loads in ((pickle.dumps, pickle.loads), (cloudpickle.dumps, cloudpickle.loads)):
        shared = []
        restored_model, restored_shared = loads(dumps((read(payload=shared), shared), protocol=5))
        assert restored_model.payload is restored_shared

        cycle = read(payload=None)
        cycle.payload = cycle
        restored_cycle = loads(dumps(cycle, protocol=5))
        assert restored_cycle.payload is restored_cycle


def test_local_generated_model_cloudpickle_preserves_local_pydantic_literal_state():
    def make_model():
        class LocalPayload(PydanticBaseModel):
            x: int
            _bonus: int = PrivateAttr(default=1)

        @Flow.model
        def read(payload: object) -> int:
            return payload.x + payload._bonus

        payload = LocalPayload(x=2)
        payload._bonus = 40
        return read(payload=payload)

    restored = cloudpickle.loads(cloudpickle.dumps(make_model(), protocol=5))

    assert restored.payload._bonus == 40
    assert restored.flow.compute().value == 42


def test_bound_model_pickle_preserves_external_pydantic_static_context_value():
    @Flow.model
    def read(payload: FromContext[object]) -> int:
        return payload.x + payload._bonus

    payload = ExternalPydanticPayload(x=2)
    payload._bonus = 40
    bound = read().flow.with_context(payload=payload)
    assert bound.flow.compute().value == 42

    for dumps, loads in ((pickle.dumps, pickle.loads), (cloudpickle.dumps, cloudpickle.loads)):
        restored = loads(dumps(bound, protocol=5))
        restored_payload = restored.context_spec.operations[0].spec.value

        assert isinstance(restored_payload, ExternalPydanticPayload)
        assert restored_payload._bonus == 40
        assert restored.flow.compute().value == 42


def test_bound_model_pickle_preserves_external_pydantic_context_transform_bound_arg():
    @Flow.model
    def read(value: FromContext[int]) -> int:
        return value

    @Flow.context_transform
    def derive(payload: object) -> int:
        return payload.x + payload._bonus

    payload = ExternalPydanticPayload(x=2)
    payload._bonus = 40
    bound = read().flow.with_context(value=derive(payload=payload))

    for dumps, loads in ((pickle.dumps, pickle.loads), (cloudpickle.dumps, cloudpickle.loads)):
        restored = loads(dumps(bound, protocol=5))
        restored_payload = restored.context_spec.operations[0].spec.bound_args["payload"]

        assert isinstance(restored_payload, ExternalPydanticPayload)
        assert restored_payload._bonus == 40
        assert restored.flow.compute().value == 42


def test_bound_model_pickle_preserves_external_ccflow_static_context_value():
    @Flow.model
    def read(payload: FromContext[object]) -> int:
        return payload.x + payload._bonus

    payload = ExternalCcflowPayload(x=2)
    payload._bonus = 40
    bound = read().flow.with_context(payload=payload)
    assert bound.flow.compute().value == 42

    for dumps, loads in ((pickle.dumps, pickle.loads), (cloudpickle.dumps, cloudpickle.loads)):
        restored = loads(dumps(bound, protocol=5))
        restored_payload = restored.context_spec.operations[0].spec.value

        assert isinstance(restored_payload, ExternalCcflowPayload)
        assert restored_payload._bonus == 40
        assert restored.flow.compute().value == 42


def test_bound_model_pickle_preserves_external_ccflow_context_transform_bound_arg():
    @Flow.model
    def read(value: FromContext[int]) -> int:
        return value

    @Flow.context_transform
    def derive(payload: object) -> int:
        return payload.x + payload._bonus

    payload = ExternalCcflowPayload(x=2)
    payload._bonus = 40
    bound = read().flow.with_context(value=derive(payload=payload))

    for dumps, loads in ((pickle.dumps, pickle.loads), (cloudpickle.dumps, cloudpickle.loads)):
        restored = loads(dumps(bound, protocol=5))
        restored_payload = restored.context_spec.operations[0].spec.bound_args["payload"]

        assert isinstance(restored_payload, ExternalCcflowPayload)
        assert restored_payload._bonus == 40
        assert restored.flow.compute().value == 42


def test_local_generated_model_plain_pickle_handles_function_default_state():
    def make_model():
        @Flow.model
        def first(xs: list[int] = [1], b: FromContext[int] = 2) -> int:
            return xs[0] + b

        return first()

    restored = pickle.loads(pickle.dumps(make_model(), protocol=5))

    assert restored.flow.compute().value == 3


def test_unresolved_lazy_nested_local_generated_dependency_identity_survives_pickle_roundtrip():
    def make_model():
        @Flow.model
        def dependency(d: FromContext[int]) -> int:
            return d

        @Flow.model
        def source(dep_value: int, a: FromContext[int]) -> int:
            return dep_value + a

        @Flow.model
        def choose(x: int, lazy_value: Lazy[int], use_lazy: FromContext[bool]) -> int:
            return lazy_value() if use_lazy else x

        return choose(x=7, lazy_value=source(dep_value=dependency()))

    model = make_model()
    context = FlowContext(use_lazy=False)
    before_eval = model.__call__.get_evaluation_context(model, context)
    before_key = cache_key(before_eval, effective=True)
    before_root = get_dependency_graph(before_eval).root_id

    for dumps, loads in ((pickle.dumps, pickle.loads), (cloudpickle.dumps, cloudpickle.loads)):
        restored = loads(dumps(model, protocol=5))
        after_eval = restored.__call__.get_evaluation_context(restored, context)

        assert cache_key(after_eval, effective=True) == before_key
        assert get_dependency_graph(after_eval).root_id == before_root


def test_generated_model_pydantic_roundtrip_via_base_model():
    @Flow.model
    def add(a: int, b: FromContext[int]) -> int:
        return a + b

    from ccflow import BaseModel

    model = add(a=10)
    dumped = model.model_dump(mode="python")

    # BaseModel.model_validate uses the _target_ field (PyObjectPath) to
    # reconstruct the correct generated class.
    restored = BaseModel.model_validate(dumped)
    assert type(restored) is type(model)
    assert restored.flow.compute(b=5).value == 15


def test_generated_model_json_roundtrip():
    @Flow.model
    def add(a: int, b: FromContext[int]) -> int:
        return a + b

    from ccflow import BaseModel

    model = add(a=10, b=3)
    json_str = model.model_dump_json()
    restored = BaseModel.model_validate_json(json_str)
    assert type(restored) is type(model)
    assert restored.flow.compute().value == 13
    assert restored.flow.compute(b=7).value == 17


def test_generated_model_dependency_input_json_roundtrip():
    from ccflow import BaseModel

    model = data_transformer(source=data_source(base_value=1), factor=2)
    restored = BaseModel.model_validate_json(model.model_dump_json())

    assert restored.flow.compute(value=10).value == 22
    assert isinstance(restored.source, CallableModel)


def test_generated_model_dep_marker_json_roundtrip():
    from ccflow import BaseModel

    @Flow.model
    def source(value: FromContext[int], offset: int) -> int:
        return value + offset

    @Flow.model
    def total(values: list[Dep[int]]) -> int:
        return sum(values)

    model = total(values=[source(offset=1), 2])
    restored = BaseModel.model_validate_json(model.model_dump_json())

    assert type(restored) is type(model)
    assert isinstance(restored.values[0], CallableModel)
    assert len(restored.__deps__(FlowContext(value=10))) == 1
    assert restored.flow.compute(value=10).value == 13


def test_generated_model_lazy_dependency_input_json_roundtrip():
    @Flow.model
    def choose(source: Lazy[int], use_value: FromContext[bool]) -> int:
        return source() if use_value else 0

    from ccflow import BaseModel

    model = choose(source=data_source(base_value=1))
    restored = BaseModel.model_validate_json(model.model_dump_json())

    assert restored.flow.compute(value=10, use_value=True).value == 11
    assert restored.flow.compute(value=10, use_value=False).value == 0
    assert isinstance(restored.source, CallableModel)


def test_generated_model_type_dict_regular_input_stays_literal_when_annotation_accepts_dict():
    @Flow.model
    def payload_type(payload: dict) -> str:
        return type(payload).__name__

    type_payload = data_source(base_value=1).model_dump(mode="python")
    model = payload_type(payload=type_payload)

    assert model.flow.compute(value=10).value == "dict"
    assert model.payload == type_payload


def test_generated_model_target_alias_dict_regular_input_stays_literal():
    @Flow.model
    def payload_type(payload: Any) -> str:
        return type(payload).__name__

    alias_payload = data_source(base_value=1).model_dump(mode="python", by_alias=True)
    model = payload_type(payload=alias_payload)

    assert model.flow.compute(value=10).value == "dict"
    assert model.payload == alias_payload


def test_generated_model_target_alias_restores_dependency_after_literal_validation_fails():
    @Flow.model
    def total(value: int) -> int:
        return value

    alias_payload = data_source(base_value=1).model_dump(mode="python", by_alias=True)
    model = total(value=alias_payload)

    assert model.flow.compute(FlowContext(value=10)).value == 11
    assert isinstance(model.value, CallableModel)


def test_generated_model_type_marker_restores_dependency_after_literal_validation_fails():
    @Flow.model
    def total(value: int) -> int:
        return value

    type_payload = data_source(base_value=1).model_dump(mode="python")
    model = total(value=type_payload)

    assert model.flow.compute(FlowContext(value=10)).value == 11
    assert isinstance(model.value, CallableModel)


@pytest.mark.parametrize(
    "payload",
    [
        {"_target_": "does.not.exist.Class", "foo": 1},
        {"_target_": "builtins.dict", "foo": 1},
        {"type_": "does.not.exist.Class", "foo": 1},
        {"type_": "builtins.dict", "foo": 1},
    ],
)
def test_serialized_dependency_fallback_preserves_literal_validation_error(payload):
    @Flow.model
    def total(value: int) -> int:
        return value

    with pytest.raises(TypeError, match="Field 'value': expected int, got dict"):
        total(value=payload)


def test_registry_lookup_preserves_literal_error_when_candidate_type_is_incompatible():
    registry = ModelRegistry.root().clear()
    registry.add("context", SimpleContext(value=1))

    @Flow.model
    def total(values: list[int]) -> int:
        return sum(values)

    try:
        with pytest.raises(TypeError, match="Field 'values': expected list, got str"):
            total(values="context")
    finally:
        registry.clear()


def test_importable_generated_model_uses_stable_module_path_for_type_serialization():
    model = basic_loader(source="library", multiplier=3)
    stable_path = f"{__name__}._basic_loader_Model"

    assert getattr(sys.modules[__name__], "_basic_loader_Model") is type(model)
    assert "__ccflow_import_path__" not in type(model).__dict__
    assert str(PyObjectPath.validate(type(model))) == stable_path
    assert str(model.model_dump(mode="python")["type_"]) == stable_path


def test_importable_generated_model_duplicate_names_raise_conflict(monkeypatch):
    module = ModuleType("ccflow_test_duplicate_generated_models")
    module.Flow = Flow
    module.FromContext = FromContext
    monkeypatch.setitem(sys.modules, module.__name__, module)

    exec(
        """
def stage(value: FromContext[int]) -> int:
    return value + 1
""",
        module.__dict__,
    )
    first_factory = Flow.model(module.stage)
    module.first = first_factory
    first = first_factory()

    exec(
        """
def stage(value: FromContext[int]) -> int:
    return value + 2
""",
        module.__dict__,
    )
    with pytest.raises(ValueError, match="already occupied"):
        Flow.model(module.stage)

    assert getattr(module, "_stage_Model") is type(first)
    assert not hasattr(module, "_stage_Model_2")
    assert str(PyObjectPath.validate(type(first))) == f"{module.__name__}._stage_Model"
    assert first.flow.compute(value=10).value == 11


def test_generated_model_pickle_path_ignores_stale_pyobjectpath_cache(monkeypatch):
    module = ModuleType("ccflow_test_stale_generated_pickle_path")
    module.Flow = Flow
    module.FromContext = FromContext
    monkeypatch.setitem(sys.modules, module.__name__, module)

    exec(
        """
def foo(a: int, x: FromContext[int]) -> int:
    return a + x + 1
""",
        module.__dict__,
    )
    first_factory = Flow.model(module.foo)
    module.foo = first_factory
    first = first_factory(a=10)
    path = f"{module.__name__}.foo"

    # Prime PyObjectPath/import_string's cache with the first factory.  Pickle
    # path selection must still inspect the live module attribute after a reload
    # or replacement, otherwise old models can serialize by a path that a clean
    # process resolves to different behavior.
    assert PyObjectPath(path).object is first_factory

    exec(
        """
def foo(a: int, x: FromContext[int]) -> int:
    return a + x + 100
""",
        module.__dict__,
    )
    second_factory = Flow.model(module.foo)
    module.foo = second_factory

    config = type(first).__flow_model_config__
    assert flow_model_module._generated_model_factory_path_for_pickle(config, type(first)) is None
    assert first.__reduce__()[0] is flow_model_module._new_local_flow_model_for_pickle
    assert pickle.loads(pickle.dumps(first, protocol=5)).flow.compute(x=1).value == 12
    assert second_factory(a=10).flow.compute(x=1).value == 111


def test_reloaded_importable_generated_model_keeps_clean_process_path(tmp_path, monkeypatch):
    module_dir = tmp_path / "reload_case"
    module_dir.mkdir()
    module_path = module_dir / "repro_mod.py"
    module_path.write_text(
        "\n".join(
            [
                "from ccflow import Flow, FromContext",
                "",
                "@Flow.model",
                "def foo(x: FromContext[int]) -> int:",
                "    return x + 1",
                "",
            ]
        )
    )
    monkeypatch.syspath_prepend(str(module_dir))

    import repro_mod

    assert str(repro_mod.foo().type_) == "repro_mod._foo_Model"
    reloaded = importlib.reload(repro_mod)
    model = reloaded.foo()
    payload = model.model_dump_json()
    script = (
        "import sys\n"
        f"sys.path.insert(0, {str(module_dir)!r})\n"
        "from ccflow import BaseModel\n"
        f"model = BaseModel.model_validate_json({payload!r})\n"
        "assert model.flow.compute(x=2).value == 3\n"
    )
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=30)

    assert str(model.type_) == "repro_mod._foo_Model"
    assert result.returncode == 0, f"Clean-process reload JSON roundtrip failed:\n{result.stderr}"


def test_reloaded_importable_generated_model_allows_stale_factory_aliases(tmp_path, monkeypatch):
    module_dir = tmp_path / "reload_alias_case"
    module_dir.mkdir()
    module_path = module_dir / "alias_mod.py"
    module_path.write_text(
        "\n".join(
            [
                "from ccflow import Flow, FromContext",
                "",
                "def foo(x: FromContext[int]) -> int:",
                "    return x + 1",
                "",
                "bar = Flow.model(foo)",
                "",
            ]
        )
    )
    monkeypatch.syspath_prepend(str(module_dir))

    import alias_mod

    assert str(alias_mod.bar().type_) == "alias_mod._foo_Model"
    reloaded = importlib.reload(alias_mod)

    assert str(reloaded.bar().type_) == "alias_mod._foo_Model"
    assert reloaded.bar().flow.compute(x=2).value == 3


def test_reloaded_importable_generated_model_allows_stale_decorator_aliases(tmp_path, monkeypatch):
    module_dir = tmp_path / "reload_decorator_alias_case"
    module_dir.mkdir()
    module_path = module_dir / "decor_alias_mod.py"
    module_path.write_text(
        "\n".join(
            [
                "from ccflow import Flow, FromContext",
                "",
                "@Flow.model",
                "def foo(x: FromContext[int]) -> int:",
                "    return x + 1",
                "",
                "foo_alias = foo",
                "",
            ]
        )
    )
    monkeypatch.syspath_prepend(str(module_dir))

    import decor_alias_mod

    assert str(decor_alias_mod.foo().type_) == "decor_alias_mod._foo_Model"
    reloaded = importlib.reload(decor_alias_mod)

    assert str(reloaded.foo().type_) == "decor_alias_mod._foo_Model"
    assert str(reloaded.foo_alias().type_) == "decor_alias_mod._foo_Model"
    assert reloaded.foo().flow.compute(x=2).value == 3


def test_importable_bound_model_context_transform_json_roundtrip_cross_process():
    model = basic_loader(source="library", multiplier=3).flow.with_context(value=increment_b(amount=3))
    payload = model.model_dump_json()
    script = (
        "from ccflow import BaseModel\n"
        f"data = {payload!r}\n"
        "model = BaseModel.model_validate_json(data)\n"
        "result = model.flow.compute(b=1)\n"
        "assert result.value == 12, f'Expected 12, got {result.value}'\n"
    )
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, f"Cross-process bound-model JSON roundtrip failed:\n{result.stderr}"


def test_dependency_graph_cloudpickle_roundtrip():
    from ccflow.evaluators import get_dependency_graph

    @Flow.model
    def source(value: FromContext[int]) -> int:
        return value * 10

    @Flow.model
    def root(x: int, penalty: FromContext[int]) -> int:
        return x + penalty

    model = root(x=source())
    ctx = FlowContext(value=10, penalty=1)
    graph = get_dependency_graph(model.__call__.get_evaluation_context(model, ctx))

    restored = cloudpickle.loads(cloudpickle.dumps(graph))
    assert restored.root_id == graph.root_id
    assert set(restored.graph.keys()) == set(graph.graph.keys())
    assert set(restored.ids.keys()) == set(graph.ids.keys())

    # The restored graph's evaluation contexts should still be functional
    for key in graph.ids:
        original_ec = graph.ids[key]
        restored_ec = restored.ids[key]
        assert type(restored_ec.model).__name__ == type(original_ec.model).__name__
        assert restored_ec.fn == original_ec.fn


def test_with_context_validates_static_override_types():
    """Static value type mismatch should be caught when context specs are normalized."""

    @Flow.model
    def add(a: int, b: FromContext[int]) -> int:
        return a + b

    with pytest.raises(TypeError, match="with_context\\(\\)"):
        add(a=1).flow.with_context(b="not_an_int")


def test_context_transform_serializes_embedded_config_and_bound_args():
    transform_factory = Flow.context_transform(increment_b.__wrapped__)
    binding = transform_factory(amount=3)
    assert isinstance(binding, flow_model_module.ContextTransform)
    assert binding.kind == "context_transform"
    assert binding.serialized_config is not None
    assert binding.bound_args == {"amount": 3}


def test_context_transform_rejects_none_for_required_param():
    with pytest.raises(TypeError, match="Context transform argument"):
        increment_b(amount=None)


def test_context_transform_rejects_lazy_params():
    with pytest.raises(TypeError, match="does not support Lazy"):
        Flow.context_transform(lazy_context_transform_for_rejection)


def test_context_transform_direct_call_uses_serialized_payload_when_original_binding_is_plain(monkeypatch):
    module = ModuleType("ccflow_test_direct_context_transform")

    def increment(value: FromContext[int]) -> int:
        return value + 1

    increment.__module__ = module.__name__
    increment.__qualname__ = increment.__name__
    module.increment = increment
    monkeypatch.setitem(sys.modules, module.__name__, module)

    transform_factory = Flow.context_transform(increment)
    binding = transform_factory()

    assert binding.serialized_config is not None

    @Flow.model
    def add(a: int, b: FromContext[int]) -> int:
        return a + b

    bound = add(a=10).flow.with_context(b=binding)
    restored = pickle.loads(pickle.dumps(bound, protocol=5))
    assert restored.flow.compute(value=4).value == 15


def test_context_transform_json_roundtrip_recoerces_regular_bound_args():
    from ccflow import BaseModel

    @Flow.model
    def load(day: FromContext[date]) -> str:
        return day.isoformat()

    @Flow.context_transform
    def shift_from_anchor(anchor: date, days: FromContext[int]) -> date:
        return anchor + timedelta(days=days)

    bound = load().flow.with_context(day=shift_from_anchor(anchor=date(2024, 1, 1)))
    restored = BaseModel.model_validate_json(bound.model_dump_json())

    assert bound.flow.compute(days=2).value == "2024-01-03"
    assert restored.flow.compute(days=2).value == "2024-01-03"


def test_context_transform_json_roundtrip_reports_malformed_payload_cleanly():
    @Flow.model
    def add(a: int, b: FromContext[int]) -> int:
        return a + b

    bound = add(a=10).flow.with_context(b=increment_b(amount=1))
    dumped = bound.model_dump(mode="python")
    dumped["context_spec"]["operations"][0]["spec"]["serialized_config"] = "not-base64"

    restored = type(bound).model_validate(dumped)
    with pytest.raises(TypeError, match="payload does not contain"):
        restored.flow.compute(b=4)


def test_context_transform_supports_nested_functions_with_serialized_payload():
    @Flow.model
    def add(a: int, b: FromContext[int]) -> int:
        return a + b

    @Flow.context_transform
    def nested_transform(b: FromContext[int], amount: int) -> int:
        return b + amount

    binding = nested_transform(amount=3)
    assert binding.serialized_config is not None

    bound = add(a=1).flow.with_context(b=binding)
    restored = cloudpickle.loads(cloudpickle.dumps(bound))
    assert restored.flow.compute(b=4).value == 8


def test_context_transform_supports_non_importable_main_functions_with_serialized_payload():
    @Flow.model
    def add(a: int, b: FromContext[int]) -> int:
        return a + b

    def main_transform(value: FromContext[int]) -> int:
        return value + 1

    main_transform.__module__ = "__main__"
    main_transform.__qualname__ = main_transform.__name__

    transformed = Flow.context_transform(main_transform)
    binding = transformed()
    assert binding.serialized_config is not None

    bound = add(a=1).flow.with_context(b=binding)
    restored = cloudpickle.loads(cloudpickle.dumps(bound))
    assert restored.flow.compute(value=4).value == 6


def test_with_context_rejects_raw_callables():
    @Flow.model
    def add(a: int, b: FromContext[int]) -> int:
        return a + b

    with pytest.raises(TypeError, match="with_context\\(\\) 'b': expected int"):
        add(a=1).flow.with_context(b=lambda ctx: ctx.b + 1)


def test_with_context_accepts_callable_literals_for_callable_context_fields():
    def increment(value: int) -> int:
        return value + 1

    @Flow.model
    def apply(fn: FromContext[Callable[[int], int]], value: FromContext[int]) -> int:
        return fn(value)

    assert apply(fn=increment).flow.compute(value=2).value == 3
    assert apply().flow.compute(fn=increment, value=2).value == 3
    assert apply().flow.with_context(fn=increment).flow.compute(value=2).value == 3


def test_with_context_rejects_wrong_transform_position():
    @Flow.model
    def load(start_date: FromContext[int], end_date: FromContext[int]) -> int:
        return start_date + end_date

    with pytest.raises(TypeError, match="Field context transforms must be passed by keyword"):
        load().flow.with_context(increment_b(amount=1))

    with pytest.raises(TypeError, match="Patch transforms must be passed positionally"):
        load().flow.with_context(start_date=shift_integer_window(amount=10))


def test_chained_with_context_preserves_patch_and_field_order():
    @Flow.context_transform
    def patch_a() -> dict[str, int]:
        return {"a": 2}

    @Flow.model
    def source(a: FromContext[int]) -> int:
        return a

    field_then_patch = source().flow.with_context(a=1).flow.with_context(patch_a())
    patch_then_field = source().flow.with_context(patch_a()).flow.with_context(a=1)

    assert (
        field_then_patch.model_dump(mode="json")["context_spec"]["operations"]
        != patch_then_field.model_dump(mode="json")["context_spec"]["operations"]
    )
    assert field_then_patch.flow.compute().value == 2
    assert patch_then_field.flow.compute().value == 1


def test_chained_with_context_transform_reads_original_context():
    @Flow.context_transform
    def patch_a() -> dict[str, int]:
        return {"a": 2}

    @Flow.context_transform
    def b_from_a(a: FromContext[int]) -> int:
        return a + 3

    @Flow.model
    def source(a: FromContext[int], b: FromContext[int]) -> int:
        return a + b

    bound = source().flow.with_context(patch_a()).flow.with_context(b=b_from_a())

    assert bound.flow.compute(a=10).value == 15
    assert bound.flow.inspect().required_inputs == {"a": int}


def test_chained_with_context_later_field_override_skips_dead_field_transform():
    @Flow.model
    def source(a: FromContext[int]) -> int:
        return a

    bound = source().flow.with_context(a=seed_plus_one()).flow.with_context(a=1)

    assert bound.flow.inspect().bound_inputs == {"a": 1}
    assert bound.flow.inspect().runtime_inputs == {}
    assert bound.flow.inspect().required_inputs == {}
    assert bound.flow.compute().value == 1

    for dumps, loads in ((pickle.dumps, pickle.loads), (cloudpickle.dumps, cloudpickle.loads)):
        restored = loads(dumps(bound, protocol=5))
        assert restored.flow.compute().value == 1


def test_with_context_accepts_wrapped_mapping_patch_annotations():
    @Flow.model
    def load(start_date: FromContext[int], end_date: FromContext[int]) -> int:
        return start_date * 1000 + end_date

    annotated = load().flow.with_context(annotated_start_patch())
    optional = load().flow.with_context(optional_start_patch())

    assert annotated.flow.compute(start_date=1, end_date=5).value == 2005
    assert optional.flow.compute(start_date=1, end_date=5).value == 3005


def test_patch_then_keyword_override_precedence():
    @Flow.model
    def load(start_date: FromContext[int], end_date: FromContext[int]) -> int:
        return start_date * 1000 + end_date

    bound = load().flow.with_context(shift_integer_window(amount=10), start_date=100)
    result = bound(FlowContext(start_date=1, end_date=2))
    assert result.value == 100_012


def test_chained_transforms_read_original_runtime_context():
    @Flow.model
    def load(start_date: FromContext[int], end_date: FromContext[int]) -> int:
        return start_date * 1000 + end_date

    bound = load().flow.with_context(
        shift_integer_window(amount=10),
        start_date=bump_start_date(amount=100),
    )

    result = bound(FlowContext(start_date=1, end_date=2))
    assert result.value == 101_012


def test_registry_integration_for_generated_models():
    registry = ModelRegistry.root().clear()
    model = basic_loader(source="library", multiplier=3)
    registry.add("loader", model)

    retrieved = registry["loader"]
    assert isinstance(retrieved, CallableModel)
    assert retrieved(SimpleContext(value=4)).value == 12


def test_any_annotation_preserves_literal_strings():
    """A parameter typed Any should keep literal strings; registry should not steal them."""
    registry = ModelRegistry.root().clear()
    dep_model = basic_loader(source="library", multiplier=1)
    registry.add("my_key", dep_model)

    @Flow.model
    def uses_any(x: Any, y: FromContext[int]) -> int:
        return y if isinstance(x, str) else 999

    model = uses_any(x="my_key")
    result = model.flow.compute(y=3)
    assert result.value == 3, "literal string should not be replaced by registry entry for Any-typed param"


def test_registry_lookup_does_not_steal_coercible_string_literals():
    registry = ModelRegistry.root().clear()
    registry.add("3", data_source(base_value=10))
    registry.add("2024-01-01", data_source(base_value=20))
    registry.add("data.csv", data_source(base_value=30))
    registry.add("loader", data_source(base_value=40))

    @Flow.model
    def uses_int(x: int) -> str:
        return f"{type(x).__name__}:{x}"

    @Flow.model
    def uses_date(day: date) -> str:
        return f"{type(day).__name__}:{day}"

    @Flow.model
    def uses_path(path: Path) -> str:
        return f"{type(path).__name__}:{path}"

    try:
        assert uses_int(x="3").flow.compute(value=1).value == "int:3"
        assert uses_date(day="2024-01-01").flow.compute(value=1).value == "date:2024-01-01"
        path_model = uses_path(path="data.csv")
        assert path_model.path == Path("data.csv")
        assert path_model.flow.compute(value=1).value.endswith(":data.csv")

        dependency_model = uses_int(x="loader")
        assert isinstance(dependency_model.x, CallableModel)
        assert dependency_model.flow.compute(value=1).value == "int:41"
    finally:
        registry.clear()


def test_unexpected_type_adapter_errors_are_not_silently_swallowed():
    class BrokenSchema:
        @classmethod
        def __get_pydantic_core_schema__(cls, source, handler):
            raise RuntimeError("boom")

    def bad(x: BrokenSchema, y: FromContext[int]) -> int:
        del x, y
        return 0

    with pytest.raises(RuntimeError, match="boom"):
        Flow.model(bad)


@pytest.mark.parametrize("error", [RuntimeError("boom"), TypeError("boom")])
def test_unexpected_type_validation_errors_are_not_rewritten(error):
    from pydantic_core import core_schema

    class BrokenValidation:
        @classmethod
        def __get_pydantic_core_schema__(cls, source, handler):
            del source, handler

            def validate(value):
                del value
                raise error

            return core_schema.no_info_plain_validator_function(validate)

    @Flow.model
    def bad(x: BrokenValidation, y: FromContext[int]) -> int:
        del x, y
        return 0

    with pytest.raises(type(error), match="boom"):
        bad(x=object())


@pytest.mark.parametrize("error", [RuntimeError("boom"), AttributeError("boom")])
def test_unexpected_type_hint_resolution_errors_propagate(monkeypatch, error):
    def broken_get_type_hints(*args, **kwargs):
        raise error

    monkeypatch.setattr(flow_model_module, "get_type_hints", broken_get_type_hints)

    def add(x: int) -> int:
        return x

    with pytest.raises(type(error), match="boom"):
        Flow.model(add)

    def transform(x: FromContext[int]) -> int:
        return x

    with pytest.raises(type(error), match="boom"):
        Flow.context_transform(transform)


def test_generated_model_flow_api_introspection_and_execution():
    @Flow.model
    def add(a: int, b: FromContext[int]) -> int:
        return a + b

    model = add(a=10)
    assert model.flow.inspect().context_inputs == {"b": int}
    assert model.flow.inspect().bound_inputs == {"a": 10}
    assert model.flow.inspect().required_inputs == {"b": int}
    assert model.flow.compute(b=5).value == 15


def test_flow_api_is_self_describing_for_interactive_sessions():
    @Flow.model
    def add(a: int, b: FromContext[int]) -> int:
        return a + b

    flow = add(a=1).flow

    assert dir(flow) == ["compute", "inspect", "with_context"]
    assert "inspect" in dir(flow)
    assert "input_specs" not in dir(flow)
    assert "argument_specs" not in dir(flow)
    assert "inspect" in repr(flow)

    inspect_signature = inspect.signature(flow.inspect)
    assert "inputs" not in inspect_signature.parameters
    assert inspect_signature.parameters["dependencies"].annotation == Literal["direct", "recursive", "none"]
    assert inspect_signature.return_annotation is flow_model_module.FlowInspection


def test_flow_api_completes_after_binding_property_value_in_ipython():
    pytest.importorskip("IPython")
    from IPython.core.interactiveshell import InteractiveShell

    @Flow.model
    def add(a: int, b: FromContext[int]) -> int:
        return a + b

    shell = InteractiveShell.instance()
    shell.user_ns["flow_model_completion_target"] = add(a=1)
    shell.user_ns["flow_model_completion_flow"] = shell.user_ns["flow_model_completion_target"].flow

    matches = shell.Completer.attr_matches("flow_model_completion_flow.")
    assert "flow_model_completion_flow.compute" in matches
    assert "flow_model_completion_flow.inspect" in matches
    assert "flow_model_completion_flow.input_specs" not in matches
    assert "flow_model_completion_flow.argument_specs" not in matches
    assert "flow_model_completion_flow.help" not in matches
    assert "flow_model_completion_flow.validate_inputs" not in matches


def test_flow_inspect_inputs_include_defaults_and_sources():
    @Flow.model
    def add(a: int, b: FromContext[int], c: FromContext[int] = 5) -> int:
        return a + b + c

    model = add(a=10)
    inspection = model.flow.inspect()

    assert inspection.inputs["a"] == flow_model_module.InputSpec("a", int, False, flow_model_module._UNSET_FLOW_INPUT, 10, "construction")
    assert inspection.inputs["b"] == flow_model_module.InputSpec(
        "b", int, True, flow_model_module._UNSET_FLOW_INPUT, flow_model_module._UNSET_FLOW_INPUT, "runtime"
    )
    assert inspection.inputs["c"] == flow_model_module.InputSpec("c", int, False, 5, 5, "function_default")
    assert inspection.inputs["b"].required
    assert not inspection.inputs["c"].required
    assert model.flow.inspect(b=2).inputs["b"] == flow_model_module.InputSpec("b", int, False, flow_model_module._UNSET_FLOW_INPUT, 2, "runtime")


def test_flow_inspect_inputs_keep_dependency_value_but_use_compact_repr():
    @Flow.model
    def add(a: int, b: int) -> int:
        return a + b

    child = add(a=1, b=1)
    model = add(a=1, b=child)

    spec = model.flow.inspect().inputs["b"]
    assert spec.value is child
    assert spec.value_repr == "<dependency _add_Model>"
    assert "meta=" not in repr(spec)
    assert "value=<dependency _add_Model>" in repr(spec)


def test_flow_inspect_with_runtime_values_is_structural_not_a_validator():
    @Flow.model
    def add(a: int, b: FromContext[int]) -> int:
        return a + b

    model = add(a=10)

    inspection = model.flow.inspect(b=2, unused=1)
    assert inspection.inputs["b"].value == 2
    assert inspection.inputs["b"].source == "runtime"
    assert "unused" not in inspection.inputs

    regular_inspection = model.flow.inspect(a=1, b=2)
    assert regular_inspection.inputs["b"].value == 2
    assert regular_inspection.inputs["b"].source == "runtime"
    assert regular_inspection.inputs["a"].value == 10
    assert "input check" not in repr(regular_inspection)

    missing_inspection = model.flow.inspect(FlowContext())
    assert missing_inspection.inputs["b"].required
    assert flow_model_module._is_unset_flow_input(missing_inspection.inputs["b"].value)


def test_flow_inspect_reports_direct_dependencies_and_unused_context():
    @Flow.model
    def source(value: FromContext[int]) -> int:
        return value * 2

    @Flow.model
    def root(x: int, bonus: FromContext[int]) -> int:
        return x + bonus

    model = root(x=source())
    explanation = model.flow.inspect(value=3, bonus=4, unused=5)

    assert explanation.inputs["x"].value is model.flow.inspect().inputs["x"].value
    assert explanation.required_inputs == {"bonus": int}
    assert len(explanation.dependencies) == 1
    assert explanation.dependencies[0].path == "x"
    assert explanation.dependencies[0].context == FlowContext(value=3)
    child_inspection = explanation.dependencies[0].model.flow.inspect(explanation.dependencies[0].context)
    assert child_inspection.runtime_inputs == {"value": int}
    assert child_inspection.inputs["value"].value == 3
    assert "dependencies" in str(explanation)
    assert repr(explanation) == str(explanation)
    assert "FlowInspection(model=_root_Model)" in repr(explanation)
    assert "inputs:" in repr(explanation)
    assert "x -> _source_Model context=FlowContext(value=3)" in repr(explanation)

    assert model.flow.inspect(dependencies="none").dependencies == ()
    assert model.flow.inspect().runtime_inputs == {"bonus": int}
    assert set(model.flow.inspect().bound_inputs) == {"x"}
    with pytest.raises(ValueError, match="dependencies must be one of"):
        model.flow.inspect(dependencies="full")


def test_flow_with_context_returns_new_bound_model_without_mutating_source():
    @Flow.model(auto_unwrap=False)
    def add(x: int, y: int, z: FromContext[int] = 2) -> int:
        return x + y + z

    @Flow.context_transform()
    def shift_1(z: FromContext[int]) -> int:
        return z + 1

    model = add(x=1, y=add(x=1, y=1))
    bound = model.flow.with_context(z=shift_1())

    assert model.flow.compute(z=2).value == 7
    assert bound.flow.compute(z=2).value == 9
    assert model.flow.inspect().bound_inputs.keys() == {"x", "y"}
    assert bound.flow.inspect().runtime_inputs == {"z": int}


def test_bound_model_inspect_reports_wrapped_argument_dependencies():
    @Flow.model(auto_unwrap=False)
    def add(x: int, y: int, z: FromContext[int] = 2) -> int:
        return x + y + z

    @Flow.context_transform()
    def shift_1(z: FromContext[int]) -> int:
        return z + 1

    bound = add(x=1, y=add(x=1, y=1)).flow.with_context(z=shift_1())

    inspection = bound.flow.inspect(z=2)

    assert "FlowInspection(model=_add_Model.flow.with_context(...))" in repr(inspection)
    assert inspection.inputs["z"] == flow_model_module.InputSpec("z", int, False, 2, 3, "context_transform")
    assert len(inspection.dependencies) == 1
    assert inspection.dependencies[0].path == "y"
    assert inspection.dependencies[0].context == FlowContext(z=3)
    assert "<wrapped>" not in repr(inspection)

    root = add(x=1, y=bound)
    root_inspection = root.flow.inspect()
    assert root_inspection.required_inputs == {}
    assert root_inspection.dependencies[0].model.flow.inspect().required_inputs == {"z": int}

    root_with_context = root.flow.inspect(z=2)
    child_with_context = root_with_context.dependencies[0].model.flow.inspect(root_with_context.dependencies[0].context)
    assert child_with_context.runtime_inputs == {"z": int}
    assert child_with_context.inputs["z"].value == 3
    assert "context=FlowContext(z=2)" in repr(root_with_context)


def test_flow_inspect_direct_dependencies_do_not_traverse_grandchildren():
    @Flow.model
    def leaf(v: FromContext[int]) -> int:
        return v

    @Flow.model
    def middle(x: int) -> int:
        return x

    @Flow.model
    def root(x: int) -> int:
        return x

    model = root(x=middle(x=leaf()))

    root_inspection = model.flow.inspect()
    child_inspection = root_inspection.dependencies[0].model.flow.inspect()

    assert root_inspection.dependencies[0].path == "x"
    assert child_inspection.dependencies[0].model.flow.inspect().required_inputs == {"v": int}


def test_flow_inspect_recursive_dependencies_report_grandchild_requirements():
    @Flow.model
    def leaf(v: FromContext[int]) -> int:
        return v

    @Flow.model
    def middle(x: int) -> int:
        return x

    @Flow.model
    def root(x: int) -> int:
        return x

    model = root(x=middle(x=leaf()))

    missing = model.flow.inspect(dependencies="recursive")

    assert tuple(dependency.path for dependency in missing.dependencies) == ("x", "x.x")
    assert missing.dependencies[1].model.flow.inspect().required_inputs == {"v": int}

    supplied = model.flow.inspect(v=2, dependencies="recursive")

    assert supplied.inputs == model.flow.inspect(v=2, dependencies="none").inputs
    assert tuple(dependency.path for dependency in model.flow.inspect(v=2, dependencies="direct").dependencies) == ("x",)
    assert tuple(dependency.context for dependency in supplied.dependencies) == (FlowContext(v=2), FlowContext(v=2))
    assert "v" not in supplied.inputs
    leaf_with_context = supplied.dependencies[1].model.flow.inspect(supplied.dependencies[1].context)
    assert leaf_with_context.inputs["v"].value == 2


def test_flow_inspect_recursive_dependencies_treat_bound_transform_inputs_as_used():
    @Flow.model
    def leaf(v: FromContext[int]) -> int:
        return v

    @Flow.model
    def middle(x: int) -> int:
        return x

    @Flow.context_transform()
    def shift(seed: FromContext[int]) -> int:
        return seed + 1

    model = middle(x=leaf().flow.with_context(v=shift()))

    missing = model.flow.inspect(dependencies="recursive")
    assert missing.dependencies[0].model.flow.inspect().required_inputs == {"seed": int}

    supplied = model.flow.inspect(seed=1, dependencies="recursive")
    assert tuple(dependency.path for dependency in supplied.dependencies) == ("x",)
    assert flow_model_module._context_values(supplied.dependencies[0].context) == {"seed": 1}
    child = supplied.dependencies[0].model.flow.inspect(supplied.dependencies[0].context)
    assert child.inputs["v"].value == 2


def test_flow_inspect_projects_bound_dependency_context_to_runtime_inputs():
    @Flow.model
    def leaf(v: FromContext[int]) -> int:
        return v

    @Flow.model
    def root(x: int, bonus: FromContext[int]) -> int:
        return x + bonus

    @Flow.context_transform()
    def shift(seed: FromContext[int]) -> int:
        return seed + 1

    model = root(x=leaf().flow.with_context(v=shift()))
    inspection = model.flow.inspect(seed=1, bonus=10, unused=99)

    assert flow_model_module._context_values(inspection.dependencies[0].context) == {"seed": 1}
    child = inspection.dependencies[0].model.flow.inspect(inspection.dependencies[0].context)
    assert child.inputs["v"].value == 2


def test_flow_inspect_dependency_requirements_skip_lazy_edges():
    @Flow.model
    def leaf(v: FromContext[int]) -> int:
        return v

    @Flow.model
    def root(x: Lazy[int]) -> int:
        return 0

    inspection = root(x=leaf()).flow.inspect(dependencies="recursive")

    assert inspection.dependencies[0].lazy
    assert inspection.dependencies[0].model.flow.inspect().required_inputs == {"v": int}
    assert "x -> _leaf_Model lazy" in repr(inspection)


def test_flow_inspect_dependency_requirements_skip_lazy_descendants():
    @Flow.model
    def leaf(v: FromContext[int]) -> int:
        return v

    @Flow.model
    def middle(x: int) -> int:
        return x

    @Flow.model
    def root(x: Lazy[int]) -> int:
        return 0

    model = root(x=middle(x=leaf()))
    inspection = model.flow.inspect(dependencies="recursive")

    assert model.flow.compute().value == 0
    assert tuple(dependency.path for dependency in inspection.dependencies) == ("x", "x.x")
    assert all(dependency.lazy for dependency in inspection.dependencies)
    assert inspection.dependencies[1].model.flow.inspect().required_inputs == {"v": int}


def test_flow_inspect_recursive_dependencies_tolerates_partial_transform_context():
    @Flow.model
    def leaf(v: FromContext[int]) -> int:
        return v

    @Flow.model
    def middle(x: int) -> int:
        return x

    @Flow.context_transform()
    def shift(seed: FromContext[int], other: FromContext[int]) -> int:
        return seed + other

    model = middle(x=leaf().flow.with_context(v=shift()))

    inspection = model.flow.inspect(seed=1, dependencies="recursive")

    assert tuple(dependency.path for dependency in inspection.dependencies) == ("x",)
    child_inspection = inspection.dependencies[0].model.flow.inspect(inspection.dependencies[0].context)
    assert child_inspection.runtime_inputs == {"seed": int, "other": int}
    assert child_inspection.inputs["v"].source == "context_transform"
    assert flow_model_module._is_unset_flow_input(child_inspection.inputs["v"].value)


def test_bound_flow_inspect_tolerates_partial_transform_context_without_dependencies():
    @Flow.model
    def leaf(v: FromContext[int]) -> int:
        return v

    @Flow.context_transform()
    def shift(seed: FromContext[int], other: FromContext[int]) -> int:
        return seed + other

    bound = leaf().flow.with_context(v=shift())

    inspection = bound.flow.inspect(seed=1, dependencies="none")

    assert inspection.runtime_inputs == {"seed": int, "other": int}
    assert inspection.required_inputs == {"seed": int, "other": int}
    assert inspection.inputs["v"].source == "context_transform"
    assert flow_model_module._is_unset_flow_input(inspection.inputs["v"].value)
    assert inspection.dependencies == ()


def test_flow_inspect_tolerates_partial_plain_callable_dependency_context():
    class PlainSource(CallableModel):
        @property
        def context_type(self):
            return SimpleContext

        @property
        def result_type(self):
            return GenericResult[int]

        @Flow.call
        def __call__(self, context: SimpleContext) -> GenericResult[int]:
            return GenericResult(value=context.value)

    @Flow.model
    def root(x: int, bonus: FromContext[int]) -> int:
        return x + bonus

    inspection = root(x=PlainSource()).flow.inspect(bonus=1)

    assert len(inspection.dependencies) == 1
    assert inspection.dependencies[0].context == FlowContext()
    assert inspection.dependencies[0].model.flow.inspect().required_inputs == {"value": int}


def test_plain_callable_flow_inspect_reports_partial_runtime_context():
    class PairContext(ContextBase):
        a: int
        b: int

    class PlainAdder(CallableModel):
        @property
        def context_type(self):
            return PairContext

        @property
        def result_type(self):
            return GenericResult[int]

        @Flow.call
        def __call__(self, context: PairContext) -> GenericResult[int]:
            return GenericResult(value=context.a + context.b)

    inspection = PlainAdder().flow.inspect(a=1)

    assert inspection.inputs["a"].value == 1
    assert flow_model_module._is_unset_flow_input(inspection.inputs["b"].value)


def test_bound_flow_inspect_uses_static_context_for_dependency_requirements():
    @Flow.model(auto_unwrap=False)
    def leaf(v: FromContext[int]) -> int:
        return v

    @Flow.model(auto_unwrap=False)
    def root(x: int, v: FromContext[int] = 0) -> int:
        return x + 1

    bound = root(x=leaf()).flow.with_context(v=1)
    inspection = bound.flow.inspect()

    assert bound.flow.compute().value == 2
    assert inspection.dependencies[0].context == FlowContext(v=1)
    assert inspection.dependencies[0].model.flow.inspect(inspection.dependencies[0].context).inputs["v"].value == 1


def test_flow_inspect_recursive_dependencies_do_not_swallow_transform_type_errors():
    @Flow.model
    def leaf(v: FromContext[int]) -> int:
        return v

    @Flow.model
    def root(x: int) -> int:
        return x

    @Flow.context_transform()
    def broken(seed: FromContext[int]) -> int:
        raise TypeError("transform bug")

    model = root(x=leaf().flow.with_context(v=broken()))

    with pytest.raises(TypeError, match="transform bug"):
        model.flow.inspect(seed=1, dependencies="recursive")


def test_dependency_evaluation_preserves_original_exception_type_with_context_note():
    class CustomDependencyError(RuntimeError):
        pass

    @Flow.model
    def child() -> int:
        raise CustomDependencyError("boom")

    @Flow.model
    def root(x: int) -> int:
        return x

    with pytest.raises(CustomDependencyError) as exc_info:
        root(x=child()).flow.compute()

    if hasattr(exc_info.value, "__notes__"):
        assert exc_info.value.__notes__ == ["Error while evaluating dependency root.x -> _child_Model."]


def test_generated_factory_signature_is_keyword_only_and_includes_model_base_fields():
    sig = inspect.signature(basic_loader)

    assert all(param.kind is inspect.Parameter.KEYWORD_ONLY for param in sig.parameters.values())
    assert list(sig.parameters) == ["source", "multiplier", "value"]
    assert sig.parameters["source"].default is flow_model_module._UNSET_FLOW_INPUT
    assert sig.parameters["multiplier"].default is flow_model_module._UNSET_FLOW_INPUT
    assert sig.parameters["value"].default is flow_model_module._UNSET_FLOW_INPUT

    with pytest.raises(TypeError, match="positional"):
        basic_loader("library", 3)

    class CustomFlowBase(CallableModel):
        multiplier: int = 1

        @Flow.call
        def __call__(self, context: SimpleContext) -> GenericResult[int]:
            return GenericResult(value=context.value)

    @Flow.model(model_base=CustomFlowBase)
    def add(a: int, b: FromContext[int]) -> int:
        return a + b

    custom_sig = inspect.signature(add)
    assert custom_sig.parameters["a"].kind is inspect.Parameter.KEYWORD_ONLY
    assert custom_sig.parameters["b"].kind is inspect.Parameter.KEYWORD_ONLY
    assert custom_sig.parameters["multiplier"].kind is inspect.Parameter.KEYWORD_ONLY
    assert custom_sig.parameters["multiplier"].default == 1


def test_context_transform_factory_signature_only_exposes_regular_bindings():
    sig = inspect.signature(increment_b)

    assert list(sig.parameters) == ["amount"]
    assert sig.parameters["amount"].kind is inspect.Parameter.KEYWORD_ONLY
    assert sig.parameters["amount"].annotation is int
    assert sig.parameters["amount"].default is inspect.Parameter.empty
    assert sig.return_annotation is inspect.Signature.empty

    with pytest.raises(TypeError, match="positional"):
        increment_b(1)


def test_plain_callable_flow_api_paths():
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

    assert dir(model.flow) == ["compute", "inspect", "with_context"]
    assert not hasattr(model.flow, "context_inputs")
    assert not hasattr(model.flow, "runtime_inputs")
    assert not hasattr(model.flow, "required_inputs")
    assert not hasattr(model.flow, "bound_inputs")
    assert model.flow.inspect().context_inputs == {"value": int}
    assert model.flow.inspect().required_inputs == {"value": int}
    assert model.flow.inspect().bound_inputs == {}
    assert model.flow.compute({"value": 3}).value == 3

    with pytest.raises(TypeError, match="either one context object or contextual keyword inputs"):
        model.flow.compute(SimpleContext(value=1), value=2)


def test_plain_callable_flow_api_is_base_property():
    class PlainModel(CallableModel):
        offset: int = 7

        @Flow.call
        def __call__(self, context: SimpleContext) -> GenericResult[int]:
            return GenericResult(value=self.offset + context.value)

    model = PlainModel()

    assert dir(model.flow) == ["compute", "inspect", "with_context"]
    assert model(SimpleContext(value=3)).value == 10


def test_plain_callable_flow_compute_preserves_matching_context_subclass():
    class RequestContext(SimpleContext):
        request_id: str

    class PlainModel(CallableModel):
        @Flow.call
        def __call__(self, context: SimpleContext) -> GenericResult[tuple[int, str]]:
            return GenericResult(value=(context.value, getattr(context, "request_id", "missing")))

    context = RequestContext(value=3, request_id="abc")

    model = PlainModel()

    assert model.flow.compute(context).value == (3, "abc")
    assert model.flow.with_context().flow.compute(context).value == (3, "abc")

    bound = model.flow.with_context(value=4)
    other_context = RequestContext(value=3, request_id="def")

    assert bound.flow.compute(context).value == (4, "abc")
    assert bound.flow.compute(other_context).value == (4, "def")
    first_key = cache_key(bound.__call__.get_evaluation_context(bound, context), effective=True)
    second_key = cache_key(bound.__call__.get_evaluation_context(bound, other_context), effective=True)
    assert first_key != second_key


def test_plain_callable_flow_compute_uses_default_context_when_available():
    class DefaultContext(ContextBase):
        value: int
        tag: str = "default"

    class PlainModel(CallableModel):
        @Flow.call
        def __call__(self, context: DefaultContext = DefaultContext(value=7)) -> GenericResult[tuple[int, str]]:
            return GenericResult(value=(context.value, context.tag))

    model = PlainModel()

    assert model.flow.inspect().required_inputs == {}
    assert model.flow.compute().value == (7, "default")
    assert model.flow.compute(value=3).value == (3, "default")
    assert model.flow.compute(tag="runtime").value == (7, "runtime")

    empty_bound = model.flow.with_context()
    assert empty_bound.flow.inspect().required_inputs == {}
    assert empty_bound.flow.compute().value == (7, "default")

    bound = model.flow.with_context(tag="bound")
    assert bound.flow.inspect().required_inputs == {}
    assert bound.flow.compute().value == (7, "bound")
    assert bound.flow.compute(value=3).value == (3, "bound")


def test_plain_callable_default_context_private_state_is_preserved():
    class DefaultContext(ContextBase):
        value: int
        _bonus: int = PrivateAttr(default=1)

    default_context = DefaultContext(value=7)
    default_context._bonus = 40

    class PlainModel(CallableModel):
        @Flow.call
        def __call__(self, context: DefaultContext = default_context) -> GenericResult[tuple[int, int, bool]]:
            return GenericResult(value=(context.value, context._bonus, context is default_context))

    model = PlainModel()

    assert model.flow.compute().value == (7, 40, True)
    assert model.flow.compute(value=8).value == (8, 40, False)
    assert model.flow.with_context().flow.compute().value == (7, 40, True)
    assert model.flow.with_context(value=8).flow.compute().value == (8, 40, False)


def test_bound_plain_callable_flow_compute_uses_default_context_for_dynamic_transforms():
    class DefaultContext(ContextBase):
        value: int
        seed: int

    class PlainModel(CallableModel):
        @Flow.call
        def __call__(self, context: DefaultContext = DefaultContext(value=7, seed=8)) -> GenericResult[tuple[int, int]]:
            return GenericResult(value=(context.value, context.seed))

    @Flow.context_transform
    def from_seed(seed: FromContext[int]) -> int:
        return seed + 1

    bound = PlainModel().flow.with_context(value=from_seed())

    assert bound.flow.inspect().required_inputs == {}
    assert bound.flow.inspect().runtime_inputs == {"seed": int}
    assert bound.flow.compute().value == (9, 8)
    assert bound.flow.compute(seed=10).value == (11, 10)


def test_bound_plain_callable_dependency_uses_default_context_baseline():
    class DefaultContext(ContextBase):
        value: int
        seed: int
        _bonus: int = PrivateAttr(default=1)

    default_context = DefaultContext(value=7, seed=8)
    default_context._bonus = 40

    class PlainModel(CallableModel):
        @Flow.call
        def __call__(self, context: DefaultContext = default_context) -> GenericResult[tuple[int, int, int]]:
            return GenericResult(value=(context.value, context.seed, context._bonus))

    @Flow.context_transform
    def from_seed(seed: FromContext[int]) -> int:
        return seed + 1

    @Flow.model
    def consume(x: tuple[int, int, int]) -> tuple[int, int, int]:
        return x

    static_bound = PlainModel().flow.with_context(value=3)
    assert static_bound.flow.compute().value == (3, 8, 40)
    assert consume(x=static_bound).flow.compute().value == (3, 8, 40)

    dynamic_bound = PlainModel().flow.with_context(value=from_seed())
    assert dynamic_bound.flow.compute().value == (9, 8, 40)
    assert consume(x=dynamic_bound).flow.compute().value == (9, 8, 40)
    assert consume(x=dynamic_bound).flow.compute(seed=10).value == (11, 10, 40)

    model = consume(x=dynamic_bound)
    graph = get_dependency_graph(model.__call__.get_evaluation_context(model, model.context_type()))
    plain_contexts = []
    for evaluation_context in graph.ids.values():
        while isinstance(evaluation_context.context, ModelEvaluationContext):
            evaluation_context = evaluation_context.context
        if isinstance(evaluation_context.model, PlainModel):
            plain_contexts.append(evaluation_context.context)

    assert plain_contexts == [DefaultContext(value=9, seed=8)]
    assert plain_contexts[0]._bonus == 40


def test_unhashable_annotations_still_validate():
    annotation = Annotated[int, []]

    @Flow.model
    def add(x: annotation, y: FromContext[annotation]) -> int:
        return x + y

    assert add(x="2").flow.compute(y="3").value == 5


def test_flow_model_internal_contract_helpers_cover_portable_edge_shapes():
    annotation = Annotated[dict[str, list[GenericResult[int]]], "contract"]
    restored = binding_module._restore_annotation(binding_module._serialize_annotation(annotation))

    assert get_args(restored)[1:] == ("contract",)
    assert binding_module._restore_annotation(binding_module._serialize_annotation(Literal["a", "b"])) == Literal["a", "b"]
    assert binding_module._restore_annotation(binding_module._serialize_annotation(int | None)) == Optional[int]
    assert binding_module._clone_function_without_annotations(5) == 5
    assert repr(get_args(Lazy[int])[1]) == "Lazy"
    assert repr(get_args(FromContext[int])[1]) == "FromContext"
    with pytest.raises(TypeError, match="Lazy is an annotation marker"):
        Lazy()
    with pytest.raises(TypeError, match="Unknown serialized annotation payload"):
        binding_module._restore_annotation(object())
    with pytest.raises(TypeError, match="Unknown serialized annotation payload kind"):
        binding_module._restore_annotation(binding_module._SerializedAnnotation(kind="bad", value=None))
    with pytest.raises(TypeError, match="Unknown Flow.model parameter payload"):
        binding_module._restore_flow_model_param(object())
    with pytest.raises(TypeError, match="Unknown Flow.model config payload"):
        binding_module._restore_flow_model_config(object())

    compatible = binding_module._context_type_annotations_compatible
    assert compatible(Any, int)
    assert compatible(int, Any)
    assert compatible(int | str, int)
    assert compatible(int | None, type(None))
    assert compatible(Literal["a", "b"], Literal["a"])
    assert compatible(str, Literal["a"])
    assert compatible(list[int], list[int])
    assert not compatible(int, int | None)
    assert not compatible(Literal["a"], Literal["b"])
    assert not compatible(Literal["a"], str)
    assert not compatible(list[int], list[str])

    flow_model_module.clear_flow_model_caches()
    unhashable_annotation = Annotated[int, []]
    assert flow_model_module._type_adapter(unhashable_annotation).validate_python("1") == 1
    assert flow_model_module._type_adapter(unhashable_annotation).validate_python("2") == 2
    assert flow_model_module._concrete_context_type(Optional[SimpleContext]) is SimpleContext
    assert flow_model_module._concrete_context_type(int) is None
    assert flow_model_module._bound_field_names(object()) == set()
    assert flow_model_module._expected_type_repr(int | str) == "int | str"
    assert flow_model_module._coerce_value("payload", object(), object(), "Regular parameter").__class__ is object
    assert flow_model_module._unwrap_model_result(3) == 3
    assert flow_model_module._resolve_registry_candidate("__missing_registry_entry__") is None
    assert flow_model_module._registry_candidate_allowed(object(), "literal")
    assert not flow_model_module._registry_candidate_allowed(int, "not-an-int")
    assert flow_model_module._registry_candidate_allowed(int, "3")
    with pytest.raises(ImportError, match="does not have a _generated_model"):
        flow_model_module._new_generated_flow_model_for_pickle("ccflow.GenericResult")
    with pytest.raises(TypeError, match="model_base must be a CallableModel subclass"):
        flow_model_module._resolve_generated_model_bases(int)
    assert flow_model_module._context_transform_identifier(static_patch()) == "static_patch"
    assert flow_model_module._context_transform_repr(static_patch()) == "static_patch()"
    assert flow_model_module._context_transform_repr(increment_b(amount=2)) == "increment_b(amount=2)"
    assert flow_model_module._context_transform_repr(5) == "5"

    @Flow.model
    def dep(value: FromContext[int]) -> int:
        return value

    with pytest.raises(TypeError, match="must return a mapping"):
        flow_model_module._validate_patch_result(dep(), 1)
    with pytest.raises(TypeError, match="string field names"):
        flow_model_module._validate_patch_result(dep(), {1: 2})
    lazy_thunk = flow_model_module._make_lazy_thunk(dep(), FlowContext(value=4))
    assert lazy_thunk() == 4
    assert lazy_thunk() == 4
    coercing_thunk = flow_model_module._make_coercing_lazy_thunk(lambda: "5", "value", int)
    assert coercing_thunk() == 5
    assert coercing_thunk() == 5


def test_compute_accepts_context_object_for_from_context_models():
    model = basic_loader(source="library", multiplier=3)

    assert model.flow.inspect().context_inputs == {"value": int}
    assert model.flow.inspect().required_inputs == {"value": int}
    assert model.flow.compute({"value": 4}).value == 12
    assert model.flow.compute(SimpleContext(value=5)).value == 15

    with pytest.raises(TypeError, match="either one context object or contextual keyword inputs"):
        model.flow.compute(SimpleContext(value=1), value=2)


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

    # Under the default (strict=False) an unconsumed required field is allowed: the declared
    # context acts as an omnibus superset. strict=True restores the full-bijection requirement.
    with pytest.raises(TypeError, match="has required fields that are not declared as FromContext parameters"):

        @Flow.model(context_type=ExtraRequiredContext, strict=True)
        def bad_extra(start_date: FromContext[date], end_date: FromContext[date]) -> int:
            return 0

    @Flow.model(context_type=ExtraRequiredContext)
    def ok_extra(start_date: FromContext[date], end_date: FromContext[date]) -> int:
        return (end_date - start_date).days

    assert ok_extra().flow.compute(ExtraRequiredContext(start_date=date(2025, 1, 1), end_date=date(2025, 1, 8), label="x")).value == 7

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
    def source(value: FromContext[int]) -> GenericResult[int]:
        return GenericResult(value=value)

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

    assert keyword_only_context(context=SimpleContext(value=3), offset=4).flow.compute().value == 7

    def missing_hints(*args, **kwargs):
        raise AttributeError("missing hints")

    monkeypatch.setattr(flow_model_module, "get_type_hints", missing_hints)

    def add(x: int, y: FromContext[int]) -> int:
        return x + y

    with pytest.raises(AttributeError, match="missing hints"):
        Flow.model(add)


def test_unresolved_forward_refs_do_not_silently_strip_from_context():
    namespace: dict[str, object] = {}

    with pytest.raises(NameError, match="MissingType"):
        exec(
            """
from __future__ import annotations
from ccflow import Flow, FromContext

@Flow.context_transform
def transform(a: MissingType, b: FromContext[int]) -> int:
    return b
""",
            namespace,
        )

    with pytest.raises(NameError, match="MissingType"):
        exec(
            """
from __future__ import annotations
from ccflow import Flow, FromContext

@Flow.model
def model(a: MissingType, b: FromContext[int]) -> int:
    return b
""",
            namespace,
        )


def test_context_type_validates_parameterized_annotations():
    class IntListContext(ContextBase):
        vals: list[int]

    @Flow.model(context_type=IntListContext)
    def total(vals: FromContext[list[int]]) -> int:
        return sum(vals)

    assert total().flow.compute(vals=["1", "2"]).value == 3

    class IntDictContext(ContextBase):
        vals: dict[str, int]

    @Flow.model(context_type=IntDictContext)
    def total_dict(vals: FromContext[dict[str, int]]) -> int:
        return sum(vals.values())

    assert total_dict().flow.compute(vals={"a": "1", "b": 2}).value == 3

    class StrListContext(ContextBase):
        vals: list[str]

    with pytest.raises(TypeError, match="annotates"):

        @Flow.model(context_type=StrListContext)
        def bad(vals: FromContext[list[int]]) -> int:
            return sum(vals)


def test_context_type_allows_compatible_union_literal_and_generic_fields():
    class RichContext(ContextBase):
        flag: Literal["a"]
        maybe_value: int | None
        values: list[int]

    @Flow.model(context_type=RichContext)
    def summarize(
        flag: FromContext[Literal["a", "b"]],
        maybe_value: FromContext[int | None],
        values: FromContext[list[int]],
    ) -> int:
        return len(flag) + (maybe_value or 0) + sum(values)

    assert summarize().flow.compute(flag="a", maybe_value=None, values=[1, 2]).value == 4


def test_context_type_rejects_nullable_field_for_non_nullable_from_context():
    class OptionalValueContext(ContextBase):
        value: int | None

    with pytest.raises(TypeError, match="annotates"):

        @Flow.model(context_type=OptionalValueContext)
        def add_one(value: FromContext[int]) -> int:
            return value + 1


def test_compute_forwards_options_with_custom_evaluator():
    calls = {"count": 0}

    @Flow.model
    def counter(value: FromContext[int]) -> int:
        calls["count"] += 1
        return value

    cache = MemoryCacheEvaluator()
    model = counter()

    result1 = model.flow.compute(value=10, _options=FlowOptions(evaluator=cache, cacheable=True))
    result2 = model.flow.compute(value=10, _options=FlowOptions(evaluator=cache, cacheable=True))

    assert result1.value == 10
    assert result2.value == 10
    assert calls["count"] == 1


def test_compute_forwards_options_with_graph_evaluator():
    @Flow.model
    def source(value: FromContext[int]) -> int:
        return value * 10

    @Flow.model
    def root(x: int, bonus: FromContext[int]) -> int:
        return x + bonus

    model = root(x=source())

    # GraphEvaluator evaluates in topo order; verify _options flows through
    # and the graph evaluator is actually used (doesn't raise CycleError, computes correctly)
    result = model.flow.compute(
        FlowContext(value=3, bonus=7),
        _options=FlowOptions(evaluator=GraphEvaluator()),
    )

    assert result.value == 37


def test_compute_forwards_options_through_bound_model():
    calls = {"count": 0}

    @Flow.model
    def add(a: int, b: FromContext[int]) -> int:
        calls["count"] += 1
        return a + b

    cache = MemoryCacheEvaluator()
    bound = add(a=10).flow.with_context(b=5)

    result1 = bound.flow.compute(_options=FlowOptions(evaluator=cache, cacheable=True))
    result2 = bound.flow.compute(_options=FlowOptions(evaluator=cache, cacheable=True))

    assert result1.value == 15
    assert result2.value == 15
    assert calls["count"] == 1


def test_compute_forwards_options_for_plain_callable_model():
    calls = {"count": 0}

    class Counter(CallableModel):
        offset: int

        @Flow.call
        def __call__(self, context: SimpleContext) -> GenericResult[int]:
            calls["count"] += 1
            return GenericResult(value=context.value + self.offset)

    cache = MemoryCacheEvaluator()
    model = Counter(offset=5)

    result1 = model.flow.compute(value=10, _options=FlowOptions(evaluator=cache, cacheable=True))
    result2 = model.flow.compute(value=10, _options=FlowOptions(evaluator=cache, cacheable=True))

    assert result1.value == 15
    assert result2.value == 15
    assert calls["count"] == 1


def test_bound_plain_callable_compute_applies_context_before_validation():
    calls = {"count": 0}

    class RequiredContext(ContextBase):
        a: int
        b: int

    class PlainSource(CallableModel):
        @Flow.call
        def __call__(self, context: RequiredContext) -> GenericResult[int]:
            calls["count"] += 1
            return GenericResult(value=context.a + context.b)

    assert PlainSource().flow.with_context(a=1).flow.compute(b=2).value == 3
    assert calls["count"] == 1


def test_bound_plain_callable_empty_with_context_preserves_optional_none_context():
    calls = {"count": 0}

    class OptionalContext(ContextBase):
        value: int = 1

    class PlainSource(CallableModel):
        @Flow.call
        def __call__(self, context: Optional[OptionalContext] = None) -> GenericResult[int]:
            calls["count"] += 1
            return GenericResult(value=0 if context is None else context.value)

    bound = PlainSource().flow.with_context()

    assert bound.flow.compute().value == 0
    assert bound(None).value == 0
    assert bound({"value": 5}).value == 5
    assert bound.flow.compute(value=5).value == 5
    assert bound.flow.compute(_options=FlowOptions(evaluator=GraphEvaluator())).value == 0
    assert calls["count"] == 5


def test_bound_plain_callable_dynamic_context_transform_runs_before_validation():
    calls = {"count": 0}

    class RequiredContext(ContextBase):
        a: int
        b: int

    class PlainSource(CallableModel):
        @Flow.call
        def __call__(self, context: RequiredContext) -> GenericResult[int]:
            calls["count"] += 1
            return GenericResult(value=context.a + context.b)

    assert PlainSource().flow.with_context(a=seed_plus_one()).flow.compute(seed=1, b=10).value == 12
    assert calls["count"] == 1


def test_bound_plain_callable_direct_call_applies_context_before_validation():
    calls = {"count": 0}

    class RequiredContext(ContextBase):
        a: int
        b: int

    class PlainSource(CallableModel):
        @Flow.call
        def __call__(self, context: RequiredContext) -> GenericResult[int]:
            calls["count"] += 1
            return GenericResult(value=context.a + context.b)

    bound = PlainSource().flow.with_context(a=1)

    assert bound(FlowContext(b=2)).value == 3
    assert bound.__deps__(FlowContext(b=2)) == [(bound.model, [RequiredContext(a=1, b=2)])]
    assert calls["count"] == 1


def test_bound_plain_callable_direct_kwargs_use_flow_context():
    class PlainSource(CallableModel):
        @Flow.call
        def __call__(self, context: FlowContext) -> GenericResult[str]:
            return GenericResult(value=type(context).__name__)

    bound = PlainSource().flow.with_context(a=1)

    assert bound(b=2).value == "FlowContext"
    assert bound.flow.compute(b=2).value == "FlowContext"


def test_bound_plain_callable_compute_preserves_bound_scoped_options():
    calls = {"source": 0, "evaluator": 0}

    class RequiredContext(ContextBase):
        a: int
        b: int

    class PlainSource(CallableModel):
        @Flow.call
        def __call__(self, context: RequiredContext) -> GenericResult[int]:
            calls["source"] += 1
            return GenericResult(value=context.a + context.b)

    class OffsetEvaluator(EvaluatorBase):
        def __call__(self, context: ModelEvaluationContext):
            calls["evaluator"] += 1
            result = context()
            return result.model_copy(update={"value": result.value + 100})

    bound = PlainSource().flow.with_context(a=1)

    with FlowOptionsOverride(options={"evaluator": OffsetEvaluator()}, models=(bound,)):
        assert bound.flow.compute(b=2).value == 103

    assert calls == {"source": 1, "evaluator": 1}


def test_bound_plain_callable_dependency_preserves_bound_scoped_options():
    calls = {"source": 0, "consumer": 0, "evaluator": 0}

    class RequiredContext(ContextBase):
        a: int
        b: int

    class PlainSource(CallableModel):
        @Flow.call
        def __call__(self, context: RequiredContext) -> GenericResult[int]:
            calls["source"] += 1
            return GenericResult(value=context.a + context.b)

    class OffsetEvaluator(EvaluatorBase):
        def __call__(self, context: ModelEvaluationContext):
            calls["evaluator"] += 1
            result = context()
            return result.model_copy(update={"value": result.value + 100})

    @Flow.model
    def consumer(x: int) -> int:
        calls["consumer"] += 1
        return x

    bound = PlainSource().flow.with_context(a=1)
    model = consumer(x=bound)

    with FlowOptionsOverride(options={"evaluator": OffsetEvaluator()}, models=(bound,)):
        assert model.flow.compute(b=2).value == 103

    assert calls == {"source": 1, "consumer": 1, "evaluator": 1}


def test_bound_plain_callable_dependency_identity_ignores_unused_ambient_context():
    calls = {"source": 0, "consumer": 0}

    class RequiredContext(ContextBase):
        a: int
        b: int

    class PlainSource(CallableModel):
        @Flow.call
        def __call__(self, context: RequiredContext) -> GenericResult[int]:
            calls["source"] += 1
            return GenericResult(value=context.a + context.b)

    @Flow.model
    def consumer(x: int) -> int:
        calls["consumer"] += 1
        return x

    cache = MemoryCacheEvaluator()
    model = consumer(x=PlainSource().flow.with_context(a=1, b=2))

    with FlowOptionsOverride(options={"evaluator": cache, "cacheable": True}):
        assert model.flow.compute(unused="one").value == 3
        assert model.flow.compute(unused="two").value == 3

    assert calls == {"source": 1, "consumer": 1}


def test_bound_dependency_identity_rewrites_dynamic_context_once():
    calls = {"source": 0, "consumer": 0}

    @Flow.model
    def source(a: FromContext[int]) -> int:
        calls["source"] += 1
        return a

    @Flow.model
    def consumer(x: int) -> int:
        calls["consumer"] += 1
        return x

    cache = MemoryCacheEvaluator()
    model = consumer(x=source().flow.with_context(a=non_idempotent_a_step()))

    with FlowOptionsOverride(options={"evaluator": cache, "cacheable": True}):
        assert model.flow.compute(a=1).value == 2
        assert model.flow.compute(a=2).value == 3

    assert calls == {"source": 2, "consumer": 2}


def test_lazy_bound_plain_callable_dependency_preserves_bound_scoped_options():
    calls = {"source": 0, "consumer": 0, "evaluator": 0}

    class RequiredContext(ContextBase):
        a: int
        b: int

    class PlainSource(CallableModel):
        @Flow.call
        def __call__(self, context: RequiredContext) -> GenericResult[int]:
            calls["source"] += 1
            return GenericResult(value=context.a + context.b)

    class OffsetEvaluator(EvaluatorBase):
        def __call__(self, context: ModelEvaluationContext):
            calls["evaluator"] += 1
            result = context()
            return result.model_copy(update={"value": result.value + 100})

    @Flow.model
    def consumer(lazy_value: Lazy[int], use_lazy: FromContext[bool]) -> int:
        calls["consumer"] += 1
        return lazy_value() if use_lazy else 0

    bound = PlainSource().flow.with_context(a=1)
    model = consumer(lazy_value=bound)

    with FlowOptionsOverride(options={"evaluator": OffsetEvaluator()}, models=(bound,)):
        assert model.flow.compute(b=2, use_lazy=True).value == 103

    assert calls == {"source": 1, "consumer": 1, "evaluator": 1}


def test_bound_flow_required_inputs_subtracts_static_context():
    class RequiredContext(ContextBase):
        a: int
        b: int

    class PlainSource(CallableModel):
        @Flow.call
        def __call__(self, context: RequiredContext) -> GenericResult[int]:
            return GenericResult(value=context.a + context.b)

    @Flow.model
    def add(a: FromContext[int], b: FromContext[int]) -> int:
        return a + b

    assert PlainSource().flow.with_context(a=1).flow.inspect().required_inputs == {"b": int}
    assert add().flow.with_context(a=1).flow.inspect().required_inputs == {"b": int}
    assert add().flow.with_context(a=static_bad()).flow.inspect().required_inputs == {"b": int}
    assert add().flow.with_context(static_patch()).flow.inspect().required_inputs == {"b": int}
    assert add().flow.with_context(a=1, b=2).flow.inspect().required_inputs == {}


def test_bound_flow_required_inputs_reflects_dynamic_field_transform_inputs():
    @Flow.model
    def add(a: FromContext[int], b: FromContext[int]) -> int:
        return a + b

    bound = add().flow.with_context(a=seed_plus_one())

    assert bound.flow.compute(seed=1, b=10).value == 12
    assert bound.flow.inspect().context_inputs == {"a": int, "b": int}
    assert bound.flow.inspect().runtime_inputs == {"b": int, "seed": int}
    assert bound.flow.inspect().required_inputs == {"b": int, "seed": int}


def test_bound_flow_bound_inputs_include_static_context_bindings():
    @Flow.model
    def add(a: int, b: FromContext[int]) -> int:
        return a + b

    bound = add(a=1).flow.with_context(b=2)

    assert bound.flow.inspect().context_inputs == {"b": int}
    assert bound.flow.inspect().runtime_inputs == {}
    assert bound.flow.inspect().required_inputs == {}
    assert bound.flow.inspect().bound_inputs == {"a": 1, "b": 2}


def test_bound_flow_bound_inputs_drops_static_patch_after_dynamic_override():
    @Flow.model
    def add(a: FromContext[int], b: FromContext[int]) -> int:
        return a + b

    bound = add().flow.with_context(static_patch()).flow.with_context(a=seed_plus_one())

    assert bound.flow.compute(seed=3, b=10).value == 14
    assert bound.flow.inspect().bound_inputs == {}
    assert bound.flow.inspect().context_inputs == {"a": int, "b": int}
    assert bound.flow.inspect().runtime_inputs == {"b": int, "seed": int}
    assert bound.flow.inspect().required_inputs == {"b": int, "seed": int}


def test_generated_model_cache_ignores_unused_flow_context_fields():
    calls = {"source": 0, "root": 0}

    @Flow.model
    def source(value: FromContext[int]) -> int:
        calls["source"] += 1
        return value * 10

    @Flow.model
    def root(x: int, bonus: FromContext[int]) -> int:
        calls["root"] += 1
        return x + bonus

    cache = MemoryCacheEvaluator()
    model = root(x=source())

    with FlowOptionsOverride(options={"evaluator": cache, "cacheable": True}):
        assert model(FlowContext(value=3, bonus=7, unused="one")).value == 37
        assert model(FlowContext(value=3, bonus=7, unused="two")).value == 37

    assert calls == {"source": 1, "root": 1}
    assert len(cache.cache) == 2


def test_generated_model_cache_uses_effective_key_through_transparent_evaluator():
    calls = {"count": 0}

    @Flow.model
    def add(a: int, b: FromContext[int]) -> int:
        calls["count"] += 1
        return a + b

    cache = MemoryCacheEvaluator()
    evaluator = combine_evaluators(LoggingEvaluator(), cache)
    model = add(a=10)

    with FlowOptionsOverride(options={"evaluator": evaluator, "cacheable": True}):
        assert model.flow.compute(b=1, unused="one").value == 11
        assert model.flow.compute(b=1, unused="two").value == 11

    assert calls["count"] == 1
    assert len(cache.cache) == 1


def test_effective_cache_key_ignores_untokenizable_unused_ambient_context():
    class BadToken:
        def __deepcopy__(self, memo):
            return self

        def __getstate__(self):
            raise RuntimeError("unused field should not be tokenized")

    @Flow.model
    def add(a: int, b: FromContext[int]) -> int:
        return a + b

    model = add(a=1)
    clean_context = FlowContext(b=2)
    noisy_context = FlowContext(b=2, unused=BadToken())
    clean_eval = model.__call__.get_evaluation_context(model, clean_context)
    noisy_eval = model.__call__.get_evaluation_context(model, noisy_context)

    assert cache_key(noisy_eval, effective=True) == cache_key(clean_eval, effective=True)


def test_recursive_effective_cache_key_ignores_untokenizable_unused_ambient_context():
    class BadToken:
        def __deepcopy__(self, memo):
            return self

        def __getstate__(self):
            raise RuntimeError("unused field should not be tokenized")

    @Flow.model
    def cycle(value: int, a: FromContext[int]) -> int:
        return a

    model = cycle(value=cycle(value=1))
    model.value = model
    context = FlowContext(a=1, unused=BadToken())
    evaluation = model.__call__.get_evaluation_context(model, context)

    cache_key(evaluation, effective=True)
    graph = get_dependency_graph(evaluation)
    assert graph.root_id in graph.ids
    with pytest.raises(graphlib.CycleError):
        tuple(graphlib.TopologicalSorter(graph.graph).static_order())


def test_cache_key_effective_option_preserves_plain_callable_structural_identity():
    calls = {"count": 0}

    class Counter(CallableModel):
        @Flow.call
        def __call__(self, context: FlowContext) -> GenericResult[int]:
            calls["count"] += 1
            return GenericResult(value=context.value)

    model = Counter()
    eval1 = model.__call__.get_evaluation_context(model, FlowContext(value=10, unused="one"))
    eval2 = model.__call__.get_evaluation_context(model, FlowContext(value=10, unused="two"))

    assert cache_key(eval1) != cache_key(eval2)
    assert cache_key(eval1, effective=True) == cache_key(eval1)
    assert cache_key(eval2, effective=True) == cache_key(eval2)


def test_generated_model_effective_cache_key_includes_behavior_token(monkeypatch):
    @Flow.model
    def add(a: int, b: FromContext[int]) -> int:
        return a + b

    def helper_v1():
        return 1

    def helper_v2():
        return 2

    model = add(a=10)
    model_type = type(model)
    context = model.__call__.get_evaluation_context(model, FlowContext(b=1, unused="same"), _options={"cacheable": True})
    cache = MemoryCacheEvaluator()

    monkeypatch.setattr(model_type, "__ccflow_tokenizer_deps__", [helper_v1], raising=False)
    key1 = cache.key(context)

    monkeypatch.setattr(model_type, "__ccflow_tokenizer_deps__", [helper_v2], raising=False)
    monkeypatch.delattr(model_type, "__ccflow_tokenizer_cache__", raising=False)
    key2 = cache.key(context)

    assert key1 != key2


def test_generated_model_cache_changes_when_consumed_context_field_changes():
    calls = {"count": 0}

    @Flow.model
    def add(a: int, b: FromContext[int]) -> int:
        calls["count"] += 1
        return a + b

    cache = MemoryCacheEvaluator()
    model = add(a=10)

    with FlowOptionsOverride(options={"evaluator": cache, "cacheable": True}):
        assert model.flow.compute(b=1, unused="same").value == 11
        assert model.flow.compute(b=2, unused="same").value == 12

    assert calls["count"] == 2
    assert len(cache.cache) == 2


def test_generated_model_cache_changes_when_regular_literal_input_changes():
    calls = {"count": 0}

    @Flow.model
    def add(a: int, b: FromContext[int]) -> int:
        calls["count"] += 1
        return a + b

    cache = MemoryCacheEvaluator()

    with FlowOptionsOverride(options={"evaluator": cache, "cacheable": True}):
        assert add(a=10).flow.compute(b=1).value == 11
        assert add(a=20).flow.compute(b=1).value == 21

    assert calls["count"] == 2
    assert len(cache.cache) == 2


def test_generated_model_cache_does_not_ignore_context_read_by_nontransparent_evaluator():
    class AddAmbient(EvaluatorBase):
        def __call__(self, context: ModelEvaluationContext):
            result = context()
            return result.model_copy(update={"value": result.value + context.context.unused})

    @Flow.model
    def add(a: int, b: FromContext[int]) -> int:
        return a + b

    cache = MemoryCacheEvaluator()
    evaluator = combine_evaluators(AddAmbient(), cache)
    model = add(a=10)

    with FlowOptionsOverride(options={"evaluator": evaluator, "cacheable": True}):
        assert model.flow.compute(b=1, unused=100).value == 111
        assert model.flow.compute(b=1, unused=200).value == 211

    assert len(cache.cache) == 2


def test_generated_model_cache_key_preserves_result_validation_option():
    calls = {"count": 0}

    @Flow.model
    def raw_result(value: FromContext[int]) -> GenericResult[int]:
        calls["count"] += 1
        return {"value": str(value)}

    cache = MemoryCacheEvaluator()
    model = raw_result()

    with FlowOptionsOverride(options=FlowOptions(evaluator=cache, cacheable=True, validate_result=False)):
        first = model.flow.compute(value=3, unused="one")

    with FlowOptionsOverride(options=FlowOptions(evaluator=cache, cacheable=True, validate_result=True)):
        second = model.flow.compute(value=3, unused="two")

    assert first == {"value": "3"}
    assert second == GenericResult[int](value=3)
    assert calls["count"] == 2
    assert len(cache.cache) == 2


def test_generated_model_cache_ignores_unresolved_unused_lazy_dependency_context():
    calls = {"choose": 0}

    @Flow.model
    def source(missing: FromContext[int]) -> int:
        return missing * 10

    @Flow.model
    def choose(x: int, lazy_value: Lazy[int], use_lazy: FromContext[bool]) -> int:
        calls["choose"] += 1
        return lazy_value() if use_lazy else x

    cache = MemoryCacheEvaluator()
    model = choose(x=7, lazy_value=source())

    with FlowOptionsOverride(options={"evaluator": cache, "cacheable": True}):
        assert model.flow.compute(use_lazy=False, unused="one").value == 7
        assert model.flow.compute(use_lazy=False, unused="two").value == 7

    assert calls["choose"] == 1
    assert len(cache.cache) == 1


def test_unused_lazy_plain_dependency_defers_missing_context_validation():
    calls = {"source": 0, "choose": 0}

    class RequiredContext(ContextBase):
        missing: int

    class PlainSource(CallableModel):
        @Flow.call
        def __call__(self, context: RequiredContext) -> GenericResult[int]:
            calls["source"] += 1
            return GenericResult(value=context.missing * 10)

    @Flow.model
    def choose(x: int, lazy_value: Lazy[int], use_lazy: FromContext[bool]) -> int:
        calls["choose"] += 1
        return lazy_value() if use_lazy else x

    cache = MemoryCacheEvaluator()
    model = choose(x=7, lazy_value=PlainSource())

    with FlowOptionsOverride(options={"evaluator": cache, "cacheable": True}):
        assert model.flow.compute(use_lazy=False, unused="one").value == 7
        assert model.flow.compute(use_lazy=False, unused="two").value == 7
        with pytest.raises(ValidationError):
            model.flow.compute(use_lazy=True)

    assert calls == {"source": 0, "choose": 2}
    assert len(cache.cache) == 1


def test_unused_lazy_bound_dependency_uses_partial_context_identity():
    calls = {"source": 0, "choose": 0}

    @Flow.model
    def source(a: FromContext[int], b: FromContext[int]) -> int:
        calls["source"] += 1
        return a + b

    @Flow.model
    def choose(x: int, lazy_value: Lazy[int], use_lazy: FromContext[bool]) -> int:
        calls["choose"] += 1
        return lazy_value() if use_lazy else x

    cache = MemoryCacheEvaluator()
    model = choose(x=7, lazy_value=source().flow.with_context(a=1))

    with FlowOptionsOverride(options={"evaluator": cache, "cacheable": True}):
        assert model.flow.compute(use_lazy=False, a=100).value == 7
        assert model.flow.compute(use_lazy=False, a=200).value == 7
        with pytest.raises(TypeError, match="Missing contextual input"):
            model.flow.compute(use_lazy=True, a=300)

    assert calls == {"source": 0, "choose": 2}
    assert len(cache.cache) == 1


def test_unused_lazy_bound_dependency_with_unresolved_transform_has_stable_identity():
    calls = {"source": 0, "choose": 0}

    @Flow.model
    def source(a: FromContext[int], b: FromContext[int]) -> int:
        calls["source"] += 1
        return a + b

    @Flow.context_transform
    def a_from_seed(seed: FromContext[int]) -> int:
        return seed + 1

    @Flow.model
    def choose(x: int, lazy_value: Lazy[int], use_lazy: FromContext[bool]) -> int:
        calls["choose"] += 1
        return lazy_value() if use_lazy else x

    cache = MemoryCacheEvaluator()
    model = choose(x=7, lazy_value=source().flow.with_context(a=a_from_seed()))

    with FlowOptionsOverride(options={"evaluator": cache, "cacheable": True}):
        assert model.flow.compute(use_lazy=False, b=1, unused="one").value == 7
        assert model.flow.compute(use_lazy=False, b=2, unused="two").value == 7
        with pytest.raises(TypeError, match="Missing contextual input"):
            model.flow.compute(use_lazy=True, b=3)

    assert calls == {"source": 0, "choose": 2}
    assert len(cache.cache) == 1


def test_unused_lazy_resolved_dependency_identity_is_conservative():
    calls = {"source": 0, "choose": 0}

    @Flow.model
    def source(a: FromContext[int]) -> int:
        calls["source"] += 1
        return a

    @Flow.model
    def choose(x: int, lazy_value: Lazy[int], use_lazy: FromContext[bool]) -> int:
        calls["choose"] += 1
        return lazy_value() if use_lazy else x

    cache = MemoryCacheEvaluator()

    with FlowOptionsOverride(options={"evaluator": cache, "cacheable": True}):
        assert choose(x=7, lazy_value=source().flow.with_context(a=1)).flow.compute(use_lazy=False).value == 7
        assert choose(x=7, lazy_value=source().flow.with_context(a=2)).flow.compute(use_lazy=False).value == 7

    assert calls == {"source": 0, "choose": 2}
    assert len(cache.cache) == 2


def test_used_lazy_bound_dependency_identity_applies_dynamic_context_transform():
    calls = {"source": 0, "choose": 0}

    @Flow.model
    def source(a: FromContext[int], b: FromContext[int]) -> int:
        calls["source"] += 1
        return a + b

    @Flow.model
    def choose(lazy_value: Lazy[int], use_lazy: FromContext[bool]) -> int:
        calls["choose"] += 1
        return lazy_value() if use_lazy else 0

    cache = MemoryCacheEvaluator()
    model = choose(lazy_value=source().flow.with_context(a=seed_plus_one()))

    with FlowOptionsOverride(options={"evaluator": cache, "cacheable": True}):
        assert model.flow.compute(use_lazy=True, seed=1, b=10).value == 12
        assert model.flow.compute(use_lazy=True, seed=2, b=10).value == 13

    assert calls == {"source": 2, "choose": 2}


def test_used_lazy_generated_dependency_identity_respects_contextual_defaults():
    calls = {"dep": 0, "source": 0, "choose": 0}

    @Flow.model
    def dep(v: FromContext[int]) -> int:
        calls["dep"] += 1
        return v

    @Flow.model
    def source(d: int, a: FromContext[int], b: FromContext[int]) -> int:
        calls["source"] += 1
        return a + b + d

    @Flow.model
    def choose(lazy_value: Lazy[int], use_lazy: FromContext[bool]) -> int:
        calls["choose"] += 1
        return lazy_value() if use_lazy else 0

    cache = MemoryCacheEvaluator()
    model = choose(lazy_value=source(d=dep(), a=1))

    with FlowOptionsOverride(options={"evaluator": cache, "cacheable": True}):
        assert model.flow.compute(use_lazy=True, b=10, v=3).value == 14
        assert model.flow.compute(use_lazy=True, b=10, v=999).value == 1010

    assert calls == {"dep": 2, "source": 2, "choose": 2}


def test_generated_model_cache_distinguishes_unresolved_lazy_dependency_models():
    calls = {"choose": 0}

    @Flow.model
    def source_one(missing: FromContext[int]) -> int:
        return missing * 10

    @Flow.model
    def source_two(missing: FromContext[int]) -> int:
        return missing * 100

    @Flow.model
    def choose(x: int, lazy_value: Lazy[int], use_lazy: FromContext[bool]) -> int:
        calls["choose"] += 1
        return lazy_value() if use_lazy else x

    cache = MemoryCacheEvaluator()

    with FlowOptionsOverride(options={"evaluator": cache, "cacheable": True}):
        assert choose(x=7, lazy_value=source_one()).flow.compute(use_lazy=False).value == 7
        assert choose(x=7, lazy_value=source_two()).flow.compute(use_lazy=False).value == 7

    assert calls["choose"] == 2
    assert len(cache.cache) == 2


def test_bound_generated_sibling_dependencies_keep_distinct_rewritten_contexts_with_graph_cache():
    calls = {"source": 0, "root": 0}

    @Flow.model
    def source(value: FromContext[int]) -> int:
        calls["source"] += 1
        return value

    @Flow.model
    def root(left: int, right: int) -> int:
        calls["root"] += 1
        return left + right

    cache = MemoryCacheEvaluator()
    evaluator = combine_evaluators(cache, GraphEvaluator())
    shared = source()
    model = root(left=shared.flow.with_context(value=1), right=shared.flow.with_context(value=2))

    with FlowOptionsOverride(options={"evaluator": evaluator, "cacheable": True}):
        assert model.flow.compute(value=99, unused="ambient").value == 3

    assert calls == {"source": 2, "root": 1}
    assert len(cache.cache) >= 3


def test_bound_generated_model_dependency_graph_traverses_collapsed_child_deps():
    @Flow.model
    def source(value: FromContext[int]) -> int:
        return value * 10

    @Flow.model
    def root(x: int, bonus: FromContext[int]) -> int:
        return x + bonus

    bound = root(x=source()).flow.with_context(bonus=1)
    graph = get_dependency_graph(bound.__call__.get_evaluation_context(bound, FlowContext(value=2, bonus=99)))

    assert graph.root_id not in graph.graph[graph.root_id]
    assert len(graph.ids) == 3
    assert len(graph.graph[graph.root_id]) == 1


def test_bound_model_cache_follows_rewritten_context_not_ambient_source_fields():
    calls = {"count": 0}

    @Flow.model
    def add(a: int, b: FromContext[int]) -> int:
        calls["count"] += 1
        return a + b

    cache = MemoryCacheEvaluator()
    bound = add(a=10).flow.with_context(b=parity_bucket())

    with FlowOptionsOverride(options={"evaluator": cache, "cacheable": True}):
        assert bound(FlowContext(raw=1)).value == 11
        assert bound(FlowContext(raw=3)).value == 11

    assert calls["count"] == 1
    assert len(cache.cache) == 2


def test_bound_model_cache_respects_wrapped_model_scoped_evaluator():
    calls = {"add": 0, "evaluator": 0}

    class OffsetEvaluator(EvaluatorBase):
        def __call__(self, context: ModelEvaluationContext):
            calls["evaluator"] += 1
            result = context()
            return result.model_copy(update={"value": result.value + 100})

    @Flow.model
    def add(a: int, b: FromContext[int]) -> int:
        calls["add"] += 1
        return a + b

    cache = MemoryCacheEvaluator()
    base = add(a=1)
    bound = base.flow.with_context(b=2)

    with FlowOptionsOverride(options={"evaluator": cache, "cacheable": True}):
        with FlowOptionsOverride(options={"evaluator": OffsetEvaluator()}, models=(base,)):
            assert bound.flow.compute().value == 103
        assert bound.flow.compute().value == 3

    assert calls == {"add": 2, "evaluator": 1}


def test_generated_models_cross_process_pickle():
    """Module-level @Flow.model instances are deserializable in a separate process."""
    model = basic_loader(source="library", multiplier=3)
    data = pickle.dumps(model, protocol=5)
    encoded = base64.b64encode(data).decode()
    script = (
        "import pickle, base64\n"
        f"data = base64.b64decode('{encoded}')\n"
        "model = pickle.loads(data)\n"
        "from ccflow import FlowContext\n"
        "result = model.flow.compute(value=4)\n"
        "assert result.value == 12, f'Expected 12, got {result.value}'\n"
    )
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, f"Cross-process unpickle failed:\n{result.stderr}"


def test_local_generated_models_cross_process_cloudpickle():
    """Local @Flow.model instances carry their generated class across processes."""

    def make_model():
        @Flow.model
        def add(a: int, b: FromContext[int]) -> int:
            return a + b

        return add(a=1)

    encoded = base64.b64encode(cloudpickle.dumps(make_model(), protocol=5)).decode()
    script = (
        "import base64\n"
        "import cloudpickle\n"
        f"data = base64.b64decode('{encoded}')\n"
        "model = cloudpickle.loads(data)\n"
        "result = model.flow.compute(b=2)\n"
        "assert result.value == 3, f'Expected 3, got {result.value}'\n"
    )
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, f"Cross-process local cloudpickle failed:\n{result.stderr}"


def test_local_generated_model_postponed_annotations_cross_process_cloudpickle():
    """Local generated models should restore from analyzed config, not worker-side type-hint resolution."""
    namespace: dict[str, Any] = {}
    exec(
        """
from __future__ import annotations
from ccflow import Flow, FromContext

def make_model():
    @Flow.model
    def add(a: int, b: FromContext[int]) -> int:
        return a + b
    return add(a=1)
""",
        namespace,
    )

    encoded = base64.b64encode(cloudpickle.dumps(namespace["make_model"](), protocol=5)).decode()
    script = (
        "import base64\n"
        "import cloudpickle\n"
        f"data = base64.b64decode('{encoded}')\n"
        "model = cloudpickle.loads(data)\n"
        "result = model.flow.compute(b=2)\n"
        "assert result.value == 3, f'Expected 3, got {result.value}'\n"
    )
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, f"Cross-process postponed-annotation cloudpickle failed:\n{result.stderr}"


def test_local_generated_model_complex_annotations_same_process_cloudpickle():
    """Local restore should reuse the analyzed contract for nested typing shapes."""

    def make_model():
        @Flow.model
        def summarize(
            values: Annotated[list[int], Field(min_length=1)],
            mode: Literal["sum", "first"],
            offsets: tuple[int, ...],
            maybe_offset: int | None,
            scale: FromContext[int],
        ) -> GenericResult[int]:
            total = values[0] if mode == "first" else sum(values)
            return GenericResult(value=(total + sum(offsets) + (maybe_offset or 0)) * scale)

        return summarize(values=[1, 2], mode="sum", offsets=(3,), maybe_offset=None)

    restored = cloudpickle.loads(cloudpickle.dumps(make_model(), protocol=5))

    assert restored.flow.compute(scale=2) == GenericResult[int](value=12)


def test_model_base_fields_visible_in_bound_inputs():
    """model_base fields that are explicitly set should appear in bound_inputs."""

    class CustomFlowBase(CallableModel):
        multiplier: int = 1

        @Flow.call
        def __call__(self, context: SimpleContext) -> GenericResult[int]:
            return GenericResult(value=context.value)

    @Flow.model(model_base=CustomFlowBase)
    def add(a: int, b: FromContext[int]) -> int:
        return a + b

    model = add(a=10, multiplier=3)
    assert model.flow.inspect().bound_inputs == {"a": 10, "multiplier": 3}

    # Default-only model_base field is NOT in bound_inputs
    model_default = add(a=10)
    assert model_default.flow.inspect().bound_inputs == {"a": 10}


def _annotation_contains(annotation: object, expected: object) -> bool:
    if annotation is expected:
        return True
    return any(_annotation_contains(arg, expected) for arg in get_args(annotation))


def _schema_contains(schema: object, predicate) -> bool:
    if isinstance(schema, dict):
        if predicate(schema):
            return True
        return any(_schema_contains(value, predicate) for value in schema.values())
    if isinstance(schema, list):
        return any(_schema_contains(value, predicate) for value in schema)
    return False


def test_generated_model_fields_preserve_construction_schema():
    @Flow.model
    def str_source(tag: FromContext[str]) -> str:
        return tag

    @Flow.model
    def consumer(x: int, lazy_value: Lazy[int], y: FromContext[int]) -> int:
        return x + lazy_value() + y

    generated_cls = getattr(consumer, "_generated_model")
    fields = generated_cls.model_fields
    properties = generated_cls.model_json_schema()["properties"]

    assert _annotation_contains(fields["x"].annotation, int)
    assert _annotation_contains(fields["x"].annotation, CallableModel)
    assert _schema_contains(properties["x"], lambda node: node.get("type") == "integer")
    assert _schema_contains(properties["x"], lambda node: node.get("$ref", "").endswith("/CallableModel"))

    assert _annotation_contains(fields["lazy_value"].annotation, CallableModel)
    assert _schema_contains(properties["lazy_value"], lambda node: node.get("$ref", "").endswith("/CallableModel"))

    assert _annotation_contains(fields["y"].annotation, int)
    assert _schema_contains(properties["y"], lambda node: node.get("type") == "integer")

    model = consumer(x=str_source(), lazy_value=str_source())
    assert model.flow.compute(tag="3", y="4").value == 10

    with pytest.raises(TypeError, match="Lazy"):
        consumer(x=1, lazy_value=1)

    with pytest.raises(TypeError, match="Regular parameter"):
        model.flow.compute(tag="not_a_number", y=1)


def test_model_base_fields_rejected_by_compute():
    """compute() should reject kwargs matching model_base field names."""

    class CustomFlowBase(CallableModel):
        multiplier: int = 1

        @Flow.call
        def __call__(self, context: SimpleContext) -> GenericResult[int]:
            return GenericResult(value=context.value)

    @Flow.model(model_base=CustomFlowBase)
    def add(a: int, b: FromContext[int]) -> int:
        return a + b

    model = add(a=10, multiplier=3)
    with pytest.raises(TypeError, match="does not accept model configuration override\\(s\\): multiplier"):
        model.flow.compute(b=5, multiplier=99)


def test_flow_model_public_exports_exclude_context_spec_models():
    assert "StaticValueSpec" not in flow_model_module.__all__
    assert "ContextTransform" not in flow_model_module.__all__
    assert "flow_context_transform" not in flow_model_module.__all__
    assert "DependencySpec" not in flow_model_module.__all__
    assert "InputCheck" not in flow_model_module.__all__
    assert not hasattr(ccflow, "StaticValueSpec")
    assert not hasattr(ccflow, "ContextTransform")
    assert not hasattr(ccflow, "flow_context_transform")
    assert not hasattr(ccflow, "DependencySpec")
    assert not hasattr(ccflow, "InputCheck")
    assert not hasattr(flow_model_module, "flow_context_transform")
    assert not hasattr(flow_model_module, "InputCheck")
