"""Focused tests for the FromContext-based Flow.model API."""

import base64
import graphlib
import inspect
import pickle
import subprocess
import sys
from datetime import date, timedelta
from types import ModuleType
from typing import Annotated, Any, Literal, Optional

import pytest
import ray
from pydantic import Field, ValidationError, model_validator
from ray.cloudpickle import dumps as rcpdumps, loads as rcploads

import ccflow
import ccflow._flow_model_binding as flow_binding_module
import ccflow.flow_model as flow_model_module
from ccflow import (
    CallableModel,
    ContextBase,
    DateRangeContext,
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
from ccflow.evaluators import GraphEvaluator, MemoryCacheEvaluator, combine_evaluators, get_dependency_graph


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

    raw = date_range_loader_previous_day(source="warehouse").flow.compute(start_date=date(2024, 1, 2), end_date=date(2024, 1, 3)).value
    assert raw == {"source": "warehouse", "start_date": "2024-01-01", "end_date": "2024-01-03"}
    assert date_range_processor(raw_data=raw).flow.compute().value.startswith("raw:warehouse")
    assert date_range_processor(raw_data=raw, normalize=True).flow.compute().value.startswith("normalized:warehouse")

    contextual = contextual_loader(source="warehouse").flow.compute(start_date=date(2024, 1, 1), end_date=date(2024, 1, 2)).value
    assert contextual_processor(prefix="p", data=contextual).flow.compute(start_date=date(2024, 1, 1), end_date=date(2024, 1, 2)).value == (
        "p:warehouse:2024-01-01 to 2024-01-02"
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


def test_context_transform_internal_error_and_repr_paths():
    assert flow_model_module._context_transform_repr(static_patch()) == "static_patch()"
    assert flow_model_module._context_transform_repr(increment_b(amount=2)) == "increment_b(amount=2)"
    assert flow_model_module._context_transform_repr(123) == "123"
    assert flow_model_module._context_transform_identifier(increment_b(amount=1)).endswith(".increment_b")

    with pytest.raises(ValidationError, match="exactly one"):
        flow_model_module.ContextTransform()

    with pytest.raises(ValidationError, match="exactly one"):
        flow_model_module.ContextTransform(path="ccflow.tests.test_flow_model.increment_b", serialized_config="also-set")

    with pytest.raises(TypeError, match="does not resolve to a Flow.context_transform binding"):
        flow_model_module._load_context_transform_config("ccflow.tests.test_flow_model.lazy_context_transform_for_rejection")

    invalid_payload = base64.b64encode(pickle.dumps({"not": "a config"}, protocol=5)).decode("ascii")
    with pytest.raises(TypeError, match="payload does not contain"):
        flow_model_module._load_serialized_context_transform_config(invalid_payload)

    invalid_binding = flow_model_module.ContextTransform.model_construct(path=None, serialized_config=None, bound_args={})
    with pytest.raises(TypeError, match="neither path nor serialized_config"):
        flow_model_module._load_context_transform_config_from_binding(invalid_binding)

    with pytest.raises(ImportError, match="does not have a _generated_model"):
        flow_model_module._restore_generated_flow_model("ccflow.tests.test_flow_model.lazy_context_transform_for_rejection", {})


def test_flow_model_low_level_value_helpers_cover_edge_paths():
    assert flow_model_module._bound_field_names(object()) == set()
    assert flow_model_module._concrete_context_type(int | None) is None
    no_name_annotation = int | str
    assert flow_model_module._expected_type_repr(no_name_annotation) == repr(no_name_annotation)
    assert flow_model_module._coerce_value("x", "still-raw", object(), "test") == "still-raw"
    assert flow_model_module._unwrap_model_result(7) == 7
    assert flow_model_module._type_accepts_str(Annotated[str, "meta"]) is True
    assert flow_model_module._type_accepts_str(int | str) is True
    assert flow_binding_module._is_result_annotation(GenericResult[int] | None) is True
    assert flow_model_module._registry_candidate_allowed(object(), data_source(base_value=1)) is True
    assert flow_model_module._registry_candidate_allowed(int, GenericResult(value=1)) is False
    assert flow_model_module._is_mapping_annotation(inspect.Signature.empty) is False
    assert flow_model_module._is_mapping_annotation(123) is False
    generated_type = type(basic_loader(source="s", multiplier=2))
    assert flow_model_module._resolve_generated_model_bases(generated_type) == (generated_type,)
    assert callable(Flow.context_transform())

    metadata: list[object] = []
    annotation = Annotated[int, metadata]
    assert flow_model_module._type_adapter(annotation) is flow_model_module._type_adapter(annotation)

    with pytest.raises(TypeError, match="only supports Python functions"):
        flow_model_module._ensure_top_level_named_function(123, decorator_name="@Flow.model")
    with pytest.raises(TypeError, match="only supports named Python functions"):
        flow_model_module._ensure_top_level_named_function(lambda: None, decorator_name="@Flow.model")


def test_lazy_thunks_and_regular_resolution_edge_paths():
    calls = {"dependency": 0, "inner": 0}

    @Flow.model
    def source(value: FromContext[int]) -> int:
        calls["dependency"] += 1
        return value + 10

    thunk = flow_model_module._make_lazy_thunk(source(), FlowContext(value=2))
    assert thunk() == 12
    assert thunk() == 12
    assert calls["dependency"] == 1

    def inner():
        calls["inner"] += 1
        return "13"

    coercing = flow_model_module._make_coercing_lazy_thunk(inner, "value", int)
    assert coercing() == 13
    assert coercing() == 13
    assert calls["inner"] == 1

    @Flow.model
    def missing_regular(x: int) -> int:
        return x

    missing_config = type(missing_regular()).__flow_model_config__
    with pytest.raises(TypeError, match="still unbound"):
        flow_model_module._resolve_regular_param_value(missing_regular(), missing_config.param("x"), FlowContext())

    @Flow.model
    def lazy_consumer(x: Lazy[int]) -> int:
        return x()

    lazy_model = getattr(lazy_consumer, "_generated_model").model_construct(x=1)
    lazy_config = type(lazy_model).__flow_model_config__
    with pytest.raises(TypeError, match="must be bound to a CallableModel"):
        flow_model_module._resolve_regular_param_value(lazy_model, lazy_config.param("x"), FlowContext())


def test_context_transform_validation_and_static_resolution_edge_paths():
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
    assert flow_model_module._evaluate_context_transform_from_values(default_seed(), {}) == 10
    with pytest.raises(TypeError, match="Missing contextual input"):
        flow_model_module._evaluate_context_transform_from_values(seed_plus_one(), {})

    dynamic_spec = flow_model_module._BoundContextSpec(
        patches=[flow_model_module.PatchContextSpec(binding=dynamic_patch())],
        field_overrides={},
    )
    assert flow_model_module._statically_resolved_context_values(add(), dynamic_spec) is None
    assert flow_model_module._statically_resolved_context_field_names(add(), dynamic_spec) == set()

    identity_values, missing_transforms = flow_model_module._apply_context_spec_values_for_identity(add(), dynamic_spec, FlowContext(b=2))
    assert identity_values == {"b": 2}
    assert missing_transforms == ((flow_model_module._context_transform_identifier(dynamic_patch()), ("seed",)),)

    missing_regular = default_amount()
    config = flow_model_module._load_context_transform_config_from_binding(default_amount())
    assert flow_model_module._bound_context_transform_regular_kwargs(config, missing_regular) == {"amount": 5}

    with pytest.raises(TypeError, match="unexpected keyword"):
        increment_b(amount=1, extra=2)
    with pytest.raises(TypeError, match="Do not pass contextual"):
        increment_b(b=1, amount=2)
    with pytest.raises(TypeError, match="missing required regular"):
        increment_b()

    with pytest.raises(TypeError, match="must return a mapping"):
        flow_model_module._validate_patch_result(add(), 1)
    with pytest.raises(TypeError, match="string field names"):
        flow_model_module._validate_patch_result(add(), {1: 2})

    class OpaqueModel:
        context_type = object

    assert flow_model_module._validate_patch_result(OpaqueModel(), {"x": 1}) == {"x": 1}
    flow_model_module._validate_with_context_field_names(OpaqueModel(), ["anything"])
    assert (
        flow_model_module._static_field_override_value(OpaqueModel(), "anything", flow_model_module.FieldContextSpec(binding=default_amount())) == 5
    )

    with pytest.raises(TypeError, match="raw callables"):
        add().flow.with_context(lambda: {"a": 1})
    with pytest.raises(TypeError, match="Positional with_context"):
        add().flow.with_context(123)


def test_additional_flow_model_source_edge_paths(monkeypatch):
    @Flow.context_transform
    def default_seed(seed: FromContext[int] = 9) -> int:
        return seed + 1

    @Flow.context_transform
    def dynamic_patch(seed: FromContext[int]) -> dict[str, object]:
        return {"a": seed}

    @Flow.model
    def add(a: FromContext[int], b: FromContext[int]) -> int:
        return a + b

    @Flow.model
    def regular_required(x: int) -> int:
        return x

    @Flow.model
    def lazy_consumer(x: Lazy[int]) -> int:
        return x()

    restored = flow_model_module._restore_generated_flow_model(
        "ccflow.tests.test_flow_model.basic_loader",
        basic_loader(source="s", multiplier=2).__getstate__(),
    )
    assert restored.flow.compute(value=3).value == 6

    class FailingPath:
        def __init__(self, path):
            self.path = path

        @property
        def object(self):
            raise ImportError(self.path)

    original_path = flow_model_module.PyObjectPath
    monkeypatch.setattr(flow_model_module, "PyObjectPath", FailingPath)
    config = type(basic_loader(source="s", multiplier=2)).__flow_model_config__
    assert flow_model_module._generated_model_factory_path_for_pickle(config, type(basic_loader(source="s", multiplier=2))) is None
    monkeypatch.setattr(flow_model_module, "PyObjectPath", original_path)

    assert flow_model_module._registry_candidate_allowed(int, 1) is True
    opaque_model = type("OpaqueModel", (), {"context_type": object})()
    assert flow_model_module._coerce_model_context_value(opaque_model, "anything", "raw", "test") == "raw"
    assert flow_model_module._generated_model_identity_payload(regular_required(), FlowContext()) is None

    context_spec = flow_model_module._BoundContextSpec(
        patches=[flow_model_module.PatchContextSpec(binding=dynamic_patch())],
        field_overrides={"b": flow_model_module.FieldContextSpec(binding=default_seed())},
    )
    values, missing = flow_model_module._apply_context_spec_values_for_identity(add(), context_spec, FlowContext(seed=1))
    assert values == {"a": 1, "b": 2, "seed": 1}
    assert missing == ()
    assert flow_model_module._statically_resolved_context_values(add(), context_spec) is None

    bound = add().flow.with_context(dynamic_patch())
    assert bound.flow.context_inputs == {"a": int, "b": int, "seed": int}
    assert bound.flow.unbound_inputs == {"a": int, "b": int, "seed": int}

    with pytest.raises(TypeError, match="missing required regular"):
        flow_model_module._bound_context_transform_regular_kwargs(
            flow_model_module._load_context_transform_config_from_binding(increment_b(amount=1)),
            increment_b(amount=1).model_copy(update={"bound_args": {}}),
        )
    with pytest.raises(TypeError, match="Missing regular parameter"):
        regular_required().__deps__(FlowContext())
    assert lazy_consumer(x=data_source(base_value=1)).__deps__(FlowContext(value=1)) == []
    assert getattr(basic_loader, "_generated_model")._resolve_registry_refs("raw") == "raw"
    assert flow_model_module._GeneratedFlowModelBase._resolve_registry_refs({}) == {}

    def transform_with_bad_hints(value: FromContext[int]) -> int:
        return value

    def raise_attribute_error(*args, **kwargs):
        raise AttributeError("bad hints")

    monkeypatch.setattr(flow_model_module, "get_type_hints", raise_attribute_error)
    assert Flow.context_transform(transform_with_bad_hints)().serialized_config is not None


def test_plain_and_bound_optional_compute_paths_and_identity_helpers():
    class AnyContextModel:
        context_type = object

    class FlowContextModel:
        context_type = FlowContext

    class OptionalContextModel(CallableModel):
        @Flow.call
        def __call__(self, context: Optional[SimpleContext] = None) -> GenericResult[int]:
            return GenericResult(value=0 if context is None else context.value)

    assert flow_model_module._model_context_contract(AnyContextModel()).input_types is None
    assert flow_model_module._model_context_contract(FlowContextModel()).input_types is None
    assert flow_model_module._identity_context_values_for_model(AnyContextModel(), FlowContext(extra=1)) == {"extra": 1}
    assert OptionalContextModel().flow.compute(None).value == 0
    assert OptionalContextModel().flow.compute().value == 0
    assert OptionalContextModel().flow.unbound_inputs == {}

    bound = OptionalContextModel().flow.with_context()
    assert bound.flow.compute(FlowContext(value=3)).value == 3
    with pytest.raises(TypeError, match="either one context object"):
        bound.flow.compute(FlowContext(value=3), value=4)
    assert bound.flow._compute_target is bound


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


def test_compute_rejects_kwargs_for_already_bound_regular_params():
    @Flow.model
    def add(a: int, b: FromContext[int]) -> int:
        return a + b

    model = add(a=1)
    with pytest.raises(TypeError, match="does not accept regular parameter override\\(s\\): a"):
        model.flow.compute(a=999, b=2)


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

    assert model.flow.context_inputs == {"mode": Literal["a"]}
    assert model.flow.unbound_inputs == {"mode": Literal["a"]}
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
    assert model.flow.bound_inputs["context"] == DateRangeContext(start_date=date(2024, 1, 1), end_date=date(2024, 1, 2))
    assert model.flow.context_inputs == {}
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
    assert model.flow.bound_inputs == {"context": SimpleContext(value=10)}
    assert model.flow.context_inputs == {"y": int}
    assert model.flow.compute(y=5).value == 15


def test_context_args_keyword_is_removed():
    with pytest.raises(TypeError, match="context_args=... has been removed"):

        @Flow.model(context_args=["x"])
        def bad(x: int) -> int:
            return x


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


def test_lazy_runtime_helper_is_removed():
    @Flow.model
    def source(value: FromContext[int]) -> GenericResult[int]:
        return GenericResult(value=value)

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


def test_generated_models_plain_pickle_roundtrip():
    @Flow.model
    def multiply(a: int, b: FromContext[int]) -> int:
        return a * b

    model = multiply(a=6)
    restored = pickle.loads(pickle.dumps(model, protocol=5))
    assert restored.flow.compute(b=7).value == 42


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


def test_generated_models_cloudpickle_preserves_unset_validation_sentinel():
    @Flow.model
    def multiply(a: int, b: FromContext[int]) -> int:
        return a * b

    model = multiply(a=6)
    restored = rcploads(rcpdumps(model, protocol=5))
    param = type(restored).__flow_model_config__.contextual_params[0]

    assert param.context_validation_annotation is flow_model_module._UNSET
    assert param.validation_annotation is int


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

    restored = rcploads(rcpdumps(graph))
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


def test_context_transform_serializes_import_path_and_bound_args():
    binding = increment_b(amount=3)
    assert isinstance(binding, flow_model_module.ContextTransform)
    assert binding.kind == "context_transform"
    assert binding.path is not None
    assert binding.serialized_config is None
    assert binding.bound_args == {"amount": 3}
    assert str(binding.path).endswith(".increment_b")


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

    assert binding.path is None
    assert binding.serialized_config is not None

    @Flow.model
    def add(a: int, b: FromContext[int]) -> int:
        return a + b

    bound = add(a=10).flow.with_context(b=binding)
    restored = pickle.loads(pickle.dumps(bound, protocol=5))
    assert restored.flow.compute(value=4).value == 15


def test_context_transform_supports_nested_functions_with_serialized_payload():
    @Flow.model
    def add(a: int, b: FromContext[int]) -> int:
        return a + b

    @Flow.context_transform
    def nested_transform(b: FromContext[int], amount: int) -> int:
        return b + amount

    binding = nested_transform(amount=3)
    assert binding.path is None
    assert binding.serialized_config is not None

    bound = add(a=1).flow.with_context(b=binding)
    restored = rcploads(rcpdumps(bound))
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
    assert binding.path is None
    assert binding.serialized_config is not None

    bound = add(a=1).flow.with_context(b=binding)
    restored = rcploads(rcpdumps(bound))
    assert restored.flow.compute(value=4).value == 6


def test_context_transform_nested_function_survives_ray_task():
    @Flow.model
    def add(a: int, b: FromContext[int]) -> int:
        return a + b

    @Flow.context_transform
    def nested_transform(b: FromContext[int], amount: int) -> int:
        return b + amount

    bound = add(a=1).flow.with_context(b=nested_transform(amount=3))

    @ray.remote
    def run_model(model):
        return model.flow.compute(b=4).value

    with ray.init(num_cpus=1):
        assert ray.get(run_model.remote(bound)) == 8


def test_with_context_rejects_raw_callables():
    @Flow.model
    def add(a: int, b: FromContext[int]) -> int:
        return a + b

    with pytest.raises(TypeError, match="no longer accepts raw callables"):
        add(a=1).flow.with_context(b=lambda ctx: ctx.b + 1)


def test_with_context_rejects_wrong_transform_position():
    @Flow.model
    def load(start_date: FromContext[int], end_date: FromContext[int]) -> int:
        return start_date + end_date

    with pytest.raises(TypeError, match="Field context transforms must be passed by keyword"):
        load().flow.with_context(increment_b(amount=1))

    with pytest.raises(TypeError, match="Patch transforms must be passed positionally"):
        load().flow.with_context(start_date=shift_integer_window(amount=10))


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

    dumped = bound.model_dump(mode="json")
    assert dumped["context_spec"]["patches"][0]["binding"]["bound_args"] == {"amount": 10}
    assert dumped["context_spec"]["field_overrides"]["start_date"]["kind"] == "static_value"


def test_transforms_evaluate_against_original_runtime_context():
    @Flow.model
    def load(start_date: FromContext[int], end_date: FromContext[int]) -> int:
        return start_date * 1000 + end_date

    bound = load().flow.with_context(
        shift_integer_window(amount=10),
        start_date=bump_start_date(amount=100),
    )

    result = bound(FlowContext(start_date=1, end_date=2))
    assert result.value == 101_012


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


def test_any_annotation_preserves_literal_strings():
    """A parameter typed Any should keep literal strings; registry should not steal them."""
    registry = ModelRegistry.root().clear()
    dep_model = basic_loader(source="warehouse", multiplier=1)
    registry.add("my_key", dep_model)

    @Flow.model
    def uses_any(x: Any, y: FromContext[int]) -> int:
        return y if isinstance(x, str) else 999

    model = uses_any(x="my_key")
    result = model.flow.compute(y=3)
    assert result.value == 3, "literal string should not be replaced by registry entry for Any-typed param"


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


def test_unexpected_type_validation_errors_are_not_rewritten():
    from pydantic_core import core_schema

    class BrokenValidation:
        @classmethod
        def __get_pydantic_core_schema__(cls, source, handler):
            del source, handler

            def validate(value):
                del value
                raise RuntimeError("boom")

            return core_schema.no_info_plain_validator_function(validate)

    @Flow.model
    def bad(x: BrokenValidation, y: FromContext[int]) -> int:
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


def test_generated_model_flow_api_introspection_and_execution():
    @Flow.model
    def add(a: int, b: FromContext[int]) -> int:
        return a + b

    model = add(a=10)
    assert model.flow.context_inputs == {"b": int}
    assert model.flow.bound_inputs == {"a": 10}
    assert model.flow.unbound_inputs == {"b": int}
    assert model.flow.compute(b=5).value == 15


def test_type_adapter_caches_are_bounded_and_clearable(monkeypatch):
    monkeypatch.setattr(flow_model_module, "_TYPE_ADAPTER_CACHE_MAXSIZE", 2)
    flow_model_module.clear_flow_model_caches()

    try:
        for annotation in (int, str, float):
            flow_model_module._type_adapter(annotation)

        assert list(flow_model_module._HASHABLE_TYPE_ADAPTER_CACHE) == [str, float]

        unhashable_annotations = (
            Annotated[int, []],
            Annotated[str, []],
            Annotated[float, []],
        )
        for annotation in unhashable_annotations:
            flow_model_module._type_adapter(annotation)

        assert len(flow_model_module._UNHASHABLE_TYPE_ADAPTER_CACHE) == 2
        assert [entry[0] for entry in flow_model_module._UNHASHABLE_TYPE_ADAPTER_CACHE.values()] == list(unhashable_annotations[-2:])

        flow_model_module.clear_flow_model_caches()
        assert not flow_model_module._HASHABLE_TYPE_ADAPTER_CACHE
        assert not flow_model_module._UNHASHABLE_TYPE_ADAPTER_CACHE
    finally:
        flow_model_module.clear_flow_model_caches()


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

    assert model.flow.context_inputs == {"value": int}
    assert model.flow.unbound_inputs == {"value": int}
    assert model.flow.bound_inputs == {}
    assert model.flow.compute({"value": 3}).value == 3

    with pytest.raises(TypeError, match="either one context object or contextual keyword arguments"):
        model.flow.compute(SimpleContext(value=1), value=2)


def test_unhashable_annotations_still_validate():
    annotation = Annotated[int, []]

    @Flow.model
    def add(x: annotation, y: FromContext[annotation]) -> int:
        return x + y

    assert add(x="2").flow.compute(y="3").value == 5


def test_compute_accepts_context_object_for_from_context_models():
    model = basic_loader(source="warehouse", multiplier=3)

    assert model.flow.context_inputs == {"value": int}
    assert model.flow.unbound_inputs == {"value": int}
    assert model.flow.compute({"value": 4}).value == 12
    assert model.flow.compute(SimpleContext(value=5)).value == 15

    with pytest.raises(TypeError, match="either one context object or contextual keyword arguments"):
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

    @Flow.model
    def add(x: int, y: FromContext[int]) -> int:
        return x + y

    assert add(x=1).flow.compute(y=2).value == 3


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


def test_context_type_rejects_nullable_field_for_non_nullable_from_context():
    class OptionalValueContext(ContextBase):
        value: int | None

    with pytest.raises(TypeError, match="annotates"):

        @Flow.model(context_type=OptionalValueContext)
        def add_one(value: FromContext[int]) -> int:
            return value + 1


@pytest.mark.parametrize(
    ("func_annotation", "context_annotation", "expected"),
    [
        (int, int, True),
        (int, str, False),
        (int | None, int, True),
        (int, int | None, False),
        (int | None, int | None, True),
        (list[int], list[str], False),
        (list[int], list[int], True),
        (int | str, int, True),
        (int | str, int | None, False),
        (int | None, type(None), True),
        (int | None, int | str, False),
        (int, int | str, False),
        (object, int | str, True),
        (Literal["a"], Literal["a"], True),
        (Literal["a"], Literal["b"], False),
        (str, Literal["a"], True),
        (Literal["a", "b"], Literal["a"], True),
        (Literal["a"], str, False),
        (list[int], Literal["a"], False),
        (Annotated[int, "meta"], int, True),
        (dict[str, list[int]], dict[str, list[int]], True),
        (dict[str, list[int]], dict[str, list[str]], False),
        (list, list[int], False),
        (tuple[int], tuple[int, str], False),
        (Any, int, True),
        (Any, str, True),
        (int, Any, True),
        (str, Any, True),
        (Any, Any, True),
    ],
)
def test_context_type_annotations_compatible_cases(func_annotation, context_annotation, expected):
    assert flow_binding_module._context_type_annotations_compatible(func_annotation, context_annotation) is expected


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


def test_unset_flow_input_pickle_roundtrip_preserves_singleton():
    restored = pickle.loads(pickle.dumps(flow_model_module._UNSET_FLOW_INPUT, protocol=5))
    assert restored is flow_model_module._UNSET_FLOW_INPUT


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


def test_bound_flow_unbound_inputs_subtracts_static_context():
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

    assert PlainSource().flow.with_context(a=1).flow.unbound_inputs == {"b": int}
    assert add().flow.with_context(a=1).flow.unbound_inputs == {"b": int}
    assert add().flow.with_context(a=static_bad()).flow.unbound_inputs == {"b": int}
    assert add().flow.with_context(static_patch()).flow.unbound_inputs == {"b": int}
    assert add().flow.with_context(a=1, b=2).flow.unbound_inputs == {}


def test_bound_flow_unbound_inputs_reflects_dynamic_field_transform_inputs():
    @Flow.model
    def add(a: FromContext[int], b: FromContext[int]) -> int:
        return a + b

    bound = add().flow.with_context(a=seed_plus_one())

    assert bound.flow.compute(seed=1, b=10).value == 12
    assert bound.flow.context_inputs == {"b": int, "seed": int}
    assert bound.flow.unbound_inputs == {"b": int, "seed": int}


def test_bound_flow_bound_inputs_include_static_context_bindings():
    @Flow.model
    def add(a: int, b: FromContext[int]) -> int:
        return a + b

    bound = add(a=1).flow.with_context(b=2)

    assert bound.flow.context_inputs == {}
    assert bound.flow.unbound_inputs == {}
    assert bound.flow.bound_inputs == {"a": 1, "b": 2}


def test_bound_flow_bound_inputs_drops_static_patch_after_dynamic_override():
    @Flow.model
    def add(a: FromContext[int], b: FromContext[int]) -> int:
        return a + b

    bound = add().flow.with_context(static_patch()).flow.with_context(a=seed_plus_one())

    assert bound.flow.compute(seed=3, b=10).value == 14
    assert bound.flow.bound_inputs == {}
    assert bound.flow.context_inputs == {"b": int, "seed": int}
    assert bound.flow.unbound_inputs == {"b": int, "seed": int}


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


def test_generated_model_effective_cache_key_includes_opaque_evaluator_behavior():
    class OpaqueA(EvaluatorBase):
        tag: str = "same"

        def __call__(self, context: ModelEvaluationContext):
            return context()

    class OpaqueB(EvaluatorBase):
        tag: str = "same"

        def __call__(self, context: ModelEvaluationContext):
            result = context()
            return result

    @Flow.model
    def add(a: int, b: FromContext[int]) -> int:
        return a + b

    model = add(a=10)
    inner = model.__call__.get_evaluation_context(model, FlowContext(b=1, unused="same"), _options={"cacheable": True})
    cache = MemoryCacheEvaluator()

    key1 = cache.key(ModelEvaluationContext(model=OpaqueA(), context=inner))
    key2 = cache.key(ModelEvaluationContext(model=OpaqueB(), context=inner))

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


def test_generated_model_cache_wraps_effective_key_with_nontransparent_evaluator():
    calls = {"count": 0}

    class OpaqueEvaluator(EvaluatorBase):
        def __call__(self, context: ModelEvaluationContext):
            return context()

    @Flow.model
    def add(a: int, b: FromContext[int]) -> int:
        calls["count"] += 1
        return a + b

    cache = MemoryCacheEvaluator()
    evaluator = combine_evaluators(OpaqueEvaluator(), cache)
    model = add(a=10)

    with FlowOptionsOverride(options={"evaluator": cache, "cacheable": True}):
        assert model.flow.compute(b=1, unused="plain").value == 11

    with FlowOptionsOverride(options={"evaluator": evaluator, "cacheable": True}):
        assert model.flow.compute(b=1, unused="one").value == 11
        assert model.flow.compute(b=1, unused="two").value == 11

    assert calls["count"] == 2
    assert len(cache.cache) == 2


def test_generated_model_cache_uses_effective_key_when_result_validation_disabled():
    calls = {"count": 0}

    @Flow.model
    def add(a: int, b: FromContext[int]) -> int:
        calls["count"] += 1
        return a + b

    cache = MemoryCacheEvaluator()
    model = add(a=10)

    with FlowOptionsOverride(options={"evaluator": cache, "cacheable": True, "validate_result": False}):
        assert model.flow.compute(b=1, unused="one").value == 11
        assert model.flow.compute(b=1, unused="two").value == 11

    assert calls["count"] == 1
    assert len(cache.cache) == 1


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


def test_unused_lazy_bound_plain_dependency_applies_static_context_identity():
    calls = {"source": 0, "choose": 0}

    class RequiredContext(ContextBase):
        a: int
        b: int

    class PlainSource(CallableModel):
        @Flow.call
        def __call__(self, context: RequiredContext) -> GenericResult[int]:
            calls["source"] += 1
            return GenericResult(value=context.a + context.b)

    @Flow.model
    def choose(x: int, lazy_value: Lazy[int], use_lazy: FromContext[bool]) -> int:
        calls["choose"] += 1
        return lazy_value() if use_lazy else x

    cache = MemoryCacheEvaluator()
    model = choose(x=7, lazy_value=PlainSource().flow.with_context(a=1))

    with FlowOptionsOverride(options={"evaluator": cache, "cacheable": True}):
        assert model.flow.compute(use_lazy=False, unused="one").value == 7
        assert model.flow.compute(use_lazy=False, unused="two").value == 7
        with pytest.raises(ValidationError):
            model.flow.compute(use_lazy=True)

    assert calls == {"source": 0, "choose": 2}
    assert len(cache.cache) == 1


def test_unused_lazy_bound_plain_dependency_dynamic_transform_can_leave_missing_context():
    calls = {"source": 0, "choose": 0}

    class RequiredContext(ContextBase):
        a: int
        b: int

    class PlainSource(CallableModel):
        @Flow.call
        def __call__(self, context: RequiredContext) -> GenericResult[int]:
            calls["source"] += 1
            return GenericResult(value=context.a + context.b)

    @Flow.model
    def choose(x: int, lazy_value: Lazy[int], use_lazy: FromContext[bool]) -> int:
        calls["choose"] += 1
        return lazy_value() if use_lazy else x

    cache = MemoryCacheEvaluator()
    model = choose(x=7, lazy_value=PlainSource().flow.with_context(b=seed_plus_one()))

    with FlowOptionsOverride(options={"evaluator": cache, "cacheable": True}):
        assert model.flow.compute(use_lazy=False, seed=1, unused="one").value == 7
        assert model.flow.compute(use_lazy=False, seed=1, unused="two").value == 7

    assert calls == {"source": 0, "choose": 1}
    assert len(cache.cache) == 1


def test_unused_lazy_bound_plain_dependency_fully_resolved_identity_ignores_ambient_context():
    calls = {"source": 0, "choose": 0}

    class RequiredContext(ContextBase):
        a: int
        b: int

    class PlainSource(CallableModel):
        @Flow.call
        def __call__(self, context: RequiredContext) -> GenericResult[int]:
            calls["source"] += 1
            return GenericResult(value=context.a + context.b)

    @Flow.model
    def choose(x: int, lazy_value: Lazy[int], use_lazy: FromContext[bool]) -> int:
        calls["choose"] += 1
        return lazy_value() if use_lazy else x

    cache = MemoryCacheEvaluator()
    model = choose(x=7, lazy_value=PlainSource().flow.with_context(a=1, b=2))

    with FlowOptionsOverride(options={"evaluator": cache, "cacheable": True}):
        assert model.flow.compute(use_lazy=False, unused="one").value == 7
        assert model.flow.compute(use_lazy=False, unused="two").value == 7

    assert calls == {"source": 0, "choose": 1}
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
    assert len(cache.cache) == 6


def test_unused_lazy_bound_dependency_records_missing_transform_context():
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
    model = choose(x=7, lazy_value=source().flow.with_context(a=seed_plus_one()))

    with FlowOptionsOverride(options={"evaluator": cache, "cacheable": True}):
        assert model.flow.compute(use_lazy=False, unused="one").value == 7
        assert model.flow.compute(use_lazy=False, unused="two").value == 7
        assert model.flow.compute(use_lazy=False, a=100, b=1).value == 7
        assert model.flow.compute(use_lazy=False, a=200, b=1).value == 7

    assert calls == {"source": 0, "choose": 1}
    assert len(cache.cache) == 1


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


def test_generated_model_diamond_cache_reuses_shared_source_and_ignores_unused_fields():
    calls = {"source": 0, "left": 0, "right": 0, "root": 0}

    @Flow.model
    def source(value: FromContext[int]) -> int:
        calls["source"] += 1
        return value + 10

    @Flow.model
    def left(x: int) -> int:
        calls["left"] += 1
        return x * 2

    @Flow.model
    def right(x: int) -> int:
        calls["right"] += 1
        return x * 5

    @Flow.model
    def root(left_value: int, right_value: int, bonus: FromContext[int]) -> int:
        calls["root"] += 1
        return left_value + right_value + bonus

    shared = source()
    model = root(left_value=left(x=shared), right_value=right(x=shared))
    cache = MemoryCacheEvaluator()

    with FlowOptionsOverride(options={"evaluator": cache, "cacheable": True}):
        assert model(FlowContext(value=3, bonus=7, unused="one")).value == 98
        assert model(FlowContext(value=3, bonus=7, unused="two")).value == 98

    assert calls == {"source": 1, "left": 1, "right": 1, "root": 1}
    assert len(cache.cache) == 4


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


def test_generated_dependency_graph_identity_ignores_unused_flow_context_fields():
    @Flow.model
    def source(value: FromContext[int]) -> int:
        return value * 10

    @Flow.model
    def root(x: int, bonus: FromContext[int]) -> int:
        return x + bonus

    model = root(x=source())
    graph1 = get_dependency_graph(model.__call__.get_evaluation_context(model, FlowContext(value=3, bonus=7, unused="one")))
    graph2 = get_dependency_graph(model.__call__.get_evaluation_context(model, FlowContext(value=3, bonus=7, unused="two")))

    assert graph1.root_id == graph2.root_id
    assert set(graph1.graph.keys()) == set(graph2.graph.keys())
    assert set(graph1.ids.keys()) == set(graph2.ids.keys())


def test_bound_generated_model_dependency_graph_has_no_self_loop():
    @Flow.model
    def add(a: int, b: FromContext[int]) -> int:
        return a + b

    bound = add(a=1).flow.with_context(b=2)
    graph = get_dependency_graph(bound.__call__.get_evaluation_context(bound, FlowContext(b=99)))

    assert graph.root_id not in graph.graph[graph.root_id]


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


def test_plain_callable_model_cache_remains_structural_for_flow_context():
    calls = {"count": 0}

    class Counter(CallableModel):
        @Flow.call
        def __call__(self, context: FlowContext) -> GenericResult[int]:
            calls["count"] += 1
            return GenericResult(value=context.value)

    cache = MemoryCacheEvaluator()
    model = Counter()

    with FlowOptionsOverride(options={"evaluator": cache, "cacheable": True}):
        assert model(FlowContext(value=10, unused="one")).value == 10
        assert model(FlowContext(value=10, unused="two")).value == 10

    assert calls["count"] == 2
    assert len(cache.cache) == 2


def test_generated_models_cross_process_pickle():
    """Module-level @Flow.model instances are deserializable in a separate process."""
    model = basic_loader(source="warehouse", multiplier=3)
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


def test_generated_models_cross_process_cloudpickle():
    """Module-level @Flow.model instances are deserializable via cloudpickle in a separate process."""
    from ray.cloudpickle import dumps as rcpdumps

    model = basic_loader(source="warehouse", multiplier=3)
    data = rcpdumps(model, protocol=5)
    encoded = base64.b64encode(data).decode()
    script = (
        "import base64\n"
        "from ray.cloudpickle import loads as rcploads\n"
        f"data = base64.b64decode('{encoded}')\n"
        "model = rcploads(data)\n"
        "from ccflow import FlowContext\n"
        "result = model.flow.compute(value=4)\n"
        "assert result.value == 12, f'Expected 12, got {result.value}'\n"
    )
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, f"Cross-process cloudpickle failed:\n{result.stderr}"


def test_local_generated_models_cross_process_cloudpickle():
    """Local @Flow.model instances carry their generated class across processes."""
    from ray.cloudpickle import dumps as rcpdumps

    def make_model():
        @Flow.model
        def add(a: int, b: FromContext[int]) -> int:
            return a + b

        return add(a=1)

    encoded = base64.b64encode(rcpdumps(make_model(), protocol=5)).decode()
    script = (
        "import base64\n"
        "from ray.cloudpickle import loads as rcploads\n"
        f"data = base64.b64decode('{encoded}')\n"
        "model = rcploads(data)\n"
        "result = model.flow.compute(b=2)\n"
        "assert result.value == 3, f'Expected 3, got {result.value}'\n"
    )
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, f"Cross-process local cloudpickle failed:\n{result.stderr}"


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
    assert model.flow.bound_inputs == {"a": 10, "multiplier": 3}

    # Default-only model_base field is NOT in bound_inputs
    model_default = add(a=10)
    assert model_default.flow.bound_inputs == {"a": 10}


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
    assert "FieldContextSpec" not in flow_model_module.__all__
    assert "PatchContextSpec" not in flow_model_module.__all__
    assert not hasattr(ccflow, "StaticValueSpec")
    assert not hasattr(ccflow, "FieldContextSpec")
    assert not hasattr(ccflow, "PatchContextSpec")
