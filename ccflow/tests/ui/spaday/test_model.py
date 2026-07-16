"""Unit tests for ccflow.ui.spaday.model module."""

from typing import Type

from pydantic import Field
from spaday.validate import validate

from ccflow import BaseModel, CallableModel, ContextBase, Flow, GenericResult, ModelRegistry
from ccflow.ui.spaday.model import model_config_view, model_type_view, model_view

from .utils import all_text, nodes_with_tag, text_of


class SimpleModel(BaseModel):
    """A documented test model."""

    name: str = Field(description="the display name")
    value: int = 0


class Ctx(ContextBase):
    """A test context."""

    a: int = 1


class MyCallable(CallableModel):
    """A callable test model."""

    x: str = "hi"

    @property
    def context_type(self) -> Type[Ctx]:
        return Ctx

    @Flow.call
    def __call__(self, context: Ctx) -> GenericResult:
        return GenericResult(value=self.x)


class TestModelTypeView:
    def test_none_is_empty(self):
        node = model_type_view(None).to_node()
        assert node["tag"] == "spa-stack"
        assert node.get("slots", {}) == {}

    def test_type_name_in_badge(self):
        node = model_type_view(SimpleModel).to_node()
        badges = nodes_with_tag(node, "wa-badge")
        assert any(text_of(b) == "SimpleModel" for b in badges)

    def test_lists_fields(self):
        text = " ".join(all_text(model_type_view(SimpleModel).to_node()))
        assert "name" in text
        assert "value" in text

    def test_includes_field_description(self):
        text = " ".join(all_text(model_type_view(SimpleModel).to_node()))
        assert "the display name" in text

    def test_includes_docstring(self):
        text = " ".join(all_text(model_type_view(SimpleModel).to_node()))
        assert "A documented test model." in text


class TestModelConfigView:
    def test_includes_path(self):
        model = SimpleModel(name="m")
        text = " ".join(all_text(model_config_view(model, "reg/m").to_node()))
        assert "reg/m" in text

    def test_no_metadata_message_when_empty(self):
        model = SimpleModel(name="m")
        text = " ".join(all_text(model_config_view(model).to_node()))
        assert "No additional metadata." in text

    def test_dependencies_rendered(self):
        registry = ModelRegistry(name="test")
        dep = SimpleModel(name="dep")
        registry.add("dep", dep)
        holder = MyCallable()
        registry.add("holder", holder)
        # A model that depends on another shows its registry dependencies (if any).
        node = model_config_view(holder, "holder").to_node()
        assert node["tag"] == "spa-stack"


class TestModelView:
    def test_is_card(self):
        node = model_view(SimpleModel(name="m"), "m").to_node()
        assert node["tag"] == "wa-card"

    def test_has_core_tabs(self):
        text = all_text(model_view(SimpleModel(name="m"), "m").to_node())
        assert "Summary" in text
        assert "Model Type" in text
        assert "Parameters" in text

    def test_plain_model_has_no_callable_tabs(self):
        text = all_text(model_view(SimpleModel(name="m"), "m").to_node())
        assert "Context Type" not in text
        assert "Result Type" not in text

    def test_callable_model_has_callable_tabs(self):
        text = all_text(model_view(MyCallable(), "m").to_node())
        assert "Context Type" in text
        assert "Result Type" in text

    def test_parameters_include_field_values(self):
        text = " ".join(all_text(model_view(SimpleModel(name="widget", value=7), "m").to_node()))
        assert "widget" in text

    def test_validates(self):
        validate(model_view(MyCallable(), "m").to_node())
