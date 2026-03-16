"""Unit tests for ccflow.ui.model module."""

import panel as pn
from pydantic import Field

from ccflow import BaseModel, CallableModel, ContextBase, Flow, GenericResult, MetaData, ModelRegistry
from ccflow.ui.model import ModelConfigViewer, ModelTypeViewer, ModelViewer

from .utils import find_components_by_type


class SimpleModel(BaseModel):
    """A simple test model with documentation."""

    name: str
    value: int = 0


class NoDocModel(BaseModel):
    field: str


class DescribedModel(BaseModel):
    """Model with field descriptions."""

    name: str = Field(description="The name field")
    count: int = Field(description="The count field")


class ContainerModel(BaseModel):
    """Model that contains another model."""

    inner: SimpleModel


class SampleContext(ContextBase):
    """Sample context for callable models."""

    input_value: str = ""


class SampleResult(GenericResult):
    """Sample result for callable models."""

    output_value: str = ""


class SimpleCallableModel(CallableModel):
    """A simple callable model for testing."""

    multiplier: int = 1

    @Flow.call
    def __call__(self, context: SampleContext) -> SampleResult:
        return SampleResult(output_value=context.input_value * self.multiplier)


class TestModelTypeViewer:
    """Tests for ModelTypeViewer class."""

    def test_init_returns_viewable(self):
        """Test ModelTypeViewer returns a Panel viewable."""
        viewer = ModelTypeViewer()
        panel = viewer.__panel__()
        assert isinstance(panel, pn.viewable.Viewable)

    def test_panel_contains_html_pane(self):
        """Test that the panel contains an HTML pane for displaying content."""
        viewer = ModelTypeViewer()
        panel = viewer.__panel__()
        html_panes = find_components_by_type(panel, pn.pane.HTML)
        assert len(html_panes) > 0

    def test_on_type_change_with_none(self):
        """Test that setting model_type to None clears the display."""
        viewer = ModelTypeViewer()
        viewer.model_type = SimpleModel
        viewer.model_type = None

        panel = viewer.__panel__()
        html_panes = find_components_by_type(panel, pn.pane.HTML)
        # All HTML panes should be empty
        for pane in html_panes:
            assert pane.object == ""

    def test_on_type_change_with_model(self):
        """Test that setting model_type displays type information."""
        viewer = ModelTypeViewer()
        viewer.model_type = SimpleModel

        panel = viewer.__panel__()
        html_panes = find_components_by_type(panel, pn.pane.HTML)
        html_content = "".join(pane.object for pane in html_panes)

        # Check that type name is displayed
        assert "SimpleModel" in html_content
        # Check that documentation is displayed
        assert "A simple test model with documentation" in html_content
        # Check that fields are displayed
        assert "name" in html_content
        assert "value" in html_content
        assert "str" in html_content
        assert "int" in html_content

    def test_on_type_change_without_docstring(self):
        """Test model type without a docstring."""
        viewer = ModelTypeViewer()
        viewer.model_type = NoDocModel

        panel = viewer.__panel__()
        html_panes = find_components_by_type(panel, pn.pane.HTML)
        html_content = "".join(pane.object for pane in html_panes)

        assert "NoDocModel" in html_content
        assert "field" in html_content
        # Should not have documentation section
        assert "Class Documentation:" not in html_content

    def test_on_type_change_with_field_descriptions(self):
        """Test that field descriptions are displayed."""
        viewer = ModelTypeViewer()
        viewer.model_type = DescribedModel

        panel = viewer.__panel__()
        html_panes = find_components_by_type(panel, pn.pane.HTML)
        html_content = "".join(pane.object for pane in html_panes)

        assert "The name field" in html_content
        assert "The count field" in html_content


class TestModelConfigViewer:
    """Tests for ModelConfigViewer class."""

    def test_init_returns_viewable(self):
        """Test ModelConfigViewer returns a Panel viewable."""
        viewer = ModelConfigViewer()
        panel = viewer.__panel__()
        assert isinstance(panel, pn.viewable.Viewable)

    def test_panel_contains_html_pane(self):
        """Test that the panel contains an HTML pane for metadata display."""
        viewer = ModelConfigViewer()
        panel = viewer.__panel__()
        html_panes = find_components_by_type(panel, pn.pane.HTML)
        assert len(html_panes) > 0

    def test_on_model_change_with_none(self):
        """Test that setting model to None clears the metadata display."""
        viewer = ModelConfigViewer()
        model = SimpleModel(name="test")
        viewer.model = model
        viewer.model = None

        panel = viewer.__panel__()
        html_panes = find_components_by_type(panel, pn.pane.HTML)
        for pane in html_panes:
            assert pane.object == ""

    def test_on_model_change_with_model(self):
        """Test that setting model displays metadata."""
        viewer = ModelConfigViewer()
        model = SimpleModel(name="test", value=42)
        viewer.model = model

        panel = viewer.__panel__()
        html_panes = find_components_by_type(panel, pn.pane.HTML)
        # Should not crash, content depends on model metadata
        assert len(html_panes) > 0

    def test_on_model_change_with_description(self):
        """Test model with meta description."""
        viewer = ModelConfigViewer()
        model = SimpleCallableModel(
            multiplier=2,
            meta=MetaData(description="This is a test model description"),
        )
        viewer.model = model

        panel = viewer.__panel__()
        html_panes = find_components_by_type(panel, pn.pane.HTML)
        html_content = "".join(pane.object for pane in html_panes)

        assert "This is a test model description" in html_content

    def test_render_dependencies_empty(self):
        """Test _render_dependencies with no dependencies."""
        viewer = ModelConfigViewer()
        model = SimpleModel(name="test")
        result = viewer._render_dependencies(model)
        assert result == ""

    def test_render_dependencies_with_deps(self):
        """Test _render_dependencies with dependencies."""
        root = ModelRegistry.root()
        registry = ModelRegistry(name="test_dep_registry")
        root.add("test_dep_registry", registry)

        try:
            model = SimpleModel(name="test")
            registry.add("my_model", model)

            container = ContainerModel(inner=model)
            registry.add("container", container)

            viewer = ModelConfigViewer()
            result = viewer._render_dependencies(container)

            assert "Registry Dependencies" in result
            assert "my_model" in result
        finally:
            root.remove("test_dep_registry")


class TestModelViewer:
    """Tests for ModelViewer class."""

    def test_init_returns_viewable(self):
        """Test ModelViewer returns a Panel viewable."""
        viewer = ModelViewer()
        panel = viewer.__panel__()
        assert isinstance(panel, pn.viewable.Viewable)

    def test_panel_contains_json_editor(self):
        """Test that the panel contains a JSON editor widget."""
        viewer = ModelViewer()
        panel = viewer.__panel__()
        json_editors = find_components_by_type(panel, pn.widgets.JSONEditor)
        assert len(json_editors) > 0

    def test_json_editor_initially_empty(self):
        """Test that JSON editor is initially empty."""
        viewer = ModelViewer()
        panel = viewer.__panel__()
        json_editors = find_components_by_type(panel, pn.widgets.JSONEditor)
        assert json_editors[0].value == {}

    def test_on_model_change_with_none(self):
        """Test that setting model to None clears the JSON editor."""
        viewer = ModelViewer()
        viewer.model = SimpleModel(name="test")
        viewer.model = None

        panel = viewer.__panel__()
        json_editors = find_components_by_type(panel, pn.widgets.JSONEditor)
        assert json_editors[0].value == {}

    def test_on_model_change_with_base_model(self):
        """Test that setting a BaseModel populates the JSON editor."""
        viewer = ModelViewer()
        model = SimpleModel(name="test", value=42)
        viewer.model = model

        panel = viewer.__panel__()
        json_editors = find_components_by_type(panel, pn.widgets.JSONEditor)
        json_value = json_editors[0].value

        assert "name" in json_value
        assert json_value["name"] == "test"
        assert json_value["value"] == 42

    def test_on_model_change_with_callable_model(self):
        """Test that setting a CallableModel sets up type viewers correctly."""
        viewer = ModelViewer()
        model = SimpleCallableModel(multiplier=2)
        viewer.model = model

        # Verify the internal type viewers are properly configured
        assert viewer._config_viewer.model == model
        assert viewer._type_viewer.model_type is type(model)
        assert viewer._context_type_viewer.model_type == model.context_type
        assert viewer._result_type_viewer.model_type == model.result_type

    def test_on_model_change_updates_viewers(self):
        """Test that changing model updates the viewers."""
        viewer = ModelViewer()

        # Set a callable model first
        callable_model = SimpleCallableModel(multiplier=2)
        viewer.model = callable_model

        # Then set a base model
        base_model = SimpleModel(name="test")
        viewer.model = base_model

        # Viewers should be updated for the new model
        assert viewer._config_viewer.model == base_model
        assert viewer._type_viewer.model_type is type(base_model)

    def test_json_serialization(self):
        """Test that JSON editor correctly serializes model data."""
        viewer = ModelViewer()
        model = SimpleModel(name="test_name", value=123)
        viewer.model = model

        panel = viewer.__panel__()
        json_editors = find_components_by_type(panel, pn.widgets.JSONEditor)
        json_value = json_editors[0].value

        assert json_value["name"] == "test_name"
        assert json_value["value"] == 123


class TestModelSwitching:
    """Tests for switching between different models to ensure proper state reset."""

    def test_switch_between_base_models_same_type(self):
        """Test switching between two BaseModel instances of the same type."""
        viewer = ModelViewer()

        model1 = SimpleModel(name="first", value=1)
        model2 = SimpleModel(name="second", value=2)

        # Set first model
        viewer.model = model1
        assert viewer._config_viewer.model == model1
        assert viewer._type_viewer.model_type == SimpleModel

        panel = viewer.__panel__()
        json_editors = find_components_by_type(panel, pn.widgets.JSONEditor)
        assert json_editors[0].value["name"] == "first"
        assert json_editors[0].value["value"] == 1

        # Switch to second model
        viewer.model = model2
        assert viewer._config_viewer.model == model2
        assert viewer._type_viewer.model_type == SimpleModel

        # Verify JSON editor updated (not still showing old model)
        assert json_editors[0].value["name"] == "second"
        assert json_editors[0].value["value"] == 2

    def test_switch_between_base_models_different_types(self):
        """Test switching between two BaseModel instances of different types."""
        viewer = ModelViewer()

        model1 = SimpleModel(name="simple", value=42)
        model2 = DescribedModel(name="described", count=100)

        # Set first model
        viewer.model = model1
        assert viewer._type_viewer.model_type == SimpleModel

        panel = viewer.__panel__()
        json_editors = find_components_by_type(panel, pn.widgets.JSONEditor)
        assert json_editors[0].value["name"] == "simple"
        assert "value" in json_editors[0].value

        # Switch to different type
        viewer.model = model2
        assert viewer._config_viewer.model == model2
        assert viewer._type_viewer.model_type == DescribedModel

        # JSON editor should show new model's fields, not old ones
        assert json_editors[0].value["name"] == "described"
        assert json_editors[0].value["count"] == 100
        assert "value" not in json_editors[0].value

    def test_switch_from_callable_to_base_model(self):
        """Test switching from CallableModel to BaseModel removes context/result tabs."""
        viewer = ModelViewer()

        callable_model = SimpleCallableModel(multiplier=5)
        base_model = SimpleModel(name="base", value=10)

        # Set callable model first
        viewer.model = callable_model
        assert viewer._context_type_viewer.model_type == callable_model.context_type
        assert viewer._result_type_viewer.model_type == callable_model.result_type

        # Should have 4 tabs: Summary, Model Type, Context Type, Result Type
        assert len(viewer._tabs) == 4

        # Switch to base model
        viewer.model = base_model
        assert viewer._config_viewer.model == base_model
        assert viewer._type_viewer.model_type == SimpleModel

        # Should have only 2 tabs now: Summary, Model Type
        assert len(viewer._tabs) == 2

        # JSON editor should show base model data
        panel = viewer.__panel__()
        json_editors = find_components_by_type(panel, pn.widgets.JSONEditor)
        assert json_editors[0].value["name"] == "base"
        assert json_editors[0].value["value"] == 10
        assert "multiplier" not in json_editors[0].value

    def test_switch_from_base_to_callable_model(self):
        """Test switching from BaseModel to CallableModel adds context/result tabs."""
        viewer = ModelViewer()

        base_model = SimpleModel(name="base", value=10)
        callable_model = SimpleCallableModel(multiplier=5)

        # Set base model first
        viewer.model = base_model
        assert len(viewer._tabs) == 2

        # Switch to callable model
        viewer.model = callable_model
        assert viewer._config_viewer.model == callable_model
        assert viewer._type_viewer.model_type == SimpleCallableModel
        assert viewer._context_type_viewer.model_type == callable_model.context_type
        assert viewer._result_type_viewer.model_type == callable_model.result_type

        # Should have 4 tabs now
        assert len(viewer._tabs) == 4

        # JSON editor should show callable model data
        panel = viewer.__panel__()
        json_editors = find_components_by_type(panel, pn.widgets.JSONEditor)
        assert "multiplier" in json_editors[0].value
        assert json_editors[0].value["multiplier"] == 5

    def test_switch_to_none_clears_state(self):
        """Test switching from a model to None clears all state."""
        viewer = ModelViewer()

        model = SimpleCallableModel(multiplier=3)

        # Set model
        viewer.model = model
        assert len(viewer._tabs) == 4
        assert viewer._json_container.visible is True

        # Clear model
        viewer.model = None
        assert len(viewer._tabs) == 0
        assert viewer._json_container.visible is False

        panel = viewer.__panel__()
        json_editors = find_components_by_type(panel, pn.widgets.JSONEditor)
        assert json_editors[0].value == {}

    def test_tabs_reset_to_first_on_model_switch(self):
        """Test that tab selection resets to first tab when switching models."""
        viewer = ModelViewer()

        model1 = SimpleCallableModel(multiplier=1)
        model2 = SimpleCallableModel(multiplier=2)

        # Set first model and change active tab
        viewer.model = model1
        viewer._tabs.active = 2  # Select "Context Type" tab

        # Switch to second model
        viewer.model = model2

        # Tab should reset to first (Summary)
        assert viewer._tabs.active == 0


class TestModelTypeViewerSwitching:
    """Tests for ModelTypeViewer switching between types."""

    def test_switch_model_types(self):
        """Test switching between different model types updates display."""
        viewer = ModelTypeViewer()

        # Set first type
        viewer.model_type = SimpleModel
        panel = viewer.__panel__()
        html_panes = find_components_by_type(panel, pn.pane.HTML)
        html_content = "".join(pane.object for pane in html_panes)

        assert "SimpleModel" in html_content
        assert "name" in html_content
        assert "value" in html_content

        # Switch to different type
        viewer.model_type = DescribedModel
        html_content = "".join(pane.object for pane in html_panes)

        # Should show new type, not old
        assert "DescribedModel" in html_content
        assert "count" in html_content
        assert "SimpleModel" not in html_content
        # "name" is in both, but "value" should not be present
        assert "value" not in html_content

    def test_switch_to_none_clears_display(self):
        """Test switching to None clears the display."""
        viewer = ModelTypeViewer()

        viewer.model_type = SimpleModel
        panel = viewer.__panel__()
        html_panes = find_components_by_type(panel, pn.pane.HTML)

        # Verify content is present
        html_content = "".join(pane.object for pane in html_panes)
        assert "SimpleModel" in html_content

        # Clear
        viewer.model_type = None
        html_content = "".join(pane.object for pane in html_panes)
        assert html_content == ""


class TestModelConfigViewerSwitching:
    """Tests for ModelConfigViewer switching between models."""

    def test_switch_models_with_different_metadata(self):
        """Test switching between models with different metadata."""
        viewer = ModelConfigViewer()

        model1 = SimpleCallableModel(
            multiplier=1,
            meta=MetaData(description="First model description"),
        )
        model2 = SimpleCallableModel(
            multiplier=2,
            meta=MetaData(description="Second model description"),
        )

        # Set first model
        viewer.model = model1
        panel = viewer.__panel__()
        html_panes = find_components_by_type(panel, pn.pane.HTML)
        html_content = "".join(pane.object for pane in html_panes)

        assert "First model description" in html_content

        # Switch to second model
        viewer.model = model2
        html_content = "".join(pane.object for pane in html_panes)

        # Should show new description, not old
        assert "Second model description" in html_content
        assert "First model description" not in html_content

    def test_switch_from_model_with_description_to_without(self):
        """Test switching from model with description to one without."""
        viewer = ModelConfigViewer()

        model_with_desc = SimpleCallableModel(
            multiplier=1,
            meta=MetaData(description="Has a description"),
        )
        model_without_desc = SimpleModel(name="no desc")

        # Set model with description
        viewer.model = model_with_desc
        panel = viewer.__panel__()
        html_panes = find_components_by_type(panel, pn.pane.HTML)
        html_content = "".join(pane.object for pane in html_panes)

        assert "Has a description" in html_content

        # Switch to model without description
        viewer.model = model_without_desc
        html_content = "".join(pane.object for pane in html_panes)

        # Old description should not be present
        assert "Has a description" not in html_content
