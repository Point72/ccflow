"""Unit tests for ccflow.ui.registry module."""

from unittest import mock

import panel as pn

from ccflow import BaseModel, ModelRegistry
from ccflow.ui.registry import ModelRegistryViewer, RegistryBrowser

from .utils import find_components_by_type


class SimpleModel(BaseModel):
    """A simple test model."""

    name: str
    value: int = 0


class AnotherModel(BaseModel):
    """Another test model."""

    data: str = ""


class TestRegistryBrowser:
    """Tests for RegistryBrowser class."""

    def test_init_returns_viewable(self):
        """Test RegistryBrowser returns a Panel viewable."""
        registry = ModelRegistry(name="test")
        browser = RegistryBrowser(registry)
        panel = browser.__panel__()
        assert isinstance(panel, pn.viewable.Viewable)

    def test_panel_contains_autocomplete_search(self):
        """Test that the panel contains an autocomplete search widget."""
        registry = ModelRegistry(name="test")
        browser = RegistryBrowser(registry)
        panel = browser.__panel__()
        autocomplete = find_components_by_type(panel, pn.widgets.AutocompleteInput)
        assert len(autocomplete) > 0

    def test_init_with_empty_registry(self):
        """Test RegistryBrowser initialization with empty registry."""
        registry = ModelRegistry(name="test")
        browser = RegistryBrowser(registry)

        assert browser.selected_model is None
        # Search options should be empty
        panel = browser.__panel__()
        autocomplete = find_components_by_type(panel, pn.widgets.AutocompleteInput)
        assert autocomplete[0].options == []

    def test_init_with_models(self):
        """Test RegistryBrowser initialization with models in registry."""
        registry = ModelRegistry(name="test")
        model1 = SimpleModel(name="test1", value=1)
        model2 = AnotherModel(data="test2")
        registry.add("model1", model1)
        registry.add("model2", model2)

        browser = RegistryBrowser(registry)

        # Search options should contain the model names
        panel = browser.__panel__()
        autocomplete = find_components_by_type(panel, pn.widgets.AutocompleteInput)
        assert "model1" in autocomplete[0].options
        assert "model2" in autocomplete[0].options

    def test_build_tree_simple(self):
        """Test _build_tree with flat registry."""
        registry = ModelRegistry(name="test")
        model = SimpleModel(name="test", value=1)
        registry.add("my_model", model)

        browser = RegistryBrowser(registry)
        tree_items = browser._tree_items

        assert len(tree_items) == 1
        assert tree_items[0]["label"] == "my_model"
        assert tree_items[0]["model"] == model
        assert tree_items[0]["_index_path"] == (0,)

    def test_build_tree_nested(self):
        """Test _build_tree with nested registries."""
        root = ModelRegistry(name="root")
        sub = ModelRegistry(name="sub")
        model = SimpleModel(name="test", value=1)

        sub.add("nested_model", model)
        root.add("subregistry", sub)

        browser = RegistryBrowser(root)
        tree_items = browser._tree_items

        assert len(tree_items) == 1
        assert tree_items[0]["label"] == "subregistry"
        assert "items" in tree_items[0]
        assert len(tree_items[0]["items"]) == 1
        assert tree_items[0]["items"][0]["label"] == "nested_model"
        assert tree_items[0]["items"][0]["model"] == model

    def test_build_node_index(self):
        """Test _build_node_index creates correct path mappings."""
        root = ModelRegistry(name="root")
        sub = ModelRegistry(name="sub")
        model1 = SimpleModel(name="test1", value=1)
        model2 = SimpleModel(name="test2", value=2)

        root.add("top_model", model1)
        sub.add("nested_model", model2)
        root.add("subregistry", sub)

        browser = RegistryBrowser(root)

        assert "top_model" in browser._node_index
        assert "subregistry/nested_model" in browser._node_index
        assert browser._node_index["top_model"]["model"] == model1
        assert browser._node_index["subregistry/nested_model"]["model"] == model2

    def test_expanded_from_index_path(self):
        """Test _expanded_from_index_path generates correct expanded paths."""
        # Single level - no expansion needed
        result = RegistryBrowser._expanded_from_index_path((0,))
        assert result == []

        # Two levels
        result = RegistryBrowser._expanded_from_index_path((0, 1))
        assert result == [(0,)]

        # Three levels
        result = RegistryBrowser._expanded_from_index_path((0, 1, 2))
        assert result == [(0,), (0, 1)]

        # Four levels
        result = RegistryBrowser._expanded_from_index_path((1, 2, 3, 4))
        assert result == [(1,), (1, 2), (1, 2, 3)]

    def test_on_search_select_empty(self):
        """Test _on_search_select with empty value."""
        registry = ModelRegistry(name="test")
        model = SimpleModel(name="test", value=1)
        registry.add("my_model", model)
        browser = RegistryBrowser(registry)

        event = mock.Mock()
        event.new = ""

        browser._on_search_select(event)
        assert browser.selected_model is None

    def test_on_search_select_invalid_path(self):
        """Test _on_search_select with non-existent path."""
        registry = ModelRegistry(name="test")
        model = SimpleModel(name="test", value=1)
        registry.add("my_model", model)
        browser = RegistryBrowser(registry)

        event = mock.Mock()
        event.new = "nonexistent"

        browser._on_search_select(event)
        assert browser.selected_model is None

    def test_on_search_select_valid_path(self):
        """Test _on_search_select with valid path."""
        registry = ModelRegistry(name="test")
        model = SimpleModel(name="test", value=1)
        registry.add("my_model", model)
        browser = RegistryBrowser(registry)

        event = mock.Mock()
        event.new = "my_model"

        browser._on_search_select(event)

        # Tree value should be set
        assert len(browser._tree.value) == 1
        assert browser._tree.value[0]["label"] == "my_model"

    def test_on_tree_select_empty(self):
        """Test _on_tree_select with empty selection."""
        registry = ModelRegistry(name="test")
        model = SimpleModel(name="test", value=1)
        registry.add("my_model", model)
        browser = RegistryBrowser(registry)

        event = mock.Mock()
        event.new = []

        browser._on_tree_select(event)
        assert browser.selected_model is None

    def test_on_tree_select_model(self):
        """Test _on_tree_select with model selection."""
        registry = ModelRegistry(name="test")
        model = SimpleModel(name="test", value=1)
        registry.add("my_model", model)
        browser = RegistryBrowser(registry)

        event = mock.Mock()
        event.new = [{"label": "my_model", "model": model}]

        browser._on_tree_select(event)
        assert browser.selected_model == model

    def test_on_tree_select_registry(self):
        """Test _on_tree_select with registry selection (no model key)."""
        registry = ModelRegistry(name="test")
        sub = ModelRegistry(name="sub")
        registry.add("subregistry", sub)
        browser = RegistryBrowser(registry)

        event = mock.Mock()
        event.new = [{"label": "subregistry", "items": []}]

        browser._on_tree_select(event)
        assert browser.selected_model is None

    def test_search_options_sorted(self):
        """Test that search widget options are sorted."""
        root = ModelRegistry(name="root")
        sub = ModelRegistry(name="sub")
        model1 = SimpleModel(name="test1", value=1)
        model2 = SimpleModel(name="test2", value=2)

        root.add("zebra", model1)
        sub.add("alpha", model2)
        root.add("subregistry", sub)

        browser = RegistryBrowser(root)
        panel = browser.__panel__()
        autocomplete = find_components_by_type(panel, pn.widgets.AutocompleteInput)

        assert autocomplete[0].options == sorted(autocomplete[0].options)


class TestModelRegistryViewer:
    """Tests for ModelRegistryViewer class."""

    def test_init_returns_viewable(self):
        """Test ModelRegistryViewer returns a Panel viewable."""
        registry = ModelRegistry(name="test")
        viewer = ModelRegistryViewer(registry)
        panel = viewer.__panel__()
        assert isinstance(panel, pn.viewable.Viewable)

    def test_panel_is_row_layout(self):
        """Test that the panel is a Row layout (browser + viewer side by side)."""
        registry = ModelRegistry(name="test")
        viewer = ModelRegistryViewer(registry)
        panel = viewer.__panel__()
        assert isinstance(panel, pn.Row)

    def test_init_with_custom_dimensions(self):
        """Test initialization with custom width/height."""
        registry = ModelRegistry(name="test")
        viewer = ModelRegistryViewer(
            registry,
            browser_width=500,
            browser_height=800,
            viewer_width=600,
        )

        assert viewer.browser_width == 500
        assert viewer.browser_height == 800
        assert viewer.viewer_width == 600

    def test_browser_viewer_wiring(self):
        """Test that browser selection updates viewer."""
        registry = ModelRegistry(name="test")
        model = SimpleModel(name="test", value=42)
        registry.add("my_model", model)

        viewer = ModelRegistryViewer(registry)

        # Simulate browser selection
        viewer._browser.selected_model = model

        # Viewer should be updated
        assert viewer._viewer.model == model

    def test_default_browser_dimensions(self):
        """Test default browser dimensions."""
        registry = ModelRegistry(name="test")
        viewer = ModelRegistryViewer(registry)

        assert viewer.browser_width == 400
        assert viewer.browser_height == 700
        assert viewer.viewer_width is None

    def test_make_browser_column(self):
        """Test _make_browser_column creates proper column."""
        registry = ModelRegistry(name="test")
        viewer = ModelRegistryViewer(registry)

        column = viewer._make_browser_column()
        assert isinstance(column, pn.Column)
        assert column.width == viewer.browser_width
        assert column.height == viewer.browser_height
        assert column.scroll is True

    def test_make_viewer_column_with_width(self):
        """Test _make_viewer_column with specified width."""
        registry = ModelRegistry(name="test")
        viewer = ModelRegistryViewer(registry, viewer_width=600)

        column = viewer._make_viewer_column()
        assert isinstance(column, pn.Column)
        assert column.width == 600

    def test_make_viewer_column_without_width(self):
        """Test _make_viewer_column without specified width (stretch)."""
        registry = ModelRegistry(name="test")
        viewer = ModelRegistryViewer(registry)

        column = viewer._make_viewer_column()
        assert isinstance(column, pn.Column)
        assert column.sizing_mode == "stretch_width"

    def test_model_param_default_none(self):
        """Test that model param starts as None."""
        registry = ModelRegistry(name="test")
        viewer = ModelRegistryViewer(registry)
        assert viewer.model is None

    def test_model_param_updated_on_selection(self):
        """Test that model param is updated when a model is selected."""
        registry = ModelRegistry(name="test")
        model = SimpleModel(name="test", value=42)
        registry.add("my_model", model)

        viewer = ModelRegistryViewer(registry)
        viewer._browser.selected_model = model

        assert viewer.model == model

    def test_model_param_cleared_on_deselection(self):
        """Test that model param is cleared when selection is cleared."""
        registry = ModelRegistry(name="test")
        model = SimpleModel(name="test", value=42)
        registry.add("my_model", model)

        viewer = ModelRegistryViewer(registry)
        viewer._browser.selected_model = model
        assert viewer.model == model

        viewer._browser.selected_model = None
        assert viewer.model is None


class TestIntegration:
    """Integration tests for UI components."""

    def test_full_workflow(self):
        """Test complete workflow from registry to viewer."""
        # Create a registry with nested structure
        root = ModelRegistry(name="root")
        models_reg = ModelRegistry(name="models")
        configs_reg = ModelRegistry(name="configs")

        model1 = SimpleModel(name="model1", value=1)
        model2 = SimpleModel(name="model2", value=2)
        config1 = AnotherModel(data="config1")

        models_reg.add("first", model1)
        models_reg.add("second", model2)
        configs_reg.add("main", config1)

        root.add("models", models_reg)
        root.add("configs", configs_reg)

        # Create viewer
        viewer = ModelRegistryViewer(root)

        # Verify browser has all paths indexed
        browser = viewer._browser
        assert "models/first" in browser._node_index
        assert "models/second" in browser._node_index
        assert "configs/main" in browser._node_index

        # Simulate selecting a model
        browser.selected_model = model1

        # Verify viewer is updated
        assert viewer._viewer.model == model1

    def test_nested_registry_expansion(self):
        """Test that nested paths generate correct expansion."""
        root = ModelRegistry(name="root")
        level1 = ModelRegistry(name="level1")
        level2 = ModelRegistry(name="level2")
        model = SimpleModel(name="deep", value=99)

        level2.add("deep_model", model)
        level1.add("level2", level2)
        root.add("level1", level1)

        browser = RegistryBrowser(root)

        # Find the deep model node
        node = browser._node_index["level1/level2/deep_model"]

        # Check expansion path
        expanded = browser._expanded_from_index_path(node["_index_path"])
        assert len(expanded) == 2  # level1 and level1/level2

    def test_switch_model_selection(self):
        """Test switching between different models updates viewer correctly."""
        root = ModelRegistry(name="root")
        model1 = SimpleModel(name="first", value=1)
        model2 = SimpleModel(name="second", value=2)

        root.add("model1", model1)
        root.add("model2", model2)

        viewer = ModelRegistryViewer(root)

        # Select first model
        viewer._browser.selected_model = model1
        assert viewer._viewer.model == model1
        assert viewer._viewer._config_viewer.model == model1

        # Switch to second model
        viewer._browser.selected_model = model2
        assert viewer._viewer.model == model2
        assert viewer._viewer._config_viewer.model == model2

        # Verify JSON editor shows second model's data
        json_editors = find_components_by_type(viewer._viewer.__panel__(), pn.widgets.JSONEditor)
        assert json_editors[0].value["name"] == "second"
        assert json_editors[0].value["value"] == 2

    def test_switch_between_different_model_types(self):
        """Test switching between models of different types in the viewer."""
        root = ModelRegistry(name="root")
        simple_model = SimpleModel(name="simple", value=42)
        another_model = AnotherModel(data="test data")

        root.add("simple", simple_model)
        root.add("another", another_model)

        viewer = ModelRegistryViewer(root)

        # Select SimpleModel
        viewer._browser.selected_model = simple_model
        assert viewer._viewer._type_viewer.model_type == SimpleModel

        json_editors = find_components_by_type(viewer._viewer.__panel__(), pn.widgets.JSONEditor)
        assert "name" in json_editors[0].value
        assert "value" in json_editors[0].value

        # Switch to AnotherModel
        viewer._browser.selected_model = another_model
        assert viewer._viewer._type_viewer.model_type == AnotherModel

        # JSON should now show AnotherModel's fields
        assert "data" in json_editors[0].value
        assert "value" not in json_editors[0].value
        assert json_editors[0].value["data"] == "test data"

    def test_deselect_model_clears_viewer(self):
        """Test that deselecting a model (selecting registry) clears the viewer."""
        root = ModelRegistry(name="root")
        sub = ModelRegistry(name="sub")
        model = SimpleModel(name="test", value=1)

        sub.add("model", model)
        root.add("subregistry", sub)

        viewer = ModelRegistryViewer(root)

        # Select model first
        viewer._browser.selected_model = model
        assert viewer._viewer.model == model
        assert viewer._viewer._json_container.visible is True

        # Deselect (simulate selecting registry node which has no model)
        viewer._browser.selected_model = None
        assert viewer._viewer.model is None
        assert viewer._viewer._json_container.visible is False

    def test_tree_selection_updates_viewer(self):
        """Test that tree selection properly updates the viewer through the wiring."""
        root = ModelRegistry(name="root")
        model1 = SimpleModel(name="first", value=1)
        model2 = SimpleModel(name="second", value=2)

        root.add("model1", model1)
        root.add("model2", model2)

        viewer = ModelRegistryViewer(root)
        browser = viewer._browser

        # Simulate tree selection of first model
        event = mock.Mock()
        event.new = [{"label": "model1", "model": model1}]
        browser._on_tree_select(event)

        assert browser.selected_model == model1
        assert viewer._viewer.model == model1

        # Simulate tree selection of second model
        event.new = [{"label": "model2", "model": model2}]
        browser._on_tree_select(event)

        assert browser.selected_model == model2
        assert viewer._viewer.model == model2

    def test_search_then_switch_models(self):
        """Test using search to select models and then switching."""
        root = ModelRegistry(name="root")
        model1 = SimpleModel(name="alpha", value=1)
        model2 = SimpleModel(name="beta", value=2)

        root.add("alpha_model", model1)
        root.add("beta_model", model2)

        viewer = ModelRegistryViewer(root)
        browser = viewer._browser

        # Search and select first model
        event = mock.Mock()
        event.new = "alpha_model"
        browser._on_search_select(event)

        # Tree should be updated
        assert len(browser._tree.value) == 1
        assert browser._tree.value[0]["label"] == "alpha_model"

        # Simulate the tree select callback that would happen
        tree_event = mock.Mock()
        tree_event.new = browser._tree.value
        browser._on_tree_select(tree_event)

        assert viewer._viewer.model == model1

        # Now search and select second model
        event.new = "beta_model"
        browser._on_search_select(event)

        tree_event.new = browser._tree.value
        browser._on_tree_select(tree_event)

        assert viewer._viewer.model == model2

        # Verify viewer shows second model
        json_editors = find_components_by_type(viewer._viewer.__panel__(), pn.widgets.JSONEditor)
        assert json_editors[0].value["name"] == "beta"

    def test_rapid_model_switching(self):
        """Test rapidly switching between multiple models."""
        root = ModelRegistry(name="root")
        models = [SimpleModel(name=f"model_{i}", value=i) for i in range(5)]

        for i, model in enumerate(models):
            root.add(f"model_{i}", model)

        viewer = ModelRegistryViewer(root)

        # Rapidly switch through all models
        for i, model in enumerate(models):
            viewer._browser.selected_model = model
            assert viewer._viewer.model == model
            assert viewer._viewer._config_viewer.model == model

            json_editors = find_components_by_type(viewer._viewer.__panel__(), pn.widgets.JSONEditor)
            assert json_editors[0].value["name"] == f"model_{i}"
            assert json_editors[0].value["value"] == i

    def test_switch_models_in_nested_registries(self):
        """Test switching between models in different nested registries."""
        root = ModelRegistry(name="root")
        reg_a = ModelRegistry(name="reg_a")
        reg_b = ModelRegistry(name="reg_b")

        model_a = SimpleModel(name="in_a", value=100)
        model_b = AnotherModel(data="in_b")

        reg_a.add("model", model_a)
        reg_b.add("model", model_b)
        root.add("registry_a", reg_a)
        root.add("registry_b", reg_b)

        viewer = ModelRegistryViewer(root)

        # Verify both paths are indexed
        assert "registry_a/model" in viewer._browser._node_index
        assert "registry_b/model" in viewer._browser._node_index

        # Select model from registry_a
        viewer._browser.selected_model = model_a
        assert viewer._viewer._type_viewer.model_type == SimpleModel

        # Switch to model from registry_b (different type)
        viewer._browser.selected_model = model_b
        assert viewer._viewer._type_viewer.model_type == AnotherModel

        json_editors = find_components_by_type(viewer._viewer.__panel__(), pn.widgets.JSONEditor)
        assert "data" in json_editors[0].value
        assert "value" not in json_editors[0].value
