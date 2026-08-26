"""Unit tests for ccflow.ui.spaday.registry module."""

from spaday.validate import validate

from ccflow import BaseModel, LazyRegistry, ModelRegistry
from ccflow.ui.spaday.registry import (
    DARK_FIELD,
    SELECTED_FIELD,
    VIEW_FIELD,
    registry_leaves,
    registry_store,
    registry_tree,
    registry_viewer,
)

from .utils import all_text, event_action, nodes_with_tag, prop_str, prop_value, show_when_value


class SimpleModel(BaseModel):
    """A simple test model."""

    name: str
    value: int = 0


class AnotherModel(BaseModel):
    """Another test model."""

    data: str = ""


def _registry():
    root = ModelRegistry(name="root")
    sub = ModelRegistry(name="sub")
    sub.add("alpha", SimpleModel(name="a", value=1))
    root.add("sub", sub)
    root.add("zeta", AnotherModel(data="z"))
    return root


class TestRegistryStore:
    def test_default_store(self):
        store = registry_store()
        assert store[SELECTED_FIELD] == ""
        assert store[VIEW_FIELD] == "details"
        assert store[DARK_FIELD] is False


class TestRegistryLeaves:
    def test_empty_registry(self):
        assert registry_leaves(ModelRegistry(name="empty")) == []

    def test_flat_registry(self):
        registry = ModelRegistry(name="test")
        model = SimpleModel(name="m", value=1)
        registry.add("my_model", model)
        assert registry_leaves(registry) == [("my_model", model)]

    def test_nested_paths(self):
        leaves = registry_leaves(_registry())
        paths = [path for path, _ in leaves]
        assert paths == ["sub/alpha", "zeta"]

    def test_sort_children_orders_subregistries_first(self):
        root = ModelRegistry(name="root")
        root.add("zzz_leaf", SimpleModel(name="leaf"))
        sub = ModelRegistry(name="sub")
        sub.add("inner", SimpleModel(name="inner"))
        root.add("aaa_sub", sub)
        # Subregistries sort before leaf models regardless of name.
        assert [p for p, _ in registry_leaves(root)] == ["aaa_sub/inner", "zzz_leaf"]

    def test_insertion_order_when_not_sorted(self):
        root = ModelRegistry(name="root")
        root.add("zebra", SimpleModel(name="z"))
        root.add("alpha", SimpleModel(name="a"))
        assert [p for p, _ in registry_leaves(root, sort_children=False)] == ["zebra", "alpha"]


class TestRegistryTree:
    def test_paths_cover_all_leaves(self):
        node = registry_tree(_registry()).to_node()
        assert node["tag"] == "spaday-tree"
        assert prop_value(node, "paths") == ["sub/alpha", "zeta"]

    def test_selection_change_sets_selected_field(self):
        node = registry_tree(_registry()).to_node()
        action = event_action(node, "selection-change")
        assert action["kind"] == "set-field"
        assert action["field"] == SELECTED_FIELD
        # The event detail is {paths: [...]}; the first entry is the newly selected model.
        assert action["value"] == {"expr": "event", "path": "paths.0"}

    def test_empty_registry_has_no_paths(self):
        node = registry_tree(ModelRegistry(name="empty")).to_node()
        assert prop_value(node, "paths") == []


class TestRegistryViewer:
    def test_returns_app(self):
        app = registry_viewer(_registry())
        assert app.to_node()["tag"] == "spa-app"

    def test_validates(self):
        validate(registry_viewer(_registry()).to_node())

    def test_title_in_header(self):
        node = registry_viewer(_registry(), title="My Registry").to_node()
        assert "My Registry" in all_text(node)

    def test_show_panel_per_leaf(self):
        node = registry_viewer(_registry()).to_node()
        show_targets = {show_when_value(n) for n in nodes_with_tag(node, "spa-show")}
        # A panel per leaf, plus the placeholder (whose condition is falsy-selection, not an equality).
        assert "sub/alpha" in show_targets
        assert "zeta" in show_targets
        assert len(nodes_with_tag(node, "spa-show")) == 3

    def test_dependency_graph_tab_present(self):
        node = registry_viewer(_registry()).to_node()
        assert nodes_with_tag(node, "spaday-dagre")

    def test_browser_width_sets_gutter(self):
        node = registry_viewer(_registry(), browser_width=500).to_node()
        gutters = nodes_with_tag(node, "spa-gutter")
        assert prop_str(gutters[0], "width") == "500px"

    def test_empty_registry_renders(self):
        node = registry_viewer(ModelRegistry(name="empty")).to_node()
        # Only the placeholder panel, and no graph to draw.
        assert len(nodes_with_tag(node, "spa-show")) == 1
        assert not nodes_with_tag(node, "spaday-dagre")

    def test_lazy_registry_renders_without_materializing_models(self):
        lazy = LazyRegistry(
            name="lazy",
            group={
                "model": {
                    "_target_": "ccflow.tests.ui.spaday.test_registry.SimpleModel",
                    "name": "pending",
                },
                "other": {
                    "_target_": "ccflow.tests.ui.spaday.test_registry.SimpleModel",
                    "name": "other",
                },
            },
        )

        node = registry_viewer(lazy).to_node()

        assert not lazy["group"].is_loaded("model")
        assert not lazy["group"].is_loaded("other")
        # Placeholder plus one shared pending-model panel, not one detail card per pending leaf.
        assert len(nodes_with_tag(node, "spa-show")) == 2
