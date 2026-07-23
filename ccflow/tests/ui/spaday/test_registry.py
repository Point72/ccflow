"""Unit tests for ccflow.ui.spaday.registry module."""

from spaday.validate import validate

from ccflow import BaseModel, LazyRegistry, ModelRegistry
from ccflow.ui.spaday.registry import (
    SELECTED_FIELD,
    registry_leaves,
    registry_store,
    registry_tree,
    registry_viewer,
)

from .utils import click_set_field, nodes_with_tag, prop_str, show_when_value


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
        assert registry_store() == {SELECTED_FIELD: ""}


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
    def test_leaf_items_carry_selection_action(self):
        nodes = registry_tree(_registry())
        # Serialize the whole set of tree items and collect leaf selection targets.
        selected = set()
        for item in nodes:
            for node in nodes_with_tag(item.to_node(), "wa-tree-item"):
                value = click_set_field(node)
                if value is not None:
                    selected.add(value)
        assert selected == {"sub/alpha", "zeta"}

    def test_branch_items_have_no_selection_action(self):
        nodes = registry_tree(_registry())
        # The top-level "sub" node is a branch; it must not carry a click action.
        sub_item = next(n for n in nodes if any(t == "sub" for t in _labels(n.to_node())))
        assert click_set_field(sub_item.to_node()) is None


def _labels(node):
    from .utils import text_of

    return [text_of(n) for n in node.get("slots", {}).get("default", [])]


class TestRegistryViewer:
    def test_returns_app(self):
        app = registry_viewer(_registry())
        assert app.to_node()["tag"] == "spa-app"

    def test_validates(self):
        validate(registry_viewer(_registry()).to_node())

    def test_title_in_header(self):
        from .utils import all_text

        node = registry_viewer(_registry(), title="My Registry").to_node()
        assert "My Registry" in all_text(node)

    def test_show_panel_per_leaf(self):
        node = registry_viewer(_registry()).to_node()
        show_targets = {show_when_value(n) for n in nodes_with_tag(node, "spa-show")}
        # A panel per leaf plus the empty-selection placeholder.
        assert "sub/alpha" in show_targets
        assert "zeta" in show_targets
        assert "" in show_targets

    def test_search_options_cover_all_leaves(self):
        node = registry_viewer(_registry()).to_node()
        options = [prop_str(n, "value") for n in nodes_with_tag(node, "wa-option")]
        # First option is the empty placeholder; the rest are sorted leaf paths.
        assert options[0] == ""
        assert options[1:] == sorted(["sub/alpha", "zeta"])

    def test_browser_width_sets_gutter(self):
        node = registry_viewer(_registry(), browser_width=500).to_node()
        gutters = nodes_with_tag(node, "spa-gutter")
        assert prop_str(gutters[0], "width") == "500px"

    def test_empty_registry_renders(self):
        node = registry_viewer(ModelRegistry(name="empty")).to_node()
        # Only the placeholder show panel, no model panels.
        assert [show_when_value(n) for n in nodes_with_tag(node, "spa-show")] == [""]

    def test_lazy_registry_renders_without_materializing_models(self):
        lazy = LazyRegistry(
            name="lazy",
            group={
                "model": {
                    "_target_": "ccflow.tests.ui.spaday.test_registry.SimpleModel",
                    "name": "pending",
                }
            },
        )

        node = registry_viewer(lazy).to_node()

        assert not lazy["group"].is_loaded("model")
        show_targets = {show_when_value(item) for item in nodes_with_tag(node, "spa-show")}
        assert "group/model" in show_targets
