"""Unit tests for ccflow.ui.spaday.registry module."""

from spaday.validate import validate

from ccflow import BaseModel, LazyRegistry, ModelRegistry
from ccflow.ui.spaday.registry import (
    DARK_FIELD,
    SELECTED_FIELD,
    model_card,
    registry_leaves,
    registry_store,
    registry_tree,
    registry_viewer,
)

from .utils import all_text, event_action, nodes_with_tag, prop_str, prop_value


class SimpleModel(BaseModel):
    """A simple test model."""

    name: str
    value: int = 0


class AnotherModel(BaseModel):
    """Another test model."""

    data: str = ""


class HolderModel(BaseModel):
    """A test model that contains another registered model."""

    child: SimpleModel


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

    def test_selected_paths_derived_so_the_tree_reveals_the_selection(self):
        node = registry_tree(_registry()).to_node()
        # A URL-seeded selection has to expand the tree to it, so the reveal is computed, not seeded.
        computed = node["bindings"]["selected_paths"]["compute"]
        assert computed["expr"] == "cond"
        assert computed["then"] == {"expr": "arr", "of": [{"expr": "field", "name": SELECTED_FIELD}]}


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
        switch = nodes_with_tag(node, "spa-switch")[0]
        # One routed case per leaf plus the no-selection default, keyed by registry path.
        assert switch["bindings"]["on"]["field"] == SELECTED_FIELD
        assert set(switch["slots"]) == {"sub/alpha", "zeta", "default"}

    def test_cards_are_deferred_not_inlined(self):
        node = registry_viewer(_registry()).to_node()
        # Each case is a placeholder that fetches its card, so the page does not carry every model.
        lazies = nodes_with_tag(node, "spa-lazy")
        assert {prop_str(n, "src") for n in lazies} == {"/card?model=sub/alpha", "/card?model=zeta"}
        assert not nodes_with_tag(node, "wa-card")

    def test_dependency_graph_is_model_local(self):
        root = ModelRegistry.root()
        root.clear()
        leaf = SimpleModel(name="leaf")
        root.add("leaf", leaf)
        root.add("holder", HolderModel(child=leaf))
        # The graph lives in the card, and only for the model that has dependencies.
        assert len(nodes_with_tag(model_card(root, "holder").to_node(), "spaday-dagre")) == 1
        assert not nodes_with_tag(model_card(root, "leaf").to_node(), "spaday-dagre")

    def test_card_for_unknown_model(self):
        node = model_card(_registry(), "nope").to_node()
        assert "Unknown model" in all_text(node)

    def test_card_for_pending_model_does_not_materialize(self):
        lazy = LazyRegistry(
            name="lazy",
            group={"model": {"_target_": "ccflow.tests.ui.spaday.test_registry.SimpleModel", "name": "pending"}},
        )
        assert "Materialize" in all_text(model_card(lazy, "group/model").to_node())
        assert not lazy["group"].is_loaded("model")

    def test_browser_width_sets_gutter(self):
        node = registry_viewer(_registry(), browser_width=500).to_node()
        gutters = nodes_with_tag(node, "spa-gutter")
        assert prop_str(gutters[0], "width") == "500px"

    def test_empty_registry_renders(self):
        node = registry_viewer(ModelRegistry(name="empty")).to_node()
        # Only the placeholder case, and no graph to draw.
        assert set(nodes_with_tag(node, "spa-switch")[0]["slots"]) == {"default"}
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
        assert set(nodes_with_tag(node, "spa-switch")[0]["slots"]) == {"group/model", "group/other", "default"}

    def test_pending_models_are_flagged_in_the_tree(self):
        lazy = LazyRegistry(
            name="lazy",
            group={"model": {"_target_": "ccflow.tests.ui.spaday.test_registry.SimpleModel", "name": "pending"}},
        )
        decorations = prop_value(registry_tree(lazy).to_node(), "decorations")
        assert set(decorations) == {"group/model"}
        assert decorations["group/model"]["badge"] == "lazy"
        assert not lazy["group"].is_loaded("model")
