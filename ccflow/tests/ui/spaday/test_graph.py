"""Unit tests for ccflow.ui.spaday.graph module."""

from spaday.validate import validate

from ccflow import BaseModel, LazyRegistry, ModelRegistry
from ccflow.ui.spaday.graph import dependency_edges, model_dependency_graph, model_dependency_view
from ccflow.ui.spaday.registry import registry_leaves

from .utils import event_action, nodes_with_tag, prop_value


def _view(path, adjacency):
    return model_dependency_view(path, adjacency, selected_field="selected")


class Leaf(BaseModel):
    """A model with no registry dependencies."""

    name: str = "leaf"


class Holder(BaseModel):
    """A model that contains another registered model."""

    child: Leaf


class Outer(BaseModel):
    """A model that contains a model which itself has a dependency."""

    inner: Holder


def _registry_with_dependency():
    root = ModelRegistry.root()
    root.clear()
    leaf = Leaf(name="a")
    sub = ModelRegistry(name="sub")
    sub.add("alpha", leaf)
    root.add("sub", sub)
    root.add("holder", Holder(child=leaf))
    return root


class TestDependencyEdges:
    def test_empty_registry(self):
        assert dependency_edges([]) == {}

    def test_edge_from_dependent_to_dependency(self):
        adjacency = dependency_edges(registry_leaves(_registry_with_dependency()))
        assert adjacency == {"sub/alpha": [], "holder": ["sub/alpha"]}

    def test_no_edges_without_dependencies(self):
        root = ModelRegistry.root()
        root.clear()
        root.add("one", Leaf(name="one"))
        root.add("two", Leaf(name="two"))
        assert dependency_edges(registry_leaves(root)) == {"one": [], "two": []}

    def test_dependencies_outside_registry_are_dropped(self):
        root = ModelRegistry.root()
        root.clear()
        leaf = Leaf(name="hidden")
        root.add("hidden", leaf)
        holder_registry = ModelRegistry(name="holder_only")
        holder_registry.add("holder", Holder(child=leaf))
        # Browsing a registry that does not contain the dependency must not invent a dangling node.
        assert dependency_edges(registry_leaves(holder_registry)) == {"holder": []}

    def test_pending_models_report_no_dependencies(self):
        lazy = LazyRegistry(
            name="lazy",
            group={"model": {"_target_": "ccflow.tests.ui.spaday.test_graph.Leaf", "name": "pending"}},
        )
        assert dependency_edges(registry_leaves(lazy)) == {"group/model": []}
        assert not lazy["group"].is_loaded("model")


class TestModelDependencyGraph:
    def test_unknown_path(self):
        assert model_dependency_graph("nope", {}) == {"nodes": [], "edges": []}

    def test_model_without_dependencies_is_a_single_node(self):
        graph = model_dependency_graph("sub/alpha", {"sub/alpha": [], "holder": ["sub/alpha"]})
        assert [node["id"] for node in graph["nodes"]] == ["sub/alpha"]
        assert graph["edges"] == []

    def test_graph_is_local_to_the_model(self):
        adjacency = {"holder": ["sub/alpha"], "sub/alpha": [], "unrelated": []}
        graph = model_dependency_graph("holder", adjacency)
        # "unrelated" is in the registry but not reachable from "holder", so it is not drawn.
        assert [node["id"] for node in graph["nodes"]] == ["holder", "sub/alpha"]
        # The edge points from the dependency into the model that uses it.
        assert graph["edges"] == [{"source": "sub/alpha", "target": "holder"}]

    def test_nodes_carry_no_styling_classes(self):
        graph = model_dependency_graph("holder", {"holder": ["sub/alpha"], "sub/alpha": []})
        # Emphasis is a bindable prop on the component, so the graph stays pure structure.
        assert all("class" not in node for node in graph["nodes"])

    def test_transitive_dependencies_are_included(self):
        adjacency = {"outer": ["holder"], "holder": ["leaf"], "leaf": []}
        graph = model_dependency_graph("outer", adjacency)
        assert [node["id"] for node in graph["nodes"]] == ["outer", "holder", "leaf"]
        # leaf -> holder -> outer: the chain reads towards the model being inspected.
        assert graph["edges"] == [
            {"source": "holder", "target": "outer"},
            {"source": "leaf", "target": "holder"},
        ]

    def test_cycles_terminate(self):
        graph = model_dependency_graph("a", {"a": ["b"], "b": ["a"]})
        assert [node["id"] for node in graph["nodes"]] == ["a", "b"]
        assert len(graph["edges"]) == 2

    def test_labels_show_the_full_registry_path(self):
        graph = model_dependency_graph("holder", {"holder": ["sub/alpha"], "sub/alpha": []})
        assert [node["label"] for node in graph["nodes"]] == ["holder", "sub/alpha"]


class TestModelDependencyView:
    def test_none_without_dependencies(self):
        adjacency = dependency_edges(registry_leaves(_registry_with_dependency()))
        assert _view("sub/alpha", adjacency) is None

    def test_renders_dagre_component(self):
        adjacency = dependency_edges(registry_leaves(_registry_with_dependency()))
        node = _view("holder", adjacency).to_node()
        dagre = nodes_with_tag(node, "spaday-dagre")
        assert len(dagre) == 1
        assert prop_value(dagre[0], "graph")["edges"] == [{"source": "sub/alpha", "target": "holder"}]

    def test_node_events_open_the_menu_instead_of_navigating(self):
        adjacency = dependency_edges(registry_leaves(_registry_with_dependency()))
        node = _view("holder", adjacency).to_node()
        dagre = nodes_with_tag(node, "spaday-dagre")[0]
        for event in ("dagre-node-click", "dagre-node-contextmenu"):
            action = event_action(dagre, event)
            # open_popup is a Sequence that captures the node, then positions and opens the popup.
            assert action["kind"] == "seq"
            writes = [step for step in action["actions"] if step["kind"] == "set-field"]
            assert writes[0]["field"] == "menu_path"
            assert writes[0]["value"] == {"expr": "event", "path": "id"}

    def test_menu_is_shared_and_bound_to_the_clicked_node(self):
        adjacency = dependency_edges(registry_leaves(_registry_with_dependency()))
        node = _view("holder", adjacency).to_node()
        assert len(nodes_with_tag(node, "spa-popup")) == 1
        # One body for every node, bound to the store rather than one gated copy per node.
        assert not nodes_with_tag(node, "spa-show")
        code = [n for n in nodes_with_tag(node, "code") if "textContent" in n.get("bindings", {})]
        assert code[0]["bindings"]["textContent"]["field"] == "menu_path"

    def test_open_model_sets_selection_and_reveals_in_tree(self):
        adjacency = dependency_edges(registry_leaves(_registry_with_dependency()))
        node = _view("holder", adjacency).to_node()
        button = next(n for n in nodes_with_tag(node, "wa-button") if "click" in n.get("events", {}))
        writes = {step["field"]: step["value"] for step in button["events"]["click"]["actions"] if step["kind"] == "set-field"}
        # Setting the selection is enough: the tree derives its reveal from it.
        assert writes["selected"] == {"expr": "field", "name": "menu_path"}

    def test_long_labels_are_capped(self):
        adjacency = dependency_edges(registry_leaves(_registry_with_dependency()))
        dagre = nodes_with_tag(_view("holder", adjacency).to_node(), "spaday-dagre")[0]
        assert prop_value(dagre, "maxLabelWidth") == 220

    def test_inspected_model_is_emphasised(self):
        adjacency = dependency_edges(registry_leaves(_registry_with_dependency()))
        dagre = nodes_with_tag(_view("holder", adjacency).to_node(), "spaday-dagre")[0]
        assert prop_value(dagre, "emphasis") == "holder"

    def test_layout_is_left_to_right(self):
        adjacency = dependency_edges(registry_leaves(_registry_with_dependency()))
        dagre = nodes_with_tag(_view("holder", adjacency).to_node(), "spaday-dagre")[0]
        assert prop_value(dagre, "layout") == {"rankdir": "LR"}

    def test_validates(self):
        adjacency = dependency_edges(registry_leaves(_registry_with_dependency()))
        validate(_view("holder", adjacency).to_node())
