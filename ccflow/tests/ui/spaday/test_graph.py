"""Unit tests for ccflow.ui.spaday.graph module."""

from spaday.validate import validate

from ccflow import BaseModel, LazyRegistry, ModelRegistry
from ccflow.ui.spaday.graph import dependency_edges, model_dependency_graph, model_dependency_view
from ccflow.ui.spaday.registry import registry_leaves

from .utils import event_action, nodes_with_tag, prop_value


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
        assert graph["edges"] == [{"source": "holder", "target": "sub/alpha"}]

    def test_focus_node_is_marked(self):
        graph = model_dependency_graph("holder", {"holder": ["sub/alpha"], "sub/alpha": []})
        assert graph["nodes"][0]["class"] == "focus"
        assert "class" not in graph["nodes"][1]

    def test_transitive_dependencies_are_included(self):
        adjacency = {"outer": ["holder"], "holder": ["leaf"], "leaf": []}
        graph = model_dependency_graph("outer", adjacency)
        assert [node["id"] for node in graph["nodes"]] == ["outer", "holder", "leaf"]
        assert graph["edges"] == [
            {"source": "outer", "target": "holder"},
            {"source": "holder", "target": "leaf"},
        ]

    def test_cycles_terminate(self):
        graph = model_dependency_graph("a", {"a": ["b"], "b": ["a"]})
        assert [node["id"] for node in graph["nodes"]] == ["a", "b"]
        assert len(graph["edges"]) == 2


class TestModelDependencyView:
    def test_none_without_dependencies(self):
        adjacency = dependency_edges(registry_leaves(_registry_with_dependency()))
        assert model_dependency_view("sub/alpha", adjacency, selected_field="selected") is None

    def test_renders_dagre_component(self):
        adjacency = dependency_edges(registry_leaves(_registry_with_dependency()))
        node = model_dependency_view("holder", adjacency, selected_field="selected").to_node()
        dagre = nodes_with_tag(node, "spaday-dagre")
        assert len(dagre) == 1
        assert prop_value(dagre[0], "graph")["edges"] == [{"source": "holder", "target": "sub/alpha"}]

    def test_node_click_sets_selection(self):
        adjacency = dependency_edges(registry_leaves(_registry_with_dependency()))
        dagre = model_dependency_view("holder", adjacency, selected_field="selected").to_node()
        action = event_action(dagre, "dagre-node-click")
        assert action["kind"] == "set-field"
        assert action["field"] == "selected"
        # The node-click detail is the node id, i.e. the registry path.
        assert action["value"] == {"expr": "event"}

    def test_layout_is_left_to_right(self):
        adjacency = dependency_edges(registry_leaves(_registry_with_dependency()))
        dagre = model_dependency_view("holder", adjacency, selected_field="selected").to_node()
        assert prop_value(dagre, "layout") == {"rankdir": "LR"}

    def test_validates(self):
        adjacency = dependency_edges(registry_leaves(_registry_with_dependency()))
        validate(model_dependency_view("holder", adjacency, selected_field="selected").to_node())
