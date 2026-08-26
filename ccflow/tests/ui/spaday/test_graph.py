"""Unit tests for ccflow.ui.spaday.graph module."""

from spaday.validate import validate

from ccflow import BaseModel, LazyRegistry, ModelRegistry
from ccflow.ui.spaday.graph import DEPENDENCY_RANKDIR_FIELD, dependency_graph, dependency_graph_view
from ccflow.ui.spaday.registry import registry_leaves

from .utils import all_text, event_action, nodes_with_tag, prop_value


class Leaf(BaseModel):
    """A model with no registry dependencies."""

    name: str = "leaf"


class Holder(BaseModel):
    """A model that contains another registered model."""

    child: Leaf


def _registry_with_dependency():
    root = ModelRegistry.root()
    root.clear()
    leaf = Leaf(name="a")
    sub = ModelRegistry(name="sub")
    sub.add("alpha", leaf)
    root.add("sub", sub)
    root.add("holder", Holder(child=leaf))
    return root


class TestDependencyGraph:
    def test_empty_registry(self):
        assert dependency_graph([]) == {"nodes": [], "edges": []}

    def test_nodes_use_leaf_name_as_label(self):
        graph = dependency_graph(registry_leaves(_registry_with_dependency()))
        labels = {node["id"]: node["label"] for node in graph["nodes"]}
        assert labels == {"sub/alpha": "alpha", "holder": "holder"}

    def test_edge_from_dependent_to_dependency(self):
        graph = dependency_graph(registry_leaves(_registry_with_dependency()))
        assert graph["edges"] == [{"source": "holder", "target": "sub/alpha"}]

    def test_no_edges_without_dependencies(self):
        root = ModelRegistry.root()
        root.clear()
        root.add("one", Leaf(name="one"))
        root.add("two", Leaf(name="two"))
        graph = dependency_graph(registry_leaves(root))
        assert len(graph["nodes"]) == 2
        assert graph["edges"] == []

    def test_dependencies_outside_registry_are_dropped(self):
        root = ModelRegistry.root()
        root.clear()
        leaf = Leaf(name="hidden")
        root.add("hidden", leaf)
        holder_registry = ModelRegistry(name="holder_only")
        holder_registry.add("holder", Holder(child=leaf))
        # Browsing a registry that does not contain the dependency must not invent a dangling node.
        graph = dependency_graph(registry_leaves(holder_registry))
        assert [node["id"] for node in graph["nodes"]] == ["holder"]
        assert graph["edges"] == []

    def test_pending_models_are_nodes_without_edges(self):
        lazy = LazyRegistry(
            name="lazy",
            group={"model": {"_target_": "ccflow.tests.ui.spaday.test_graph.Leaf", "name": "pending"}},
        )
        graph = dependency_graph(registry_leaves(lazy))
        assert [node["id"] for node in graph["nodes"]] == ["group/model"]
        assert graph["nodes"][0]["class"] == "pending"
        assert graph["edges"] == []
        assert not lazy["group"].is_loaded("model")


class TestDependencyGraphView:
    def test_renders_dagre_component(self):
        view = dependency_graph_view(registry_leaves(_registry_with_dependency()), selected_field="selected")
        node = view.to_node()
        dagre = nodes_with_tag(node, "spaday-dagre")
        assert len(dagre) == 1
        assert prop_value(dagre[0], "graph")["edges"] == [{"source": "holder", "target": "sub/alpha"}]

    def test_node_click_sets_selection(self):
        view = dependency_graph_view(registry_leaves(_registry_with_dependency()), selected_field="selected")
        dagre = nodes_with_tag(view.to_node(), "spaday-dagre")[0]
        action = event_action(dagre, "dagre-node-click")
        assert action["kind"] == "set-field"
        assert action["field"] == "selected"
        # The node-click detail is the node id, i.e. the registry path.
        assert action["value"] == {"expr": "event"}

    def test_layout_bound_to_rankdir_field(self):
        view = dependency_graph_view(registry_leaves(_registry_with_dependency()), selected_field="selected")
        dagre = nodes_with_tag(view.to_node(), "spaday-dagre")[0]
        layout = dagre["bindings"]["layout"]["compute"]
        assert layout["fields"]["rankdir"] == {"expr": "field", "name": DEPENDENCY_RANKDIR_FIELD}

    def test_empty_registry_message(self):
        view = dependency_graph_view([], selected_field="selected")
        node = view.to_node()
        assert not nodes_with_tag(node, "spaday-dagre")
        assert "No models" in all_text(node)

    def test_validates(self):
        view = dependency_graph_view(registry_leaves(_registry_with_dependency()), selected_field="selected")
        validate(view.to_node())
