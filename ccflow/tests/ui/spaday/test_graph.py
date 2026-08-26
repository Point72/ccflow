"""Unit tests for ccflow.ui.spaday.graph module."""

from spaday.validate import validate

from ccflow import BaseModel, LazyRegistry, ModelRegistry
from ccflow.ui.spaday.graph import dependency_edges, model_dependency_graph, model_dependency_view
from ccflow.ui.spaday.registry import registry_leaves

from .utils import event_action, nodes_with_tag, prop_value


def _view(path, adjacency):
    return model_dependency_view(path, adjacency, selected_field="selected", selected_paths_field="selected_paths")


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

    def test_focus_node_is_marked(self):
        graph = model_dependency_graph("holder", {"holder": ["sub/alpha"], "sub/alpha": []})
        assert graph["nodes"][0]["class"] == "focus"
        assert "class" not in graph["nodes"][1]

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

    def test_node_click_opens_the_menu_instead_of_navigating(self):
        adjacency = dependency_edges(registry_leaves(_registry_with_dependency()))
        node = _view("holder", adjacency).to_node()
        dagre = nodes_with_tag(node, "spaday-dagre")[0]
        for event in ("click", "contextmenu"):
            action = event_action(dagre, event)
            # Guarded on the node id, so a click on blank canvas opens nothing.
            assert action["kind"] == "if"
            assert action["cond"] == {"expr": "event-prop", "path": "target.parentElement.dataset.nodeId"}
            assert action["then"]["kind"] == "seq"

    def test_menu_has_a_body_per_node(self):
        adjacency = dependency_edges(registry_leaves(_registry_with_dependency()))
        node = _view("holder", adjacency).to_node()
        assert nodes_with_tag(node, "spa-popup")
        shows = nodes_with_tag(node, "spa-show")
        assert len(shows) == 2  # one per node in the graph

    def test_open_model_sets_selection_and_reveals_in_tree(self):
        adjacency = dependency_edges(registry_leaves(_registry_with_dependency()))
        node = _view("holder", adjacency).to_node()
        buttons = [n for n in nodes_with_tag(node, "wa-button") if "click" in n.get("events", {})]
        writes = {}
        for button in buttons:
            for action in button["events"]["click"]["actions"]:
                if action["kind"] == "set-field":
                    writes.setdefault(action["field"], []).append(action["value"]["value"])
        # Literal per node, because the DSL cannot build the tree's single-element path list.
        assert set(writes["selected"]) == {"holder", "sub/alpha"}
        assert ["holder"] in writes["selected_paths"]
        assert ["sub/alpha"] in writes["selected_paths"]

    def test_layout_is_left_to_right(self):
        adjacency = dependency_edges(registry_leaves(_registry_with_dependency()))
        dagre = nodes_with_tag(_view("holder", adjacency).to_node(), "spaday-dagre")[0]
        assert prop_value(dagre, "layout") == {"rankdir": "LR"}

    def test_validates(self):
        adjacency = dependency_edges(registry_leaves(_registry_with_dependency()))
        validate(_view("holder", adjacency).to_node())
