"""Registry dependency graph, rendered with ``spaday-dagre``.

ccflow models declare which other registered models they contain via
:meth:`ccflow.BaseModel.get_registry_dependencies`. That relation is a DAG over registry paths, which
this module turns into the serializable ``{nodes, edges}`` config the ``spaday-dagre`` component lays
out. Clicking a node selects that model, so the graph doubles as a navigator.
"""

from collections.abc import Mapping

from spaday import Component, Strong, Text
from spaday.actions import SetField, event_value, field, obj
from spaday.components import Column, Row
from spaday_dagre import Dagre
from spaday_webawesome import WaButton

__all__ = ("DEPENDENCY_RANKDIR_FIELD", "dependency_graph", "dependency_graph_view")

#: The signal-store field holding the dagre ``rankdir`` layout direction.
DEPENDENCY_RANKDIR_FIELD = "rankdir"


def _is_pending(model) -> bool:
    """Whether the entry is an un-instantiated (lazy) registry config rather than a model."""
    return isinstance(model, Mapping) and "_target_" in model


def _normalize(name: str) -> str:
    """Registered names are root-relative and leading-slashed ("/a/b"); leaf paths are not."""
    return name.removeprefix("/")


def dependency_graph(leaves: list[tuple[str, object]]) -> dict:
    """Build the dagre ``{nodes, edges}`` config for the registry's dependency relation.

    Only edges between models present in ``leaves`` are emitted, so a dependency on something outside
    the browsed registry does not introduce a dangling node. Pending (lazy) models are shown, but
    contribute no edges because resolving them would instantiate the model.
    """
    known = {path for path, _ in leaves}
    nodes = []
    edges = []

    for path, model in leaves:
        pending = _is_pending(model)
        node = {"id": path, "label": path.rsplit("/", 1)[-1]}
        if pending:
            node["class"] = "pending"
        nodes.append(node)
        if pending:
            continue
        for group in model.get_registry_dependencies():
            # A group holds equivalent names for one dependency; the first is the canonical path.
            target = _normalize(group[0])
            if target in known and target != path:
                edges.append({"source": path, "target": target})

    # Deduplicate edges while preserving order (a model may reference the same dependency twice).
    seen = set()
    unique_edges = []
    for edge in edges:
        key = (edge["source"], edge["target"])
        if key not in seen:
            seen.add(key)
            unique_edges.append(edge)

    return {"nodes": nodes, "edges": unique_edges}


def _rankdir_button(label: str, rankdir: str) -> WaButton:
    return WaButton(appearance="outlined", size="s").text(label).on("click", SetField(DEPENDENCY_RANKDIR_FIELD, rankdir))


def dependency_graph_view(leaves: list[tuple[str, object]], *, selected_field: str) -> Component:
    """The dependency graph panel: layout controls plus the graph itself.

    ``selected_field`` is the signal-store field a node click writes to, so the graph drives the same
    selection as the sidebar tree.
    """
    graph = dependency_graph(leaves)

    if not graph["nodes"]:
        return Column(Strong("No models"), Text("This registry has no models to graph."), gap="0.5rem")

    if not graph["edges"]:
        header = Text("No registry dependencies between these models.")
    else:
        header = Text(f"{len(graph['nodes'])} models, {len(graph['edges'])} dependencies. Click a node to inspect it.")

    controls = Row(
        _rankdir_button("Left to right", "LR"),
        _rankdir_button("Top down", "TB"),
        gap="0.5rem",
        align="center",
    )

    dagre = (
        Dagre(id="dependency-graph", zoomable=True)
        .prop("graph", graph)
        .compute("layout", obj({"rankdir": field(DEPENDENCY_RANKDIR_FIELD)}))
        .on("dagre-node-click", SetField(selected_field, event_value()))
        .style(display="block", min_height="60vh")
    )

    return Column(header, controls, dagre, gap="0.75rem")
