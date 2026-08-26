"""Per-model dependency graphs, rendered with ``spaday-dagre``.

ccflow models declare which other registered models they contain via
:meth:`ccflow.BaseModel.get_registry_dependencies`. That relation is a DAG over registry paths; each
model's detail card shows only the part reachable from that model, so the graph stays about the model
in front of you. Clicking a node selects it, so the graph doubles as a navigator.
"""

from collections.abc import Mapping

from spaday import Component, Strong, element
from spaday.actions import If, Sequence, SetField, by_id, close_popup, eq, event_prop, field, lit, open_popup
from spaday.components import Column, Popup, Show
from spaday_dagre import Dagre
from spaday_webawesome import WaButton, WaCard, WaDivider

__all__ = ("MENU_PATH_FIELD", "dependency_edges", "model_dependency_graph", "model_dependency_view")

#: Dependencies flow left to right into the model.
_LAYOUT = {"rankdir": "LR"}

#: The signal-store field holding the node a context menu was opened on.
MENU_PATH_FIELD = "menu_path"

_MENU_ID = "dependency-node-menu"

#: dagre draws each node as ``<g data-node-id>`` wrapping a ``<rect>`` and a ``<text>``, so a pointer
#: event lands on a child and the node is one level up.
_EVENT_NODE_ID = "target.parentElement.dataset.nodeId"


def _is_pending(model) -> bool:
    """Whether the entry is an un-instantiated (lazy) registry config rather than a model."""
    return isinstance(model, Mapping) and "_target_" in model


def _normalize(name: str) -> str:
    """Registered names are root-relative and leading-slashed ("/a/b"); leaf paths are not."""
    return name.removeprefix("/")


def dependency_edges(leaves: list[tuple[str, object]]) -> dict[str, list[str]]:
    """Map each leaf path to the paths it depends on.

    Only dependencies present in ``leaves`` are kept, so a reference to something outside the browsed
    registry does not introduce a dangling node. Pending (lazy) models report none, because resolving
    them would instantiate the model.
    """
    known = {path for path, _ in leaves}
    adjacency: dict[str, list[str]] = {}
    for path, model in leaves:
        targets: list[str] = []
        if not _is_pending(model):
            for group in model.get_registry_dependencies():
                # A group holds equivalent names for one dependency; the first is the canonical path.
                target = _normalize(group[0])
                if target in known and target != path and target not in targets:
                    targets.append(target)
        adjacency[path] = targets
    return adjacency


def model_dependency_graph(path: str, adjacency: dict[str, list[str]]) -> dict:
    """The dagre ``{nodes, edges}`` config for everything reachable from ``path``.

    Edges point from a dependency to the model that uses it, so the graph reads in dataflow order and
    the model in front of you is the last node.
    """
    if path not in adjacency:
        return {"nodes": [], "edges": []}

    order: list[str] = []
    seen = {path}
    queue = [path]
    while queue:
        current = queue.pop(0)
        order.append(current)
        for target in adjacency.get(current, ()):
            if target not in seen:
                seen.add(target)
                queue.append(target)

    nodes = [{"id": node, "label": node, **({"class": "focus"} if node == path else {})} for node in order]
    edges = [{"source": dependency, "target": node} for node in order for dependency in adjacency.get(node, ())]
    return {"nodes": nodes, "edges": edges}


def _node_menu(paths: list[str], *, selected_field: str, selected_paths_field: str) -> Popup:
    """The context menu shown for a graph node.

    One body per node, gated on which node was clicked, so each carries literal actions: the action DSL
    has no way to build the single-element list the tree's ``selected_paths`` needs from a store value.
    """
    entries = []
    for path in paths:
        open_model = Sequence(
            SetField(selected_field, lit(path)),
            SetField(selected_paths_field, lit([path])),
            close_popup(by_id(_MENU_ID)),
        )
        entries.append(
            Show(
                Column(
                    Strong(path.rsplit("/", 1)[-1]),
                    element("code").text(path).style(font_size="0.8em", overflow_wrap="anywhere"),
                    WaDivider(),
                    WaButton(appearance="filled", size="s").text("Open model").on("click", open_model),
                    gap="0.4rem",
                ),
                when=eq(field(MENU_PATH_FIELD), lit(path)),
            )
        )

    card = WaCard(appearance="outlined").child(Column(*entries, gap="0.4rem")).style(min_width="14rem")
    return Popup(card, id=_MENU_ID)


def model_dependency_view(path: str, adjacency: dict[str, list[str]], *, selected_field: str, selected_paths_field: str) -> Component | None:
    """The model's dependency graph, or ``None`` when it depends on nothing worth drawing.

    A node opens a context menu rather than navigating on the spot; choosing "Open model" from it sets
    ``selected_field`` and reveals the model in the sidebar tree via ``selected_paths_field``.
    """
    graph = model_dependency_graph(path, adjacency)
    if len(graph["nodes"]) < 2:
        return None

    # The node id lives on the group above the clicked shape, and is absent when the pointer misses a
    # node, which is what keeps the menu from opening over blank canvas.
    node_id = event_prop(_EVENT_NODE_ID)
    show_menu = If(node_id, open_popup(by_id(_MENU_ID), context_field=MENU_PATH_FIELD, context=node_id))

    dagre = (
        Dagre(zoomable=True)
        .prop("graph", graph)
        .prop("layout", _LAYOUT)
        .on("click", show_menu)
        .on("contextmenu", show_menu)
        .style(display="block", height="22rem")
    )

    paths = [node["id"] for node in graph["nodes"]]
    return Column(dagre, _node_menu(paths, selected_field=selected_field, selected_paths_field=selected_paths_field), gap="0")
