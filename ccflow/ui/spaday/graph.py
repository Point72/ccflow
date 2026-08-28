"""Per-model dependency graphs, rendered with ``spaday-dagre``.

ccflow models declare which other registered models they contain via
:meth:`ccflow.BaseModel.get_registry_dependencies`. That relation is a DAG over registry paths; each
model's detail card shows only the part reachable from that model, so the graph stays about the model
in front of you. A node opens a context menu, from which the model can be selected.
"""

from collections.abc import Mapping

from spaday import Component, element
from spaday.actions import Sequence, SetField, by_id, close_popup, event_value, field, open_popup
from spaday.components import Column, Popup
from spaday_dagre import Dagre
from spaday_webawesome import WaButton, WaCard, WaDivider

__all__ = ("MENU_PATH_FIELD", "dependency_edges", "model_dependency_graph", "model_dependency_view")

#: Dependencies flow left to right into the model.
_LAYOUT = {"rankdir": "LR"}

#: Full registry paths make good labels but poor node widths; wider ones ellipsise.
_MAX_LABEL_WIDTH = 220

#: The signal-store field holding the node a context menu was opened on.
MENU_PATH_FIELD = "menu_path"

_MENU_ID = "dependency-node-menu"


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

    nodes = [{"id": node, "label": node} for node in order]
    edges = [{"source": dependency, "target": node} for node in order for dependency in adjacency.get(node, ())]
    return {"nodes": nodes, "edges": edges}


def _node_menu(*, selected_field: str) -> Popup:
    """The context menu shown for a graph node, bound to whichever node opened it."""
    open_model = Sequence(
        SetField(selected_field, field(MENU_PATH_FIELD)),
        close_popup(by_id(_MENU_ID)),
    )
    body = Column(
        element("code").bind("textContent", MENU_PATH_FIELD).style(font_size="0.85em", overflow_wrap="anywhere"),
        WaDivider(),
        WaButton(appearance="filled", size="s").text("Open model").on("click", open_model),
        gap="0.4rem",
    )
    return Popup(WaCard(appearance="outlined").child(body).style(min_width="14rem"), id=_MENU_ID)


def model_dependency_view(path: str, adjacency: dict[str, list[str]], *, selected_field: str) -> Component | None:
    """The model's dependency graph, or ``None`` when it depends on nothing worth drawing.

    A node opens a context menu rather than navigating on the spot; choosing "Open model" from it sets
    ``selected_field``, which routes the page and reveals the model in the sidebar tree.
    """
    graph = model_dependency_graph(path, adjacency)
    if len(graph["nodes"]) < 2:
        return None

    show_menu = open_popup(
        by_id(_MENU_ID),
        x=event_value("x"),
        y=event_value("y"),
        context_field=MENU_PATH_FIELD,
        context=event_value("id"),
    )

    dagre = (
        Dagre(zoomable=True, controls=True, max_label_width=_MAX_LABEL_WIDTH, emphasis=path)
        .prop("graph", graph)
        .prop("layout", _LAYOUT)
        .on("dagre-node-click", show_menu)
        .on("dagre-node-contextmenu", show_menu)
        .style(display="block", height="22rem")
    )

    menu = _node_menu(selected_field=selected_field)
    return Column(dagre, menu, gap="0")
