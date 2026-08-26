"""Per-model dependency graphs, rendered with ``spaday-dagre``.

ccflow models declare which other registered models they contain via
:meth:`ccflow.BaseModel.get_registry_dependencies`. That relation is a DAG over registry paths; each
model's detail card shows only the part reachable from that model, so the graph stays about the model
in front of you. Clicking a node selects it, so the graph doubles as a navigator.
"""

from collections.abc import Mapping

from spaday import Component
from spaday.actions import SetField, event_value
from spaday_dagre import Dagre

__all__ = ("dependency_edges", "model_dependency_graph", "model_dependency_view")

#: Dependencies read left to right.
_LAYOUT = {"rankdir": "LR"}


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
    """The dagre ``{nodes, edges}`` config for everything reachable from ``path``."""
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

    nodes = [{"id": node, "label": node.rsplit("/", 1)[-1], **({"class": "focus"} if node == path else {})} for node in order]
    edges = [{"source": node, "target": target} for node in order for target in adjacency.get(node, ())]
    return {"nodes": nodes, "edges": edges}


def model_dependency_view(path: str, adjacency: dict[str, list[str]], *, selected_field: str) -> Component | None:
    """The model's dependency graph, or ``None`` when it depends on nothing worth drawing.

    ``selected_field`` is the signal-store field a node click writes to, so the graph drives the same
    selection as the sidebar tree.
    """
    graph = model_dependency_graph(path, adjacency)
    if len(graph["nodes"]) < 2:
        return None

    return (
        Dagre(zoomable=True)
        .prop("graph", graph)
        .prop("layout", _LAYOUT)
        .on("dagre-node-click", SetField(selected_field, event_value()))
        .style(display="block", height="22rem")
    )
