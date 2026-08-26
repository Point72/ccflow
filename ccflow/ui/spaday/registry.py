"""Registry browser and top-level viewer as a spaday component tree.

Selection is driven entirely client-side through the runtime's signal store: picking a leaf in the
``spaday-tree`` writes the model's path to the ``selected`` field, and each model's detail card is
wrapped in a :class:`~spaday.components.shell.Show` that mounts only when ``selected`` equals its path.
No round-trip to Python is needed to change the selection.
"""

from collections.abc import Mapping

from spaday import Component, Strong, Text
from spaday.actions import SetField, any_, eq, event_value, field, lit, not_
from spaday.components import App, Body, Column, Gutter, Main, Nav, Row, Show
from spaday_trees import Tree
from spaday_webawesome import WaSwitch

import ccflow

from .graph import MENU_PATH_FIELD, dependency_edges, model_dependency_view
from .model import model_view, pending_model_view

__all__ = (
    "DARK_FIELD",
    "SELECTED_FIELD",
    "SELECTED_PATHS_FIELD",
    "registry_leaves",
    "registry_store",
    "registry_tree",
    "registry_viewer",
)

#: The signal-store field holding the selected model's registry path ("" when nothing is selected).
SELECTED_FIELD = "selected"

#: The tree's own selection, as the list of paths it takes. Seeding it makes the tree expand to reveal
#: that model, which is what keeps the tree open across the reload a materialize triggers.
SELECTED_PATHS_FIELD = "selected_paths"

#: The signal-store field driving the ``wa-dark`` page theme.
DARK_FIELD = "dark"


def registry_store(selected: str = "") -> dict:
    """The initial signal-store state the viewer is mounted with."""
    return {
        SELECTED_FIELD: selected,
        SELECTED_PATHS_FIELD: [selected] if selected else [],
        MENU_PATH_FIELD: "",
        DARK_FIELD: False,
    }


def _sorted_items(registry, sort_children: bool):
    """Registry entries, optionally with subregistries first and each group sorted alphabetically."""
    if isinstance(registry, ccflow.LazyRegistry):
        items = []
        for name in registry.models:
            loaded = registry.get_loaded(name)
            items.append((name, loaded if loaded is not None else registry.get_pending_config(name)))
    else:
        items = list(registry.models.items())
    if sort_children:
        items = sorted(items, key=lambda kv: (not isinstance(kv[1], ccflow.ModelRegistry), kv[0]))
    return list(items)


def registry_leaves(registry, *, sort_children: bool = True, _prefix: str = "") -> list[tuple[str, object]]:
    """Return ``(path, model)`` for every leaf model in the registry, depth-first."""
    leaves: list[tuple[str, object]] = []
    for name, model in _sorted_items(registry, sort_children):
        path = f"{_prefix}/{name}" if _prefix else name
        if isinstance(model, ccflow.ModelRegistry):
            leaves.extend(registry_leaves(model, sort_children=sort_children, _prefix=path))
        else:
            leaves.append((path, model))
    return leaves


def registry_tree(registry, *, sort_children: bool = True) -> Tree:
    """Build the registry browser: a path-driven tree whose leaf selection sets ``selected``.

    ``spaday-tree`` derives the hierarchy from the ``/``-separated paths itself and provides its own
    search box, so the whole registry is described by the flat leaf-path list.
    """
    paths = [path for path, _ in registry_leaves(registry, sort_children=sort_children)]
    # The tree virtualizes its rows, so it renders nothing unless it is given a height to fill.
    return (
        Tree(paths=paths, id="registry-tree")
        .bind("selected_paths", SELECTED_PATHS_FIELD)
        .on("selection-change", SetField(SELECTED_FIELD, event_value("paths.0")))
        .style(display="block", flex="1", min_height="70vh")
    )


def _placeholder() -> Component:
    """The main-area hint shown when no model is selected."""
    return Column(
        Strong("Select a model"),
        Text("Choose a model from the registry on the left to inspect its configuration, type, and parameters."),
        gap="0.5rem",
    )


def _details_view(leaves: list[tuple[str, object]]) -> Component:
    """The per-model detail cards, one mounted at a time based on ``selected``."""
    adjacency = dependency_edges(leaves)
    panels: list[Component] = [Show(_placeholder(), when=not_(field(SELECTED_FIELD)))]
    pending_paths = []
    for path, model in leaves:
        if isinstance(model, Mapping) and "_target_" in model:
            pending_paths.append(path)
        else:
            dependency_view = model_dependency_view(path, adjacency, selected_field=SELECTED_FIELD, selected_paths_field=SELECTED_PATHS_FIELD)
            panels.append(Show(model_view(model, path, dependency_view), when=eq(field(SELECTED_FIELD), lit(path))))
    if pending_paths:
        pending_selected = any_(*(eq(field(SELECTED_FIELD), lit(path)) for path in pending_paths))
        panels.append(Show(pending_model_view(field(SELECTED_FIELD)), when=pending_selected))
    return Column(*panels, gap="1rem")


def registry_viewer(registry, *, title: str = "ccflow Model Registry", browser_width: int = 400, sort_children: bool = True) -> App:
    """Compose the full page: a sidebar registry tree and the selected model's detail card."""
    leaves = registry_leaves(registry, sort_children=sort_children)

    sidebar = Gutter(
        Column(Strong("Registry"), registry_tree(registry, sort_children=sort_children), gap="0.75rem"),
        width=f"{browser_width}px",
        gap="0.75rem",
    )

    theme = Row(WaSwitch().text("Dark").bind("checked", DARK_FIELD, mode="two-way"), gap="0.5rem", align="center")

    return App(
        Nav(Row(Strong(title), theme, gap="1rem", align="center", justify="space-between")),
        Body(sidebar, Main(_details_view(leaves))),
    ).bind_root_class("wa-dark", DARK_FIELD)
