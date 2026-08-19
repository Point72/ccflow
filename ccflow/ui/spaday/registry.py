"""Registry browser and top-level viewer as a spaday component tree.

Selection is driven entirely client-side through the runtime's signal store: clicking a leaf in the
``wa-tree`` (or picking it from the search ``wa-select``) writes the model's path to the ``selected``
field, and each model's detail card is wrapped in a :class:`~spaday.components.shell.Show` that mounts
only when ``selected`` equals its path. No round-trip to Python is needed to change the selection.
"""

from typing import List, Tuple

from spaday import Component, Strong, Text
from spaday.actions import SetField, eq, field, lit
from spaday.components import App, Body, Column, Gutter, Main, Nav, Show, WaOption, WaSelect, WaTree, WaTreeItem

import ccflow

from .model import model_view, pending_model_view

__all__ = ("SELECTED_FIELD", "registry_store", "registry_leaves", "registry_tree", "registry_viewer")

#: The signal-store field holding the selected model's registry path ("" when nothing is selected).
SELECTED_FIELD = "selected"


def registry_store() -> dict:
    """The initial signal-store state the viewer is mounted with."""
    return {SELECTED_FIELD: ""}


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


def registry_leaves(registry, *, sort_children: bool = True, _prefix: str = "") -> List[Tuple[str, object]]:
    """Return ``(path, model)`` for every leaf model in the registry, depth-first."""
    leaves: List[Tuple[str, object]] = []
    for name, model in _sorted_items(registry, sort_children):
        path = f"{_prefix}/{name}" if _prefix else name
        if isinstance(model, ccflow.ModelRegistry):
            leaves.extend(registry_leaves(model, sort_children=sort_children, _prefix=path))
        else:
            leaves.append((path, model))
    return leaves


def registry_tree(registry, *, sort_children: bool = True, _prefix: str = "") -> List[WaTreeItem]:
    """Build the ``wa-tree-item`` nodes for the registry; leaf clicks select the model by path."""
    nodes: List[WaTreeItem] = []
    for name, model in _sorted_items(registry, sort_children):
        path = f"{_prefix}/{name}" if _prefix else name
        if isinstance(model, ccflow.ModelRegistry):
            children = registry_tree(model, sort_children=sort_children, _prefix=path)
            nodes.append(WaTreeItem(Text(name), *children))
        else:
            nodes.append(WaTreeItem(Text(name)).on("click", SetField(SELECTED_FIELD, lit(path))))
    return nodes


def _placeholder() -> Component:
    """The main-area hint shown when no model is selected."""
    return Column(
        Strong("Select a model"),
        Text("Choose a model from the registry on the left to inspect its configuration, type, and parameters."),
        gap="0.5rem",
    )


def _search(leaves: List[Tuple[str, object]]) -> WaSelect:
    """A select of every model path, two-way bound to the selection so it both jumps and reflects."""
    options = [WaOption(value="").text("— jump to a model —")]
    options += [WaOption(value=path).text(path) for path, _ in sorted(leaves)]
    return WaSelect(placeholder="Search / jump to model", with_clear=True).child(*options).bind("value", SELECTED_FIELD, mode="two-way")


def registry_viewer(registry, *, title: str = "ccflow Model Registry", browser_width: int = 400, sort_children: bool = True) -> App:
    """Compose the full page: a sidebar registry tree + search, and the selected model's detail card."""
    leaves = registry_leaves(registry, sort_children=sort_children)
    tree = WaTree(*registry_tree(registry, sort_children=sort_children), selection="leaf")

    sidebar = Gutter(
        Column(Strong("Registry"), _search(leaves), tree, gap="0.75rem"),
        width=f"{browser_width}px",
        gap="0.75rem",
    )

    panels: List[Component] = [Show(_placeholder(), when=eq(field(SELECTED_FIELD), lit("")))]
    for path, model in leaves:
        detail = pending_model_view(model, path) if isinstance(model, dict) and "_target_" in model else model_view(model, path)
        panels.append(Show(detail, when=eq(field(SELECTED_FIELD), lit(path))))

    return App(
        Nav(Strong(title)),
        Body(sidebar, Main(Column(*panels, gap="1rem"))),
    )
