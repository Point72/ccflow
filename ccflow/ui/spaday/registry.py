"""Registry browser and top-level viewer as a spaday component tree.

Selection is driven client-side through the runtime's signal store: picking a leaf in the
``spaday-tree`` writes the model's path to the ``selected`` field, and a :class:`Switch` routes to that
model's detail card. ``selected`` is bound to a URL query parameter by the server, so a model is
linkable and back/forward navigate between models.
"""

from collections.abc import Mapping
from urllib.parse import quote

from spaday import Component, Strong, Text, element
from spaday.actions import SetField, arr, cond, event_value, field, lit
from spaday.components import App, Body, Column, Gutter, Lazy, Main, Nav, Row, Switch, Toast
from spaday_trees import Tree
from spaday_webawesome import WaSwitch

import ccflow

from .graph import MENU_PATH_FIELD, dependency_edges, model_dependency_view
from .model import MATERIALIZE_RESULT_FIELD, model_view, pending_model_view

__all__ = (
    "CARD_ENDPOINT",
    "DARK_FIELD",
    "SELECTED_FIELD",
    "model_card",
    "registry_leaves",
    "registry_store",
    "registry_tree",
    "registry_viewer",
)

#: Path of the endpoint (served by :func:`ccflow.ui.spaday.cli.serve_registry`) returning one model's
#: card, so the initial page carries a placeholder per model rather than every card.
CARD_ENDPOINT = "/card"

#: The signal-store field holding the selected model's registry path ("" when nothing is selected).
SELECTED_FIELD = "selected"

#: The signal-store field driving the ``wa-dark`` page theme.
DARK_FIELD = "dark"


def registry_store() -> dict:
    """The initial signal-store state the viewer is mounted with."""
    return {
        SELECTED_FIELD: "",
        MENU_PATH_FIELD: "",
        MATERIALIZE_RESULT_FIELD: {},
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


def _is_pending(model) -> bool:
    """Whether the entry is an un-instantiated (lazy) registry config rather than a model."""
    return isinstance(model, Mapping) and "_target_" in model


def registry_tree(registry, *, sort_children: bool = True) -> Tree:
    """Build the registry browser: a path-driven tree whose leaf selection sets ``selected``.

    ``spaday-tree`` derives the hierarchy from the ``/``-separated paths itself and provides its own
    search box, so the whole registry is described by the flat leaf-path list.
    """
    leaves = registry_leaves(registry, sort_children=sort_children)
    decorations = {
        path: {"badge": "lazy", "tone": "warning", "tooltip": "Configured but not yet instantiated"} for path, model in leaves if _is_pending(model)
    }
    # The tree virtualizes its rows, so it renders nothing unless it is given a height to fill.
    return (
        Tree(paths=[path for path, _ in leaves], decorations=decorations, id="registry-tree")
        # Revealing the selection is what expands the tree to a deep-linked model.
        .compute("selected_paths", cond(field(SELECTED_FIELD), arr(field(SELECTED_FIELD)), lit([])))
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


def _loading() -> Component:
    """Shown while a card is being fetched."""
    return Text("Loading…")


def _code(text: str) -> Component:
    return element("code").text(text).style(overflow_wrap="anywhere")


def model_card(registry, path: str, *, sort_children: bool = True) -> Component:
    """The detail card for one model, built on demand for :data:`CARD_ENDPOINT`.

    Dependencies are resolved against the whole registry, so the graph is the same as it would be if
    the card had been inlined.
    """
    leaves = registry_leaves(registry, sort_children=sort_children)
    model = next((candidate for leaf_path, candidate in leaves if leaf_path == path), None)
    if model is None:
        return Column(Strong("Unknown model"), _code(path), gap="0.5rem")
    if _is_pending(model):
        return pending_model_view(path)
    dependency_view = model_dependency_view(path, dependency_edges(leaves), selected_field=SELECTED_FIELD)
    return model_view(model, path, dependency_view)


def _details_view(leaves: list[tuple[str, object]]) -> Component:
    """Route to the selected model's card, each deferred so only the visible one is ever fetched."""
    cases = {path: Lazy(_loading(), src=f"{CARD_ENDPOINT}?model={quote(path)}") for path, _ in leaves}
    return Switch(SELECTED_FIELD, cases, default=_placeholder())


def registry_viewer(registry, *, title: str = "ccflow Model Registry", browser_width: int = 400, sort_children: bool = True) -> App:
    """Compose the full page: a sidebar registry tree and the selected model's detail card."""
    leaves = registry_leaves(registry, sort_children=sort_children)

    sidebar = Gutter(
        Column(Strong("Registry"), registry_tree(registry, sort_children=sort_children), gap="0.75rem"),
        width=f"{browser_width}px",
        gap="0.75rem",
    )

    theme = Row(WaSwitch().text("Dark").bind("checked", DARK_FIELD, mode="two-way"), gap="0.5rem", align="center")

    # A materialize that fails server-side (a bad _target_, a missing dependency) reports here.
    toasts = Toast(tone="danger", timeout=0, id="materialize-toasts").compute(
        "message",
        cond(field(f"{MATERIALIZE_RESULT_FIELD}.ok"), lit(""), field(f"{MATERIALIZE_RESULT_FIELD}.body.message")),
    )

    return App(
        Nav(Row(Strong(title), theme, gap="1rem", align="center", justify="space-between")),
        Body(sidebar, Main(_details_view(leaves))),
        toasts,
    ).bind_root_class("wa-dark", DARK_FIELD)
