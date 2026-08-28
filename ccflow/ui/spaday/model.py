"""Model-detail components for the spaday registry viewer.

Each function builds a piece of the model inspector as a :class:`spaday.Component` tree (rendered to the
browser by the spaday runtime), mirroring the tabs of the Panel viewer in :mod:`ccflow.ui.panel.model`:
an instance summary, the model / context / result types with their fields, and the serialized parameters.
"""

import json

from pydantic._internal._repr import display_as_type
from spaday import Component, Strong, Text, element
from spaday.actions import CallEndpoint, Expr, RefreshTree, Sequence, obj
from spaday.components import Column, Row
from spaday_webawesome import Tabs, WaBadge, WaButton, WaCard, WaDivider

import ccflow

#: Path of the endpoint (served by :func:`ccflow.ui.spaday.cli.serve_registry`) that materializes a
#: pending model server-side.
MATERIALIZE_ENDPOINT = "/materialize"

#: Signal-store field holding the materialize call's ``{status, ok, body}`` outcome.
MATERIALIZE_RESULT_FIELD = "materialize_result"

__all__ = (
    "MATERIALIZE_ENDPOINT",
    "MATERIALIZE_RESULT_FIELD",
    "model_config_view",
    "model_type_view",
    "model_view",
    "pending_model_view",
)

_PRE_STYLE = {
    "white_space": "pre-wrap",
    "font_family": "monospace",
    "background": "#f6f8fa",
    "padding": "8px",
    "margin": "0",
    "border_radius": "4px",
    "overflow_wrap": "anywhere",
}


def _labeled(label: str, *body: Component) -> Component:
    """A bold label above its content."""
    return Column(Strong(label), *body, gap="0.25rem")


def _code(text: str | Expr, *, color: str = "") -> Component:
    """An inline ``<code>`` element that wraps long identifiers."""
    node = element("code").text(text).style(overflow_wrap="anywhere")
    return node.style(color=color) if color else node


def _pre(text: str) -> Component:
    """A preformatted code block."""
    return element("pre").text(text).style(**_PRE_STYLE)


def model_type_view(model_cls) -> Component:
    """Show a Pydantic model type's name, class docstring, and fields."""
    if model_cls is None:
        return Column()

    children = [Row(Strong("Type:"), WaBadge(variant="brand").text(display_as_type(model_cls)), gap="0.5rem", align="center")]

    docs = (model_cls.__doc__ or "").strip()
    if docs:
        children.append(_labeled("Class Documentation", _pre(docs)))

    fields = getattr(model_cls, "model_fields", {})
    if fields:
        items = element("ul").style(margin="0", padding_left="18px")
        for name, field in fields.items():
            entry = element("li").style(overflow_wrap="anywhere")
            entry.child(_code(name, color="#0550ae"))
            entry.child(Text(f" ({display_as_type(field.annotation)})"))
            if field.description:
                entry.child(Text(f" — {field.description}"))
            items.child(entry)
        children.append(_labeled("Fields", items))

    return Column(*children, gap="0.75rem")


def _dependencies_view(model) -> Component:
    """A bulleted list of the model's registry dependencies, or ``None`` if it has none."""
    deps = model.get_registry_dependencies()
    if not deps:
        return None

    rows = sorted({group[0] if len(group) == 1 else " | ".join(group) for group in deps})
    items = element("ul").style(margin="0", padding_left="18px")
    for row in rows:
        items.child(element("li").child(_code(row)))
    return _labeled("Registry Dependencies", items)


def model_config_view(model, path: str = "") -> Component:
    """Show instance-level metadata: registry path, description, and dependencies."""
    children = []

    if path:
        children.append(_labeled("Registry Path", _code(path)))

    description = model.meta.description.strip() if hasattr(model, "meta") and model.meta.description else ""
    if description:
        children.append(_labeled("Instance Description", element("div").text(description)))

    dependencies = _dependencies_view(model)
    if dependencies is not None:
        children.append(dependencies)

    if not children:
        children.append(Text("No additional metadata."))

    return Column(*children, gap="0.75rem")


def model_view(model, path: str = "", dependency_view: Component | None = None) -> Component:
    """A card with tabs inspecting a single ccflow model instance.

    ``dependency_view`` is this model's dependency graph, added as a tab when it has one.
    """
    type_name = display_as_type(type(model))

    tabs = Tabs(active="summary")
    tabs.tab("Summary", model_config_view(model, path), name="summary")
    tabs.tab("Model Type", model_type_view(type(model)), name="model-type")
    if isinstance(model, ccflow.CallableModel):
        tabs.tab("Context Type", model_type_view(model.context_type), name="context-type")
        tabs.tab("Result Type", model_type_view(model.result_type), name="result-type")
    if dependency_view is not None:
        tabs.tab("Dependencies", dependency_view, name="dependencies")

    params = model.__pydantic_serializer__.to_python(model, fallback=str, mode="json")
    tabs.tab("Parameters", _pre(json.dumps(params, indent=2, default=str)), name="parameters")

    header = Row(WaBadge(variant="brand").text(type_name), Strong(path or type_name), gap="0.5rem", align="center")
    return WaCard(appearance="outlined").child(Column(header, WaDivider(), tabs, gap="0.75rem"))


def _materialize_button(path: str | Expr) -> Component:
    """Instantiate the pending model server-side, then diff the refreshed tree into the page."""
    materialize = Sequence(
        CallEndpoint("POST", MATERIALIZE_ENDPOINT, obj({"path": path}), result=MATERIALIZE_RESULT_FIELD),
        RefreshTree(),
    )
    return WaButton(variant="brand").text("Materialize").on("click", materialize)


def pending_model_view(path: str | Expr) -> Component:
    """A card for a selected model that has not been instantiated.

    ``Materialize`` instantiates it on the server and refreshes the tree in place, so the card becomes
    the full :func:`model_view` detail without a reload. Failures are reported from
    :data:`MATERIALIZE_RESULT_FIELD`.
    """
    tabs = Tabs(active="summary")
    tabs.tab(
        "Summary",
        Column(
            _labeled("Registry Path", _code(path)),
            Text("This model has not been instantiated. Materialize it to inspect its type, context, result, and parameters."),
            _materialize_button(path),
            gap="0.75rem",
        ),
        name="summary",
    )
    header = Row(WaBadge(variant="neutral").text("Pending"), Strong("Pending model"), gap="0.5rem", align="center")
    return WaCard(appearance="outlined").child(Column(header, WaDivider(), tabs, gap="0.75rem"))
