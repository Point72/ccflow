import html

import bleach
import panel as pn

# Register extensions
import panel_material_ui  # noqa: F401
import panel_material_ui as pmui
import param
from pydantic._internal._repr import display_as_type

import ccflow

pn.extension()
pn.extension("jsoneditor")


__all__ = ("ModelTypeViewer", "ModelViewer", "ModelConfigViewer")


class ModelTypeViewer(param.Parameterized):
    """
    Displays type name, class docstring, and fields for a Pydantic model type.
    """

    model_type = param.Parameter(default=None)

    def __init__(self, **params):
        super().__init__(**params)

        self._pane = pn.pane.HTML("", width=1200)
        self._layout = pn.Column(
            self._pane,
        )

        self.param.watch(self._on_type_change, "model_type")

    def __panel__(self):
        return self._layout

    def _on_type_change(self, event):
        model_cls = event.new
        if model_cls is None:
            self._pane.object = ""
            return

        type_name = display_as_type(model_cls)

        # Class documentation
        docs = (model_cls.__doc__ or "").strip()
        docs_html = ""
        if docs:
            escaped = html.escape(docs).replace("\n", "<br>")
            docs_html = f"""
            <div style="margin:8px 0;">
              <div style="font-weight:600;">Class Documentation:</div>
              <pre style="
                  white-space:normal;
                  font-family:monospace;
                  background:#f6f8fa;
                  padding:8px;
                  margin:0;
                  border-radius:4px;
              "><code>{escaped}</code></pre>
            </div>
            """

        # Fields
        fields = getattr(model_cls, "model_fields", {})
        field_items = []

        for name, field in fields.items():
            field_type = display_as_type(field.annotation)
            desc = field.description or ""
            field_items.append(
                f"<li><code>{html.escape(name)}</code> (<code>{html.escape(field_type)}</code>){': ' + html.escape(desc) if desc else ''}</li>"
            )

        fields_html = ""
        if field_items:
            fields_html = f"""
            <div style="margin-top:8px;">
              <div style="font-weight:600;">Fields:</div>
              <ul style="margin:0;padding-left:18px;">
                {"".join(field_items)}
              </ul>
            </div>
            """

        self._pane.object = f"""
        <div>
          <div style="margin-bottom:6px;">
            <span style="font-weight:600;">Type:</span>
            <code>{html.escape(type_name)}</code>
          </div>
          {docs_html}
          {fields_html}
        </div>
        """


class ModelConfigViewer(param.Parameterized):
    """
    Displays instance-level metadata (description + dependencies).
    """

    model = param.Parameter(default=None)

    def __init__(self, **params):
        super().__init__(**params)

        self._metadata = pn.pane.HTML("", width=1200)

        self._layout = pn.Column(
            self._metadata,
        )

        self.param.watch(self._on_model_change, "model")

    def __panel__(self):
        return self._layout

    # ------------------------------------------------------------

    def _render_dependencies(self, model):
        deps = model.get_registry_dependencies()
        if not deps:
            return ""

        # Collect all values, deduplicate, and sort
        all_paths = []
        for group in deps:
            if len(group) == 1:
                all_paths.append(group[0])
            else:
                all_paths.append(" | ".join(group))

        # Unique elements, sorted
        rows = sorted(set(all_paths))

        items = "".join(f"<li>{html.escape(row)}</li>" for row in rows)

        return f"""
        <div style="margin-top:8px;">
          <div style="font-weight:600;margin-bottom:4px;">
            Registry Dependencies
          </div>
          <ul style="margin:0;padding-left:18px;">
            {items}
          </ul>
        </div>
        """

    def _on_model_change(self, event):
        model = event.new
        if model is None:
            self._metadata.object = ""
            return

        description = model.meta.description.strip() if hasattr(model, "meta") and model.meta.description else ""

        desc_html = ""
        if description:
            desc_html = f"""
            <div style="margin-bottom:6px;">
              <div style="font-weight:600;">Instance Description</div>
              <div>{bleach.linkify(html.escape(description))}</div>
            </div>
            """

        self._metadata.object = desc_html + self._render_dependencies(model)


class ModelViewer(param.Parameterized):
    """
    Displays a tabbed view of a ccflow Model instance, including description, registry dependencies, docstrings and json representation.
    """

    model = param.Parameter(default=None)

    def __init__(self, **params):
        super().__init__(**params)

        # Sub-viewers (no JSONEditor inside)
        self._config_viewer = ModelConfigViewer()
        self._type_viewer = ModelTypeViewer()
        self._context_type_viewer = ModelTypeViewer()
        self._result_type_viewer = ModelTypeViewer()

        # Material UI Tabs (metadata only)
        self._tabs = pmui.Tabs(
            active=0,
            sizing_mode="stretch_width",
        )

        # JSON editor (stable, but hidden until a model is selected)
        self._json_editor = pn.widgets.JSONEditor(
            value={},
            mode="view",
            menu=False,
            width=600,
        )

        self._json_container = pn.Column(
            "## Parameters",
            self._json_editor,
            visible=False,  # hidden initially
        )

        self._layout = pn.Column(
            "## Model Viewer",
            self._tabs,
            pn.Spacer(height=12),
            self._json_container,
        )

        self.param.watch(self._on_model_change, "model")

    def __panel__(self):
        return self._layout

    # ------------------------------------------------------------

    def _on_model_change(self, event):
        model = event.new
        self._tabs.clear()

        if model is None:
            # hide JSON editor if no model
            self._json_editor.value = {}
            self._json_container.visible = False
            return

        # ---------------- Config tab ----------------
        self._config_viewer.model = model
        self._tabs.append(("Summary", self._config_viewer))

        # ---------------- Model Type tab ----------------
        self._type_viewer.model_type = type(model)
        self._tabs.append(("Model Type", self._type_viewer))

        # ---------------- CallableModel extras ----------------
        if isinstance(model, ccflow.CallableModel):
            self._context_type_viewer.model_type = model.context_type
            self._tabs.append(("Context Type", self._context_type_viewer))

            self._result_type_viewer.model_type = model.result_type
            self._tabs.append(("Result Type", self._result_type_viewer))

        # Default to Config tab
        self._tabs.active = 0

        # Update & show JSONEditor
        self._json_editor.value = model.__pydantic_serializer__.to_python(model, fallback=str, mode="json")
        self._json_container.visible = True
