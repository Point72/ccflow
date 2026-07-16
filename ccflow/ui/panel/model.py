import html

import panel as pn
import panel_material_ui  # noqa: F401  Must be imported like this to register the extension
import panel_material_ui as pmui
import param
from pydantic._internal._repr import display_as_type

import ccflow

pn.extension()
pn.extension("jsoneditor")


__all__ = ("ModelConfigViewer", "ModelTypeViewer", "ModelViewer")


_FIELD_STYLES = {
    "name": "color:#0550ae;",  # blue
    "type": "color:#8250df;",  # purple
    "description": "color:#57606a;font-style:italic;",  # muted gray
}


class ModelTypeViewer(param.Parameterized):
    """
    Displays type name, class docstring, and fields for a Pydantic model type.
    """

    model_type = param.Parameter(default=None)

    def __init__(self, **params):
        super().__init__(**params)

        self._pane = pn.pane.HTML("", sizing_mode="stretch_width")
        self._layout = pn.Column(
            self._pane,
            sizing_mode="stretch_width",
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
            name_html = f'<code style="{_FIELD_STYLES["name"]}">{html.escape(name)}</code>'
            type_html = f'<code style="{_FIELD_STYLES["type"]}">{html.escape(field_type)}</code>'
            desc_html = f' — <span style="{_FIELD_STYLES["description"]}">{html.escape(desc)}</span>' if desc else ""
            field_items.append(f'<li style="overflow-wrap:anywhere;">{name_html} ({type_html}){desc_html}</li>')

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
            <code style="{_FIELD_STYLES["type"]}overflow-wrap:anywhere;">{html.escape(type_name)}</code>
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

        self.model_path = ""
        self._metadata = pn.pane.HTML("", sizing_mode="stretch_width")

        self._layout = pn.Column(
            self._metadata,
            sizing_mode="stretch_width",
        )

        self.param.watch(self._on_model_change, "model")

    def __panel__(self):
        return self._layout

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

        items = "".join(f'<li><code style="overflow-wrap:anywhere;">{html.escape(row)}</code></li>' for row in rows)

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

        path_html = ""
        if self.model_path:
            path_html = f"""
            <div style="margin-bottom:6px;">
              <div style="font-weight:600;">Registry Path</div>
              <code>{html.escape(self.model_path)}</code>
            </div>
            """

        description = model.meta.description.strip() if hasattr(model, "meta") and model.meta.description else ""

        desc_html = ""
        if description:
            try:
                import bleach

                description = bleach.linkify(html.escape(description))
            except ImportError:
                description = html.escape(description)
            desc_html = f"""
            <div style="margin-bottom:6px;">
              <div style="font-weight:600;">Instance Description</div>
              <div>{description}</div>
            </div>
            """

        self._metadata.object = path_html + desc_html + self._render_dependencies(model)


class ModelViewer(param.Parameterized):
    """
    Displays a tabbed view of a ccflow Model instance, including description, registry dependencies, docstrings and json representation.
    """

    model = param.Parameter(default=None)

    def __init__(self, **params):
        super().__init__(**params)

        self.model_path = ""

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
            sizing_mode="stretch_width",
            min_width=400,
        )

        self._json_container = pn.Column(
            "## Parameters",
            self._json_editor,
            visible=False,  # hidden initially
            sizing_mode="stretch_width",
        )

        self._layout = pn.Column(
            "## Model Viewer",
            self._tabs,
            pn.Spacer(height=12),
            self._json_container,
            sizing_mode="stretch_width",
        )

        self.param.watch(self._on_model_change, "model")

    def __panel__(self):
        return self._layout

    def _on_model_change(self, event):
        model = event.new
        self._tabs.clear()

        if model is None:
            # hide JSON editor if no model
            self._json_editor.value = {}
            self._json_container.visible = False
            return

        # Config tab
        self._config_viewer.model_path = self.model_path
        self._config_viewer.model = model
        self._tabs.append(("Summary", self._config_viewer))

        # Model Type tab
        self._type_viewer.model_type = type(model)
        self._tabs.append(("Model Type", self._type_viewer))

        # CallableModel extras
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
