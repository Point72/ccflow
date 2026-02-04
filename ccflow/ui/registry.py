import panel as pn
import panel_material_ui  # noqa: F401  Must be imported like this to register the extension
import panel_material_ui as pmui
import param

from .model import ModelViewer

pn.extension()

__all__ = ("RegistryBrowser", "ModelRegistryViewer")


class RegistryBrowser(param.Parameterized):
    selected_model = param.Parameter(default=None)

    def __init__(self, registry, **params):
        super().__init__(**params)
        self._registry = registry

        self._tree_items = self._build_tree(registry)
        self._node_index = self._build_node_index(self._tree_items)

        self._tree = pmui.Tree(
            items=self._tree_items,
            multi_select=False,
        )

        self._search = pn.widgets.AutocompleteInput(
            name="Search",
            options=sorted(self._node_index.keys()),
            placeholder="Search full path…",
            case_sensitive=False,
            search_strategy="includes",
            min_characters=1,
            sizing_mode="stretch_width",
        )

        self._search.param.watch(self._on_search_select, "value")
        self._tree.param.watch(self._on_tree_select, "value")

        self._layout = pn.Column(
            "## Registry",
            self._search,
            self._tree,
        )

    def __panel__(self):
        return self._layout

    # ---------------- Tree construction ----------------

    def _build_tree(self, registry, index_prefix=()):
        import ccflow

        items = []
        for i, (name, model) in enumerate(registry.models.items()):
            index_path = index_prefix + (i,)
            entry = {
                "label": name,
                "_index_path": index_path,
            }

            if isinstance(model, ccflow.ModelRegistry):
                entry["items"] = self._build_tree(model, index_prefix=index_path)
            else:
                entry["model"] = model

            items.append(entry)

        return items

    def _build_node_index(self, tree_items):
        index = {}

        def walk(items, prefix=""):
            for node in items:
                path = f"{prefix}/{node['label']}" if prefix else node["label"]
                if "model" in node:
                    index[path] = node
                walk(node.get("items", []), path)

        walk(tree_items)
        return index

    @staticmethod
    def _expanded_from_index_path(index_path):
        return [index_path[:i] for i in range(1, len(index_path))]

    # ---------------- Callbacks ----------------

    def _on_search_select(self, event):
        path = event.new
        if not path:
            return
        node = self._node_index.get(path)
        if not node:
            return
        self._tree.expanded = self._expanded_from_index_path(node["_index_path"])
        self._tree.value = [node]
        self._search.value = ""

    def _on_tree_select(self, event):
        self.selected_model = event.new[0].get("model") if event.new else None


class ModelRegistryViewer(param.Parameterized):
    """
    Top-level viewer that composes the RegistryBrowser and ModelViewer
    into a scrollable two-panel layout.
    """

    # ---------------- Layout parameters ----------------
    browser_width = param.Integer(
        default=400,
        bounds=(200, None),
        doc="Width of the registry browser panel (px)",
    )

    browser_height = param.Integer(
        default=700,
        bounds=(300, None),
        doc="Height of the registry browser panel (px)",
    )

    viewer_width = param.Integer(
        default=None,
        allow_None=True,
        doc="Optional fixed width for the model viewer panel (px)",
    )

    model = param.Parameter(
        default=None,
        doc="The currently selected model from the registry browser",
    )

    def __init__(self, registry, **params):
        super().__init__(**params)

        # Core components
        self._browser = RegistryBrowser(registry)
        self._viewer = ModelViewer()

        # Wire browser → viewer and model param
        def _on_selection(e):
            self.model = e.new
            self._viewer.model = e.new

        self._browser.param.watch(_on_selection, "selected_model")

        # Build layout
        self._layout = pn.Row(
            self._make_browser_column(),
            self._make_viewer_column(),
        )

    def __panel__(self):
        return self._layout

    # ---------------- Internal helpers ----------------

    def _make_browser_column(self):
        return pn.Column(
            self._browser,
            width=self.browser_width,
            height=self.browser_height,
            scroll=True,  # ✅ only left panel scrolls
        )

    def _make_viewer_column(self):
        if self.viewer_width is not None:
            return pn.Column(
                self._viewer,
                width=self.viewer_width,
            )
        else:
            return pn.Column(
                self._viewer,
                sizing_mode="stretch_width",
            )
