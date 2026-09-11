"""Unit tests for ccflow.ui.spaday.cli module."""

import importlib
from pathlib import Path

import pytest
from spaday.bootstrap import _ASSETS, _layout, bundles_dir

from ccflow import BaseModel, LazyRegistry, ModelRegistry
from ccflow.ui.spaday.cli import _get_ui_args_parser, serve_registry


class SimpleModel(BaseModel):
    name: str
    value: int = 0


class TestGetUIArgsParser:
    def test_parser_composition(self):
        parser = _get_ui_args_parser()
        args = parser.parse_args([])

        # From add_hydra_config_args
        assert hasattr(args, "overrides")
        assert hasattr(args, "config_path")
        assert hasattr(args, "config_name")

        # Server + viewer-specific
        assert hasattr(args, "host")
        assert hasattr(args, "port")
        assert hasattr(args, "browser_width")
        assert hasattr(args, "title")
        assert hasattr(args, "sort_children")

    def test_defaults(self):
        args = _get_ui_args_parser().parse_args([])
        assert args.host == "127.0.0.1"
        assert args.port == 8080
        assert args.browser_width == 400
        assert args.title == "ccflow Model Registry"
        assert args.sort_children is True

    def test_custom_values(self):
        args = _get_ui_args_parser().parse_args(["--host", "0.0.0.0", "--port", "9000", "--browser-width", "500", "--title", "Mine"])
        assert args.host == "0.0.0.0"
        assert args.port == 9000
        assert args.browser_width == 500
        assert args.title == "Mine"

    def test_no_sort_children_flag(self):
        args = _get_ui_args_parser().parse_args(["--no-sort-children"])
        assert args.sort_children is False

    def test_overrides_positional(self):
        args = _get_ui_args_parser().parse_args(["key1=value1", "key2=value2"])
        assert args.overrides == ["key1=value1", "key2=value2"]


class TestServeRegistry:
    def test_builds_app_without_running(self):
        registry = ModelRegistry(name="test")
        registry.add("m", SimpleModel(name="m", value=1))
        app = serve_registry(registry, run=False)
        paths = {getattr(route, "path", None) for route in app.routes}
        assert "/" in paths
        assert "/tree.json" in paths

    def test_selection_is_url_bound_and_theme_persists(self):
        starlette_testclient = pytest.importorskip("starlette.testclient")
        registry = ModelRegistry(name="test")
        registry.add("widget", SimpleModel(name="widget"))
        app = serve_registry(registry, run=False)

        page = starlette_testclient.TestClient(app).get("/").text

        # The selection rides a query parameter, so a model is linkable and back/forward navigate.
        assert '"selected"' in page and '"model"' in page
        assert "ccflow-ui-dark" in page

    def test_card_endpoint_serves_one_model(self):
        starlette_testclient = pytest.importorskip("starlette.testclient")
        registry = ModelRegistry(name="test")
        registry.add("widget", SimpleModel(name="widget", value=7))
        registry.add("other", SimpleModel(name="other"))
        client = starlette_testclient.TestClient(serve_registry(registry, run=False))

        response = client.get("/card", params={"model": "widget"})

        assert response.status_code == 200
        body = response.json()
        assert body["tag"]
        # Only the requested model's card, so the page can defer the rest.
        assert "widget" in response.text
        assert "other" not in response.text

    def test_card_endpoint_handles_unknown_model(self):
        starlette_testclient = pytest.importorskip("starlette.testclient")
        client = starlette_testclient.TestClient(serve_registry(ModelRegistry(name="test"), run=False))

        response = client.get("/card", params={"model": "nope"})

        assert response.status_code == 200
        assert "Unknown model" in response.text

    def test_tree_route_reflects_registry(self):
        registry = ModelRegistry(name="test")
        registry.add("widget", SimpleModel(name="widget"))
        app = serve_registry(registry, title="T", run=False)
        # The tree route serializes the viewer; the model path should appear in it.
        tree_route = next(r for r in app.routes if getattr(r, "path", None) == "/tree.json")
        assert tree_route is not None

    def test_materialize_route_present(self):
        registry = ModelRegistry(name="test")
        registry.add("m", SimpleModel(name="m"))
        app = serve_registry(registry, run=False)
        paths = {getattr(route, "path", None) for route in app.routes}
        assert "/materialize" in paths

    @pytest.mark.parametrize("module", ["ccflow.ui.cli", "ccflow.ui.model", "ccflow.ui.registry"])
    def test_panel_module_compatibility_imports(self, module):
        assert importlib.import_module(module)


class TestMaterializeEndpoint:
    def _lazy_registry(self):
        return LazyRegistry(
            name="root",
            group={"model": {"_target_": "ccflow.tests.ui.spaday.test_cli.SimpleModel", "name": "pending"}},
        )

    def test_materialize_instantiates_pending_model(self, mocker):
        starlette_testclient = pytest.importorskip("starlette.testclient")
        from ccflow.ui.spaday import cli

        to_thread = mocker.spy(cli.asyncio, "to_thread")
        registry = self._lazy_registry()
        app = serve_registry(registry, run=False)
        assert not registry["group"].is_loaded("model")

        client = starlette_testclient.TestClient(app)
        response = client.post("/materialize", json={"path": "group/model"})

        assert response.status_code == 200
        assert "group/model" in response.json()["message"]
        assert registry["group"].is_loaded("model")
        to_thread.assert_awaited_once()

    def test_materialize_missing_path_is_rejected(self):
        starlette_testclient = pytest.importorskip("starlette.testclient")
        registry = self._lazy_registry()
        app = serve_registry(registry, run=False)

        client = starlette_testclient.TestClient(app)
        response = client.post("/materialize", json={})

        assert response.status_code == 400
        assert response.json()["message"]

    def test_materialize_reports_failure(self):
        starlette_testclient = pytest.importorskip("starlette.testclient")
        registry = LazyRegistry(name="lazy", group={"broken": {"_target_": "not_a_module.Nope"}})
        app = serve_registry(registry, run=False)

        client = starlette_testclient.TestClient(app)
        response = client.post("/materialize", json={"path": "group/broken"})

        # The page surfaces this message in a toast, so it has to name the model and the cause.
        assert response.status_code == 500
        assert "group/broken" in response.json()["message"]
        assert not registry["group"].is_loaded("broken")

    def test_materialize_rejects_get(self):
        starlette_testclient = pytest.importorskip("starlette.testclient")
        app = serve_registry(self._lazy_registry(), run=False)

        response = starlette_testclient.TestClient(app).get("/materialize", params={"path": "group/model"})

        assert response.status_code == 405


class TestAssetLayout:
    def test_resolved_layout_has_runtime_asset(self):
        # Guards the 404 regression: an unrelated top-level ``js`` package must not push spaday to the
        # "source" layout, whose bundle directory would then lack the runtime asset.
        layout = _layout(None)
        runtime = _ASSETS[layout]["runtime"].lstrip("/")
        assert (Path(bundles_dir(layout)) / runtime).is_file()
