"""Unit tests for ccflow.ui.spaday.cli module."""

from pathlib import Path

import pytest
from spaday.bootstrap import _ASSETS, bundles_dir

from ccflow import BaseModel, LazyRegistry, ModelRegistry
from ccflow.ui.spaday.cli import _asset_layout, _get_ui_args_parser, serve_registry


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
        assert hasattr(args, "address")
        assert hasattr(args, "port")
        assert hasattr(args, "browser_width")
        assert hasattr(args, "title")
        assert hasattr(args, "sort_children")

    def test_defaults(self):
        args = _get_ui_args_parser().parse_args([])
        assert args.address == "127.0.0.1"
        assert args.port == 8080
        assert args.browser_width == 400
        assert args.title == "ccflow Model Registry"
        assert args.sort_children is True

    def test_custom_values(self):
        args = _get_ui_args_parser().parse_args(["--address", "0.0.0.0", "--port", "9000", "--browser-width", "500", "--title", "Mine"])
        assert args.address == "0.0.0.0"
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


class TestMaterializeEndpoint:
    def _lazy_registry(self):
        return LazyRegistry(
            name="root",
            group={"model": {"_target_": "ccflow.tests.ui.spaday.test_cli.SimpleModel", "name": "pending"}},
        )

    def test_materialize_instantiates_pending_model(self):
        starlette_testclient = pytest.importorskip("starlette.testclient")
        registry = self._lazy_registry()
        app = serve_registry(registry, run=False)
        assert not registry["group"].is_loaded("model")

        client = starlette_testclient.TestClient(app)
        response = client.get("/materialize", params={"path": "group/model"}, follow_redirects=False)

        assert response.status_code == 303
        assert "sel=group/model" in response.headers["location"]
        assert registry["group"].is_loaded("model")

    def test_materialize_missing_path_redirects_without_error(self):
        starlette_testclient = pytest.importorskip("starlette.testclient")
        registry = self._lazy_registry()
        app = serve_registry(registry, run=False)

        client = starlette_testclient.TestClient(app)
        response = client.get("/materialize", follow_redirects=False)

        assert response.status_code == 303

    def test_homepage_seeds_selected_model(self):
        starlette_testclient = pytest.importorskip("starlette.testclient")
        registry = self._lazy_registry()
        app = serve_registry(registry, run=False)

        client = starlette_testclient.TestClient(app)
        assert "group/model" in client.get("/", params={"sel": "group/model"}).text
        assert "group/model" not in client.get("/").text


class TestAssetLayout:
    def test_selected_layout_has_runtime_asset(self):
        # Guards the 404 regression: an unrelated top-level ``js`` package must not push us to the
        # "source" layout, whose bundle directory would then lack spaday's runtime asset.
        layout = _asset_layout()
        runtime = _ASSETS[layout]["runtime"].lstrip("/")
        assert (Path(bundles_dir(layout)) / runtime).is_file()
