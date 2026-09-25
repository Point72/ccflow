"""Unit tests for ccflow.ui.spaday.cli module."""

import importlib
import sys
from pathlib import Path

import pytest
from spaday.bootstrap import _ASSETS, _layout, bundles_dir

from ccflow import BaseModel, LazyRegistry, ModelRegistry
from ccflow.ui.spaday.cli import _get_ui_args_parser, registry_viewer_cli, serve_registry


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

    @pytest.mark.parametrize(
        ("argv", "expected"),
        [([], "--config-path"), (["--config-path", "cfg"], "--config-name")],
    )
    def test_missing_config_args_report_usage(self, monkeypatch, capsys, argv, expected):
        monkeypatch.setattr(sys, "argv", ["ccflow-ui-spaday", *argv])

        with pytest.raises(SystemExit) as excinfo:
            registry_viewer_cli()

        # argparse usage error, not a traceback out of resolve_config_paths.
        assert excinfo.value.code == 2
        assert expected in capsys.readouterr().err

    def test_config_dir_is_optional(self):
        args = _get_ui_args_parser().parse_args(["--config-path", "cfg", "--config-name", "conf"])
        assert args.config_dir is None
        assert args.config_dir_config_name is None

    def test_config_dir_and_name_load_without_config_path(self, monkeypatch):
        from ccflow.ui.spaday import cli

        example = Path(cli.__file__).parents[2] / "examples" / "calculator"
        monkeypatch.setattr(sys, "argv", ["ccflow-ui-spaday", "-cd", "config", "-cn", "base", "--basepath", str(example)])
        monkeypatch.chdir(example)
        served = {}
        monkeypatch.setattr(cli, "serve_registry", lambda registry, **kwargs: served.update(registry=registry))
        ModelRegistry.root().clear()

        cli.registry_viewer_cli()

        # The config dir stood in for the root config, so the registry actually got populated.
        assert served["registry"] is ModelRegistry.root()
        assert served["registry"].models

    def test_config_dir_without_config_name_reports_usage(self, monkeypatch, capsys):
        monkeypatch.setattr(sys, "argv", ["ccflow-ui-spaday", "-cd", "config"])

        with pytest.raises(SystemExit) as excinfo:
            registry_viewer_cli()

        assert excinfo.value.code == 2
        assert "--config-name" in capsys.readouterr().err


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
        starlette_testclient = pytest.importorskip("starlette.testclient")
        registry = ModelRegistry(name="test")
        registry.add("widget", SimpleModel(name="widget"))
        client = starlette_testclient.TestClient(serve_registry(registry, title="T", run=False))

        response = client.get("/tree.json")

        assert response.status_code == 200
        assert "widget" in response.text

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

    def test_materialize_rejects_path_escaping_served_registry(self):
        starlette_testclient = pytest.importorskip("starlette.testclient")
        # A leading slash resolves against the process-global root rather than the served registry.
        root = ModelRegistry.root()
        root.clear()
        served = self._lazy_registry()
        root.add("served", served)

        client = starlette_testclient.TestClient(serve_registry(served, run=False))
        response = client.post("/materialize", json={"path": "/served/group/model"})

        assert response.status_code == 404
        assert not served["group"].is_loaded("model")

    def test_materialize_requires_json_content_type(self):
        starlette_testclient = pytest.importorskip("starlette.testclient")
        registry = self._lazy_registry()
        client = starlette_testclient.TestClient(serve_registry(registry, run=False))

        # A form post is a CSRF-able "simple request", so it never reaches the registry.
        response = client.post("/materialize", data={"path": "group/model"})

        assert response.status_code == 415
        assert not registry["group"].is_loaded("model")

    @pytest.mark.parametrize("content_type", ["application/json", "Application/JSON", "application/json; charset=utf-8"])
    def test_materialize_accepts_content_type_case_insensitively(self, content_type):
        starlette_testclient = pytest.importorskip("starlette.testclient")
        registry = self._lazy_registry()
        client = starlette_testclient.TestClient(serve_registry(registry, run=False))

        response = client.post("/materialize", content='{"path": "group/model"}', headers={"content-type": content_type})

        assert response.status_code == 200
        assert registry["group"].is_loaded("model")

    @pytest.mark.parametrize("body", ["not json", "null", "[]"])
    def test_materialize_rejects_malformed_body(self, body):
        starlette_testclient = pytest.importorskip("starlette.testclient")
        client = starlette_testclient.TestClient(serve_registry(self._lazy_registry(), run=False))

        response = client.post("/materialize", content=body, headers={"content-type": "application/json"})

        assert response.status_code == 400

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
