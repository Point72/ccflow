"""CLI for serving the ccflow ModelRegistry as a spaday application.

Mirrors :mod:`ccflow.ui.panel.cli` but renders the spaday viewer and serves it with Starlette + uvicorn
instead of Panel. ``serve_registry`` is the importable entry point; ``registry_viewer_cli`` is the
hydra-config-driven command wrapped by the ``ccflow-ui-spaday`` console script.
"""

import argparse
import logging
import os
from pathlib import Path
from typing import Callable, Optional
from urllib.parse import quote

from ccflow import ModelRegistry
from ccflow.utils.hydra import add_hydra_config_args, load_config, resolve_config_paths

from .model import MATERIALIZE_ENDPOINT
from .registry import SELECTED_FIELD, registry_store, registry_viewer

__all__ = ("serve_registry", "registry_viewer_cli", "main")

log = logging.getLogger(__name__)


def _asset_layout() -> str:
    """Select spaday's asset layout ("source" vs "installed").

    spaday auto-detects this from whether ``<spaday>/../js`` is a directory, but an unrelated
    top-level ``js`` package on ``sys.path`` (common in site-packages) makes it wrongly choose the
    "source" layout, whose bundle URLs then 404. Only a real spaday source checkout ships ``js/dist``,
    so require that before trusting the source layout; otherwise use the packaged extension assets.
    """
    import spaday

    source_js = Path(spaday.__file__).resolve().parent.parent / "js"
    return "source" if (source_js / "dist").is_dir() else "installed"


def serve_registry(
    registry: ModelRegistry,
    *,
    title: str = "ccflow Model Registry",
    browser_width: int = 400,
    sort_children: bool = True,
    address: str = "127.0.0.1",
    port: int = 8080,
    run: bool = True,
):
    """Build the spaday registry viewer and serve it as a Starlette app.

    Args:
        registry: The registry to browse. The page tree is rebuilt per request, so it reflects the
            registry's current contents.
        title: Title shown in the page header.
        browser_width: Initial width of the registry sidebar, in pixels.
        sort_children: Sort registry entries alphabetically at every level (subregistries first).
        address, port: Interface and port uvicorn binds to (only used when ``run`` is True).
        run: When True, start a blocking uvicorn server. When False, return the app without serving.

    Returns:
        starlette.applications.Starlette: The mounted spaday application.
    """
    try:
        import uvicorn
        from spaday.backends.starlette import serve
        from spaday.bootstrap import bootstrap
        from starlette.responses import HTMLResponse, RedirectResponse
        from starlette.routing import Route
    except ImportError:
        raise ImportError(
            "spaday, starlette and uvicorn must be installed to serve the spaday UI. Pip install ccflow[full] to install all optional dependencies."
        ) from None

    layout = _asset_layout()

    def page():
        return registry_viewer(registry, title=title, browser_width=browser_width, sort_children=sort_children)

    async def materialize(request):
        """Instantiate a pending (lazily-loaded) model, then redirect back with it selected.

        Materialization is best-effort: if the model cannot be constructed (e.g. it needs live data
        or an unavailable dependency) the failure is logged and the page still reloads, leaving the
        entry pending so it can be retried.
        """
        path = request.query_params.get("path", "")
        if path:
            try:
                registry[path]
            except Exception:
                log.exception("Failed to materialize lazy registry model %r", path)
        return RedirectResponse(url=f"/?sel={quote(path)}", status_code=303)

    def homepage(request):
        """Serve the page with the ``?sel=`` model preselected (used by the materialize redirect)."""
        selected = request.query_params.get("sel", "")
        return HTMLResponse(bootstrap(bundles=["webawesome"], store={SELECTED_FIELD: selected}, title=title, layout=layout))

    app = serve(
        page,
        bundles=["webawesome"],
        store=registry_store(),
        title=title,
        layout=layout,
        routes=[Route(MATERIALIZE_ENDPOINT, materialize, methods=["GET"])],
    )
    # Prepend a homepage that seeds the selection from ?sel= so the freshly materialized model's detail
    # card is shown immediately after the materialize redirect (Starlette matches routes in order).
    app.routes.insert(0, Route("/", homepage, methods=["GET"]))

    if run:
        uvicorn.run(app, host=address, port=port)
    return app


def _get_ui_args_parser() -> argparse.ArgumentParser:
    """Create the argument parser for the spaday viewer server."""
    parser = argparse.ArgumentParser(
        add_help=True,
        description="Serve the ccflow ModelRegistry viewer as a spaday application",
    )

    add_hydra_config_args(parser)

    parser.add_argument("--address", type=str, default="127.0.0.1", help="Address to bind the server to (default: 127.0.0.1).")
    parser.add_argument("--port", type=int, default=8080, help="Port to bind the server to (default: 8080).")
    parser.add_argument(
        "--browser-width",
        type=int,
        default=400,
        help="Initial width of the registry browser sidebar in px (default: 400).",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="ccflow Model Registry",
        help="Title shown in the page header (default: 'ccflow Model Registry').",
    )
    parser.add_argument(
        "--no-sort-children",
        dest="sort_children",
        action="store_false",
        help="Keep registry entries in insertion order instead of sorting them alphabetically.",
    )

    return parser


def registry_viewer_cli(
    config_path: str = "",
    config_name: str = "",
    hydra_main: Optional[Callable] = None,
):
    """CLI entry point for serving the spaday ModelRegistry viewer.

    Args:
        config_path: The config_path specified in hydra.main().
        config_name: The config_name specified in hydra.main().
        hydra_main: The function decorated with hydra.main(). Used to resolve config_path relative to
            the decorated function's file location.
    """
    parser = _get_ui_args_parser()
    args = parser.parse_args()

    root_config_dir, root_config_name = resolve_config_paths(args, config_path, config_name, hydra_main)
    # hydra's initialize_config_dir requires an absolute directory; resolve a relative --config-path
    # against the current working directory.
    root_config_dir = os.path.abspath(root_config_dir)

    result = load_config(
        root_config_dir=root_config_dir,
        root_config_name=root_config_name,
        config_dir=args.config_dir,
        config_name=args.config_dir_config_name,
        overrides=args.overrides,
        basepath=args.basepath,
    )

    registry = ModelRegistry.root()
    registry.load_config(cfg=result.cfg, overwrite=True)

    serve_registry(
        registry,
        title=args.title,
        browser_width=args.browser_width,
        sort_children=args.sort_children,
        address=args.address,
        port=args.port,
    )


def main():
    """Console-script entry point (``ccflow-ui-spaday``)."""
    registry_viewer_cli()
