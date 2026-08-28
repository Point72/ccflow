"""CLI for serving the ccflow ModelRegistry as a spaday application.

Mirrors :mod:`ccflow.ui.panel.cli` but renders the spaday viewer and serves it with Starlette + uvicorn
instead of Panel. ``serve_registry`` is the importable entry point; ``registry_viewer_cli`` is the
hydra-config-driven command wrapped by the ``ccflow-ui-spaday`` console script.
"""

import argparse
import asyncio
import logging
import os
from collections.abc import Callable

from spaday_dagre import package as dagre_package
from spaday_trees import package as trees_package
from spaday_webawesome import package as webawesome_package

from ccflow import ModelRegistry
from ccflow.utils.hydra import add_hydra_config_args, load_config, resolve_config_paths

from .model import MATERIALIZE_ENDPOINT
from .registry import CARD_ENDPOINT, DARK_FIELD, SELECTED_FIELD, model_card, registry_store, registry_viewer

__all__ = ("main", "registry_viewer_cli", "serve_registry")

log = logging.getLogger(__name__)

#: Component packages whose assets the page needs (webawesome controls, the tree, the dependency graph).
_PACKAGES = (webawesome_package, trees_package, dagre_package)

#: spaday-trees follows ``wa-dark``/``wa-light`` itself; the page only ever sets ``wa-dark``, so pin the
#: unset case to light at zero specificity, letting the package's own dark rule win when it applies.
_STYLES = (
    ":where(spaday-tree), :where(spaday-tree file-tree-container) { color-scheme: light; }",
    # Un-materialized models are drawn as outlines so they read as configuration, not instances.
    "spaday-dagre .spaday-dagre-node.pending :is(rect, ellipse, polygon) { stroke-dasharray: 4 3; }",
)


def serve_registry(
    registry: ModelRegistry,
    *,
    title: str = "ccflow Model Registry",
    browser_width: int = 400,
    sort_children: bool = True,
    host: str = "127.0.0.1",
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
        host, port: Interface and port uvicorn binds to (only used when ``run`` is True).
        run: When True, start a blocking uvicorn server. When False, return the app without serving.

    Returns:
        starlette.applications.Starlette: The mounted spaday application.
    """
    try:
        import uvicorn
        from spaday.backends.starlette import serve
        from spaday.bootstrap import tree_json
        from starlette.responses import JSONResponse, Response
        from starlette.routing import Route
    except ImportError:
        raise ImportError(
            "spaday, starlette and uvicorn must be installed to serve the spaday UI. Pip install ccflow[full] to install all optional dependencies."
        ) from None

    def page():
        return registry_viewer(registry, title=title, browser_width=browser_width, sort_children=sort_children)

    async def materialize(request):
        """Instantiate a pending (lazily-loaded) model.

        The client refreshes the tree afterwards, so the response only has to report the outcome: a
        model that cannot be constructed (a bad ``_target_``, an unavailable dependency) stays pending
        and its error is returned for the page to surface.
        """
        path = (await request.json()).get("path", "")
        if not path:
            return JSONResponse({"message": "No model selected."}, status_code=400)
        try:
            await asyncio.to_thread(registry.__getitem__, path)
        except Exception as error:
            log.exception("Failed to materialize lazy registry model %r", path)
            return JSONResponse({"message": f"Could not materialize {path}: {error}"}, status_code=500)
        return JSONResponse({"message": f"Materialized {path}."})

    def card(request):
        """Return one model's detail card, fetched by the page when that model is first shown."""
        path = request.query_params.get("model", "")
        component = model_card(registry, path, sort_children=sort_children)
        return Response(tree_json(component), media_type="application/json")

    app = serve(
        page,
        packages=_PACKAGES,
        styles=_STYLES,
        store=registry_store(),
        url={SELECTED_FIELD: "model"},
        persist={DARK_FIELD: "ccflow-ui-dark"},
        title=title,
        routes=[
            Route(MATERIALIZE_ENDPOINT, materialize, methods=["POST"]),
            Route(CARD_ENDPOINT, card),
        ],
    )

    if run:
        uvicorn.run(app, host=host, port=port)
    return app


def _get_ui_args_parser() -> argparse.ArgumentParser:
    """Create the argument parser for the spaday viewer server."""
    parser = argparse.ArgumentParser(
        add_help=True,
        description="Serve the ccflow ModelRegistry viewer as a spaday application",
    )

    add_hydra_config_args(parser)

    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host interface to bind the server to (default: 127.0.0.1).")
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
    hydra_main: Callable | None = None,
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
        host=args.host,
        port=args.port,
    )


def main():
    """Console-script entry point (``ccflow-ui-spaday``)."""
    registry_viewer_cli()
