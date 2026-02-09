"""CLI for serving ModelRegistryViewer as a Panel application."""

import argparse
from typing import Callable, Optional

import panel as pn

from ccflow import ModelRegistry
from ccflow.utils.hydra import add_hydra_config_args, add_panel_server_args, load_config, resolve_config_paths

from .registry import ModelRegistryViewer

__all__ = ("registry_viewer_cli",)


def _get_ui_args_parser() -> argparse.ArgumentParser:
    """Create argument parser with UI server configuration options."""
    parser = argparse.ArgumentParser(
        add_help=True,
        description="Serve ModelRegistryViewer as a Panel application",
    )

    # Standard hydra config loading arguments
    add_hydra_config_args(parser)

    # Standard Panel server arguments
    add_panel_server_args(parser)

    # Viewer-specific arguments
    parser.add_argument(
        "--browser-width",
        type=int,
        default=400,
        help="Width of the registry browser panel (default: 400)",
    )
    parser.add_argument(
        "--browser-height",
        type=int,
        default=700,
        help="Height of the registry browser panel (default: 700)",
    )
    parser.add_argument(
        "--viewer-width",
        type=int,
        default=None,
        help="Fixed width for model viewer panel (default: stretch)",
    )

    return parser


def registry_viewer_cli(
    config_path: str = "",
    config_name: str = "",
    hydra_main: Optional[Callable] = None,
):
    """CLI entry point for serving ModelRegistryViewer.

    Parameters
    ----------
    config_path
        The config_path specified in hydra.main()
    config_name
        The config_name specified in hydra.main()
    hydra_main
        The function decorated with hydra.main(). Used to resolve config_path
        relative to the decorated function's file location.
    """
    parser = _get_ui_args_parser()
    args = parser.parse_args()

    # Resolve config paths using shared helper
    root_config_dir, root_config_name = resolve_config_paths(args, config_path, config_name, hydra_main)

    # Load config using hydra utilities
    result = load_config(
        root_config_dir=root_config_dir,
        root_config_name=root_config_name,
        config_dir=args.config_dir,
        config_name=args.config_dir_config_name,
        overrides=args.overrides,
        basepath=args.basepath,
    )

    # Load registry from config
    registry = ModelRegistry.root()
    registry.load_config(cfg=result.cfg, overwrite=True)

    # Create app factory for per-session instances
    def create_app():
        viewer = ModelRegistryViewer(
            registry,
            browser_width=args.browser_width,
            browser_height=args.browser_height,
            viewer_width=args.viewer_width,
        )
        return viewer.__panel__()

    # Serve the panel app (callable = fresh instance per session)
    pn.serve(
        create_app,
        address=args.address,
        port=args.port,
        allow_websocket_origin=args.allow_websocket_origin,
        show=args.show,
    )
