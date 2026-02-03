"""CLI for serving ModelRegistryViewer as a Panel application."""

import argparse
import inspect
from pathlib import Path
from typing import Callable, Optional

import panel as pn

from ccflow import ModelRegistry
from ccflow.utils.hydra import load_config

from .registry import ModelRegistryViewer

__all__ = ("registry_viewer_cli",)


def _get_ui_args_parser() -> argparse.ArgumentParser:
    """Create argument parser with UI server configuration options."""
    parser = argparse.ArgumentParser(
        add_help=True,
        description="Serve ModelRegistryViewer as a Panel application",
    )

    # Registry loading arguments (similar to utils.hydra)
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Key=value arguments to override config values",
    )
    parser.add_argument(
        "--config-path",
        "-cp",
        help="Path to the Hydra config directory",
    )
    parser.add_argument(
        "--config-name",
        "-cn",
        help="Name of the config file (without .yaml extension)",
    )
    parser.add_argument(
        "--config-dir",
        "-cd",
        help="Additional config directory to add to search path",
    )
    parser.add_argument(
        "--config-dir-config-name",
        "-cdcn",
        help="Config name to look for within config-dir",
    )
    parser.add_argument(
        "--basepath",
        help="Base path for searching config directories",
    )

    # UI server arguments
    parser.add_argument(
        "--address",
        type=str,
        default="127.0.0.1",
        help="Address to bind the server to (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port to bind the server to (default: 8080)",
    )
    parser.add_argument(
        "--allow-websocket-origin",
        type=str,
        nargs="+",
        default=["*"],
        help="Allowed websocket origins (default: *)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Open browser automatically",
    )

    # Viewer layout arguments
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

    # Resolve config path (same logic as cfg_explain_cli)
    if args.config_path:
        root_config_dir = args.config_path
    elif hydra_main and config_path:
        root_config_dir = str(Path(inspect.getfile(hydra_main.__wrapped__)).parent / config_path)
    else:
        raise ValueError("Must provide --config-path.")

    # Resolve config name
    if args.config_name:
        root_config_name = args.config_name
    elif config_name:
        root_config_name = config_name
    else:
        raise ValueError("Must provide --config-name.")

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
