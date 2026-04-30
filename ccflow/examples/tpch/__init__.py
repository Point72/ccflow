"""TPC-H example for ccflow.

This package is a *teaching* example showing how to compose a workflow from
``CallableModel``s wired together through the ``ModelRegistry``. The
canonical usage is::

    from ccflow import ModelRegistry
    from ccflow.examples.tpch import load_config

    load_config()                     # populate the root ModelRegistry from conf.yaml
    registry = ModelRegistry.root()
    result = registry["/query/Q1"]()  # run TPC-H query 1
    print(result.df.to_native())

To run the same example at a different TPC-H scale factor, override the
single shared backend on load (every table / answer / query references it,
so the change flows through everywhere)::

    load_config(overrides=["tpch.backend.scale_factor=1.0"])
"""

from pathlib import Path
from typing import List, Optional

from ccflow import RootModelRegistry, load_config as _load_config_base

from .data_generators import TPCHAnswerProvider, TPCHDuckDBBackend, TPCHTable, TPCHTableProvider
from .query import TPCHQuery

__all__ = (
    "TPCHTable",
    "TPCHDuckDBBackend",
    "TPCHTableProvider",
    "TPCHAnswerProvider",
    "TPCHQuery",
    "load_config",
)


def load_config(
    config_dir: str = "",
    config_name: str = "",
    overrides: Optional[List[str]] = None,
    *,
    overwrite: bool = True,
    basepath: str = "",
) -> RootModelRegistry:
    """Load the TPC-H example registry into the root ``ModelRegistry``.

    Pass hydra-style ``overrides`` to reconfigure entries on load — most
    usefully ``["tpch.backend.scale_factor=1.0"]`` to run the example at a
    different TPC-H scale factor. Every table / answer / query references the
    single ``/tpch/backend`` entry, so this one override flows through to all
    22+8 providers.

    Args:
        config_dir: Optional extra hydra config directory to overlay on top
            of the bundled ``config/conf.yaml``. Empty string (the default)
            means "use only the bundled config".
        config_name: Optional config name within ``config_dir`` to load.
        overrides: Hydra override strings, e.g.
            ``["tpch.backend.scale_factor=1.0"]``.
        overwrite: When True (the default), entries already present in the
            registry are replaced. This is what you want in notebooks where
            you re-call ``load_config()`` after tweaking overrides; set to
            False to require a fresh registry.
        basepath: Base path for resolving a relative ``config_dir``.
    """
    return _load_config_base(
        root_config_dir=str(Path(__file__).resolve().parent / "config"),
        root_config_name="conf",
        config_dir=config_dir,
        config_name=config_name,
        overrides=overrides,
        overwrite=overwrite,
        basepath=basepath,
    )
