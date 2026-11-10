"""Expose the selected configuration backend through one API.

Hydra is preferred when available, otherwise Lerna is used. Set ``CCFLOW_CONFIG_BACKEND=lerna`` to require Lerna instead.
The backend is loaded on first API access, so importing ccflow does not require either package.
"""

import importlib
import os
from functools import cache
from types import ModuleType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hydra import compose, initialize, initialize_config_dir, main
    from hydra.core.hydra_config import HydraConfig
    from hydra.errors import InstantiationException
    from hydra.utils import instantiate

    CONFIG_FRAMEWORK: str

__all__ = (
    "CONFIG_FRAMEWORK",
    "HydraConfig",
    "InstantiationException",
    "compose",
    "initialize",
    "initialize_config_dir",
    "instantiate",
    "main",
)

_ENV_VAR = "CCFLOW_CONFIG_BACKEND"
_SUBMODULES = {
    "DefaultsList": "_internal.defaults_list",
    "GlobalHydra": "core.global_hydra",
    "HydraConfig": "core.hydra_config",
    "InstantiationException": "errors",
    "ObjectType": "core.object_type",
    "RunMode": "types",
    "instantiate": "utils",
}


@cache
def _get_framework() -> ModuleType:
    requested = os.environ.get(_ENV_VAR, "").strip().lower()
    frameworks = ("lerna",) if requested == "lerna" else ("hydra", "lerna")
    for name in frameworks:
        try:
            return importlib.import_module(name)
        except ModuleNotFoundError as error:
            if error.name != name:
                raise
    if requested == "lerna":
        raise ImportError(f"{_ENV_VAR}=lerna requires the 'lerna' package")
    raise ImportError("Configuration requires 'hydra-core' or 'lerna'; install one to use ccflow.config")


def __getattr__(name: str):
    if name == "CONFIG_FRAMEWORK":
        return _get_framework().__name__
    if name in ("main", "compose", "initialize", "initialize_config_dir"):
        return getattr(_get_framework(), name)
    if name in _SUBMODULES:
        module = importlib.import_module(f"{_get_framework().__name__}.{_SUBMODULES[name]}")
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
