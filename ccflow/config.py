"""Expose the selected configuration framework through one API.

Lerna is preferred when available unless ``CCFLOW_CONFIG_FRAMEWORK=hydra``.
Applications that select Hydra must install ``hydra-core``.
"""

import importlib
import os
from types import ModuleType

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

_ENV_VAR = "CCFLOW_CONFIG_FRAMEWORK"
_requested = os.environ.get(_ENV_VAR, "").strip().lower()

if _requested not in ("", "hydra", "lerna"):
    raise ValueError(f"{_ENV_VAR} must be 'hydra', 'lerna', or unset, got {_requested!r}")


def _import_framework(name: str) -> ModuleType:
    package = "hydra-core" if name == "hydra" else name
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as error:
        if error.name != name:
            raise
        raise ImportError(f"{_ENV_VAR}={name} requires the '{package}' package") from error


if _requested == "hydra":
    _framework = _import_framework("hydra")
elif _requested == "lerna":
    _framework = _import_framework("lerna")
else:
    try:
        _framework = importlib.import_module("lerna")
    except ModuleNotFoundError as error:
        if error.name != "lerna":
            raise
        _framework = _import_framework("hydra")

CONFIG_FRAMEWORK = _framework.__name__
main = _framework.main
compose = _framework.compose
initialize = _framework.initialize
initialize_config_dir = _framework.initialize_config_dir

if CONFIG_FRAMEWORK == "lerna":
    DefaultsList = importlib.import_module("lerna._internal.defaults_list").DefaultsList
    GlobalHydra = importlib.import_module("lerna.core.global_hydra").GlobalHydra
    HydraConfig = importlib.import_module("lerna.core.hydra_config").HydraConfig
    InstantiationException = importlib.import_module("lerna.errors").InstantiationException
    ObjectType = importlib.import_module("lerna.core.object_type").ObjectType
    RunMode = importlib.import_module("lerna.types").RunMode
    instantiate = importlib.import_module("lerna.utils").instantiate
else:
    DefaultsList = importlib.import_module("hydra._internal.defaults_list").DefaultsList
    GlobalHydra = importlib.import_module("hydra.core.global_hydra").GlobalHydra
    HydraConfig = importlib.import_module("hydra.core.hydra_config").HydraConfig
    InstantiationException = importlib.import_module("hydra.errors").InstantiationException
    ObjectType = importlib.import_module("hydra.core.object_type").ObjectType
    RunMode = importlib.import_module("hydra.types").RunMode
    instantiate = importlib.import_module("hydra.utils").instantiate
