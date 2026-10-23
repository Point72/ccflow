__version__ = "0.9.4"

# Import exttypes early so modules that import `from ccflow import PyObjectPath` during
# initialization find it (avoids circular import issues with functions that import utilities
# which, in turn, import `ccflow`).
from .exttypes import *  # noqa: I001

from .base import *
from .compose import *
from .callable import *
from .context import *
from .enums import Enum
from .flow_model import *
from .global_state import *
from .local_persistence import *
from .models import *
from .object_config import *
from .publisher import *
from .result import *
from .utils import FileHandler, StreamHandler

# Arrow models and types import pyarrow, so load them on first access.
_LAZY_ARROW = {
    "ArrowDateFilter": "arrow",
    "ArrowDatetimeFilter": "arrow",
    "ArrowFileSystem": "arrow",
    "ArrowFilter": "arrow",
    "ArrowLocalFileSystem": "arrow",
    "ArrowPartitioning": "arrow",
    "ArrowS3FileSystem": "arrow",
    "ArrowSchemaModel": "arrow",
    "ArrowTemplateFilter": "arrow",
    "ArrowTimeFilter": "arrow",
    "render_filters": "arrow",
    "ArrowSchema": "exttypes.arrow",
    "ArrowTable": "exttypes.arrow",
    "PyArrowDatatype": "exttypes.arrow",
}


def __getattr__(name):
    if name in _LAZY_ARROW:
        from importlib import import_module

        return getattr(import_module(f".{_LAZY_ARROW[name]}", __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
