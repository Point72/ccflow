from .exprtk import *
from .frequency import *
from .jinja import *

# Do NOT import .polars. We don't want ccflow (without flow) to have a dependency on polars!
from .pydantic_numpy import *
from .pyobjectpath import *

_LAZY_ARROW = ("ArrowSchema", "ArrowTable", "PyArrowDatatype")


def __getattr__(name):
    # Defer pyarrow until an arrow type is used.
    if name in _LAZY_ARROW:
        from . import arrow

        return getattr(arrow, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
