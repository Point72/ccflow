__version__ = "0.1.1"

from .arrow import *
from .base import *
from .callable import *
from .context import *
from .exttypes import *
from .generic_base import *
from .object_config import *
from .publisher import *
from .result import *
from .serialization import *
from .utils import *


def _initialize_s3_log_level():
    # See https://github.com/apache/arrow/issues/35575
    # https://github.com/apache/arrow/issues/36115
    # https://github.com/apache/arrow/pull/38267
    import os
    import pyarrow
    import pyarrow._s3fs

    if pyarrow.__version__ < "15":
        if os.environ.get("ARROW_S3_LOG_LEVEL"):
            level = getattr(
                pyarrow._s3fs.S3LogLevel,
                os.environ.get("ARROW_S3_LOG_LEVEL").capitalize(),
            )
            pyarrow._s3fs.initialize_s3(level)


_initialize_s3_log_level()
