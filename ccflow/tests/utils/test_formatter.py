import logging
from io import StringIO
from unittest import TestCase

import narwhals as nw
import pyarrow as pa

from ccflow.result.narwhals import NarwhalsDataFrameResult
from ccflow.result.pyarrow import ArrowResult
from ccflow.utils.formatter import PolarsTableFormatter


class TestPolarsTableFormatter(TestCase):
    """Test the PolarsTableFormatter."""

    def setUp(self):
        # Set up a logger with a StringIO stream to capture output
        self.logger = logging.getLogger("test_logger")
        self.logger.setLevel(logging.DEBUG)

        self.log_stream = StringIO()
        handler = logging.StreamHandler(self.log_stream)

        # Use the custom formatter
        formatter = PolarsTableFormatter("%(name)s - %(levelname)s - %(message)s", polars_config={"tbl_rows": 5})
        handler.setFormatter(formatter)

        self.logger.addHandler(handler)

    def tearDown(self):
        # Remove handlers after each test
        self.logger.handlers.clear()

    def test_arrow(self):
        table = pa.table({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        self.logger.debug("Result:", extra={"result": ArrowResult(table=table)})
        log = self.log_stream.getvalue().strip()
        assert log.startswith("test_logger - DEBUG - Result:\n{'_target_': 'ccflow.result.pyarrow.ArrowResult',\n 'table': shape: (3, 2)\n")

    def test_narwhals(self):
        df = nw.from_dict({"a": [1, 2, 3], "b": ["x", "y", "z"]}, backend="polars")
        self.logger.debug("Result:", extra={"result": NarwhalsDataFrameResult(df=df)})
        log = self.log_stream.getvalue().strip()
        assert log.startswith("test_logger - DEBUG - Result:\n{'_target_': 'ccflow.result.narwhals.NarwhalsDataFrameResult',\n 'df': shape: (3, 2)\n")
