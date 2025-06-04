from datetime import date
from unittest import TestCase
from unittest.mock import MagicMock, patch

from pydantic import BaseModel as PydanticBaseModel

from ccflow.exttypes import NDArray
from ccflow.publishers import (
    LogPublisher,
    PrintJSONPublisher,
    PrintPublisher,
    PrintPydanticJSONPublisher,
    PrintYAMLPublisher,
)


class MyTestModel(PydanticBaseModel):
    foo: int
    bar: date
    baz: NDArray[float]


class TestPrintPublishers(TestCase):
    def test_print(self):
        with patch("ccflow.publishers.print.print") as mock_print:
            p = PrintPublisher(
                name="test_{{param}}",
                name_params={"param": "JSON"},
            )
            p.data = {"foo": 5, "bar": date(2020, 1, 1)}
            p()
            assert mock_print.call_count == 1
            mock_print.assert_called_with(p.data)

    def test_log(self):
        with patch("logging.getLogger") as mock_getLogger:
            mock_getLogger.return_value = MagicMock()
            p = LogPublisher(
                name="test_{{param}}",
                name_params={"param": "JSON"},
            )
            p.data = {"foo": 5, "bar": date(2020, 1, 1)}
            p()
            assert mock_getLogger.return_value.log.call_count == 1
            mock_getLogger.return_value.log.assert_called_with(level=20, msg=p.data)

    def test_json(self):
        with patch("ccflow.publishers.print.print") as mock_print:
            p = PrintJSONPublisher(
                name="test_{{param}}",
                name_params={"param": "JSON"},
                kwargs=dict(default=str),
            )
            p.data = {"foo": 5, "bar": date(2020, 1, 1)}
            p()
            assert mock_print.call_count == 1
            mock_print.assert_called_with('{"foo":5,"bar":"2020-01-01"}')

    def test_yaml(self):
        with patch("ccflow.publishers.print.print") as mock_print:
            p = PrintYAMLPublisher(
                name="test_{{param}}",
                name_params={"param": "JSON"},
            )
            p.data = {"foo": 5, "bar": date(2020, 1, 1)}
            p()
            assert mock_print.call_count == 1
            mock_print.assert_called_with("bar: 2020-01-01\nfoo: 5\n")

    def test_json_pydantic(self):
        with patch("ccflow.publishers.print.print") as mock_print:
            p = PrintPydanticJSONPublisher(
                name="test_{{param}}",
                name_params={"param": "JSON"},
            )
            p.data = {"foo": 5, "bar": date(2020, 1, 1)}
            p()
            assert mock_print.call_count == 1
            mock_print.assert_called_with('{"foo":5,"bar":"2020-01-01"}')
