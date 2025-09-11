from unittest.mock import patch

from ccflow import CallableModel, DictResult, Flow, GenericResult, NullContext
from ccflow.models import PublisherModel
from ccflow.publishers import PrintPublisher


class TestModel(CallableModel):
    @Flow.call
    def __call__(self, context: NullContext) -> DictResult[str, str]:
        return DictResult[str, str](value={"message": "Hello, World!"})


class TestPublisherModel:
    def test_run(self):
        with patch("ccflow.publishers.print.print") as mock_print:
            model = PublisherModel(model=TestModel(), publisher=PrintPublisher())
            res = model(None)
            assert isinstance(res, GenericResult)  # from PrintPublisher
            assert isinstance(res.value, DictResult[str, str])
            assert res.value.value == {"message": "Hello, World!"}
            assert mock_print.call_count == 1
            assert mock_print.call_args[0][0].value == {"message": "Hello, World!"}
