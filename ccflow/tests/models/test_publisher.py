import gc
import weakref
from typing import Any
from unittest.mock import patch

from typing_extensions import override

from ccflow import BasePublisher, CallableModel, DictResult, Flow, GenericResult, NullContext
from ccflow.models import PublisherModel
from ccflow.publishers import PrintPublisher


class ModelTest(CallableModel):
    @Flow.call
    def __call__(self, context: NullContext) -> DictResult[str, str]:
        return DictResult[str, str](value={"message": "Hello, World!"})


class Frame:
    """A weak-referenceable stand-in for a large frame."""


_OBSERVATIONS: list = []


class FrameModel(CallableModel):
    @Flow.call
    def __call__(self, context: NullContext) -> GenericResult:
        return GenericResult(value=Frame())


class OwnershipCheckPublisher(BasePublisher):
    """Publisher that records whether it holds the sole reference to the frame.

    It drops its own reference to the frame and forces a collection; if the driver
    has released its references, the frame is collected and the weak reference dies.
    The observation is appended to the module-level ``_OBSERVATIONS`` list, which
    survives the ``model_copy`` that ``PublisherModel`` performs on the publisher.
    """

    @override
    def __call__(self) -> Any:
        ref = weakref.ref(self.data)
        self.data = None
        gc.collect()
        _OBSERVATIONS.append(ref() is not None)
        return "published"


class TestPublisherModel:
    def test_run(self):
        with patch("ccflow.publishers.print.print") as mock_print:
            model = PublisherModel(model=ModelTest(), publisher=PrintPublisher())
            res = model(None)
            assert isinstance(res, GenericResult)  # from PrintPublisher
            assert isinstance(res.value, DictResult[str, str])
            assert res.value.value == {"message": "Hello, World!"}
            assert mock_print.call_count == 1
            assert mock_print.call_args[0][0].value == {"message": "Hello, World!"}

    def test_release_references_when_not_returning_data(self):
        _OBSERVATIONS.clear()
        model = PublisherModel(model=FrameModel(), publisher=OwnershipCheckPublisher(), field="value", return_data=False)
        res = model(None)
        assert isinstance(res, GenericResult)
        assert res.value == "published"
        # The driver dropped its references, so the publisher owned the sole reference.
        assert _OBSERVATIONS == [False]

    def test_retains_references_when_returning_data(self):
        _OBSERVATIONS.clear()
        model = PublisherModel(model=FrameModel(), publisher=OwnershipCheckPublisher(), field="value", return_data=True)
        res = model(None)
        assert isinstance(res, GenericResult)
        assert isinstance(res.value, Frame)
        # The driver must keep the frame alive to return it, so it is not collected.
        assert _OBSERVATIONS == [True]
