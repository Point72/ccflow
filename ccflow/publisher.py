import abc
from typing import Any, Dict, TypeVar

from pydantic import ConfigDict, Field
from typing_extensions import override

from .base import BaseModel
from .exttypes import JinjaTemplate

__all__ = ("BasePublisher", "NullPublisher", "PublisherType")


class BasePublisher(BaseModel, abc.ABC):
    """A publisher is a configurable object (flow base model) that knows how to "publish" typed python objects.

    We use pydantic's type declarations to define the "type" of data that the publisher knows how to publish.
    The naming convention for publishers is WhatWherePublisher or just WherePublisher if Any type is supported.
    """

    model_config = ConfigDict(
        # Want to validate assignment so that when new data is set on a publisher, validation gets applied
        validate_assignment=True,
        # Many publishers will require arbitrary types set on data
        arbitrary_types_allowed=True,
    )
    name: JinjaTemplate = Field(None, description="The 'name' by which to publish that data element")
    name_params: Dict[str, Any] = Field(default_factory=dict, description="The parameters for the name template")
    data: Any = Field(None, description="The data we are going to publish")

    def get_name(self):
        """Get the name with the template parameters filled in."""
        if not self.name:
            raise ValueError("Name must be set")
        return self.name.template.render(**self.name_params)

    @abc.abstractmethod
    def __call__(self) -> Any:
        """Publish the data."""


PublisherType = TypeVar("CallableModelType", bound=BasePublisher)


class NullPublisher(BasePublisher):
    """A publisher which does nothing!"""

    @override
    def __call__(self) -> Any:
        pass
