import abc
import pydantic
from packaging import version
from pydantic import Field
from typing import Any, Dict, TypeVar
from typing_extensions import override

from .base import BaseModel
from .exttypes import JinjaTemplate

__all__ = (
    "BasePublisher",
    "NullPublisher",
    "PublisherType",
)


class BasePublisher(BaseModel, abc.ABC):
    """A publisher is a configurable object (flow base model) that knows how to "publish" typed python objects.

    We use pydantic's type declarations to define the "type" of data that the publisher knows how to publish.
    Some examples of publishing destinations include: file (local or cloud), email, database, Rest API, Kafka, etc
    The naming convention for publishers is WhatWherePublisher or just WherePublisher if Any type is supported.
    """

    # The "name" by which to publish that data element
    name: JinjaTemplate = None
    # The parameters for the name template
    name_params: Dict[str, Any] = Field(default_factory=dict)
    # The data is a field on the publisher model so that we can use pydantic validation/coercion on it
    data: Any = None

    class Config(BaseModel.Config):
        # Want to validate assignment so that when new data is set on a publisher, validation gets applied
        validate_assignment = True
        # Many publishers will require arbitrary types set on data
        arbitrary_types_allowed = True
        if version.parse(pydantic.__version__) < version.parse("2"):
            # Do not copy on validate as "data" might be large
            copy_on_model_validation = "none"

    def get_name(self):
        """Get the name with the template parameters filled in."""
        if self.name is None:
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
