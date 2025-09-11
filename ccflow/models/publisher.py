from typing import Generic, Type

from pydantic import Field
from typing_extensions import override

from ..callable import CallableModelType, ContextType, Flow, ResultType, WrapperModel
from ..publisher import PublisherType
from ..result import GenericResult

__all__ = ("PublisherModel",)


class PublisherModel(
    WrapperModel[CallableModelType],
    Generic[CallableModelType, PublisherType],
):
    """Model that chains together a callable model and a publisher to publish the results of the callable model."""

    publisher: PublisherType
    field: str = Field(None, description="Specific field on model output to publish")
    return_data: bool = Field(
        False,
        description="Whether to return the underlying model result as the output instead of the publisher output",
    )

    @property
    def result_type(self) -> Type[ResultType]:
        """Result type that will be returned. Could be over-ridden by child class."""
        if self.return_data:
            return self.model.result_type
        else:
            return GenericResult

    def _get_publisher(self, context):
        publisher = self.publisher.model_copy()
        # Set the name, if needed
        if not publisher.name and self.meta.name:
            publisher.name = self.meta.name
        # Augment any existing name parameters with the context parameters
        name_params = publisher.name_params.copy()
        name_params.update(context.model_dump(exclude={"type_"}))
        publisher.name_params = name_params
        return publisher

    @override
    @Flow.call
    def __call__(self, context: ContextType) -> ResultType:
        """This method gets the result from the underlying model, and publishes it."""
        publisher = self._get_publisher(context)
        data = self.model(context)
        if self.field:
            pub_data = getattr(data, self.field)
        else:
            pub_data = data
        publisher.data = pub_data
        out = publisher()
        if self.return_data:
            return data
        else:
            return self.result_type(value=out)
