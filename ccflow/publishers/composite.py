import logging
from typing import Dict, Generic, List, Optional

from pydantic import ValidationError, field_validator
from typing_extensions import override

from ..publisher import BasePublisher
from ..utils import PydanticDictOptions, PydanticModelType, dict_to_model

__all__ = ("CompositePublisher",)

log = logging.getLogger(__name__)

ROOT_KEY = "__root__"  # Used even outside the context of pydantic version 1


class CompositePublisher(BasePublisher, Generic[PydanticModelType]):
    """Highly configurable, publisher that decomposes a pydantic BaseModel or a dictionary into pieces
    and publishes each piece separately."""

    data: PydanticModelType = None
    sep: str = "/"
    # Map of field names to publisher
    field_publishers: Dict[str, BasePublisher] = {}
    # List of publishers that will be tried in order based on validation against "data" type
    default_publishers: List[BasePublisher] = []
    # Publisher for any remaining fields not covered by the above.
    root_publisher: Optional[BasePublisher] = None

    # Whether to expand fields that contain pydantic models into dictionaries.
    models_as_dict: bool = True
    # Options for iterating through the pydantic model.
    options: PydanticDictOptions = PydanticDictOptions()

    _normalize_data = field_validator("data", mode="before")(dict_to_model)

    def _get_dict(self):
        if self.data is None:
            raise ValueError("'data' field must be set before publishing")
        if self.models_as_dict:
            data = self.data.model_dump(**self.options.model_dump())
        else:
            data = dict(self.data)
        return data

    def _get_publishers(self, data):
        publishers = {}
        for field, value in data.items():
            full_name = self.sep.join((self.name, field)) if self.name else field
            publisher = self.field_publishers.get(field, None)

            if publisher is None:
                for try_publisher in self.default_publishers:
                    try:
                        try_publisher.data = value
                        publishers[field] = try_publisher.model_copy()
                        publishers[field].name = full_name
                        publishers[field].name_params = self.name_params
                        break
                    except ValidationError:
                        continue  # try next publisher in default_publishers
                if field not in publishers:
                    log.info(
                        "No sub-publisher found for field %s on %s named %s",
                        field,
                        self.__class__.__name__,
                        self.name,
                    )
            else:
                # If value is the wrong type for the configured publisher, it will raise
                # User should provide the right type of publisher for a given field.
                publisher.data = value
                if not publisher.name:
                    publisher = publisher.model_copy()
                    publisher.name = full_name
                    publisher.name_params = self.name_params
                publishers[field] = publisher
        return publishers

    def _get_root_publisher(self, data, publishers):
        root_publisher = self.root_publisher.model_copy(deep=True)
        if not root_publisher.name:
            root_publisher.name = self.name or ROOT_KEY
            root_publisher.name_params = self.name_params
        root_publisher.data = {f: v for f, v in data.items() if f not in publishers}
        # Only return a publisher if there is data to publish!
        if root_publisher.data:
            return root_publisher

    @override
    def __call__(self):
        data = self._get_dict()
        publishers = self._get_publishers(data)

        # At this point, we have a dict of publishers, each with "data" set.
        # Some publishers might be missing, i.e. if we failed to find a valid publisher.
        # We run through all publishers, and try to call each one
        outputs = {}
        exceptions = {}
        for field, publisher in publishers.items():
            try:
                outputs[field] = publisher()
            except Exception as e:
                exceptions[field] = e
                continue

        # Take "remaining" fields
        if self.root_publisher:
            root_publisher = self._get_root_publisher(data, publishers)
            if root_publisher:
                try:
                    outputs[ROOT_KEY] = root_publisher()
                except Exception as e:
                    exceptions[ROOT_KEY] = e

        # Re-raise any exceptions that occurred
        if exceptions:
            raise Exception(exceptions)

        return outputs
