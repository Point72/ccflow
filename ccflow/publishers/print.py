import logging
from typing import Any, Dict, Generic

import yaml
from pydantic import Field, field_validator
from typing_extensions import Literal, override

from ..publisher import BasePublisher
from ..serialization import orjson_dumps
from ..utils import PydanticDictOptions, PydanticModelType, dict_to_model

__all__ = (
    "PrintPublisher",
    "LogPublisher",
    "PrintJSONPublisher",
    "PrintYAMLPublisher",
    "PrintPydanticJSONPublisher",
)


def _orjson_dumps(data: Any, **kwargs):
    default = kwargs.pop("default", None)
    return orjson_dumps(data, default=default)


class PrintPublisher(BasePublisher):
    """Print data using python standard print."""

    @override
    def __call__(self) -> Any:
        if self.data is None:
            raise ValueError("'data' field must be set before publishing")
        print(self.data)
        return self.data


class LogPublisher(BasePublisher):
    """Print data using python standard logging."""

    level: Literal["debug", "info", "warning", "error", "critical"] = Field(
        "info",
        description="The log level to use for logging the data",
    )
    logger_name: str = Field(
        "ccflow",
        description="The name of the logger to use for logging the data",
    )

    @override
    def __call__(self) -> Any:
        if self.data is None:
            raise ValueError("'data' field must be set before publishing")

        logging.getLogger(self.logger_name).log(
            level=getattr(logging, self.level.upper()),
            msg=self.data,
        )

        return self.data


class PrintJSONPublisher(BasePublisher):
    """Print data in JSON format."""

    kwargs: Dict[str, Any] = Field(default_factory=dict)

    @override
    def __call__(self) -> Any:
        print(_orjson_dumps(self.data, **self.kwargs))
        return self.data


class PrintYAMLPublisher(BasePublisher):
    """Print data in YAML format."""

    kwargs: Dict[str, Any] = Field(default_factory=dict)

    @override
    def __call__(self) -> Any:
        print(yaml.dump(self.data, **self.kwargs))
        return self.data


class PrintPydanticJSONPublisher(BasePublisher, Generic[PydanticModelType]):
    """Print pydantic model as json.

    See https://docs.pydantic.dev/latest/concepts/serialization/#modelmodel_dump_json
    """

    data: PydanticModelType = None
    options: PydanticDictOptions = Field(default_factory=PydanticDictOptions)
    kwargs: Dict[str, Any] = Field(default_factory=dict)

    _normalize_data = field_validator("data", mode="before")(dict_to_model)

    @override
    def __call__(self) -> Any:
        kwargs = self.options.model_dump(mode="python")
        kwargs.update(self.kwargs)
        print(self.data.model_dump_json(**kwargs))
        return self.data
