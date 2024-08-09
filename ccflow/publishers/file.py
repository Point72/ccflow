import pandas as pd
import pickle
import yaml
from cloudpathlib import AnyPath
from pydantic import Field, validator
from typing import IO, Any, Callable, Dict, Generic
from typing_extensions import Literal, override

from ..exttypes import JinjaTemplate
from ..publisher import BasePublisher
from ..serialization import orjson_dumps
from ..utils import PydanticDictOptions, PydanticModelType, dict_to_model
from ..utils.pydantic1to2 import GenericModel

__all__ = (
    "DictTemplateFilePublisher",
    "GenericFilePublisher",
    "JSONPublisher",
    "PandasFilePublisher",
    "PicklePublisher",
    "PydanticJSONPublisher",
    "YAMLPublisher",
)

DEFAULT_SEPARATOR = "/"


def _write_to_io(data: Any, f: IO, **kwargs):
    """Default write function that converts data to string and writes to IO"""
    return f.write(str(data))


def _orjson_file_dump(data, file, **kw):
    default = kw.pop("default", None)
    orjson_serialized_obj = orjson_dumps(data, default=default)
    file.write(orjson_serialized_obj)


class GenericFilePublisher(BasePublisher):
    dump: Callable[[Any, IO, Any], Any] = _write_to_io
    suffix: str
    mode: str = "w"
    kwargs: Dict[str, Any] = Field(default_factory=dict)

    @override
    def __call__(self) -> AnyPath:
        if self.data is None:
            raise ValueError("'data' field must be set before publishing")
        # This uses cloudpathlib's AnyPath so that S3 paths are supported
        path = AnyPath(self.get_name()).with_suffix(self.suffix)

        # Ensure the directory exists, and create if not
        path.parent.mkdir(parents=True, exist_ok=True)

        # This uses smart_open.open so that cloud paths are supported
        from smart_open import open  # Expensive import

        with open(path, self.mode) as f:
            self.dump(self.data, f, **self.kwargs)
        return path


class JSONPublisher(BasePublisher):
    """Publish results to file in JSON format."""

    kwargs: Dict[str, Any] = Field(default_factory=dict)

    @override
    def __call__(self) -> AnyPath:
        return GenericFilePublisher(
            name=self.name,
            name_params=self.name_params,
            data=self.data,
            dump=_orjson_file_dump,
            suffix=".json",
            kwargs=self.kwargs,
        )()


class YAMLPublisher(BasePublisher):
    """Publish results to file in YAML format."""

    kwargs: Dict[str, Any] = Field(default_factory=dict)

    @override
    def __call__(self) -> AnyPath:
        return GenericFilePublisher(
            name=self.name,
            name_params=self.name_params,
            data=self.data,
            dump=yaml.dump,
            suffix=".yaml",
            kwargs=self.kwargs,
        )()


class PicklePublisher(BasePublisher):
    """Pickle results to file."""

    protocol: int = -1
    kwargs: Dict[str, Any] = Field(default_factory=dict)

    @override
    def __call__(self) -> AnyPath:
        kwargs = self.kwargs.copy()
        kwargs["protocol"] = self.protocol
        return GenericFilePublisher(
            name=self.name,
            name_params=self.name_params,
            data=self.data,
            dump=pickle.dump,
            suffix=".pickle",
            mode="wb",
            kwargs=kwargs,
        )()


class DictTemplateFilePublisher(BasePublisher):
    """Write a dictionary of data to a JinjaTemplate"""

    data: Dict = None
    suffix: str
    template: JinjaTemplate

    @override
    def __call__(self) -> AnyPath:
        data = self.template.template.render(**self.data)
        return GenericFilePublisher(
            name=self.name,
            name_params=self.name_params,
            data=data,
            suffix=self.suffix,
        )()


class PydanticJSONPublisher(BasePublisher, GenericModel, Generic[PydanticModelType]):
    """See https://pydantic-docs.helpmanual.io/usage/exporting_models/#modeljson"""

    data: PydanticModelType = None
    options: PydanticDictOptions = Field(default_factory=PydanticDictOptions)
    kwargs: Dict[str, Any] = Field(default_factory=dict)

    _normalize_data = validator("data", pre=True, allow_reuse=True)(dict_to_model)

    @classmethod
    def dump(cls, data, file, **kwargs):
        out = data.json(**kwargs)
        file.write(out)

    @override
    def __call__(self) -> AnyPath:
        kwargs = self.options.dict()
        kwargs.update(self.kwargs)
        return GenericFilePublisher(
            name=self.name,
            name_params=self.name_params,
            data=self.data,
            dump=self.dump,
            suffix=".json",
            kwargs=kwargs,
        )()


class PandasFilePublisher(BasePublisher):
    """Generic way of publishing a pandas DataFrame to a file.

    For large-scale exporting (using parquet), see .parquet.PandasParquetPublisher.
    """

    data: pd.DataFrame = None
    kwargs: Dict[str, Any] = Field(default_factory=dict)
    func: str = "to_html"  # The access function must be able to write to a buffer or file-like object.
    suffix: str = ".html"
    mode: Literal["w", "wb"] = "w"

    @override
    def __call__(self) -> AnyPath:
        return GenericFilePublisher(
            name=self.name,
            name_params=self.name_params,
            data=self.data,
            dump=getattr(pd.DataFrame, self.func),
            suffix=self.suffix,
            mode=self.mode,
            kwargs=self.kwargs,
        )()