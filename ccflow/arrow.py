"""This modules holds BaseModels that facilitate working with Arrow classes.
Note that arrow related extension types are in exttypes.arrow.
"""

import abc
from datetime import date, datetime, time
from typing import Any, Dict, List, Optional, Union

import pyarrow as pa
from pydantic import Field, model_validator
from typing_extensions import Literal  # For pydantic 1 compatibility on python 3.9

from .base import BaseModel
from .exttypes import JinjaTemplate, PyArrowDatatype, PyObjectPath
from .object_config import ObjectConfig

__all__ = (
    "ArrowSchemaModel",
    "ArrowPartitioning",
    "ArrowDateFilter",
    "ArrowDatetimeFilter",
    "ArrowFilter",
    "ArrowTimeFilter",
    "ArrowTemplateFilter",
    "ArrowFileSystem",
    "ArrowLocalFileSystem",
    "ArrowS3FileSystem",
    "render_filters",
)


class ArrowFilter(BaseModel):
    """A custom model to help represent the filters to read parquet datasets.
    Allows for better typing for non-basic types in configs.
    """

    key: str
    op: str
    value: Any

    def tuple(self):
        """Convert the filter back to a tuple"""
        return self.key, self.op, self.value

    @model_validator(mode="before")
    def _validate_fields(cls, v, info):
        if isinstance(v, (list, tuple)):
            key, op, value = v
            return dict(key=key, op=op, value=value)
        return v


class ArrowDateFilter(ArrowFilter):
    value: date


class ArrowDatetimeFilter(ArrowFilter):
    value: datetime


class ArrowTimeFilter(ArrowFilter):
    value: time


class ArrowTemplateFilter(ArrowFilter):
    value: JinjaTemplate


def _render_filter(f, template_args):
    if isinstance(f, ArrowTemplateFilter):
        return (f.key, f.op, f.value.template.render(**template_args))
    else:
        return f.tuple()


def render_filters(
    filters: Optional[Union[List[ArrowFilter], List[List[ArrowFilter]]]] = None,
    template_args: Dict[str, Any] = None,
):
    """Fill in template arguments in a list of filters, and return the standard Arrow form."""
    if isinstance(filters[0], ArrowFilter):
        return [_render_filter(f, template_args) for f in filters]
    else:
        return [[_render_filter(f, template_args) for f in flist] for flist in filters]


class ArrowFileSystem(ObjectConfig, abc.ABC):
    """ArrowFilesystem is a wrapping of pyarrow.fs.Filesystem.
    See https://arrow.apache.org/docs/python/filesystems.html
    """


_LOCAL_FILE_SYSTEM = PyObjectPath("pyarrow.fs.LocalFileSystem")


class ArrowLocalFileSystem(ArrowFileSystem):
    """Wrapping of pyarrow.fs.LocalFilesystem.
    See https://arrow.apache.org/docs/python/generated/pyarrow.fs.LocalFileSystem.html
    """

    object_type: Literal[_LOCAL_FILE_SYSTEM] = _LOCAL_FILE_SYSTEM


_S3_FILE_SYSTEM = PyObjectPath("pyarrow.fs.S3FileSystem")


class ArrowS3FileSystem(ArrowFileSystem):
    """Wrapping of pyarrow.fs.S3FileSystem.
    See https://arrow.apache.org/docs/python/generated/pyarrow.fs.S3FileSystem.html"""

    object_type: Literal[_S3_FILE_SYSTEM] = _S3_FILE_SYSTEM


class ArrowSchemaModel(BaseModel):
    """ArrowSchemaModel is a pydantic model of pyarrow schema"""

    fields: Dict[str, PyArrowDatatype]
    metadata: Optional[Dict[bytes, bytes]] = None

    @model_validator(mode="before")
    def _validate_fields(cls, values, info):
        if isinstance(values, dict) and "fields" in values:
            values["fields"] = dict(values["fields"])
        return values

    @model_validator(mode="wrap")
    def _schema_validator(cls, v, handler, info):
        if isinstance(v, pa.Schema):
            v = dict(
                fields={name: v.field(name).type for name in v.names},
                metadata=v.metadata,
            )
        return handler(v)

    @property
    def schema(self) -> pa.Schema:
        if isinstance(self.fields, Dict):
            return pa.schema(
                {k: v.datatype if isinstance(v, PyArrowDatatype) else v for k, v in self.fields.items()},
                self.metadata,
            )
        if isinstance(self.fields, List):
            return pa.schema(
                [(k, v.datatype if isinstance(v, PyArrowDatatype) else v) for (k, v) in self.fields],
                self.metadata,
            )

    @property
    def object(self) -> pa.Schema:
        """Alias to schema method as it's more consistent with other object wrappings"""
        return self.schema


class ArrowPartitioning(BaseModel):
    """ArrowPartitioning is a pydantic wrapping of pyarrow.dataset.Partitioning
    See https://arrow.apache.org/docs/python/generated/pyarrow.dataset.partitioning.html
    """

    arrow_schema: Optional[ArrowSchemaModel] = Field(None, alias="schema")
    field_names: Optional[List[str]] = None
    flavor: Optional[str] = None
    dictionaries: Optional[Dict[str, Any]] = None

    @property
    def object(self):  # -> Union[ds.Partitioning, ds.PartitioningFactory]:
        import pyarrow.dataset as ds  # Heavy import, only import if used.

        return ds.partitioning(
            schema=self.arrow_schema.schema if self.arrow_schema else None,
            field_names=self.field_names,
            flavor=self.flavor,
            dictionaries=self.dictionaries,
        )

    def get_partition_columns(self) -> List[str]:
        """Return the list of partition columns"""
        if self.arrow_schema is not None:
            return self.arrow_schema.object.names
        elif self.field_names is not None:
            return self.field_names
        else:
            return []
