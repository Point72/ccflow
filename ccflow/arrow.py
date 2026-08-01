"""This modules holds BaseModels that facilitate working with Arrow classes.
Note that arrow related extension types are in exttypes.arrow.
"""

import abc
from datetime import date, datetime, time
from typing import Any, Literal

import pyarrow as pa
from pydantic import Field, model_validator

from .base import BaseModel
from .exttypes import JinjaTemplate, PyArrowDatatype, PyObjectPath
from .object_config import ObjectConfig

__all__ = (
    "ArrowDateFilter",
    "ArrowDatetimeFilter",
    "ArrowFileSystem",
    "ArrowFilter",
    "ArrowLocalFileSystem",
    "ArrowPartitioning",
    "ArrowS3FileSystem",
    "ArrowSchemaModel",
    "ArrowTemplateFilter",
    "ArrowTimeFilter",
    "render_filters",
)


class ArrowFilter(BaseModel):
    """Wrapping of pyarrow filter tuples as a ccflow BaseModel.

    Allows for better typing of filters in configs for pyarrow file reads.
    """

    key: str
    op: str
    value: Any

    def tuple(self) -> tuple:
        """Convert the filter back to a tuple"""
        return self.key, self.op, self.value

    @model_validator(mode="before")
    def _validate_fields(cls, v, info):
        if isinstance(v, (list, tuple)):
            key, op, value = v
            return {"key": key, "op": op, "value": value}
        return v


class ArrowDateFilter(ArrowFilter):
    """An ArrowFilter where value is validated as a date"""

    value: date


class ArrowDatetimeFilter(ArrowFilter):
    """An ArrowFilter where value is validated as a datetime"""

    value: datetime


class ArrowTimeFilter(ArrowFilter):
    """An ArrowFilter where value is validated as a time"""

    value: time


class ArrowTemplateFilter(ArrowFilter):
    """An ArrowFilter where value is validated as a Jinja template"""

    value: JinjaTemplate


def _render_filter(f, template_args):
    if isinstance(f, ArrowTemplateFilter):
        return (f.key, f.op, f.value.template.render(**template_args))
    else:
        return f.tuple()


def render_filters(
    filters: list[ArrowFilter] | list[list[ArrowFilter]],
    template_args: dict[str, Any],
) -> list[ArrowFilter] | list[list[ArrowFilter]]:
    """Fill in template arguments in a list of filters, and return the standard Arrow form.

    Args:
        filters: The collection of filters to render.
        template_args: The template arguments for the filters
    """
    if isinstance(filters[0], ArrowFilter):
        return [_render_filter(f, template_args) for f in filters]
    else:
        return [[_render_filter(f, template_args) for f in flist] for flist in filters]


class ArrowFileSystem(ObjectConfig, abc.ABC):
    """Wrapping of pyarrow.fs.Filesystem as a generic ObjectConfig.

    See https://arrow.apache.org/docs/python/filesystems.html
    """


_LOCAL_FILE_SYSTEM = PyObjectPath("pyarrow.fs.LocalFileSystem")
_S3_FILE_SYSTEM = PyObjectPath("pyarrow.fs.S3FileSystem")


class ArrowLocalFileSystem(ArrowFileSystem):
    """Wrapping of pyarrow.fs.LocalFilesystem as a generic ObjectConfig.

    See https://arrow.apache.org/docs/python/generated/pyarrow.fs.LocalFileSystem.html
    """

    object_type: Literal[_LOCAL_FILE_SYSTEM] = _LOCAL_FILE_SYSTEM


class ArrowS3FileSystem(ArrowFileSystem):
    """Wrapping of pyarrow.fs.S3FileSystem as a generic ObjectConfig.

    See https://arrow.apache.org/docs/python/generated/pyarrow.fs.S3FileSystem.html"""

    object_type: Literal[_S3_FILE_SYSTEM] = _S3_FILE_SYSTEM


class ArrowSchemaModel(BaseModel):
    """Wrapping of pyarrow.Schema as a ccflow BaseModel."""

    fields: dict[str, PyArrowDatatype]
    metadata: dict[bytes, bytes] | None = None

    @model_validator(mode="before")
    def _validate_fields(cls, values, info):
        if isinstance(values, dict) and "fields" in values:
            values["fields"] = dict(values["fields"])
        return values

    @model_validator(mode="wrap")
    def _schema_validator(cls, v, handler, info):
        if isinstance(v, pa.Schema):
            v = {
                "fields": {name: v.field(name).type for name in v.names},
                "metadata": v.metadata,
            }
        return handler(v)

    @property
    def schema(self) -> pa.Schema:
        if isinstance(self.fields, dict):
            return pa.schema(
                {k: v.datatype if isinstance(v, PyArrowDatatype) else v for k, v in self.fields.items()},
                self.metadata,
            )
        if isinstance(self.fields, list):
            return pa.schema(
                [(k, v.datatype if isinstance(v, PyArrowDatatype) else v) for (k, v) in self.fields],
                self.metadata,
            )

    @property
    def object(self) -> pa.Schema:
        """Alias to schema method as it's more consistent with other object wrappings"""
        return self.schema


class ArrowPartitioning(BaseModel):
    """Wrapping of pyarrow.dataset.Partitioning as a ccflow BaseModel.

    See https://arrow.apache.org/docs/python/generated/pyarrow.dataset.partitioning.html
    """

    arrow_schema: ArrowSchemaModel | None = Field(None, alias="schema")
    field_names: list[str] | None = None
    flavor: str | None = None
    dictionaries: dict[str, Any] | None = None

    @property
    def object(self):  # -> Union[ds.Partitioning, ds.PartitioningFactory]:
        """Return the underlying pyarrow dataset Partitioning object"""
        import pyarrow.dataset as ds  # Heavy import, only import if used.

        return ds.partitioning(
            schema=self.arrow_schema.schema if self.arrow_schema else None,
            field_names=self.field_names,
            flavor=self.flavor,
            dictionaries=self.dictionaries,
        )

    def get_partition_columns(self) -> list[str]:
        """Return the list of partition columns"""
        if self.arrow_schema:
            return self.arrow_schema.object.names
        elif self.field_names:
            return self.field_names
        else:
            return []
