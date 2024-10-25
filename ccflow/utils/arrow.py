"""Various arrow tools"""

from typing import Any, Dict, Optional

import orjson
import pyarrow as pa

from ccflow.serialization import orjson_dumps

__all__ = (
    "convert_decimal_types_to_float",
    "convert_large_types",
    "add_field_metadata",
    "get_field_metadata",
)


def convert_decimal_types_to_float(table: pa.table, target_type: Optional[pa.DataType] = None) -> pa.Table:
    """Converts decimal types to float or other user-provided type

    Args:
        table: The table to convert schema for
        target_type: The target type to convert decimal types to. If not supplied, will default to Float64

    Returns:
        A pyarrow table whose decimal types have been converted to the specified target type
    """

    if target_type is None:
        target_type = pa.float64()

    fields = []
    for field in table.schema:
        if pa.types.is_decimal(field.type):
            new_field = pa.field(field.name, target_type, field.nullable)
        else:
            new_field = field
        fields.append(new_field)

    schema = pa.schema(fields)
    return table.cast(schema)


def convert_large_types(table: pa.Table) -> pa.Table:
    """Converts the large types to their regular counterparts in pyarrow.

    This is necessary because polars always uses large list, but pyarrow
    recommends using the regular one, as it is more accepted (e.g. by csp)
    https://arrow.apache.org/docs/python/generated/pyarrow.large_list.html
    """
    fields = []
    for field in table.schema:
        if pa.types.is_large_list(field.type):
            new_field = pa.field(field.name, pa.list_(field.type.value_type), field.nullable)
        elif pa.types.is_large_binary(field.type):
            new_field = pa.field(field.name, pa.binary(), field.nullable)
        elif pa.types.is_large_string(field.type):
            new_field = pa.field(field.name, pa.string(), field.nullable)
        else:
            new_field = field
        fields.append(new_field)
    schema = pa.schema(fields)
    return table.cast(schema)


def add_field_metadata(table: pa.Table, metadata: Dict[str, Any]):
    """Helper function to add column-level meta data to an arrow table for multiple columns at once."""
    # There does not seem to be a pyarrow function to do this easily
    new_schema = []
    for field in table.schema:
        if field.name in metadata:
            field_metadata = {k: orjson_dumps(v) for k, v in metadata[field.name].items()}
            new_field = field.with_metadata(field_metadata)
        else:
            new_field = field
        new_schema.append(new_field)
    return table.cast(pa.schema(new_schema))


def get_field_metadata(table: pa.Table) -> Dict[str, Any]:
    """Helper function to retrieve all the field level metadata in an arrow table."""
    metadata = {}
    for field in table.schema:
        raw_metadata = field.metadata
        if raw_metadata:
            metadata[field.name] = {k.decode("UTF-8"): orjson.loads(v) for k, v in raw_metadata.items()}
    return metadata
