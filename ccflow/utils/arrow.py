"""Various arrow tools"""

import orjson
import pandas as pd
import pyarrow as pa
from typing import Any, Dict, Optional, Sequence

from ccflow.serialization import orjson_dumps


def convert_large_types(table: pa.Table) -> pa.Table:
    """Converts the large types to their regular counterparts in pyarrow.

    This is necessary because polars always using large list, but pyarrow
    recommends using the regular one, as it is more accepted (i.e. by csp)
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


def pandas_dict_to_arrow_lists(dfs: Dict[str, pd.DataFrame], index_name: Optional[str] = None) -> pa.Table:
    """Convert a dictionary of pandas frames to an arrow table of list types, with one column per frame.
    The indices of the dataframes must all be equal, and the index data will be in the "index_name" column.
    """
    if not dfs:
        raise ValueError("Input dictionary must be non-empty")
    # Need to make sure the index in the same for all frames
    all_dfs = list(dfs.values())
    for df in all_dfs[1:]:
        if not all_dfs[0].index.equals(df.index):
            # One could try to get clever by aligning the indices, but here we just throw
            raise ValueError("indices of all dataframes must be equal")
    data = {}
    if index_name:
        data[index_name] = pa.Array.from_pandas(all_dfs[0].index)
    data.update({col: pa.array(list(df.values)) for col, df in dfs.items()})
    table = pa.table(data)
    metadata = {col: {"columns": df.columns.to_list()} for col, df in dfs.items()}
    return add_field_metadata(table, metadata)


def arrow_lists_to_pandas_dict(
    table: pa.Table,
    *,
    fields: Sequence[str] = None,
    index_name: Optional[str] = None,
    columns: Optional[Sequence[str]] = None,
) -> Dict[str, pd.DataFrame]:
    """Convert an arrow table with list types into a dictionary of data frames.

    fields: The fields from the arrow table to include in the output. If not provided, will use all list-typed columns.
    index_name: The column that provides the index of the frames.
    columns: The columns to map to the lists. If not provided, will look for "columns" in field-level metadata
    """
    index = table[index_name] if index_name else None
    if fields is None:
        fields = [field.name for field in table.schema if pa.types.is_list(field.type)]
    metadata = get_field_metadata(table)
    outputs = {}
    for field in fields:
        if columns is None:
            field_columns = metadata.get(field, {}).get("columns")
        else:
            field_columns = columns
        if field_columns is None:
            raise ValueError("Must either provide column names or they must be on the field-level metadata of the table")
        data = table[field].to_numpy()
        outputs[field] = pd.DataFrame.from_records(
            [[] if d is None else d for d in data],
            columns=field_columns,
            index=index,
            nrows=len(data),
        )
    return outputs
