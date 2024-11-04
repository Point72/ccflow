from decimal import Decimal
from unittest import TestCase

import pandas as pd
import polars as pl
import pyarrow as pa
from packaging import version

from ccflow.utils.arrow import add_field_metadata, convert_decimal_types_to_float, convert_large_types, get_field_metadata


class TestArrowUtil(TestCase):
    def test_convert_large_types(self):
        schema = pa.schema(
            [
                pa.field("int", pa.int16()),
                pa.field("str", pa.string()),
                pa.field("large_list_float", pa.large_list(pa.float32())),
                pa.field("large_list_str", pa.large_list(pa.string())),
                pa.field("large_string", pa.large_string()),
                pa.field("large_binary", pa.large_binary()),
            ]
        )
        t = pa.Table.from_pydict(
            {
                "int": [1, 2],
                "str": ["foo", "bar"],
                "large_list_float": [[1.0, 2.0], [3.0, 4.0]],
                "large_list_str": [["foo", "bar"], ["baz", "qux"]],
                "large_string": ["foo", "bar"],
                "large_binary": [bytes(b"abc"), bytes(b"def")],
            },
            schema=schema,
        )
        out = convert_large_types(t)

        target_schema = pa.schema(
            [
                pa.field("int", pa.int16()),
                pa.field("str", pa.string()),
                pa.field("large_list_float", pa.list_(pa.float32())),
                pa.field("large_list_str", pa.list_(pa.string())),
                pa.field("large_string", pa.string()),
                pa.field("large_binary", pa.binary()),
            ]
        )
        self.assertEqual(out.schema, target_schema)
        pd.testing.assert_frame_equal(out.to_pandas(), t.to_pandas())

    def test_polars_large_list(self):
        """The function convert_large_types is necessary because polars uses large list.
        This test continues to verify that this is the case in polars. If it fails,
        it may be possible to remove the convert_large_types logic."""
        df = pl.DataFrame({"a": [1, 2, 3]})
        if version.parse(pl.__version__) < version.parse("0.18"):
            t = df.select(pl.col("a").list()).to_arrow()
        else:
            # list() was renamed to implode().
            t = df.select(pl.col("a").implode()).to_arrow()

        # This is the schema that polars returns, which necessitates convert_large_types
        target_schema = pa.schema([pa.field("a", pa.large_list(pa.int64()))])
        self.assertEqual(t.schema, target_schema)

    def test_convert_decimal_types_to_float(self):
        table = pa.Table.from_pydict(
            {"sym": ["A", "B", "C"], "price": [Decimal(2.0), Decimal(3.0), Decimal(50.0)]},
            pa.schema([pa.field("sym", pa.string()), pa.field("price", pa.decimal128(8, 3))]),
        )

        converted_table = convert_decimal_types_to_float(table, pa.float64())
        target_schema = pa.schema([pa.field("sym", pa.string()), pa.field("price", pa.float64())])
        target_df = pd.DataFrame({"sym": ["A", "B", "C"], "price": [2.0, 3.0, 50.0]})

        self.assertEqual(converted_table.schema, target_schema)
        pd.testing.assert_frame_equal(target_df, converted_table.to_pandas())


class TestMetaData(TestCase):
    def test_add_field_metadata(self):
        t = pa.Table.from_pydict(
            {
                "int": [1, 2],
                "str": ["foo", "bar"],
                "list_float": [[1.0, 2.0], [3.0, 4.0]],
                "list_str": [["foo", "bar"], ["baz", "qux"]],
            },
        )
        self.assertEqual(get_field_metadata(t), {})
        metadata = {"int": {"foo": 4}, "str": {"bar": ["a", "b"], "baz": None}}
        t2 = add_field_metadata(t, metadata)
        self.assertEqual(t2.schema.field("int").metadata[b"foo"], b"4")
        self.assertEqual(t2.schema.field("str").metadata[b"bar"], b'["a","b"]')
        self.assertEqual(t2.schema.field("str").metadata[b"baz"], b"null")

        # Get it back in "normal" form
        self.assertDictEqual(get_field_metadata(t2), metadata)
