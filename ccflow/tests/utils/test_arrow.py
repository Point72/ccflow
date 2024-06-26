import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
from packaging import version
from unittest import TestCase

from ccflow.utils.arrow import add_field_metadata, arrow_lists_to_pandas_dict, convert_large_types, get_field_metadata, pandas_dict_to_arrow_lists


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


class TestMetaData(TestCase):
    def test_metadata(self):
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


class TestListConversions(TestCase):
    def setUp(self) -> None:
        self.df1 = pd.DataFrame([["foo", "bar"], ["baz", "qux"]], columns=["a", "b"], index=[1, 2])
        self.df2 = pd.DataFrame([[1.0, 2.0], [3.0, 4.0]], columns=["A", "B"], index=[1, 2])
        self.t = pa.Table.from_pydict(
            {
                "int": [1, 2],
                "str": ["foo", "bar"],
                "list_float": [[1.0, 2.0], [3.0, 4.0]],
                "list_str": [["foo", "bar"], ["baz", "qux"]],
            },
        )

    def test_arrow_lists_to_pandas_dict(self):
        fields = ["list_float", "list_str"]
        dfs = arrow_lists_to_pandas_dict(self.t, columns=["a", "b"])
        self.assertEqual(list(dfs.keys()), fields)
        pd.testing.assert_frame_equal(dfs["list_str"], self.df1.reset_index(drop=True))
        df2 = pd.DataFrame([[1.0, 2.0], [3.0, 4.0]], columns=["a", "b"])
        pd.testing.assert_frame_equal(dfs["list_float"], df2)

    def test_arrow_lists_to_pandas_dict_fields(self):
        fields = ["list_str"]
        dfs = arrow_lists_to_pandas_dict(self.t, fields=fields, columns=["a", "b"])
        self.assertEqual(list(dfs.keys()), fields)
        pd.testing.assert_frame_equal(dfs["list_str"], self.df1.reset_index(drop=True))

    def test_arrow_lists_to_pandas_dict_index(self):
        dfs = arrow_lists_to_pandas_dict(self.t, columns=["a", "b"], index_name="int")
        pd.testing.assert_frame_equal(
            dfs["list_str"],
            pd.DataFrame([["foo", "bar"], ["baz", "qux"]], columns=["a", "b"], index=[1, 2]),
        )
        pd.testing.assert_frame_equal(
            dfs["list_float"],
            pd.DataFrame([[1.0, 2.0], [3.0, 4.0]], columns=["a", "b"], index=[1, 2]),
        )

    def test_pandas_dict_to_arrow_lists(self):
        t = pandas_dict_to_arrow_lists({"list_str": self.df1, "list_float": self.df2})
        self.assertEqual(t, t.select(["list_str", "list_float"]))

        t = pandas_dict_to_arrow_lists({"list_str": self.df1, "list_float": self.df2}, index_name="int")
        self.assertEqual(t, t.select(["int", "list_str", "list_float"]))

        metadata = get_field_metadata(t)
        self.assertDictEqual(
            metadata,
            {
                "list_str": {"columns": ["a", "b"]},
                "list_float": {"columns": ["A", "B"]},
            },
        )

        self.assertRaises(ValueError, pandas_dict_to_arrow_lists, {})

    def test_pandas_dict_to_arrow_lists_roundtrip(self):
        # Test round-trip! This includes embedding the column information in the metadata
        t = pandas_dict_to_arrow_lists({"list_str": self.df1, "list_float": self.df2}, index_name="int")
        dfs = arrow_lists_to_pandas_dict(t, index_name="int")
        pd.testing.assert_frame_equal(dfs["list_str"], self.df1)
        pd.testing.assert_frame_equal(dfs["list_float"], self.df2)

    def test_nulls(self):
        t = pa.Table.from_pydict(
            {
                "int": [1, 2, 3],
                "str": ["foo", "bar", "baz"],
                "list_float": [[1.0, 2.0], None, [3.0, 4.0]],
                "list_str": [["foo", "bar"], None, ["baz", "qux"]],
            },
        )
        df1 = pd.DataFrame(
            [["foo", "bar"], [np.nan, np.nan], ["baz", "qux"]],
            columns=["a", "b"],
        )
        fields = ["list_float", "list_str"]
        dfs = arrow_lists_to_pandas_dict(t, columns=["a", "b"])
        self.assertEqual(list(dfs.keys()), fields)
        pd.testing.assert_frame_equal(dfs["list_str"], df1)
        df2 = pd.DataFrame([[1.0, 2.0], [np.nan, np.nan], [3.0, 4.0]], columns=["a", "b"])
        pd.testing.assert_frame_equal(dfs["list_float"], df2)
