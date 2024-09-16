from datetime import date
from unittest import TestCase

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.fs as fs
from pydantic import TypeAdapter

from ccflow import (
    ArrowDateFilter,
    ArrowFilter,
    ArrowLocalFileSystem,
    ArrowPartitioning,
    ArrowS3FileSystem,
    ArrowSchemaModel,
    ArrowTemplateFilter,
    render_filters,
)


class TestArrowParquetOptions(TestCase):
    def test_schemamodel_definition(self):
        # test equivalence of schema definition using Dict vs List and PyArrowDatatype vs pa.lib.DataType
        tuple_str_schema = ArrowSchemaModel(
            fields=[
                ("str_field", "pa.string()"),
                ("int_field", "pa.int32()"),
                ("float_field", "pa.float32()"),
                ("bool_field", "pa.bool_()"),
                ("date_field", "pa.date32()"),
                ("timestamp_field", "pa.timestamp('ns')"),
                ("list_field", "pa.list_(pa.int32())"),
            ]
        )
        dict_str_schema = ArrowSchemaModel(
            fields={
                "str_field": "pa.string()",
                "int_field": "pa.int32()",
                "float_field": "pa.float32()",
                "bool_field": "pa.bool_()",
                "date_field": "pa.date32()",
                "timestamp_field": "pa.timestamp('ns')",
                "list_field": "pa.list_(pa.int32())",
            }
        )
        tuple_pyarrow_schema = ArrowSchemaModel(
            fields=[
                ("str_field", pa.string()),
                ("int_field", pa.int32()),
                ("float_field", pa.float32()),
                ("bool_field", pa.bool_()),
                ("date_field", pa.date32()),
                ("timestamp_field", pa.timestamp("ns")),
                ("list_field", pa.list_(pa.int32())),
            ]
        )
        dict_pyarrow_schema = ArrowSchemaModel(
            fields={
                "str_field": pa.string(),
                "int_field": pa.int32(),
                "float_field": pa.float32(),
                "bool_field": pa.bool_(),
                "date_field": pa.date32(),
                "timestamp_field": pa.timestamp("ns"),
                "list_field": pa.list_(pa.int32()),
            }
        )
        # all of the above schemas must be identical
        self.assertEqual(tuple_str_schema.schema, dict_str_schema.schema)
        self.assertEqual(tuple_str_schema.schema, tuple_pyarrow_schema.schema)
        self.assertEqual(tuple_str_schema.schema, dict_pyarrow_schema.schema)

        # schema and object properties point to the same thing
        self.assertEqual(tuple_str_schema.schema, tuple_str_schema.object)

    def test_schemamodel_metadata(self):
        # test ability to define and retain metadata in ArrowSchemaModel
        schemamodel_with_metadata = ArrowSchemaModel(
            fields=[
                ("str_field", "pa.string()"),
                ("int_field", "pa.int32()"),
            ],
            metadata={
                "str_field": "this is a string field",
                "int_field": "this is an int field",
            },
        )

        expected_schema = pa.schema(
            [
                ("str_field", pa.string()),
                ("int_field", pa.int32()),
            ],
            metadata={
                "str_field": "this is a string field",
                "int_field": "this is an int field",
            },
        )
        self.assertEqual(schemamodel_with_metadata.schema, expected_schema)

    def test_bad_schemamodel(self):
        # test validation errors on bad schemas
        self.assertRaises(ValueError, ArrowSchemaModel, fields=[("str_field", True)])
        self.assertRaises(ValueError, ArrowSchemaModel, fields={"str_field": 7})
        self.assertRaises(ValueError, ArrowSchemaModel, fields=[("str_field", "foo")])
        self.assertRaises(
            ValueError,
            ArrowSchemaModel,
            fields={"str_field": "pa.timestamp('foo')"},
        )

    def test_schemamodel_validate(self):
        s = pa.schema(
            [pa.field("date", pa.date32()), pa.field("x", pa.float64())],
            metadata={"A": "b"},
        )
        model = TypeAdapter(ArrowSchemaModel).validate_python(s)

        target = ArrowSchemaModel(fields={"date": pa.date32(), "x": pa.float64()}, metadata={"A": "b"})
        self.assertEqual(model, target)


class TestArrowFilters(TestCase):
    def test_filters(self):
        f = ArrowFilter(key="foo", op="==", value=5)
        self.assertEqual(f.tuple(), ("foo", "==", 5))

    def test_render(self):
        filters = [
            ArrowFilter(key="foo", op="==", value=5),
            ArrowTemplateFilter(key="bar", op="==", value="{{x}}"),
        ]
        target = [
            ("foo", "==", 5),
            ("bar", "==", "hello"),
        ]
        self.assertListEqual(render_filters(filters, {"x": "hello"}), target)

    def test_render_nested(self):
        filters = [
            [
                ArrowFilter(key="foo", op="==", value=5),
                ArrowTemplateFilter(key="bar", op="==", value="{{x}}"),
            ],
            [ArrowDateFilter(key="baz", op=">", value=date(2020, 1, 1))],
        ]
        target = [
            [
                ("foo", "==", 5),
                ("bar", "==", "hello"),
            ],
            [("baz", ">", date(2020, 1, 1))],
        ]
        self.assertListEqual(render_filters(filters, {"x": "hello"}), target)


class TestArrowPartitioning(TestCase):
    def test_schema(self):
        schema = pa.schema([pa.field("date", pa.date32())])
        model = ArrowPartitioning(schema=schema)
        p = model.object
        self.assertEqual(p.schema, schema)
        self.assertIsInstance(p, ds.DirectoryPartitioning)
        self.assertEqual(model.get_partition_columns(), ["date"])

    def test_schema_hive(self):
        schema = pa.schema([pa.field("date", pa.date32())])
        model = ArrowPartitioning(schema=schema, flavor="hive")
        p = model.object
        self.assertEqual(p.schema, schema)
        self.assertIsInstance(p, ds.HivePartitioning)

    def test_field_names(self):
        field_names = ["date", "symbol"]
        model = ArrowPartitioning(field_names=field_names)
        p = model.object
        self.assertIsInstance(p, ds.PartitioningFactory)
        self.assertEqual(model.get_partition_columns(), field_names)


class TestArrowFileSystem(TestCase):
    def test_local(self):
        f = ArrowLocalFileSystem()
        self.assertIsInstance(f.object, fs.LocalFileSystem)

    def test_s3(self):
        f = ArrowS3FileSystem(access_key="foo", secret_key="bar")
        self.assertIsInstance(f.object, fs.S3FileSystem)
