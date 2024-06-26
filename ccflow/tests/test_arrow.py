import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.fs as fs
import pydantic
from datetime import date
from packaging import version
from pydantic import ValidationError
from unittest import TestCase

from ccflow import (
    ArrowDateFilter,
    ArrowFilter,
    ArrowLocalFileSystem,
    ArrowPartitioning,
    ArrowS3FileSystem,
    ArrowSchemaModel,
    ArrowSchemaTransformModel,
    ArrowTableTransform,
    ArrowTemplateFilter,
    render_filters,
)
from ccflow.utils.pydantic1to2 import ValidationTypeError


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
        self.assertRaises(ValidationTypeError, ArrowSchemaModel, fields=[("str_field", True)])
        self.assertRaises(ValidationTypeError, ArrowSchemaModel, fields={"str_field": 7})
        self.assertRaises(ValidationTypeError, ArrowSchemaModel, fields=[("str_field", "foo")])
        self.assertRaises(
            ValidationTypeError,
            ArrowSchemaModel,
            fields={"str_field": "pa.timestamp('foo')"},
        )

    def test_schemamodel_validate(self):
        s = pa.schema(
            [pa.field("date", pa.date32()), pa.field("x", pa.float64())],
            metadata={"A": "b"},
        )
        if version.parse(pydantic.__version__) < version.parse("2"):
            model = ArrowSchemaModel.validate(s)
        else:
            from pydantic import TypeAdapter

            model = TypeAdapter(ArrowSchemaModel).validate_python(s)

        target = ArrowSchemaModel(fields={"date": pa.date32(), "x": pa.float64()}, metadata={"A": "b"})
        self.assertEqual(model, target)


class TestArrowSchemaTransform(TestCase):
    transform_list = [
        ("str_field", "STR_FIELD", "pa.string()"),
        ("int_field", "FLOAT_FIELD", "pa.float32()"),
        ("date_field", "TIMESTAMP_FIELD", "pa.timestamp('ns')"),
    ]

    transform_dict = {
        "str_field": ("STR_FIELD", "pa.string()"),
        "int_field": ("FLOAT_FIELD", "pa.float32()"),
        "date_field": ("TIMESTAMP_FIELD", "pa.timestamp('ns')"),
    }

    metadata = {
        "STR_FIELD": "this is a string field",
        "FLOAT_FIELD": "this is an float field",
    }

    # missing element
    bad_transform1 = [
        ("STR_FIELD", "pa.string()"),
        ("int_field", "INT_FIELD", "pa.int32()"),
        ("date_field", "DATE_FIELD", "pa.date32()"),
    ]

    # bad pyarrow datatype
    bad_transform2 = [
        ("str_field", "STR_FIELD", "pa.foo()"),
        ("int_field", "INT_FIELD", "pa.int32()"),
        ("date_field", "DATE_FIELD", "pa.date32()"),
    ]

    # duplicate schema field
    bad_transform3 = [
        ("str_field", "STR_FIELD", "pa.foo()"),
        ("int_field", "STR_FIELD", "pa.int32()"),
        ("date_field", "DATE_FIELD", "pa.date32()"),
    ]

    data = {
        "str_field": ["abc", "def", "ghi"],
        "int_field": [1, 2, 3],
        "date_field": [date(2023, 1, 1), date(2023, 1, 2), date(2023, 1, 3)],
        "dummy_field": ["foo", "bar", "baz"],
    }

    expected_schema = pa.schema(
        fields={
            "STR_FIELD": pa.string(),
            "FLOAT_FIELD": pa.float32(),
            "TIMESTAMP_FIELD": pa.timestamp("ns"),
        },
        metadata=metadata,
    )

    def test_transform_model(self):
        expected_input_fields = ["str_field", "int_field", "date_field"]
        expected_output_fields = ["STR_FIELD", "FLOAT_FIELD", "TIMESTAMP_FIELD"]

        m = ArrowSchemaTransformModel(fields=self.transform_list, metadata=self.metadata)
        self.assertListEqual(expected_input_fields, m.input_fields)
        self.assertListEqual(expected_output_fields, m.output_fields)
        self.assertEqual(self.expected_schema, m.schema)

        m = ArrowSchemaTransformModel(fields=self.transform_dict, metadata=self.metadata)
        self.assertListEqual(expected_input_fields, m.input_fields)
        self.assertListEqual(expected_output_fields, m.output_fields)
        self.assertEqual(self.expected_schema, m.schema)

        self.assertRaises(
            ValidationError,
            ArrowSchemaTransformModel,
            fields=self.bad_transform1,
        )
        self.assertRaises(
            ValidationTypeError,
            ArrowSchemaTransformModel,
            fields=self.bad_transform2,
        )
        self.assertRaises(
            ValidationTypeError,
            ArrowSchemaTransformModel,
            fields=self.bad_transform3,
        )

    def test_transform(self):
        expected_table = pa.Table.from_pydict(
            {
                "STR_FIELD": ["abc", "def", "ghi"],
                "FLOAT_FIELD": [1, 2, 3],
                "TIMESTAMP_FIELD": [
                    date(2023, 1, 1),
                    date(2023, 1, 2),
                    date(2023, 1, 3),
                ],
            }
        ).cast(self.expected_schema)

        # below call should:
        # 0. accept input data
        # 1. select requested fields only
        # 2. rename fields per transform schema
        # 3. cast fields per output schema
        # 4. return transformed arrow table
        output_table = ArrowTableTransform(
            table=self.data,
            transform_schema=ArrowSchemaTransformModel(fields=self.transform_list, metadata=self.metadata),
        ).table
        self.assertEqual(output_table, expected_table)

        output_table = ArrowTableTransform(
            table=self.data,
            transform_schema=ArrowSchemaTransformModel(fields=self.transform_dict, metadata=self.metadata),
        ).table
        self.assertEqual(output_table, expected_table)

    def test_empty_schema(self):
        # if no schema is provided then return teh table
        output_table = ArrowTableTransform(
            table=self.data,
        ).table
        self.assertEqual(output_table, pa.Table.from_pydict(self.data))


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
