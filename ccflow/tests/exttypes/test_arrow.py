from unittest import TestCase

import pandas as pd
import polars as pl
import pyarrow as pa
from packaging import version
from pydantic import ValidationError

from ccflow import BaseModel
from ccflow.exttypes.arrow import ArrowSchema, ArrowTable, PyArrowDatatype

RAW_SCHEMA = pa.schema({"a": pa.int32(), "b": pa.float32(), "c": pa.utf8()})
RAW_SCHEMA_FULL = pa.schema(
    {
        "a": pa.int32(),
        "b": pa.float32(),
        "c": pa.utf8(),
        "d": pa.int64(),
    }
)

SCHEMA_WEAK = ArrowSchema.make(RAW_SCHEMA, strict=False)
SCHEMA_FILTER = ArrowSchema.make(RAW_SCHEMA, strict="filter")
SCHEMA_STRICT = ArrowSchema.make(RAW_SCHEMA, strict=True)


class Model(BaseModel):
    table: ArrowTable = None
    weak: ArrowTable[SCHEMA_WEAK] = None
    filter: ArrowTable[SCHEMA_FILTER] = None
    strict: ArrowTable[SCHEMA_STRICT] = None


class TestArrowSchema(TestCase):
    def test_init(self):
        self.assertIsInstance(SCHEMA_WEAK, type)
        self.assertIsInstance(SCHEMA_WEAK.schema, pa.Schema)
        self.assertEqual(SCHEMA_FILTER.strict, "filter")

        # You can't actually construct the schema type (because the schema is on the type, not the instance)
        self.assertRaises(TypeError, SCHEMA_WEAK)

    def test_validate(self):
        # Validation failures should raise
        self.assertRaises(ValueError, ArrowSchema.validate, "foo")
        self.assertRaises(ValueError, ArrowSchema.validate, 20.0)
        self.assertRaises(ValueError, ArrowSchema.validate, False)

        # Validation of a pyarrow schema should provide an ArrowSchema
        pyarrow_schema = pa.schema({"A": pa.float64(), "B": pa.int64(), "C": pa.string()})
        result = ArrowSchema.validate(pyarrow_schema)
        self.assertIsInstance(result, ArrowSchema)
        self.assertTrue(pyarrow_schema.equals(result.schema))

        # Validation of an ArrowSchema should provide an ArrowSchema
        schema = ArrowSchema.make(pyarrow_schema, strict=True)
        result = ArrowSchema.validate(schema)
        self.assertIsInstance(result, ArrowSchema)
        self.assertEqual(schema, result)


class TestArrowTable(TestCase):
    def setUp(self) -> None:
        self.data = {
            "a": [1.0, 2.0, 3.0],
            "b": [4, 5, 6],
            "c": ["foo", "bar", "baz"],
            "d": [0, 0, 0],
        }

    def test_table_dict(self):
        data = self.data
        m = Model(table=data, filter=data, weak=data)
        self.assertEqual(m.table, pa.Table.from_pydict(data))
        self.assertEqual(m.weak, pa.Table.from_pydict(data, RAW_SCHEMA_FULL))
        self.assertEqual(m.filter, pa.Table.from_pydict(data, RAW_SCHEMA))

        self.assertEqual(m.table.column_names, ["a", "b", "c", "d"])
        self.assertEqual(m.weak.column_names, ["a", "b", "c", "d"])
        self.assertEqual(m.filter.column_names, ["a", "b", "c"])

        self.assertRaises(ValidationError, Model, strict=data)
        data.pop("d")
        m = Model(strict=data)
        self.assertEqual(m.strict, pa.Table.from_pydict(data, RAW_SCHEMA))

    def test_table_pandas(self):
        df = pd.DataFrame(self.data)
        m = Model(table=df, filter=df, weak=df)
        self.assertEqual(m.table, pa.Table.from_pandas(df))
        self.assertEqual(m.weak, pa.Table.from_pandas(df, RAW_SCHEMA_FULL))
        self.assertEqual(m.filter, pa.Table.from_pandas(df, RAW_SCHEMA))

        self.assertEqual(m.table.column_names, ["a", "b", "c", "d"])
        self.assertEqual(m.weak.column_names, ["a", "b", "c", "d"])
        self.assertEqual(m.filter.column_names, ["a", "b", "c"])

        self.assertRaises(ValidationError, Model, strict=df)
        self.data.pop("d")
        df = pd.DataFrame(self.data)
        m = Model(strict=df)
        self.assertEqual(m.strict, pa.Table.from_pandas(df, RAW_SCHEMA))

    def test_table_polars(self):
        df = pl.DataFrame(self.data)
        m = Model(table=df, filter=df, weak=df)
        self.assertEqual(m.table, df.to_arrow())
        self.assertEqual(m.weak, df.to_arrow().cast(RAW_SCHEMA_FULL))
        self.assertEqual(m.filter, df.to_arrow().drop(["d"]).cast(RAW_SCHEMA))

        self.assertEqual(m.table.column_names, ["a", "b", "c", "d"])
        self.assertEqual(m.weak.column_names, ["a", "b", "c", "d"])
        self.assertEqual(m.filter.column_names, ["a", "b", "c"])

        self.assertRaises(ValidationError, Model, strict=df)
        self.data.pop("d")
        df = pl.DataFrame(self.data)
        m = Model(strict=df)
        self.assertEqual(m.strict, df.to_arrow().cast(RAW_SCHEMA))

    def test_table_arrow(self):
        t = pa.Table.from_pydict(self.data)
        m = Model(table=t, filter=t, weak=t)
        self.assertEqual(m.table, t)
        self.assertEqual(m.weak, t.cast(RAW_SCHEMA_FULL))
        self.assertEqual(m.filter, t.drop(["d"]).cast(RAW_SCHEMA))

        self.assertEqual(m.table.column_names, ["a", "b", "c", "d"])
        self.assertEqual(m.weak.column_names, ["a", "b", "c", "d"])
        self.assertEqual(m.filter.column_names, ["a", "b", "c"])

        self.assertRaises(ValidationError, Model, strict=t)
        self.data.pop("d")
        t = pa.Table.from_pydict(self.data)
        m = Model(strict=t)
        self.assertEqual(m.strict, t.cast(RAW_SCHEMA))

    def test_record_batches(self):
        data = self.data
        batch = pa.RecordBatch.from_pydict(data)
        m = Model(table=[batch], filter=[batch], weak=[batch])
        self.assertEqual(m.table, pa.Table.from_pydict(data))
        self.assertEqual(m.weak, pa.Table.from_pydict(data, RAW_SCHEMA_FULL))
        self.assertEqual(m.filter, pa.Table.from_pydict(data, RAW_SCHEMA))

        self.assertEqual(m.table.column_names, ["a", "b", "c", "d"])
        self.assertEqual(m.weak.column_names, ["a", "b", "c", "d"])
        self.assertEqual(m.filter.column_names, ["a", "b", "c"])

        self.assertRaises(ValidationError, Model, strict=[batch])
        data.pop("d")
        batch = pa.RecordBatch.from_pydict(data)
        m = Model(strict=[batch])
        self.assertEqual(m.strict, pa.Table.from_pydict(data, RAW_SCHEMA))

    def test_bad_type(self):
        t = pa.Table.from_pydict(self.data)
        data_bad = {"a": ["oops", 2, 3], "b": [4, 5, 6], "c": ["foo", "bar", "baz"]}
        if version.parse(pl.__version__) < version.parse("1.0.0"):
            t_bad = pl.DataFrame(data_bad).to_arrow()
        else:
            # polars raises a TypeError on encountering mixed datatypes by default.
            t_bad = pl.DataFrame(data_bad, strict=False).to_arrow()
        self.assertRaises(ValidationError, Model, table=t, filter=t_bad, weak=t)
        self.assertRaises(ValidationError, Model, table=t, filter=t, weak=t_bad)

        m = Model(table=t_bad, filter=t, weak=t)
        self.assertEqual(m.table, t_bad)

    def test_missing_column(self):
        t = pa.Table.from_pydict(self.data)
        data_bad = {"a": [1, 2, 3], "b": [4, 5, 6]}  # Missing column
        t_bad = pl.DataFrame(data_bad).to_arrow()
        self.assertRaises(ValidationError, Model, table=t, filter=t_bad, weak=t)
        self.assertRaises(ValidationError, Model, table=t, filter=t, weak=t_bad)

        m = Model(table=t_bad, filter=t, weak=t)
        self.assertEqual(m.table, t_bad)

    def test_out_of_order(self):
        t = pa.Table.from_pydict(self.data)
        data_bad = {"c": self.data["c"], "a": self.data["a"], "b": self.data["c"]}
        t_bad = pl.DataFrame(data_bad).to_arrow()

        self.assertRaises(ValidationError, Model, table=t, filter=t_bad, weak=t)
        self.assertRaises(ValidationError, Model, table=t, filter=t, weak=t_bad)

        m = Model(table=t_bad, filter=t, weak=t)
        self.assertEqual(m.table, t_bad)


class TestPyArrowDatatype(TestCase):
    def test_validate(self):
        # test validation logic to accpet pa.lib.DataType or a string representation of it
        self.assertRaises(ValueError, PyArrowDatatype.validate, True)
        self.assertRaises(ValueError, PyArrowDatatype.validate, "foo")
        self.assertRaises(ValueError, PyArrowDatatype.validate, 7)
        self.assertRaises(ValueError, PyArrowDatatype.validate, "pa.string")
        self.assertIsInstance(PyArrowDatatype.validate("pa.string()"), PyArrowDatatype)
        self.assertIsInstance(PyArrowDatatype.validate(pa.int32()), pa.lib.DataType)
