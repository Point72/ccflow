import importlib
import inspect
import re
from datetime import date, datetime, timedelta, timezone
from unittest import TestCase

import pandas as pd
from pydantic import TypeAdapter, ValidationError

import ccflow.context as ctx
from ccflow.context import (
    ContextBase,
    DateContext,
    DateRangeContext,
    DatetimeContext,
    FreqContext,
    FreqDateContext,
    FreqDateRangeContext,
    GenericContext,
    ModelContext,
    ModelDateContext,
    ModelDateRangeContext,
    ModelDateRangeSourceContext,
    ModelFreqDateRangeContext,
    NullContext,
    UniverseContext,
    UniverseDateContext,
    UniverseDateRangeContext,
)
from ccflow.result import GenericResult


class TestContexts(TestCase):
    def test_null_context(self):
        n1 = NullContext()
        n2 = NullContext()
        self.assertEqual(n1, n2)
        self.assertEqual(hash(n1), hash(n2))

    def test_null_context_validation(self):
        self.assertEqual(NullContext.model_validate([]), NullContext())
        self.assertEqual(NullContext.model_validate({}), NullContext())
        self.assertEqual(NullContext.model_validate(None), NullContext())
        self.assertRaises(ValueError, NullContext.model_validate, [True])

    def test_date_validation(self):
        c = DateContext(date=date.today())
        self.assertEqual(DateContext(date=str(date.today())), c)
        self.assertEqual(DateContext(date=pd.Timestamp(date.today())), c)
        self.assertEqual(DateContext(date="0d"), c)
        c1 = DateContext(date=date.today() - timedelta(1))
        self.assertEqual(DateContext(date="-1d"), c1)
        self.assertRaises(ValueError, DateContext, date="foo")

        # Test validation from other types
        self.assertEqual(TypeAdapter(DateContext).validate_python({"date": date.today()}), c)
        self.assertEqual(TypeAdapter(DateContext).validate_python(date.today()), c)
        self.assertEqual(TypeAdapter(DateContext).validate_python([date.today()]), c)
        self.assertEqual(TypeAdapter(DateContext).validate_python(str(date.today())), c)
        self.assertEqual(TypeAdapter(DateContext).validate_python("0d"), c)
        self.assertEqual(TypeAdapter(DateContext).validate_python("-1d"), c1)
        self.assertRaises(ValidationError, TypeAdapter(DateContext).validate_python, "foo")
        self.assertRaises(ValidationError, TypeAdapter(DateContext).validate_python, None)

    def test_date_from_datetime_validation(self):
        dt = datetime(2022, 1, 1, 12, tzinfo=timezone.utc)
        self.assertEqual(TypeAdapter(DateContext).validate_python(dt), DateContext(date=dt.date()))
        self.assertEqual(TypeAdapter(DateContext).validate_python(dt.isoformat()), DateContext(date=dt.date()))

    def test_datetime_validation(self):
        dt = datetime(2022, 1, 1, 12, 0, tzinfo=timezone.utc)
        c = DatetimeContext(dt=dt)
        self.assertEqual(DatetimeContext(dt=str(dt)), c)
        self.assertEqual(DatetimeContext(dt=dt.date()), DatetimeContext(dt=datetime(2022, 1, 1)))

        # Test validation from other types
        self.assertEqual(TypeAdapter(DatetimeContext).validate_python({"dt": dt}), c)
        self.assertEqual(TypeAdapter(DatetimeContext).validate_python(dt), c)
        self.assertEqual(TypeAdapter(DatetimeContext).validate_python([dt]), c)
        self.assertEqual(TypeAdapter(DatetimeContext).validate_python(str(dt)), c)
        self.assertEqual(TypeAdapter(DatetimeContext).validate_python(dt.isoformat()), c)
        self.assertRaises(ValueError, TypeAdapter(DatetimeContext).validate_python, "foo")

    def test_coercion(self):
        d = DateContext(date=date(2022, 1, 1))
        f = FreqDateContext(freq="5T", date=date(2022, 1, 1))
        self.assertEqual(DateContext.model_validate(f), f)
        self.assertRaises(ValidationError, FreqDateContext.model_validate, d)

    def test_date_range(self):
        d0 = date.today() - timedelta(1)
        d1 = date.today()
        c = DateRangeContext(start_date=d0, end_date=d1)
        self.assertEqual(DateRangeContext(start_date=str(d0), end_date=pd.Timestamp(date.today())), c)
        self.assertEqual(DateRangeContext(start_date="-1d", end_date="0d"), c)
        self.assertRaises(ValueError, DateRangeContext, start_date="foo", end_date=d1)

        # Test validation from other types
        self.assertEqual(TypeAdapter(DateRangeContext).validate_python({"start_date": d0, "end_date": d1}), c)
        self.assertEqual(TypeAdapter(DateRangeContext).validate_python(("-1d", "0d")), c)
        self.assertEqual(TypeAdapter(DateRangeContext).validate_python(["-1d", "0d"]), c)
        self.assertEqual(TypeAdapter(DateRangeContext).validate_python(["-1d", datetime.now()]), c)

    def test_date_range_from_date(self):
        # Test validation from a DateContext
        d0 = date.today()
        self.assertEqual(TypeAdapter(DateRangeContext).validate_python(DateContext(date=d0)), DateRangeContext(start_date=d0, end_date=d0))

    def test_freq(self):
        self.assertEqual(
            FreqDateContext.model_validate("5min,2022-01-01"),
            FreqDateContext(freq="5T", date=date(2022, 1, 1)),
        )
        self.assertEqual(
            FreqDateRangeContext.model_validate("5min,2022-01-01,2022-02-01"),
            FreqDateRangeContext(freq="5T", start_date=date(2022, 1, 1), end_date=date(2022, 2, 1)),
        )

    def test_universe(self):
        self.assertEqual(
            UniverseDateContext.model_validate("US,2022-01-01"),
            UniverseDateContext(universe="US", date=date(2022, 1, 1)),
        )
        self.assertEqual(
            UniverseDateRangeContext.model_validate("US,2022-01-01,2022-02-01"),
            UniverseDateRangeContext(universe="US", start_date=date(2022, 1, 1), end_date=date(2022, 2, 1)),
        )

    def test_model(self):
        self.assertEqual(
            ModelDateContext.model_validate("EULTS,2022-01-01"),
            ModelDateContext(model="EULTS", date=date(2022, 1, 1)),
        )
        self.assertEqual(
            ModelDateRangeContext.model_validate("EULTS,2022-01-01,2022-02-01"),
            ModelDateRangeContext(model="EULTS", start_date=date(2022, 1, 1), end_date=date(2022, 2, 1)),
        )
        self.assertEqual(
            ModelFreqDateRangeContext.model_validate("EULTS,2 min,2022-01-01,2022-02-01"),
            ModelFreqDateRangeContext(
                model="EULTS",
                freq="2T",
                start_date=date(2022, 1, 1),
                end_date=date(2022, 2, 1),
            ),
        )

    def test_model_source(self):
        self.assertEqual(
            ModelDateRangeSourceContext.model_validate("USE4S,2022-01-01,2022-02-01,barra"),
            ModelDateRangeSourceContext(model="USE4S", source="barra", start_date=date(2022, 1, 1), end_date=date(2022, 2, 1)),
        )
        self.assertEqual(
            ModelDateRangeSourceContext.model_validate("USE4S,2022-01-01,2022-02-01"),
            ModelDateRangeSourceContext(model="USE4S", source=None, start_date=date(2022, 1, 1), end_date=date(2022, 2, 1)),
        )

    def test_list_scalar_consistency(self):
        """Test that for the contexts with one field, validation from scalar and list of length 1 is consistent."""
        test_cases = [(UniverseContext, "US"), (FreqContext, "1D"), (DateContext, date(2024, 1, 1)), (ModelContext, "USFASTD")]
        for context_type, v in test_cases:
            context_from_scalar = context_type.model_validate(v)
            context_from_list = context_type.model_validate([v])
            self.assertEqual(context_from_scalar, context_from_list)


class TestGenericResult(TestCase):
    def test_generic(self):
        v = {"a": 1, "b": [2, 3]}
        result = GenericResult(value=v)
        self.assertEqual(GenericResult.model_validate(v), result)
        self.assertIs(GenericResult.model_validate(result), result)

        v = {"value": 5}
        self.assertEqual(GenericResult.model_validate(v), GenericResult(value=5))
        self.assertEqual(GenericResult[int].model_validate(v), GenericResult[int](value=5))
        self.assertEqual(GenericResult[str].model_validate(v), GenericResult[str](value="5"))

        self.assertEqual(GenericResult.model_validate("foo"), GenericResult(value="foo"))
        self.assertEqual(GenericResult[str].model_validate(5), GenericResult[str](value="5"))

        result = GenericResult(value=5)
        # Note that this will work, even though GenericResult is not a subclass of GenericResult[str]
        self.assertEqual(GenericResult[str].model_validate(result), GenericResult[str](value="5"))


class TestGenericContext(TestCase):
    def test_generic_context(self):
        v = (1, [2, 3], {4, 5, 6})
        result = GenericContext(value=v)
        self.assertEqual(GenericContext.model_validate(v), result)

        v = {"value": 5}
        self.assertEqual(GenericContext.model_validate(v), GenericContext(value=5))
        self.assertEqual(GenericContext[int].model_validate(v), GenericContext[int](value=5))
        self.assertEqual(GenericContext[str].model_validate(v), GenericContext[str](value="5"))

        self.assertEqual(GenericContext.model_validate("foo"), GenericContext(value="foo"))
        self.assertEqual(GenericContext[str].model_validate(5), GenericContext[str](value="5"))

        result = GenericContext(value=5)
        # Note that this will work, even though GenericContext is not a subclass of GenericContext[str]
        self.assertEqual(GenericContext[str].model_validate(result), GenericContext[str](value="5"))

    def test_generics_conversion(self):
        v = (1, [2, 3], {4, 5, 6})
        self.assertEqual(GenericContext(value=GenericResult(value=v)), GenericContext(value=v))

        v = 5
        self.assertEqual(GenericContext[str](value=GenericResult(value=v)), GenericContext[str](value=v))
        self.assertEqual(GenericContext[str](value=GenericResult[str](value=v)), GenericContext[str](value=v))
        self.assertEqual(GenericContext[int](value=GenericResult[str](value=v)), GenericContext[int](value=v))
        self.assertEqual(GenericContext[int](value=GenericResult[int](value=v)), GenericContext[int](value=v))

        v = "5"
        self.assertEqual(GenericContext[str](value=GenericResult(value=v)), GenericContext[str](value=v))
        self.assertEqual(GenericContext[str](value=GenericResult[str](value=v)), GenericContext[str](value=v))
        self.assertEqual(GenericContext[int](value=GenericResult[str](value=v)), GenericContext[int](value=v))
        self.assertEqual(GenericContext[int](value=GenericResult[int](value=v)), GenericContext[int](value=v))

        self.assertEqual(GenericContext[str].model_validate(GenericResult(value=5)), GenericContext[str](value="5"))


class TestContextInheritance(TestCase):
    def setUp(self):
        self.classes = {
            name: obj
            for name, obj in inspect.getmembers(ctx, inspect.isclass)
            if obj.__module__ == ctx.__name__ and issubclass(obj, ContextBase) and not getattr(obj, "__deprecated__", False)
        }

    def test_field_ordering(self):
        """Test that complex contexts have fields in the same order as the basic contexts they are composed of."""

        def split_camel(name: str):
            return re.findall(r"[A-Z][a-z]*", name)

        basic_contexts_map = {}
        complex_contexts_map = {}

        for name, cls in self.classes.items():
            name_noralized = name.replace("Context", "").replace("EntryTime", "Entrytime").replace("Range", "range")
            if ContextBase in cls.__bases__:
                basic_contexts_map[name_noralized] = list(cls.model_fields.keys())
            else:
                complex_contexts_map[name_noralized] = list(cls.model_fields.keys())

        for complex_context, complex_context_fields in complex_contexts_map.items():
            expected_fields = []
            for basic_context in split_camel(complex_context):
                expected_fields.extend(basic_contexts_map[basic_context])
            self.assertEqual(
                expected_fields, complex_context_fields, f"{expected_fields} do not match {complex_context_fields} for context {complex_context}."
            )

    def test_inheritance(self):
        """Test that if a context has a superset of fields of another context, it is a subclass of that context."""

        for parent_name, parent_class in self.classes.items():
            for child_name, child_class in self.classes.items():
                if parent_class is child_class:
                    continue

                parent_fields = set(parent_class.model_fields.keys())
                child_fields = set(child_class.model_fields.keys())
                if parent_fields.issubset(child_fields):
                    self.assertTrue(
                        issubclass(child_class, parent_class),
                        f"ERROR: {child_name} has a superset of {parent_name}'s fields but is NOT a subclass of {parent_name}",
                    )


class TestDeprecated(TestCase):
    # TODO - remove once deprecated contexts are removed
    def test_deprecated(self):
        """
        Tests that all deprecated context classes are:
        1. importable,
        2. marked with __deprecated__ = True,
        3. accompanied by a DeprecationWarning when instantiated.
        """

        deprecated_class_names = [
            "SeededContext",
            "SeededDateRangeContext",
            "SeededDatetimeRangeContext",
            "UniverseFrequencyDateRangeContext",
            "UniverseFrequencyDatetimeRangeContext",
            "UniverseFrequencyHorizonDateRangeContext",
            "UniverseFrequencyHorizonDatetimeRangeContext",
            "VersionedUniverseDateContext",
            "VersionedUniverseDatetimeContext",
            "VersionedUniverseDateRangeContext",
            "VersionedUniverseDatetimeRangeContext",
            "VersionedModelDateContext",
            "VersionedModelDatetimeContext",
            "VersionedModelDateRangeContext",
            "VersionedModelDatetimeRangeContext",
            "VersionedDateContext",
            "VersionedDatetimeContext",
            "VersionedDateRangeContext",
            "VersionedDatetimeRangeContext",
        ]

        module = importlib.import_module("ccflow.context")
        for cls_name in deprecated_class_names:
            cls = getattr(module, cls_name)
            self.assertTrue(cls.__deprecated__)
            with self.assertWarns(DeprecationWarning):
                cls.model_construct()
