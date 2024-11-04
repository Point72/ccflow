from datetime import date, timedelta
from unittest import TestCase

import pandas as pd
from pydantic import BaseModel, ValidationError

from ccflow.context import (
    DateContext,
    DateRangeContext,
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


class MyModel(BaseModel):
    context: DateContext


class MyRangeModel(BaseModel):
    context: DateRangeContext


class TestContexts(TestCase):
    def test_null_context(self):
        n1 = NullContext()
        n2 = NullContext()
        self.assertEqual(n1, n2)
        self.assertEqual(hash(n1), hash(n2))

    def test_date_validation(self):
        c = DateContext(date=date.today())
        self.assertEqual(DateContext(date=str(date.today())), c)
        self.assertEqual(DateContext(date=pd.Timestamp(date.today())), c)
        self.assertEqual(DateContext(date="0d"), c)
        c1 = DateContext(date=date.today() - timedelta(1))
        self.assertEqual(DateContext(date="-1d"), c1)
        self.assertRaises(ValueError, DateContext, date="foo")

        # Test coercion to DateContext on nested models
        self.assertEqual(MyModel(context={"date": date.today()}).context, c)
        self.assertEqual(MyModel(context=date.today()).context, c)
        self.assertEqual(MyModel(context=str(date.today())).context, c)
        self.assertEqual(MyModel(context="0d").context, c)
        self.assertEqual(MyModel(context="-1d").context, c1)
        self.assertRaises(ValueError, MyModel, context="foo")

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

        # Test coercion to DateContext on nested models
        self.assertEqual(MyRangeModel(context={"start_date": d0, "end_date": d1}).context, c)
        self.assertEqual(MyRangeModel(context=("-1d", "0d")).context, c)
        self.assertEqual(MyRangeModel(context=["-1d", "0d"]).context, c)

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

        self.assertEqual(GenericContext[str].validate(GenericResult(value=5)), GenericContext[str](value="5"))
