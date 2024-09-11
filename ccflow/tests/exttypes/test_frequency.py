from datetime import timedelta
from unittest import TestCase

import pandas as pd
from packaging.version import parse
from pandas.tseries.frequencies import to_offset
from pydantic import TypeAdapter

from ccflow.exttypes.frequency import Frequency

IS_PD_22 = parse(pd.__version__) >= parse("2.2")


class TestFrequency(TestCase):
    def test_basic(self):
        f = Frequency("5min")
        self.assertIsInstance(f, str)
        self.assertEqual(f.offset, to_offset("5min"))
        self.assertEqual(f.timedelta, timedelta(minutes=5))

    def test_validate_bad(self):
        ta = TypeAdapter(Frequency)
        self.assertRaises(ValueError, ta.validate_python, None)
        self.assertRaises(ValueError, ta.validate_python, "foo")

    def test_validate_1D(self):
        f = Frequency("1D")
        ta = TypeAdapter(Frequency)
        self.assertEqual(ta.validate_python(f), f)
        self.assertEqual(ta.validate_python(str(f)), f)
        self.assertEqual(ta.validate_python(f.offset), f)
        self.assertEqual(ta.validate_python("1d"), f)
        self.assertEqual(ta.validate_python(Frequency("1d")), f)
        self.assertEqual(ta.validate_python(timedelta(days=1)), f)

    def test_validate_5T(self):
        if IS_PD_22:
            f = Frequency("5min")
        else:
            f = Frequency("5T")
        ta = TypeAdapter(Frequency)
        self.assertEqual(ta.validate_python(f), f)
        self.assertEqual(ta.validate_python(str(f)), f)
        self.assertEqual(ta.validate_python(f.offset), f)
        self.assertEqual(ta.validate_python("5T"), f)
        self.assertEqual(ta.validate_python("5min"), f)
        self.assertEqual(ta.validate_python(Frequency("5T")), f)
        self.assertEqual(ta.validate_python(Frequency("5min")), f)
        self.assertEqual(ta.validate_python(timedelta(minutes=5)), f)

    def test_validate_1M(self):
        if IS_PD_22:
            f = Frequency("1ME")
        else:
            f = Frequency("1M")
        ta = TypeAdapter(Frequency)
        self.assertEqual(ta.validate_python(f), f)
        self.assertEqual(ta.validate_python(str(f)), f)
        self.assertEqual(ta.validate_python(f.offset), f)
        self.assertEqual(ta.validate_python("1m"), f)
        self.assertEqual(ta.validate_python("1M"), f)
        self.assertEqual(ta.validate_python(Frequency("1m")), f)
        self.assertEqual(ta.validate_python(Frequency("1M")), f)

    def test_validate_1Y(self):
        if IS_PD_22:
            f = Frequency("1YE-DEC")
        else:
            f = Frequency("1A-DEC")
        ta = TypeAdapter(Frequency)
        self.assertEqual(ta.validate_python(f), f)
        self.assertEqual(ta.validate_python(str(f)), f)
        self.assertEqual(ta.validate_python(f.offset), f)
        self.assertEqual(ta.validate_python("1A-DEC"), f)
        self.assertEqual(ta.validate_python("1y"), f)
        self.assertEqual(ta.validate_python(Frequency("1A-DEC")), f)
        self.assertEqual(ta.validate_python(Frequency("1y")), f)
