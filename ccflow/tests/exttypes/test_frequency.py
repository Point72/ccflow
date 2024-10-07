from datetime import timedelta
from unittest import TestCase

import pandas as pd
from packaging.version import parse
from pandas.tseries.frequencies import to_offset

from ccflow.exttypes.frequency import Frequency

IS_PD_22 = parse(pd.__version__) >= parse("2.2")


class TestFrequency(TestCase):
    def test_basic(self):
        f = Frequency("5min")
        self.assertIsInstance(f, str)
        self.assertEqual(f.offset, to_offset("5min"))
        self.assertEqual(f.timedelta, timedelta(minutes=5))

    def test_validate_bad(self):
        self.assertRaises(ValueError, Frequency.validate, None)
        self.assertRaises(ValueError, Frequency.validate, "foo")

    def test_validate_1D(self):
        f = Frequency("1D")
        self.assertEqual(Frequency.validate(f), f)
        self.assertEqual(Frequency.validate(str(f)), f)
        self.assertEqual(Frequency.validate(f.offset), f)
        self.assertEqual(Frequency.validate("1d"), f)
        self.assertEqual(Frequency.validate(Frequency("1d")), f)
        self.assertEqual(Frequency.validate(timedelta(days=1)), f)

    def test_validate_5T(self):
        if IS_PD_22:
            f = Frequency("5min")
        else:
            f = Frequency("5T")
        self.assertEqual(Frequency.validate(f), f)
        self.assertEqual(Frequency.validate(str(f)), f)
        self.assertEqual(Frequency.validate(f.offset), f)
        self.assertEqual(Frequency.validate("5T"), f)
        self.assertEqual(Frequency.validate("5min"), f)
        self.assertEqual(Frequency.validate(Frequency("5T")), f)
        self.assertEqual(Frequency.validate(Frequency("5min")), f)
        self.assertEqual(Frequency.validate(timedelta(minutes=5)), f)

    def test_validate_1M(self):
        if IS_PD_22:
            f = Frequency("1ME")
        else:
            f = Frequency("1M")
        self.assertEqual(Frequency.validate(f), f)
        self.assertEqual(Frequency.validate(str(f)), f)
        self.assertEqual(Frequency.validate(f.offset), f)
        self.assertEqual(Frequency.validate("1m"), f)
        self.assertEqual(Frequency.validate("1M"), f)
        self.assertEqual(Frequency.validate(Frequency("1m")), f)
        self.assertEqual(Frequency.validate(Frequency("1M")), f)

    def test_validate_1Y(self):
        if IS_PD_22:
            f = Frequency("1YE-DEC")
        else:
            f = Frequency("1A-DEC")
        self.assertEqual(Frequency.validate(f), f)
        self.assertEqual(Frequency.validate(str(f)), f)
        self.assertEqual(Frequency.validate(f.offset), f)
        self.assertEqual(Frequency.validate("1A-DEC"), f)
        self.assertEqual(Frequency.validate("1y"), f)
        self.assertEqual(Frequency.validate(Frequency("1A-DEC")), f)
        self.assertEqual(Frequency.validate(Frequency("1y")), f)
