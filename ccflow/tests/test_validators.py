import logging
from calendar import monthrange
from datetime import date, datetime, timedelta
from unittest import TestCase, mock
from zoneinfo import ZoneInfo

import pytest

from ccflow import Frequency
from ccflow.validators import _shift, eval_or_load_object, load_object, normalize_date, normalize_datetime, str_to_log_level


class A:
    pass


class TestValidators(TestCase):
    def test_normalize_date(self):
        c = datetime.now().astimezone().date()
        self.assertEqual(normalize_date(c), c)
        self.assertEqual(normalize_date("0d"), c)
        c1 = datetime.now().astimezone().date() - timedelta(1)
        self.assertEqual(normalize_date("-1d"), c1)
        with pytest.warns(FutureWarning):
            self.assertEqual(normalize_date("D"), normalize_date("1D"))

        self.assertEqual(normalize_date(datetime.now()), c)
        self.assertEqual(normalize_date(datetime.now().isoformat()), c)

        self.assertEqual(normalize_date("foo"), "foo")
        self.assertEqual(normalize_date(None), None)

    def test_normalize_date_frequency_errors_surface(self):
        # Frequency-like strings with an invalid or sub-day unit raise instead of falling through to the date parser.
        for bad in ("-1M", "-1ME", "-1B", "1MS", "1bh", "1NS", "-6m", "1h", "1h30min", "1000000000d"):
            self.assertRaises(ValueError, normalize_date, bad)
        # Other strings, including ones that start with a number but have no frequency unit, fall through to the date parser.
        self.assertEqual(normalize_date("15Mar"), "15Mar")
        self.assertEqual(normalize_date("2022-01-01"), date(2022, 1, 1))

    def test_normalize_datetime(self):
        today = datetime.today()  # noqa: DTZ002
        now = datetime.now()
        c = datetime(today.year, today.month, today.day)

        self.assertEqual(normalize_datetime(c), c)
        self.assertEqual(normalize_datetime("0d"), c)

        c1 = c - timedelta(1)
        self.assertEqual(normalize_datetime("-1d"), c1)

        self.assertEqual(normalize_datetime(now), now)

        # check passthrough validation error
        self.assertEqual(normalize_datetime("foo"), "foo")
        self.assertEqual(normalize_datetime(None), None)

        # check dict
        self.assertEqual(
            normalize_datetime({"dt": now.isoformat(), "tz": "US/Hawaii"}),
            now.astimezone(tz=ZoneInfo("US/Hawaii")),
        )
        # check list
        self.assertEqual(
            normalize_datetime([now.isoformat(), "US/Hawaii"]),
            now.astimezone(tz=ZoneInfo("US/Hawaii")),
        )

    def test_calendar_offsets_keep_day_of_month(self):
        with mock.patch("ccflow.validators._local_today", return_value=date(2024, 3, 31)):
            self.assertEqual(normalize_date("-1mo"), date(2024, 2, 29))
            self.assertEqual(normalize_date("P1M"), date(2024, 4, 30))
            self.assertEqual(normalize_datetime("-1y"), datetime(2023, 3, 31))
            # Nanoseconds floor to the containing microsecond.
            self.assertEqual(normalize_datetime("-1ns"), datetime(2024, 3, 30, 23, 59, 59, 999999))

    def test_calendar_offsets_match_polars(self):
        pl = pytest.importorskip("polars")
        dates = [date(y, m, d) for y in (2000, 2023, 2024) for m in (1, 2, 3, 12) for d in (1, 15, 28, 29, 30, 31) if d <= monthrange(y, m)[1]]
        for value in ("1mo", "-1mo", "2q", "-1q", "1y", "-3y", "13mo", "3d", "-2w"):
            f = Frequency.validate(value)
            expected = pl.Series(dates).dt.offset_by(value).to_list()
            self.assertEqual([_shift(d, f).date() for d in dates], expected, value)

    def test_load_object(self):
        self.assertEqual(load_object("ccflow.tests.test_validators.A"), A)
        self.assertIsNone(load_object(None))
        self.assertEqual(load_object(5), 5)

        # Special case, if the object to load is string, you might want to load it from an object path
        # or you might want to provide it explicitly. Thus, if no object path found, return the value
        self.assertEqual(load_object("foo"), "foo")

    def test_eval_or_load_object(self):
        f1 = eval_or_load_object("lambda x: x+1")
        self.assertEqual(f1(2), 3)
        self.assertEqual(eval_or_load_object("A", {"locals": {"A": A}}), A)

        self.assertEqual(eval_or_load_object("ccflow.tests.test_validators.A"), A)
        self.assertIsNone(eval_or_load_object(None))
        self.assertEqual(eval_or_load_object(5), 5)

    def test_str_to_log_level(self):
        self.assertEqual(str_to_log_level("INFO"), logging.INFO)
        self.assertEqual(str_to_log_level("debug"), logging.DEBUG)
        self.assertEqual(str_to_log_level(logging.WARNING), logging.WARNING)
