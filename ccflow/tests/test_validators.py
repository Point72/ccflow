import logging
from datetime import date, datetime, timedelta
from unittest import TestCase
from zoneinfo import ZoneInfo

from ccflow.validators import eval_or_load_object, load_object, normalize_date, normalize_datetime, str_to_log_level


class A:
    pass


class TestValidators(TestCase):
    def test_normalize_date(self):
        c = date.today()
        self.assertEqual(normalize_date(c), c)
        self.assertEqual(normalize_date("0d"), c)
        c1 = date.today() - timedelta(1)
        self.assertEqual(normalize_date("-1d"), c1)

        self.assertEqual(normalize_date(datetime.now()), c)
        self.assertEqual(normalize_date(datetime.now().isoformat()), c)

        self.assertEqual(normalize_date("foo"), "foo")
        self.assertEqual(normalize_date(None), None)

    def test_normalize_datetime(self):
        today = datetime.today()
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
