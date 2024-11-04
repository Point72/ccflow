import logging
from datetime import date, timedelta
from unittest import TestCase

from ccflow.validators import eval_or_load_object, load_object, normalize_date, str_to_log_level


class A:
    pass


class TestValidators(TestCase):
    def test_normalize_date(self):
        c = date.today()
        self.assertEqual(normalize_date("0d"), c)
        c1 = date.today() - timedelta(1)
        self.assertEqual(normalize_date("-1d"), c1)

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
