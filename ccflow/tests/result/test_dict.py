from unittest import TestCase

from ccflow.result import DictResult


class TestResult(TestCase):
    def test_dict(self):
        context = DictResult[str, float].model_validate({"value": {"a": 0, "b": 1.1}})
        self.assertEqual(context.value, {"a": 0.0, "b": 1.1})
