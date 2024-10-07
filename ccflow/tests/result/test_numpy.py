from unittest import TestCase

import numpy as np

from ccflow.result.numpy import NumpyResult


class TestResult(TestCase):
    def test_numpy(self):
        x = np.array([1.0, 3.0])
        r = NumpyResult[np.float64](array=x)
        np.testing.assert_equal(r.array, x)

        # Check you can also construct from list
        r = NumpyResult[np.float64](array=x.tolist())
        np.testing.assert_equal(r.array, x)

        self.assertRaises(TypeError, NumpyResult[np.float64], np.array(["foo"]))

        # Test generic
        r = NumpyResult[object](array=x)
        np.testing.assert_equal(r.array, x)
        r = NumpyResult[object](array=[None, "foo", 4.0])
        np.testing.assert_equal(r.array, np.array([None, "foo", 4.0]))
