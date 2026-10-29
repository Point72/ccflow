import pickle
import warnings
from datetime import timedelta
from unittest import TestCase

import pytest

from ccflow.exttypes.frequency import CalendarFrequency, Frequency, IntradayFrequency


class TestCanonicalContract(TestCase):
    """The canonical string is used as a cache key and path segment, so equal frequencies must serialize identically regardless of how they were
    supplied."""

    def test_cross_representation_canonical(self):
        pd = pytest.importorskip("pandas")
        groups = {
            "1h": ["60m", "3600s", "PT60M", "PT1H", timedelta(hours=1), pd.offsets.Minute(60), pd.offsets.Hour(1)],
            "24h": ["1440m", "86400s", "PT24H", pd.offsets.Hour(24)],
            "1d": ["P1D", timedelta(days=1), timedelta(hours=24), pd.offsets.Day(1), pd.DateOffset(days=1)],
            "1us": ["1000ns", pd.offsets.Nano(1000), pd.offsets.Micro(1)],
            "1mo": ["P1M", pd.DateOffset(months=1)],
            "1q": ["3mo", "P3M", pd.DateOffset(months=3)],
            "1y": ["12mo", "4q", pd.DateOffset(years=1)],
            "1w": ["P1W", pd.offsets.Week(1), pd.DateOffset(weeks=1)],
        }
        for canonical, inputs in groups.items():
            for value in inputs:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    self.assertEqual(str(Frequency.validate(value)), canonical, repr(value))

    def test_zero_is_canonical(self):
        self.assertEqual(str(Frequency.validate("0d")), "0s")
        self.assertEqual(str(Frequency.validate("0w")), "0s")
        self.assertEqual(str(Frequency.validate("0s")), "0s")

    def test_offset_roundtrip(self):
        pytest.importorskip("pandas")
        for value in ("5m", "1h", "1d", "1w", "1mo", "1q", "1y", "1ns"):
            f = Frequency.validate(value)
            self.assertEqual(Frequency.validate(f.offset), f)
            self.assertNotIn(b"pandas", pickle.dumps(f))


class TestFrequency(TestCase):
    def test_canonical_roundtrip(self):
        # As in polars, 7d and 1w bucket differently and a calendar day differs from 24 elapsed hours.
        for value in ("5m", "1h", "30s", "500ms", "1us", "1d", "24h", "7d", "2w", "1mo", "2q", "1y"):
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                self.assertEqual(str(Frequency.validate(value)), value)
        # Canonical units are coerced without a warning: reductions, signs, leading zeros and padding.
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            self.assertEqual(str(Frequency.validate("60m")), "1h")
            self.assertEqual(str(Frequency.validate("6mo")), "2q")
            self.assertEqual(str(Frequency.validate(" +01d ")), "1d")
        self.assertEqual((Frequency.validate("90m").n, Frequency.validate("90m").unit), (90, "m"))

    def test_legacy_pandas_aliases(self):
        cases = {
            "5min": "5m",
            "5T": "5m",
            "5N": "5ns",
            "1min": "1m",
            "1H": "1h",
            "1S": "1s",
            "1D": "1d",
        }
        for value, expected in cases.items():
            with pytest.warns(FutureWarning) as record:
                self.assertEqual(str(Frequency.validate(value)), expected)
            # The warning points at the caller, not at ccflow or pydantic.
            self.assertEqual(record[0].filename, __file__)

    def test_iso8601_input(self):
        cases = {"PT15M": "15m", "PT1H30M": "90m", "P3D": "3d", "P1W": "1w", "P1M": "1mo", "P2Y": "2y"}
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            for value, expected in cases.items():
                self.assertEqual(str(Frequency.validate(value)), expected)
        # Compound calendar durations are rejected rather than silently flattened.
        self.assertRaises(ValueError, Frequency.validate, "P1Y2M")
        self.assertRaises(ValueError, Frequency.validate, "P1WT0H")
        for value in ("P1D1D", "P1DT", "PT1.5H", "P1DT12H"):
            self.assertRaises(ValueError, Frequency.validate, value)

    def test_signed_iso_duration(self):
        self.assertEqual(str(Frequency.validate("-P1D")), "-1d")
        self.assertEqual(str(Frequency.validate("-PT15M")), "-15m")
        self.assertEqual(str(Frequency.validate("-PT36H")), "-36h")

    def test_ambiguous_M_raises(self):
        for value in ("M", "1M", "5M", "-1M"):
            with self.assertRaises(ValueError):
                Frequency.validate(value)

    def test_unsupported_units_and_anchors_rejected(self):
        # pandas period-end and anchored spellings have no equivalent, including uppercase Q/Y/W.
        for value in ("1ME", "1QE-DEC", "1YE", "1A-DEC", "1Q", "1Y", "1W", "1W-SUN", "1w-SUN", "1me"):
            self.assertRaises(ValueError, Frequency.validate, value)
        for value in ("1B", "MS", "QS", "1W-MON", "1QE-NOV", "1mon", "2 days", "1d12h", "５m", "foo", ""):
            self.assertRaises(ValueError, Frequency.validate, value)

    def test_timedelta_input(self):
        self.assertEqual(str(Frequency.validate(timedelta(minutes=5))), "5m")
        self.assertEqual(str(Frequency.validate(timedelta(days=14))), "14d")
        self.assertEqual(str(Frequency.validate(timedelta(milliseconds=500))), "500ms")
        self.assertEqual(str(Frequency.validate(timedelta(hours=36))), "36h")

    def test_pandas_timedelta_nanoseconds(self):
        pd = pytest.importorskip("pandas")
        self.assertEqual(str(Frequency.validate(pd.Timedelta(5, "ns"))), "5ns")
        self.assertEqual(str(Frequency.validate(pd.Timedelta(-1, "ns"))), "-1ns")

    def test_nanoseconds(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            self.assertEqual(str(Frequency.validate("1ns")), "1ns")
            self.assertEqual(str(Frequency.validate("100ns")), "100ns")
        self.assertEqual(str(Frequency.validate("1000ns")), "1us")
        self.assertEqual(str(IntradayFrequency.validate("100ns")), "100ns")
        self.assertRaises(ValueError, Frequency.validate, "1NS")
        # datetime.timedelta cannot hold nanoseconds.
        self.assertRaises(ValueError, lambda: Frequency.validate("1ns").timedelta)

    def test_bad_input(self):
        self.assertRaises(ValueError, Frequency.validate, None)

    def test_directly_constructed_iso_value(self):
        self.assertEqual(Frequency("PT5M").timedelta, timedelta(minutes=5))

    def test_timedelta_property(self):
        self.assertEqual(Frequency.validate("5m").timedelta, timedelta(minutes=5))
        self.assertEqual(Frequency.validate("1w").timedelta, timedelta(weeks=1))
        # Months, quarters and years have no fixed length.
        for value in ("1mo", "1q", "1y"):
            self.assertRaises(ValueError, lambda v=value: Frequency.validate(v).timedelta)
        self.assertRaises(ValueError, lambda: Frequency.validate("1000000000d").timedelta)

    def test_offset_matches_pandas(self):
        pd = pytest.importorskip("pandas")
        to_offset = pd.tseries.frequencies.to_offset
        self.assertEqual(Frequency.validate("5m").offset, to_offset("5min"))
        self.assertEqual(Frequency.validate("1h").offset, to_offset("h"))
        self.assertEqual(Frequency.validate("3d").offset, pd.offsets.Day(3))
        self.assertEqual(Frequency.validate("2w").offset, pd.offsets.Week(2))
        self.assertEqual(Frequency.validate("1mo").offset, pd.DateOffset(months=1))
        self.assertEqual(Frequency.validate("2q").offset.n, 2)

    def test_pandas_offset_input(self):
        pd = pytest.importorskip("pandas")
        self.assertEqual(str(Frequency.validate(pd.offsets.Minute(5))), "5m")
        self.assertEqual(str(Frequency.validate(pd.DateOffset(2, months=1))), "2mo")
        # Period-end, anchored and multi-keyword offsets have no equivalent and are rejected.
        for value in (pd.offsets.MonthEnd(), pd.offsets.QuarterEnd(), pd.offsets.YearEnd(), pd.offsets.Week(weekday=6), pd.offsets.BusinessDay()):
            self.assertRaises(ValueError, Frequency.validate, value)
        for value in (pd.DateOffset(months=1, days=1), pd.DateOffset(days=1.5), pd.DateOffset(month=1)):
            self.assertRaises(ValueError, Frequency.validate, value)
        self.assertRaises(ValueError, Frequency.validate, pd.DateOffset(months=1, normalize=True))


class TestIntradayFrequency(TestCase):
    def test_accepts_sub_day(self):
        self.assertEqual(str(IntradayFrequency.validate("15m")), "15m")
        self.assertEqual(str(IntradayFrequency.validate("23h")), "23h")
        self.assertEqual(IntradayFrequency.validate("15m").timedelta, timedelta(minutes=15))

    def test_rejects_non_intraday(self):
        for value in ("1d", "1w", "1mo", "1y", "24h", "1440m", "0m", "-5m"):
            self.assertRaises(ValueError, IntradayFrequency.validate, value)

    def test_field_validation(self):
        from pydantic import BaseModel, ValidationError

        class M(BaseModel):
            freq: IntradayFrequency

        self.assertIsInstance(M(freq="5m").freq, IntradayFrequency)
        self.assertIsInstance(pickle.loads(pickle.dumps(M(freq="5m").freq)), IntradayFrequency)
        self.assertRaises(ValidationError, M, freq="1mo")


class TestCalendarFrequency(TestCase):
    def test_accepts_calendar(self):
        for value, expected in {"1d": "1d", "2w": "2w", "1mo": "1mo", "1q": "1q", "1y": "1y"}.items():
            self.assertEqual(str(CalendarFrequency.validate(value)), expected)

    def test_rejects_non_calendar(self):
        for value in ("15m", "1h", "24h"):
            self.assertRaisesRegex(ValueError, "requires a d, w, mo, q or y unit", CalendarFrequency.validate, value)
        for value in ("0d", "0mo", "-1mo"):
            self.assertRaisesRegex(ValueError, "must be positive", CalendarFrequency.validate, value)
