"""This module contains common validators."""

import calendar
import logging
from datetime import date, datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

from pydantic import TypeAdapter, ValidationError

from .exttypes import PyObjectPath
from .exttypes.frequency import Frequency, _coerce, _looks_like_offset

_DatetimeAdapter = TypeAdapter(datetime)
# Months in each month-based unit, for shifting a date.
_MONTHS = {"mo": 1, "q": 3, "y": 12}

__all__ = (
    "eval_or_load_object",
    "load_object",
    "normalize_date",
    "normalize_datetime",
    "str_to_log_level",
)


def _local_today() -> date:
    return datetime.now().astimezone().date()


def _shift(d: date, freq: Frequency) -> datetime:
    """Shift midnight on ``d`` by ``freq`` with polars ``offset_by`` semantics.

    Months, quarters and years keep the day of month, clamped to the end of a shorter month. Nanoseconds round down to the containing microsecond,
    because datetime has microsecond resolution.
    """
    try:
        if freq.unit in _MONTHS:
            year, month = divmod(d.year * 12 + d.month - 1 + freq.n * _MONTHS[freq.unit], 12)
            return datetime(year, month + 1, min(d.day, calendar.monthrange(year, month + 1)[1]))
        midnight = datetime(d.year, d.month, d.day)
        if freq.unit == "ns":
            return midnight + timedelta(microseconds=freq.n // 1000)
        return midnight + freq.timedelta
    except (OverflowError, ValueError):
        raise ValueError(f"frequency offset is out of range: {freq}") from None


def normalize_date(v: Any) -> Any:
    """Convert a relative date such as ``-1d`` or ``1mo`` to a date counted from today, and convert anything datetime-like to a date.

    Relative dates use the calendar units of ``CalendarFrequency`` (``d``, ``w``, ``mo``, ``q``, ``y``), with any sign. A string that looks like a
    frequency but is invalid, or uses a sub-day unit, raises.
    """
    if isinstance(v, str) and _looks_like_offset(v):
        # A frequency-like string (-1d, -1M, 1NS, 1h30min, P1D) is parsed as a frequency and raises if invalid; other strings go to the date parser.
        freq = _coerce(v)
        if freq.n != 0 and freq.unit not in ("d", "w", *_MONTHS):
            raise ValueError(f"a relative date needs a calendar unit (d, w, mo, q or y), got {v!r}; m is minutes and mo is months")
        return _shift(_local_today(), freq).date()
    # Convert from anything that can be converted to a datetime to a date via datetime
    # This is not normally allowed by pydantic.
    try:
        v = _DatetimeAdapter.validate_python(v)
        if isinstance(v, datetime):
            return v.date()
    except ValidationError:
        pass
    return v


def normalize_datetime(v: Any) -> Any:
    """Convert a relative datetime such as ``-1d`` or ``90m`` to a datetime counted from local midnight today, and validate anything datetime-like.

    A ``dict`` or ``list`` input holds a datetime and an optional time zone name. A string that looks like a frequency but is invalid raises.
    """
    if isinstance(v, str) and _looks_like_offset(v):
        return _shift(_local_today(), _coerce(v))
    if isinstance(v, dict):
        # e.g. DatetimeContext object, {"dt": datetime(...)}
        dt = next(iter(v.values()))
        tz = list(v.values())[1] if len(v) > 1 else None
    elif isinstance(v, list):
        dt = v[0]
        tz = v[1] if len(v) > 1 else None
    else:
        dt = v
        tz = None
    try:
        dt = TypeAdapter(datetime).validate_python(dt)
        if tz and isinstance(tz, str):
            tz = ZoneInfo(tz)
        if tz:
            dt = dt.astimezone(tz)
        return dt
    except ValidationError:
        return v


def load_object(v: Any) -> Any:
    """Validator that loads an object from path if a string is provided"""
    if isinstance(v, str):
        try:
            return PyObjectPath(v).object
        except (ImportError, ValidationError):
            pass
    return v


def eval_or_load_object(v: Any, values: dict[str, Any] | None = None) -> Any:
    """Validator that evaluates or loads an object from path if a string is provided.

    Useful for fields that could be either lambda functions or callables.
    """
    if isinstance(v, str):
        try:
            return eval(v, (values or {}).get("locals", {}))
        except NameError:
            if isinstance(v, str):
                return PyObjectPath(v).object
    return v


def str_to_log_level(v: Any) -> Any:
    """Validator to convert string to a log level."""
    if isinstance(v, str):
        return getattr(logging, v.upper())
    return v
