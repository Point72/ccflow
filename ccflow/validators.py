"""This module contains common validators."""

import logging
from datetime import date, datetime
from typing import Any, Dict, Optional
from zoneinfo import ZoneInfo

import pandas as pd
from pydantic import TypeAdapter, ValidationError

from .exttypes import PyObjectPath

_DatetimeAdapter = TypeAdapter(datetime)

__all__ = (
    "normalize_date",
    "normalize_datetime",
    "load_object",
    "eval_or_load_object",
    "str_to_log_level",
)


def normalize_date(v: Any) -> Any:
    """Validator that will convert string offsets to date based on today, and convert datetime to date."""
    if isinstance(v, str):  # Check case where it's an offset
        try:
            timestamp = pd.tseries.frequencies.to_offset(v) + date.today()
            return timestamp.date()
        except ValueError:
            pass
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
    """Validator that will convert string offsets to datetime based on today, and convert datetime to date."""
    if isinstance(v, str):  # Check case where it's an offset
        try:
            return (pd.tseries.frequencies.to_offset(v) + date.today()).to_pydatetime()
        except ValueError:
            pass
    if isinstance(v, dict):
        # e.g. DatetimeContext object, {"dt": datetime(...)}
        dt = list(v.values())[0]
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


def eval_or_load_object(v: Any, values: Optional[Dict[str, Any]] = None) -> Any:
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
