"""This module contains common validators."""

import logging
from datetime import date, datetime
from typing import Any, Dict, Optional

import pandas as pd
from pydantic import TypeAdapter, ValidationError

from .exttypes import PyObjectPath

_DatetimeAdapter = TypeAdapter(datetime)


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
