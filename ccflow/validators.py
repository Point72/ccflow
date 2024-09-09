"""This module contains common validators."""

import logging
from datetime import date

import pandas as pd
from pydantic import ValidationError

from .exttypes import PyObjectPath


def normalize_date(v):
    """Validator that will convert string offsets to date based on today."""
    if isinstance(v, str):  # Check case where it's an offset
        try:
            timestamp = pd.tseries.frequencies.to_offset(v) + date.today()
            return timestamp.date()
        except ValueError:
            pass
    return v


def load_object(v):
    """Validator that loads an object from path if a string is provided"""
    if isinstance(v, str):
        try:
            return PyObjectPath(v).object
        except (ImportError, ValidationError):
            pass
    return v


def eval_or_load_object(v, values=None):
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


def str_to_enum(v, field):
    """Validator for enum fields that will try to convert a string value to the enum type."""
    if isinstance(v, str):
        return field.type_[v]
    return v


def str_to_log_level(v):
    """Validator to convert string to a log level."""
    if isinstance(v, str):
        return getattr(logging, v.upper())
    return v
