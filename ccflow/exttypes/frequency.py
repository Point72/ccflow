"""This module contains a frequency extension type for pydantic."""

import warnings
from datetime import timedelta
from functools import cached_property
from typing import Any, Type

import pandas as pd
from pandas.tseries.frequencies import to_offset


class Frequency(str):
    """Represents a frequency string that can be converted to a pandas offset."""

    validate_always = True

    @cached_property
    def offset(self) -> Type:
        """Return the underlying pandas DateOffset object."""
        return to_offset(str(self))

    @cached_property
    def timedelta(self) -> timedelta:
        return pd.to_timedelta(self.offset).to_pytimedelta()

    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, value, field=None) -> Any:
        if isinstance(value, cls):
            return cls.validate(str(value))

        if isinstance(value, (timedelta, str)):
            try:
                with warnings.catch_warnings():
                    # Because pandas 2.2 deprecated many frequency strings (i.e. "Y", "M", "T" still in common use)
                    # We should consider switching away from pandas on this and supporting ISO
                    warnings.simplefilter("ignore", category=FutureWarning)
                    value = to_offset(value)
            except ValueError as e:
                raise ValueError(f"ensure this value can be converted to a pandas offset: {e}")

        if isinstance(value, pd.offsets.DateOffset):
            return cls(f"{value.n}{value.base.freqstr}")

        raise ValueError(f"ensure this value can be converted to a pandas offset: {value}")
