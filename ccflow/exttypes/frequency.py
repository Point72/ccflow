"""This module contains a frequency extension type for pydantic."""

import warnings
from datetime import timedelta
from functools import cached_property
from typing import Any, Type

import pandas as pd
from pandas.tseries.frequencies import to_offset
from pydantic_core import core_schema


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
    def __get_pydantic_core_schema__(cls, source_type, handler):
        return core_schema.no_info_plain_validator_function(cls._validate)

    @classmethod
    def _validate(cls, value) -> Any:
        if isinstance(value, cls):
            return cls._validate(str(value))

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
