import re
import warnings
from datetime import timedelta
from functools import cached_property
from typing import Type

import pandas as pd
from pandas.tseries.frequencies import to_offset
from pydantic import TypeAdapter
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
    def _validate(cls, value) -> "Frequency":
        if isinstance(value, cls):
            return cls._validate(str(value))

        if isinstance(value, timedelta):
            # Keep ccflow's day-sized spellings stable when pandas round-trips
            # timedeltas through its newer canonical offset machinery.
            # Context for the offset alias changes:
            # https://pandas.pydata.org/docs/dev/whatsnew/v3.0.0.html#enforced-deprecation-of-aliases-m-q-y-etc-in-favour-of-me-qe-ye-etc-for-offsets
            if value.total_seconds() % 86400 == 0:
                return cls(f"{int(value.total_seconds() // 86400)}D")

        if isinstance(value, str):
            # Pandas deprecated legacy aliases like M/Q/Y in 2.2 and enforced the
            # new ME/QE/YE forms in 3.0. Normalize old user-facing inputs before
            # handing them to pandas.
            # Exact sections:
            # https://pandas.pydata.org/docs/whatsnew/v2.2.0.html#deprecate-aliases-m-q-y-etc-in-favour-of-me-qe-ye-etc-for-offsets
            # https://pandas.pydata.org/docs/dev/whatsnew/v3.0.0.html#enforced-deprecation-of-aliases-m-q-y-etc-in-favour-of-me-qe-ye-etc-for-offsets
            value = _normalize_frequency_alias(value)

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

    @classmethod
    def validate(cls, value) -> "Frequency":
        """Try to convert/validate an arbitrary value to a Frequency."""
        return _TYPE_ADAPTER.validate_python(value)


_TYPE_ADAPTER = TypeAdapter(Frequency)


_LEGACY_FREQ_PATTERN = re.compile(
    r"^(?P<count>[+-]?\d+)?(?P<unit>T|M|A|Y)(?:-(?P<suffix>[A-Za-z]{3}))?$",
    re.IGNORECASE,
)


def _normalize_frequency_alias(value: str) -> str:
    # Pandas 2.2 deprecated, and pandas 3.0 enforced removal of, several
    # legacy offset aliases. Keep accepting the old forms at the ccflow API
    # boundary and translate them to pandas' canonical spellings here.
    # Exact sections:
    # https://pandas.pydata.org/docs/whatsnew/v2.2.0.html#deprecate-aliases-m-q-y-etc-in-favour-of-me-qe-ye-etc-for-offsets
    # https://pandas.pydata.org/docs/dev/whatsnew/v3.0.0.html#enforced-deprecation-of-aliases-m-q-y-etc-in-favour-of-me-qe-ye-etc-for-offsets
    normalized = value.strip()
    if not normalized:
        return normalized

    match = _LEGACY_FREQ_PATTERN.fullmatch(normalized)
    if not match:
        day_match = re.fullmatch(r"(?P<count>[+-]?\d+)?d", normalized, re.IGNORECASE)
        if day_match:
            return f"{day_match.group('count') or 1}D"
        return normalized

    count = match.group("count") or "1"
    unit = match.group("unit").upper()
    suffix = (match.group("suffix") or "DEC").upper()
    replacements = {
        "T": f"{count}min",
        "M": f"{count}ME",
        "A": f"{count}YE-{suffix}",
        "Y": f"{count}YE-{suffix}",
    }
    return replacements[unit]
