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


_PD_GE_22 = tuple(int(x) for x in pd.__version__.split(".")[:2]) >= (2, 2)

_LEGACY_FREQ_PATTERN = re.compile(
    r"^(?P<count>[+-]?\d+)?(?P<unit>T|M|A|Y)(?:-(?P<suffix>[A-Za-z]{3}))?$",
    re.IGNORECASE,
)

_CANONICAL_FREQ_PATTERN = re.compile(
    r"^(?P<count>[+-]?\d+)?(?P<unit>ME|QE|YE)(?:-(?P<suffix>[A-Za-z]{3}))?$",
    re.IGNORECASE,
)


def _normalize_frequency_alias(value: str) -> str:
    # Pandas 2.2 deprecated, and pandas 3.0 enforced removal of, several
    # legacy offset aliases. Keep accepting both old and new forms at the
    # ccflow API boundary and translate them to the correct spellings for the
    # installed pandas version.
    # Exact sections:
    # https://pandas.pydata.org/docs/whatsnew/v2.2.0.html#deprecate-aliases-m-q-y-etc-in-favour-of-me-qe-ye-etc-for-offsets
    # https://pandas.pydata.org/docs/dev/whatsnew/v3.0.0.html#enforced-deprecation-of-aliases-m-q-y-etc-in-favour-of-me-qe-ye-etc-for-offsets
    normalized = value.strip()
    if not normalized:
        return normalized

    if _PD_GE_22:
        # Pandas >= 2.2 deprecated legacy aliases (M, Q, Y, A, T).
        # Convert them to the canonical forms (ME, QE, YE, min).
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
    else:
        # Pandas < 2.2 doesn't understand the canonical aliases (ME, QE, YE).
        # Convert them back to legacy forms that old pandas accepts.
        match = _CANONICAL_FREQ_PATTERN.fullmatch(normalized)
        if match:
            count = match.group("count") or "1"
            unit = match.group("unit").upper()
            suffix = match.group("suffix")
            if unit == "ME":
                return f"{count}M"
            elif unit == "QE":
                return f"{count}Q" + (f"-{suffix.upper()}" if suffix else "")
            elif unit == "YE":
                return f"{count}A" + (f"-{suffix.upper()}" if suffix else "-DEC")

        # Also normalize case for legacy aliases.
        match = _LEGACY_FREQ_PATTERN.fullmatch(normalized)
        if match:
            count = match.group("count") or "1"
            unit = match.group("unit").upper()
            suffix = (match.group("suffix") or "DEC").upper()
            replacements = {
                "T": f"{count}T",
                "M": f"{count}M",
                "A": f"{count}A-{suffix}",
                "Y": f"{count}A-{suffix}",
            }
            return replacements[unit]

        day_match = re.fullmatch(r"(?P<count>[+-]?\d+)?d", normalized, re.IGNORECASE)
        if day_match:
            return f"{day_match.group('count') or 1}D"

        return normalized
