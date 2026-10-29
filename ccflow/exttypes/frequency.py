"""Frequency types for sampling intervals and calendar periods, following the polars duration language.

The canonical string form is an integer count followed by a lowercase unit, e.g. ``5m`` (five minutes), ``1h``, ``1d``, ``1mo`` (one month), ``1q``
(one quarter), ``1y`` (one year). The units mean what they mean in polars: ``ns``, ``us``, ``ms``, ``s``, ``m`` and ``h`` are fixed durations, while
``d``, ``w``, ``mo``, ``q`` and ``y`` are calendar units. ``1d`` (a calendar day) and ``24h`` (24 elapsed hours) are therefore distinct values.

Canonical units are case-sensitive: ``m`` is minute, and ``M`` is rejected as ambiguous (use ``mo`` for month). The pandas spellings ``T``, ``min``,
``H``, ``S``, ``L``, ``U``, ``N`` and ``D`` are accepted in any case with a :class:`FutureWarning`. ISO-8601 durations (``PT5M``, ``P1M``), timedeltas
and pandas offsets are also accepted and normalized to the canonical form. pandas period-end, anchored and business-day spellings (``ME``, ``QE-DEC``,
``W-SUN``, ``B``) are rejected because they have no equivalent here.
"""

from __future__ import annotations

import numbers
import os
import re
import sys
import warnings
from datetime import timedelta
from functools import cache, cached_property

import pydantic
from pydantic import TypeAdapter
from pydantic_core import core_schema

__all__ = ("CalendarFrequency", "Frequency", "IntradayFrequency")

# Each canonical unit as (kind, size, pandas offset class). Fixed sizes are in nanoseconds, day sizes in days and month sizes in months. Within a
# kind the order is coarse to fine, so ``_reduce`` can pick the coarsest unit that divides a value exactly.
_UNITS: dict[str, tuple[str, int, str | None]] = {
    "y": ("month", 12, None),
    "q": ("month", 3, None),
    "mo": ("month", 1, None),
    "w": ("day", 7, "Week"),
    "d": ("day", 1, "Day"),
    "h": ("fixed", 3_600_000_000_000, "Hour"),
    "m": ("fixed", 60_000_000_000, "Minute"),
    "s": ("fixed", 1_000_000_000, "Second"),
    "ms": ("fixed", 1_000_000, "Milli"),
    "us": ("fixed", 1_000, "Micro"),
    "ns": ("fixed", 1, "Nano"),
}
_DAY_NS = 86_400_000_000_000

# pandas spellings accepted with a warning, keyed by the lower-cased token. ``m``, ``ms``, ``us`` and ``ns`` are left out so only their exact
# lowercase forms match: ``M`` is ambiguous and ``MS`` is pandas month-start.
_ALIASES: dict[str, str] = {"t": "m", "min": "m", "h": "h", "s": "s", "d": "d", "l": "ms", "u": "us", "n": "ns"}
# pandas units with no equivalent here (period ends, anchored weeks, business days and hours, period starts, semi-months). They are rejected like
# any unknown unit, and listed only so that ``normalize_date`` raises for them instead of reading them as dates.
_PANDAS_ONLY_UNITS: frozenset[str] = frozenset(
    {"ME", "QE", "YE", "A", "Q", "Y", "W", "B", "C", "BH", "CBH", "BM", "BME", "CBM", "CBME", "MS", "BMS", "CBMS", "SM", "SME", "SMS", "BQ", "BQE",
     "QS", "BQS", "BA", "BY", "BYE", "AS", "YS", "BAS", "BYS"}
)  # fmt: skip
_KNOWN_TOKENS: frozenset[str] = frozenset(_UNITS) | frozenset(_ALIASES) | frozenset(u.lower() for u in _PANDAS_ONLY_UNITS)

_OFFSET_RE = re.compile(r"(?P<n>[+-]?\d+)?\s*(?P<unit>[A-Za-z]+)(?:-(?P<anchor>[A-Za-z]{1,4}))?", re.ASCII)
# Several count-and-unit pairs (``1h30min``): not a valid frequency, but recognised so it is reported rather than read as a date.
_COMPOUND_RE = re.compile(r"[+-]?(?:\d+\s*[A-Za-z]+\s*)+", re.ASCII)
_ISO_SHAPE_RE = re.compile(r"[+-]?P(?=[0-9T])", re.ASCII)
# ISO-8601 durations with integer components: either one date part (``P1D``) or time parts only (``PT1H30M``).
_ISO_DATE_RE = re.compile(r"([+-]?)P(\d+)([YMWD])", re.ASCII)
_ISO_TIME_RE = re.compile(r"([+-]?)PT(?=\d)(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", re.ASCII)
_ISO_DATE_UNITS: dict[str, str] = {"Y": "y", "M": "mo", "W": "w", "D": "d"}

_DATEOFFSET_UNITS: dict[str, str] = {"days": "d", "weeks": "w", "months": "mo", "years": "y"}
_PANDAS_OFFSET_UNITS: dict[str, str] = {name: unit for unit, (_, _, name) in _UNITS.items() if name}

# Directories skipped when attributing a warning, so it points at the caller: ccflow itself (apart from its tests) and pydantic's validation.
_CCFLOW_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + os.sep
_INTERNAL_DIRS = (_CCFLOW_DIR, os.path.dirname(pydantic.__file__) + os.sep)
_CCFLOW_TESTS_DIR = os.path.join(_CCFLOW_DIR, "tests") + os.sep


def _looks_like_offset(value: str) -> bool:
    """Whether a string is a frequency unit with an optional count (or an ISO-8601 duration), as opposed to a date-like string such as ``15Mar``."""
    value = value.strip()
    if _ISO_SHAPE_RE.match(value):
        return True
    if match := _OFFSET_RE.fullmatch(value):
        return match.group("unit").lower() in _KNOWN_TOKENS
    return _COMPOUND_RE.fullmatch(value) is not None and all(t.lower() in _KNOWN_TOKENS for t in re.findall("[A-Za-z]+", value))


def _reduce(n: int, unit: str) -> tuple[int, str]:
    """Collapse to one canonical spelling so equal values serialize identically: ``60m`` becomes ``1h`` and ``3mo`` becomes ``1q``.

    Fixed units reduce up to hours, never into days, and month-based units reduce among ``mo``/``q``/``y``. Days and weeks are kept as written because
    they bucket differently (``7d`` and ``1w`` chunk on different boundaries). Zero is ``0s`` whatever its unit.
    """
    if n == 0:
        return 0, "s"
    kind, size, _ = _UNITS[unit]
    if kind == "day":
        return n, unit
    total = n * size
    return next((total // s, u) for u, (k, s, _) in _UNITS.items() if k == kind and total % s == 0)


def _from_iso(value: str) -> tuple[int, str]:
    if match := _ISO_DATE_RE.fullmatch(value):
        sign, n, letter = match.groups()
        return _reduce(int(sign + n), _ISO_DATE_UNITS[letter])
    if match := _ISO_TIME_RE.fullmatch(value):
        sign, *counts = match.groups()
        total = sum(int(count or 0) * _UNITS[unit][1] for count, unit in zip(counts, ("h", "m", "s")))
        return _reduce(-total if sign == "-" else total, "ns")
    raise ValueError(f"invalid ISO-8601 duration {value!r}; use one date part such as P1D or time parts only such as PT1H30M")


def _parse(value: str) -> tuple[int, str]:
    """Parse a duration literal or ISO-8601 duration into ``(count, canonical_unit)``."""
    text = value.strip()
    if _ISO_SHAPE_RE.match(text):
        return _from_iso(text)
    match = _OFFSET_RE.fullmatch(text)
    if match and not match.group("anchor"):
        token = match.group("unit")
        unit = token if token in _UNITS else _ALIASES.get(token.lower())
        if unit is not None:
            return _reduce(int(match.group("n") or 1), unit)
    raise ValueError(f"cannot parse frequency {value!r}; write a count and one of ns, us, ms, s, m, h, d, w, mo, q or y (m is minute, mo is month)")


def _from_timedelta(value: timedelta) -> tuple[int, str]:
    """Whole days become ``d``; anything else is a fixed duration such as ``36h``."""
    # A pandas Timedelta subclasses timedelta and adds a sub-microsecond ``nanoseconds`` component.
    total_ns = value // timedelta(microseconds=1) * 1000 + getattr(value, "nanoseconds", 0)
    days, remainder = divmod(total_ns, _DAY_NS)
    return _reduce(days, "d") if remainder == 0 else _reduce(total_ns, "ns")


def _from_pandas_offset(value) -> tuple[int, str]:
    name = type(value).__name__
    if value.normalize:
        raise ValueError("pandas offsets with normalize=True are not supported")
    if name == "DateOffset" and len(value.kwds) == 1:
        [(key, count)] = value.kwds.items()
        if key in _DATEOFFSET_UNITS and isinstance(count, numbers.Integral) and not isinstance(count, bool):
            return _reduce(value.n * int(count), _DATEOFFSET_UNITS[key])
    elif name in _PANDAS_OFFSET_UNITS and value.kwds.get("weekday") is None:
        return _reduce(value.n, _PANDAS_OFFSET_UNITS[name])
    raise ValueError(
        f"unsupported pandas offset {value!r}; use Nano through Hour, Day, Week() or DateOffset with one of days, weeks, months or years"
    )


def _warn(message: str) -> None:
    """Emit a ``FutureWarning`` attributed to the first frame outside ccflow and pydantic."""
    frame, level = sys._getframe(1), 2
    while frame is not None and frame.f_code.co_filename.startswith(_INTERNAL_DIRS) and not frame.f_code.co_filename.startswith(_CCFLOW_TESTS_DIR):
        frame, level = frame.f_back, level + 1
    warnings.warn(message, FutureWarning, stacklevel=level)


def _coerce(value: object) -> Frequency:
    """Coerce an arbitrary input to a canonical ``Frequency``, warning on pandas spellings."""
    if isinstance(value, Frequency):
        n, unit = value._parts
    elif isinstance(value, timedelta):
        n, unit = _from_timedelta(value)
    elif (pd := sys.modules.get("pandas")) is not None and isinstance(value, pd.offsets.BaseOffset):  # no pandas offset exists unless it's imported
        n, unit = _from_pandas_offset(value)
    elif isinstance(value, str):
        n, unit = _parse(value)
        # Only a pandas unit spelling warns; signs, leading zeros, padding and reductions are coerced silently.
        if (match := _OFFSET_RE.fullmatch(value.strip())) and match.group("unit") not in _UNITS:
            _warn(f"{value!r} is a deprecated pandas frequency spelling; ccflow uses polars duration units, so write {f'{n}{unit}'!r}")
    else:
        raise ValueError(f"cannot convert {value!r} to a frequency")
    return Frequency(f"{n}{unit}")


class Frequency(str):
    """A sampling frequency such as ``5m``, ``1h`` or ``1mo``.

    Accepts every unit; use :class:`IntradayFrequency` or :class:`CalendarFrequency` to restrict a field to sub-day or calendar frequencies.
    """

    def __reduce__(self):
        # Pickle only the string, not the cached properties, so the pandas offset does not travel with the value.
        return type(self), (str(self),)

    @cached_property
    def _parts(self) -> tuple[int, str]:
        return _parse(str(self))

    @property
    def n(self) -> int:
        """Number of units, e.g. ``90`` for ``90m``."""
        return self._parts[0]

    @property
    def unit(self) -> str:
        """Canonical unit: one of ``ns``, ``us``, ``ms``, ``s``, ``m``, ``h``, ``d``, ``w``, ``mo``, ``q`` or ``y``."""
        return self._parts[1]

    @cached_property
    def timedelta(self) -> timedelta:
        """Equivalent timedelta; raises for months, quarters, years and nanoseconds. Days and weeks give their nominal 24-hour and 7-day length."""
        n, unit = self._parts
        kind, size, _ = _UNITS[unit]
        if kind == "month" or unit == "ns":
            raise ValueError(f"{str(self)!r} has no fixed datetime.timedelta")
        try:
            return timedelta(days=n * size) if kind == "day" else timedelta(microseconds=n * size // 1000)
        except OverflowError:
            raise ValueError(f"{str(self)!r} is too large to represent as a timedelta") from None

    @cached_property
    def offset(self):
        """Equivalent pandas offset for adding to a timestamp (requires pandas); its ``n`` is the count.

        ``ns`` to ``h`` give ``Nano`` to ``Hour``, ``d`` gives ``Day`` and ``w`` gives ``Week``; pandas 2 treats ``Day`` as 24 hours, where polars and
        pandas 3 use a calendar day. ``mo``, ``q`` and ``y`` give ``DateOffset(n, months=k)``, which is not a pandas frequency: ``resample`` rejects
        it, ``date_range`` applies it step by step so month-end dates drift (Jan 31, Feb 29, Mar 29), and pandas 1 also clamps the day at each step
        when adding a count above 1 to a single timestamp.
        """
        from pandas import offsets

        n, unit = self._parts
        _, size, name = _UNITS[unit]
        return getattr(offsets, name)(n) if name else offsets.DateOffset(n, months=size)

    @classmethod
    def __get_pydantic_core_schema__(cls, source_type, handler):
        return core_schema.no_info_plain_validator_function(cls._validate)

    @classmethod
    def _check_allowed(cls, n: int, unit: str) -> None:
        """Raise ``ValueError`` if this type does not accept ``n`` steps of ``unit``; subclasses narrow the grammar here."""

    @classmethod
    def _validate(cls, value) -> Frequency:
        freq = _coerce(value)
        cls._check_allowed(freq.n, freq.unit)
        return cls(freq)

    @classmethod
    def validate(cls, value) -> Frequency:
        """Validate/convert an arbitrary value to this frequency type."""
        return _type_adapter(cls).validate_python(value)


class IntradayFrequency(Frequency):
    """A fixed sub-day frequency: a positive duration shorter than one day."""

    @classmethod
    def _check_allowed(cls, n: int, unit: str) -> None:
        kind, size, _ = _UNITS[unit]
        if kind != "fixed":
            raise ValueError(f"IntradayFrequency requires a sub-day unit (ns/us/ms/s/m/h), got {n}{unit}")
        if not 0 < n * size < _DAY_NS:
            raise ValueError(f"IntradayFrequency must be positive and under one day, got {n}{unit}")


class CalendarFrequency(Frequency):
    """A calendar frequency: a positive count of days, weeks, months, quarters or years."""

    @classmethod
    def _check_allowed(cls, n: int, unit: str) -> None:
        if n <= 0:
            raise ValueError(f"CalendarFrequency must be positive, got {n}{unit}")
        if _UNITS[unit][0] == "fixed":
            raise ValueError(f"CalendarFrequency requires a d, w, mo, q or y unit, got {n}{unit}")


@cache
def _type_adapter(cls: type) -> TypeAdapter:
    return TypeAdapter(cls)
