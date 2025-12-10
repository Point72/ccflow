"""This module defines re-usable contexts for the "Callable Model" framework defined in flow.callable.py."""

import warnings
from datetime import date, datetime
from typing import Generic, Hashable, Optional, Sequence, Set, TypeVar

from deprecated import deprecated
from pydantic import field_validator, model_validator

from .base import ContextBase
from .exttypes import Frequency
from .validators import normalize_date, normalize_datetime

warnings.simplefilter("always", DeprecationWarning)


__all__ = (
    "NullContext",
    "GenericContext",
    "DateContext",
    "DatetimeContext",
    "EntryTimeContext",
    "SeededContext",
    "SourceContext",
    "DateRangeContext",
    "DatetimeRangeContext",
    "SeededDateRangeContext",
    "SeededDatetimeRangeContext",
    "DateVersionedContext",
    "DatetimeVersionedContext",
    "DateRangeVersionedContext",
    "DatetimeRangeVersionedContext",
    "FreqContext",
    "FreqDateContext",
    "FreqDatetimeContext",
    "FreqDateRangeContext",
    "FreqDatetimeRangeContext",
    "HorizonContext",
    "FreqHorizonContext",
    "FreqHorizonDateContext",
    "FreqHorizonDatetimeContext",
    "FreqHorizonDateRangeContext",
    "FreqHorizonDatetimeRangeContext",
    "UniverseContext",
    "UniverseDateContext",
    "UniverseDatetimeContext",
    "UniverseDateRangeContext",
    "UniverseDatetimeRangeContext",
    "UniverseFreqDateRangeContext",
    "UniverseFreqDatetimeRangeContext",
    "UniverseFreqHorizonDateRangeContext",
    "UniverseFreqHorizonDatetimeRangeContext",
    "UniverseDateVersionedContext",
    "UniverseDatetimeVersionedContext",
    "UniverseDateRangeVersionedContext",
    "UniverseDatetimeRangeVersionedContext",
    "ModelContext",
    "ModelDateContext",
    "ModelDatetimeContext",
    "ModelDateRangeContext",
    "ModelDatetimeRangeContext",
    "ModelDateRangeSourceContext",
    "ModelFreqDateRangeContext",
    "ModelFreqDatetimeRangeContext",
    "ModelDateVersionedContext",
    "ModelDatetimeVersionedContext",
    "ModelDateRangeVersionedContext",
    "ModelDatetimeRangeVersionedContext",
    # deprecated aliases
    "UniverseFrequencyDateRangeContext",
    "UniverseFrequencyDatetimeRangeContext",
    "UniverseFrequencyHorizonDateRangeContext",
    "UniverseFrequencyHorizonDatetimeRangeContext",
    "VersionedDateContext",
    "VersionedDatetimeContext",
    "VersionedDateRangeContext",
    "VersionedDatetimeRangeContext",
    "VersionedUniverseDateContext",
    "VersionedUniverseDatetimeContext",
    "VersionedUniverseDateRangeContext",
    "VersionedUniverseDatetimeRangeContext",
    "VersionedModelDateContext",
    "VersionedModelDatetimeContext",
    "VersionedModelDateRangeContext",
    "VersionedModelDatetimeRangeContext",
)

_SEPARATOR = ","


class NullContext(ContextBase):
    """A Null Context that is used when no context is provided."""

    @model_validator(mode="wrap")
    def _validate_none(cls, v, handler, info):
        v = v or {}
        return handler(v)


C = TypeVar("C", bound=Hashable)


class GenericContext(ContextBase, Generic[C]):
    """Holds anything."""

    value: C

    @model_validator(mode="wrap")
    def _validate_generic_context(cls, v, handler, info):
        if isinstance(v, GenericContext) and not isinstance(v, cls):
            v = {"value": v.value}
        elif not isinstance(v, GenericContext) and not (isinstance(v, dict) and "value" in v):
            v = {"value": v}
        if isinstance(v, dict) and "value" in v:
            from .result import GenericResult

            if isinstance(v["value"], GenericResult):
                v["value"] = v["value"].value
            if isinstance(v["value"], Sequence) and not isinstance(v["value"], Hashable):
                v["value"] = tuple(v["value"])
            if isinstance(v["value"], Set) and not isinstance(v["value"], Hashable):
                v["value"] = frozenset(v["value"])
        return handler(v)


class DateContext(ContextBase):
    date: date

    # validators
    _normalize_date = field_validator("date", mode="before")(normalize_date)

    @model_validator(mode="wrap")
    def _date_context_validator(cls, v, handler, info):
        if cls is DateContext and not isinstance(v, (DateContext, dict)):
            if isinstance(v, (tuple, list)) and len(v) == 1:
                v = v[0]

            v = DateContext(date=v)
        return handler(v)


class DatetimeContext(ContextBase):
    dt: datetime

    # validators
    _normalize_dt = field_validator("dt", mode="before")(normalize_datetime)

    @model_validator(mode="wrap")
    def _datetime_context_validator(cls, v, handler, info):
        if cls is DatetimeContext and not isinstance(v, (DatetimeContext, dict)):
            if isinstance(v, (tuple, list)) and len(v) == 1:
                v = v[0]

            v = DatetimeContext(dt=v)
        return handler(v)


class EntryTimeContext(ContextBase):
    entry_time_cutoff: Optional[datetime] = None


class SeededContext(ContextBase):
    seed: int = 1234


class SourceContext(ContextBase):
    source: Optional[str] = None


class DateRangeContext(ContextBase):
    start_date: date
    end_date: date

    _normalize_start = field_validator("start_date", mode="before")(normalize_date)
    _normalize_end = field_validator("end_date", mode="before")(normalize_date)

    @model_validator(mode="wrap")
    def _date_context_validator(cls, v, handler, info):
        if isinstance(v, DateContext):
            v = dict(start_date=v.date, end_date=v.date)
        return handler(v)


class DatetimeRangeContext(ContextBase):
    start_datetime: datetime
    end_datetime: datetime

    _normalize_start = field_validator("start_datetime", mode="before")(normalize_datetime)
    _normalize_end = field_validator("end_datetime", mode="before")(normalize_datetime)


class SeededDateRangeContext(DateRangeContext, SeededContext):
    pass


class SeededDatetimeRangeContext(DatetimeRangeContext, SeededContext):
    pass


class DateVersionedContext(EntryTimeContext, DateContext):
    pass


class DatetimeVersionedContext(EntryTimeContext, DatetimeContext):
    pass


class DateRangeVersionedContext(EntryTimeContext, DateRangeContext):
    pass


class DatetimeRangeVersionedContext(EntryTimeContext, DatetimeRangeContext):
    pass


class FreqContext(ContextBase):
    freq: Frequency


class FreqDateContext(DateContext, FreqContext):
    pass


class FreqDatetimeContext(DatetimeContext, FreqContext):
    pass


class FreqDateRangeContext(DateRangeContext, FreqContext):
    pass


class FreqDatetimeRangeContext(DatetimeRangeContext, FreqContext):
    pass


class HorizonContext(ContextBase):
    horizon: Frequency


class HorizonDateContext(DateContext, HorizonContext):
    pass


class HorizonDatetimeContext(DatetimeContext, HorizonContext):
    pass


class HorizonDateRangeContext(DateRangeContext, HorizonContext):
    pass


class HorizonDatetimeRangeContext(DatetimeRangeContext, HorizonContext):
    pass


class FreqHorizonContext(HorizonContext, FreqContext):
    pass


class FreqHorizonDateContext(HorizonDateContext, FreqDateContext, FreqHorizonContext):
    pass


class FreqHorizonDatetimeContext(HorizonDatetimeContext, FreqDatetimeContext, FreqHorizonContext):
    pass


class FreqHorizonDateRangeContext(HorizonDateRangeContext, FreqDateRangeContext, FreqHorizonContext):
    pass


class FreqHorizonDatetimeRangeContext(HorizonDatetimeRangeContext, FreqDatetimeRangeContext, FreqHorizonContext):
    pass


class UniverseContext(ContextBase):
    universe: str


class UniverseDateContext(DateContext, UniverseContext):
    pass


class UniverseDatetimeContext(DatetimeContext, UniverseContext):
    pass


class UniverseDateRangeContext(DateRangeContext, UniverseContext):
    pass


class UniverseDatetimeRangeContext(DatetimeRangeContext, UniverseContext):
    pass


class UniverseFreqDateRangeContext(FreqDateRangeContext, UniverseDateRangeContext):
    pass


class UniverseFreqDatetimeRangeContext(FreqDatetimeRangeContext, UniverseDatetimeRangeContext):
    pass


class UniverseFreqHorizonDateRangeContext(FreqHorizonDateRangeContext, UniverseFreqDateRangeContext):
    pass


class UniverseFreqHorizonDatetimeRangeContext(UniverseFreqDatetimeRangeContext, FreqHorizonDatetimeRangeContext):
    pass


class UniverseDateVersionedContext(DateVersionedContext, UniverseDateContext):
    pass


class UniverseDatetimeVersionedContext(DatetimeVersionedContext, UniverseDatetimeContext):
    pass


class UniverseDateRangeVersionedContext(DateRangeVersionedContext, UniverseDateRangeContext):
    pass


class UniverseDatetimeRangeVersionedContext(DatetimeRangeVersionedContext, UniverseDatetimeRangeContext):
    pass


class ModelContext(ContextBase):
    model: str


class ModelDateContext(DateContext, ModelContext):
    pass


class ModelDatetimeContext(DatetimeContext, ModelContext):
    pass


class ModelDateRangeContext(DateRangeContext, ModelContext):
    pass


class ModelDatetimeRangeContext(DatetimeRangeContext, ModelContext):
    pass


class ModelDateRangeSourceContext(SourceContext, ModelDateRangeContext):
    pass


class ModelFreqDateRangeContext(FreqDateRangeContext, ModelDateRangeContext):
    pass


class ModelFreqDatetimeRangeContext(FreqDatetimeRangeContext, ModelDatetimeRangeContext):
    pass


class ModelDateVersionedContext(DateVersionedContext, ModelDateContext):
    pass


class ModelDatetimeVersionedContext(DatetimeVersionedContext, ModelDatetimeContext):
    pass


class ModelDateRangeVersionedContext(DateRangeVersionedContext, ModelDateRangeContext):
    pass


class ModelDatetimeRangeVersionedContext(DatetimeRangeVersionedContext, ModelDatetimeRangeContext):
    pass


# TODO - remove later to avoid breaking changes for now
@deprecated(version="0.8.0", reason="Use UniverseFreqDateRangeContext instead")
class UniverseFrequencyDateRangeContext(UniverseFreqDateRangeContext):
    __deprecated__ = True
    pass


@deprecated(version="0.8.0", reason="Use UniverseFreqDatetimeRangeContext instead")
class UniverseFrequencyDatetimeRangeContext(UniverseFreqDatetimeRangeContext):
    __deprecated__ = True
    pass


@deprecated(version="0.8.0", reason="Use UniverseFreqHorizonDateRangeContext instead")
class UniverseFrequencyHorizonDateRangeContext(UniverseFreqHorizonDateRangeContext):
    __deprecated__ = True
    pass


@deprecated(version="0.8.0", reason="Use UniverseFreqHorizonDatetimeRangeContext instead")
class UniverseFrequencyHorizonDatetimeRangeContext(UniverseFreqHorizonDatetimeRangeContext):
    __deprecated__ = True
    pass


@deprecated(version="0.8.0", reason="Use UniverseDateVersionedContext instead")
class VersionedUniverseDateContext(UniverseDateVersionedContext):
    __deprecated__ = True
    pass


@deprecated(version="0.8.0", reason="Use UniverseDatetimeVersionedContext instead")
class VersionedUniverseDatetimeContext(UniverseDatetimeVersionedContext):
    __deprecated__ = True
    pass


@deprecated(version="0.8.0", reason="Use UniverseDateRangeVersionedContext instead")
class VersionedUniverseDateRangeContext(UniverseDateRangeVersionedContext):
    __deprecated__ = True
    pass


@deprecated(version="0.8.0", reason="Use UniverseDatetimeRangeVersionedContext instead")
class VersionedUniverseDatetimeRangeContext(UniverseDatetimeRangeVersionedContext):
    __deprecated__ = True
    pass


@deprecated(version="0.8.0", reason="Use ModelDateVersionedContext instead")
class VersionedModelDateContext(ModelDateVersionedContext):
    __deprecated__ = True
    pass


@deprecated(version="0.8.0", reason="Use ModelDatetimeVersionedContext instead")
class VersionedModelDatetimeContext(ModelDatetimeVersionedContext):
    __deprecated__ = True
    pass


@deprecated(version="0.8.0", reason="Use ModelDateRangeVersionedContext instead")
class VersionedModelDateRangeContext(ModelDateRangeVersionedContext):
    __deprecated__ = True
    pass


@deprecated(version="0.8.0", reason="Use ModelDatetimeRangeVersionedContext instead")
class VersionedModelDatetimeRangeContext(ModelDatetimeRangeVersionedContext):
    __deprecated__ = True
    pass


@deprecated(version="0.8.0", reason="Use DateVersionedContext instead")
class VersionedDateContext(EntryTimeContext, DateContext):
    __deprecated__ = True
    pass


@deprecated(version="0.8.0", reason="Use DatetimeVersionedContext instead")
class VersionedDatetimeContext(EntryTimeContext, DatetimeContext):
    __deprecated__ = True
    pass


@deprecated(version="0.8.0", reason="Use DateRangeVersionedContext instead")
class VersionedDateRangeContext(EntryTimeContext, DateRangeContext):
    __deprecated__ = True
    pass


@deprecated(version="0.8.0", reason="Use DatetimeRangeVersionedContext instead")
class VersionedDatetimeRangeContext(EntryTimeContext, DatetimeRangeContext):
    __deprecated__ = True
    pass
