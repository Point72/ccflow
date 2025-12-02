"""This module defines re-usable contexts for the "Callable Model" framework defined in flow.callable.py."""

from datetime import date, datetime
from typing import Generic, Hashable, Optional, Sequence, Set, TypeVar

from pydantic import field_validator, model_validator

from .base import ContextBase
from .exttypes import Frequency
from .validators import normalize_date, normalize_datetime

__all__ = (
    "NullContext",
    "GenericContext",
    "DateContext",
    "DatetimeContext",
    "EntryTimeContext",
    "SourceContext",
    "DateRangeContext",
    "DatetimeRangeContext",
    "SeededDateRangeContext",
    "SeededDatetimeRangeContext",
    "VersionedDateContext",
    "VersionedDatetimeContext",
    "VersionedDateRangeContext",
    "VersionedDatetimeRangeContext",
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
    "UniverseFrequencyDateRangeContext",
    "UniverseFrequencyDatetimeRangeContext",
    "UniverseFrequencyHorizonDateRangeContext",
    "UniverseFrequencyHorizonDatetimeRangeContext",
    "VersionedUniverseDateContext",
    "VersionedUniverseDatetimeContext",
    "VersionedUniverseDateRangeContext",
    "VersionedUniverseDatetimeRangeContext",
    "ModelContext",
    "ModelDateContext",
    "ModelDatetimeContext",
    "ModelDateRangeContext",
    "ModelDatetimeRangeContext",
    "ModelDateRangeSourceContext",
    "ModelFreqDateRangeContext",
    "ModelFreqDatetimeRangeContext",
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


class SeededDateRangeContext(DateRangeContext):
    seed: int = 1234


class SeededDatetimeRangeContext(DatetimeRangeContext):
    seed: int = 1234


class VersionedDateContext(DateContext, EntryTimeContext):
    pass


class VersionedDatetimeContext(DatetimeContext, EntryTimeContext):
    pass


class VersionedDateRangeContext(DateRangeContext, EntryTimeContext):
    pass


class VersionedDatetimeRangeContext(DatetimeRangeContext, EntryTimeContext):
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


class FreqHorizonContext(HorizonContext, FreqContext):
    pass


class FreqHorizonDateContext(DateContext, HorizonContext, FreqContext):
    pass


class FreqHorizonDatetimeContext(DatetimeContext, HorizonContext, FreqContext):
    pass


class FreqHorizonDateRangeContext(DateRangeContext, HorizonContext, FreqContext):
    pass


class FreqHorizonDatetimeRangeContext(DatetimeRangeContext, HorizonContext, FreqContext):
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


class UniverseFrequencyDateRangeContext(DateRangeContext, FreqContext, UniverseContext):
    pass


class UniverseFrequencyDatetimeRangeContext(DatetimeRangeContext, FreqContext, UniverseContext):
    pass


class UniverseFrequencyHorizonDateRangeContext(DateRangeContext, HorizonContext, FreqContext, UniverseContext):
    pass


class UniverseFrequencyHorizonDatetimeRangeContext(DatetimeRangeContext, HorizonContext, FreqContext, UniverseContext):
    pass


class VersionedUniverseDateContext(VersionedDateContext, UniverseContext):
    pass


class VersionedUniverseDatetimeContext(VersionedDatetimeContext, UniverseContext):
    pass


class VersionedUniverseDateRangeContext(VersionedDateRangeContext, UniverseContext):
    pass


class VersionedUniverseDatetimeRangeContext(VersionedDatetimeRangeContext, UniverseContext):
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


class ModelFreqDateRangeContext(FreqDateRangeContext, ModelContext):
    pass


class ModelFreqDatetimeRangeContext(FreqDatetimeRangeContext, ModelContext):
    pass


class VersionedModelDateContext(VersionedDateContext, ModelContext):
    pass


class VersionedModelDatetimeContext(VersionedDatetimeContext, ModelContext):
    pass


class VersionedModelDateRangeContext(VersionedDateRangeContext, ModelContext):
    pass


class VersionedModelDatetimeRangeContext(VersionedDatetimeRangeContext, ModelContext):
    pass
