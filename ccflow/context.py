"""This module defines re-usable contexts for the "Callable Model" framework defined in flow.callable.py."""

from datetime import date, datetime
from typing import Optional

from pydantic import field_validator, model_validator

from .base import ContextBase
from .exttypes import Frequency
from .generic_base import GenericContext
from .validators import normalize_date

__all__ = [
    "NullContext",
    "GenericContext",
    "DateContext",
    "EntryTimeContext",
    "DateRangeContext",
    "VersionedDateContext",
    "VersionedDateRangeContext",
    "FreqContext",
    "FreqDateContext",
    "FreqDateRangeContext",
    "HorizonContext",
    "FreqHorizonContext",
    "FreqHorizonDateContext",
    "FreqHorizonDateRangeContext",
    "SeededDateRangeContext",
    "SourceContext",
    "UniverseContext",
    "UniverseDateContext",
    "UniverseDateRangeContext",
    "UniverseFrequencyDateRangeContext",
    "UniverseFrequencyHorizonDateRangeContext",
    "VersionedUniverseDateContext",
    "VersionedUniverseDateRangeContext",
    "ModelContext",
    "ModelDateContext",
    "ModelDateRangeContext",
    "ModelDateRangeSourceContext",
    "ModelFreqDateRangeContext",
    "VersionedModelDateContext",
    "VersionedModelDateRangeContext",
]

_SEPARATOR = ","


class NullContext(ContextBase):
    pass


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


class EntryTimeContext(ContextBase):
    entry_time_cutoff: Optional[datetime] = None


class SourceContext(ContextBase):
    source: Optional[str] = None


class DateRangeContext(ContextBase):
    start_date: date
    end_date: date

    _normalize_start = field_validator("start_date", mode="before")(normalize_date)
    _normalize_end = field_validator("end_date", mode="before")(normalize_date)


class SeededDateRangeContext(DateRangeContext):
    seed: int = 1234


class VersionedDateContext(DateContext, EntryTimeContext):
    pass


class VersionedDateRangeContext(DateRangeContext, EntryTimeContext):
    pass


class FreqContext(ContextBase):
    freq: Frequency


class FreqDateContext(DateContext, FreqContext):
    pass


class FreqDateRangeContext(DateRangeContext, FreqContext):
    pass


class HorizonContext(ContextBase):
    horizon: Frequency


class FreqHorizonContext(HorizonContext, FreqContext):
    pass


class FreqHorizonDateContext(DateContext, HorizonContext, FreqContext):
    pass


class FreqHorizonDateRangeContext(DateRangeContext, HorizonContext, FreqContext):
    pass


class UniverseContext(ContextBase):
    universe: str


class UniverseDateContext(DateContext, UniverseContext):
    pass


class UniverseDateRangeContext(DateRangeContext, UniverseContext):
    pass


class UniverseFrequencyDateRangeContext(DateRangeContext, FreqContext, UniverseContext):
    pass


class UniverseFrequencyHorizonDateRangeContext(DateRangeContext, HorizonContext, FreqContext, UniverseContext):
    pass


class VersionedUniverseDateContext(VersionedDateContext, UniverseContext):
    pass


class VersionedUniverseDateRangeContext(VersionedDateRangeContext, UniverseContext):
    pass


class ModelContext(ContextBase):
    model: str


class ModelDateContext(DateContext, ModelContext):
    pass


class ModelDateRangeContext(DateRangeContext, ModelContext):
    pass


class ModelDateRangeSourceContext(SourceContext, ModelDateRangeContext):
    pass


class ModelFreqDateRangeContext(FreqDateRangeContext, ModelContext):
    pass


class VersionedModelDateContext(VersionedDateContext, ModelContext):
    pass


class VersionedModelDateRangeContext(VersionedDateRangeContext, ModelContext):
    pass
