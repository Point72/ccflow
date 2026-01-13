"""This module defines re-usable contexts for the "Callable Model" framework defined in flow.callable.py."""

import warnings
from datetime import date, datetime
from typing import Any, Generic, Hashable, Optional, Sequence, Set, TypeVar

from deprecated import deprecated
from pydantic import ConfigDict, field_validator, model_validator

from .base import ContextBase
from .exttypes import Frequency
from .validators import normalize_date, normalize_datetime

warnings.simplefilter("always", DeprecationWarning)


__all__ = (
    "FlowContext",
    "NullContext",
    "GenericContext",
    "DateContext",
    "DatetimeContext",
    "EntryTimeContext",
    "SeedContext",
    "SourceContext",
    "DateRangeContext",
    "DatetimeRangeContext",
    "SeedDateRangeContext",
    "SeedDatetimeRangeContext",
    "DateEntryTimeContext",
    "DatetimeEntryTimeContext",
    "DateRangeEntryTimeContext",
    "DatetimeRangeEntryTimeContext",
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
    "UniverseDateEntryTimeContext",
    "UniverseDatetimeEntryTimeContext",
    "UniverseDateRangeEntryTimeContext",
    "UniverseDatetimeRangeEntryTimeContext",
    "ModelContext",
    "ModelDateContext",
    "ModelDatetimeContext",
    "ModelDateRangeContext",
    "ModelDatetimeRangeContext",
    "ModelDateRangeSourceContext",
    "ModelFreqDateRangeContext",
    "ModelFreqDatetimeRangeContext",
    "ModelDateEntryTimeContext",
    "ModelDatetimeEntryTimeContext",
    "ModelDateRangeEntryTimeContext",
    "ModelDatetimeRangeEntryTimeContext",
    # deprecated aliases
    "SeededContext",
    "SeededDateRangeContext",
    "SeededDatetimeRangeContext",
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

# Starting 0.8.0 Nullcontext is an alias to ContextBase
NullContext = ContextBase


class FlowContext(ContextBase):
    """Universal context for @Flow.model functions.

    Instead of generating a new ContextBase subclass for each @Flow.model,
    this single class with extra="allow" serves as the universal carrier.
    Validation happens via TypedDict + TypeAdapter at compute() time.

    This design avoids:
    - Proliferation of dynamic _funcname_Context classes
    - Class registration overhead for serialization
    - Pickling issues with Ray/distributed computing

    Fields are stored in __pydantic_extra__ and accessed via __getattr__.
    """

    model_config = ConfigDict(extra="allow", frozen=True)

    def __getattr__(self, name: str) -> Any:
        """Access fields stored in __pydantic_extra__."""
        # Use object.__getattribute__ to avoid infinite recursion
        try:
            extra = object.__getattribute__(self, "__pydantic_extra__")
            if extra is not None and name in extra:
                return extra[name]
        except AttributeError:
            pass
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __repr__(self) -> str:
        """Show all fields including extra fields."""
        extra = object.__getattribute__(self, "__pydantic_extra__") or {}
        fields = ", ".join(f"{k}={v!r}" for k, v in extra.items())
        return f"FlowContext({fields})"


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


class SeedContext(ContextBase):
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


class SeedDateRangeContext(DateRangeContext, SeedContext):
    pass


class SeedDatetimeRangeContext(DatetimeRangeContext, SeedContext):
    pass


class DateEntryTimeContext(EntryTimeContext, DateContext):
    pass


class DatetimeEntryTimeContext(EntryTimeContext, DatetimeContext):
    pass


class DateRangeEntryTimeContext(EntryTimeContext, DateRangeContext):
    pass


class DatetimeRangeEntryTimeContext(EntryTimeContext, DatetimeRangeContext):
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


class UniverseDateEntryTimeContext(DateEntryTimeContext, UniverseDateContext):
    pass


class UniverseDatetimeEntryTimeContext(DatetimeEntryTimeContext, UniverseDatetimeContext):
    pass


class UniverseDateRangeEntryTimeContext(DateRangeEntryTimeContext, UniverseDateRangeContext):
    pass


class UniverseDatetimeRangeEntryTimeContext(DatetimeRangeEntryTimeContext, UniverseDatetimeRangeContext):
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


class ModelDateEntryTimeContext(DateEntryTimeContext, ModelDateContext):
    pass


class ModelDatetimeEntryTimeContext(DatetimeEntryTimeContext, ModelDatetimeContext):
    pass


class ModelDateRangeEntryTimeContext(DateRangeEntryTimeContext, ModelDateRangeContext):
    pass


class ModelDatetimeRangeEntryTimeContext(DatetimeRangeEntryTimeContext, ModelDatetimeRangeContext):
    pass


# TODO - remove later to avoid breaking changes for now
@deprecated(version="0.8.0", reason="Use SeedContext instead")
class SeededContext(SeedContext):
    __deprecated__ = True
    pass


@deprecated(version="0.8.0", reason="Use SeedDateRangeContext instead")
class SeededDateRangeContext(SeedDateRangeContext):
    __deprecated__ = True
    pass


@deprecated(version="0.8.0", reason="Use SeedDatetimeRangeContext instead")
class SeededDatetimeRangeContext(SeedDatetimeRangeContext):
    __deprecated__ = True
    pass


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


@deprecated(version="0.8.0", reason="Use UniverseDateEntryTimeContext instead")
class VersionedUniverseDateContext(UniverseDateEntryTimeContext):
    __deprecated__ = True
    pass


@deprecated(version="0.8.0", reason="Use UniverseDatetimeEntryTimeContext instead")
class VersionedUniverseDatetimeContext(UniverseDatetimeEntryTimeContext):
    __deprecated__ = True
    pass


@deprecated(version="0.8.0", reason="Use UniverseDateRangeEntryTimeContext instead")
class VersionedUniverseDateRangeContext(UniverseDateRangeEntryTimeContext):
    __deprecated__ = True
    pass


@deprecated(version="0.8.0", reason="Use UniverseDatetimeRangeEntryTimeContext instead")
class VersionedUniverseDatetimeRangeContext(UniverseDatetimeRangeEntryTimeContext):
    __deprecated__ = True
    pass


@deprecated(version="0.8.0", reason="Use ModelDateEntryTimeContext instead")
class VersionedModelDateContext(ModelDateEntryTimeContext):
    __deprecated__ = True
    pass


@deprecated(version="0.8.0", reason="Use ModelDatetimeEntryTimeContext instead")
class VersionedModelDatetimeContext(ModelDatetimeEntryTimeContext):
    __deprecated__ = True
    pass


@deprecated(version="0.8.0", reason="Use ModelDateRangeEntryTimeContext instead")
class VersionedModelDateRangeContext(ModelDateRangeEntryTimeContext):
    __deprecated__ = True
    pass


@deprecated(version="0.8.0", reason="Use ModelDatetimeRangeEntryTimeContext instead")
class VersionedModelDatetimeRangeContext(ModelDatetimeRangeEntryTimeContext):
    __deprecated__ = True
    pass


@deprecated(version="0.8.0", reason="Use DateEntryTimeContext instead")
class VersionedDateContext(EntryTimeContext, DateContext):
    __deprecated__ = True
    pass


@deprecated(version="0.8.0", reason="Use DatetimeEntryTimeContext instead")
class VersionedDatetimeContext(EntryTimeContext, DatetimeContext):
    __deprecated__ = True
    pass


@deprecated(version="0.8.0", reason="Use DateRangeEntryTimeContext instead")
class VersionedDateRangeContext(EntryTimeContext, DateRangeContext):
    __deprecated__ = True
    pass


@deprecated(version="0.8.0", reason="Use DatetimeRangeEntryTimeContext instead")
class VersionedDatetimeRangeContext(EntryTimeContext, DatetimeRangeContext):
    __deprecated__ = True
    pass
