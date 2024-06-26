import logging
import pandas as pd
from datetime import date, datetime
from pydantic import PrivateAttr
from typing import Any, ClassVar, List, Optional

from ccflow import CallableModel, DateContext, DateRangeContext, Flow, GenericResult, GraphDepList, ResultBase

log = logging.getLogger(__name__)


class MyResult(ResultBase):
    x: int


class MyDateCallable(CallableModel):
    """Set up some basic underlying model"""

    offset: int
    any: Any = None

    @Flow.call(volatile=True, validate_result=False)
    def current_time(self, context: DateContext):
        out = datetime.utcnow()
        log.info("Current datetime: %s", out)
        return out

    @Flow.call
    def __call__(self, context: DateContext) -> MyResult:
        return MyResult(x=context.date.day + self.offset)


class MyRaisingCallable(MyDateCallable):
    """Set up some basic underlying model"""

    @Flow.call
    def __call__(self, context: DateContext) -> MyResult:
        raise ValueError("Expected raising")


class MySometimesRaisingCallable(CallableModel):
    offset: int
    any: Any = None
    dates_to_raise: List[date]
    _called_dates: List[date] = PrivateAttr(default_factory=lambda **kwargs: [])

    @Flow.call
    def __call__(self, context: DateContext) -> MyResult:
        self._called_dates.append(context.date)

        if context.date in self.dates_to_raise:
            raise ValueError("Expected raising")

        return MyResult(x=context.date.day + self.offset)

    def get_called_dates(self) -> List[date]:
        return self._called_dates


class MyDateRangeCallable(CallableModel):
    """Set up some more complex model that references the underlying ones"""

    model1: CallableModel
    model2: Optional[MyDateCallable] = None

    @Flow.call
    def __call__(self, context: DateRangeContext) -> MyResult:
        dates = pd.date_range(context.start_date, context.end_date)
        result = sum(self.model1(DateContext(date=d)).x for d in dates)
        if self.model2:
            result += sum(self.model2(DateContext(date=d)).x for d in dates)
        return MyResult(x=result)

    @Flow.deps
    def __deps__(self, context: DateRangeContext) -> GraphDepList:
        dates = pd.date_range(context.start_date, context.end_date)
        deps = [(self.model1, [DateContext(date=d) for d in dates])]
        if self.model2:
            deps.append((self.model2, [DateContext(date=d) for d in dates]))
        return deps


class NodeModel(CallableModel):
    """Useful helper model for testing dependencies"""

    deps_model: List["NodeModel"] = []
    deps_context: List[DateContext] = []
    run_deps: bool = False  # Whether to call the dependencies in the call function

    _calls: ClassVar[List[Any]] = []  # To hold the outputs for testing
    _deps_calls: ClassVar[List[Any]] = []  # To hold the outputs for testing

    @Flow.call
    def __call__(self, context: DateContext) -> GenericResult:
        if self.run_deps:
            for model in self.deps_model:
                for c in [context] + self.deps_context:
                    model(c)
        self._calls.append((self.meta.name, context.date))
        return GenericResult(value=True)

    @Flow.deps
    def __deps__(self, context: DateContext):
        self._deps_calls.append((self.meta.name, context.date))
        return [(model, [context] + self.deps_context) for model in self.deps_model]


class CircularModel(CallableModel):
    @Flow.call
    def __call__(self, context: DateContext) -> GenericResult:
        return GenericResult(value=True)

    @Flow.deps
    def __deps__(self, context: DateContext):
        return [(self, [context])]
