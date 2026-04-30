"""Generic TPC-H query runner.

A single ``TPCHQuery`` class can express any of the 22 TPC-H queries; each
query gets one configured instance in the registry (``query/Q1`` ...
``query/Q22``). The query's table dependencies are explicit Pydantic fields
on each instance (via ``inputs``).
"""

from importlib import import_module
from typing import Tuple

from pydantic import conint

from ccflow import CallableModel, CallableModelGenericType, Flow, NullContext
from ccflow.result.narwhals import NarwhalsFrameResult

__all__ = ("TPCHQuery",)


class TPCHQuery(CallableModel):
    """Runs one TPC-H query (``q{query_id}``) against a tuple of table providers.

    The query body itself comes from ``ccflow.examples.tpch.queries.q{N}.query``
    (vendored from narwhals); each input is called to produce a frame, and the
    frames are passed positionally to that function in the order given.

    A few ccflow features worth noting on this class:

    * The ``inputs`` field is typed ``CallableModelGenericType[NullContext,
      NarwhalsFrameResult]``. Using a ``CallableModelGenericType[C, R]`` as a
      Pydantic field type causes ccflow to validate, when the registry is
      loaded, that each resolved provider's ``__call__`` actually takes a
      ``NullContext`` (or subclass) and returns a ``NarwhalsFrameResult`` (or
      subclass). The configured ``TPCHTableProvider`` instances satisfy this
      because their return type, ``NarwhalsDataFrameResult``, is a subclass
      of ``NarwhalsFrameResult``.
    * Each provider is invoked with no arguments (``provider()``). The
      ``@Flow.call`` decorator on the provider's ``__call__`` reads the
      ``context: NullContext = NullContext()`` default from the signature
      when no context is passed in, so a "no-arg" call is well-defined.
    """

    query_id: conint(ge=1, le=22)
    inputs: Tuple[CallableModelGenericType[NullContext, NarwhalsFrameResult], ...]

    @Flow.call
    def __call__(self, context: NullContext = NullContext()) -> NarwhalsFrameResult:
        query_module = import_module(f"ccflow.examples.tpch.queries.q{self.query_id}")
        # Materialise the frames eagerly into a tuple before unpacking, so the
        # query body can iterate its inputs more than once if it wants to.
        frames = tuple(provider().df for provider in self.inputs)
        return NarwhalsFrameResult(df=query_module.query(*frames))
