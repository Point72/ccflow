"""Reusable narwhals pipeline abstractions for ccflow.

This module provides three layers of composition for building data-frame pipelines on top of
`narwhals <https://narwhals-dev.github.io/narwhals/>`_:

1. :class:`NarwhalsFrameTransform` -- a pure ``LazyFrame -> LazyFrame`` step. Framework-agnostic;
   usable standalone via ``lf.pipe(transform)`` without any other ccflow machinery.
2. :class:`SequenceTransform` -- a transform that bundles several transforms together. Itself a
   :class:`NarwhalsFrameTransform`, so it nests and composes via ``.pipe()``.
3. :class:`NarwhalsPipelineModel` -- a :class:`~ccflow.CallableModel` that wires a
   :class:`~ccflow.NarwhalsFrameResult`-returning source to a list of transforms, producing a new
   :class:`~ccflow.NarwhalsFrameResult`. Its :attr:`~NarwhalsPipelineModel.context_type` is
   delegated to the source, so the pipeline transparently adopts the source's context type.

Two generic transform implementations are also shipped:

- :class:`JoinTransform` -- joins another callable model's frame onto the input frame.
- :class:`JoinBackTransform` -- runs an inner transform on the input, joins the result back.

Together these are sufficient for a wide range of multi-source enrichment pipelines while
keeping the linear ``.pipe()`` contract that makes the rest of the design composable.
"""

from typing import Callable, List, Type, Union

import narwhals.stable.v1 as nw
from pydantic import Field, SerializeAsAny, model_validator

from ..base import BaseModel
from ..callable import CallableModel, Flow, GraphDepList
from ..context import ContextBase, NullContext
from ..result.narwhals import NarwhalsFrameResult

__all__ = (
    "NarwhalsFrameTransform",
    "SequenceTransform",
    "NarwhalsPipelineModel",
    "JoinTransform",
    "JoinBackTransform",
)


class NarwhalsFrameTransform(BaseModel):
    """Base class for a pure ``narwhals.LazyFrame -> narwhals.LazyFrame`` transform.

    Subclasses configure their behavior via pydantic fields and implement :meth:`__call__`.
    Instances are callable (``transform(lf)``), which makes them directly usable as the argument
    to :py:meth:`narwhals.LazyFrame.pipe`::

        class MultiplyColumn(NarwhalsFrameTransform):
            col: str
            factor: float

            def __call__(self, df):
                return df.with_columns(nw.col(self.col) * self.factor)

        out = lf.pipe(MultiplyColumn(col="x", factor=2.0))

    No ccflow machinery (sources, contexts, evaluators) is required to use a transform on its
    own -- it is just a configurable, JSON-serializable function on lazy frames.
    """

    def __call__(self, df: nw.LazyFrame) -> nw.LazyFrame:
        raise NotImplementedError


class SequenceTransform(NarwhalsFrameTransform):
    """Compose a list of :class:`NarwhalsFrameTransform` instances into a single transform.

    The transforms are applied in order via :py:meth:`narwhals.LazyFrame.pipe`. ``SequenceTransform``
    is itself a :class:`NarwhalsFrameTransform`, so it can be nested inside other sequences and
    used directly as a ``.pipe()`` argument.

    Because the ``transforms`` field is typed strictly as a list of :class:`NarwhalsFrameTransform`,
    a ``SequenceTransform`` is always JSON-roundtrippable.
    """

    transforms: List[NarwhalsFrameTransform] = Field(
        default_factory=list,
        description="Transforms applied in order via `.pipe()`.",
    )

    def __call__(self, df: nw.LazyFrame) -> nw.LazyFrame:
        for t in self.transforms:
            df = df.pipe(t)
        return df


# A loose union type used by NarwhalsPipelineModel.transforms. We deliberately put
# NarwhalsFrameTransform first in the union so pydantic prefers the BaseModel branch over
# the duck-typed Callable branch (NarwhalsFrameTransform instances are themselves callable).
NarwhalsFrameTransformOrCallable = Union[NarwhalsFrameTransform, Callable[[nw.LazyFrame], nw.LazyFrame]]


def _coerce_lazy(df) -> nw.LazyFrame:
    """Coerce any narwhals frame to a LazyFrame, accepting native frames as well.

    The pipeline contract is "always lazy" -- transforms returning eager frames or native objects
    are normalized so that subsequent stages see a ``narwhals.LazyFrame``.
    """
    if isinstance(df, nw.LazyFrame):
        return df
    if isinstance(df, nw.DataFrame):
        return df.lazy()
    # Fall back to nw.from_native to handle native frames returned by user-supplied callables.
    return nw.from_native(df).lazy()


class NarwhalsPipelineModel(CallableModel):
    """A callable model that pipes the output of a source through a list of transforms.

    The pipeline is itself a :class:`~ccflow.CallableModel` returning :class:`~ccflow.NarwhalsFrameResult`.
    Its :attr:`context_type` is delegated to the source -- so the pipeline transparently adopts
    whatever context type the source uses, without requiring users to parameterize a generic class.

    The output frame is always a ``narwhals.LazyFrame``; users that need an eager result should
    call ``.collect()`` on the returned :class:`~ccflow.NarwhalsFrameResult`.

    Parameters
    ----------
    source:
        A :class:`~ccflow.CallableModel` returning a :class:`~ccflow.NarwhalsFrameResult`. The
        pipeline's :attr:`context_type` is delegated to the source's.
        ``SerializeAsAny`` is applied explicitly because ccflow's metaclass does not unwrap
        generic-aliased ``BaseModel`` fields, and we want JSON roundtrip to preserve subclass info.
    transforms:
        A list of :class:`NarwhalsFrameTransform` instances or plain callables of one
        ``LazyFrame`` argument. Plain callables are accepted at runtime but cannot be JSON
        serialized; pipelines that need full serialization should use only
        :class:`NarwhalsFrameTransform` instances.
    """

    source: SerializeAsAny[CallableModel] = Field(
        ...,
        description="Upstream callable model that produces a NarwhalsFrameResult.",
    )
    transforms: List[NarwhalsFrameTransformOrCallable] = Field(
        default_factory=list,
        description="Transforms (or plain callables) applied in order via `.pipe()`.",
    )

    @model_validator(mode="after")
    def _validate_source_result_type(self) -> "NarwhalsPipelineModel":
        rt = self.source.result_type
        if not (isinstance(rt, type) and issubclass(rt, NarwhalsFrameResult)):
            raise ValueError(f"NarwhalsPipelineModel source must return NarwhalsFrameResult (or subclass); got source with result_type={rt!r}.")
        return self

    @property
    def context_type(self) -> Type[ContextBase]:
        """Delegate context type to the source so the pipeline adopts its context."""
        return self.source.context_type

    @property
    def result_type(self) -> Type[NarwhalsFrameResult]:
        return NarwhalsFrameResult

    @Flow.call
    def __call__(self, context) -> NarwhalsFrameResult:
        df = _coerce_lazy(self.source(context).df)
        for t in self.transforms:
            df = _coerce_lazy(df.pipe(t))
        return NarwhalsFrameResult(df=df)

    @Flow.deps
    def __deps__(self, context) -> GraphDepList:
        # Surface the source so that GraphEvaluator can see this edge. Transforms that themselves
        # invoke other CallableModels (e.g. JoinTransform) are NOT surfaced here -- multi-source
        # graph awareness is intentionally out of scope for v1; see module docstring.
        return [(self.source, [context])]


class JoinTransform(NarwhalsFrameTransform):
    """Join another callable model's frame onto the input frame.

    The ``other`` model is invoked with :class:`~ccflow.NullContext` -- this transform is for the
    common case where the secondary input is independent of any pipeline-level context. Pipelines
    needing context-dependent secondary sources should compose at the
    :class:`~ccflow.CallableModel` graph layer instead.

    Parameters mirror :py:meth:`narwhals.LazyFrame.join`. The ``other`` model is expected to
    return a :class:`~ccflow.NarwhalsFrameResult`; this is enforced at construction time.
    """

    other: SerializeAsAny[CallableModel] = Field(
        ...,
        description="Callable model producing the right-hand frame to join.",
    )
    on: Union[str, List[str], None] = Field(
        None,
        description="Column name(s) used as the join key. Mutually exclusive with left_on/right_on.",
    )
    left_on: Union[str, List[str], None] = Field(
        None,
        description="Column name(s) on the left frame. Use with right_on when join keys differ.",
    )
    right_on: Union[str, List[str], None] = Field(
        None,
        description="Column name(s) on the right frame. Use with left_on when join keys differ.",
    )
    how: str = Field("left", description="Join strategy: 'inner', 'left', 'right', 'full', 'cross', 'semi', 'anti'.")
    suffix: str = Field("_right", description="Suffix appended to overlapping column names from the right frame.")

    @model_validator(mode="after")
    def _validate_join_keys(self) -> "JoinTransform":
        has_on = self.on is not None
        has_left_right = self.left_on is not None or self.right_on is not None
        if self.how == "cross":
            if has_on or has_left_right:
                raise ValueError("Cross joins must not specify on/left_on/right_on.")
            return self
        if has_on and has_left_right:
            raise ValueError("Specify either on=... or left_on=... and right_on=..., not both.")
        if has_left_right and (self.left_on is None or self.right_on is None):
            raise ValueError("left_on and right_on must be specified together.")
        if not has_on and not has_left_right:
            raise ValueError("Must specify either on=... or left_on=... and right_on=...")
        return self

    @model_validator(mode="after")
    def _validate_other_result_type(self) -> "JoinTransform":
        rt = self.other.result_type
        if not (isinstance(rt, type) and issubclass(rt, NarwhalsFrameResult)):
            raise ValueError(f"JoinTransform.other must return NarwhalsFrameResult (or subclass); got other with result_type={rt!r}.")
        # NullContext-only invocation is by design for v1; verify the source can accept it.
        ct = self.other.context_type
        if not (isinstance(ct, type) and issubclass(NullContext, ct)):
            raise ValueError(f"JoinTransform.other must accept a NullContext (or supertype); got other with context_type={ct!r}.")
        return self

    def __call__(self, df: nw.LazyFrame) -> nw.LazyFrame:
        right = _coerce_lazy(self.other(NullContext()).df)
        if self.how == "cross":
            return df.join(right, how="cross", suffix=self.suffix)
        if self.on is not None:
            return df.join(right, on=self.on, how=self.how, suffix=self.suffix)
        return df.join(right, left_on=self.left_on, right_on=self.right_on, how=self.how, suffix=self.suffix)


class JoinBackTransform(NarwhalsFrameTransform):
    """Run an inner transform on the input frame and join its result back to the input.

    Useful for "fork-and-rejoin" patterns where a per-group summary needs to be enriched onto the
    original rows but window functions (``.over()``) don't fit -- e.g. when the summary changes
    row count, or aggregates a derived projection.

    For straightforward per-group statistics, prefer ``df.with_columns(expr.over(...))`` since it
    expresses intent more directly. ``JoinBackTransform`` is the right tool when the summary has
    a different shape from the input.
    """

    inner: NarwhalsFrameTransform = Field(
        ...,
        description="Transform applied to the input to produce the right-hand side of the join.",
    )
    on: Union[str, List[str]] = Field(..., description="Column name(s) used as the join key.")
    how: str = Field("left", description="Join strategy.")
    suffix: str = Field("_right", description="Suffix appended to overlapping column names from the right frame.")

    def __call__(self, df: nw.LazyFrame) -> nw.LazyFrame:
        right = _coerce_lazy(df.pipe(self.inner))
        return df.join(right, on=self.on, how=self.how, suffix=self.suffix)
