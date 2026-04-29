"""Tests for ccflow.models.narwhals."""

from typing import List

import narwhals.stable.v1 as nw
import polars as pl
import pytest
from pydantic import ValidationError

from ccflow import (
    CallableModel,
    Flow,
    JoinBackTransform,
    JoinTransform,
    NarwhalsFrameTransform,
    NarwhalsPipelineModel,
    NullContext,
    SequenceTransform,
)
from ccflow.result.generic import GenericResult
from ccflow.result.narwhals import NarwhalsFrameResult

# --- helpers --- #


class MultiplyColumn(NarwhalsFrameTransform):
    """Test transform: multiplies a column by a scalar."""

    col: str
    factor: float

    def __call__(self, df: nw.LazyFrame) -> nw.LazyFrame:
        return df.with_columns(nw.col(self.col) * self.factor)


class FilterGreater(NarwhalsFrameTransform):
    col: str
    threshold: int

    def __call__(self, df: nw.LazyFrame) -> nw.LazyFrame:
        return df.filter(nw.col(self.col) > self.threshold)


class CollectThenReturn(NarwhalsFrameTransform):
    """Transform that returns an eager frame -- exercises the lazy-coercion contract."""

    def __call__(self, df: nw.LazyFrame) -> nw.LazyFrame:  # type: ignore[override]
        return df.collect()  # type: ignore[return-value]


class ReturnNative(NarwhalsFrameTransform):
    """Transform that returns a native polars LazyFrame -- exercises lazy-coercion via from_native."""

    def __call__(self, df: nw.LazyFrame) -> nw.LazyFrame:  # type: ignore[override]
        return df.to_native()  # type: ignore[return-value]


class FrameSource(CallableModel):
    """A source returning a fixed in-memory polars LazyFrame as a NarwhalsFrameResult."""

    data: dict

    @Flow.call
    def __call__(self, context: NullContext) -> NarwhalsFrameResult:
        return NarwhalsFrameResult(df=pl.LazyFrame(self.data))


def _values(df: nw.LazyFrame, col: str) -> List:
    return df.collect().to_native()[col].to_list()


# --- 1. NarwhalsFrameTransform base contract --- #


class TestBaseTransform:
    def test_pipe_compatibility(self):
        """A NarwhalsFrameTransform is callable and works directly with .pipe()."""
        lf = nw.from_native(pl.LazyFrame({"x": [1, 2, 3]}))
        out = lf.pipe(MultiplyColumn(col="x", factor=10.0))
        assert _values(out, "x") == [10.0, 20.0, 30.0]

    def test_default_call_raises(self):
        """The base class __call__ raises NotImplementedError so subclasses must override."""
        t = NarwhalsFrameTransform()
        with pytest.raises(NotImplementedError):
            t(nw.from_native(pl.LazyFrame({"x": [1]})))

    def test_json_roundtrip(self):
        """Concrete subclasses are pydantic models and roundtrip through JSON."""
        t = MultiplyColumn(col="x", factor=3.0)
        t2 = MultiplyColumn.model_validate_json(t.model_dump_json())
        assert t == t2


# --- 2. SequenceTransform --- #


class TestSequenceTransform:
    def test_applies_in_order(self):
        lf = nw.from_native(pl.LazyFrame({"x": [1, 2, 3]}))
        seq = SequenceTransform(transforms=[MultiplyColumn(col="x", factor=2), MultiplyColumn(col="x", factor=3)])
        assert _values(lf.pipe(seq), "x") == [6.0, 12.0, 18.0]

    def test_pipe_compatible(self):
        """A SequenceTransform is itself a NarwhalsFrameTransform."""
        seq = SequenceTransform(transforms=[MultiplyColumn(col="x", factor=2)])
        assert isinstance(seq, NarwhalsFrameTransform)

    def test_nestable(self):
        lf = nw.from_native(pl.LazyFrame({"x": [1, 2, 3]}))
        inner = SequenceTransform(transforms=[MultiplyColumn(col="x", factor=2)])
        outer = SequenceTransform(transforms=[inner, MultiplyColumn(col="x", factor=5)])
        assert _values(lf.pipe(outer), "x") == [10.0, 20.0, 30.0]

    def test_json_roundtrip(self):
        seq = SequenceTransform(transforms=[MultiplyColumn(col="x", factor=2), FilterGreater(col="x", threshold=2)])
        seq2 = SequenceTransform.model_validate_json(seq.model_dump_json())
        lf = nw.from_native(pl.LazyFrame({"x": [1, 2, 3]}))
        assert lf.pipe(seq).collect().to_native().equals(lf.pipe(seq2).collect().to_native())

    def test_empty_is_identity(self):
        lf = nw.from_native(pl.LazyFrame({"x": [1, 2, 3]}))
        assert _values(lf.pipe(SequenceTransform()), "x") == [1, 2, 3]


# --- 3. NarwhalsPipelineModel --- #


class TestPipelineModel:
    def test_basic_pipeline(self):
        src = FrameSource(data={"x": [1, 2, 3]})
        p = NarwhalsPipelineModel(source=src, transforms=[MultiplyColumn(col="x", factor=10)])
        res = p(NullContext())
        assert isinstance(res, NarwhalsFrameResult)
        assert isinstance(res.df, nw.LazyFrame)
        assert _values(res.df, "x") == [10.0, 20.0, 30.0]

    def test_context_type_delegated_to_source(self):
        src = FrameSource(data={"x": [1]})
        p = NarwhalsPipelineModel(source=src)
        assert p.context_type is src.context_type

    def test_result_type(self):
        p = NarwhalsPipelineModel(source=FrameSource(data={"x": [1]}))
        assert p.result_type is NarwhalsFrameResult

    def test_deps_includes_source(self):
        src = FrameSource(data={"x": [1]})
        p = NarwhalsPipelineModel(source=src)
        deps = p.__deps__(NullContext())
        assert len(deps) == 1
        dep_model, dep_contexts = deps[0]
        assert dep_model is src
        assert len(dep_contexts) == 1
        assert isinstance(dep_contexts[0], NullContext)

    def test_lazy_contract_enforced_on_eager_transform(self):
        """A transform that collects the frame should still leave the pipeline in a lazy state."""
        src = FrameSource(data={"x": [1, 2, 3]})
        p = NarwhalsPipelineModel(
            source=src,
            transforms=[CollectThenReturn(), MultiplyColumn(col="x", factor=2)],
        )
        res = p(NullContext())
        assert isinstance(res.df, nw.LazyFrame)
        assert _values(res.df, "x") == [2, 4, 6]

    def test_lazy_contract_enforced_on_native_transform(self):
        src = FrameSource(data={"x": [1, 2, 3]})
        p = NarwhalsPipelineModel(
            source=src,
            transforms=[ReturnNative(), MultiplyColumn(col="x", factor=2)],
        )
        res = p(NullContext())
        assert isinstance(res.df, nw.LazyFrame)
        assert _values(res.df, "x") == [2, 4, 6]

    def test_loose_callable_accepted(self):
        src = FrameSource(data={"x": [1, 2, 3]})
        p = NarwhalsPipelineModel(
            source=src,
            transforms=[lambda d: d.with_columns(nw.col("x") * 5)],
        )
        assert _values(p(NullContext()).df, "x") == [5, 10, 15]

    def test_json_roundtrip_strict_transforms(self):
        src = FrameSource(data={"x": [1, 2, 3]})
        p = NarwhalsPipelineModel(source=src, transforms=[MultiplyColumn(col="x", factor=10)])
        p2 = NarwhalsPipelineModel.model_validate_json(p.model_dump_json())
        assert _values(p(NullContext()).df, "x") == _values(p2(NullContext()).df, "x")

    def test_rejects_non_narwhals_source(self):
        class BadSrc(CallableModel):
            @Flow.call
            def __call__(self, context: NullContext) -> GenericResult[int]:
                return GenericResult[int](value=1)

        with pytest.raises(ValidationError, match="NarwhalsFrameResult"):
            NarwhalsPipelineModel(source=BadSrc())

    def test_dependency_injection_swap_source(self):
        """Same pipeline, two sources -- demonstrates DI."""
        transforms = [MultiplyColumn(col="x", factor=2)]
        p1 = NarwhalsPipelineModel(source=FrameSource(data={"x": [1, 2, 3]}), transforms=transforms)
        p2 = NarwhalsPipelineModel(source=FrameSource(data={"x": [10, 20, 30]}), transforms=transforms)
        assert _values(p1(NullContext()).df, "x") == [2, 4, 6]
        assert _values(p2(NullContext()).df, "x") == [20, 40, 60]


# --- 4. JoinTransform --- #


class TestJoinTransform:
    def test_basic_join(self):
        lf = nw.from_native(pl.LazyFrame({"x": [1, 2, 3]}))
        right_src = FrameSource(data={"x": [1, 2, 3], "y": ["a", "b", "c"]})
        jt = JoinTransform(other=right_src, on="x", how="left")
        out = lf.pipe(jt).collect().to_native()
        assert out.columns == ["x", "y"]
        assert out["y"].to_list() == ["a", "b", "c"]

    def test_pipe_compatible(self):
        assert isinstance(JoinTransform(other=FrameSource(data={"x": [1]}), on="x"), NarwhalsFrameTransform)

    def test_json_roundtrip(self):
        right_src = FrameSource(data={"x": [1, 2, 3], "y": ["a", "b", "c"]})
        jt = JoinTransform(other=right_src, on="x", how="left", suffix="_r")
        jt2 = JoinTransform.model_validate_json(jt.model_dump_json())
        lf = nw.from_native(pl.LazyFrame({"x": [1, 2, 3]}))
        assert lf.pipe(jt).collect().to_native().equals(lf.pipe(jt2).collect().to_native())

    def test_rejects_non_narwhals_other(self):
        class BadSrc(CallableModel):
            @Flow.call
            def __call__(self, context: NullContext) -> GenericResult[int]:
                return GenericResult[int](value=1)

        with pytest.raises(ValidationError, match="NarwhalsFrameResult"):
            JoinTransform(other=BadSrc(), on="x")

    def test_in_pipeline(self):
        """Use a JoinTransform as a stage in a NarwhalsPipelineModel (multi-source enrichment)."""
        main = FrameSource(data={"x": [1, 2, 3], "v": [10, 20, 30]})
        side = FrameSource(data={"x": [1, 2, 3], "label": ["a", "b", "c"]})
        p = NarwhalsPipelineModel(source=main, transforms=[JoinTransform(other=side, on="x")])
        out = p(NullContext()).df.collect().to_native()
        assert out.columns == ["x", "v", "label"]
        assert out["label"].to_list() == ["a", "b", "c"]

    def test_left_on_right_on(self):
        lf = nw.from_native(pl.LazyFrame({"a": [1, 2, 3]}))
        right_src = FrameSource(data={"b": [1, 2, 3], "y": ["a", "b", "c"]})
        jt = JoinTransform(other=right_src, left_on="a", right_on="b", how="inner")
        out = lf.pipe(jt).collect().to_native()
        assert out["y"].to_list() == ["a", "b", "c"]

    def test_cross_join(self):
        lf = nw.from_native(pl.LazyFrame({"a": [1, 2]}))
        right_src = FrameSource(data={"b": [10, 20]})
        jt = JoinTransform(other=right_src, how="cross")
        out = lf.pipe(jt).collect().to_native()
        assert len(out) == 4

    def test_rejects_missing_join_keys(self):
        with pytest.raises(ValidationError, match="either on="):
            JoinTransform(other=FrameSource(data={"x": [1]}))

    def test_rejects_both_on_and_left_right(self):
        with pytest.raises(ValidationError, match="not both"):
            JoinTransform(other=FrameSource(data={"x": [1]}), on="x", left_on="x", right_on="x")

    def test_rejects_partial_left_right(self):
        with pytest.raises(ValidationError, match="must be specified together"):
            JoinTransform(other=FrameSource(data={"x": [1]}), left_on="x")


# --- 5. JoinBackTransform --- #


class TestJoinBackTransform:
    def test_basic_join_back(self):
        lf = nw.from_native(pl.LazyFrame({"x": [1, 2, 3], "v": [10, 20, 30]}))

        class Project(NarwhalsFrameTransform):
            def __call__(self, df: nw.LazyFrame) -> nw.LazyFrame:
                return df.select(nw.col("x"), (nw.col("v") * 100).alias("hund"))

        jb = JoinBackTransform(inner=Project(), on="x")
        out = lf.pipe(jb).collect().to_native()
        assert out.columns == ["x", "v", "hund"]
        assert out["hund"].to_list() == [1000, 2000, 3000]

    def test_pipe_compatible(self):
        class Noop(NarwhalsFrameTransform):
            def __call__(self, df):
                return df

        assert isinstance(JoinBackTransform(inner=Noop(), on="x"), NarwhalsFrameTransform)

    def test_json_roundtrip(self):
        jb = JoinBackTransform(inner=MultiplyColumn(col="x", factor=2), on="x")
        jb2 = JoinBackTransform.model_validate_json(jb.model_dump_json())
        assert jb == jb2


# --- 6. Confluence: source pipeline composed of two pipelines --- #


class TestConfluence:
    def test_pipeline_as_source_of_another_pipeline(self):
        """A NarwhalsPipelineModel can itself be the `source` of another NarwhalsPipelineModel."""
        upstream = NarwhalsPipelineModel(
            source=FrameSource(data={"x": [1, 2, 3]}),
            transforms=[MultiplyColumn(col="x", factor=2)],
        )
        downstream = NarwhalsPipelineModel(source=upstream, transforms=[FilterGreater(col="x", threshold=3)])
        out = downstream(NullContext()).df.collect().to_native()
        assert out["x"].to_list() == [4, 6]

    def test_pipeline_as_other_in_join(self):
        """A NarwhalsPipelineModel can be the `other` of a JoinTransform (confluence pattern)."""
        side_pipeline = NarwhalsPipelineModel(
            source=FrameSource(data={"x": [1, 2, 3], "y": ["a", "b", "c"]}),
            transforms=[FilterGreater(col="x", threshold=1)],
        )
        main = FrameSource(data={"x": [1, 2, 3], "v": [10, 20, 30]})
        p = NarwhalsPipelineModel(source=main, transforms=[JoinTransform(other=side_pipeline, on="x", how="inner")])
        out = p(NullContext()).df.collect().to_native()
        assert sorted(out["x"].to_list()) == [2, 3]
