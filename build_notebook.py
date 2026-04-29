"""Build ccflow/examples/narwhals_pipelines.ipynb.

Run from the repo root: ``python build_notebook.py``

This script is the source of truth for the notebook -- editing the .ipynb directly
risks losing structure. After editing this script, run it and commit the regenerated
notebook alongside.
"""

import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []


def md(text: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(text.strip("\n")))


def code(text: str) -> None:
    cells.append(nbf.v4.new_code_cell(text.strip("\n")))


# ============================================================================
# §1. Motivation
# ============================================================================

md(r"""
# Reusable narwhals pipelines with `ccflow`

This notebook walks through three layers of composition for building data-frame pipelines
on top of [narwhals](https://narwhals-dev.github.io/narwhals/), provided by `ccflow.models.narwhals`:

1. **`NarwhalsFrameTransform`** — a pure `LazyFrame -> LazyFrame` step. Framework-agnostic;
   usable standalone via `lf.pipe(transform)` without any other ccflow machinery.
2. **`SequenceTransform`** — bundles a list of transforms into a single transform.
3. **`NarwhalsPipelineModel`** — a `CallableModel` that wires a frame source to a list of
   transforms, producing a new `NarwhalsFrameResult`.

Plus two generic implementations:

- **`JoinTransform`** — joins another callable model's frame onto the input frame.
- **`JoinBackTransform`** — runs an inner transform on the input, joins the result back.

The TPC-H benchmark is used as a worked example (Q1 → Q3).
""")

md(r"""
## Why?

A typical narwhals query is just a function:

```python
def query(lineitem):
    return (
        lineitem.filter(...)
        .with_columns(...)
        .group_by(...).agg(...)
        .sort(...)
    )
```

This is fine for a one-shot script. But in a real codebase you usually want:

- **Configurability** — the date cutoff, the segment, the group-by columns shouldn't be hard-coded.
- **Reusability** — small pieces (e.g. "filter by ship date") used in many queries.
- **Dependency injection** — swap the data source (DuckDB, parquet, mocked) without rewriting the query.
- **Serialization** — store the entire pipeline as JSON or YAML; reload from config.
- **Graph awareness** — when a pipeline composes other pipelines, ccflow's graph evaluator can
  parallelize and cache shared upstream computations.

The three layers below give you these properties incrementally — pay only for the level
of structure you actually need.
""")

md(r"""
## The three layers, at a glance

| Layer | What it is | What you get |
|-------|------------|--------------|
| `NarwhalsFrameTransform` | A pydantic model that is callable on a `LazyFrame`. | Configurable, JSON-serializable, `.pipe()`-compatible. **No ccflow runtime required.** |
| `SequenceTransform` | A transform whose state is a list of transforms. | Composition + JSON roundtrip + nestable. Still just `.pipe()`-compatible. |
| `NarwhalsPipelineModel` | A `CallableModel` returning `NarwhalsFrameResult`. | Sources, contexts, dependency graph, evaluators, caching — full ccflow integration. |

You can stop at layer 1 if you only want better-behaved transforms in an existing codebase.
You can stop at layer 2 if you want bundling without ccflow.
You graduate to layer 3 when you need sourcing, configuration, or graph composition.

---

### Table of contents

1. Setup & the starting point
2. The `NarwhalsFrameTransform` base class
3. Refactoring Q1 into transforms
4. `SequenceTransform`: bundling transforms
5. `NarwhalsPipelineModel`: source + transforms
6. Multi-source enrichment with `JoinTransform`
7. Confluence: pipelines as inputs to pipelines (Q3)
8. `JoinBackTransform`: fork-and-rejoin patterns
9. When *not* to use `NarwhalsPipelineModel`
10. Summary
""")

# ============================================================================
# §2. Setup & the starting point
# ============================================================================

md(r"""
## 1. Setup & the starting point

We use the [TPC-H benchmark](https://www.tpc.org/tpch/) as our working data — a synthetic
sales/orders schema with 8 tables. ccflow ships a generator that produces these tables at
arbitrary scale factors (we'll use `0.01`, ~10MB per table).
""")

code(r"""
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

import narwhals.stable.v1 as nw
import polars as pl

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
from ccflow.examples.tpch.base import TPCHTable, TPCHTableContext
from ccflow.examples.tpch.data_generators import TPCHDataGenerator
from ccflow.result.narwhals import NarwhalsFrameResult

generator = TPCHDataGenerator(scale_factor=0.01)
""")

md(r"""
### Q1 today

Here is TPC-H Q1, written as a plain narwhals function — adapted from
[`ccflow/examples/tpch/queries/q1.py`](../examples/tpch/queries/q1.py). It computes
pricing summary statistics for line items shipped before a cutoff date, grouped by
return-flag and line-status.
""")

code(r"""
from datetime import datetime


def q1_monolith(lineitem):
    cutoff = datetime(1998, 9, 2)
    return (
        lineitem.filter(nw.col("l_shipdate") <= cutoff)
        .with_columns(
            disc_price=nw.col("l_extendedprice") * (1 - nw.col("l_discount")),
            charge=nw.col("l_extendedprice") * (1 - nw.col("l_discount")) * (1 + nw.col("l_tax")),
        )
        .group_by("l_returnflag", "l_linestatus")
        .agg(
            nw.sum("l_quantity").alias("sum_qty"),
            nw.sum("l_extendedprice").alias("sum_base_price"),
            nw.sum("disc_price").alias("sum_disc_price"),
            nw.sum("charge").alias("sum_charge"),
            nw.mean("l_quantity").alias("avg_qty"),
            nw.mean("l_extendedprice").alias("avg_price"),
            nw.mean("l_discount").alias("avg_disc"),
            nw.len().alias("count_order"),
        )
        .sort("l_returnflag", "l_linestatus")
    )


# Pull the lineitem table; TPCHDataGenerator returns an eager NarwhalsDataFrameResult.
lineitem_eager = generator(TPCHTableContext(table="lineitem")).df
lineitem = lineitem_eager.lazy()  # work lazily for plan optimization

q1_monolith(lineitem).collect().to_native()
""")

md(r"""
### What's wrong with this picture?

This works, but the function locks in:

- A specific **date cutoff** (`1998-09-02`).
- A specific **set of group-by columns**.
- A specific **set of aggregation outputs**.
- The expectation that the input is exactly `lineitem` from a hard-coded place.

Each of those is a reasonable thing to *configure*. And while we can certainly add parameters
and turn this into a 12-arg function, that scales poorly: it conflates "what the pipeline does"
with "where the data comes from", and gives no good way to swap individual stages.

The next sections build up structure gradually until each of these concerns has its own home.
""")

# ============================================================================
# §3. The base class
# ============================================================================

md(r"""
## 2. The `NarwhalsFrameTransform` base class

A `NarwhalsFrameTransform` is just a callable pydantic model that takes a `narwhals.LazyFrame`
and returns a `narwhals.LazyFrame`. Subclasses declare configuration via fields and implement
`__call__`.

Crucially: **a transform is usable on its own, with no ccflow runtime required.** It works
directly as the argument to `narwhals.LazyFrame.pipe(...)`.
""")

code(r"""
class MultiplyColumn(NarwhalsFrameTransform):
    '''Toy transform: multiply a column by a constant factor.'''

    col: str
    factor: float

    def __call__(self, df):
        return df.with_columns(nw.col(self.col) * self.factor)


lf = nw.from_native(pl.LazyFrame({"x": [1, 2, 3]}))

# Instances are pydantic models -- configurable, validated, JSON-serializable.
t = MultiplyColumn(col="x", factor=10.0)
print("config:", t.model_dump_json())

# And they are callables, so they slot into .pipe() naturally.
print("output:", lf.pipe(t).collect().to_native())
""")

md(r"""
That's it. `MultiplyColumn` reuses cleanly anywhere narwhals is used — no ccflow imports,
sources, or contexts are needed in client code that just wants to call `lf.pipe(t)`.

The benefits of being a pydantic model show up when you start composing or persisting:
field validation, JSON roundtrip, IDE autocompletion, hydra config support.
""")

# ============================================================================
# §4. Refactor Q1 into transforms
# ============================================================================

md(r"""
## 3. Refactoring Q1 into business-meaningful transforms

The Q1 monolith naturally decomposes into four phases. Each becomes its own
`NarwhalsFrameTransform`, each named after what it *does* in business terms.
""")

md(r"""
### 3.1. Filter by ship date
""")

code(r"""
class FilterByShipDate(NarwhalsFrameTransform):
    cutoff: datetime

    def __call__(self, df):
        return df.filter(nw.col("l_shipdate") <= self.cutoff)


# Standalone use:
filtered = lineitem.pipe(FilterByShipDate(cutoff=datetime(1998, 9, 2)))
filtered.collect().shape
""")

md(r"""
### 3.2. Derive line-item metrics
""")

code(r"""
class DeriveLineItemMetrics(NarwhalsFrameTransform):
    '''Adds disc_price = extendedprice * (1 - discount), charge = disc_price * (1 + tax).'''

    def __call__(self, df):
        return df.with_columns(
            disc_price=nw.col("l_extendedprice") * (1 - nw.col("l_discount")),
            charge=nw.col("l_extendedprice") * (1 - nw.col("l_discount")) * (1 + nw.col("l_tax")),
        )


lineitem.pipe(FilterByShipDate(cutoff=datetime(1998, 9, 2))).pipe(DeriveLineItemMetrics()).collect().columns
""")

md(r"""
### 3.3. Aggregate by return-status
""")

code(r"""
class AggregateByReturnStatus(NarwhalsFrameTransform):
    def __call__(self, df):
        return df.group_by("l_returnflag", "l_linestatus").agg(
            nw.sum("l_quantity").alias("sum_qty"),
            nw.sum("l_extendedprice").alias("sum_base_price"),
            nw.sum("disc_price").alias("sum_disc_price"),
            nw.sum("charge").alias("sum_charge"),
            nw.mean("l_quantity").alias("avg_qty"),
            nw.mean("l_extendedprice").alias("avg_price"),
            nw.mean("l_discount").alias("avg_disc"),
            nw.len().alias("count_order"),
        )
""")

md(r"""
### 3.4. Sort the aggregated result
""")

code(r"""
class SortByReturnStatus(NarwhalsFrameTransform):
    def __call__(self, df):
        return df.sort("l_returnflag", "l_linestatus")
""")

md(r"""
### Compose by hand

Now we can assemble the query as a chain of `.pipe()` calls. The chain reads top-to-bottom
like a description of what the query *does*.
""")

code(r"""
result = (
    lineitem
    .pipe(FilterByShipDate(cutoff=datetime(1998, 9, 2)))
    .pipe(DeriveLineItemMetrics())
    .pipe(AggregateByReturnStatus())
    .pipe(SortByReturnStatus())
)
result.collect().to_native()
""")

# ============================================================================
# §5. SequenceTransform
# ============================================================================

md(r"""
## 4. `SequenceTransform`: bundling transforms

When you find yourself repeating the same chain of transforms in multiple places, bundle
them into a `SequenceTransform`. The bundle is itself a `NarwhalsFrameTransform`, so it
remains `.pipe()`-compatible and can be nested inside other sequences.

Because `SequenceTransform.transforms` is typed as `List[NarwhalsFrameTransform]`,
sequences are always JSON-roundtrippable.
""")

code(r"""
q1_preproc = SequenceTransform(transforms=[
    FilterByShipDate(cutoff=datetime(1998, 9, 2)),
    DeriveLineItemMetrics(),
    AggregateByReturnStatus(),
    SortByReturnStatus(),
])

# Pipe-compatible: same usage as any other transform.
lineitem.pipe(q1_preproc).collect().head().to_native()
""")

code(r"""
# JSON roundtrip preserves structure exactly.
serialized = q1_preproc.model_dump_json(indent=2)
print(serialized[:500] + " ...")

q1_preproc_restored = SequenceTransform.model_validate_json(serialized)
restored = lineitem.pipe(q1_preproc_restored).collect().to_native()
original = lineitem.pipe(q1_preproc).collect().to_native()
from polars.testing import assert_frame_equal as _afe
_afe(restored, original, check_exact=False)
print("\nroundtrip OK ✓")
""")

# ============================================================================
# §6. NarwhalsPipelineModel
# ============================================================================

md(r"""
## 5. `NarwhalsPipelineModel`: source + transforms

`SequenceTransform` is still backend-agnostic — it just composes transforms. To get
**dependency injection**, **graph composition**, and **full configuration via Hydra/JSON**,
graduate to `NarwhalsPipelineModel`.

A pipeline takes:

- a **`source`**: any `CallableModel` whose `result_type` is `NarwhalsFrameResult` (or a subclass).
- a list of **`transforms`**: applied in order via `.pipe()`.

The pipeline is itself a `CallableModel` returning `NarwhalsFrameResult`. Its
`context_type` is delegated to the source — so the pipeline transparently adopts whatever
context type the source uses.

The output is **always lazy**: even if a transform returns an eager frame, the pipeline
re-coerces to a `LazyFrame` so subsequent stages can rely on the lazy contract. Users that
want an eager result call `.collect()` on the returned `NarwhalsFrameResult`.

### A source for the lineitem table

`TPCHDataGenerator` is parameterized by a `TPCHTableContext` (which table to fetch). Our
pipeline wants a `NullContext`-shaped source — so we wrap it in a small adapter that pins
the table.
""")

code(r"""
class TPCHTableProvider(CallableModel):
    '''A NullContext-shaped source that pulls one fixed TPC-H table.'''

    generator: TPCHDataGenerator
    table: TPCHTable

    @Flow.call
    def __call__(self, context: NullContext) -> NarwhalsFrameResult:
        return self.generator(TPCHTableContext(table=self.table))


lineitem_provider = TPCHTableProvider(generator=generator, table="lineitem")
""")

code(r"""
q1_pipeline = NarwhalsPipelineModel(
    source=lineitem_provider,
    transforms=[
        FilterByShipDate(cutoff=datetime(1998, 9, 2)),
        DeriveLineItemMetrics(),
        AggregateByReturnStatus(),
        SortByReturnStatus(),
    ],
)

# The pipeline is a CallableModel; call it with a context to get a NarwhalsFrameResult.
q1_result = q1_pipeline(NullContext())
print("type:", type(q1_result).__name__)
print("frame is lazy:", isinstance(q1_result.df, nw.LazyFrame))
q1_result.df.collect().to_native()
""")

md(r"""
### Dependency injection: swap the source

The same transform list works against any source returning a `NarwhalsFrameResult`. To
demonstrate, here's a parquet-backed source that round-trips lineitem through disk.
""")

code(r"""
import tempfile
from pathlib import Path


class ParquetSource(CallableModel):
    '''A NullContext-shaped source that reads a parquet file lazily.'''

    path: Path

    @Flow.call
    def __call__(self, context: NullContext) -> NarwhalsFrameResult:
        return NarwhalsFrameResult(df=pl.scan_parquet(str(self.path)))


# Persist lineitem to parquet, then build a pipeline against that source.
tmpdir = Path(tempfile.mkdtemp())
parquet_path = tmpdir / "lineitem.parquet"
lineitem_eager.to_native().write_parquet(parquet_path)

q1_from_parquet = NarwhalsPipelineModel(
    source=ParquetSource(path=parquet_path),
    transforms=q1_pipeline.transforms,  # exact same transforms, different source
)

from polars.testing import assert_frame_equal as _afe
_afe(
    q1_from_parquet(NullContext()).df.collect().to_native(),
    q1_pipeline(NullContext()).df.collect().to_native(),
    check_exact=False,
)
print("DI swap OK ✓")
""")

md(r"""
### Full pipeline serialization

Because every component is a pydantic model, the entire pipeline serializes as JSON or YAML
and reloads via Hydra. Here's a JSON snapshot of just the structure:
""")

code(r"""
print(q1_pipeline.model_dump_json(indent=2)[:600], "\n  ...")
""")

md(r"""
The same data renders directly to YAML and is consumable by Hydra config files. A typical
production setup would have a `config/q1.yaml` like:

```yaml
_target_: ccflow.NarwhalsPipelineModel
source:
  _target_: my_project.TPCHTableProvider
  generator:
    _target_: ccflow.examples.tpch.TPCHDataGenerator
    scale_factor: 0.01
  table: lineitem
transforms:
  - _target_: my_project.FilterByShipDate
    cutoff: 1998-09-02
  - _target_: my_project.DeriveLineItemMetrics
  - _target_: my_project.AggregateByReturnStatus
  - _target_: my_project.SortByReturnStatus
```

(See ccflow's Hydra integration docs for the full config-file workflow.)

### Graph awareness

`NarwhalsPipelineModel` declares its source as an explicit dependency via `__deps__`,
so ccflow's graph evaluator sees the edge:
""")

code(r"""
deps = q1_pipeline.__deps__(NullContext())
print(deps)
""")

# ============================================================================
# §7. JoinTransform
# ============================================================================

md(r"""
## 6. Multi-source enrichment with `JoinTransform`

Most queries involve more than one table. `JoinTransform` is a transform that takes the
input frame and joins another `CallableModel`'s frame onto it.

The signature mirrors `LazyFrame.join`. The "other" model is invoked with `NullContext()`
— this matches the common case where the secondary input is independent of any
pipeline-level context. Both same-named (`on=...`) and cross-named (`left_on=..., right_on=...`)
join key forms are supported.

### TPC-H Q3

Q3 picks the top 10 customer-segment shipments unshipped before a given date and groups
them by order. Three tables: customer, orders, lineitem.
""")

code(r"""
# Three providers, one per table.
customer_provider = TPCHTableProvider(generator=generator, table="customer")
orders_provider = TPCHTableProvider(generator=generator, table="orders")
lineitem_provider_q3 = TPCHTableProvider(generator=generator, table="lineitem")


# Filters and derivations as transforms.
class FilterCustomerSegment(NarwhalsFrameTransform):
    segment: str

    def __call__(self, df):
        return df.filter(nw.col("c_mktsegment") == self.segment)


class FilterOrderDateBefore(NarwhalsFrameTransform):
    cutoff: datetime

    def __call__(self, df):
        return df.filter(nw.col("o_orderdate") < self.cutoff)


class FilterShipDateAfter(NarwhalsFrameTransform):
    cutoff: datetime

    def __call__(self, df):
        return df.filter(nw.col("l_shipdate") > self.cutoff)


class DeriveRevenue(NarwhalsFrameTransform):
    def __call__(self, df):
        return df.with_columns((nw.col("l_extendedprice") * (1 - nw.col("l_discount"))).alias("revenue"))


class AggregateOrderRevenue(NarwhalsFrameTransform):
    def __call__(self, df):
        return (
            df.group_by(["o_orderkey", "o_orderdate", "o_shippriority"])
            .agg(nw.sum("revenue"))
            .select(
                nw.col("o_orderkey").alias("l_orderkey"),
                "revenue",
                "o_orderdate",
                "o_shippriority",
            )
            .sort(by=["revenue", "o_orderdate"], descending=[True, False])
            .head(10)
        )


cutoff = datetime(1995, 3, 15)

q3_pipeline = NarwhalsPipelineModel(
    source=customer_provider,
    transforms=[
        FilterCustomerSegment(segment="BUILDING"),
        JoinTransform(other=orders_provider, left_on="c_custkey", right_on="o_custkey", how="inner"),
        JoinTransform(other=lineitem_provider_q3, left_on="o_orderkey", right_on="l_orderkey", how="inner"),
        FilterOrderDateBefore(cutoff=cutoff),
        FilterShipDateAfter(cutoff=cutoff),
        DeriveRevenue(),
        AggregateOrderRevenue(),
    ],
)

q3_pipeline(NullContext()).df.collect().to_native()
""")

md(r"""
We can sanity-check this against the canonical Q3 (which uses the raw narwhals function form):
""")

code(r"""
from ccflow.examples.tpch.queries import q3 as canonical_q3

canonical = canonical_q3.query(
    customer_provider(NullContext()).df.lazy(),
    lineitem_provider_q3(NullContext()).df.lazy(),
    orders_provider(NullContext()).df.lazy(),
).collect().to_native()

ours = q3_pipeline(NullContext()).df.collect().to_native()
from polars.testing import assert_frame_equal as _afe
_afe(canonical, ours, check_exact=False)
print("Q3 matches canonical ✓")
""")

# ============================================================================
# §8. Confluence
# ============================================================================

md(r"""
## 7. Confluence: pipelines as inputs to pipelines

Now level up: each side input becomes its own `NarwhalsPipelineModel`. The "main" pipeline
joins them together. This is the **confluence** pattern.

```text
   customer_provider                 orders_provider           lineitem_provider
          │                                  │                          │
          ▼                                  ▼                          ▼
   customer_pipeline                 orders_pipeline             lineitem_pipeline
   (filter segment)                 (raw, or pre-filter)         (raw, or pre-derive)
          │                                  │                          │
          └────────────── JoinTransform ───┴────── JoinTransform ───┘
                                             │
                                             ▼
                                   aggregate / sort / head
                                             │
                                             ▼
                                       Q3 result
```

Why bother? Now each side input is independently:

- **Testable** — run `customer_pipeline(NullContext())` on its own.
- **Configurable** — swap the segment via config without touching the join code.
- **Reusable** — drop the orders pre-filter into another query unchanged.
- **Cacheable** — ccflow's evaluator can memoize each pipeline's result if used in multiple places.
""")

code(r"""
# Step 1: build per-table pipelines.
customer_filtered_pipeline = NarwhalsPipelineModel(
    source=customer_provider,
    transforms=[FilterCustomerSegment(segment="BUILDING")],
)

orders_filtered_pipeline = NarwhalsPipelineModel(
    source=orders_provider,
    transforms=[FilterOrderDateBefore(cutoff=cutoff)],
)

lineitem_with_revenue_pipeline = NarwhalsPipelineModel(
    source=lineitem_provider_q3,
    transforms=[FilterShipDateAfter(cutoff=cutoff), DeriveRevenue()],
)

# Step 2: glue them together via JoinTransform.
q3_confluence = NarwhalsPipelineModel(
    source=customer_filtered_pipeline,
    transforms=[
        JoinTransform(other=orders_filtered_pipeline, left_on="c_custkey", right_on="o_custkey", how="inner"),
        JoinTransform(other=lineitem_with_revenue_pipeline, left_on="o_orderkey", right_on="l_orderkey", how="inner"),
        AggregateOrderRevenue(),
    ],
)

# Step 3: verify against canonical.
ours_conf = q3_confluence(NullContext()).df.collect().to_native()
from polars.testing import assert_frame_equal as _afe
_afe(ours_conf, canonical, check_exact=False)
print("Q3 confluence matches canonical ✓")
ours_conf
""")

md(r"""
Notice the structural shift compared to §6:

- Each pre-filter / derivation lives **inside** the pipeline that owns the relevant table,
  rather than being mixed into a single linear list.
- The "main" pipeline now describes only the join-and-aggregate logic.
- The graph is properly a tree: ccflow's `__deps__` tracks the source edge, and (in v1)
  the side inputs invoked by `JoinTransform` are not auto-surfaced — but they remain
  configurable and serializable.
""")

# ============================================================================
# §9. JoinBackTransform + schema validation aside
# ============================================================================

md(r"""
## 8. `JoinBackTransform`: fork-and-rejoin patterns

Sometimes a transform needs to compute a per-group summary and **enrich** the original rows
with it. Two tools cover the spectrum:

- **`with_columns(expr.over(window))`** — the right answer when the summary fits a window
  function (same row count as input). Prefer this when it applies.
- **`JoinBackTransform`** — runs an inner transform on the input to produce a summary
  with potentially different shape, then joins it back. Useful when the summary aggregates
  across groups, or projects to a different column set.
""")

code(r"""
# Toy example: enrich each line item with the total revenue of its order.
toy_lineitem = nw.from_native(pl.LazyFrame({
    "l_orderkey": [1, 1, 2, 2, 3],
    "l_extendedprice": [100.0, 50.0, 200.0, 80.0, 30.0],
    "l_discount": [0.0, 0.1, 0.0, 0.05, 0.0],
}))


class TotalRevenuePerOrder(NarwhalsFrameTransform):
    '''Inner transform: compute total revenue per order key.'''

    def __call__(self, df):
        return (
            df.with_columns(revenue=nw.col("l_extendedprice") * (1 - nw.col("l_discount")))
            .group_by("l_orderkey")
            .agg(nw.sum("revenue").alias("order_total_revenue"))
        )


# JoinBackTransform threads through .pipe() like any other transform.
enrich_with_order_total = JoinBackTransform(inner=TotalRevenuePerOrder(), on="l_orderkey", how="left")

toy_lineitem.pipe(enrich_with_order_total).collect().to_native()
""")

md(r"""
For per-row deltas (e.g. "this line item's revenue minus its order's average"), the
window-function form is cleaner:

```python
df.with_columns(
    delta=(nw.col("l_extendedprice") * (1 - nw.col("l_discount")))
          - (nw.col("l_extendedprice") * (1 - nw.col("l_discount"))).mean().over("l_orderkey")
)
```

`JoinBackTransform` earns its place when the summary's shape genuinely differs from the
input — e.g. multi-column projections, or aggregations across multiple keys.

### Aside: schema validation

A natural follow-on is "validate the frame's schema mid-pipeline". This is intentionally
left for a future addition.

The right semantics is for validation to participate in narwhals' lazy schema inspection
(`collect_schema()`) — i.e. to be checked *during planning*, not at construction time. As of
this writing, the underlying primitive (`pipe_with_schema`) is unstable in polars and not
yet present in `narwhals.stable.v1`. Once it lands upstream, ccflow will ship a
`NarwhalsSchemaTransform` base class. In the meantime, an eager user-defined validator
called inside `__call__` (i.e. `value.collect_schema()`) is the workable workaround — at the
cost of forcing a partial schema-resolution at every call.
""")

# ============================================================================
# §10. When not to use
# ============================================================================

md(r"""
## 9. When *not* to use `NarwhalsPipelineModel`

The linear `.pipe()` contract is the design's strength — but it's also a constraint.
Some queries don't fit:

- **True DAGs.** TPC-H Q15 (CTE referenced from two places) and Q21 (correlated EXISTS)
  share intermediate frames across multiple downstream consumers. Forcing them through
  a single `.pipe()` chain duplicates work.
- **Scalar subqueries.** Q4, Q11, Q17, Q22 compute a scalar (or a tiny frame) from one
  query and use it as a parameter in another. The "pipeline" view is the wrong abstraction.
- **Multi-output stages.** Anything that wants to fork into multiple labeled outputs
  (e.g. `frame -> {"by_segment": ..., "by_region": ...}`) breaks the `LazyFrame -> LazyFrame`
  contract.

The right move for these is to step **up** a layer: split the query into multiple
`CallableModel`s, compose them at the graph level, and let ccflow's evaluator handle
parallelization and caching of shared upstream computations. The pipeline abstractions in
this notebook live *inside* each node of that graph; they're not a replacement for it.
""")

# ============================================================================
# §11. Summary
# ============================================================================

md(r"""
## 10. Summary

Three layers, each opt-in:

1. **`NarwhalsFrameTransform`** — pydantic + callable + `.pipe()`-compatible. Use anywhere
   narwhals is used; no ccflow runtime required by callers.
2. **`SequenceTransform`** — bundle. Strictly typed, JSON-roundtrippable, nestable.
3. **`NarwhalsPipelineModel`** — full ccflow integration: sources, contexts, dependency
   declarations, evaluator, caching, Hydra config.

Plus two ready-made transforms:

- **`JoinTransform`** — multi-source enrichment with another `CallableModel`.
- **`JoinBackTransform`** — fork-and-rejoin when window functions don't fit.

When pipelines compose pipelines, you get **confluence** — small, testable units glued
together at the join boundary, each independently configurable and (in future) cacheable.

When the query exceeds linear `.pipe()`, step up to multi-`CallableModel` graphs.

### Pointers

- Source: `ccflow/models/narwhals.py`
- Tests: `ccflow/tests/models/test_narwhals.py`
- TPC-H assets: `ccflow/examples/tpch/`
- Canonical Q3: `ccflow/examples/tpch/queries/q3.py`
- Result types: `ccflow/result/narwhals.py`
""")

nb["cells"] = cells

import os

out = os.path.join("ccflow", "examples", "narwhals_pipelines.ipynb")
nbf.write(nb, out)
print(f"wrote {out} ({len(cells)} cells)")
