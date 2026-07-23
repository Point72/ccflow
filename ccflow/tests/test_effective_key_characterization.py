"""Characterization tests: effective identity must not change existing models.

The dependency-graph builder and cache-key path now route every node through
``_effective_evaluation_key()``. For models that do NOT opt into effective identity
(everything except generated ``@Flow.model`` / ``BoundModel``), the result must remain
byte-for-byte identical to the structural ``cache_key()``. These tests pin that
equivalence so future changes to the effective path cannot silently shift cache or graph
identity for ordinary ``CallableModel`` graphs.
"""

from datetime import date

from ccflow import DateContext
from ccflow.evaluators.common import (
    _build_dependency_graph,
    _effective_evaluation_key,
    cache_key,
    get_dependency_graph,
)

from .evaluators.util import NodeModel


def _structural_graph(evaluation_context):
    """Build a graph using only the structural cache_key (the pre-effective behavior)."""
    from ccflow.evaluators.common import CallableModelGraph

    graph = CallableModelGraph(ids={}, graph={}, root_id=cache_key(evaluation_context))

    def walk(ctx, parent_key=None):
        key = cache_key(ctx)
        if parent_key:
            graph.graph[parent_key].add(key)
        if key not in graph.ids:
            graph.ids[key] = ctx
        if key not in graph.graph:
            graph.graph[key] = set()
            for model, contexts in ctx.model.__deps__(ctx.context):
                for context in contexts:
                    walk(model.__call__.get_evaluation_context(model, context), parent_key=key)

    walk(evaluation_context)
    return graph


def _models():
    ctx = DateContext(date=date(2022, 1, 1))
    leaf = NodeModel(meta={"name": "leaf"})
    # diamond: root -> n1, n2 -> shared leaf
    n1 = NodeModel(meta={"name": "n1"}, deps_model=[leaf])
    n2 = NodeModel(meta={"name": "n2"}, deps_model=[leaf])
    diamond = NodeModel(meta={"name": "root"}, deps_model=[n1, n2])
    # simple chain: a -> b -> c
    c = NodeModel(meta={"name": "c"})
    b = NodeModel(meta={"name": "b"}, deps_model=[c])
    a = NodeModel(meta={"name": "a"}, deps_model=[b])
    return ctx, {"leaf": leaf, "diamond": diamond, "chain": a}


class TestEffectiveKeyEqualsStructural:
    def test_cache_key_effective_equals_structural(self):
        ctx, models = _models()
        for model in models.values():
            evaluation = model.__call__.get_evaluation_context(model, ctx)
            assert cache_key(evaluation, effective=True) == cache_key(evaluation, effective=False)
            assert _effective_evaluation_key(evaluation) == cache_key(evaluation)

    def test_dependency_graph_matches_structural(self):
        ctx, models = _models()
        for name, model in models.items():
            evaluation = model.__call__.get_evaluation_context(model, ctx)
            effective_graph = get_dependency_graph(evaluation)
            structural_graph = _structural_graph(evaluation)
            assert effective_graph.root_id == structural_graph.root_id, name
            assert set(effective_graph.graph) == set(structural_graph.graph), name
            assert {k: set(v) for k, v in effective_graph.graph.items()} == {k: set(v) for k, v in structural_graph.graph.items()}, name

    def test_diamond_dedupes_shared_leaf(self):
        ctx, models = _models()
        graph = get_dependency_graph(models["diamond"].__call__.get_evaluation_context(models["diamond"], ctx))
        # 4 distinct nodes: root, n1, n2, shared leaf.
        assert len(graph.graph) == 4

    def test_build_dependency_graph_returns_structural_root(self):
        from ccflow.evaluators.common import CallableModelGraph

        ctx, models = _models()
        evaluation = models["diamond"].__call__.get_evaluation_context(models["diamond"], ctx)
        graph = CallableModelGraph(ids={}, graph={}, root_id=b"")
        root = _build_dependency_graph(evaluation, graph)
        assert root == cache_key(evaluation)
