import logging
import pandas as pd
from datetime import date
from unittest import TestCase

from ccflow import DateContext, DateRangeContext, Evaluator, FlowOptionsOverride, ModelEvaluationContext
from ccflow.evaluators import (
    GraphEvaluator,
    LoggingEvaluator,
    MemoryCacheEvaluator,
    MultiEvaluator,
    cache_key,
    combine_evaluators,
    get_dependency_graph,
)

from .util import CircularModel, MyDateCallable, MyDateRangeCallable, MyRaisingCallable, NodeModel


class TestCombineEvaluator(TestCase):
    def test_combine_evaluators(self):
        evaluator1 = LoggingEvaluator(log_level=logging.DEBUG)
        evaluator2 = LoggingEvaluator(log_level=logging.INFO)
        evaluator3 = LoggingEvaluator(log_level=logging.WARNING)
        evaluator4 = LoggingEvaluator(log_level=logging.ERROR)
        self.assertIs(combine_evaluators(None, None), None)
        self.assertIs(combine_evaluators(evaluator1, None), evaluator1)
        self.assertIs(combine_evaluators(None, evaluator2), evaluator2)
        self.assertEqual(combine_evaluators(evaluator1, evaluator2), MultiEvaluator(evaluators=[evaluator1, evaluator2]))
        self.assertEqual(
            combine_evaluators(MultiEvaluator(evaluators=[evaluator1, evaluator2]), evaluator3),
            MultiEvaluator(evaluators=[evaluator1, evaluator2, evaluator3]),
        )
        self.assertEqual(
            combine_evaluators(evaluator1, MultiEvaluator(evaluators=[evaluator2, evaluator3])),
            MultiEvaluator(evaluators=[evaluator1, evaluator2, evaluator3]),
        )
        self.assertEqual(
            combine_evaluators(MultiEvaluator(evaluators=[evaluator1, evaluator2]), MultiEvaluator(evaluators=[evaluator3, evaluator4])),
            MultiEvaluator(evaluators=[evaluator1, evaluator2, evaluator3, evaluator4]),
        )

    def test_multi_evaluator(self):
        # Test that the pass-throughs work correctly
        evaluator1 = Evaluator()
        evaluator2 = Evaluator()
        m1 = MyDateCallable(offset=1)
        context = DateContext(date=date(2022, 1, 1))
        model_evaluation_context = ModelEvaluationContext(model=m1, context=context)
        multi_evaluator = MultiEvaluator(evaluators=[evaluator1, evaluator2])
        self.assertEqual(multi_evaluator(model_evaluation_context), m1(context))

        deps_model_context = ModelEvaluationContext(model=m1, context=context, fn="__deps__")
        self.assertEqual(multi_evaluator(deps_model_context), m1.__deps__(context))
        self.assertEqual(multi_evaluator.__deps__(model_evaluation_context), m1.__deps__(context))


class TestLoggingEvaluator(TestCase):
    def test_logging(self):
        m1 = MyDateCallable(offset=1)
        context = DateContext(date=date(2022, 1, 1))
        model_evaluation_context = ModelEvaluationContext(model=m1, context=context)
        evaluator = LoggingEvaluator(log_level=logging.INFO)
        with self.assertLogs(level=logging.INFO) as captured:
            evaluator(model_evaluation_context)
            evaluator(model_evaluation_context)
        self.assertEqual(len(captured.records), 6)
        for i in range(6):
            self.assertEqual(captured.records[i].levelno, logging.INFO)
        self.assertIn("Start evaluation of __call__", captured.records[0].getMessage())
        self.assertIn("MyDateCallable", captured.records[1].getMessage())
        self.assertIn("End evaluation of __call__", captured.records[2].getMessage())
        self.assertIn("time elapsed", captured.records[2].getMessage())

    def test_logging_deps(self):
        m1 = MyDateCallable(offset=1)
        context = DateContext(date=date(2022, 1, 1))
        model_evaluation_context = ModelEvaluationContext(model=m1, context=context)
        evaluator = LoggingEvaluator(log_level=logging.INFO)
        with self.assertLogs(level=logging.INFO) as captured:
            evaluator.__deps__(model_evaluation_context)
            evaluator.__deps__(model_evaluation_context)
        self.assertEqual(len(captured.records), 6)
        for i in range(6):
            self.assertEqual(captured.records[i].levelno, logging.INFO)
        self.assertIn("Start evaluation of __deps__", captured.records[0].getMessage())
        self.assertIn("MyDateCallable", captured.records[1].getMessage())
        self.assertIn("End evaluation of __deps__", captured.records[2].getMessage())
        self.assertIn("time elapsed", captured.records[2].getMessage())

    def test_logging_exception(self):
        """Test that even when an evaluation fails, both start and end still get logged."""
        m1 = MyRaisingCallable(offset=1)
        context = DateContext(date=date(2022, 1, 1))
        model_evaluation_context = ModelEvaluationContext(model=m1, context=context)
        evaluator = LoggingEvaluator(log_level=logging.INFO)
        with self.assertLogs(level=logging.INFO) as captured:
            self.assertRaises(ValueError, evaluator, model_evaluation_context)

        self.assertEqual(len(captured.records), 3)
        for i in range(3):
            self.assertEqual(captured.records[i].levelno, logging.INFO)
        self.assertIn("Start evaluation of __call__", captured.records[0].getMessage())
        self.assertIn("MyRaisingCallable", captured.records[1].getMessage())
        self.assertIn("End evaluation of __call__", captured.records[2].getMessage())
        self.assertIn("time elapsed", captured.records[2].getMessage())

    def test_verbose(self):
        m1 = MyDateCallable(offset=1)
        context = DateContext(date=date(2022, 1, 1))
        model_evaluation_context = ModelEvaluationContext(model=m1, context=context)
        evaluator = LoggingEvaluator(log_level=logging.INFO, verbose=False)
        with self.assertLogs(level=logging.INFO) as captured:
            evaluator(model_evaluation_context)
            evaluator(model_evaluation_context)
        self.assertEqual(len(captured.records), 4)
        for i in range(4):
            self.assertEqual(captured.records[i].levelno, logging.INFO)
        self.assertIn("Start evaluation of __call__", captured.records[0].getMessage())
        self.assertIn("End evaluation of __call__", captured.records[1].getMessage())
        self.assertIn("time elapsed", captured.records[1].getMessage())

    def test_logging_options_nested(self):
        m1 = MyDateCallable(offset=1)
        context = DateContext(date=date(2022, 1, 1))
        # Test that the logging options are applied even when the logging evaluator is nested in a Multi evaluator
        with FlowOptionsOverride(options={"evaluator": MultiEvaluator(evaluators=[LoggingEvaluator()]), "log_level": logging.INFO}):
            with self.assertLogs(level=logging.INFO) as captured:
                m1(context)
                m1(context)
        self.assertEqual(len(captured.records), 6)
        for i in range(6):
            self.assertEqual(captured.records[i].levelno, logging.INFO)
        self.assertIn("Start evaluation of __call__", captured.records[0].getMessage())
        self.assertIn("MyDateCallable", captured.records[1].getMessage())
        self.assertIn("End evaluation of __call__", captured.records[2].getMessage())
        self.assertIn("time elapsed", captured.records[2].getMessage())


class SubContext(DateContext):
    pass


class SubMyDateCallable(MyDateCallable):
    pass


class TestCacheKey(TestCase):
    def test_context(self):
        context = DateContext(date=date(2022, 1, 1))
        context2 = DateContext(date=date(2022, 1, 1))
        context3 = DateContext(date=date(2022, 1, 2))
        assert cache_key(context) == cache_key(context2)
        assert cache_key(context) != cache_key(context3)

        subcontext = SubContext(date=date(2022, 1, 1))
        assert cache_key(subcontext) != cache_key(context)

    def test_model(self):
        any = {
            "a": [1, 2, set(), float("nan")],
            "b": pd.DataFrame(data=100.0, columns=["a", "b"], index=[1, 2, 3]),
            "c": MyDateCallable(offset=-1),
        }
        m1 = MyDateCallable(offset=1, any=any)
        m2 = MyDateCallable(offset=1, any=any)
        assert cache_key(m1) == cache_key(m2)
        m3 = SubMyDateCallable(offset=1, any=any)
        assert cache_key(m3) != cache_key(m1)

    def test_model_evaluation_context(self):
        c1 = DateContext(date=date(2022, 1, 1))
        c2 = DateContext(date=date(2022, 1, 1))
        m1 = MyDateCallable(offset=1)
        m2 = MyDateCallable(offset=1)
        m3 = SubMyDateCallable(offset=1)
        mec1 = ModelEvaluationContext(model=m1, context=c1)
        mec2 = ModelEvaluationContext(model=m2, context=c2)
        mec3 = ModelEvaluationContext(model=m3, context=c1)
        assert cache_key(mec1) == cache_key(mec2)
        assert cache_key(mec3) != cache_key(mec1)


class TestMemoryCacheEvaluator(TestCase):
    def test_basic(self):
        m1 = MyDateCallable(offset=1)
        evaluator = MemoryCacheEvaluator()
        context = DateContext(date=date(2022, 1, 1))
        model_evaluation_context = ModelEvaluationContext(model=m1, context=context, options=dict(cacheable=True))

        out = evaluator(model_evaluation_context)
        target = {evaluator.key(model_evaluation_context): out}
        self.assertEqual(evaluator.cache, target)
        ids = {evaluator.key(model_evaluation_context): model_evaluation_context}
        self.assertEqual(evaluator.ids, ids)

        context = DateContext(date=date(2022, 1, 2))
        model_evaluation_context = ModelEvaluationContext(model=m1, context=context, options=dict(cacheable=True))
        out2 = evaluator(model_evaluation_context)
        target[evaluator.key(model_evaluation_context)] = out2
        self.assertEqual(evaluator.cache, target)
        ids[evaluator.key(model_evaluation_context)] = model_evaluation_context
        self.assertEqual(evaluator.ids, ids)

        context = DateContext(date=date(2022, 1, 3))
        model_evaluation_context = ModelEvaluationContext(model=m1, context=context, options=dict(cacheable=True, volatile=True))
        evaluator(model_evaluation_context)
        key = evaluator.key(model_evaluation_context)
        self.assertNotIn(key, evaluator.cache)
        self.assertNotIn(key, evaluator.ids)

        context = DateContext(date=date(2022, 1, 4))
        model_evaluation_context = ModelEvaluationContext(model=m1, context=context, options=dict(cacheable=False))
        evaluator(model_evaluation_context)
        key = evaluator.key(model_evaluation_context)
        self.assertNotIn(key, evaluator.cache)
        self.assertNotIn(key, evaluator.ids)

    def test_caching(self):
        # Create some hard-to hash structure with all kinds of custom types
        # We will put this on the callable to make sure caching still works
        # It is a stress-test of the implementation of MemoryCacheEvaluator.key
        any = {
            "a": [1, 2, set(), float("nan")],
            "b": pd.DataFrame(data=100.0, columns=["a", "b"], index=[1, 2, 3]),
            "c": MyDateCallable(offset=-1),
        }
        m1 = MyDateCallable(offset=1, any=any)
        m2 = MyDateCallable(offset=1, any=any)
        m3 = MyDateRangeCallable(model1=m1, model2=m2)

        # Apply logging first, then apply the memory cache on top
        # (i.e. only calls to the original model will be logged)
        evaluators = [LoggingEvaluator(log_level=logging.INFO), MemoryCacheEvaluator()]
        evaluator = MultiEvaluator(evaluators=evaluators)
        context = DateRangeContext(start_date=date(2022, 1, 1), end_date=date(2022, 1, 3))
        # We apply this evaluator to all the call functions via the decorator
        with FlowOptionsOverride(options={"evaluator": evaluator, "cacheable": True}):
            with self.assertLogs(level=logging.INFO) as captured:
                out = m3(context)
                self.assertEqual(out.x, 18)
        # Without caching, there would be 3+3+1=7 calls, but some of these can be re-used,
        # so there should be only 3+1=4 calls
        start_records = [r.getMessage() for r in captured.records if "Start evaluation of __call__" in r.getMessage()]
        self.assertEqual(len(start_records), 4)

    def test_caching_of_deps(self):
        any = {
            "a": [1, 2],
            "c": MyDateCallable(offset=-1),
        }
        m1 = MyDateCallable(offset=1, any=any)
        m2 = MyDateCallable(offset=1, any=any)
        m3 = MyDateRangeCallable(model1=m1, model2=m2)

        # Apply logging first, then apply the memory cache on top
        # (i.e. only calls to the original model will be logged)
        evaluators = [LoggingEvaluator(log_level=logging.INFO), MemoryCacheEvaluator()]
        evaluator = MultiEvaluator(evaluators=evaluators)
        context = DateRangeContext(start_date=date(2022, 1, 1), end_date=date(2022, 1, 3))
        target_deps = m3.__deps__(context)
        with FlowOptionsOverride(options={"evaluator": evaluator, "cacheable": True}):
            with self.assertLogs(level=logging.INFO) as captured:
                out = m3.__deps__(context)
                self.assertEqual(out, target_deps)
                out = m3.__deps__(context)
        # Without caching, there would be 2 calls, but there should only be 1
        start_records = [r.getMessage() for r in captured.records if "Start evaluation of __deps__" in r.getMessage()]
        self.assertEqual(len(start_records), 1)

    def test_decorator_volatile(self):
        m1 = MyDateCallable(offset=1)
        # Even though we specify cacheable=True, the volatile=True flag on current_time takes precedence.
        with FlowOptionsOverride(options={"evaluator": MemoryCacheEvaluator(), "cacheable": True}):
            with self.assertLogs(level=logging.INFO) as captured:
                out1 = m1.current_time(DateContext(date=date(2022, 1, 1)))
                out2 = m1.current_time(DateContext(date=date(2022, 1, 1)))
                self.assertGreater(out2, out1)
        self.assertEqual(len(captured.records), 2)


class TestGraphDeps(TestCase):
    def test_graph_deps_diamond(self):
        n0 = NodeModel(meta=dict(name="n0"))
        n1 = NodeModel(meta=dict(name="n1"), deps_model=[n0])
        n2 = NodeModel(meta=dict(name="n2"), deps_model=[n0])
        root = NodeModel(meta=dict(name="n3"), deps_model=[n1, n2])
        context = DateContext(date=date(2022, 1, 1))
        graph = get_dependency_graph(ModelEvaluationContext(model=root, context=context))
        self.assertEqual(graph.ids.keys(), graph.graph.keys())
        self.assertEqual(len(graph.ids), 4)
        for k, v in graph.ids.items():
            if v.model.meta.name == "n3":
                self.assertEqual(set(graph.ids[dep_key].context.model.meta.name for dep_key in graph.graph[k]), set(["n1", "n2"]))
            elif v.model.meta.name in ("n1", "n2"):
                self.assertEqual(set(graph.ids[dep_key].context.model.meta.name for dep_key in graph.graph[k]), set(["n0"]))
            elif v.model.meta.name == "n0":
                self.assertEqual(set(graph.ids[dep_key].context.model.meta.name for dep_key in graph.graph[k]), set())

    def test_graph_deps_circular(self):
        root = CircularModel()
        context = DateContext(date=date(2022, 1, 1))
        graph = get_dependency_graph(root.__call__.get_evaluation_context(root, context))
        self.assertEqual(len(graph.graph), 1)
        key = list(graph.graph.keys())[0]
        self.assertEqual(graph.graph[key], set([key]))


class TestGraphEvaluator(TestCase):
    def test_graph_evaluator_basic(self):
        n0 = NodeModel(meta=dict(name="n0"))
        n1 = NodeModel(meta=dict(name="n1"))
        n2 = NodeModel(meta=dict(name="n2"), deps_model=[n0, n1])
        context = DateContext(date=date(2022, 1, 1))

        NodeModel._calls = []
        NodeModel._deps_calls = []
        n2(context)
        original_calls = NodeModel._calls
        self.assertEqual(NodeModel._deps_calls, [])

        # We apply this evaluator to all the call functions via the decorator
        evaluator = GraphEvaluator()
        NodeModel._calls = []
        with FlowOptionsOverride(options={"evaluator": evaluator}):
            n2(context)
        graph_calls = NodeModel._calls
        deps_calls = NodeModel._deps_calls

        self.assertEqual(deps_calls, [("n2", date(2022, 1, 1)), ("n0", date(2022, 1, 1)), ("n1", date(2022, 1, 1))])
        self.assertEqual(original_calls, [("n2", date(2022, 1, 1))])
        self.assertEqual(len(graph_calls), 3)
        self.assertEqual(graph_calls[-1], ("n2", date(2022, 1, 1)))
        self.assertIn(("n0", date(2022, 1, 1)), graph_calls[:2])
        self.assertIn(("n1", date(2022, 1, 1)), graph_calls[:2])

    def test_graph_evaluator_diamond(self):
        n0 = NodeModel(meta=dict(name="n0"))
        n1 = NodeModel(meta=dict(name="n1"), deps_model=[n0])
        n2 = NodeModel(meta=dict(name="n2"), deps_model=[n0])
        root = NodeModel(meta=dict(name="n3"), deps_model=[n1, n2])
        context = DateContext(date=date(2022, 1, 1))

        NodeModel._calls = []
        NodeModel._deps_calls = []
        root(context)
        original_calls = NodeModel._calls
        self.assertEqual(NodeModel._deps_calls, [])

        # We apply this evaluator to the root model only (as we don't need each model to be sorting its dependencies)
        evaluator = GraphEvaluator()
        NodeModel._calls = []
        with FlowOptionsOverride(options={"evaluator": evaluator}):
            root(context)
        graph_calls = NodeModel._calls
        deps_calls = NodeModel._deps_calls

        self.assertEqual(
            deps_calls,
            [("n3", date(2022, 1, 1)), ("n1", date(2022, 1, 1)), ("n0", date(2022, 1, 1)), ("n2", date(2022, 1, 1))],
        )
        self.assertEqual(original_calls, [("n3", date(2022, 1, 1))])
        self.assertEqual(len(graph_calls), 4)
        self.assertEqual(graph_calls[0], ("n0", date(2022, 1, 1)))
        self.assertEqual(graph_calls[-1], ("n3", date(2022, 1, 1)))
        self.assertIn(("n1", date(2022, 1, 1)), graph_calls[1:3])
        self.assertIn(("n2", date(2022, 1, 1)), graph_calls[1:3])

    def test_graph_evaluator_cache(self):
        """Test that stacking the graph evaluator with caching/logging works as expected."""
        n0 = NodeModel(meta=dict(name="n0"), run_deps=True)
        n1 = NodeModel(meta=dict(name="n1"), deps_model=[n0], run_deps=True)
        n2 = NodeModel(meta=dict(name="n2"), deps_model=[n0], run_deps=True)
        root = NodeModel(meta=dict(name="n3"), deps_model=[n1, n2], run_deps=True)
        context = DateContext(date=date(2022, 1, 1))

        evaluators = [LoggingEvaluator(log_level=logging.INFO), MemoryCacheEvaluator(), GraphEvaluator()]
        evaluator = MultiEvaluator(evaluators=evaluators)

        # We apply this evaluator to the root model only (as we don't need each model to be sorting its dependencies)
        NodeModel._calls = []
        NodeModel._deps_calls = []
        with FlowOptionsOverride(options={"evaluator": evaluator, "cacheable": True}):
            with self.assertLogs(level=logging.INFO) as captured:
                root(context)
        graph_calls = NodeModel._calls
        deps_calls = NodeModel._deps_calls

        self.assertEqual(len(deps_calls), 4)
        self.assertEqual(deps_calls, [("n3", date(2022, 1, 1)), ("n1", date(2022, 1, 1)), ("n0", date(2022, 1, 1)), ("n2", date(2022, 1, 1))])

        self.assertEqual(len(graph_calls), 4)
        self.assertEqual(graph_calls[0], ("n0", date(2022, 1, 1)))
        self.assertEqual(graph_calls[-1], ("n3", date(2022, 1, 1)))
        self.assertIn(("n1", date(2022, 1, 1)), graph_calls[1:3])
        self.assertIn(("n2", date(2022, 1, 1)), graph_calls[1:3])

        self.assertEqual(len(captured.records), (4 + 4) * 3)

    def test_graph_evaluator_circular(self):
        root = CircularModel()
        context = DateContext(date=date(2022, 1, 1))
        evaluator = GraphEvaluator()
        with FlowOptionsOverride(options={"evaluator": evaluator}):
            self.assertRaises(Exception, root, context)
