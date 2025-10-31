import json
from datetime import date

from ccflow import DateContext
from ccflow.callable import ModelEvaluationContext
from ccflow.evaluators import GraphEvaluator, LoggingEvaluator, MultiEvaluator
from ccflow.tests.evaluators.util import NodeModel

# NOTE: for these tests, round-tripping via JSON does not work
# because the ModelEvaluationContext just has an InstanceOf validation check
# and so we do not actually construct a full MEC on load.


def _make_nested_mec(model):
    ctx = DateContext(date=date(2022, 1, 1))
    mec = model.__call__.get_evaluation_context(model, ctx)
    assert isinstance(mec, ModelEvaluationContext)
    # ensure nested: outer model is an evaluator, inner is a ModelEvaluationContext
    assert isinstance(mec.context, ModelEvaluationContext)
    return mec


def test_mec_model_dump_basic():
    m = NodeModel()
    mec = _make_nested_mec(m)

    d = mec.model_dump()
    assert isinstance(d, dict)
    assert "fn" in d and "model" in d and "context" in d and "options" in d

    s = mec.model_dump_json()
    parsed = json.loads(s)
    assert parsed["fn"] == d["fn"]
    # Also verify mode-specific dumps
    d_py = mec.model_dump(mode="python")
    assert isinstance(d_py, dict)
    d_json = mec.model_dump(mode="json")
    assert isinstance(d_json, dict)
    json.dumps(d_json)


def test_mec_model_dump_diamond_graph():
    n0 = NodeModel()
    n1 = NodeModel(deps_model=[n0])
    n2 = NodeModel(deps_model=[n0])
    root = NodeModel(deps_model=[n1, n2])

    mec = _make_nested_mec(root)

    d = mec.model_dump()
    assert isinstance(d, dict)
    assert set(["fn", "model", "context", "options"]).issubset(d.keys())

    s = mec.model_dump_json()
    json.loads(s)
    # verify mode dumps
    d_py = mec.model_dump(mode="python")
    assert isinstance(d_py, dict)
    d_json = mec.model_dump(mode="json")
    assert isinstance(d_json, dict)
    json.dumps(d_json)


def test_mec_model_dump_with_multi_evaluator():
    m = NodeModel()
    _ = LoggingEvaluator()  # ensure import/validation
    evaluator = MultiEvaluator(evaluators=[LoggingEvaluator(), GraphEvaluator()])

    # Simulate how Flow builds evaluation context with a custom evaluator
    ctx = DateContext(date=date(2022, 1, 1))
    mec = ModelEvaluationContext(model=evaluator, context=m.__call__.get_evaluation_context(m, ctx))

    d = mec.model_dump()
    assert isinstance(d, dict)
    assert "fn" in d and "model" in d and "context" in d
    s = mec.model_dump_json()
    json.loads(s)
    # verify mode dumps
    d_py = mec.model_dump(mode="python")
    assert isinstance(d_py, dict)
    d_json = mec.model_dump(mode="json")
    assert isinstance(d_json, dict)
    json.dumps(d_json)
