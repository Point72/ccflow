import pickle

import pytest

from ccflow import BaseModel, EvaluatorBase, FlowOptionsOverride, GenericResult, GlobalState, ModelEvaluationContext, ModelRegistry


@pytest.fixture
def root_registry():
    r = ModelRegistry.root()
    r.clear()
    yield r
    r.clear()


class DummyModel(BaseModel):
    name: str


class DummyEvaluator(EvaluatorBase):
    def __call__(self, context: ModelEvaluationContext):
        return GenericResult(value="test")


def test_global_state(root_registry):
    root_registry.add("foo", DummyModel(name="foo"))
    evaluator = DummyEvaluator()
    with FlowOptionsOverride(options=dict(evaluator=evaluator)) as override:
        state = GlobalState()

    # Now clear the registry, and add a different model
    root_registry.clear()
    root_registry.add("bar", DummyModel(name="bar"))
    assert "foo" in state.root_registry.models
    assert "bar" not in state.root_registry.models
    assert state.open_overrides == {id(override): override}

    with state:
        state2 = GlobalState()
    assert "foo" in state2.root_registry.models
    assert "bar" not in state2.root_registry.models
    assert state2.open_overrides == {id(override): override}

    # Test that global state doesn't persist outside of the context manager
    state3 = GlobalState()
    assert "foo" not in state3.root_registry.models
    assert "bar" in state3.root_registry.models
    assert state3.open_overrides == {}


def test_global_state_pickle(root_registry):
    root_registry.add("foo", DummyModel(name="foo"))
    evaluator = DummyEvaluator()
    with FlowOptionsOverride(options=dict(evaluator=evaluator)) as override:
        state = GlobalState()

    # Now pickle and unpickle the state
    state_pickled = pickle.dumps(state)
    state_unpickled = pickle.loads(state_pickled)

    assert "foo" in state_unpickled.root_registry.models
    assert state_unpickled.open_overrides == {id(override): override}
