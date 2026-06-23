"""Hydra integration tests for the FromContext-based Flow.model API."""

from datetime import date
from pathlib import Path

from ccflow import CallableModel, FlowContext, ModelRegistry

CONFIG_PATH = str(Path(__file__).parent / "config" / "conf_flow.yaml")


def setup_function():
    ModelRegistry.root().clear()


def teardown_function():
    ModelRegistry.root().clear()


def test_basic_loader_from_yaml():
    registry = ModelRegistry.root()
    registry.load_config_from_path(CONFIG_PATH)

    loader = registry["flow_loader"]
    assert isinstance(loader, CallableModel)
    assert loader.flow.compute(value=10).value == 50


def test_registry_dependency_from_yaml():
    registry = ModelRegistry.root()
    registry.load_config_from_path(CONFIG_PATH)

    transformer = registry["flow_transformer"]
    assert transformer.source is registry["flow_source"]
    assert transformer.flow.compute(value=5).value == 315


def test_diamond_dependency_from_yaml_reuses_shared_source():
    registry = ModelRegistry.root()
    registry.load_config_from_path(CONFIG_PATH)

    aggregator = registry["diamond_aggregator"]
    assert aggregator.input_a.source is registry["diamond_source"]
    assert aggregator.input_b.source is registry["diamond_source"]
    assert aggregator.flow.compute(value=10).value == 140


def test_from_context_pipeline_from_yaml():
    registry = ModelRegistry.root()
    registry.load_config_from_path(CONFIG_PATH)

    loader = registry["contextual_loader_model"]
    processor = registry["contextual_processor_model"]

    assert processor.data is loader
    result = processor.flow.compute(FlowContext(start_date=date(2024, 3, 1), end_date=date(2024, 3, 31)))
    assert result.value == "output:data_source:2024-03-01 to 2024-03-31"
