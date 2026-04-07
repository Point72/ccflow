"""Hydra integration tests for the FromContext-based Flow.model API."""

from datetime import date
from pathlib import Path

from omegaconf import OmegaConf

from ccflow import CallableModel, DateRangeContext, FlowContext, ModelRegistry

from .test_flow_model import SimpleContext

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
    assert loader(SimpleContext(value=10)).value == 50


def test_basic_processor_from_yaml():
    registry = ModelRegistry.root()
    registry.load_config_from_path(CONFIG_PATH)

    processor = registry["flow_processor"]
    assert processor(SimpleContext(value=42)).value == "value=42!"


def test_two_stage_pipeline_from_yaml():
    registry = ModelRegistry.root()
    registry.load_config_from_path(CONFIG_PATH)

    transformer = registry["flow_transformer"]
    assert transformer(SimpleContext(value=5)).value == 315


def test_three_stage_pipeline_from_yaml():
    registry = ModelRegistry.root()
    registry.load_config_from_path(CONFIG_PATH)

    stage3 = registry["flow_stage3"]
    assert stage3(SimpleContext(value=10)).value == 90


def test_diamond_dependency_from_yaml():
    registry = ModelRegistry.root()
    registry.load_config_from_path(CONFIG_PATH)

    aggregator = registry["diamond_aggregator"]
    assert aggregator(SimpleContext(value=10)).value == 140


def test_date_range_pipeline_from_yaml():
    registry = ModelRegistry.root()
    registry.load_config_from_path(CONFIG_PATH)

    processor = registry["flow_date_processor"]
    ctx = DateRangeContext(start_date=date(2024, 1, 10), end_date=date(2024, 1, 31))
    result = processor(ctx)

    assert "normalized:" in result.value
    assert "2024-01-09" in result.value


def test_from_context_pipeline_from_yaml():
    registry = ModelRegistry.root()
    registry.load_config_from_path(CONFIG_PATH)

    loader = registry["contextual_loader_model"]
    processor = registry["contextual_processor_model"]

    assert loader.flow.context_inputs == {"start_date": date, "end_date": date}
    result = processor.flow.compute(start_date=date(2024, 3, 1), end_date=date(2024, 3, 31))
    assert result.value == "output:data_source:2024-03-01 to 2024-03-31"
    assert processor.data is loader


def test_registry_name_references_share_instances():
    registry = ModelRegistry.root()
    registry.load_config_from_path(CONFIG_PATH)

    transformer = registry["flow_transformer"]
    source = registry["flow_source"]
    assert transformer.source is source

    stage2 = registry["flow_stage2"]
    stage3 = registry["flow_stage3"]
    assert stage2.stage1_output is registry["flow_stage1"]
    assert stage3.stage2_output is stage2


def test_instantiate_with_omegaconf():
    cfg = OmegaConf.create(
        {
            "loader": {
                "_target_": "ccflow.tests.test_flow_model.basic_loader",
                "source": "dynamic_source",
                "multiplier": 7,
            },
            "contextual": {
                "_target_": "ccflow.tests.test_flow_model.contextual_loader",
                "source": "warehouse",
            },
        }
    )

    registry = ModelRegistry.root()
    registry.load_config(cfg)

    assert registry["loader"](SimpleContext(value=3)).value == 21
    assert registry["contextual"].flow.compute(start_date=date(2024, 1, 1), end_date=date(2024, 1, 2)).value == {
        "source": "warehouse",
        "start_date": "2024-01-01",
        "end_date": "2024-01-02",
    }


def test_flow_context_execution_with_yaml_models():
    registry = ModelRegistry.root()
    registry.load_config_from_path(CONFIG_PATH)

    processor = registry["contextual_processor_model"]
    result = processor.flow.compute(FlowContext(start_date=date(2024, 4, 1), end_date=date(2024, 4, 30)))
    assert result.value == "output:data_source:2024-04-01 to 2024-04-30"
