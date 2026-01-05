"""Hydra integration tests for Flow.model.

These tests verify that Flow.model decorated functions work correctly when
loaded from YAML configuration files using ModelRegistry.load_config_from_path().

Key feature: Registry name references (e.g., `source: flow_source`) ensure the same
object instance is shared across all consumers.
"""

from datetime import date
from pathlib import Path
from unittest import TestCase

from omegaconf import OmegaConf

from ccflow import CallableModel, DateRangeContext, GenericResult, ModelRegistry

from .test_flow_model import SimpleContext

CONFIG_PATH = str(Path(__file__).parent / "config" / "conf_flow.yaml")


class TestFlowModelHydraYAML(TestCase):
    """Tests loading Flow.model from YAML config files using ModelRegistry."""

    def setUp(self) -> None:
        ModelRegistry.root().clear()

    def tearDown(self) -> None:
        ModelRegistry.root().clear()

    def test_basic_loader_from_yaml(self):
        """Test basic model instantiation from YAML."""
        r = ModelRegistry.root()
        r.load_config_from_path(CONFIG_PATH)

        loader = r["flow_loader"]

        self.assertIsInstance(loader, CallableModel)

        ctx = SimpleContext(value=10)
        result = loader(ctx)
        self.assertEqual(result.value, 50)  # 10 * 5

    def test_string_processor_from_yaml(self):
        """Test string processor model from YAML."""
        r = ModelRegistry.root()
        r.load_config_from_path(CONFIG_PATH)

        processor = r["flow_processor"]

        ctx = SimpleContext(value=42)
        result = processor(ctx)
        self.assertEqual(result.value, "value=42!")

    def test_two_stage_pipeline_from_yaml(self):
        """Test two-stage pipeline from YAML config."""
        r = ModelRegistry.root()
        r.load_config_from_path(CONFIG_PATH)

        transformer = r["flow_transformer"]

        self.assertIsInstance(transformer, CallableModel)

        ctx = SimpleContext(value=5)
        result = transformer(ctx)
        # flow_source: 5 + 100 = 105
        # flow_transformer: 105 * 3 = 315
        self.assertEqual(result.value, 315)

    def test_three_stage_pipeline_from_yaml(self):
        """Test three-stage pipeline from YAML config."""
        r = ModelRegistry.root()
        r.load_config_from_path(CONFIG_PATH)

        stage3 = r["flow_stage3"]

        ctx = SimpleContext(value=10)
        result = stage3(ctx)
        # stage1: 10 + 10 = 20
        # stage2: 20 * 2 = 40
        # stage3: 40 + 50 = 90
        self.assertEqual(result.value, 90)

    def test_diamond_dependency_from_yaml(self):
        """Test diamond dependency pattern from YAML config."""
        r = ModelRegistry.root()
        r.load_config_from_path(CONFIG_PATH)

        aggregator = r["diamond_aggregator"]

        ctx = SimpleContext(value=10)
        result = aggregator(ctx)
        # source: 10 + 10 = 20
        # branch_a: 20 * 2 = 40
        # branch_b: 20 * 5 = 100
        # aggregator: 40 + 100 = 140
        self.assertEqual(result.value, 140)

    def test_date_range_pipeline_from_yaml(self):
        """Test DateRangeContext pipeline with transforms from YAML."""
        r = ModelRegistry.root()
        r.load_config_from_path(CONFIG_PATH)

        processor = r["flow_date_processor"]

        ctx = DateRangeContext(start_date=date(2024, 1, 10), end_date=date(2024, 1, 31))
        result = processor(ctx)

        # The transform extends start_date back by one day
        self.assertIn("2024-01-09", result.value)
        self.assertIn("normalized:", result.value)

    def test_context_args_from_yaml(self):
        """Test context_args model from YAML config."""
        r = ModelRegistry.root()
        r.load_config_from_path(CONFIG_PATH)

        loader = r["ctx_args_loader"]

        self.assertIsInstance(loader, CallableModel)
        # context_args models use DateRangeContext
        self.assertEqual(loader.context_type, DateRangeContext)

        ctx = DateRangeContext(start_date=date(2024, 1, 1), end_date=date(2024, 1, 31))
        result = loader(ctx)
        self.assertEqual(result.value, "data_source:2024-01-01 to 2024-01-31")

    def test_context_args_pipeline_from_yaml(self):
        """Test context_args pipeline with dependencies from YAML."""
        r = ModelRegistry.root()
        r.load_config_from_path(CONFIG_PATH)

        processor = r["ctx_args_processor"]

        ctx = DateRangeContext(start_date=date(2024, 3, 1), end_date=date(2024, 3, 31))
        result = processor(ctx)
        # loader: "data_source:2024-03-01 to 2024-03-31"
        # processor: "output:data_source:2024-03-01 to 2024-03-31"
        self.assertEqual(result.value, "output:data_source:2024-03-01 to 2024-03-31")

    def test_context_args_shares_instance(self):
        """Test that context_args pipeline shares dependency instance."""
        r = ModelRegistry.root()
        r.load_config_from_path(CONFIG_PATH)

        loader = r["ctx_args_loader"]
        processor = r["ctx_args_processor"]

        self.assertIs(processor.data, loader)


class TestFlowModelHydraInstanceSharing(TestCase):
    """Tests that registry name references share the same object instance."""

    def setUp(self) -> None:
        ModelRegistry.root().clear()

    def tearDown(self) -> None:
        ModelRegistry.root().clear()

    def test_pipeline_shares_instance(self):
        """Test that pipeline stages share the same dependency instance."""
        r = ModelRegistry.root()
        r.load_config_from_path(CONFIG_PATH)

        transformer = r["flow_transformer"]
        source = r["flow_source"]

        self.assertIs(transformer.source, source)

    def test_three_stage_pipeline_shares_instances(self):
        """Test that three-stage pipeline shares instances correctly."""
        r = ModelRegistry.root()
        r.load_config_from_path(CONFIG_PATH)

        stage1 = r["flow_stage1"]
        stage2 = r["flow_stage2"]
        stage3 = r["flow_stage3"]

        self.assertIs(stage2.stage1_output, stage1)
        self.assertIs(stage3.stage2_output, stage2)

    def test_diamond_pattern_shares_source_instance(self):
        """Test that diamond pattern branches share the same source instance."""
        r = ModelRegistry.root()
        r.load_config_from_path(CONFIG_PATH)

        source = r["diamond_source"]
        branch_a = r["diamond_branch_a"]
        branch_b = r["diamond_branch_b"]
        aggregator = r["diamond_aggregator"]

        # Both branches should share the SAME source instance
        self.assertIs(branch_a.source, source)
        self.assertIs(branch_b.source, source)
        self.assertIs(branch_a.source, branch_b.source)

        self.assertIs(aggregator.input_a, branch_a)
        self.assertIs(aggregator.input_b, branch_b)

    def test_date_range_shares_instance(self):
        """Test that date range pipeline shares dependency instance."""
        r = ModelRegistry.root()
        r.load_config_from_path(CONFIG_PATH)

        loader = r["flow_date_loader"]
        processor = r["flow_date_processor"]

        self.assertIs(processor.raw_data, loader)


class TestFlowModelHydraOmegaConf(TestCase):
    """Tests using OmegaConf.create for dynamic config creation."""

    def setUp(self) -> None:
        ModelRegistry.root().clear()

    def tearDown(self) -> None:
        ModelRegistry.root().clear()

    def test_instantiate_with_omegaconf(self):
        """Test instantiation using OmegaConf.create via ModelRegistry."""
        cfg = OmegaConf.create(
            {
                "loader": {
                    "_target_": "ccflow.tests.test_flow_model.basic_loader",
                    "source": "dynamic_source",
                    "multiplier": 7,
                },
            }
        )

        r = ModelRegistry.root()
        r.load_config(cfg)
        loader = r["loader"]

        ctx = SimpleContext(value=3)
        result = loader(ctx)
        self.assertEqual(result.value, 21)  # 3 * 7

    def test_nested_deps_with_omegaconf(self):
        """Test nested dependencies using OmegaConf with registry names."""
        cfg = OmegaConf.create(
            {
                "source": {
                    "_target_": "ccflow.tests.test_flow_model.data_source",
                    "base_value": 50,
                },
                "transformer": {
                    "_target_": "ccflow.tests.test_flow_model.data_transformer",
                    "source": "source",
                    "factor": 4,
                },
            }
        )

        r = ModelRegistry.root()
        r.load_config(cfg)
        transformer = r["transformer"]

        ctx = SimpleContext(value=10)
        result = transformer(ctx)
        # source: 10 + 50 = 60
        # transformer: 60 * 4 = 240
        self.assertEqual(result.value, 240)

        self.assertIs(transformer.source, r["source"])

    def test_diamond_with_omegaconf(self):
        """Test diamond pattern with OmegaConf using registry names."""
        cfg = OmegaConf.create(
            {
                "source": {
                    "_target_": "ccflow.tests.test_flow_model.data_source",
                    "base_value": 10,
                },
                "branch_a": {
                    "_target_": "ccflow.tests.test_flow_model.data_transformer",
                    "source": "source",
                    "factor": 2,
                },
                "branch_b": {
                    "_target_": "ccflow.tests.test_flow_model.data_transformer",
                    "source": "source",
                    "factor": 3,
                },
                "aggregator": {
                    "_target_": "ccflow.tests.test_flow_model.data_aggregator",
                    "input_a": "branch_a",
                    "input_b": "branch_b",
                    "operation": "multiply",
                },
            }
        )

        r = ModelRegistry.root()
        r.load_config(cfg)
        aggregator = r["aggregator"]

        ctx = SimpleContext(value=5)
        result = aggregator(ctx)
        # source: 5 + 10 = 15
        # branch_a: 15 * 2 = 30
        # branch_b: 15 * 3 = 45
        # aggregator: 30 * 45 = 1350
        self.assertEqual(result.value, 1350)

        # Verify SAME source instance is shared
        self.assertIs(r["branch_a"].source, r["source"])
        self.assertIs(r["branch_b"].source, r["source"])


class TestFlowModelHydraDefaults(TestCase):
    """Tests that default parameter values work with Hydra."""

    def setUp(self) -> None:
        ModelRegistry.root().clear()

    def tearDown(self) -> None:
        ModelRegistry.root().clear()

    def test_defaults_used_when_not_specified(self):
        """Test that default values are used when not in config."""
        cfg = OmegaConf.create(
            {
                "loader": {
                    "_target_": "ccflow.tests.test_flow_model.basic_loader",
                    "source": "test",
                },
            }
        )

        r = ModelRegistry.root()
        r.load_config(cfg)
        loader = r["loader"]

        ctx = SimpleContext(value=10)
        result = loader(ctx)
        self.assertEqual(result.value, 10)  # 10 * 1 (default)

    def test_defaults_can_be_overridden(self):
        """Test that defaults can be overridden in config."""
        cfg = OmegaConf.create(
            {
                "loader": {
                    "_target_": "ccflow.tests.test_flow_model.basic_loader",
                    "source": "test",
                    "multiplier": 100,
                },
            }
        )

        r = ModelRegistry.root()
        r.load_config(cfg)
        loader = r["loader"]

        ctx = SimpleContext(value=10)
        result = loader(ctx)
        self.assertEqual(result.value, 1000)  # 10 * 100


class TestFlowModelHydraModelProperties(TestCase):
    """Tests that model properties are correct after Hydra instantiation."""

    def setUp(self) -> None:
        ModelRegistry.root().clear()

    def tearDown(self) -> None:
        ModelRegistry.root().clear()

    def test_context_type_property(self):
        """Test that context_type is correct."""
        r = ModelRegistry.root()
        r.load_config_from_path(CONFIG_PATH)

        loader = r["flow_loader"]
        self.assertEqual(loader.context_type, SimpleContext)

    def test_result_type_property(self):
        """Test that result_type is correct."""
        r = ModelRegistry.root()
        r.load_config_from_path(CONFIG_PATH)

        loader = r["flow_loader"]
        self.assertEqual(loader.result_type, GenericResult[int])

    def test_deps_method_works(self):
        """Test that __deps__ method works after Hydra instantiation."""
        r = ModelRegistry.root()
        r.load_config_from_path(CONFIG_PATH)

        transformer = r["flow_transformer"]

        ctx = SimpleContext(value=5)
        deps = transformer.__deps__(ctx)

        self.assertEqual(len(deps), 1)
        self.assertIsInstance(deps[0][0], CallableModel)
        self.assertEqual(deps[0][1], [ctx])
        self.assertIs(deps[0][0], r["flow_source"])


class TestFlowModelHydraDateRangeTransforms(TestCase):
    """Tests transforms with DateRangeContext from Hydra config."""

    def setUp(self) -> None:
        ModelRegistry.root().clear()

    def tearDown(self) -> None:
        ModelRegistry.root().clear()

    def test_transform_applied_from_yaml(self):
        """Test that transform is applied when loaded from YAML."""
        r = ModelRegistry.root()
        r.load_config_from_path(CONFIG_PATH)

        processor = r["flow_date_processor"]

        ctx = DateRangeContext(start_date=date(2024, 1, 10), end_date=date(2024, 1, 31))
        deps = processor.__deps__(ctx)

        self.assertEqual(len(deps), 1)
        dep_model, dep_contexts = deps[0]

        # The transform should extend start_date back by one day
        transformed_ctx = dep_contexts[0]
        self.assertEqual(transformed_ctx.start_date, date(2024, 1, 9))
        self.assertEqual(transformed_ctx.end_date, date(2024, 1, 31))

        self.assertIs(dep_model, r["flow_date_loader"])


if __name__ == "__main__":
    import unittest

    unittest.main()
