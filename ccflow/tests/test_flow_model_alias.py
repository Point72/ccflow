"""Parity tests: model_alias vs bare-string vs root-relative registry aliases.

In ccflow a bare string field value is resolved from the model registry, so an explicit
`ccflow.compose.model_alias` wrapper, a bare-string alias, and a root-relative `/name`
alias should all dereference to the same registered model instance. `model_alias` is a
Hydra convenience for the existing bare-string convention, not a separate mechanism.
"""

from pathlib import Path

from ccflow import CallableModel, ModelRegistry

ALIAS_CONFIG_PATH = str(Path(__file__).parent / "config" / "conf_flow_alias.yaml")


def setup_function():
    ModelRegistry.root().clear()


def teardown_function():
    ModelRegistry.root().clear()


def test_alias_forms_resolve_to_same_instance():
    registry = ModelRegistry.root()
    registry.load_config_from_path(ALIAS_CONFIG_PATH)

    source = registry["alias_source"]
    via_model_alias = registry["via_model_alias"]
    via_bare_string = registry["via_bare_string"]
    via_root_relative = registry["via_root_relative"]

    # Each `source` must be the registered model instance, not a literal string.
    assert isinstance(via_bare_string.source, CallableModel)
    assert via_model_alias.source is source
    assert via_bare_string.source is source
    assert via_root_relative.source is source


def test_alias_forms_compute_identically():
    registry = ModelRegistry.root()
    registry.load_config_from_path(ALIAS_CONFIG_PATH)

    results = [registry[name].flow.compute(value=5).value for name in ("via_model_alias", "via_bare_string", "via_root_relative")]
    # data_source(base_value=100)(value=5) -> 105; data_transformer(factor=3) -> 315
    assert results == [315, 315, 315]
