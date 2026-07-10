"""Smoke tests for packaged Flow.model examples."""

import subprocess
import sys

import pytest


@pytest.mark.parametrize(
    ("module_name", "expected_text"),
    [
        ("ccflow.examples.flow_model.flow_model_example", "Flow.model Example"),
        ("ccflow.examples.flow_model.flow_model_hydra_builder_demo", "Hydra + Flow.model Builder Demo"),
    ],
)
def test_flow_model_examples_run_as_package_modules(module_name: str, expected_text: str):
    result = subprocess.run(
        [sys.executable, "-m", module_name],
        check=True,
        capture_output=True,
        text=True,
    )

    assert expected_text in result.stdout
    assert "counts=[model, model]" in result.stdout
