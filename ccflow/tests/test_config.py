import os
import subprocess
import sys

import pytest


def _config_framework(
    env_value: str | None,
    *,
    hide_hydra: bool = False,
    hide_lerna: bool = False,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    if env_value is None:
        env.pop("CCFLOW_CONFIG_FRAMEWORK", None)
    else:
        env["CCFLOW_CONFIG_FRAMEWORK"] = env_value
    hidden_modules = []
    if hide_hydra:
        hidden_modules.append("hydra")
    if hide_lerna:
        hidden_modules.append("lerna")
    hide_modules_statement = "".join(f'sys.modules["{name}"] = None;' for name in hidden_modules)
    return subprocess.run(
        [
            sys.executable,
            "-c",
            f"import sys; {hide_modules_statement} from ccflow import config; print(config.CONFIG_FRAMEWORK)",
        ],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )


def _compose_config(config_dir, framework: str, overrides: list[str] | None = None) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["CCFLOW_CONFIG_FRAMEWORK"] = framework
    return subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from ccflow import config; "
                f"context = config.initialize_config_dir(config_dir={str(config_dir)!r}, version_base=None); "
                f"context.__enter__(); cfg = config.compose(config_name='config', overrides={overrides or []!r}); "
                "print(cfg); context.__exit__(None, None, None)"
            ),
        ],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )


def test_lerna_is_preferred_when_available():
    pytest.importorskip("lerna")

    result = _config_framework(None)

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "lerna"


def test_hydra_can_be_selected_explicitly():
    pytest.importorskip("hydra")

    result = _config_framework("hydra")

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "hydra"


def test_lerna_can_be_selected_explicitly():
    pytest.importorskip("lerna")

    result = _config_framework("lerna")

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "lerna"


def test_hydra_is_used_when_lerna_is_unavailable():
    pytest.importorskip("hydra")

    result = _config_framework(None, hide_lerna=True)

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "hydra"


def test_explicit_unavailable_lerna_fails_clearly():
    result = _config_framework("lerna", hide_lerna=True)

    assert result.returncode != 0
    assert "CCFLOW_CONFIG_FRAMEWORK=lerna requires the 'lerna' package" in result.stderr


def test_explicit_unavailable_hydra_fails_clearly():
    result = _config_framework("hydra", hide_hydra=True)

    assert result.returncode != 0
    assert "CCFLOW_CONFIG_FRAMEWORK=hydra requires the 'hydra-core' package" in result.stderr


def test_unknown_framework_is_rejected():
    result = _config_framework("hydraa")

    assert result.returncode != 0
    assert "CCFLOW_CONFIG_FRAMEWORK must be 'hydra', 'lerna', or unset" in result.stderr


def test_public_api_used_by_downstream_applications():
    from ccflow import config

    assert config.CONFIG_FRAMEWORK in ("hydra", "lerna")
    assert callable(config.main)
    assert callable(config.compose)
    assert callable(config.initialize)
    assert callable(config.initialize_config_dir)
    assert callable(config.instantiate)
    assert hasattr(config.HydraConfig, "get")
    assert issubclass(config.InstantiationException, Exception)


def test_lerna_patch_is_available_through_facade(tmp_path):
    pytest.importorskip("hydra")
    group = tmp_path / "group"
    group.mkdir()
    (group / "base.yaml").write_text("drop_me: true\nkeep: 42\n")
    (tmp_path / "config.yaml").write_text(
        """defaults:
  - group/base@_here_
  - _self_
  - _patch_:
    - ~drop_me
"""
    )

    lerna_result = _compose_config(tmp_path, "lerna")
    hydra_result = _compose_config(tmp_path, "hydra")

    assert lerna_result.returncode == 0, lerna_result.stderr
    assert "drop_me" not in lerna_result.stdout
    assert "'keep': 42" in lerna_result.stdout
    assert hydra_result.returncode != 0
    assert "Could not load '_patch_/~drop_me'" in hydra_result.stderr


def test_lerna_list_operations_are_available_through_facade(tmp_path):
    pytest.importorskip("hydra")
    (tmp_path / "config.yaml").write_text("tags: [one, two]\n")

    lerna_result = _compose_config(tmp_path, "lerna", ["tags=append(three)"])
    hydra_result = _compose_config(tmp_path, "hydra", ["tags=append(three)"])

    assert lerna_result.returncode == 0, lerna_result.stderr
    assert "one" in lerna_result.stdout
    assert "two" in lerna_result.stdout
    assert "three" in lerna_result.stdout
    assert hydra_result.returncode != 0
    assert "Unknown function 'append'" in hydra_result.stderr
