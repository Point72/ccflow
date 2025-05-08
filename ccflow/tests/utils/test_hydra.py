import sys
from pathlib import Path

import pytest

from ccflow.utils.hydra import load_config


@pytest.fixture
def basepath():
    # Because os.cwd may change depending on how tests are run
    return str(Path(__file__).resolve().parent.parent)


def test_root_config(basepath):
    root_config_dir = str(Path(__file__).resolve().parent.parent / "config")
    result = load_config(root_config_dir=root_config_dir, root_config_name="conf", basepath=basepath, debug=False)
    assert result.root_config_dir == root_config_dir
    assert result.root_config_name == "conf"
    assert result.cfg
    assert result.cfg_sources is None
    assert result.defaults_list is None
    assert result.group_options is None


def test_config_dir(basepath):
    root_config_dir = str(Path(__file__).resolve().parent.parent / "config")
    config_dir = str(Path(__file__).resolve().parent.parent / "config_user")
    result = load_config(
        root_config_dir=root_config_dir,
        root_config_name="conf",
        config_dir=config_dir,
        basepath=basepath,
    )
    assert result.cfg
    assert "foo" in result.cfg
    assert "bar" in result.cfg


def test_config_name(basepath):
    root_config_dir = str(Path(__file__).resolve().parent.parent / "config")
    config_dir = str(Path(__file__).resolve().parent.parent / "config_user")
    result = load_config(
        root_config_dir=root_config_dir,
        root_config_name="conf",
        config_dir=config_dir,
        config_name="sample",
        basepath=basepath,
    )
    assert result.cfg
    assert "foo" in result.cfg
    assert "bar" in result.cfg
    assert "config_user" in result.cfg
    assert "user_foo" in result.cfg["config_user"]


def test_config_dir_with_overrides(basepath):
    root_config_dir = str(Path(__file__).resolve().parent.parent / "config")
    config_dir = str(Path(__file__).resolve().parent.parent)
    result = load_config(
        root_config_dir=root_config_dir,
        root_config_name="conf",
        config_dir=config_dir,
        overrides=["+config_user=sample"],
        basepath=basepath,
    )
    assert result.cfg
    assert "foo" in result.cfg
    assert "bar" in result.cfg
    assert "config_user" in result.cfg
    assert "user_foo" in result.cfg["config_user"]


def test_config_name_yml_not_yaml(basepath):
    root_config_dir = str(Path(__file__).resolve().parent.parent / "config")
    config_dir = str(Path(__file__).resolve().parent.parent / "config_user")
    with pytest.raises(ValueError):
        load_config(
            root_config_dir=root_config_dir,
            root_config_name="conf",
            config_dir=config_dir,
            config_name="sample2",
            basepath=basepath,
        )


def test_config_dir_basepath_malformed():
    root_config_dir = str(Path(__file__).resolve().parent.parent / "config")
    # By putting "config_user" in both the base path and the config dir, we are technically listing it twice,
    # so it needs to go up a level to actually find the "config_user" directory.
    basepath = str(Path(__file__).resolve().parent.parent / "config_user")
    result = load_config(
        root_config_dir=root_config_dir,
        root_config_name="conf",
        config_dir="config_user",
        config_name="sample",
        basepath=basepath,
    )
    assert result.cfg
    assert "foo" in result.cfg
    assert "bar" in result.cfg
    assert "config_user" in result.cfg
    assert "user_foo" in result.cfg["config_user"]


def test_debug(basepath):
    root_config_dir = str(Path(__file__).resolve().parent.parent / "config")
    config_dir = str(Path(__file__).resolve().parent.parent)
    result = load_config(
        root_config_dir=root_config_dir,
        root_config_name="conf",
        config_dir=config_dir,
        overrides=["+config_user=sample"],
        basepath=basepath,
        debug=True,
    )
    assert result.cfg
    assert result.cfg_sources
    assert result.defaults_list
    assert result.group_options

    # cfg checks
    assert "foo" in result.cfg
    assert "bar" in result.cfg
    assert "config_user" in result.cfg
    assert "user_foo" in result.cfg["config_user"]

    # cfg_sources checks
    assert "foo" in result.cfg_sources
    assert result.cfg_sources["foo"]["a"]["value_raw"] == "test"
    assert result.cfg_sources["foo"]["a"]["source_file"] == f"{root_config_dir}/conf.yaml"
    assert result.cfg_sources["config_user"]["user_foo"]["a"]["value_raw"] == "test"
    assert result.cfg_sources["config_user"]["user_foo"]["a"]["source_file"] == f"{config_dir}/config_user/sample.yaml"

    # defaults_list checks
    assert result.defaults_list.defaults
    assert any(default.config_path == "config_user/sample" for default in result.defaults_list.defaults)
    assert len(result.defaults_list.overrides.append_group_defaults) == 1
    assert result.defaults_list.overrides.append_group_defaults[0].group == "config_user"
    assert result.defaults_list.overrides.append_group_defaults[0].value == "sample"

    # group options
    assert result.group_options
    assert "hydra/job_logging" in result.group_options
    assert len(result.group_options["hydra/job_logging"]) > 1
    assert "config_user" in result.group_options
    assert result.group_options["config_user"] == ["sample"]
    # Arguable whether these should be here
    assert "conf_out_of_order" in result.group_options[""]

    # merged value
    merged = result.merge()
    assert merged
    assert "foo" in merged
    assert "config_user" in merged
    assert merged["config_user"]["__options__"] == ["sample"]
    assert merged["config_user"]["__parent__"] == "conf"  # Maybe this should be a path to a file
    assert merged["config_user"]["__selected__"] == "sample"
    assert "user_foo" in merged["config_user"]
    assert "a" in merged["config_user"]["user_foo"]
    assert "source_file" in merged["config_user"]["user_foo"]["a"]
    assert "value_raw" in merged["config_user"]["user_foo"]["a"]
    assert "value_interp" in merged["config_user"]["user_foo"]["a"]


@pytest.fixture(params=["--no-gui", "--gui"])
def mock_args(mocker, basepath, request):
    root_config_dir = str(Path(__file__).resolve().parent.parent / "config")
    config_dir = str(Path(__file__).resolve().parent.parent)

    argv = ["dummy", "--config-path", root_config_dir, "--config-name", "conf", "--config-dir", config_dir, "--basepath", basepath]
    if request.param == "--no-gui":
        argv += ["--no-gui"]
    argv += ["+config_user=sample"]
    mocker.patch.object(sys, "argv", argv)


def test_cfg_explain_cli_args(mock_args):
    """Test that the mocking of the argparse args works correctly in isolation"""
    from ccflow.utils.hydra import _get_args_parser

    parser = _get_args_parser()
    args = parser.parse_args()
    assert args.overrides == ["+config_user=sample"]
    assert args.config_name == "conf"
    assert args.port == 8080


def test_cfg_explain_cli(mock_args, mocker, capsys):
    """Test that the CLI works correctly with the mocked args"""
    from ccflow.utils.hydra import cfg_explain_cli

    mocker.patch("panel.serve")

    cfg_explain_cli()
