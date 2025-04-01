from pathlib import Path

from ccflow.base import load_config


def test_root_config():
    root_config_dir = str(Path(__file__).resolve().parent / "config")
    r = load_config(
        root_config_dir=root_config_dir,
        root_config_name="conf",
        overwrite=True,
    )
    try:
        assert len(r.models)
        assert "foo" in r.models
        assert "bar" in r.models
    finally:
        r.clear()


def test_config_dir():
    root_config_dir = str(Path(__file__).resolve().parent / "config")
    config_dir = str(Path(__file__).resolve().parent / "config_user")
    r = load_config(
        root_config_dir=root_config_dir,
        root_config_name="conf",
        config_dir=config_dir,
        overwrite=True,
    )
    try:
        assert len(r.models)
        assert "foo" in r.models
        assert "bar" in r.models
    finally:
        r.clear()


def test_config_name():
    root_config_dir = str(Path(__file__).resolve().parent / "config")
    config_dir = str(Path(__file__).resolve().parent / "config_user")
    r = load_config(
        root_config_dir=root_config_dir,
        root_config_name="conf",
        config_dir=config_dir,
        config_name="sample",
        overwrite=True,
    )
    try:
        assert len(r.models)
        assert "foo" in r.models
        assert "bar" in r.models
        assert "config_user" in r.models
        assert "user_foo" in r["config_user"]
    finally:
        r.clear()


def test_config_dir_with_overrides():
    root_config_dir = str(Path(__file__).resolve().parent / "config")
    config_dir = str(Path(__file__).resolve().parent)
    r = load_config(
        root_config_dir=root_config_dir,
        root_config_name="conf",
        config_dir=config_dir,
        overrides=["+config_user=sample"],
        overwrite=True,
    )
    try:
        assert len(r.models)
        assert "foo" in r.models
        assert "bar" in r.models
        assert "config_user" in r.models
        assert "user_foo" in r["config_user"]
    finally:
        r.clear()


def test_config_name_yml_not_yaml():
    # TODO: Doesn't look like hydra supports yml
    root_config_dir = str(Path(__file__).resolve().parent / "config")
    config_dir = str(Path(__file__).resolve().parent / "config_user")
    r = load_config(
        root_config_dir=root_config_dir,
        root_config_name="conf",
        config_dir=config_dir,
        config_name="sample2",
        overwrite=True,
    )
    try:
        assert len(r.models)
        assert "foo" in r.models
        assert "bar" in r.models
        assert "config_user" in r.models
        assert "user_bar" in r["config_user"]
    finally:
        r.clear()


def test_config_dir_basepath():
    root_config_dir = str(Path(__file__).resolve().parent / "config")
    config_dir = "."
    basepath = str(Path(__file__).resolve().parent)
    r = load_config(
        root_config_dir=root_config_dir,
        root_config_name="conf",
        config_dir=config_dir,
        overrides=["+config_user=sample"],
        overwrite=True,
        basepath=basepath,
    )
    try:
        assert len(r.models)
        assert "foo" in r.models
        assert "bar" in r.models
        assert "config_user" in r.models
        assert "user_foo" in r["config_user"]
    finally:
        r.clear()


def test_config_dir_basepath_malformed():
    root_config_dir = str(Path(__file__).resolve().parent / "config")
    # By putting "config_user" in both the base path and the config dir, we are technically listing it twice,
    # so it needs to go up a level to actually find the "config_user" directory.
    basepath = str(Path(__file__).resolve().parent / "config_user")
    r = load_config(
        root_config_dir=root_config_dir,
        root_config_name="conf",
        config_dir="config_user",
        config_name="sample",
        overwrite=True,
        basepath=basepath,
    )
    try:
        assert len(r.models)
        assert "foo" in r.models
        assert "bar" in r.models
        assert "config_user" in r.models
        assert "user_foo" in r["config_user"]
    finally:
        r.clear()
