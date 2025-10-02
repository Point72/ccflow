from pathlib import Path

from ccflow.base import load_config as load_registry


def test_hydra_mixin_loading():
    root_config_dir = str(Path(__file__).resolve().parent.parent / "config")
    # Load the dedicated mixin config file
    r = load_registry(
        root_config_dir=root_config_dir, root_config_name="conf_mixin", overwrite=True, basepath=str(Path(__file__).resolve().parent.parent)
    )
    try:
        assert "mixin_simple" in r.models
        assert "mixin_db" in r.models
        assert "mixin_parent" in r.models

        ms = r["mixin_simple"]
        assert ms.name == "test_model"
        assert ms.version == "1.0"
        assert ms.enabled is True

        md = r["mixin_db"]
        assert md.host == "localhost"
        assert md.port == 5432
        assert md.name == "test_db"

        mp = r["mixin_parent"]
        # Parent resolved from SIMPLE_CONFIG
        assert mp.name == "test_model"
        assert mp.version == "1.0"
        assert mp.enabled is True
        # Child resolved via path/key
        assert mp.child.host == "localhost"
        assert mp.child.port == 5432
        assert mp.child.name == "test_db"
        # Plain child uses provided explicit values
        assert mp.plain_child.host == "plain.example.com"
        assert mp.plain_child.port == 3306
        assert mp.plain_child.name == "plain_db"
    finally:
        r.clear()


def test_hydra_mixin_override_path_only():
    root_config_dir = str(Path(__file__).resolve().parent.parent / "config")
    overrides = [
        "mixin_db.ccflow_path=ccflow.tests.data.path_key_resolver_samples.OTHER_NESTED_CONFIG",
    ]
    r = load_registry(
        root_config_dir=root_config_dir,
        root_config_name="conf_mixin",
        overrides=overrides,
        overwrite=True,
        basepath=str(Path(__file__).resolve().parent.parent),
    )
    try:
        md = r["mixin_db"]
        # Should resolve from OTHER_NESTED_CONFIG["database"]
        assert md.host == "override.local"
        assert md.port == 6543
        assert md.name == "other_db"
    finally:
        r.clear()


def test_hydra_mixin_override_path_and_key():
    root_config_dir = str(Path(__file__).resolve().parent.parent / "config")
    overrides = [
        "mixin_db.ccflow_path=ccflow.tests.data.path_key_resolver_samples.OTHER_NESTED_CONFIG",
        "mixin_db.ccflow_keys=database_alt",
    ]
    r = load_registry(
        root_config_dir=root_config_dir,
        root_config_name="conf_mixin",
        overrides=overrides,
        overwrite=True,
        basepath=str(Path(__file__).resolve().parent.parent),
    )
    try:
        md = r["mixin_db"]
        # Should resolve from OTHER_NESTED_CONFIG["database_alt"]
        assert md.host == "alt.local"
        assert md.port == 7777
        assert md.name == "alt_db"
    finally:
        r.clear()
