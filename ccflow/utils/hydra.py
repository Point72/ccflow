import argparse
import inspect
import os
from dataclasses import dataclass
from pathlib import Path
from pprint import pprint
from typing import Any, Callable, Dict, List, Optional

from hydra._internal.defaults_list import DefaultsList
from omegaconf import DictConfig, OmegaConf

__all__ = ("load_config",)


@dataclass
class ConfigLoadResult:
    root_config_dir: str
    root_config_name: str
    cfg: DictConfig
    cfg_sources: Optional[DictConfig] = None
    defaults_list: Optional[DefaultsList] = None
    group_options: Optional[Dict[str, List[str]]] = None

    def merge(self) -> Dict:
        """Returns a single nested dict of information by config key"""
        cfg_resolved = OmegaConf.to_container(self.cfg, resolve=True)
        cfg_dict = OmegaConf.to_container(self.cfg_sources, resolve=False)

        # Process cfg_resolved to embed values in the `value` key
        cfg_resolved = _transform_leaves(cfg_resolved, lambda x: dict(value_interp=x))
        # process the group options
        group_options = {group: {"__options__": options} for group, options in self.group_options.items()}
        group_options = _to_nested_dict(group_options, "/")
        # process the defaults list
        defaults_dict = {
            default.package: {
                "__selected__": default.config_path.split("/")[-1],
                "__parent__": default.parent,
                # WARNING: Can't do the below because it's not the right path for groups loaded from searchpaths!
                # "__selected_path__": f"{self.root_config_dir}/{default.config_path}.yaml",
                # "__parent_path__": f"{self.root_config_dir}/{default.parent}.yaml",
            }
            for default in self.defaults_list.defaults
        }
        defaults_dict = _to_nested_dict(defaults_dict, ".")

        # Merge the different dicts
        cfg = OmegaConf.merge(cfg_dict, cfg_resolved, group_options, defaults_dict)
        return _sorted_nested_dict(OmegaConf.to_container(cfg, resolve=False))


def _to_nested_dict(flat_dict, separator):
    dicts = []
    for k, v in flat_dict.items():
        for part in reversed(k.split(separator)):
            v = {part: v}
        dicts.append(v)

    # This is probably slower than it needs to be
    return OmegaConf.merge(*dicts)


def _transform_leaves(d, f):
    if isinstance(d, dict):
        return {k: _transform_leaves(v, f) for k, v in d.items()}
    else:
        return f(d)


def _sorted_nested_dict(d):
    if not isinstance(d, dict):
        return d

    return {key: _sorted_nested_dict(d[key]) for key in sorted(d.keys())}


def _dict_add_source(d, source):
    if d is not None:
        for k, v in d.items():
            if k in ("defaults", "hydra"):  # Can't do this hacking on defaults or hydra (i.e. if searchpaths is set).
                continue
            if isinstance(v, dict):
                d[k] = _dict_add_source(v, source)
            else:
                d[k] = dict(value_raw=v, source_file=source)
    return d


def _find_group_options(config_loader, path, config_name, overrides, results):
    """This function finds the names of all the options in the config group hierarchy.

    Note that it will pick up config files that are not intended to be used as config group options,
    but that exist to provide common config options to other files in the group (i.e. to default)
    """
    from hydra.core.object_type import ObjectType

    groups = config_loader.get_group_options(path, ObjectType.GROUP, config_name, overrides)
    options = config_loader.get_group_options(path, ObjectType.CONFIG, config_name, overrides)
    if options:
        results[path] = [o for o in options if not o.startswith("_")]  # Hide options starting with "_"
    for group in groups:  # Iterate through all the sub-groups
        group_path = f"{path}/{group}".strip("/")
        _find_group_options(config_loader, group_path, config_name, overrides, results)
    return results


def _find_parent_config_folder(config_dir: str = "config", config_name: str = "", *, basepath: str = ""):
    folder = Path(basepath).resolve()
    exists = (folder / config_dir).exists() if not config_name else (folder / config_dir / f"{config_name}.yaml").exists()
    if not exists and (folder / config_dir / f"{config_name}.yml").exists():
        raise ValueError(
            f"Found config_name `{config_name}` with `.yml` suffix, which is not recognized by hydra. Please rename to `{config_name}.yaml`."
        )
    while not exists:
        folder = folder.parent
        if str(folder) == os.path.abspath(os.sep):
            raise FileNotFoundError(f"Could not find config folder: {config_dir} in folder {basepath}")
        exists = (folder / config_dir).exists() if not config_name else (folder / config_dir / f"{config_name}.yaml").exists()
        if not exists and (folder / config_dir / f"{config_name}.yml").exists():
            raise ValueError(
                f"Found config_name `{config_name}` with `.yml` suffix, which is not recognized by hydra. Please rename to `{config_name}.yaml`."
            )

    config_dir = (folder / config_dir).resolve()
    if not config_name:
        return folder.resolve(), config_dir, ""
    else:
        return folder.resolve(), config_dir, (folder / config_dir / f"{config_name}.yaml").resolve()


def load_config(
    root_config_dir: str,
    root_config_name: str,
    config_dir: str = "config",
    config_name: str = "",
    overrides: Optional[List[str]] = None,
    *,
    version_base: Optional[str] = None,
    basepath: str = "",
    debug: bool = False,
) -> ConfigLoadResult:
    """Helper function to load a hydra config into the root model registry.

    Hydra configs can be pulled from multiple places:
      1. A root configuration
      2. An optional user-provided config directory, and within that, an optional config name.

    Optionally, we want to pull a bunch of other information for debugging

    Arguments:
        root_config_dir: The directory containing the root hydra config. This is typically the location of the configs in, i.e. "config"
         This is passed to hydra.initialize_config_dir to get the loading started.
        root_config_name: The config name within the base directory, i.e. "conf"
        config_dir: End user-provided additional directory to search for hydra configs.
        config_name: An optional config name to look for within the `config_dir`. This allows you to specify a particular config file to load.
        overrides: A list of hydra-style override strings to apply when loading the config.
        version_base: See https://hydra.cc/docs/upgrades/version_base/
        basepath: The base path to start searching for the `config_dir`. This is useful when you want to load from an absolute (rather than relative) path.
        debug: (Experimental) Whether to enable debug mode. This will return more information about the configs on ConfigLoadResult.
    """
    # Heavy import, only import if used
    import os

    from hydra import compose, initialize_config_dir

    overrides = overrides or []
    with initialize_config_dir(config_dir=root_config_dir, version_base=version_base):
        if config_dir:
            hydra_folder, config_dir, _ = _find_parent_config_folder(config_dir=config_dir, config_name=config_name, basepath=basepath or os.getcwd())

            cfg = compose(config_name=root_config_name, overrides=[], return_hydra_config=True)
            searchpaths = cfg["hydra"]["searchpath"]
            searchpaths.extend([hydra_folder, config_dir])
            if config_name:
                config_group = Path(config_dir).resolve().name
                overrides = [f"+{config_group}={config_name}", *overrides.copy(), f"hydra.searchpath=[{','.join(searchpaths)}]"]
            else:
                overrides = [*overrides.copy(), f"hydra.searchpath=[{','.join(searchpaths)}]"]

        cfg = compose(config_name=root_config_name, overrides=overrides)
        result = ConfigLoadResult(root_config_dir=root_config_dir, root_config_name=root_config_name, cfg=cfg)
        if debug:
            import yaml
            from hydra.core.global_hydra import GlobalHydra
            from hydra.types import RunMode

            # To track the source file for each config value, we need to monkey patch the yaml loader
            original_yaml_load = yaml.load
            try:

                def yaml_load(*args, **kwargs):
                    res = original_yaml_load(*args, **kwargs)
                    return _dict_add_source(res, args[0].name)

                yaml.load = yaml_load
                result.cfg_sources = compose(config_name=root_config_name, overrides=overrides)
            finally:
                yaml.load = original_yaml_load

            config_loader = GlobalHydra.instance().config_loader()
            # Load defaults list using the standard hydra function
            result.defaults_list = config_loader.compute_defaults_list(root_config_name, overrides=overrides, run_mode=RunMode.RUN)
            # Load all config group options by recursively calling config_loader.get_group_options
            # Ideally, one would use the two lines below, but these functions do not process the overrides
            #   groups = [x for x in GlobalHydra.instance().hydra.list_all_config_groups()]
            #   result.group_options = {group: sorted(config_loader.get_group_options(group)) for group in groups}
            # Thus, we have our own implementation
            result.group_options = {}
            _find_group_options(config_loader, "", root_config_name, overrides, result.group_options)

    return result


def get_args_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=True, description="Hydra Config Audit Tool")
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Any key=value arguments to override config values (use dots for.nested=overrides)",
    )

    parser.add_argument(
        "--config-path",
        "-cp",
        help="""Overrides the config_path specified in hydra.main().
                    The config_path is absolute or relative to the Python file declaring @hydra.main()""",
    )

    parser.add_argument(
        "--config-name",
        "-cn",
        help="Overrides the config_name specified in hydra.main()",
    )

    parser.add_argument(
        "--config-dir",
        "-cd",
        help="Adds an additional config dir to the config search path",
    )
    parser.add_argument(
        "--config-dir-config-name",
        "-cdcn",
        help="An optional config name to look for within the `config_dir`. This allows you to specify a particular config file to load.",
    )
    parser.add_argument(
        "--basepath",
        help="The base path to start searching for the `config_dir` (if not the cwd). This is useful when you want to load from an absolute (rather than relative) path.",
    )
    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="Disable the GUI",
    )
    return parser


def cfg_explain_cli(
    config_path: str = "",
    config_name: str = "",
    hydra_main: Optional[Callable] = None,
    args_parser: argparse.ArgumentParser = None,
    ui_launcher: Callable[[Dict[str, Any], ...], None] = None,
):
    """CLI entry point for hydra config explain

    Parameters:
        config_path: The config_path specified in hydra.main()
        config_name: The config_name specified in hydra.main()
        hydra_main: The module in which hydra.main() is declared. This is used to find the config_path and config_name.
        args_parser: An optionally extended version of the argparser returned by `get_args_parser` to add UI arguments
        ui_launcher: A callable that takes the config dict and the parsed args and launches a custom display.
    """
    parser = args_parser or get_args_parser()
    args = parser.parse_args()

    if args.config_path:
        root_config_dir = args.config_path
    elif hydra_main and config_path:
        root_config_dir = str(Path(inspect.getfile(hydra_main.__wrapped__)).parent / config_path)
    else:
        raise ValueError("Must provide --config-path.")

    if args.config_name:
        root_config_name = args.config_name
    elif config_name:
        root_config_name = config_name
    else:
        raise ValueError("Must provide --config-name.")

    result = load_config(
        root_config_dir=root_config_dir,
        root_config_name=root_config_name,
        config_dir=args.config_dir,
        config_name=args.config_dir_config_name,
        overrides=args.overrides,
        basepath=args.basepath,
        debug=True,
    )
    merged_cfg = result.merge()

    if args.no_gui:
        pprint(merged_cfg, width=120, indent=2)
    elif ui_launcher is not None:
        ui_launcher(merged_cfg, **vars(args))
    else:
        raise ValueError("Cannot launch UI, no ui_launcher provided. Use --no-gui to print the results.")


if __name__ == "__main__":
    cfg_explain_cli()
