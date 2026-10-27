## Pre-requisites

You need Python >=3.10 on your machine to install `ccflow`.

## Install with `pip`

```bash
pip install ccflow
```

## Install with `conda`

```bash
conda install ccflow --channel conda-forge
```

## Install Hydra for configuration

Loading registries from YAML configs, the `_target_`-based `ModelRegistry.load_config`, `LazyRegistry` materialization, and the config-driven CLIs use [Hydra](https://hydra.cc). `ccflow` does not install it by default:

```bash
pip install hydra-core
```

## Source installation

For other platforms and for development installations, [build `ccflow` from source](Build-from-Source).
