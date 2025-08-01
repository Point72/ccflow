<a href="https://github.com/point72/ccflow">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://github.com/point72/ccflow/raw/main/docs/img/dark.png?raw=true">
    <img alt="ccflow logo, 'ccflow' with letters in color" src="https://github.com/point72/ccflow/raw/main/docs/img/light.png?raw=true" width="400">
  </picture>
</a>

<br/>
<br/>

[![Build Status](https://github.com/Point72/ccflow/actions/workflows/build.yaml/badge.svg?branch=main&event=push)](https://github.com/Point72/ccflow/actions/workflows/build.yaml)
[![codecov](https://codecov.io/gh/Point72/ccflow/branch/main/graph/badge.svg)](https://codecov.io/gh/Point72/ccflow)
[![GitHub issues](https://img.shields.io/github/issues/point72/ccflow.svg)](https://github.com/point72/ccflow/issues)
[![License](https://img.shields.io/github/license/Point72/ccflow)](https://github.com/Point72/ccflow)
[![PyPI Version](https://img.shields.io/pypi/v/ccflow.svg)](https://pypi.python.org/pypi/ccflow)

`ccflow` (Composable Configuration Flow) is a collection of tools for workflow configuration, orchestration, and dependency injection.
It is intended to be flexible enough to handle diverse use cases, including data retrieval, validation, transformation, and loading (i.e. ETL workflows), model training, microservice configuration, and automated report generation.

The framework provides:

- a way to manage hierarchical, strongly typed configurations and the relationships between them through composition
- a way to associate user-defined functions with configurations, and in doing so, to define and name configurable workflow graphs
- a way to manage dependency injection and inversion of control for objects in these graphs
- flexibility in how to interact with configurations and workflows, including files/command line, native python/Jupyter notebook, Airflow/job scheduler, REST API, etc (in progress)

It heavily leverages [pydantic](https://docs.pydantic.dev/latest/), and users are expected to implement their own configuration and workflow building blocks by implementing pydantic models.

It also integrates closely with [hydra](https://hydra.cc/) for file-based configuration and command line interaction, but can also be used natively from Python without it.

This library was partially inspired by this [blog post](https://towardsdatascience.com/configuration-management-for-model-training-experiments-using-pydantic-and-hydra-d14a6ae84c13) by Suneeta Mall ([@suneeta-mall](https://github.com/suneeta-mall)).
We have taken these ideas a step further by introducing the concept of the `ModelRegistry`, which allows for the configs to be managed without `hydra`, and also allows us to implement dependency injection.

We aim to provide additional (and optional) tools for workflow orchestration on top of the configuration framework.

[More information is available in our wiki](https://github.com/Point72/ccflow/wiki)

## Installation

`ccflow` can be installed via [pip](https://pip.pypa.io) or [conda](https://docs.conda.io/en/latest/), the two primary package managers for the Python ecosystem.

To install `ccflow` via **pip**, run this command in your terminal:

```bash
pip install ccflow
```

To install `ccflow` via **conda**, run this command in your terminal:

```bash
conda install ccflow -c conda-forge
```

## Community

- [Contribute](https://github.com/Point72/ccflow/wiki/Contribute) to `ccflow` and help improve the project

## License

This software is licensed under the Apache 2.0 license. See the [LICENSE](https://github.com/Point72/ccflow/blob/main/LICENSE) file for details.
