# Key Features

The `ccflow` framework is a collection of tools and patterns for application and workflow configuration.
Its intended uses include ETL, data validation, model training, live trading configuration, backtesting, hyperparameter search, and automated report generation.

## Base Model

Central to `ccflow` is the `BaseModel` class.
`BaseModel` is the base class for models in the `ccflow` framework.
A model is basically a data class (class with attributes).
It has nothing to do with mathematical models (sorry if this causes confusion).
The naming was inspired by the open source library [Pydantic](https://docs.pydantic.dev/latest/)(`BaseModel` actually inherits from the Pydantic base model class).

## Callable Model

`CallableModel` is the base class for a special type of `BaseModel` which can be called.
`CallableModel`'s are called with a context (something that derives from `ContextBase`) and returns a result (something that derives from `ResultBase`).
As an example, you may have a `SQLReader` callable model that when called with a `DateRangeContext` returns a `ArrowResult` (wrapper around a Arrow table) with data in the date range defined by the context by querying some SQL database.


## Model Registry

A `ModelRegistry` is a named collection of models.
A `ModelRegistry` can be loaded from YAML configuration, which means you can define a collection of models using YAML.
This is really powerful because this gives you a easy way to define a collection of Python objects via configuration.


## Models

Although you are free to define your own models (`BaseModel` implementations) to use in your flow graph,
`ccflow` comes with some models that you can use off the shelf to solve common problems.

# Readers

`ccflow` comes with a range of models for reading data.
The following table summarizes the "reader" models.

TODO

## Publishers

`ccflow` also comes with a range of models for writing data.
These are referred to as publishers.
The following table summarizes the "publisher" models:

TODO

You can "chain" publishers and callable models using `PublisherModel` to call a `CallableModel` and publish
the results in one step.
In fact, `ccflow` comes with several implementations of `PublisherModel` for common publishing use cases.

TODO


## Utilities

`ccflow` also comes with some utility models that might be useful in your workflow.
The following table summaries these:

TODO

## Evaluators

`ccflow` comes with "evaluators" that allows you to evaluate (i.e. run) `CallableModel` s in different
ways.

TODO

## Hydra

`ccflow` is integrated with [Hydra](https://hydra.cc/docs/intro/).
This allows for composable and hierarchical configurations.
Please refer to Hydra's documentation for more details.
