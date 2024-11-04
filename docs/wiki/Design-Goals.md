- [Configuration Design Goals](#configuration-design-goals)
  - [Interactivity](#interactivity)
  - [Schemas](#schemas)
  - [Hierarchy](#hierarchy)
- [Workflow Design Goals](#workflow-design-goals)
  - [Ease of Use](#ease-of-use)
  - [Configuration](#configuration)
  - [Parameterization](#parameterization)

## Configuration Design Goals

In both production applications and research workflows, the need arises to configure various components. As these applications and workflows get increasingly complex, so do the patterns and frameworks that people use for configuration.
While some of this complexity is unavoidable, in an ideal world, there is a single well-designed (hopefully!) framework that can be used across all use cases, spanning data retrieval, validation, transformation, and loading (i.e. ETL workflows),
model training, microservice configuration, and automated report generation.

In order to meet the demands of these varying applications, the ideal configuration framework must satisfy several needs.

### Interactivity

Since the aim is to use configuration for both production applications/workflows as well as for research, there needs to be both a relatively static, well-controlled way of defining the entirety of the configuration, as well as much more dynamic ways of interacting and iterating over the configuration options.
Versioned file-based configurations are almost always used to accomplish the former, and flexible command line interfaces are often used to satisfy the latter, but in an ideal world, it should be possible to both modify and add completely new configurations directly from a python script or notebook for research,
without having to resort to modifying files or leveraging command line overrides (though this is also useful functionality to have).

### Schemas

As the framework scales, the chances increase of two common types of errors:

1. Misnaming or mistyping of configuration options, that could cause the configuration to silently fail (if the parameter was optional) i.e. "threshold" vs "threshold"
1. Type errors and value constraints, i.e. the identifier string "12345" vs the integer 1234, or specifying that a parameter "sigma" should be non-negative.

Typically, we want to catch these errors as soon as possible: when the configurations are loaded rather than when they are used. This also allows for writing testing of configurations that is decoupled from testing the logic that depends on the configurations, making it easier to spot issues quickly and easily.

In order to solve these issues, configurations need strongly typed schemas, with the option to perform additional (and custom) validations. There may be a need to coerce values (i.e. if the string "1234" is passed to a parameter that expects an int, it may be desirable to coerce it to 1234), and additional validation may be needed on the entire structure (to test validity of combinations of parameters rather than just parameters themselves). The use of schemas also means there must be a way to evolve the schema over time (adding and removing attributes), and even to version it if necessary.

In `ccflow`, we leverage the power of the very popular [pydantic](https://pydantic-docs.helpmanual.io/) library to tackle these issues, with some additional extensions. Note that while python's [dataclasses](https://docs.python.org/3/library/dataclasses.html) solve the misnaming/mistyping problem, they do not provide type checking or additional run-time validation. One can think of pydantic as a powerful extension of dataclasses which does.

### Hierarchy

Quantitative workflows are typically very hierarchical in nature.
For example, a feature generation workflow might depend on construction (and retrieval) of multiple feature sets, and each feature set may depend on its own techniques and data sources, and each technique will have its own configuration parameters,
and each data source will also have parameters that configure how it was cleaned/transformed and how to access it.
Thus, the configuration framework must have a modular and hierarchical structure, which means that entire parts of the hierarchy must be easy to add and remove without affecting the rest of the configuration.
In a file-based representation, this means that the configuration should be spreadable across multiple files spanning several sub-directories. The interactive representation of these configs must mirror the same kind of structure.

Furthermore, the hierarchy of configurations can have complex dependencies, forming a graph structure, rather than a simple tree.
For example, a data source may be configured to be transformed in a particular way, and then used in multiple signals, which are then all used as part of portfolio construction.
If changing the configuration of the data source, it is then important that all the signals pick up this change. The challenge lies in defining this graph structure both statically (i.e. for trading) as well as dynamically in the python code (for research).

Lastly, there should ideally be a way to automatically map a piece of configuration to the code that decides how to use it.
Without this, the configuration can end up acting like a large catalog of global variables that proliferate throughout the codebase, with all the same drawbacks as global variables (including increased coupling between everything).
So, each piece of configuration should get used by as few high-level pieces of code as possible, rather than by multiple low-level pieces of code.
We tackle this problem by frequently binding together the configuration parameters and the code which uses the configuration in a single object.

In `ccflow`, we leverage the power of Meta's [hydra](https://hydra.cc/) library for file-based and command line configuration management, but add some of our own functionality to support the interactive configuration use case.

## Workflow Design Goals

We define a "workflow" solution to mean a library to help define and run a collection of inter-dependent tasks (or steps). We can break this down further into separate components

1. Defining (via configuration) what tasks/steps make up the workflow
1. Passing data between tasks, so that each task has the information that it needs from upstream tasks
1. Determining the order in which to run the tasks (often referred to scheduling, or more appropriately "task scheduling")
1. Advanced features such as caching, distributed evaluation, monitoring, UI's, etc
1. Automating the launch of the workflow so that it runs regularly according to some rules (also referred to as scheduling, or more appropriately "workflow scheduling")

There are numerous existing packages and products in the Python ecosystem which tackle the problem of workflow management, each written with different use cases in mind and supporting different sets of requirements and features.
We feel that most of the existing open source solutions to this problem tend to focus on (or are marketed on) the later elements in the list above rather than the earlier ones. Our approach is to focus on the components roughly in the order they are listed.

### Ease of Use

As much as possible, it should be easy and intuitive to define workflows in the framework. Simple things should be easy, but arbitrarily complex things should remain possible.

Furthermore, we should not impose too many constraints on how users write their code - they should be able to bring their existing analytics, no matter what underlying python packages or tools they use, and hook it into the framework.

We aim to leverage standard and familiar programming paradigms as much as possible (writing objects and functions), as they are time-tested and easy for users to understand.
By using composition of classes and functions and their return values, the python language essentially handles items 2. and 3. above for us, without a need to do anything special or for users to learn anything new.

We are not trying to write a new language that people have to learn in order to implement their analytics. However, the framework should support workflow steps that use any such "language" that already exists (i.e. polars/pytorch/jax/csp/etc).

We do not wish to make assumptions about how data is represented within the framework, or even that all data should be tabular or array-like; we should be able to support documents, charts, event streams or any other kinds of objects as part of the workflows.

We should be able to offer common tools (within the framework) to facilitate common tabular data processing tasks (i.e. reading and writing) on some popular formats. Since our aim is not to support all use cases and integrations out of the box, it should be straightforward for users to write their own integrations.

Once defined/configured, we would like a workflow to have a very simple way to run it, whether interactively or from command line. Furthermore, it should be equally easy to run any intermediate step of a workflow (and it's dependencies) to promote reusability and make debugging easy.

### Configuration

Given a configuration framework meeting the design goals laid out above, suitable for both production and research configuration, a key requirement is to be able to configure workflows using the same framework.

This implies that workflows should be easy to define from version configs (i.e. using files) in production, or to change and re-run interactively from python (for research). Thus, the workflow, the steps in the workflow, and the objects used by those steps will all belong to the same configuration paradigm.

### Parameterization

While the configuration framework allows for arbitrary complexity in the configuration and definition of the workflow steps (and thus of the workflow), it is often natural to think of the workflows as being parameterized (or templatized) across certain dimensions, and to treat these "context" parameters separately from other configuration options to make it easier to run multiple templatized workflows without needing to re-configure anything.

For example, many data processing workflows are parameterized by date. However, this is not the only option; in many cases it is more efficient to process data across a date range. Taking the idea further, one may also want to specify a workflow that applies to a specific region and time range, or even down to a symbol and date range level. In the realm of data orchestration workflows, another way to think about the "context" is as the definition of the smallest "chunk" of data we are willing to operate on in a step.

Thus, we want to be able to define a flexible, parametric "context" for each step, such that the step can be easily run across multiple contexts, and depend on other steps, each of which may use the same context or even a different context.

A technical reason for parameterizing the steps separately from the configuration is to prevent run-time mutation of the configuration, which is dangerous as configurations are shared across multiple components.
