# Tutorials

These tutorials are hands-on lessons. Work through them at the keyboard, in order — each one builds on the last, and every step is meant to succeed exactly as written. They teach by *doing*; the reasoning behind what you are doing lives in the [Explanation](Explanation) pages, and task-focused recipes live in the [How-to Guides](How-to-Guides).

1. **[First Steps](First-Steps)** — define a couple of configuration objects, register them, and see how the registry links them together. The shortest path to the core idea.
1. **[Configuring Models](Configuring-Models)** — build up strongly typed, hierarchical configuration with pydantic models, register them, and wire dependencies between them.
1. **[Defining Workflows](Defining-Workflows)** — turn configuration into runnable steps with contexts, results, and callable models, and meet the evaluators that run them.
1. **[Building an ETL Pipeline](Building-an-ETL-Pipeline)** — assemble extract, transform, and load steps into an end-to-end pipeline driven by a single Hydra config.
1. **[Composing an ETL Application](Composing-an-ETL-Application)** — grow that pipeline into a reusable, config-group-driven application with command-line dispatch.
1. **[Building a Configurable Calculator](Building-a-Configurable-Calculator)** — the capstone: combine the functional `@Flow.model` API with config groups and CLI dispatch to build a fully configurable program from the command line.

New to `ccflow`? Start at [First Steps](First-Steps). If you want to understand *why* the framework is shaped this way before diving in, read [Core Concepts](Core-Concepts) first.
