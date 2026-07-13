# Explanation

These pages build understanding of `ccflow` — what it is for, the ideas behind its design, and why it leans on the tools it does. They are for reading and reflection rather than for following along at the keyboard; when you want to *do* something, reach for a [How-to Guide](How-to-Guides) or work through a [Tutorial](Tutorials) instead.

- **[Core Concepts](Core-Concepts)** — the vocabulary of `ccflow` (models, the registry, contexts, results, callable models, evaluators, publishers) and how the configuration half and the workflow half fit together into one framework.
- **[Design Goals](Design-Goals)** — the problems `ccflow` set out to solve for configuration and workflow management, and the requirements that shaped it.
- **[Configuration and Hydra](Configuration-and-Hydra)** — why file-based configuration and a command line matter, what `hydra` and its config groups bring, and how the `ModelRegistry` complements them to enable composable, dispatchable applications.

If you are new, the [Design Goals](Design-Goals) explain the "why does this exist at all", [Core Concepts](Core-Concepts) give you the shared language used everywhere else, and [Configuration and Hydra](Configuration-and-Hydra) motivates the composition style used in the [Composing an ETL Application](Composing-an-ETL-Application) tutorial.
