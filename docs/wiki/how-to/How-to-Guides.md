# How-to Guides

These guides are recipes for getting a specific job done. They assume you already know the basics from the [Tutorials](Tutorials) and get straight to the task. If you want to understand *why* rather than *how*, see the [Explanation](Explanation) pages; for exact signatures and catalogs, see the [Reference](Reference).

**Setup**

- [Install ccflow](Installation) — pip, conda, or from source.

**Configuration**

- [Configure Complex Values](Configure-Complex-Values) — custom validation and coercion, Jinja/SQL templates, Polars expressions, NumPy arrays, arbitrary types, and object-by-path references.
- [Bind Logic to Configs](Bind-Logic-to-Configs) — attach business logic to configuration classes, including custom publishers and data pipelines.

**Running workflows**

- [Run Workflows from the CLI](Run-Workflows-from-the-CLI) — run configured callables, apply overrides, and inspect the composed configuration.
- [Cache Results](Cache-Results) — avoid redundant work with in-memory caching and graph evaluation, and write your own evaluator.
- [Retry on Failure](Retry-on-Failure) — make flaky steps resilient with retry evaluators and `RetryModel`.
