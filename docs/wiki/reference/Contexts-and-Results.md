# Contexts and Results

The built-in context and result types. Contexts parameterize workflow steps; results carry their output. Both are `BaseModel` subclasses, so they validate and serialize like any config. See [Defining Workflows](Defining-Workflows) for usage.

## Contexts

All contexts derive from `ContextBase` and are frozen and hashable so they can serve as cache keys.

| Type                   | Fields                      | Notes                                        |
| :--------------------- | :-------------------------- | :------------------------------------------- |
| `NullContext`          | none                        | For steps with no runtime parameters.        |
| `GenericContext[T]`    | `value: T`                  | Holds an arbitrary (optionally typed) value. |
| `DateContext`          | `date`                      | Single date.                                 |
| `DatetimeContext`      | `dt`                        | Single datetime.                             |
| `DateRangeContext`     | `start_date`, `end_date`    | A date range.                                |
| `VersionedDateContext` | `date`, `entry_time_cutoff` | Date with an as-of cutoff.                   |
| `UniverseContext`      | `universe`                  | A named universe.                            |
| `UniverseDateContext`  | `universe`, `date`          | Universe plus date.                          |
| `ModelDateContext`     | `mode`, `date`              | Model name plus date.                        |

### Validation conveniences

`NullContext` validates from `None` or an empty container:

```python
NullContext.model_validate(None)   # NullContext()
NullContext.model_validate([])     # NullContext()
```

`GenericContext[T]` coerces its value to `T`:

```python
GenericContext[str].model_validate(100)   # GenericContext[str](value='100')
```

Date-based contexts accept strings and tuples, including relative offsets (handy from the command line):

```python
DateContext.model_validate("2025-01-01")        # explicit date
DateContext.model_validate("0d")                # today
DateRangeContext.model_validate(("-7d", "0d"))  # last 7 days through today
```

## Results

All results derive from `ResultBase`.

| Type                  | Holds                                                                                                  |
| :-------------------- | :----------------------------------------------------------------------------------------------------- |
| `GenericResult[T]`    | An arbitrary (optionally typed) `value`.                                                               |
| `PandasResult`        | A pandas `DataFrame`.                                                                                  |
| `NDArrayResult`       | A NumPy array.                                                                                         |
| `ArrowResult`         | A PyArrow table.                                                                                       |
| `XArrayResult`        | An xarray structure.                                                                                   |
| `NarwhalsFrameResult` | A dataframe via [Narwhals](https://narwhals-dev.github.io/narwhals/), for cross-library compatibility. |

### `GenericResult`

Holds anything, with optional generic typing and boilerplate-reducing validation:

```python
GenericResult(value={"x": "foo", "y": 5.0})     # any value
GenericResult[str](value="Any string")          # typed
GenericResult.model_validate("Any string")      # bare value validated into the wrapper
```

### Custom results

Define a schema by subclassing `ResultBase`:

```python
from ccflow import ResultBase

class MyResult(ResultBase):
    x: str
    y: float
```

## See also

- [Core Types](Core-Types) — `ContextBase`, `ResultBase`, and the surrounding machinery.
- [Defining Workflows](Defining-Workflows) — contexts and results in a running workflow.
