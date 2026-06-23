"""Flow.model fixtures used by Hydra integration tests."""

from datetime import date

from ccflow import Flow, FromContext, GenericResult


@Flow.model
def basic_loader(source: str, multiplier: int, value: FromContext[int]) -> GenericResult[int]:
    return GenericResult(value=value * multiplier)


@Flow.model
def data_source(base_value: int, value: FromContext[int]) -> GenericResult[int]:
    return GenericResult(value=value + base_value)


@Flow.model
def data_transformer(source: int, factor: int) -> GenericResult[int]:
    return GenericResult(value=source * factor)


@Flow.model
def data_aggregator(input_a: int, input_b: int, operation: str = "add") -> GenericResult[int]:
    if operation == "add":
        return GenericResult(value=input_a + input_b)
    raise ValueError(f"unsupported operation: {operation}")


@Flow.model
def contextual_loader(source: str, start_date: FromContext[date], end_date: FromContext[date]) -> GenericResult[dict]:
    return GenericResult(
        value={
            "source": source,
            "start_date": str(start_date),
            "end_date": str(end_date),
        }
    )


@Flow.model
def contextual_processor(
    prefix: str,
    data: dict,
    start_date: FromContext[date],
    end_date: FromContext[date],
) -> GenericResult[str]:
    del start_date, end_date
    return GenericResult(value=f"{prefix}:{data['source']}:{data['start_date']} to {data['end_date']}")
