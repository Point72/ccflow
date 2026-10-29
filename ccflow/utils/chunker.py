"""Functionality to chunk query parameters or simple data structures.

This is useful as part of creating caching schemes for time series data, where one wants to choose a chunk/page size,
i.e. monthly, daily, etc, and any time any data is needed from the chunk, to load and cache the entire chunk.
Control over the chunk size is important: if it's too big, too much un-necessary data gets loaded, but if it's too
small, performance suffers from too many repeated trips to the underlying data store for long-range queries.
"""

from datetime import date, timedelta

from ccflow.exttypes.frequency import CalendarFrequency, Frequency

# Chunks are polars ``truncate`` buckets: counted from 1970-01-01, and for weeks from the first Monday after it so weeks run Monday..Sunday.
_EPOCH = date(1970, 1, 1)
_WEEK_EPOCH = date(1970, 1, 5)
_ONE_DAY = timedelta(days=1)
# Months in each month-based unit, for counting month buckets.
_MONTHS = {"mo": 1, "q": 3, "y": 12}

__all__ = ("dates_to_chunks",)


def dates_to_chunks(start: date, end: date, chunk_size: str = "1mo", trim: bool = False) -> list[tuple[date, date]]:
    """
    Chunks a date range in a consistent way (i.e. the same middle chunks will always be generated for overlapping
    ranges).

    Args:
        start: The start date of the time interval to convert to chunks
        end: The end date of the time interval to convert to chunks
        chunk_size: A positive calendar frequency, e.g. '1d', '2w', '1mo'. Chunks are counted from 1970-01-01 (weeks from Monday 1970-01-05), so any
            two ranges get the same chunks for the dates they share.
        trim: Whether to trim the ends to match start and end date exactly (versus standard-sized chunks that cover the interval)

    Returns:
        List of tuples of (start date, end date) for each of the chunks; empty if ``start > end``
    """
    freq = CalendarFrequency.validate(chunk_size)
    if start > end:
        return []
    chunks = [(_bucket_start(k, freq), _bucket_start(k + 1, freq) - _ONE_DAY) for k in range(_bucket(start, freq), _bucket(end, freq) + 1)]
    if trim:
        chunks[0] = (max(chunks[0][0], start), chunks[0][1])
        chunks[-1] = (chunks[-1][0], min(chunks[-1][1], end))
    return chunks


def _bucket(d: date, freq: Frequency) -> int:
    """Index of the chunk holding ``d``, counted from the epoch."""
    if freq.unit in _MONTHS:
        return (12 * (d.year - _EPOCH.year) + d.month - 1) // (freq.n * _MONTHS[freq.unit])
    return (d - (_WEEK_EPOCH if freq.unit == "w" else _EPOCH)) // freq.timedelta


def _bucket_start(k: int, freq: Frequency) -> date:
    if freq.unit in _MONTHS:
        years, month = divmod(k * freq.n * _MONTHS[freq.unit], 12)
        return date(_EPOCH.year + years, month + 1, 1)
    return (_WEEK_EPOCH if freq.unit == "w" else _EPOCH) + k * freq.timedelta
