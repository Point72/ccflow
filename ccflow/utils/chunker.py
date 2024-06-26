"""Functionality to chunk query parameters or simple data structures.

This is useful as part of creating caching schemes for time series data, where one wants to choose a chunk/page size,
i.e. monthly, daily, etc, and any time any data is needed from the chunk, to load and cache the entire chunk.
Control over the chunk size is important: if it's too big, too much un-necessary data gets loaded, but if it's too
small, performance suffers from too many repeated trips to the underlying data store for long-range queries.
"""

import pandas as pd
from datetime import date
from typing import List, Tuple

_MIN_END_DATE = date(1969, 12, 31)

__all__ = ("dates_to_chunks",)


def dates_to_chunks(start: date, end: date, chunk_size: str = "M", trim: bool = False) -> List[Tuple[date, date]]:
    """
    Chunks a date range in a consistent way (i.e. the same middle chunks will always be generated for overlapping
    ranges).

    :param start: The start date of the time interval to convert to chunks
    :type start: date
    :param end: The end date of the time interval to convert to chunks
    :type end: date
    :param chunk_size: Any valid Pandas frequency string (with no multiplier). i.e. 'W', 'D', 'M' but not '2W'.
    :type chunk_size: str
    :param trim: Whether to trim the ends to match start and end date exactly
    :type trim: bool
    :return: generator that produces tuples: (start date, end date)
    """
    offset = pd.tseries.frequencies.to_offset(chunk_size)
    if offset.n == 1:
        end_dates = pd.date_range(start - offset, end + offset, freq=chunk_size)
    else:
        # Need to anchor the timeline at some absolute date, because otherwise chunks might depend on the start date
        # and end up overlappig each other, i.e. with 2M, would end up with
        # i.e. (Jan-Feb) or (Feb,Mar) depending on whether start date was in Jan or Feb,
        # instead of always returning (Jan,Feb) for any start date in either of those two months.
        end_dates = pd.date_range(_MIN_END_DATE, end + offset, freq=chunk_size)
    start_dates = end_dates + pd.DateOffset(1)
    chunks = [(s, e) for s, e in zip(start_dates[:-1].date, end_dates[1:].date) if e >= start and s <= end]
    if trim:
        if chunks[0][0] < start:
            chunks[0] = (start, chunks[0][1])
        if chunks[-1][-1] > end:
            chunks[-1] = (chunks[-1][0], end)
    return chunks
