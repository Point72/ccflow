import unittest
from datetime import date

from ccflow.utils.chunker import dates_to_chunks


class TestChunker(unittest.TestCase):
    def test_dates_to_chunks_M(self):
        out = dates_to_chunks(date(2022, 4, 6), date(2022, 4, 7), "1mo")
        self.assertListEqual(out, [(date(2022, 4, 1), date(2022, 4, 30))])

        out = dates_to_chunks(date(2022, 4, 1), date(2022, 4, 30), "1mo")
        self.assertListEqual(out, [(date(2022, 4, 1), date(2022, 4, 30))])

        out = dates_to_chunks(date(2022, 3, 6), date(2022, 4, 7), "1mo")
        self.assertListEqual(
            out,
            [
                (date(2022, 3, 1), date(2022, 3, 31)),
                (date(2022, 4, 1), date(2022, 4, 30)),
            ],
        )
        out = dates_to_chunks(date(2022, 1, 6), date(2022, 4, 7), "1mo")
        self.assertListEqual(
            out,
            [
                (date(2022, 1, 1), date(2022, 1, 31)),
                (date(2022, 2, 1), date(2022, 2, 28)),
                (date(2022, 3, 1), date(2022, 3, 31)),
                (date(2022, 4, 1), date(2022, 4, 30)),
            ],
        )

        # Using 2mo chunks, make sure the chunk is the first two months of the year
        out = dates_to_chunks(date(2022, 1, 6), date(2022, 1, 7), "2mo")
        self.assertListEqual(out, [(date(2022, 1, 1), date(2022, 2, 28))])
        out = dates_to_chunks(date(2022, 2, 6), date(2022, 2, 7), "2mo")
        self.assertListEqual(out, [(date(2022, 1, 1), date(2022, 2, 28))])
        out = dates_to_chunks(date(1950, 2, 6), date(1950, 2, 7), "2mo")
        self.assertListEqual(out, [(date(1950, 1, 1), date(1950, 2, 28))])
        self.assertListEqual(dates_to_chunks(date(2024, 3, 31), date(2024, 3, 31), "13mo"), [(date(2024, 3, 1), date(2025, 3, 31))])
        self.assertListEqual(dates_to_chunks(date(2022, 2, 6), date(2022, 2, 7), "1q"), [(date(2022, 1, 1), date(2022, 3, 31))])

    def test_dates_to_chunks_d(self):
        out = dates_to_chunks(date(2022, 4, 6), date(2022, 4, 6), "1d")
        self.assertListEqual(out, [(date(2022, 4, 6), date(2022, 4, 6))])
        out = dates_to_chunks(date(2022, 4, 1), date(2022, 4, 6), "1d")
        self.assertListEqual(
            out,
            [
                (date(2022, 4, 1), date(2022, 4, 1)),
                (date(2022, 4, 2), date(2022, 4, 2)),
                (date(2022, 4, 3), date(2022, 4, 3)),
                (date(2022, 4, 4), date(2022, 4, 4)),
                (date(2022, 4, 5), date(2022, 4, 5)),
                (date(2022, 4, 6), date(2022, 4, 6)),
            ],
        )

        # The choice if whether the chunk is 2022-04-06:2022-04-07 or 2022-04-05:2022-04-06
        # is arbitrary and not important, as long as it's the same for both.
        chunk = (date(2022, 4, 6), date(2022, 4, 7))
        out = dates_to_chunks(chunk[0], chunk[0], "2d")
        self.assertListEqual(out, [chunk])
        out = dates_to_chunks(chunk[1], chunk[1], "2d")
        self.assertListEqual(out, [chunk])

        out = dates_to_chunks(chunk[1], chunk[0], "1d")
        self.assertListEqual(out, [])

    def test_dates_to_chunks_trim(self):
        out = dates_to_chunks(date(2022, 4, 6), date(2022, 4, 7), "1mo", trim=True)
        self.assertListEqual(out, [(date(2022, 4, 6), date(2022, 4, 7))])

        out = dates_to_chunks(date(2022, 4, 1), date(2022, 4, 30), "1mo", trim=True)
        self.assertListEqual(out, [(date(2022, 4, 1), date(2022, 4, 30))])

        out = dates_to_chunks(date(2022, 3, 6), date(2022, 4, 7), "1mo", trim=True)
        self.assertListEqual(
            out,
            [
                (date(2022, 3, 6), date(2022, 3, 31)),
                (date(2022, 4, 1), date(2022, 4, 7)),
            ],
        )
        out = dates_to_chunks(date(2022, 1, 6), date(2022, 4, 7), "1mo", trim=True)
        self.assertListEqual(
            out,
            [
                (date(2022, 1, 6), date(2022, 1, 31)),
                (date(2022, 2, 1), date(2022, 2, 28)),
                (date(2022, 3, 1), date(2022, 3, 31)),
                (date(2022, 4, 1), date(2022, 4, 7)),
            ],
        )

        # Using 2mo chunks, make sure the chunk is the first two months of the year
        out = dates_to_chunks(date(2022, 1, 6), date(2022, 3, 7), "2mo", trim=True)
        self.assertListEqual(
            out,
            [
                (date(2022, 1, 6), date(2022, 2, 28)),
                (date(2022, 3, 1), date(2022, 3, 7)),
            ],
        )
        out = dates_to_chunks(date(2022, 2, 6), date(2022, 3, 7), "2mo", trim=True)
        self.assertListEqual(
            out,
            [
                (date(2022, 2, 6), date(2022, 2, 28)),
                (date(2022, 3, 1), date(2022, 3, 7)),
            ],
        )

    def test_dates_to_chunks_rejects_sub_day_and_period_end(self):
        for chunk_size in ("1h", "24h", "2ME"):
            self.assertRaises(ValueError, dates_to_chunks, date(2024, 1, 1), date(2024, 1, 2), chunk_size)

    def test_dates_to_chunks_weekly_monday_aligned(self):
        # Weeks run Monday..Sunday regardless of the query start (ISO weeks).
        self.assertListEqual(
            dates_to_chunks(date(2024, 1, 3), date(2024, 1, 20), "1w"),
            [
                (date(2024, 1, 1), date(2024, 1, 7)),
                (date(2024, 1, 8), date(2024, 1, 14)),
                (date(2024, 1, 15), date(2024, 1, 21)),
            ],
        )
        # Multi-week chunks are counted from Monday 1970-01-05.
        self.assertListEqual(
            dates_to_chunks(date(2024, 1, 3), date(2024, 1, 20), "2w"),
            [(date(2023, 12, 25), date(2024, 1, 7)), (date(2024, 1, 8), date(2024, 1, 21))],
        )

    def test_dates_to_chunks_reversed_within_one_chunk(self):
        self.assertListEqual(dates_to_chunks(date(2024, 1, 2), date(2024, 1, 1), "1mo", trim=True), [])
