import unittest
from datetime import date

from ccflow.utils.chunker import dates_to_chunks


class TestChunker(unittest.TestCase):
    def test_dates_to_chunks_M(self):
        out = dates_to_chunks(date(2022, 4, 6), date(2022, 4, 7), "M")
        self.assertListEqual(out, [(date(2022, 4, 1), date(2022, 4, 30))])

        out = dates_to_chunks(date(2022, 4, 1), date(2022, 4, 30), "M")
        self.assertListEqual(out, [(date(2022, 4, 1), date(2022, 4, 30))])

        out = dates_to_chunks(date(2022, 3, 6), date(2022, 4, 7), "M")
        self.assertListEqual(
            out,
            [
                (date(2022, 3, 1), date(2022, 3, 31)),
                (date(2022, 4, 1), date(2022, 4, 30)),
            ],
        )
        out = dates_to_chunks(date(2022, 1, 6), date(2022, 4, 7), "M")
        self.assertListEqual(
            out,
            [
                (date(2022, 1, 1), date(2022, 1, 31)),
                (date(2022, 2, 1), date(2022, 2, 28)),
                (date(2022, 3, 1), date(2022, 3, 31)),
                (date(2022, 4, 1), date(2022, 4, 30)),
            ],
        )

        # Using 2M chunks, make sure the chunk is the first two months of the year
        out = dates_to_chunks(date(2022, 1, 6), date(2022, 1, 7), "2M")
        self.assertListEqual(out, [(date(2022, 1, 1), date(2022, 2, 28))])
        out = dates_to_chunks(date(2022, 2, 6), date(2022, 2, 7), "2M")
        self.assertListEqual(out, [(date(2022, 1, 1), date(2022, 2, 28))])

    def test_dates_to_chunks_d(self):
        out = dates_to_chunks(date(2022, 4, 6), date(2022, 4, 6), "d")
        self.assertListEqual(out, [(date(2022, 4, 6), date(2022, 4, 6))])
        out = dates_to_chunks(date(2022, 4, 1), date(2022, 4, 6), "d")
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

        out = dates_to_chunks(chunk[1], chunk[0], "d")
        self.assertListEqual(out, [])

    def test_dates_to_chunks_trim(self):
        out = dates_to_chunks(date(2022, 4, 6), date(2022, 4, 7), "M", trim=True)
        self.assertListEqual(out, [(date(2022, 4, 6), date(2022, 4, 7))])

        out = dates_to_chunks(date(2022, 4, 1), date(2022, 4, 30), "M", trim=True)
        self.assertListEqual(out, [(date(2022, 4, 1), date(2022, 4, 30))])

        out = dates_to_chunks(date(2022, 3, 6), date(2022, 4, 7), "M", trim=True)
        self.assertListEqual(
            out,
            [
                (date(2022, 3, 6), date(2022, 3, 31)),
                (date(2022, 4, 1), date(2022, 4, 7)),
            ],
        )
        out = dates_to_chunks(date(2022, 1, 6), date(2022, 4, 7), "M", trim=True)
        self.assertListEqual(
            out,
            [
                (date(2022, 1, 6), date(2022, 1, 31)),
                (date(2022, 2, 1), date(2022, 2, 28)),
                (date(2022, 3, 1), date(2022, 3, 31)),
                (date(2022, 4, 1), date(2022, 4, 7)),
            ],
        )

        # Using 2M chunks, make sure the chunk is the first two months of the year
        out = dates_to_chunks(date(2022, 1, 6), date(2022, 3, 7), "2M", trim=True)
        self.assertListEqual(
            out,
            [
                (date(2022, 1, 6), date(2022, 2, 28)),
                (date(2022, 3, 1), date(2022, 3, 7)),
            ],
        )
        out = dates_to_chunks(date(2022, 2, 6), date(2022, 3, 7), "2M", trim=True)
        self.assertListEqual(
            out,
            [
                (date(2022, 2, 6), date(2022, 2, 28)),
                (date(2022, 3, 1), date(2022, 3, 7)),
            ],
        )
