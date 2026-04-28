"""Unit tests for ``agglovar.bed.merge.melt_depth``.

The melt encoding is a key correctness contract; pin it explicitly via the
docstring examples plus a few extra cases.
"""

from __future__ import annotations

import polars as pl

from agglovar.bed.merge import melt_depth


def _melt(df: pl.DataFrame) -> list[tuple]:
    """Return melt_depth output as a sorted list of tuples for comparison."""
    return (
        melt_depth(df.lazy(), col_names=('chrom', 'pos', 'end'))
        .collect()
        .rows()
    )


class TestMeltDepth:
    """Pin the event encoding."""

    def test_single_record(self) -> None:
        df = pl.DataFrame(
            {'chrom': ['chr1'], 'pos': [100], 'end': [200]},
            schema={'chrom': pl.String, 'pos': pl.Int64, 'end': pl.Int64},
        )

        rows = _melt(df)
        # Expected events at (100, +1), (200, 0), (200, -1) → cumsum 1, 1, 0
        assert rows == [
            ('chr1', 100, 1),
            ('chr1', 200, 1),
            ('chr1', 200, 0),
        ]

    def test_two_overlapping_docstring_first(self) -> None:
        """Match the first docstring example."""
        df = pl.DataFrame(
            {
                'chrom': ['chr1', 'chr1'],
                'pos': [100, 120],
                'end': [160, 200],
            },
            schema={'chrom': pl.String, 'pos': pl.Int64, 'end': pl.Int64},
        )

        rows = _melt(df)
        assert rows == [
            ('chr1', 100, 1),
            ('chr1', 120, 2),
            ('chr1', 160, 2),
            ('chr1', 160, 1),
            ('chr1', 200, 1),
            ('chr1', 200, 0),
        ]

    def test_two_overlapping_docstring_second(self) -> None:
        """Match the second docstring example (alternate input order)."""
        df = pl.DataFrame(
            {
                'chrom': ['chr1', 'chr1'],
                'pos': [100, 120],
                'end': [200, 160],
            },
            schema={'chrom': pl.String, 'pos': pl.Int64, 'end': pl.Int64},
        )

        rows = _melt(df)
        assert rows == [
            ('chr1', 100, 1),
            ('chr1', 120, 2),
            ('chr1', 160, 2),
            ('chr1', 160, 1),
            ('chr1', 200, 1),
            ('chr1', 200, 0),
        ]

    def test_adjacent_records_match_docstring(self) -> None:
        """Adjacent (touching) records peak at depth 2 at the boundary."""
        df = pl.DataFrame(
            {
                'chrom': ['chr1', 'chr1'],
                'pos': [100, 200],
                'end': [200, 300],
            },
            schema={'chrom': pl.String, 'pos': pl.Int64, 'end': pl.Int64},
        )

        rows = _melt(df)
        assert rows == [
            ('chr1', 100, 1),
            ('chr1', 200, 2),
            ('chr1', 200, 2),
            ('chr1', 200, 1),
            ('chr1', 300, 1),
            ('chr1', 300, 0),
        ]

    def test_multiple_chromosomes_partition_independently(self) -> None:
        """cum_sum is partitioned by chromosome."""
        df = pl.DataFrame(
            {
                'chrom': ['chr1', 'chr2'],
                'pos': [100, 100],
                'end': [200, 200],
            },
            schema={'chrom': pl.String, 'pos': pl.Int64, 'end': pl.Int64},
        )

        rows = _melt(df)
        # Each chromosome should reach depth 1 then return to 0 independently.
        chr1 = [r for r in rows if r[0] == 'chr1']
        chr2 = [r for r in rows if r[0] == 'chr2']

        assert max(r[2] for r in chr1) == 1
        assert max(r[2] for r in chr2) == 1
        assert chr1[-1][2] == 0
        assert chr2[-1][2] == 0
