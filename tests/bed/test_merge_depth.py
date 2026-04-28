"""Unit tests for ``agglovar.bed.merge.merge_depth``.

Compared against the brute-force oracle in :mod:`tests.bed.oracle` over
hand-crafted DataFrames and the synthetic1 dataset.
"""

from __future__ import annotations

import polars as pl
import pytest

from agglovar.bed.merge import merge_depth

from tests.bed.datasets import BedDataset
from tests.bed.oracle import MergedRegion, expected_merge_depth


def _df(rows: list[tuple[str, int, int]]) -> pl.DataFrame:
    return pl.DataFrame(
        rows,
        schema={'chrom': pl.String, 'pos': pl.Int64, 'end': pl.Int64},
        orient='row',
    )


def _run(df: pl.DataFrame, distance: int = 0, **kwargs) -> list[tuple]:
    return (
        merge_depth(df.lazy(), distance=distance, **kwargs)
        .collect()
        .sort('chrom', 'pos', 'end')
        .rows()
    )


def _expected_rows(
        df: pl.DataFrame, distance: int = 0, with_depth: bool = True,
) -> list[tuple]:
    regions = expected_merge_depth(df, distance=distance)
    if with_depth:
        return [(r.chrom, r.pos, r.end, r.max_depth) for r in regions]
    return [(r.chrom, r.pos, r.end) for r in regions]


class TestMergeDepthBasics:
    """Hand-traced expected results for small inputs."""

    def test_empty(self) -> None:
        df = pl.DataFrame(
            {'chrom': [], 'pos': [], 'end': []},
            schema={'chrom': pl.String, 'pos': pl.Int64, 'end': pl.Int64},
        )
        assert _run(df) == []

    def test_single_record(self) -> None:
        df = _df([('chr1', 100, 200)])
        assert _run(df) == [('chr1', 100, 200, 1)]

    def test_disjoint_records(self) -> None:
        df = _df([('chr1', 100, 200), ('chr1', 300, 400)])
        assert _run(df) == [
            ('chr1', 100, 200, 1),
            ('chr1', 300, 400, 1),
        ]

    def test_overlapping_records_merge(self) -> None:
        df = _df([('chr1', 100, 200), ('chr1', 150, 250)])
        assert _run(df) == [('chr1', 100, 250, 2)]

    def test_nested_records(self) -> None:
        df = _df([('chr1', 100, 500), ('chr1', 200, 300)])
        assert _run(df) == [('chr1', 100, 500, 2)]

    def test_three_stacked(self) -> None:
        df = _df([
            ('chr1', 100, 400),
            ('chr1', 150, 350),
            ('chr1', 200, 300),
        ])
        assert _run(df) == [('chr1', 100, 400, 3)]

    def test_adjacent_records_merge(self) -> None:
        """Touching records (a.end == b.pos) merge under the bed module's spec."""
        df = _df([('chr1', 100, 200), ('chr1', 200, 300)])
        assert _run(df) == [('chr1', 100, 300, 2)]

    def test_distance_pads_merge(self) -> None:
        """Records within ``distance`` of each other merge.

        With distance=50, the padded ends become 250 and 350, which makes
        them adjacent at loc=250 and depth peaks at 2 there.
        """
        df = _df([('chr1', 100, 200), ('chr1', 250, 300)])
        assert _run(df, distance=50) == [('chr1', 100, 300, 2)]

    def test_distance_below_gap_keeps_separate(self) -> None:
        df = _df([('chr1', 100, 200), ('chr1', 250, 300)])
        assert _run(df, distance=49) == [
            ('chr1', 100, 200, 1),
            ('chr1', 250, 300, 1),
        ]

    def test_multi_chrom(self) -> None:
        df = _df([
            ('chr1', 100, 200),
            ('chr2', 100, 200),
            ('chr1', 150, 250),
        ])
        assert _run(df) == [
            ('chr1', 100, 250, 2),
            ('chr2', 100, 200, 1),
        ]


class TestMergeDepthOptions:
    """Optional behaviours and error paths."""

    def test_no_max_depth_column(self) -> None:
        df = _df([('chr1', 100, 200), ('chr1', 150, 250)])
        rows = _run(df, col_max_depth=None)
        assert rows == [('chr1', 100, 250)]

    def test_max_depth_collision_with_coord_raises(self) -> None:
        df = _df([('chr1', 100, 200)])
        with pytest.raises(ValueError):
            merge_depth(
                df.lazy(),
                col_names=('chrom', 'pos', 'end'),
                col_max_depth='end',
            ).collect()

    def test_custom_col_names(self) -> None:
        df = pl.DataFrame(
            {'qry_id': ['q1', 'q1'], 'qry_pos': [100, 150], 'qry_end': [200, 250]},
            schema={
                'qry_id': pl.String,
                'qry_pos': pl.Int64,
                'qry_end': pl.Int64,
            },
        )
        rows = (
            merge_depth(df.lazy(), col_names='qry')
            .collect()
            .sort('qry_id', 'qry_pos', 'qry_end')
            .rows()
        )
        assert rows == [('q1', 100, 250, 2)]


@pytest.mark.usefixtures('df_a', 'df_b')
class TestMergeDepthAgainstOracle:
    """Oracle equivalence over every discovered dataset, both sides."""

    def test_a_default_distance(self, df_a: pl.DataFrame) -> None:
        actual = _run(df_a, distance=0)
        expected = _expected_rows(df_a, distance=0)
        assert actual == expected

    def test_b_default_distance(self, df_b: pl.DataFrame) -> None:
        actual = _run(df_b, distance=0)
        expected = _expected_rows(df_b, distance=0)
        assert actual == expected

    @pytest.mark.parametrize('distance', [1, 100, 10_000])
    def test_a_with_distance(self, df_a: pl.DataFrame, distance: int) -> None:
        actual = _run(df_a, distance=distance)
        expected = _expected_rows(df_a, distance=distance)
        assert actual == expected

    @pytest.mark.parametrize('distance', [1, 100, 10_000])
    def test_b_with_distance(self, df_b: pl.DataFrame, distance: int) -> None:
        actual = _run(df_b, distance=distance)
        expected = _expected_rows(df_b, distance=distance)
        assert actual == expected
