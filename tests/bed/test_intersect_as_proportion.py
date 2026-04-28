"""Unit tests for ``agglovar.bed.intersect.as_proportion``."""

from __future__ import annotations

import math

import polars as pl
import pytest

from agglovar.bed.intersect import as_proportion


def _df(rows: list[tuple[str, int, int]]) -> pl.LazyFrame:
    return pl.DataFrame(
        rows,
        schema={'chrom': pl.String, 'pos': pl.Int64, 'end': pl.Int64},
        orient='row',
    ).lazy()


def _run(df_a, df_b, name: str = 'prop', **kwargs) -> list[float]:
    if isinstance(df_a, pl.DataFrame):
        df_a = df_a.lazy()
    if isinstance(df_b, pl.DataFrame):
        df_b = df_b.lazy()
    out = as_proportion(df_a, df_b, name=name, **kwargs).collect()
    return out.sort('_index').get_column(name).to_list()


class TestAsProportionBasics:
    """Hand-crafted minimal cases."""

    def test_full_overlap(self) -> None:
        df_a = _df([('chr1', 1_000, 2_000)])
        df_b = _df([('chr1', 1_000, 2_000)])
        assert _run(df_a, df_b) == [1.0]

    def test_no_overlap_proportion_zero(self) -> None:
        df_a = _df([('chr1', 1_000, 2_000)])
        df_b = _df([('chr1', 3_000, 4_000)])
        assert _run(df_a, df_b) == [0.0]

    def test_half_overlap(self) -> None:
        df_a = _df([('chr1', 1_000, 2_000)])
        df_b = _df([('chr1', 1_500, 2_500)])
        assert _run(df_a, df_b) == [0.5]

    def test_disjoint_chromosomes_zero(self) -> None:
        df_a = _df([('chr1', 1_000, 2_000)])
        df_b = _df([('chr2', 1_000, 2_000)])
        assert _run(df_a, df_b) == [0.0]

    def test_b_contains_a_proportion_one(self) -> None:
        df_a = _df([('chr1', 1_500, 1_600)])
        df_b = _df([('chr1', 1_000, 2_000)])
        assert _run(df_a, df_b) == [1.0]

    def test_a_contains_b_partial(self) -> None:
        df_a = _df([('chr1', 1_000, 2_000)])
        df_b = _df([('chr1', 1_200, 1_400)])  # 200/1000 = 0.2
        assert _run(df_a, df_b) == [0.2]

    def test_multiple_disjoint_b_intervals(self) -> None:
        df_a = _df([('chr1', 1_000, 2_000)])
        df_b = _df([
            ('chr1', 1_000, 1_300),  # 300
            ('chr1', 1_500, 1_700),  # 200
            ('chr1', 1_800, 2_200),  # 200 inside [1000, 2000)
        ])
        assert _run(df_a, df_b) == [pytest.approx(0.7)]

    def test_overlapping_b_intervals_merged(self) -> None:
        """B intervals overlap each other — coverage must not double-count."""
        df_a = _df([('chr1', 1_000, 2_000)])
        df_b = _df([
            ('chr1', 1_000, 1_500),
            ('chr1', 1_300, 1_800),  # merges with above to [1000, 1800)
            ('chr1', 1_900, 2_100),  # → covers [1900, 2000)
        ])
        # covered: 800 + 100 = 900 / 1000
        assert _run(df_a, df_b) == [pytest.approx(0.9)]

    def test_a_empty_returns_empty(self) -> None:
        df_a = _df([])
        df_b = _df([('chr1', 100, 200)])
        assert _run(df_a, df_b) == []

    def test_b_empty_all_zero(self) -> None:
        df_a = _df([('chr1', 100, 200), ('chr1', 1_000, 1_100)])
        df_b = _df([])
        assert _run(df_a, df_b) == [0.0, 0.0]


class TestAsProportionZeroLength:
    """Zero-length intervals.

    Zero-length A produces NaN (0/0). Zero-length B contributes 0 covered bp
    (zero-length intervals do not contribute to coverage).
    """

    def test_zero_length_a_yields_nan(self) -> None:
        df_a = _df([('chr1', 1_000, 1_000)])
        df_b = _df([('chr1', 1_000, 1_000)])
        result = _run(df_a, df_b)
        assert len(result) == 1
        assert math.isnan(result[0])

    def test_zero_length_b_does_not_cover(self) -> None:
        df_a = _df([('chr1', 1_000, 2_000)])
        df_b = _df([('chr1', 1_500, 1_500)])
        assert _run(df_a, df_b) == [0.0]


class TestAsProportionCustomColumns:
    """Custom ``CoordCol`` specifications."""

    def test_qry_columns(self) -> None:
        df = pl.DataFrame(
            {
                'qry_id': ['q1'],
                'qry_pos': [1_000],
                'qry_end': [2_000],
            },
            schema={
                'qry_id': pl.String,
                'qry_pos': pl.Int64,
                'qry_end': pl.Int64,
            },
        ).lazy()

        df_b = pl.DataFrame(
            {
                'qry_id': ['q1'],
                'qry_pos': [1_500],
                'qry_end': [2_500],
            },
            schema={
                'qry_id': pl.String,
                'qry_pos': pl.Int64,
                'qry_end': pl.Int64,
            },
        ).lazy()

        result = _run(df, df_b, col_names_a='qry', col_names_b='qry')
        assert result == [pytest.approx(0.5)]


class TestAsProportionErrors:
    """Required-input errors."""

    def test_name_none_raises(self) -> None:
        df = _df([('chr1', 100, 200)])
        with pytest.raises(ValueError):
            as_proportion(df, df, name=None).collect()

    def test_name_empty_raises(self) -> None:
        df = _df([('chr1', 100, 200)])
        with pytest.raises(ValueError):
            as_proportion(df, df, name='').collect()

    def test_name_whitespace_raises(self) -> None:
        df = _df([('chr1', 100, 200)])
        with pytest.raises(ValueError):
            as_proportion(df, df, name='   ').collect()


class TestAsProportionNullCoords:
    """Rows with null pos/end are preserved with a null proportion."""

    def test_null_pos_preserved_as_null(self) -> None:
        df_a = pl.DataFrame(
            {
                'chrom': ['chr1', 'chr1'],
                'pos': [None, 1_000],
                'end': [200, 2_000],
            },
            schema={
                'chrom': pl.String,
                'pos': pl.Int64,
                'end': pl.Int64,
            },
        ).lazy()
        df_b = _df([('chr1', 1_500, 2_500)])

        result = (
            as_proportion(df_a, df_b, name='prop')
            .collect()
            .sort('_index')
        )
        assert result.height == 2
        assert result.get_column('prop').to_list() == [None, 0.5]

    def test_null_end_preserved_as_null(self) -> None:
        df_a = pl.DataFrame(
            {
                'chrom': ['chr1', 'chr1'],
                'pos': [100, 1_000],
                'end': [None, 2_000],
            },
            schema={
                'chrom': pl.String,
                'pos': pl.Int64,
                'end': pl.Int64,
            },
        ).lazy()
        df_b = _df([('chr1', 1_500, 2_500)])

        result = (
            as_proportion(df_a, df_b, name='prop')
            .collect()
            .sort('_index')
        )
        assert result.height == 2
        assert result.get_column('prop').to_list() == [None, 0.5]


class TestAsProportionAgainstOracle:
    """Oracle equivalence over every dataset, both directions."""

    @staticmethod
    def _expected_props(df_a: pl.DataFrame, df_b: pl.DataFrame) -> list[float]:
        from tests.bed.oracle import expected_as_proportion
        return list(expected_as_proportion(df_a, df_b))

    def test_a_vs_b(self, df_a: pl.DataFrame, df_b: pl.DataFrame) -> None:
        actual = _run(df_a, df_b)
        expected = self._expected_props(df_a, df_b)
        assert actual == pytest.approx(expected, abs=1e-9, nan_ok=True)

    def test_b_vs_a(self, df_a: pl.DataFrame, df_b: pl.DataFrame) -> None:
        actual = _run(df_b, df_a)
        expected = self._expected_props(df_b, df_a)
        assert actual == pytest.approx(expected, abs=1e-9, nan_ok=True)
