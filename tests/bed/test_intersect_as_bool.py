"""Unit tests for ``agglovar.bed.intersect.as_bool``."""

from __future__ import annotations

import polars as pl
import pytest

from agglovar.bed.intersect import as_bool

from tests.bed.oracle import expected_join_pairs


def _df(rows: list[tuple[str, int, int]]) -> pl.LazyFrame:
    """Build a synthetic LazyFrame for as_bool tests."""
    return pl.DataFrame(
        rows,
        schema={'chrom': pl.String, 'pos': pl.Int64, 'end': pl.Int64},
        orient='row',
    ).lazy()


def _run(df_a, df_b, name: str = 'hit', **kwargs) -> list[bool]:
    if isinstance(df_a, pl.DataFrame):
        df_a = df_a.lazy()
    if isinstance(df_b, pl.DataFrame):
        df_b = df_b.lazy()
    df_out = as_bool(df_a, df_b, name=name, **kwargs).collect()
    return df_out.sort('_index').get_column(name).to_list()


class TestAsBoolBasics:
    """Hand-crafted minimal cases."""

    def test_no_overlap(self) -> None:
        df_a = _df([('chr1', 100, 200)])
        df_b = _df([('chr1', 300, 400)])
        assert _run(df_a, df_b) == [False]

    def test_single_hit(self) -> None:
        df_a = _df([('chr1', 100, 200)])
        df_b = _df([('chr1', 150, 250)])
        assert _run(df_a, df_b) == [True]

    def test_negate_flips_output(self) -> None:
        df_a = _df([('chr1', 100, 200), ('chr1', 1_000, 1_100)])
        df_b = _df([('chr1', 150, 250)])
        assert _run(df_a, df_b, negate=False) == [True, False]
        assert _run(df_a, df_b, negate=True) == [False, True]

    def test_disjoint_chromosomes(self) -> None:
        df_a = _df([('chr1', 100, 200)])
        df_b = _df([('chr2', 100, 200)])
        assert _run(df_a, df_b) == [False]

    def test_a_empty_returns_empty(self) -> None:
        df_a = _df([])
        df_b = _df([('chr1', 100, 200)])
        assert _run(df_a, df_b) == []

    def test_b_empty_all_miss(self) -> None:
        df_a = _df([('chr1', 100, 200), ('chr1', 300, 400)])
        df_b = _df([])
        assert _run(df_a, df_b) == [False, False]

    def test_b_empty_negate_all_hit(self) -> None:
        df_a = _df([('chr1', 100, 200), ('chr1', 300, 400)])
        df_b = _df([])
        assert _run(df_a, df_b, negate=True) == [True, True]

    def test_duplicates_dedup_per_a_row(self) -> None:
        """as_bool reports one boolean per A row; duplicate B hits do not duplicate."""
        df_a = _df([('chr1', 100, 200)])
        df_b = _df([('chr1', 150, 250), ('chr1', 150, 250)])
        assert _run(df_a, df_b) == [True]


class TestAsBoolDataFrameInput:
    """``as_bool`` accepts both ``DataFrame`` and ``LazyFrame`` inputs."""

    def test_eager_dataframe_input(self) -> None:
        df_a = pl.DataFrame(
            {'chrom': ['chr1'], 'pos': [100], 'end': [200]},
            schema={'chrom': pl.String, 'pos': pl.Int64, 'end': pl.Int64},
        )
        df_b = pl.DataFrame(
            {'chrom': ['chr1'], 'pos': [150], 'end': [250]},
            schema={'chrom': pl.String, 'pos': pl.Int64, 'end': pl.Int64},
        )
        out = as_bool(df_a, df_b, name='hit').collect()
        assert out.get_column('hit').to_list() == [True]

    def test_lazy_dataframe_input(self) -> None:
        df_a = _df([('chr1', 100, 200)])
        df_b = _df([('chr1', 150, 250)])
        out = as_bool(df_a, df_b, name='hit').collect()
        assert out.get_column('hit').to_list() == [True]


class TestAsBoolErrors:
    """Required-input errors."""

    def test_name_none_raises(self) -> None:
        df = _df([('chr1', 100, 200)])
        with pytest.raises(ValueError):
            as_bool(df, df, name=None).collect()

    def test_name_empty_raises(self) -> None:
        df = _df([('chr1', 100, 200)])
        with pytest.raises(ValueError):
            as_bool(df, df, name='').collect()

    def test_name_whitespace_raises(self) -> None:
        df = _df([('chr1', 100, 200)])
        with pytest.raises(ValueError):
            as_bool(df, df, name='   ').collect()


class TestAsBoolDistance:
    """``distance`` parameter behaviour."""

    def test_distance_zero_strict_overlap(self) -> None:
        df_a = _df([('chr1', 100, 200)])
        df_b = _df([('chr1', 250, 350)])
        assert _run(df_a, df_b, distance=0) == [False]

    def test_distance_includes_records_within(self) -> None:
        df_a = _df([('chr1', 100, 200)])
        df_b = _df([('chr1', 250, 350)])  # gap = 50
        assert _run(df_a, df_b, distance=50) == [True]
        assert _run(df_a, df_b, distance=49) == [False]

    def test_negative_distance_requires_overlap(self) -> None:
        df_a = _df([('chr1', 100, 200)])
        df_b = _df([('chr1', 150, 250)])  # 50 bp overlap
        assert _run(df_a, df_b, distance=-50) == [True]
        assert _run(df_a, df_b, distance=-51) == [False]


class TestAsBoolCustomColumns:
    """Custom ``CoordCol`` specifications."""

    def test_qry_columns(self) -> None:
        df = pl.DataFrame(
            {'qry_id': ['q1'], 'qry_pos': [100], 'qry_end': [200]},
            schema={'qry_id': pl.String, 'qry_pos': pl.Int64, 'qry_end': pl.Int64},
        ).lazy()
        out = as_bool(df, df, name='hit', col_names_a='qry', col_names_b='qry').collect()
        assert out.get_column('hit').to_list() == [True]


class TestAsBoolAgainstOracle:
    """Oracle equivalence over every dataset, both directions."""

    @staticmethod
    def _expected(df_a: pl.DataFrame, df_b: pl.DataFrame) -> list[bool]:
        pairs = expected_join_pairs(df_a, df_b)
        hits = {idx_a for idx_a, _ in pairs}
        return [i in hits for i in range(df_a.height)]

    def test_a_vs_b(self, df_a: pl.DataFrame, df_b: pl.DataFrame) -> None:
        assert _run(df_a, df_b) == self._expected(df_a, df_b)

    def test_b_vs_a(self, df_a: pl.DataFrame, df_b: pl.DataFrame) -> None:
        assert _run(df_b, df_a) == self._expected(df_b, df_a)

    def test_negate_matches_inverse(self, df_a: pl.DataFrame, df_b: pl.DataFrame) -> None:
        positive = _run(df_a, df_b, negate=False)
        negative = _run(df_a, df_b, negate=True)
        assert negative == [not x for x in positive]
