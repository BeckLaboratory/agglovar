"""Unit tests for ``agglovar.bed.join``.

Tests run all three implementations (``pairwise_join``,
``pairwise_join_iter``, ``pairwise_join_tree``) against the brute-force
oracle in :mod:`tests.bed.oracle`. Each is parametrized over the discovered
datasets and over both directions (A→B and B→A).
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import polars as pl
import pytest

from agglovar.bed.join import (
    pairwise_join,
    pairwise_join_iter,
    pairwise_join_tree,
)

from tests.bed.oracle import (
    JoinRow,
    expected_join,
    expected_join_pairs,
)


def _join_via_iter(df_a, df_b, **kwargs) -> pl.DataFrame:
    return pl.concat(list(pairwise_join_iter(df_a, df_b, **kwargs))).collect()


def _join_via_collect(df_a, df_b, **kwargs) -> pl.DataFrame:
    return pairwise_join(df_a, df_b, **kwargs).collect()


def _join_via_tree(df_a, df_b, **kwargs) -> pl.DataFrame:
    return pairwise_join_tree(df_a, df_b, **kwargs)


ALL_IMPLS: tuple[tuple[str, Callable], ...] = (
    ('pairwise_join', _join_via_collect),
    ('pairwise_join_iter', _join_via_iter),
    ('pairwise_join_tree', _join_via_tree),
)

ALL_IMPL_PARAMS = [pytest.param(name, fn, id=name) for name, fn in ALL_IMPLS]


def _df(rows: list[tuple[str, int, int]]) -> pl.DataFrame:
    return pl.DataFrame(
        rows,
        schema={'chrom': pl.String, 'pos': pl.Int64, 'end': pl.Int64},
        orient='row',
    )


def _pairs(df_join: pl.DataFrame) -> set[tuple[int, int]]:
    return set(df_join.select('index_a', 'index_b').iter_rows())


def _join_to_rows(df_join: pl.DataFrame) -> list[JoinRow]:
    rows = (
        df_join
        .select('index_a', 'index_b', 'chrom', 'pos', 'end', 'distance')
        .sort('index_a', 'index_b')
        .iter_rows()
    )
    return [JoinRow(*r) for r in rows]


# --- direct edge-case tests --------------------------------------------------

class TestPairwiseJoinSchema:
    """Output schema is stable across implementations."""

    EXPECTED_COLS = ('index_a', 'index_b', 'chrom', 'pos', 'end', 'distance')

    @pytest.mark.parametrize('name,fn', ALL_IMPL_PARAMS)
    def test_columns_present(self, name: str, fn: Callable) -> None:
        df_a = _df([('chr1', 100, 200)])
        df_b = _df([('chr1', 150, 250)])
        out = fn(df_a, df_b)
        for col in self.EXPECTED_COLS:
            assert col in out.columns, f'{name} missing column {col}'

    @pytest.mark.parametrize('name,fn', ALL_IMPL_PARAMS)
    def test_empty_inputs_return_empty_with_schema(
            self, name: str, fn: Callable,
    ) -> None:
        df_a = _df([])
        df_b = _df([])
        out = fn(df_a, df_b)
        assert out.height == 0
        for col in self.EXPECTED_COLS:
            assert col in out.columns


class TestPairwiseJoinEdgeCases:
    """Hand-crafted minimal cases."""

    @pytest.mark.parametrize('name,fn', ALL_IMPL_PARAMS)
    def test_no_overlap(self, name: str, fn: Callable) -> None:
        df_a = _df([('chr1', 100, 200)])
        df_b = _df([('chr1', 300, 400)])
        assert fn(df_a, df_b).height == 0

    @pytest.mark.parametrize('name,fn', ALL_IMPL_PARAMS)
    def test_partial_overlap(self, name: str, fn: Callable) -> None:
        df_a = _df([('chr1', 100, 200)])
        df_b = _df([('chr1', 150, 250)])
        out = fn(df_a, df_b)
        assert _pairs(out) == {(0, 0)}

    @pytest.mark.parametrize('name,fn', ALL_IMPL_PARAMS)
    def test_disjoint_chromosomes(self, name: str, fn: Callable) -> None:
        df_a = _df([('chr1', 100, 200)])
        df_b = _df([('chr2', 100, 200)])
        assert fn(df_a, df_b).height == 0

    @pytest.mark.parametrize('name,fn', ALL_IMPL_PARAMS)
    def test_a_empty_with_b_present(self, name: str, fn: Callable) -> None:
        df_a = _df([])
        df_b = _df([('chr1', 100, 200)])
        assert fn(df_a, df_b).height == 0

    @pytest.mark.parametrize('name,fn', ALL_IMPL_PARAMS)
    def test_b_empty_with_a_present(self, name: str, fn: Callable) -> None:
        df_a = _df([('chr1', 100, 200)])
        df_b = _df([])
        assert fn(df_a, df_b).height == 0

    @pytest.mark.parametrize('name,fn', ALL_IMPL_PARAMS)
    def test_touching_boundary_included(
            self, name: str, fn: Callable,
    ) -> None:
        """Touching at ``distance=0`` is included; output ``distance`` is 0."""
        df_a = _df([('chr1', 100, 200)])
        df_b = _df([('chr1', 200, 300)])
        out = fn(df_a, df_b)
        assert _pairs(out) == {(0, 0)}
        assert out['distance'].to_list() == [0]

    @pytest.mark.parametrize('name,fn', ALL_IMPL_PARAMS)
    def test_nested_overlap(self, name: str, fn: Callable) -> None:
        df_a = _df([('chr1', 100, 500)])
        df_b = _df([('chr1', 200, 300)])
        out = fn(df_a, df_b)
        assert _pairs(out) == {(0, 0)}
        # intersection coords are the inner interval
        assert out['pos'].to_list() == [200]
        assert out['end'].to_list() == [300]
        # distance is negative for overlap
        assert out['distance'].to_list()[0] == 200 - 300

    @pytest.mark.parametrize('name,fn', ALL_IMPL_PARAMS)
    def test_duplicates_yield_full_cross(self, name: str, fn: Callable) -> None:
        df_a = _df([('chr1', 100, 200), ('chr1', 100, 200)])
        df_b = _df([('chr1', 150, 250), ('chr1', 150, 250)])
        out = fn(df_a, df_b)
        assert _pairs(out) == {(0, 0), (0, 1), (1, 0), (1, 1)}


class TestPairwiseJoinDistance:
    """Distance parameter behaviour."""

    @pytest.mark.parametrize('name,fn', ALL_IMPL_PARAMS)
    def test_distance_strictly_below_gap_excludes(
            self, name: str, fn: Callable,
    ) -> None:
        df_a = _df([('chr1', 100, 200)])
        df_b = _df([('chr1', 250, 350)])  # gap = 50
        assert fn(df_a, df_b, distance=49).height == 0

    @pytest.mark.parametrize('name,fn', ALL_IMPL_PARAMS)
    def test_distance_above_gap_includes(
            self, name: str, fn: Callable,
    ) -> None:
        df_a = _df([('chr1', 100, 200)])
        df_b = _df([('chr1', 250, 350)])  # gap = 50
        assert _pairs(fn(df_a, df_b, distance=51)) == {(0, 0)}

    @pytest.mark.parametrize('name,fn', ALL_IMPL_PARAMS)
    def test_distance_exactly_at_gap_includes(
            self, name: str, fn: Callable,
    ) -> None:
        """Inclusive boundary: distance == gap is included."""
        df_a = _df([('chr1', 100, 200)])
        df_b = _df([('chr1', 250, 350)])  # gap = 50
        assert _pairs(fn(df_a, df_b, distance=50)) == {(0, 0)}

    @pytest.mark.parametrize('name,fn', ALL_IMPL_PARAMS)
    def test_negative_distance_below_overlap_excludes(
            self, name: str, fn: Callable,
    ) -> None:
        df_a = _df([('chr1', 100, 200)])
        df_b = _df([('chr1', 150, 250)])  # 50 bp overlap
        assert fn(df_a, df_b, distance=-51).height == 0

    @pytest.mark.parametrize('name,fn', ALL_IMPL_PARAMS)
    def test_negative_distance_within_overlap_includes(
            self, name: str, fn: Callable,
    ) -> None:
        df_a = _df([('chr1', 100, 200)])
        df_b = _df([('chr1', 150, 250)])  # 50 bp overlap
        assert _pairs(fn(df_a, df_b, distance=-49)) == {(0, 0)}

    @pytest.mark.parametrize('name,fn', ALL_IMPL_PARAMS)
    def test_negative_distance_exactly_at_overlap_includes(
            self, name: str, fn: Callable,
    ) -> None:
        """Inclusive boundary: distance == -|overlap| is included."""
        df_a = _df([('chr1', 100, 200)])
        df_b = _df([('chr1', 150, 250)])  # 50 bp overlap
        assert _pairs(fn(df_a, df_b, distance=-50)) == {(0, 0)}


class TestPairwiseJoinChunkSize:
    """Cross-chunk-size consistency."""

    @pytest.mark.parametrize('chunk_size', [1, 2, 5, 17, 1_000, 10_000])
    def test_iter_chunk_invariance(
            self, df_a: pl.DataFrame, df_b: pl.DataFrame, chunk_size: int,
    ) -> None:
        small = _pairs(_join_via_iter(df_a, df_b, chunk_size=chunk_size))
        big = _pairs(_join_via_iter(df_a, df_b, chunk_size=1_000_000))
        assert small == big

    @pytest.mark.parametrize('chunk_size', [1, 2, 5, 17, 1_000])
    def test_join_chunk_invariance(
            self, df_a: pl.DataFrame, df_b: pl.DataFrame, chunk_size: int,
    ) -> None:
        small = _pairs(_join_via_collect(df_a, df_b, chunk_size=chunk_size))
        big = _pairs(_join_via_collect(df_a, df_b, chunk_size=1_000_000))
        assert small == big

    def test_join_iter_match(
            self, df_a: pl.DataFrame, df_b: pl.DataFrame,
    ) -> None:
        """``pairwise_join`` and ``pairwise_join_iter`` produce the same pairs."""
        a = _pairs(_join_via_collect(df_a, df_b))
        b = _pairs(_join_via_iter(df_a, df_b))
        assert a == b


class TestPairwiseJoinAgainstOracle:
    """Equivalence with the brute-force oracle across all three implementations."""

    @pytest.mark.parametrize('side', ['a_b', 'b_a'])
    @pytest.mark.parametrize('name,fn', ALL_IMPL_PARAMS)
    def test_oracle_pairs(
            self,
            df_a: pl.DataFrame,
            df_b: pl.DataFrame,
            side: str,
            name: str,
            fn: Callable,
    ) -> None:
        a, b = (df_a, df_b) if side == 'a_b' else (df_b, df_a)
        actual = _pairs(fn(a, b))
        expected = expected_join_pairs(a, b)
        assert actual == expected

    @pytest.mark.parametrize('side', ['a_b', 'b_a'])
    @pytest.mark.parametrize('name,fn', ALL_IMPL_PARAMS)
    def test_oracle_metadata(
            self,
            df_a: pl.DataFrame,
            df_b: pl.DataFrame,
            side: str,
            name: str,
            fn: Callable,
    ) -> None:
        a, b = (df_a, df_b) if side == 'a_b' else (df_b, df_a)
        actual = sorted(_join_to_rows(fn(a, b)), key=lambda r: (r.index_a, r.index_b))
        expected = expected_join(a, b)
        assert actual == expected


class TestPairwiseJoinCustomColumns:
    """Custom ``CoordCol`` specifications."""

    def test_qry_columns(self) -> None:
        df = pl.DataFrame(
            {'qry_id': ['q1'], 'qry_pos': [100], 'qry_end': [200]},
            schema={'qry_id': pl.String, 'qry_pos': pl.Int64, 'qry_end': pl.Int64},
        )
        out = pairwise_join(
            df, df,
            col_names_a='qry', col_names_b='qry',
        ).collect()
        assert out.height == 1
        assert _pairs(out) == {(0, 0)}

    def test_mismatched_column_names(self) -> None:
        df_a = pl.DataFrame(
            {'qry_id': ['q1'], 'qry_pos': [100], 'qry_end': [200]},
            schema={'qry_id': pl.String, 'qry_pos': pl.Int64, 'qry_end': pl.Int64},
        )
        df_b = pl.DataFrame(
            {'chrom': ['q1'], 'pos': [150], 'end': [250]},
            schema={'chrom': pl.String, 'pos': pl.Int64, 'end': pl.Int64},
        )
        out = pairwise_join(
            df_a, df_b, col_names_a='qry', col_names_b='ref',
        ).collect()
        assert _pairs(out) == {(0, 0)}


class TestPairwiseJoinTempDir:
    """``temp_dir`` materialisation policy.

    All three modes (``False`` -> in-memory, ``True`` -> system temp dir,
    explicit path) must produce identical results.
    """

    @pytest.fixture
    def expected_pairs(self, df_a: pl.DataFrame, df_b: pl.DataFrame) -> set[tuple[int, int]]:
        return _pairs(_join_via_collect(df_a, df_b, temp_dir=False))

    def test_false_in_memory(
            self, df_a: pl.DataFrame, df_b: pl.DataFrame,
            expected_pairs: set[tuple[int, int]],
    ) -> None:
        assert _pairs(pairwise_join(df_a, df_b, temp_dir=False).collect()) == expected_pairs

    def test_true_system_temp(
            self, df_a: pl.DataFrame, df_b: pl.DataFrame,
            expected_pairs: set[tuple[int, int]],
    ) -> None:
        assert _pairs(pairwise_join(df_a, df_b, temp_dir=True).collect()) == expected_pairs

    def test_explicit_path(
            self, df_a: pl.DataFrame, df_b: pl.DataFrame,
            expected_pairs: set[tuple[int, int]],
            tmp_path: Path,
    ) -> None:
        assert _pairs(pairwise_join(df_a, df_b, temp_dir=tmp_path).collect()) == expected_pairs

    def test_explicit_path_str(
            self, df_a: pl.DataFrame, df_b: pl.DataFrame,
            expected_pairs: set[tuple[int, int]],
            tmp_path: Path,
    ) -> None:
        assert _pairs(pairwise_join(df_a, df_b, temp_dir=str(tmp_path)).collect()) == expected_pairs

    def test_iter_temp_dir_modes_match(
            self, df_a: pl.DataFrame, df_b: pl.DataFrame,
            expected_pairs: set[tuple[int, int]],
            tmp_path: Path,
    ) -> None:
        for mode in (False, True, tmp_path):
            assert _pairs(_join_via_iter(df_a, df_b, temp_dir=mode)) == expected_pairs

    def test_tree_temp_dir_modes_match(
            self, df_a: pl.DataFrame, df_b: pl.DataFrame,
            tmp_path: Path,
    ) -> None:
        a = df_a.filter(pl.col('end') > pl.col('pos'))  # tree skipped earlier on zero-len; now ok
        b = df_b.filter(pl.col('end') > pl.col('pos'))
        baseline = _pairs(pairwise_join_tree(a, b, temp_dir=False))
        assert _pairs(pairwise_join_tree(a, b, temp_dir=True)) == baseline
        assert _pairs(pairwise_join_tree(a, b, temp_dir=tmp_path)) == baseline

    def test_explicit_path_does_not_leave_files(
            self, df_a: pl.DataFrame, df_b: pl.DataFrame,
            tmp_path: Path,
    ) -> None:
        """Temp parquet files must be cleaned up on success."""
        pairwise_join(df_a, df_b, temp_dir=tmp_path).collect()
        assert list(tmp_path.iterdir()) == []


class TestPairwiseJoinErrors:
    """Error paths."""

    def test_invalid_chunk_size_raises(self) -> None:
        df = _df([('chr1', 100, 200)])
        with pytest.raises(ValueError):
            pairwise_join(df, df, chunk_size=0).collect()
        with pytest.raises(ValueError):
            pairwise_join(df, df, chunk_size=-5).collect()

    def test_bad_col_names_a_raises(self) -> None:
        df = _df([('chr1', 100, 200)])
        with pytest.raises(ValueError):
            pairwise_join(df, df, col_names_a=('only_two', 'columns')).collect()
