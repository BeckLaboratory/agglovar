"""Unit tests for ``agglovar.bed.col``."""

from __future__ import annotations

import polars as pl
import pytest

from agglovar.bed.col import (
    CoordCol,
    CoordColExpr,
    get_coord_cols,
    make_unique_col,
    standardize,
)


class TestGetCoordCols:
    """Resolution of coordinate column specs to ``CoordCol``."""

    def test_default_returns_ref_columns(self) -> None:
        assert get_coord_cols() == CoordCol('chrom', 'pos', 'end')

    def test_none_returns_ref_columns(self) -> None:
        assert get_coord_cols(None) == CoordCol('chrom', 'pos', 'end')

    def test_ref_keyword(self) -> None:
        assert get_coord_cols('ref') == CoordCol('chrom', 'pos', 'end')

    def test_qry_keyword(self) -> None:
        assert get_coord_cols('qry') == CoordCol('qry_id', 'qry_pos', 'qry_end')

    def test_passthrough_coordcol(self) -> None:
        cc = CoordCol('a', 'b', 'c')
        assert get_coord_cols(cc) is cc

    def test_tuple_input(self) -> None:
        assert get_coord_cols(('a', 'b', 'c')) == CoordCol('a', 'b', 'c')

    def test_list_input(self) -> None:
        assert get_coord_cols(['x', 'y', 'z']) == CoordCol('x', 'y', 'z')

    def test_strips_whitespace(self) -> None:
        assert get_coord_cols((' a ', ' b ', ' c ')) == CoordCol('a', 'b', 'c')

    def test_wrong_length_raises(self) -> None:
        with pytest.raises(ValueError):
            get_coord_cols(('a', 'b'))

    def test_empty_value_raises(self) -> None:
        with pytest.raises(ValueError):
            get_coord_cols(('a', '', 'c'))

    def test_none_value_raises(self) -> None:
        with pytest.raises(ValueError):
            get_coord_cols(('a', None, 'c'))


class TestCoordCol:
    """Iteration, containment, and expression generation."""

    def test_iteration_order(self) -> None:
        cc = CoordCol('chrom', 'pos', 'end')
        assert list(cc) == ['chrom', 'pos', 'end']

    def test_contains(self) -> None:
        cc = CoordCol('chrom', 'pos', 'end')
        assert 'pos' in cc
        assert 'unknown' not in cc

    def test_exprs_no_alias(self) -> None:
        cc = CoordCol('chrom', 'pos', 'end')
        result = cc.exprs()
        assert isinstance(result, CoordColExpr)
        assert result.chrom_name == 'chrom'
        assert result.pos_name == 'pos'
        assert result.end_name == 'end'

    def test_exprs_with_suffix(self) -> None:
        cc = CoordCol('chrom', 'pos', 'end')
        result = cc.exprs(suffix='_a')
        assert result.chrom_name == 'chrom_a'
        assert result.pos_name == 'pos_a'
        assert result.end_name == 'end_a'

    def test_exprs_alias_mapping(self) -> None:
        cc = CoordCol('a', 'b', 'c')
        ref = CoordCol('chrom', 'pos', 'end')
        result = cc.exprs(alias=ref, suffix='_b')

        assert result.chrom_name == 'chrom_b'
        assert result.pos_name == 'pos_b'
        assert result.end_name == 'end_b'

    def test_exprs_against_dataframe(self) -> None:
        cc = CoordCol('chrom', 'pos', 'end')
        df = pl.DataFrame(
            {'chrom': ['chr1'], 'pos': [10], 'end': [20]}
        )

        out = df.select(*cc.exprs(suffix='_a'))
        assert list(out.columns) == ['chrom_a', 'pos_a', 'end_a']


class TestStandardize:
    """Column-name normalization."""

    def test_lowercase(self) -> None:
        assert standardize('Chrom') == 'chrom'

    def test_strip_whitespace(self) -> None:
        assert standardize('  end  ') == 'end'

    def test_collapse_internal_spaces(self) -> None:
        assert standardize('column  name') == 'column_name'

    def test_drop_special_chars(self) -> None:
        assert standardize('col!@#name') == 'colname'

    def test_keep_dot_and_underscore(self) -> None:
        assert standardize('a_b.c') == 'a_b.c'

    def test_empty_after_clean_raises(self) -> None:
        with pytest.raises(ValueError):
            standardize('!!!')


class TestMakeUniqueCol:
    """Conflict resolution for column names."""

    def test_no_conflict_returns_input(self) -> None:
        assert make_unique_col('foo', {'bar'}) == 'foo'

    def test_first_conflict_appends_one(self) -> None:
        assert make_unique_col('foo', {'foo'}) == 'foo_1'

    def test_chained_conflict(self) -> None:
        assert make_unique_col('foo', {'foo', 'foo_1'}) == 'foo_2'

    def test_multiple_containers(self) -> None:
        assert make_unique_col('foo', {'foo'}, {'foo_1'}) == 'foo_2'
