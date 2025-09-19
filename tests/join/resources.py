"""
Resources for join tests.
"""

import itertools
import numpy as np
import polars as pl
import pytest
from typing import Any

import agglovar

from tests.assets.variant_tables.load import load_variant_table

PARAMS_CROSS = [
    {
        'vartype': ['ins', 'del'],
        'ro_min': [0.0, 0.2, 0.5, 0.9, 1.0],
        'offset_max': None,
        'size_ro_min': None,
        'offset_prop_max': None,
        'match_prop_min': [None, 0.5, 0.8, 1.0],
        'match_ref': None,
        'match_alt': None
    },
    {
        'vartype': ['ins', 'del'],
        'ro_min': None,
        'offset_max': [None, 200, 1000],
        'size_ro_min': [None, 0.8],
        'offset_prop_max': [None, 2.0],
        'match_prop_min': [None, 0.5, 0.8, 1.0],
        'match_ref': None,
        'match_alt': None
    },
    {
        'vartype': 'snv',
        'ro_min': None,
        'offset_max': [None, 5, 200, 1000],
        'size_ro_min': None,
        'offset_prop_max': None,
        'match_prop_min': None,
        'match_ref': [None, True, False],
        'match_alt': [None, True, False]
    }
]
"""Parameters for test cases."""

PARAM_KEYS = [
    'vartype',
    'ro_min',
    'offset_max',
    'size_ro_min',
    'offset_prop_max',
    'match_prop_min',
    'match_ref',
    'match_alt'
]
"""Parameter keys."""

PARAM_KEY_SPEC = ','.join(PARAM_KEYS)
"""String representation of parameter keys."""

def get_params():
    """Generate sets of parameters to test."""

    def set_param(value):
        if value is None or not isinstance(value, (list, tuple)):
            return [value]
        return value

    for param_dict in PARAMS_CROSS:
        param_dict = {
            key: set_param(param_dict[key] if key in param_dict else None)
                for key in PARAM_KEYS
        }

        for val_list in itertools.product(*[
            param_dict[key] for key in PARAM_KEYS
        ]):
            yield val_list

PARAM_TUPLES = list(get_params())
"""Parameter tuples."""

JOIN_SCHEMA = {
    'index_a': pl.UInt32,
    'index_b': pl.UInt32,
    'id_a': pl.String,
    'id_b': pl.String,
    'ro': pl.Float32,
    'offset_dist': pl.Int32,
    'offset_prop': pl.Float32,
    'size_ro': pl.Float32,
    'match_prop': pl.Float32
}
"""Join table schema."""


#
# Fixtures
#

@pytest.fixture(scope='class')
def df_a(
        vartype: str
) -> pl.DataFrame:
    """Get variant table A.

    :param vartype: Variant type.

    :return: Variant table A.
    """

    return load_variant_table(
        vartype, 'a'
    )


@pytest.fixture(scope='class')
def df_b(
        vartype: str
) -> pl.DataFrame:
    """Get variant table B.

    :param vartype: Variant type.

    :return: Variant table B.
    """

    return load_variant_table(
        vartype, 'b'
    )

@pytest.fixture(scope='class')
def df_exp(
        df_a: pl.DataFrame,
        df_b: pl.DataFrame,
        ro_min: float|None,
        offset_max: int|None,
        size_ro_min: float|None,
        offset_prop_max: float|None,
        match_prop_min: float,
        match_ref: bool|None,
        match_alt: bool|None
):
    """Get a table of expected join records.

    :param df_exp_all: A table of all possible join records.
    :param vartype: Variant type.
    :param ro_min: Reciprocal overlap threshold.
    :param offset_max: Maximum offset.
    :param size_ro_min: Minimum size reciprocal overlap.
    :param offset_prop_max: Maximum offset proportion.
    :param match_prop_min: Minimum match proportion.
    :param match_ref: Match reference base.
    :param match_alt: Match alternate base.

    :return: Expected join records.
    """

    df_exp_all = make_expected_join_table(df_a, df_b, match_ref, match_alt)

    filters = list()
    with_exprs = list()

    if ro_min is not None:
        filters.append(
            pl.col('ro') >= ro_min
        )

    if offset_max is not None:
        filters.append(
            pl.col('offset_dist') <= offset_max
        )

    if size_ro_min is not None:
        filters.append(
            pl.col('size_ro') >= size_ro_min
        )

    if offset_prop_max is not None:
        filters.append(
            pl.col('offset_prop') <= offset_prop_max
        )

    if match_prop_min is not None:
        filters.append(
            pl.col('match_prop') >= match_prop_min
        )
    else:
        with_exprs.append(
            pl.lit(None).cast(pl.Float32).alias('match_prop')
        )

    return (
        df_exp_all
        .filter(*filters)
        .with_columns(*with_exprs)
    )

@pytest.fixture(scope='class')
def df_join(
        df_a: pl.DataFrame,
        df_b: pl.DataFrame,
        ro_min: float|None,
        offset_max: int|None,
        size_ro_min: float|None,
        offset_prop_max: float|None,
        match_prop_min: float|None,
        match_ref: bool|None,
        match_alt: bool|None
) -> pl.DataFrame:
    """Get a table of joined records.

    :param df_a: DataFrame A.
    :param df_b: DataFrame B.
    :param vartype: Variant type.
    :param ro_min: Reciprocal overlap threshold.
    :param offset_max: Maximum offset.
    :param size_ro_min: Minimum size reciprocal overlap.
    :param offset_prop_max: Maximum offset proportion.
    :param match_prop_min: Minimum match proportion.
    :param match_ref: Match reference base.
    :param match_alt: Match alternate base.

    :return: Actual join records.
    """

    return agglovar.join.pair.join(
        **{
            key: val for key, val in locals().items() if val is not None and key not in {'vartype'}
        }

        # ro_min=ro_min,
        # offset_max=offset_max,
        # size_ro_min=size_ro_min,
        # offset_prop_max=offset_prop_max,
        # match_prop_min=match_prop_min,
        # match_ref=match_ref,
        # match_alt=match_alt
    ).collect()


#
# Helper functions
#

def subset_miss(
        df_join: pl.DataFrame,
        df_exp: pl.DataFrame
) -> pl.DataFrame:
    """Get expected records missing from df_join.

    :param df_join: Actual join.
    :param df_exp: Expected join.

    :return: A table of missing intersect records.
    """

    return (
        df_exp
        .join(
            df_join,
            on=['index_a', 'index_b'],
            how='anti',
            nulls_equal=True,
            suffix='_join'
        )
    )


def subset_extra(
        df_join: pl.DataFrame,
        df_exp: pl.DataFrame
) -> pl.DataFrame:
    """Get extraneous record in df_join that are not in df_exp.

    :param df_join: Actual join.
    :param df_exp: Expected join.

    :return: A table of extraneous intersect records.
    """

    return (
        df_join
        .join(
            df_exp,
            on=['index_a', 'index_b'],
            how='anti',
            nulls_equal=True,
            suffix='_exp'
        )
    )


def row_to_dict(
        df: pl.DataFrame,
        index: int,
        approx: bool=False,
        rel: float=1e-2
) -> dict[str, Any]:
    """Convert a row to a dictionary for assert comparisons between dictionaries.

    :param df: DataFrame.
    :param index: Row index.
    :param approx: If True, wrap float values in pytest.approx().
    :param rel: Relative tolerance for pytest.approx().

    :return: A dictionary for this row.
    """

    row_dict = df.row(index, named=True)

    if approx:
        row_dict = {
            key: (
                pytest.approx(value, rel=rel) if df.schema[key].is_float() else value
            ) for key, value in row_dict.items()
        }

    return row_dict


def make_expected_join_table(
        df_a: pl.DataFrame,
        df_b: pl.DataFrame,
        match_ref: bool|None,
        match_alt: bool|None
) -> pl.DataFrame:
    """Construct expected join tables independently of the join function.

    :param df_a: DataFrame A.
    :param df_b: DataFrame B.
    :param match_ref: Match reference base.
    :param match_alt: Match alternate base.

    :return: Expected join table.
    """

    match_score_model = agglovar.seqmatch.MatchScoreModel()

    row_list = list()

    if 'varlen' not in df_a.columns:
        df_a = (
            df_a
            .with_columns(
                (pl.col('end') - pl.col('pos')).alias('varlen').cast(agglovar.schema.VARIANT['varlen'])
            )
        )

    if 'varlen' not in df_b.columns:
        df_b = (
            df_b
            .with_columns(
                (pl.col('end') - pl.col('pos')).alias('varlen').cast(agglovar.schema.VARIANT['varlen'])
            )
        )

    has_seq = 'seq' in df_a.columns and 'seq' in df_b.columns

    for i_a in range(df_a.height):
        for i_b in range(df_b.height):

            row_a = df_a.row(i_a, named=True)
            row_b = df_b.row(i_b, named=True)

            if row_a['chrom'] != row_b['chrom']:
                continue

            if match_ref and not row_a['ref'].upper() == row_b['ref'].upper():
                continue

            if match_alt and not row_a['alt'].upper() == row_b['alt'].upper():
                continue

            row_list.append({
                'index_a': i_a,
                'index_b': i_b,
                'id_a': row_a['id'],
                'id_b': row_b['id'],
                'ro': get_ro(row_a, row_b),
                'offset_dist': (offset_dist := get_offset_dist(row_a, row_b)),
                'offset_prop': offset_dist / np.min([row_a['varlen'], row_b['varlen']]),
                'size_ro': np.min([row_a['varlen'], row_b['varlen']]) / np.max([row_a['varlen'], row_b['varlen']]),
                'match_prop': match_score_model.match_prop(row_a['seq'], row_b['seq']) if has_seq else None
            })

    return pl.DataFrame(
        row_list,
        schema=JOIN_SCHEMA
    )

def get_ro(
        row_a: dict,
        row_b: dict
) -> float:
    """Compute expected reciprocal overlap. Assumes "chrom" was already checked.

    :param row_a: Row A.
    :param row_b: Row B.

    :return: RO.
    """

    pos_a = row_a['pos']
    pos_b = row_b['pos']

    if row_a['vartype'] == 'INS':
        end_a = row_a['pos'] + row_a['varlen']
        end_b = row_b['pos'] + row_b['varlen']
    else:
        end_a = row_a['end']
        end_b = row_b['end']

    if (pos_b >= end_a) or (pos_a >= end_b):
        return 0.0

    return (
        np.min([end_a, end_b]) - np.max([pos_a, pos_b])
    ) / np.max([row_a['varlen'], row_b['varlen']])

def get_offset_dist(
        row_a: dict,
        row_b: dict
) -> int:
    """Compute expected offset distance. Assumes "chrom" was already checked.

    :param row_a: Row A.
    :param row_b: Row B.

    :return: Offset distance.
    """

    return np.max([
        np.abs(row_a['pos'] - row_b['pos']),
        np.abs(row_a['end'] - row_b['end'])
    ])
