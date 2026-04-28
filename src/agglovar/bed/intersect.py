"""Table intersects."""

from pathlib import Path
from typing import Iterable, Optional

import polars as pl

from .join import pairwise_join_tree, pairwise_join_iter
from .merge import merge_depth
from .col import CoordCol, get_coord_cols

__all__ = [
    'as_bool',
    'as_proportion',
]


def as_bool(
        df_a: pl.LazyFrame | pl.DataFrame,
        df_b: pl.LazyFrame | pl.DataFrame,
        name: str,
        distance: int = 0,
        negate: bool = False,
        col_names_a: Optional[CoordCol | Iterable[str] | str] = None,
        col_names_b: Optional[CoordCol | Iterable[str] | str] = None,
        temp_dir: bool | str | Path = False,
) -> pl.LazyFrame:
    """Add a boolean column to df_a indicating whether each record intersects with df_b.

    :param df_a: Table a.
    :param df_b: Table b.
    :param name: Name of the column to add.
    :param distance: Maximum distance between two records. May be negative to require overlap.
    :param negate: If True, negate the boolean column to annotate misses instead of hits.
    :param col_names_a: Columns in a (chromosome or query ID, pos, end).
    :param col_names_b: Columns in b (chromosome or query ID, pos, end).
    :param temp_dir: How to materialise the prepared tables before iterating. See
        :func:`agglovar.bed.join.pairwise_join`.

    :return: A LazyFrame with two columns: ``_index`` and ``name``.
    """
    if name is None or not (name := name.strip()):
        raise ValueError('Name must be a non-empty string')

    col_names_a = get_coord_cols(col_names_a)
    col_names_b = get_coord_cols(col_names_b)

    if isinstance(df_a, pl.DataFrame):
        df_a = df_a.lazy()

    if isinstance(df_b, pl.DataFrame):
        df_b = df_b.lazy()

    hit_val = not negate

    join_list = []

    for df_join in pairwise_join_iter(
            df_a=df_a,
            df_b=df_b,
            distance=distance,
            col_names_a=col_names_a,
            col_names_b=col_names_b,
            temp_dir=temp_dir,
    ):
        join_list.append(
            df_join
            .select(pl.col('index_a').alias('_index'))
            .collect()
            .lazy()
        )

    if '_index' in df_a.collect_schema().names():
        df_a_indexed = df_a
    else:
        df_a_indexed = df_a.with_row_index('_index').with_columns(pl.col('_index').cast(pl.UInt64))

    if not join_list:
        return (
            df_a_indexed
            .select(
                '_index',
                pl.lit(not hit_val).alias(name),
            )
        )

    hits = pl.concat(join_list).unique('_index')

    return (
        df_a_indexed
        .select('_index')
        .join(
            hits.with_columns(pl.lit(hit_val).alias(name)),
            on='_index', how='left',
        )
        .select(
            '_index',
            pl.col(name).fill_null(not hit_val),
        )
    )


def as_proportion(
        df_a: pl.LazyFrame | pl.DataFrame,
        df_b: pl.LazyFrame | pl.DataFrame,
        name: str,
        col_names_a: Optional[CoordCol | Iterable[str] | str] = None,
        col_names_b: Optional[CoordCol | Iterable[str] | str] = None,
        temp_dir: bool | str | Path = False,
) -> pl.LazyFrame:
    """Compute the proportion of each interval in ``df_a`` covered by intervals in ``df_b``.

    Rows in ``df_a`` with null ``pos`` or ``end`` are preserved in the output with a null
    proportion. Zero-length intervals (``pos == end``) produce ``NaN`` (0 / 0).

    :param df_a: Table a.
    :param df_b: Table b.
    :param name: Name of the column to add.
    :param col_names_a: Columns in a (chromosome or query ID, pos, end).
    :param col_names_b: Columns in b (chromosome or query ID, pos, end).
    :param temp_dir: How to materialise the prepared tables before iterating. See
        :func:`agglovar.bed.join.pairwise_join`.

    :return: A LazyFrame with two columns: ``_index`` and ``name``.
    """
    if name is None or not (name := name.strip()):
        raise ValueError('Name must be a non-empty string')

    col_names_a = get_coord_cols(col_names_a)
    col_names_b = get_coord_cols(col_names_b)

    col_expr_a = col_names_a.exprs()

    if isinstance(df_a, pl.DataFrame):
        df_a = df_a.lazy()

    if isinstance(df_b, pl.DataFrame):
        df_b = df_b.lazy()

    if '_index' not in df_a.collect_schema().names():
        df_a = df_a.with_row_index('_index').with_columns(pl.col('_index').cast(pl.UInt64))

    # Records with null pos/end are kept and assigned a null proportion at the end.
    df_a_clean = df_a.filter(
        pl.col(col_names_a.pos).is_not_null(),
        pl.col(col_names_a.end).is_not_null(),
    )

    # Collapse overlapping b intervals so coverage is not double-counted.
    df_b_nr = (
        merge_depth(df_b, 0, col_names_b)
        .select(*col_names_b.exprs())
        .collect()
    )

    df_join = pairwise_join_tree(
        df_a=df_a_clean,
        df_b=df_b_nr,
        col_names_a=col_names_a,
        col_names_b=col_names_b,
        temp_dir=temp_dir,
    )

    df_overlap = (
        df_join
        .filter(pl.col('end') > pl.col('pos'))
        .select(
            pl.col('index_a').alias('_index'),
            (pl.col('end') - pl.col('pos')).alias('_overlap'),
        )
        .group_by('_index')
        .agg(pl.col('_overlap').sum())
    )

    return (
        df_a
        .select(
            '_index',
            (col_expr_a.end - col_expr_a.pos).alias('_len'),
        )
        .join(df_overlap.lazy(), on='_index', how='left')
        .select(
            '_index',
            (pl.col('_overlap').fill_null(0.0) / pl.col('_len')).alias(name),
        )
    )
