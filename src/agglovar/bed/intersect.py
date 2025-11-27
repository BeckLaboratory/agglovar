"""Table intersects."""

from typing import Iterable, Optional

import polars as pl

from .join import pairwise_join_tree, pairwise_join_iter
from .merge import merge_depth
from .col import CoordCol, get_coord_cols


def as_bool(
        df_a: pl.LazyFrame | pl.DataFrame,
        df_b: pl.LazyFrame | pl.DataFrame,
        name: Optional[str] = None,
        negate: bool = False,
        col_names_a: Optional[CoordCol | Iterable[str] | str] = None,
        col_names_b: Optional[CoordCol | Iterable[str] | str] = None,
) -> pl.LazyFrame:
    """Add a boolean column to df_a indicating whether each record intersects with df_b.

    :param df_a: Table a.
    :param df_b: Table b.
    :param name: Nome of the column to add.
    :param negate: If True, negate the boolean column to annotate misses instead of hits.
    :param col_names_a: Columns in a (chromosome or query ID, pos, end).
    :param col_names_b: Columns in b (chromosome or query ID, pos, end).

    :return: A LazyFrame with the boolean column added.
    """
    col_names_a = get_coord_cols(col_names_a)
    col_names_b = get_coord_cols(col_names_b)

    hit_val = False if negate else True

    if name is None or not (name := name.strip()):
        raise ValueError('Name must be a non-empty string')

    join_list = []

    for df_join in pairwise_join_iter(
            df_a=df_a,
            df_b=df_b,
            col_names_a=col_names_a,
            col_names_b=col_names_b,
    ):
        join_list.append(
            df_join
            .select(
                pl.col('index_a').alias('_index'),
                pl.lit(hit_val).alias(name)
            )
            .collect()
            .lazy()
        )

    if not join_list:
        return pl.DataFrame([], schema={name: pl.Boolean}).with_row_index('_index').lazy()

    return (
        df_a
        .drop('_index', strict=False)
        .with_row_index('_index')
        .join(
            pl.concat(join_list).unique('_index'),
            on='_index', how='left'
        )
        .select(
            '_index',
            pl.col(name).fill_null(not hit_val)
        )
    )

def as_proportion(
        df_a: pl.LazyFrame | pl.DataFrame,
        df_b: pl.LazyFrame | pl.DataFrame,
        name: Optional[str] = None,
        col_names_a: Optional[CoordCol | Iterable[str] | str] = None,
        col_names_b: Optional[CoordCol | Iterable[str] | str] = None,
) -> pl.LazyFrame:
    col_names_a = get_coord_cols(col_names_a)
    col_names_b = get_coord_cols(col_names_b)

    col_expr_a = col_names_a.exprs()

    if name is None or not (name := name.strip()):
        raise ValueError('Name must be a non-empty string')

    # Collapse b, index a
    df_b_nr = (
        merge_depth(df_b, 0, col_names_b)
        .filter(pl.col('max_depth') > 0)
        .select(*col_names_b.exprs())
        .collect()
    )

    df_a = (
        df_a
        .drop('_index', strict=False)
        .with_row_index('_index')
        .filter(
            pl.col(col_names_a.pos).is_not_null(),
            pl.col(col_names_a.end).is_not_null(),
        )
    )

    df_join = pairwise_join_tree(
        df_a=df_a,
        df_b=df_b_nr,
        col_names_a=col_names_a,
        col_names_b=col_names_b,
    )

    df_join = (
        df_join
        .filter(pl.col('end') > pl.col('pos'))
        .select(
            pl.col('index_a').alias('_index'),
            (pl.col(col_names_b.end) - pl.col(col_names_b.pos)).alias('_overlap')
        )
        .group_by('_index')
        .agg(
            pl.col('_overlap').sum()
        )
    )

    return (
        df_a
        .select(
            '_index',
            (col_expr_a.end - col_expr_a.pos).alias('len')
        )
        .join(
            df_join.lazy(),
            on='_index',
            how='left',
        )
        .select(
            '_index',
            (pl.col('_overlap').fill_null(0.0) / pl.col('len')).alias(name)
        )
    )

    # Join in chunks
    # df_overlap_list = []
    #
    # for df_join in pairwise_join_iter(
    #         df_a=df_a,
    #         df_b=df_b_nr,
    #         col_names_a=col_names_a,
    #         col_names_b=col_names_b,
    # ):
    #     df_overlap_list.append(
    #         df_join
    #         .filter(pl.col('end') > pl.col('pos'))
    #         .select(
    #             pl.col('index_a').alias('_index'),
    #             (pl.col(col_names_b.end) - pl.col(col_names_b.pos)).alias('_overlap')
    #         )
    #         .group_by('_index')
    #         .agg(
    #             pl.col('_overlap').sum()
    #         )
    #     )
    #
    # if not df_overlap_list:
    #     return pl.DataFrame([], schema={name: pl.String}).with_row_index('_index').lazy()
    #
    # return (
    #     df_a
    #     .select(
    #         '_index',
    #         (col_expr_a.end - col_expr_a.pos).alias('len')
    #     )
    #     .join(
    #         pl.concat(df_overlap_list),
    #         on='_index',
    #         how='left',
    #     )
    #     .select(
    #         '_index',
    #         (pl.col('_overlap').fill_null(0.0) / pl.col('len')).alias(name)
    #     )
    # )
