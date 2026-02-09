"""Base join operations for intersects."""

__all__ = [
    'pairwise_join',
    'pairwise_join_iter',
    'pairwise_join_tree',
]

import collections
import operator
from typing import Iterable, Iterator, Optional

import intervaltree
import polars as pl

from .col import COL_CHROM, CoordCol, get_coord_cols

CHUNK_SIZE: int = 2_500
"""Default size of join chunks. Breaks up tables into batches of this size or less."""

class _JoinResources:
    """Resources for joining tables."""
    df_a: pl.LazyFrame
    df_b: pl.LazyFrame
    distance: int
    chunk_size: int
    col_a: CoordCol
    col_b: CoordCol

    def __init__(
            self,
            df_a: pl.LazyFrame,
            df_b: pl.LazyFrame,
            distance: int = 0,
            chunk_size: int = CHUNK_SIZE,
            col_names_a: Optional[CoordCol | Iterable[str] | str] = None,
            col_names_b: Optional[CoordCol | Iterable[str] | str] = None,
    ):

        if chunk_size < 1:
            raise ValueError('chunk_size must be greater than 0')

        if isinstance(df_a, pl.DataFrame):
            df_a = df_a.lazy()

        if isinstance(df_b, pl.DataFrame):
            df_b = df_b.lazy()

        cols_a = set(df_a.collect_schema().keys())
        cols_b = set(df_b.collect_schema().keys())

        if '_index' not in cols_a:
            df_a = df_a.with_row_index('_index')

        if '_index' not in cols_b:
            df_b = df_b.with_row_index('_index')

        # Set column names
        ref_cols = get_coord_cols('ref')

        try:
            col_select_a = get_coord_cols(col_names_a).exprs(alias=ref_cols, suffix='_a')
        except (ValueError, TypeError) as e:
            raise ValueError(f'col_names_a: {e}')

        try:
            col_select_b = get_coord_cols(col_names_b).exprs(alias=ref_cols, suffix='_b')
        except (ValueError, TypeError) as e:
            raise ValueError(f'col_names_b: {e}')

        col_a = col_select_a.col_names()
        col_b = col_select_b.col_names()

        # Prepare tables
        df_a = (
            df_a
            .select(
                pl.col('_index').alias('_index_a'),
                *col_select_a
            )
        )

        df_b = (
            df_b
            .select(
                pl.col('_index').alias('_index_b'),
                *col_select_b
            )
        )

        self.df_a = df_a
        self.df_b = df_b
        self.distance = distance
        self.chunk_size = chunk_size
        self.col_a = col_a
        self.col_b = col_b


def pairwise_join(
        df_a: pl.LazyFrame | pl.DataFrame,
        df_b: pl.LazyFrame | pl.DataFrame,
        distance: int = 0,
        chunk_size: int = CHUNK_SIZE,
        col_names_a: Optional[CoordCol | Iterable[str] | str] = None,
        col_names_b: Optional[CoordCol | Iterable[str] | str] = None,
) -> pl.LazyFrame:
    """Join two tables.

    Returns a table with columns:

        * index_a: Index in table a.
        * index_b: Index in table b.
        * chrom: Chromosome matched.
        * pos: Start position of intersection.
        * end: End position of intersection.
        * distance: Distance between the two intervals with negative values representing overlapping intervals.

    Note that if padding is greater than 0, the "pos" and "end" will have been modified to include padding.

    :param df_a: Table a.
    :param df_b: Table b.
    :param distance: Maximum distance between two records. May be negative to force overlapping.
    :param chunk_size: Chunk tables by this size to limit the effects of combinatorial explosions.
    :param col_names_a: Columns to select from `df_a` if not None, otherwise, use object defaults.
    :param col_names_b: Columns to select from `df_b` if not None, otherwise, use object defaults.

    :return: A LazyFrame with the joined tables.
    """
    # Do join
    return pl.concat(_join_chunks(
        _JoinResources(
            df_a=df_a,
            df_b=df_b,
            distance=distance,
            chunk_size=chunk_size,
            col_names_a=col_names_a,
            col_names_b=col_names_b,
        )
    ))

def pairwise_join_iter(
        df_a: pl.LazyFrame | pl.DataFrame,
        df_b: pl.LazyFrame | pl.DataFrame,
        distance: int = 0,
        chunk_size: int = 5_000,
        col_names_a: Optional[CoordCol | Iterable[str] | str] = None,
        col_names_b: Optional[CoordCol | Iterable[str] | str] = None,
) -> Iterator[pl.LazyFrame]:
    """Join two tables.

    Returns a table with columns:

        * index_a: Index in table a.
        * index_b: Index in table b.
        * chrom: Chromosome matched.
        * pos: Start position of intersection.
        * end: End position of intersection.
        * distance: Distance between the two intervals with negative values representing overlapping intervals.

    Note that if padding is greater than 0, the "pos" and "end" will have been modified to include padding.

    :param df_a: Table a.
    :param df_b: Table b.
    :param distance: Maximum distance between two records. May be negative to force overlapping.
    :param chunk_size: Chunk tables by this size to limit the effects of combinatorial explosions.
    :param col_names_a: Columns to select from `df_a` if not None, otherwise, use object defaults.
    :param col_names_b: Columns to select from `df_b` if not None, otherwise, use object defaults.

    :return: A LazyFrame with the joined tables.
    """
    # Do join
    return _join_chunks(
        _JoinResources(
            df_a=df_a,
            df_b=df_b,
            distance=distance,
            chunk_size=chunk_size,
            col_names_a=col_names_a,
            col_names_b=col_names_b,
        )
    )

def _join_chunks(
        join_resources: _JoinResources
) -> Iterator[pl.LazyFrame]:
    """An iterator for joining by chunks."""

    df_a = join_resources.df_a
    df_b = join_resources.df_b
    distance = join_resources.distance
    chunk_size = join_resources.chunk_size
    col_a = join_resources.col_a
    col_b = join_resources.col_b

    for chrom, last_index_a in (
        df_a
        .group_by(col_a.chrom)
        .agg(pl.len().alias('last_index'))
        .sort(col_a.chrom)
    ).collect().rows():
        start_index_a = 0

        df_a_chrom = (
            df_a.filter(pl.col(col_a.chrom) == chrom)
            .with_row_index('_index_chrom_a')
        )

        while start_index_a < last_index_a:
            end_index_a = start_index_a + chunk_size

            df_a_chunk = df_a_chrom.filter(
                pl.col('_index_chrom_a') >= start_index_a,
                pl.col('_index_chrom_a') < end_index_a
            )

            end_max, pos_min = pl.collect_all([
                df_a_chunk.select(pl.col(col_a.end).max()),
                df_a_chunk.select(pl.col(col_a.pos).min()),
            ])

            end_max = end_max.item()
            pos_min = pos_min.item()

            if end_max is None or pos_min is None:
                start_index_a = end_index_a
                continue

            df_b_chunk = (
                df_b.filter(
                    pl.col(col_b.chrom) == chrom,
                    pl.col(col_b.pos) - distance < end_max,
                    pl.col(col_b.end) + distance > pos_min,
                )
                .with_row_index('_index_chunk_b')
            )

            start_index_b = 0
            last_index_b = df_b_chunk.select(pl.col('_index_chunk_b').max() + 1).collect().item()

            if last_index_b is None:
                start_index_a = end_index_a
                continue

            while start_index_b < last_index_b:
                end_index_b = start_index_b + chunk_size

                yield (
                    df_a_chunk
                    .join(
                        df_b_chunk.filter(
                            pl.col('_index_chunk_b') >= start_index_b,
                            pl.col('_index_chunk_b') < end_index_b,
                        ),
                        left_on=col_a.chrom,
                        right_on=col_b.chrom,
                        how='inner',
                    )
                    .filter(
                        pl.col(col_b.pos) - distance <= pl.col(col_a.end),
                        pl.col(col_b.end) + distance >= pl.col(col_a.pos),
                    )
                    .select(
                        pl.col('_index_a').alias('index_a'),
                        pl.col('_index_b').alias('index_b'),
                        pl.col(col_a.chrom).alias('chrom'),
                        pl.max_horizontal(col_a.pos, col_b.pos).alias('pos'),
                        pl.min_horizontal(col_a.end, col_b.end).alias('end'),
                    )
                    .with_columns(
                        pl.min_horizontal('pos', 'end').alias('pos'),
                        pl.max_horizontal('pos', 'end').alias('end'),
                        (pl.col('pos') - pl.col('end')).alias('distance')
                    )
                ).collect().lazy()

                start_index_b = end_index_b

            start_index_a = end_index_a


def pairwise_join_tree(
        df_a: pl.LazyFrame | pl.DataFrame,
        df_b: pl.LazyFrame | pl.DataFrame,
        distance: int = 0,
        chunk_size: int = CHUNK_SIZE,
        col_names_a: Optional[CoordCol | Iterable[str] | str] = None,
        col_names_b: Optional[CoordCol | Iterable[str] | str] = None,
) -> pl.DataFrame:
    """Join two tables using an interval tree to load df_b into memory.

    This is often more efficient than a pure-polars solution.

    Returns a table with columns:

        * index_a: Index in table a.
        * index_b: Index in table b.
        * chrom: Chromosome matched.
        * pos: Start position of intersection.
        * end: End position of intersection.
        * distance: Distance between the two intervals with negative values representing overlapping intervals.

    Note that if padding is greater than 0, the "pos" and "end" will have been modified to include padding.

    :param df_a: Table a.
    :param df_b: Table b.
    :param distance: Maximum distance between two records. May be negative to force overlapping.
    :param chunk_size: Chunk tables by this size to limit the effects of combinatorial explosions.
    :param col_names_a: Columns to select from `df_a` if not None, otherwise, use object defaults.
    :param col_names_b: Columns to select from `df_b` if not None, otherwise, use object defaults.

    :return: A DataFrame with the joined tables.
    """

    # Use Join Resources to normalize and check
    join_resources = _JoinResources(
        df_a=df_a,
        df_b=df_b,
        distance=distance,
        chunk_size=chunk_size,
        col_names_a=col_names_a,
        col_names_b=col_names_b,
    )

    df_a = join_resources.df_a
    df_b = join_resources.df_b
    distance = join_resources.distance
    chunk_size = join_resources.chunk_size
    col_a = join_resources.col_a
    col_b = join_resources.col_b

    # Load tree
    itree = collections.defaultdict(intervaltree.IntervalTree)

    for df_batch_b in (
        df_b
        .select(pl.col('_index_b'), *col_b.exprs())
        .collect_batches(chunk_size=chunk_size)
    ):
        for index_b, chrom_b, pos_b, end_b in df_batch_b.iter_rows():
            itree[chrom_b].addi(pos_b - distance, end_b + distance, index_b)

    # Intersect
    match_list = []

    for df_chunk in (
        df_a
        .select(pl.col('_index_a'), *col_a.exprs())
        .collect_batches(chunk_size=chunk_size)
    ):
        for index_a, chrom_a, pos_a, end_a in df_chunk.iter_rows():
            for interval in sorted(itree[chrom_a][pos_a:end_a], key=operator.attrgetter('data')):

                pos = max(pos_a, interval.begin + distance)
                end = min(end_a, interval.end - distance)

                match_list.append((
                    index_a,  # index_a
                    interval.data,  # index_b
                    chrom_a,  # chrom
                    min((pos, end)),  # pos
                    max((pos, end)),  # end
                    pos - end  # distance
                ))

    return pl.DataFrame(
        match_list,
        orient='row',
        schema={
            'index_a': pl.UInt32,
            'index_b': pl.UInt32,
            'chrom': pl.String,
            'pos': pl.Int64,
            'end': pl.Int64,
            'distance': pl.Int64
        }
    )
