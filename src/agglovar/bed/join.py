"""Base join operations for intersects."""

__all__ = [
    'pairwise_join',
    'pairwise_join_iter',
    'pairwise_join_tree',
]

from pathlib import Path
from typing import Iterable, Iterator, Optional

import polars as pl

from .col import CoordCol, get_coord_cols
from ..util.lazy import materialize_pair

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
            df_a = df_a.with_row_index('_index').with_columns(pl.col('_index').cast(pl.UInt64))

        if '_index' not in cols_b:
            df_b = df_b.with_row_index('_index').with_columns(pl.col('_index').cast(pl.UInt64))

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
        temp_dir: bool | str | Path = False,
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
    :param temp_dir: How to materialise the prepared tables before the chunked loop.
        ``False`` (default) collects both into memory; ``True`` writes them to the
        system temp directory as parquet files; a ``str``/``Path`` writes them to
        that directory. Temp files are always removed on exit.

    :return: A LazyFrame with the joined tables.
    """
    resources = _JoinResources(
        df_a=df_a,
        df_b=df_b,
        distance=distance,
        chunk_size=chunk_size,
        col_names_a=col_names_a,
        col_names_b=col_names_b,
    )

    with materialize_pair(
        resources.df_a, resources.df_b, temp_dir, prefix='bed_join_prep_',
    ) as (df_a_mat, df_b_mat):
        resources.df_a = df_a_mat
        resources.df_b = df_b_mat
        chunks = list(_join_chunks(resources))

    if not chunks:
        return _empty_join(resources)

    return pl.concat(chunks)


def pairwise_join_iter(
        df_a: pl.LazyFrame | pl.DataFrame,
        df_b: pl.LazyFrame | pl.DataFrame,
        distance: int = 0,
        chunk_size: int = 5_000,
        col_names_a: Optional[CoordCol | Iterable[str] | str] = None,
        col_names_b: Optional[CoordCol | Iterable[str] | str] = None,
        temp_dir: bool | str | Path = False,
) -> Iterator[pl.LazyFrame]:
    """Join two tables, yielding one LazyFrame per chunk.

    Returns chunks with the same columns as :func:`pairwise_join`. At least one chunk
    is always yielded; an empty schema-only frame is yielded when no chunk would otherwise
    have been produced (so callers can safely call ``pl.concat`` on the result).

    :param df_a: Table a.
    :param df_b: Table b.
    :param distance: Maximum distance between two records. May be negative to force overlapping.
    :param chunk_size: Chunk tables by this size to limit the effects of combinatorial explosions.
    :param col_names_a: Columns to select from `df_a` if not None, otherwise, use object defaults.
    :param col_names_b: Columns to select from `df_b` if not None, otherwise, use object defaults.
    :param temp_dir: How to materialise the prepared tables before the chunked loop.
        See :func:`pairwise_join`.

    :return: An iterator of LazyFrames.
    """
    resources = _JoinResources(
        df_a=df_a,
        df_b=df_b,
        distance=distance,
        chunk_size=chunk_size,
        col_names_a=col_names_a,
        col_names_b=col_names_b,
    )

    with materialize_pair(
        resources.df_a, resources.df_b, temp_dir, prefix='bed_join_prep_',
    ) as (df_a_mat, df_b_mat):
        resources.df_a = df_a_mat
        resources.df_b = df_b_mat

        yielded = False

        for chunk in _join_chunks(resources):
            yielded = True
            yield chunk

        if not yielded:
            yield _empty_join(resources)


def _build_join_pair(
        df_a: pl.LazyFrame,
        df_b: pl.LazyFrame,
        distance: int,
        col_a: CoordCol,
        col_b: CoordCol,
) -> pl.LazyFrame:
    """Apply the cross-join, distance filter, and output projection.

    Both ``df_a`` and ``df_b`` must already have ``_index_a``/``_index_b`` columns
    and the coordinate columns named per ``col_a``/``col_b``. The output schema
    is invariant across all callers.
    """
    return (
        df_a
        .join(df_b, left_on=col_a.chrom, right_on=col_b.chrom, how='inner')
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
            (pl.col('pos') - pl.col('end')).alias('distance'),
        )
    )


def _empty_join(join_resources: _JoinResources) -> pl.LazyFrame:
    """Schema-only LazyFrame for the empty-input case."""
    return _build_join_pair(
        join_resources.df_a.head(0),
        join_resources.df_b.head(0),
        join_resources.distance,
        join_resources.col_a,
        join_resources.col_b,
    )


def _join_chunks(
        join_resources: _JoinResources
) -> Iterator[pl.LazyFrame]:
    """Iterate over join results by chunks. Yields nothing if no chunks have rows."""
    df_a = join_resources.df_a
    df_b = join_resources.df_b
    distance = join_resources.distance
    chunk_size = join_resources.chunk_size
    col_a = join_resources.col_a
    col_b = join_resources.col_b

    # Restrict the chrom loop to chroms present in both tables. Chroms unique to A
    # would otherwise drive a full chunk loop whose B filter always returns empty.
    for chrom, last_index_a in (
        df_a
        .group_by(col_a.chrom)
        .agg(pl.len().alias('last_index'))
        .join(
            df_b.select(pl.col(col_b.chrom)).unique(),
            left_on=col_a.chrom, right_on=col_b.chrom,
            how='inner',
        )
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
                pl.col('_index_chrom_a') < end_index_a,
            )

            end_max, pos_min = (
                df_a_chunk
                .select(
                    pl.col(col_a.end).max().alias('end_max'),
                    pl.col(col_a.pos).min().alias('pos_min'),
                )
                .collect()
                .row(0)
            )

            if end_max is None or pos_min is None:
                start_index_a = end_index_a
                continue

            # Pre-filter df_b to records that could possibly match any A record in
            # this chunk. The bounds use <= and >= to match the inner filter so
            # boundary cases (touching, exact-overlap-at-negative-distance) are not
            # silently dropped before they reach the inner join.
            df_b_chunk = (
                df_b.filter(
                    pl.col(col_b.chrom) == chrom,
                    pl.col(col_b.pos) - distance <= end_max,
                    pl.col(col_b.end) + distance >= pos_min,
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

                yield _build_join_pair(
                    df_a_chunk,
                    df_b_chunk.filter(
                        pl.col('_index_chunk_b') >= start_index_b,
                        pl.col('_index_chunk_b') < end_index_b,
                    ),
                    distance,
                    col_a,
                    col_b,
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
        temp_dir: bool | str | Path = False,
) -> pl.DataFrame:
    """Join two tables using a per-chromosome IEJoin (Polars ``join_where``).

    Differs from :func:`pairwise_join` in the inner join algorithm: each A chunk is
    paired with the chrom-matched B records via ``join_where`` instead of an
    equi-join on chrom plus a post-filter. IEJoin sorts on the inequality keys and
    is selective on small overlap rates, but its working set scales with the
    cross-product of the chunk pair, so chunking on A is required to bound memory.

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
    :param chunk_size: Chunk A by this size per chromosome to bound the IEJoin working set.
    :param col_names_a: Columns to select from `df_a` if not None, otherwise, use object defaults.
    :param col_names_b: Columns to select from `df_b` if not None, otherwise, use object defaults.
    :param temp_dir: How to materialise the prepared tables before iterating. See :func:`pairwise_join`.

    :return: A DataFrame with the joined tables.
    """
    join_resources = _JoinResources(
        df_a=df_a,
        df_b=df_b,
        distance=distance,
        chunk_size=chunk_size,
        col_names_a=col_names_a,
        col_names_b=col_names_b,
    )

    with materialize_pair(
        join_resources.df_a, join_resources.df_b, temp_dir, prefix='bed_join_tree_prep_',
    ) as (df_a_mat, df_b_mat):
        join_resources.df_a = df_a_mat
        join_resources.df_b = df_b_mat
        chunks = list(_join_chunks_iejoin(join_resources))

    if not chunks:
        return _empty_join(join_resources).collect()

    return pl.concat(chunks).collect()


def _join_chunks_iejoin(
        join_resources: _JoinResources
) -> Iterator[pl.LazyFrame]:
    """Iterate over join results by chunks using ``join_where`` per A chunk.

    Mirrors :func:`_join_chunks` but replaces the equi-join + filter inner with an
    IEJoin. The chrom equality is enforced by filtering each side to ``chrom``
    upstream so the predicates passed to ``join_where`` are pure inequalities (the
    shape Polars's IEJoin path is optimised for).
    """
    df_a = join_resources.df_a
    df_b = join_resources.df_b
    distance = join_resources.distance
    chunk_size = join_resources.chunk_size
    col_a = join_resources.col_a
    col_b = join_resources.col_b

    # Restrict the chrom loop to chroms present in both tables. Same prefilter as
    # _join_chunks; chroms unique to A would otherwise drive a wasted A chunk loop.
    for chrom, last_index_a in (
        df_a
        .group_by(col_a.chrom)
        .agg(pl.len().alias('last_index'))
        .join(
            df_b.select(pl.col(col_b.chrom)).unique(),
            left_on=col_a.chrom, right_on=col_b.chrom,
            how='inner',
        )
        .sort(col_a.chrom)
    ).collect().rows():
        df_a_chrom = (
            df_a.filter(pl.col(col_a.chrom) == chrom)
            .with_row_index('_index_chrom_a')
        )
        df_b_chrom = df_b.filter(pl.col(col_b.chrom) == chrom)

        start_index_a = 0
        while start_index_a < last_index_a:
            end_index_a = start_index_a + chunk_size

            df_a_chunk = df_a_chrom.filter(
                pl.col('_index_chrom_a') >= start_index_a,
                pl.col('_index_chrom_a') < end_index_a,
            )

            end_max, pos_min = (
                df_a_chunk
                .select(
                    pl.col(col_a.end).max().alias('end_max'),
                    pl.col(col_a.pos).min().alias('pos_min'),
                )
                .collect()
                .row(0)
            )

            if end_max is None or pos_min is None:
                start_index_a = end_index_a
                continue

            # Pre-filter B to records that could possibly match any A record in this
            # chunk. Bounds use <= and >= to match the inner predicate (touching cases
            # at distance == 0 must survive the prefilter).
            df_b_chunk = df_b_chrom.filter(
                pl.col(col_b.pos) - distance <= end_max,
                pl.col(col_b.end) + distance >= pos_min,
            )

            yield (
                df_a_chunk
                .join_where(
                    df_b_chunk,
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
                    (pl.col('pos') - pl.col('end')).alias('distance'),
                )
                .collect()
                .lazy()
            )

            start_index_a = end_index_a
