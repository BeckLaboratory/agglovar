"""Routines for removing duplicates from pairwise join tables."""

from collections.abc import Iterator
from enum import Enum
from typing import Optional

import polars as pl

from .weights import WeightStrategy


class DedupPriority(Enum):
    """Priority for de-duplication on "index_a" and "index_b" in a join table.

        * MAX: Keep the record with the maximum weight over "index_a" and "index_b". For tied
            weights, the first record is kept.
        * FIRST: Keep the first record and discard subsequent records with the same "index_a" and
            "index_b".
    """
    MAX = 'max'
    FIRST = 'first'


def dedup_iter(
        join_iter: Iterator[pl.LazyFrame],
        dedup_priority: Optional[DedupPriority | str] = DedupPriority.FIRST
) -> Iterator[pl.LazyFrame]:
    """De-duplicates tables from an iterator.

    :param join_iter: Iterator of tables to de-duplicate.
    :param dedup_priority: De-duplication priority.

    :return: An iterator of de-duplicated join tables.
    """

    if isinstance(dedup_priority, str):
        try:
            dedup_priority = DedupPriority(dedup_priority)
        except ValueError:
            raise ValueError(f'Unknown dedup priority (from string): {dedup_priority}')

    if dedup_priority == DedupPriority.FIRST:
        return dedup_iter_first(join_iter)
    elif dedup_priority == DedupPriority.MAX:
        return dedup_iter_max(join_iter)
    else:
        raise ValueError(f'Unknown dedup priority: {dedup_priority}')


def dedup_iter_first(
        join_iter: Iterator[pl.LazyFrame],
) -> Iterator[pl.LazyFrame]:
    """De-duplicate with strategy FIRST.

    Keep the first record and discard subsequent records with the same "index_a" and "index_b".
    This is faster than MAX. Each de-duplicated table is output by the iterator.

    :param join_iter: Iterator of input tables.

    :return: An iterator of de-duplicated tables.
    """
    seen_pair_list: list[pl.DataFrame] = [
        pl.DataFrame([], schema=[('index_a', pl.UInt32), ('index_b', pl.UInt32)])
    ]

    for df in join_iter:
        df = (
            df
            .unique(
                ['index_a', 'index_b'],
                keep='first',
                maintain_order=True,
            )
            .join(
                pl.concat(seen_pair_list).lazy(),
                on=['index_a', 'index_b'],
                how='anti',
            )
        )

        seen_pair_list.append(
            df.select('index_a', 'index_b').collect()
        )

        yield df


def dedup_iter_max(
        join_iter: Iterator[pl.LazyFrame],
        missing_weight_strategy: Optional[WeightStrategy] = None
) -> Iterator[pl.LazyFrame]:
    """De-duplicate with strategy MAX.

    This iterator must first consume every table, then concat and de-duplicate. The iterator
    returned by this strategy always has exactly one item (one whole de-duplicated table).

    :param join_iter: Iterator of input tables.

    :return: An iterator of de-duplicated tables.
    """

    # Fill missing weights
    if missing_weight_strategy is not None:
        weight_expr = pl.coalesce(
            pl.col(r'^weight$'),
            missing_weight_strategy.expr,
            pl.lit(0.0)
        ).cast(pl.Float32)
    else:
        weight_expr = pl.coalesce(
            pl.col(r'^weight$'),
            pl.lit(0.0)
        ).cast(pl.Float32)

    # Filter for max weight
    yield (
        pl.concat(join_iter)
        .with_columns(
            weight_expr.alias('weight')
        )
        .filter(
            pl.col('weight')
            .rank(method='ordinal', descending=True)
            .over('index_a', 'index_b')
            == 1
        )
        .collect()
        .lazy()
    )
