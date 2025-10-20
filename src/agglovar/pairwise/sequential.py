"""Run one or more pairwise intersects sequentially and concatenate join tables.

Provides an interface for executing multiple pairwise join strategies and generating a unified
join table.

Some intersect strategies are implemented by iterating through an ordered set of join strategies
with different parameters. For example, parameters joining large SVs may differ from parameters for
joining small SVs and indels. Instead of bifurcating a callset into "large" and "small" variants,
which leads to all kinds of difficulty and arbitrary cutoff values, a sequential strategy can run a
series of pairwise intersects with different parameters and concatenate the results into one
unified join table.

Like other pairwise joins, this sequential strategy is also pairwise and will only intersect two
tables.
"""

__all__ = [
    'PairwiseSequential',
]

from collections.abc import Iterator
from typing import Optional

import polars as pl

from .base import PairwiseJoin
from .dedup import DedupPriority, dedup_iter
from .weights import WeightStrategy, DEFAULT_WEIGHT_STRATEGY

class PairwiseSequential(PairwiseJoin):
    """Run multiple pairwise joins and produce a unified join table."""

    _pairwise_list: list[PairwiseJoin]
    dedup_priority: DedupPriority

    def __init__(
            self,
            weight_strategy: Optional[WeightStrategy] = None,
            dedup_priority: DedupPriority = DedupPriority.FIRST,
    ) -> None:
        """Initialize a MultiPair object.

        :param weight_strategy: Default weight strategy for join operations in this class. If none,
             a global default weight strategy is set.
        """
        super().__init__(weight_strategy)

        if dedup_priority is None:
            dedup_priority = DedupPriority.FIRST

        self._pairwise_list = []
        self.dedup_priority = dedup_priority

    def add_join(
            self,
            pairwise_join: PairwiseJoin,
    ) -> None:
        """Add a pairwise join.

        Add a pairwise join object to this class and a weight strategy for computing weights for
        each this join. If the weight expression is not

        :param pairwise_join: Join object.
        """
        if self.is_locked:
            raise AttributeError('Pairwise intersect object is locked and cannot be modified.')

        if pairwise_join is None:
            raise ValueError('Cannot add join: Missing join object.')

        self._pairwise_list.append(pairwise_join)

    def join(
            self,
            df_a: pl.DataFrame,
            df_b: pl.DataFrame,
            collect: bool = False
    ) -> pl.DataFrame:
        """Join all pairwise intersects.

        :param df_a: Source dataframe.
        :param df_b: Target dataframe.
        :param collect: Whether to collect the join table.

        :returns: A join table.
        """
        return pl.concat(*self.join_iter(df_a, df_b))

    def join_iter(
            self,
            df_a: pl.DataFrame | pl.LazyFrame,
            df_b: pl.DataFrame | pl.LazyFrame
    ) -> Iterator[pl.LazyFrame]:
        """Find all pairs of variants in two sources that meet a set of criteria.

        :param df_a: Source dataframe.
        :param df_b: Target dataframe.

        :yields: A LazyFrame for each chunk.
        """
        return dedup_iter(
            (
                df_join
                for pairwise_join in self._pairwise_list
                for df_join in pairwise_join.join_iter(df_a, df_b)
            ),
            dedup_priority=self.dedup_priority,
        )

    @property
    def required_cols(self) -> set[str]:
        """Get the set of required columns from all pairwise intersects."""
        return set.union(
            *(pairwise_join.required_cols for pairwise_join in self._pairwise_list)
        )
