"""A merging strategy that adds callsets cumulatively to the merge.

This strategy maintains a current state of a callset and cumulatively adds callsets to the merge.
The merge callset is initially empty. The first callset added becomes the current state. Each
additional callset is added by joining the current state with the new callset. Variants that do
not join with the existing callset are added to the merge state for the next callset. Variants
that do join with the existing callset are tracked so that one variant call in the current state
may represent a call from one or more sources.

This strategy is fast and uses minimal memory, but is not necessarily optimal. The order variants
are input into the callset may alter the merged results in nontrivial ways, especially in loci
where multiple join choices are possible.
"""

__all__ = [
    'MergeCumulative',
]

from collections.abc import Iterable
from typing import Optional

import polars as pl

from ..pairwise.base import PairwiseJoin
from ..meta.decorators import immutable

from .base import (
    CallsetDefType,
    MergeBase
)

@immutable
class MergeCumulative(MergeBase):
    """Iterative intersection.

    :ivar join: Pairwise join strategy for intersects.
    """
    join: PairwiseJoin

    def __init__(
            self,
            join: PairwiseJoin,
    ) -> None:
        """Create an iterative intersection object."""
        super().__init__()

        if join is None:
            raise ValueError('Missing join object.')

        self.join = join

    def __call__(
            self,
            callsets: Iterable[CallsetDefType],
            pre_check_schema: bool = True
    ) -> pl.LazyFrame:
        """
        Intersect callsets.

        :param callsets: Callsets to intersect.
        :param pre_check_schema: If True, check the schema on all input tables before intersecting.
            this catches errors early and produces better error messages, but can be expensive if
            a lazy frame was input with non-trivial transformations. Should be True for most
            input.

        :return: A merged callset table.
        """
        if callsets is None:
            raise ValueError('Missing callsets.')

        callsets = MergeBase.get_intersect_tuples(callsets)

        if len(callsets) == 0:
            raise ValueError('No callsets to intersect.')

        # Check required columns
        if pre_check_schema:
            for df_next, next_name, source_index in callsets:
                if missing_cols := self.join.required_cols - set(df_next.collect_schema().names()):
                    raise ValueError(f'Missing columns for source ({next_name}, index {source_index}): "{", ".join(sorted(missing_cols))}"')

        # Run merge
        df_cumulative: Optional[pl.LazyFrame] = None
        cols: list[str] = []

        for df_next, next_name, source_index in callsets:
            cols = [
                col for col in df_next.collect_schema().names()
                    if col in self.join.required_cols
            ]

            # Initialize with the first table
            if df_cumulative is None:
                df_cumulative = (
                    df_next
                    .select(cols)
                    .with_row_index('_index')
                    .with_columns(
                        pl.concat_list(
                            pl.struct([
                                pl.lit(source_index).cast(pl.Int32).alias('source'),
                                pl.col("_index").cast(pl.Int32).alias('index'),
                            ])
                        ).alias('_source')
                    )
                    .drop('_index')
                    .collect()
                    .lazy()
                )

                continue

            # Add next callset
            df_join = pl.concat(
                [
                    (
                        df_join
                        .select('index_a', 'index_b')
                    )
                        for df_join in self.join.join_iter(df_cumulative, df_next)
                ]
            ).collect()





        raise NotImplementedError()
        return None
