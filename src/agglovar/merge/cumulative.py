"""A merging strategy that adds callsets cumulatively to the merge.

This strategy uses a table of variants that accumulates as callsets are added (the cumulative
table). The cumulative table is initially empty. As callsets are added, variant intersecting the
cumulative table are added to existing entries, and variants not intersecting the cumulative table
are appended as new variants. This process is repeated for each callset added.

After all callsets are procesesd, the cumulative table represents a nonredundant callset where
each entry is one variant that was found in one or more of the original callsets. Columns
tracking the sources and the variant within each source.

This strategy is fast and uses minimal memory, but is not necessarily optimal. The order variants
are input into the callset may alter the merged results in nontrivial ways, especially in loci
where multiple join choices are possible.
"""

__all__ = [
    'MergeCumulative',
]

from collections.abc import Iterable
from enum import Enum
from typing import Optional

import polars as pl
import polars.selectors as cs

from ..meta.decorators import immutable
from ..pairwise.base import PairwiseJoin
from ..meta.decorators import immutable

from .base import (
    CallsetDefType,
    MergeBase
)

class LeadStrategy(Enum):
    """Strategy for choosing the lead variant.

    When variants from multiple sources join into one record, this strategy determines which
    variant is chosen as the lead variant. The lead variant represents the merged records in the
    merged callset.
    """
    LEFT = 'left'
    FIRST = 'right'

def _init_cumulative(
        df_next: pl.LazyFrame,
        src_index: int,
        src_name: str,
        required_cols: Iterable[str],
        merge_stat_cols: dict[str, pl.DataType]
) -> pl.LazyFrame:
    """Initialize a cumulative table from a callset table.

    Use this function to transform an input table into the cumulative table format. This is needed
    to initialize the cumulative table for the first callset and to add new callsets to the cumulative
    table (un-merged variants).

    Steps:

        1. Drop columns that are not neededfor the merge.
        2. Add columns for tracking variant source ("_mg_src_" prefix).
        3. Add columns for merge stats ("_mg_stat_" prefix).

    :param df_next: Table to convert.
    :param src_index: Merge source index (0 for the first table, increment by 1).
    :param src_name: Name for the source.
    :param required_cols: Columns required for the merge. The cumulative table is "df_a" for joins
        and must have all the columns the join requries.
    :param merge_stat_cols: A dict of merge stat column names (key) and their data type (value).

    :return: A table ready to be used as a cumulative table or appended to an existing one.
    """
    mg_cols = ('_mg_src', '_mg_src_id', '_mg_src_name', '_mg_src_pos')
    required_cols = set(required_cols) - set(mg_cols)

    return (
        df_next
        .select(
            pl.col('_index').alias('_mg_index'),
            *required_cols
        )
        .with_columns(  # Source columns
            pl.lit([int(src_index)]).cast(pl.List(pl.Int32)).alias('_mg_src'),
            pl.lit([str(src_name)]).cast(pl.List(pl.String)).alias('_mg_src_name'),
            pl.lit([]).list.concat(pl.col('_mg_index')).alias('_mg_src_index'),
            pl.lit([]).list.concat(pl.col('id').cast(pl.String)).alias('_mg_src_id'),
            pl.lit([]).list.concat(pl.col('pos').cast(pl.Int64)).alias('_mg_src_pos'),
        )
        .with_columns(  # Merge stat columns
            *(
                pl.lit([]).cast(pl.List(dtype)).alias(f'_mg_stat_{col}')
                for col, dtype in merge_stat_cols.items()
            )
        )
    )

def _get_match(
        df_cumulative: pl.LazyFrame,
        df_next: pl.LazyFrame,
        df_join: pl.DataFrame,
        src_index: int,
        src_name: str,
        merge_stat_cols: dict[str, pl.DataType]
) -> pl.LazyFrame:
    return (  # Update matching variants
        df_cumulative
        .join(
            df_join.lazy().select(pl.all().name.prefix('_mg_join_')),
            left_on='_mg_index', right_on='_mg_join_index_a', how='inner'
        )
        .with_columns(
            pl.col('_mg_src').list.concat(pl.lit(int(src_index))),
            pl.col('_mg_src_name').list.concat(pl.lit(str(src_name))),
            pl.col('_mg_src_index').list.concat(pl.col('_mg_join_index_b')),
            pl.col('_mg_src_id').list.concat(pl.col('_mg_join_id')),
            pl.col('_mg_src_pos').list.concat(pl.col('_mg_join_pos')),
            *(
                pl.col(f'_mg_stat_{col}').list.concat(pl.col(f'_mg_join_{col}').cast(dtype))
                    for col, dtype in merge_stat_cols.items()
            )
        )
        .drop('_mg_index', cs.starts_with('_mg_join_'))
    )


@immutable
class MergeCumulative(MergeBase):
    """Iterative intersection.

    :ivar join: Pairwise join strategy for intersects.
    """
    join: PairwiseJoin

    def __init__(
            self,
            pairwise_join: PairwiseJoin,
            lead_strategy: LeadStrategy = LeadStrategy.LEFT,
    ) -> None:
        """Create an iterative intersection object."""
        super().__init__()

        if pairwise_join is None:
            raise ValueError('Missing pairwise_join')

        self.pairwise_join = pairwise_join

    def __call__(
            self,
            callsets: Iterable[CallsetDefType],
            retain_index: bool = False
    ) -> pl.DataFrame:
        """
        Intersect callsets.

        :param callsets: Callsets to intersect.
        :param pre_check_schema: If True, check the schema on all input tables before intersecting.
            this catches errors early and produces better error messages, but can be expensive if
            a lazy frame was input with non-trivial transformations. Should be True for most
            input.

        :return: A merged callset table.
        """
        callsets = self.get_intersect_tuples(callsets, retain_index)

        if len(callsets) == 0:
            raise ValueError('No callsets to intersect.')

        # Check required columns
        required_cols = self.pairwise_join.required_cols | {'chrom', 'pos', 'end', 'id'}
        all_col_dict = {}

        for df_next, src_name, src_index in callsets:
            if missing_cols := required_cols - set(df_next.collect_schema().names()):
                raise ValueError(f'Missing columns for source ({next_name}, index {next_index}): "{", ".join(sorted(missing_cols))}"')

            for col, dtype in df_next.collect_schema().items():
                if col not in all_col_dict:
                    all_col_dict[col] = dtype
                # elif all_col_dict[col] != dtype:
                #     raise ValueError(f'Column {col} has different types in different sources: {all_col_dict[col]} vs {dtype}')

        required_cols = [col for col in all_col_dict.keys() if col in required_cols]  # To list in order found in tables

        # Initialize cumulative table
        df_next, src_name, src_index = callsets[0]

        merge_stat_cols = {  # Empty join, gets names of columns this join will return
            col: dtype for col, dtype in self.pairwise_join.join(df_next.head(0).drop('_index'), df_next.head(0).drop('_index')).collect_schema().items()
                if col not in {'index_a', 'index_b', 'id_a', 'id_b'}
        }

        # Cumulative merge
        df_cumulative = None

        for df_next, src_name, src_index in callsets:
            if df_cumulative is None:
                df_cumulative = (
                    _init_cumulative(df_next, src_index, src_name, required_cols, merge_stat_cols)
                    .sort('chrom', 'pos', 'end', 'id')
                    .collect()
                    .lazy()
                )

                continue

            # Intersect
            df_join = (
                (
                    self.pairwise_join.join(df_cumulative, df_next.drop('_index'))
                    .sort('weight', descending=False)
                    .unique('index_a', keep='first')
                    .unique('index_b', keep='first')
                    .sort('index_a', 'index_b')
                )
                .join(
                    df_next.select('_index', 'pos', 'id'),
                    left_on='index_b', right_on='_index',
                    how='left'
                )
                .collect()
            )

            # Update cumulative in three parts:
            # 1) Match: Update existing cumulative records for matches
            # 2) No-match: Keep cumulative records that were not updated
            # 3) New: New records in this callset

            df_match = (
                _get_match(
                    df_cumulative, df_next, df_join, src_index, src_name, merge_stat_cols
                )
            )

            df_nomatch = (  # Gather records that were not updated
                df_cumulative
                .join(
                    df_join.lazy(),
                    left_on='_mg_index', right_on='index_a', how='anti'
                )
                .drop('_mg_index')
            )

            df_new = (
                _init_cumulative(
                    (
                        df_next
                        .join(
                            df_join.lazy(),
                            left_on='_index', right_on='index_b', how='anti'
                        )
                    ),
                    src_index, src_name, required_cols, merge_stat_cols
                )
                .drop('_mg_index')
            )

            df_cumulative = (
                pl.concat([df_match, df_nomatch, df_new])
                .sort('chrom', 'pos', 'end')
                .with_row_index('_mg_index')
                .collect()
                .lazy()
            )

        # Subset to merge columns (drop columns needed for join)
        df_cumulative = (
            df_cumulative
            .select(cs.starts_with('_mg_').exclude('_mg_index'))
            .rename(lambda col: col.removeprefix('_'))
        )

        # Choose lead variant
        if lead_strategy is LeadStrategy.LEFT:
            src_index_expr = pl.col('mg_src_pos').list.arg_min()
        elif lead_strategy is LeadStrategy.FIRST:
            src_index_expr = pl.lit(0)
        else:
            raise ValueError(f'Unknown lead_strategy: {lead_strategy!r}')

        df_cumulative = (
            df_cumulative.with_columns(
                pl.col('mg_src').list.get(src_index_expr).alias('mg_src_lead'),
                pl.col('mg_src_index').list.get(src_index_expr).alias('mg_src_lead_index'),
            )
            .with_row_index('_mg_index')
            .collect()
        )

        # Finalize, copy variant columns from the
        lead_list = []

        mg_cols = [col for col in df_cumulative.columns if not col.startswith('_')]
        table_cols = [col for col in all_col_dict.keys() if col not in mg_cols and not col.startswith('_')]
        col_order = table_cols + mg_cols

        for df_next, src_name, src_index in callsets:
            df_next_cols = set(df_next.collect_schema().names())

            df_next = (
                df_next
                .join(
                    (
                        df_cumulative
                        .lazy()
                        .filter(pl.col('mg_src_lead') == src_index)
                        .select('_mg_index', 'mg_src_lead_index')
                    ),
                    left_on='_index', right_on='mg_src_lead_index', how='inner'
                )
                .drop('_index')
            )

            for col in set(table_cols) - df_next_cols:
                df_next = df_next.with_columns(pl.lit(None).cast(all_col_dict[col]).alias(col))

            lead_list.append(df_next)

        return (
            pl.concat(lead_list)
            .join(
                (
                    df_cumulative
                    .lazy()
                    .with_columns(pl.col('mg_src_lead_index').alias('_mg_index'))  # Copy so "mg_src_lead_index" is not lost
                ),
                on='_mg_index', how='inner'
            )
            .drop('_mg_index')
            .select(col_order)
            .sort('chrom', 'pos', 'end', 'id')
        )
