"""A merging strategy that adds callsets cumulatively to the merge.

This strategy uses a table of variants that accumulates as callsets are added (the cumulative
table). The cumulative table is initially empty. As callsets are added, variants intersecting the
cumulative table are added to existing entries, and variants not intersecting the cumulative table
are appended as new variants. This process is repeated for each callset added.

After all callsets are processed, the cumulative table represents a nonredundant callset where
each entry is one variant that was found in one or more of the original callsets. Columns
tracking the sources and the variant within each source.

This strategy is fast and uses minimal memory, but is not necessarily optimal. The order variants
are input into the callset may alter the merged results in nontrivial ways, especially in loci
where multiple join choices are possible.

Internally, source tracking is held in a long-form sidecar table (``df_sources``) that grows by
row-append per iteration. The public list columns (``mg_src``, ``mg_stat``) and the ``mg_src_lead``
index are materialized once at finalization via a single ``group_by(_mg_index).agg(...)``. Each
``mg_src`` entry carries ``src_index``, ``src_name``, ``src_meta``, ``var_index``, and ``var_id``;
``mg_src_lead`` is the position within ``mg_src`` of the lead (representative) source entry.
The cumulative variant table (``df_cumulative``) keeps chrom-sorted ordering across iterations via
``merge_sorted(key='chrom')`` to avoid the per-iteration O(N log N) global sort.
"""

__all__ = [
    'LeadStrategy',
    'MergeCumulative',
]

from collections.abc import Iterable
from enum import Enum
from pathlib import Path
from typing import Optional, Any

import polars as pl

from ..pairwise.base import PairwiseJoin
from ..meta.decorators import immutable
from ..util.var import id_version_expr

from .base import (
    CallsetDefInputType,
    CallsetDef,
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


def _ensure_chrom_sorted(
        df: pl.DataFrame | pl.LazyFrame
) -> pl.DataFrame:
    """Ensure ``df`` is sorted on ``chrom`` with the ``SORTED_ASC`` flag set.

    ``merge_sorted`` silently produces wrong output when either side isn't sorted on the key,
    so this function is the choke point that enforces the invariant. If the flag is already
    set (e.g. ``df`` came from an upstream sort or anti-join that preserved order), the call
    is a no-op aside from a flag check.

    :param df: A DataFrame containing a ``chrom`` column.

    :return: ``df`` itself if already chrom-sorted, otherwise a sorted copy.
    """

    if isinstance(df, pl.LazyFrame):
        df = df.collect()

    if df.height == 0 or df['chrom'].flags.get('SORTED_ASC', False):
        return df

    return df.sort('chrom').set_sorted('chrom')


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
        self.lead_strategy = lead_strategy

    def __call__(
            self,
            callsets: Iterable[CallsetDefInputType],
            retain_index: bool = False,
            pre_filter: Optional[Iterable[pl.Expr] | pl.Expr] = None,
            sort: bool = True,
            add_id: bool = True,
            temp_dir: bool | str | Path = False,
    ) -> pl.LazyFrame:
        """
        Intersect callsets.

        :param callsets: Callsets to intersect.
        :param retain_index: If `True`, do not drop an existing "_index" column in callset tables
            if they exist.
        :param pre_filter: If set, filter each table with these expressions. Filter is applied
            last (after "_index" is set).
        :param temp_dir: Forwarded to the pairwise intersect for each cumulative step. See
            :meth:`agglovar.pairwise.base.PairwiseJoin.join_iter`.

        :return: A merged callset table.
        """
        callsets: list[CallsetDef] = self.get_intersect_tuples(callsets, retain_index, pre_filter)

        if len(callsets) == 0:
            raise ValueError('No callsets to intersect.')

        # Column inventory across all callsets (preserves first-seen order, captures dtypes).
        required_cols = self.pairwise_join.required_cols | {'chrom', 'pos', 'end', 'id'}
        all_col_dict: dict[str, pl.DataType] = {}

        for callset_def in callsets:
            schema = callset_def.table.collect_schema()

            if missing_cols := required_cols - set(schema.names()):
                raise ValueError(
                    f'Missing columns for source {callset_def}: '
                    f'"{", ".join(sorted(missing_cols))}"'
                )

            for col, dtype in schema.items():
                if col not in all_col_dict:
                    all_col_dict[col] = dtype

        # Columns carried on the cumulative table — preserve discovery order. Includes the
        # pairwise join's required columns plus "id" (needed for the cumulative sort and join
        # reporting), via the local "required_cols" superset.
        pairwise_cols = [
            col for col in all_col_dict.keys() if col in required_cols
        ]

        # Materialize each callset once (chrom-sorted). Source LazyFrames are not
        # re-evaluated in the merge loop or the final lead-extraction pass.
        callsets_mat: list[dict[str, Any]] = [
            {
                'table': _ensure_chrom_sorted(callset_def.table),
                'name': callset_def.name,
                'metadata': callset_def.metadata,
            } for callset_def in callsets
        ]

        # Derive merge_stat_cols (the per-pair stats schema from the pairwise join) once, using
        # an empty pairwise call. Polars short-circuits on empty inputs so this is cheap.
        first_df = callsets_mat[0]['table']

        merge_stat_cols: dict[str, pl.DataType] = {
            col: dtype
            for col, dtype in (
                self.pairwise_join.join(first_df.head(0).lazy(), first_df.head(0).lazy())
                .collect_schema()
                .items()
            )
            if col not in {'index_a', 'index_b', 'id_a', 'id_b'}
        }
        stat_struct_dtype = pl.Struct(merge_stat_cols)

        # Sources sidecar schema. Append-only across iterations; lists built once at finalization.
        sources_schema = pl.Schema({
            '_mg_index': pl.UInt64,
            '_mg_src_order': pl.Int32,
            'src_name': pl.String,
            'src_meta': pl.String,
            'var_index': pl.UInt32,
            'var_id': pl.String,
            'src_pos': pl.Int64,
            '_mg_stat': stat_struct_dtype,
        })

        # Initialise cumulative tables from the first callset.
        df_first = callsets_mat[0]['table']
        first_src_name = callsets_mat[0]['name']
        first_src_meta = callsets_mat[0]['metadata']
        first_src_index = 0

        df_cumulative = (
            df_first
            .select(*pairwise_cols)
            .with_columns(
                pl.int_range(0, pl.len(), dtype=pl.UInt64).alias('_mg_index')
            )
            .sort('chrom', 'pos', 'end', 'id')
            .with_columns(pl.col('chrom').set_sorted())
        )

        df_sources = df_first.select(
            pl.col('_mg_src_index').cast(pl.UInt32).alias('var_index'),
            pl.col('id').cast(pl.String).alias('var_id'),
            pl.col('pos').cast(pl.Int64).alias('src_pos'),
        ).with_columns(
            pl.int_range(0, pl.len(), dtype=pl.UInt64).alias('_mg_index'),
            pl.lit(first_src_index, dtype=pl.Int32).alias('_mg_src_order'),
            pl.lit(first_src_name, dtype=pl.String).alias('src_name'),
            pl.lit(first_src_meta, dtype=pl.String).alias('src_meta'),
            pl.lit(None, dtype=stat_struct_dtype).alias('_mg_stat'),
        ).select(*sources_schema.names())

        next_mg_index: int = df_first.height

        # Iterate remaining callsets.
        for iter_idx, callset_mat in enumerate(callsets_mat[1:], start=1):
            df_join = (
                self.pairwise_join.join(df_cumulative.lazy(), callset_mat['table'].lazy(), temp_dir=temp_dir)
                .sort('weight', descending=True)
                .unique('index_a', keep='first')
                .unique('index_b', keep='first')
                .collect()
            )

            # Resolve positional indices (item 4): lookup _mg_index from cumulative and
            # (_mg_src_index, id, pos) from df_next — single small in-memory join each.
            df_join_resolved = (
                df_join.lazy()
                .join(
                    df_cumulative.lazy()
                    .with_row_index('_idx_a_lookup')
                    .select('_idx_a_lookup', '_mg_index'),
                    left_on='index_a', right_on='_idx_a_lookup', how='inner',
                )
                .join(
                    callset_mat['table'].lazy()
                    .with_row_index('_idx_b_lookup')
                    .select(
                        '_idx_b_lookup',
                        '_mg_src_index',
                        pl.col('id').alias('_match_var_id'),
                        pl.col('pos').alias('_match_pos'),
                    ),
                    left_on='index_b', right_on='_idx_b_lookup', how='inner',
                )
                .collect()
            )

            # Append source rows for matched pairs (items 1+2: long-form append, no list mutation).
            stat_struct_expr = (
                pl.struct(*[
                    pl.col(col).cast(dtype).alias(col)
                    for col, dtype in merge_stat_cols.items()
                ])
                .alias('_mg_stat')
                if merge_stat_cols
                else pl.lit(None, dtype=stat_struct_dtype).alias('_mg_stat')
            )

            df_match_sources = df_join_resolved.select(
                pl.col('_mg_index').cast(pl.UInt64),
                pl.lit(iter_idx, dtype=pl.Int32).alias('_mg_src_order'),
                pl.lit(callset_mat['name'], dtype=pl.String).alias('src_name'),
                pl.lit(callset_mat['metadata'], dtype=pl.String).alias('src_meta'),
                pl.col('_mg_src_index').cast(pl.UInt32).alias('var_index'),
                pl.col('_match_var_id').cast(pl.String).alias('var_id'),
                pl.col('_match_pos').cast(pl.Int64).alias('src_pos'),
                stat_struct_expr,
            ).select(*sources_schema.names())

            # df_new: rows in df_next not matched (anti-join preserves left order → chrom-sorted).
            df_new = (
                callset_mat['table'].lazy()
                .join(
                    df_join_resolved.lazy().select('_mg_src_index'),
                    on='_mg_src_index', how='anti',
                )
                .collect()
            )

            n_new = df_new.height

            if n_new > 0:
                df_new_with_index = df_new.with_columns(
                    (pl.int_range(0, pl.len(), dtype=pl.UInt64) + next_mg_index).alias('_mg_index')
                )

                df_new_cumulative = df_new_with_index.select(*pairwise_cols, '_mg_index')

                df_new_sources = df_new_with_index.select(
                    pl.col('_mg_index'),
                    pl.lit(iter_idx, dtype=pl.Int32).alias('_mg_src_order'),
                    pl.lit(callset_mat['name'], dtype=pl.String).alias('src_name'),
                    pl.lit(callset_mat['metadata'], dtype=pl.String).alias('src_meta'),
                    pl.col('_mg_src_index').cast(pl.UInt32).alias('var_index'),
                    pl.col('id').cast(pl.String).alias('var_id'),
                    pl.col('pos').cast(pl.Int64).alias('src_pos'),
                    pl.lit(None, dtype=stat_struct_dtype).alias('_mg_stat'),
                ).select(*sources_schema.names())

                # Item 3: merge_sorted on chrom — O(N+M) vs full sort's O((N+M) log (N+M)).
                # Both sides must carry the SORTED_ASC flag, or merge_sorted will silently
                # produce an interleaved-but-unsorted result. _ensure_chrom_sorted is the
                # invariant choke point.
                df_cumulative = _ensure_chrom_sorted(df_cumulative)
                df_new_cumulative = _ensure_chrom_sorted(df_new_cumulative)

                _assert_chrom_sorted(df_cumulative, 'df_cumulative')
                _assert_chrom_sorted(df_new_cumulative, 'df_new_cumulative')

                df_cumulative = (
                    df_cumulative
                    .merge_sorted(df_new_cumulative, key='chrom')
                    .with_columns(pl.col('chrom').set_sorted())
                )

                df_sources = pl.concat([df_sources, df_match_sources, df_new_sources])
                next_mg_index += n_new
            else:
                df_sources = pl.concat([df_sources, df_match_sources])

        return self._finalize(
            df_sources=df_sources,
            callsets_mat=callsets_mat,
            all_col_dict=all_col_dict,
            sort=sort,
            add_id=add_id,
        )

    def _finalize(
            self,
            df_sources: pl.DataFrame,
            callsets_mat: list[dict[str, Any]],
            all_col_dict: dict[str, pl.DataType],
            sort: bool,
            add_id: bool,
    ) -> pl.LazyFrame:
        """Build the final merged callset from the long-form sources sidecar.

        Materialises the public ``mg_src``/``mg_stat`` list columns and the ``mg_src_lead``
        index (the position of the lead entry within ``mg_src``), then joins lead variant
        columns from the materialised callsets.
        """
        # "mg_src_lead" is the index into the "mg_src" list identifying the lead source entry.
        # It is computed FROM the materialised lists (not a separate argmin over the long-form
        # rows), so it cannot drift from "mg_src" regardless of how the list is ordered. A lead
        # strategy only chooses which list field the index minimises over.
        if self.lead_strategy is LeadStrategy.LEFT:
            # Earliest start position among the contributing sources.
            lead_index_expr = pl.col('mg_src_pos').list.arg_min()
        elif self.lead_strategy is LeadStrategy.FIRST:
            # Earliest source in input order (smallest src_index).
            lead_index_expr = (
                pl.col('mg_src')
                .list.eval(pl.element().struct.field('src_index'))
                .list.arg_min()
            )
        else:
            raise ValueError(f'Unknown lead_strategy: {self.lead_strategy!r}')

        df_agg = (
            df_sources
            .sort('_mg_index', '_mg_src_order')
            .group_by('_mg_index', maintain_order=True)
            .agg(
                # Per-source entries. "src_index" is the source index (0-based input order),
                # which is tracked as "_mg_src_order".
                pl.struct(
                    pl.col('_mg_src_order').alias('src_index'),
                    pl.col('src_name'),
                    pl.col('src_meta'),
                    pl.col('var_index'),
                    pl.col('var_id'),
                ).alias('mg_src'),
                pl.col('src_pos').alias('mg_src_pos'),
                pl.col('_mg_stat').drop_nulls().alias('mg_stat'),
            )
            .with_columns(
                lead_index_expr.cast(pl.UInt32).alias('mg_src_lead'),
            )
            .with_columns(
                # Lead-variant keys for column extraction, gathered from the same "mg_src" entry
                # "mg_src_lead" points at — guaranteed consistent with it.
                pl.col('mg_src').list.get(pl.col('mg_src_lead')).struct.field('src_index').alias('_lead_src_index'),
                pl.col('mg_src').list.get(pl.col('mg_src_lead')).struct.field('var_index').alias('_lead_var_index'),
            )
        )

        # filter column may be missing across all sources; the original code synthesised an
        # empty list to keep the column present, then dropped it at the end. Preserve that.
        drop_filter = 'filter' not in all_col_dict
        if drop_filter:
            all_col_dict['filter'] = pl.List(pl.String)

        mg_cols = ['mg_src', 'mg_stat', 'mg_src_lead']
        table_cols = [col for col in all_col_dict.keys() if col not in mg_cols and not col.startswith('_')]
        col_order = table_cols + mg_cols

        if drop_filter:
            col_order = [col for col in col_order if col != 'filter']

        # Lead-variant column extraction (item 6): read from already-materialised callsets.
        lead_list: list[pl.LazyFrame] = []

        for src_index, callset_mat in enumerate(callsets_mat):
            df_next_cols = set(callset_mat['table'].columns)

            df_next_lead = (
                callset_mat['table'].lazy()
                .join(
                    df_agg.lazy()
                    .filter(pl.col('_lead_src_index') == src_index)
                    .select(
                        '_mg_index',
                        pl.col('_lead_var_index').cast(pl.UInt32).alias('_mg_src_index'),
                    ),
                    on='_mg_src_index', how='inner',
                )
                .drop('_mg_src_index')
            )

            for col in set(table_cols) - df_next_cols:
                df_next_lead = df_next_lead.with_columns(
                    pl.lit(None).cast(all_col_dict[col]).alias(col)
                )

            df_next_lead = (
                df_next_lead
                .with_columns(pl.col('filter').fill_null([]))
                .select('_mg_index', *table_cols)
            )

            lead_list.append(df_next_lead)

        df_merge = (
            pl.concat(lead_list)
            .join(
                df_agg.lazy().select('_mg_index', 'mg_src', 'mg_src_lead', 'mg_stat'),
                on='_mg_index', how='inner',
            )
        )

        if add_id:
            df_merge = df_merge.with_columns(id_version_expr())

        if sort:
            df_merge = df_merge.sort('chrom', 'pos', 'end', 'id')

        drop_cols = ['_mg_index']
        if drop_filter:
            drop_cols.append('filter')

        return (
            df_merge
            .drop(drop_cols)
            .select(col_order)
        )


def _assert_chrom_sorted(df: pl.DataFrame, name: str) -> None:
    """Raise if ``df`` does not carry the ``chrom`` SORTED_ASC flag.

    ``merge_sorted`` does not validate that its inputs are actually sorted on the merge key —
    if either side has lost the flag (e.g. a join scrambled order without us noticing), the
    result is silently mis-interleaved. This assertion makes that failure mode loud.
    """
    if df.height == 0:
        return
    if not df['chrom'].flags.get('SORTED_ASC', False):
        raise AssertionError(
            f'{name}: chrom column missing SORTED_ASC flag — merge_sorted would silently '
            f'produce incorrect output. This indicates a transformation upstream stripped '
            f'the sorted invariant without re-establishing it via sort()/set_sorted().'
        )
