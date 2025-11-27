"""Pairwise overlap runner object."""

__all__ = [
    'PairwiseOverlap',
]

from collections.abc import (
    Iterable,
    Iterator,
    Mapping,
)
import functools
import operator
from typing import Optional
from warnings import warn

import polars as pl

from ... import schema
from ...meta.decorators import immutable
from ...meta.descriptors import (
    CheckedBool,
    CheckedObject,
    BoundedInt,
)
from ...seqmatch import MatchScoreModel

from ..base import PairwiseJoin
from ..weights import (
    WeightStrategy,
    DEFAULT_WEIGHT_STRATEGY,
)

from ._const import (
    AUTOGEN_COLS,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_JOIN_COLS,
    INVARIANT_JOIN_COLS,
    JOIN_COL_EXPR,
    RESERVED_COLS,
)

from ._stage import PairwiseOverlapStage

@immutable
class PairwiseOverlap(PairwiseJoin):
    """Pairwise overlap class.

    Join by overlapping variants by position.

    :ivar match_score_model: (Advanced) Configured model for scoring similarity between pairs of
        sequences. If `None` and `match_prop_min` is set, then a default aligner will be used.
    :ivar force_end_ro: (Advanced) By default, reciprocal overlap is calculated with the end
        position set to the start position plus the variant length. For all variants except
        insertions, this will typically match the end value in the source DataFrame. If `True`, the
        end position in the DataFrame is also used for reciprocal overlap without changes.
        Typically, this option should not be used and will break reciprocal overlap for insertions.
    :ivar chunk_size: (Advanced) Chunk df_a into partitions of this size, and for each chunk,
        subset df_b to include only variants that may overlap with variants in the chunk. If
        None, each chromosome is a single chunk, which will lead to a combinatorial explosion
        unless offset_max is greater than 0.
    """

    # Advanced Configuration Attributes
    match_score_model: MatchScoreModel = CheckedObject(default=MatchScoreModel())
    force_end_ro: bool = CheckedBool()
    chunk_size: Optional[int] = BoundedInt(0, default=DEFAULT_CHUNK_SIZE)

    # Join control
    stages: tuple[PairwiseOverlapStage, ...]
    join_cols: tuple[pl.Expr, ...]
    join_col_exprs: tuple[pl.Expr, ...]
    equi_join_exprs: tuple[pl.Expr, ...]

    # Special join columns
    compute_seg_ro: bool = CheckedBool(default=False)
    compute_weight: bool = CheckedBool(default=False)

    expected_cols: frozenset[str]

    def __init__(
            self,
            stages: Iterable[PairwiseOverlapStage],
            join_cols: Optional[Iterable[str | pl.Expr]] = None,
            drop_default_join_cols: bool = False,
            match_score_model: Optional[MatchScoreModel] = None,
            weight_strategy: WeightStrategy = DEFAULT_WEIGHT_STRATEGY,
            force_end_ro: bool = False,
            chunk_size: Optional[int] = None,
    ) -> None:
        super().__init__(weight_strategy)

        # Set join stages
        self.stages = tuple(stages)

        if not self.stages:
            raise ValueError(f'No defined overlap stages')

        # Set parameters
        if match_score_model is not None:
            self.match_score_model = match_score_model

        self.force_end_ro = force_end_ro

        # Chunking control
        self.chunk_size = chunk_size
        self._chunk_range = dict()

        # Match proportion expression
        self.expr_match_prop = (
            pl.struct('seq_a', 'seq_b')
            .map_elements(
                lambda s: self.match_score_model.match_prop(s['seq_a'], s['seq_b']),
                return_dtype=pl.Float32
            )
        ) if any(stage.has_match for stage in self.stages) else (
            pl.lit(None).cast(pl.Float32)
        )

        # Set join columns
        self._set_join_cols(
            join_cols=join_cols,
            drop_default_join_cols=drop_default_join_cols,
        )

        # Set expected columns
        self.expected_cols = self._get_expected_cols()

        # Create a tuple of common equi-join expressions. Speeds up join operations.
        equi_join_list = []

        if all(stage.offset_max == 0 for stage in self.stages):
            equi_join_list.append(pl.col('pos_a') == pl.col('pos_b'))
            equi_join_list.append(pl.col('end_a') == pl.col('end_b'))

        if all(stage.match_ref for stage in self.stages):
            equi_join_list.append(pl.col('ref_a') == pl.col('ref_b'))

        if all(stage.match_alt for stage in self.stages):
            equi_join_list.append(pl.col('alt_a') == pl.col('alt_b'))

        self.equi_join_exprs = tuple(equi_join_list)

    @property
    def required_cols(self) -> set[str]:
        """The minimum set of columns that must be present in input tables.

        This is set based on parameters needed to perform the join. For example, if sequence
        matching is required, then "seq" will be in this list, and if "seq" does not exist
        in both df_a and df_b, then an error is raised.
        """
        return set(self.expected_cols - AUTOGEN_COLS - RESERVED_COLS)

    @property
    def reserved_cols(self) -> set[str]:
        return set(RESERVED_COLS)

    @property
    def has_match(self) -> bool:
        return any(stage.has_match for stage in self.stages)

    @property
    def has_seg_ro(self) -> bool:
        return any(stage.has_seg_ro for stage in self.stages)

    @property
    def match_prop_expr(self):
        return (
            pl.struct('seq_a', 'seq_b')
            .map_elements(
                lambda s: self.match_score_model.match_prop(s['seq_a'], s['seq_b']),
                return_dtype=pl.Float32
            )
        ) if any(stage.has_match for stage in self.stages) else (
            pl.lit(None).cast(pl.Float32)
        )

    @property
    def has_equi_join(self) -> bool:
        """True if all stages have an equivalent equi-join expression."""
        return bool(self.equi_join_exprs)

    @property
    def is_equi_offset(self) -> bool:
        """True if all stages have an equivalent equi-join expression for position."""
        return all(stage.offset_max == 0 for stage in self.stages)

    @property
    def weight_expr(self):
        return self.weight_strategy.expr


    def join_iter(
            self,
            df_a: pl.DataFrame | pl.LazyFrame,
            df_b: pl.DataFrame | pl.LazyFrame,
    ) -> Iterator[pl.LazyFrame]:
        """Find all pairs of variants in two sources that meet a set of criteria.

        :param df_a: Source dataframe.
        :param df_b: Target dataframe.

        :yields: A LazyFrame for each chunk.
        """

        if self.is_equi_offset or self.chunk_size == 0:
            return self._join_iter_notchunked(df_a, df_b)

        return self._join_iter_chunked(df_a, df_b)

    def _join_iter_chunked(
            self,
            df_a: pl.DataFrame | pl.LazyFrame,
            df_b: pl.DataFrame | pl.LazyFrame,
    ) -> Iterator[pl.LazyFrame]:
        """Find all pairs of variants in two sources that meet a set of criteria.

        :param df_a: Source dataframe.
        :param df_b: Target dataframe.

        :yields: A LazyFrame for each chunk.
        """

        # print('_join_iter_chunked(): Starting')

        join_empty = True  # Detects if no joins were written

        chunk_range = self._get_chunk_range()
        chunk_size = self.chunk_size if self.chunk_size is not None else DEFAULT_CHUNK_SIZE

        # Prepare tables
        df_a, df_b = self._prepare_tables(df_a, df_b, warn_on_reserved=True)

        for chrom, last_index_a in (
            df_a
            .group_by('chrom_a')
            .agg(pl.len().alias('last_index'))
            .sort('chrom_a')
        ).collect().rows():
            start_index_a = 0

            df_a_chrom = (
                df_a.filter(pl.col('chrom_a') == chrom)
                .with_row_index('_index_chrom_a')
            )

            while start_index_a < last_index_a:
                end_index_a = start_index_a + chunk_size

                # print(f'Chunk A: {chrom}: {start_index_a} - {end_index_a}', flush=True)

                chunk_list = []

                df_a_chunk = df_a_chrom.filter(
                    pl.col('_index_chrom_a') >= start_index_a,
                    pl.col('_index_chrom_a') < end_index_a
                )

                df_b_chunk = (
                    self._chunk_relative(df_a_chunk, df_b, chrom, chunk_range)
                    .with_row_index('_index_chunk_b')
                )

                start_index_b = 0
                last_index_b = df_b_chunk.select(pl.col('_index_chunk_b').max() + 1).collect().item()

                if last_index_b is None:
                    start_index_a = end_index_a
                    continue

                while start_index_b < last_index_b:
                    end_index_b = start_index_b + chunk_size

                    # print(f'* Chunk B: {start_index_b} - {end_index_b}', flush=True)

                    chunk_list.append(
                        self._join_pairwise(
                            df_a_chunk,
                            df_b_chunk.filter(
                                pl.col('_index_chunk_b') >= start_index_b,
                                pl.col('_index_chunk_b') < end_index_b
                            )
                        )
                    )

                    start_index_b = end_index_b

                if chunk_list:
                    yield (
                        pl.concat(chunk_list)
                        .collect()
                        .lazy()
                    )

                    join_empty = False

                start_index_a = end_index_a

        if join_empty:
            # If no join tables were yielded, yield an empty one. This creates an empty join table
            # with the correct structure and prevents pl.concat from failing on an empty list.
            yield self._join_pairwise(df_a.head(0), df_b.head(0))

    def _join_iter_notchunked(
            self,
            df_a: pl.DataFrame | pl.LazyFrame,
            df_b: pl.DataFrame | pl.LazyFrame,
    ) -> Iterator[pl.LazyFrame]:
        """Find all pairs of variants in two sources that meet a set of criteria.

        :param df_a: Source dataframe.
        :param df_b: Target dataframe.

        :yields: A LazyFrame for each chunk.
        """
        # print('_join_iter_notchunked(): Starting')

        join_empty = True  # Detects if no joins were written

        # Prepare tables
        df_a, df_b = self._prepare_tables(df_a, df_b, warn_on_reserved=True)

        chrom_list = sorted(
            set(df_a.select('chrom_a').unique().collect().to_series().to_list())
            | set(df_b.select('chrom_b').unique().collect().to_series().to_list())
        )

        for chrom in chrom_list:
            yield self._join_pairwise(
                df_a.filter(pl.col('chrom_a') == chrom),
                df_b.filter(pl.col('chrom_b') == chrom),
            )

            join_empty = False

        if join_empty:
            # If no join tables were yielded, yield an empty one. This creates an empty join table
            # with the correct structure and prevents pl.concat from failing on an empty list.
            yield self._join_pairwise(df_a.head(0), df_b.head(0))

    def _join_pairwise(
            self,
            df_a: pl.LazyFrame,
            df_b: pl.LazyFrame,
    ) -> pl.LazyFrame:
        """Non-equi-join (pos and end not the same).

        Assumes both tables are filtered to the same chromosome.
        """
        df_join_head = (
            df_a
            .join(
                df_b,
                how='cross'
            )
            .filter(
                *self.equi_join_exprs,
            )
        )

        join_list = []

        for stage in self.stages:
            df_join = (
                df_join_head
                .filter(
                    *stage.join_predicates,
                )
                .select(
                    *self.join_col_exprs,
                )
            )

            if self.compute_weight:
                df_join = df_join.with_columns(
                    self.weight_expr.alias('weight'),
                )

            if self.compute_seg_ro:
                df_join = self._seg_ro(df_join, df_a, df_b)

            df_join = (
                df_join
                .filter(*stage.join_filters)
                .select(*self.join_cols)
                # .sort(['index_a', 'index_b'])
            )

            join_list.append(df_join)

        return (
            pl.concat(join_list)
            .group_by('index_a', 'index_b')
            .agg(
                pl.all().get(pl.col('weight').fill_null(0.0).arg_max())
            )
            # .sort('index_a', 'index_b')
        )

    def _get_chunk_range(self) -> dict[tuple[str, str], list[pl.Expr]]:
        """Get a dict of chunk ranges including each column and a limit."""
        chunk_range: dict[tuple[str, str], list[pl.Expr]] = {}

        for stage in self.stages:
            for col, limit, exprs in stage.chunk_range:
                if (col, limit) not in chunk_range:
                    chunk_range[(col, limit)] = []

                chunk_range[(col, limit)].extend(exprs)

        return chunk_range


    def _prepare_tables(
            self,
            df_a: pl.DataFrame | pl.LazyFrame,
            df_b: pl.DataFrame | pl.LazyFrame,
            warn_on_reserved: bool = False
    ) -> tuple[pl.LazyFrame, pl.LazyFrame]:
        """Prepares tables for join.

        Checks for expected columns and formats, adds missing columns as needed, and
        appends "_a" and "_b" suffixes to column names.

        :param df_a: Table A.
        :param df_b: Table B.
        :param warn_on_reserved: If True, generate a warning if reserved columns are found and
            drop them. If false, raise an error.

        :returns: Tuple of normalized tables (df_a, df_b).

        :raises ValueError: If missing or malform columns are detected.
        :raises TypeError: If input is not a DataFrame or LazyFrame.
        """
        # Check input types
        if isinstance(df_a, pl.DataFrame):
            df_a = df_a.lazy()
        elif not isinstance(df_a, pl.LazyFrame):
            raise TypeError(f'Variant source: Expected DataFrame or LazyFrame, got {type(df_a)}')

        if isinstance(df_b, pl.DataFrame):
            df_b = df_b.lazy()
        elif not isinstance(df_b, pl.LazyFrame):
            raise TypeError(f'Variant target: Expected DataFrame or LazyFrame, got {type(df_b)}')

        # Check for expected columns
        columns_a = set(df_a.collect_schema().names())
        columns_b = set(df_b.collect_schema().names())

        missing_cols_a = sorted(self.check_required_cols(columns_a))
        missing_cols_b = sorted(self.check_required_cols(columns_b))

        if missing_cols_a or missing_cols_b:
            if missing_cols_a == missing_cols_b:
                raise ValueError(f'DataFrames "A" and "B" are missing expected column(s): {", ".join(missing_cols_a)}')

            raise ValueError(
                f'DataFrame "A" missing expected column(s): {", ".join(missing_cols_a)}; '
                f'DataFrame "B" missing expected column(s): {", ".join(missing_cols_b)}'
            )

        # Drop reserved columns
        if reserved_cols := sorted(self.check_reserved_cols(columns_a)):
            err_str = f'Reserved columns in table "A": {", ".join(reserved_cols)}'

            if warn_on_reserved:
                warn(f'{err_str}: Dropping column(s)')
            else:
                raise ValueError(err_str)

            df_a = df_a.drop(reserved_cols, strict=False)

        if reserved_cols := sorted(self.check_reserved_cols(columns_b)):
            err_str = f'Reserved columns in table "B": {", ".join(reserved_cols)}'

            if warn_on_reserved:
                warn(f'{err_str}: Dropping column(s)')
            else:
                raise ValueError(err_str)

            df_b = df_b.drop(reserved_cols, strict=False)

        # Cast columns
        try:
            df_a = df_a.cast({col: schema.VARIANT[col] for col in columns_a if col in schema.VARIANT.keys()})
        except pl.exceptions.InvalidOperationError as e:
            raise ValueError(f'Unexpected columns types encountered in DataFrame "A": {e}') from e

        try:
            df_b = df_b.cast({col: schema.VARIANT[col] for col in columns_b if col in schema.VARIANT.keys()})
        except pl.exceptions.InvalidOperationError as e:
            raise ValueError(f'Unexpected columns types encountered in DataFrame "B": {e}') from e

        # Set and prepare varlen
        if 'varlen' not in columns_a:
            df_a = (
                df_a
                .with_columns(
                    (pl.col('end') - pl.col('pos'))
                    .cast(schema.VARIANT['varlen'])
                    .alias('varlen')
                )
            )

        if 'varlen' not in columns_b:
            df_b = (
                df_b
                .with_columns(
                    (pl.col('end') - pl.col('pos'))
                    .cast(schema.VARIANT['varlen'])
                    .alias('varlen')
                )
            )

        # Ensure positive values
        df_a = df_a.with_columns(
            pl.col('varlen')
            .cast(schema.VARIANT['varlen'])
            .abs()
        )

        df_b = df_b.with_columns(
            pl.col('varlen')
            .cast(schema.VARIANT['varlen'])
            .abs()
        )

        # Set index
        df_a = (
            df_a
            .with_row_index('_index')
        )

        df_b = (
            df_b
            .with_row_index('_index')
        )

        # Set ID
        if 'id' not in columns_a:
            df_a = df_a.with_columns(
                pl.lit(None).alias('id').cast(schema.VARIANT['id'])
            )

        if 'id' not in columns_b:
            df_b = df_b.with_columns(
                pl.lit(None).alias('id').cast(schema.VARIANT['id'])
            )

        # Prepare REF & ALT
        if 'ref' in self.required_cols:
            df_a = df_a.with_columns(
                pl.col('ref').str.to_uppercase().str.strip_chars()
            )

            df_b = df_b.with_columns(
                pl.col('ref').str.to_uppercase().str.strip_chars()
            )

        if 'alt' in self.required_cols:
            df_a = df_a.with_columns(
                pl.col('alt').str.to_uppercase().str.strip_chars()
            )

            df_b = df_b.with_columns(
                pl.col('alt').str.to_uppercase().str.strip_chars()
            )

        # Get END for RO
        if not self.force_end_ro:
            df_a = df_a.with_columns(
                (pl.col('pos') + pl.col('varlen')).alias('_end_ro')
            )

            df_b = df_b.with_columns(
                (pl.col('pos') + pl.col('varlen')).alias('_end_ro')
            )

        else:
            df_a = df_a.with_columns(
                pl.col('end').alias('_end_ro')
            )

            df_b = df_b.with_columns(
                pl.col('end').alias('_end_ro')
            )

        # Append suffixes to all columns
        df_a = df_a.select(pl.all().name.suffix('_a'))
        df_b = df_b.select(pl.all().name.suffix('_b'))

        return df_a, df_b

    def _set_join_cols(
            self,
            join_cols: Optional[Iterable[str | pl.Expr]] = None,
            drop_default_join_cols: bool = False,
    ) -> None:
        """Set join columns.

        :param join_cols: Join columns to include (expressions or names).
        :param drop_default_join_cols: Drop default join columns if True.
        """

        join_expr_map: dict[str, Optional[pl.Expr]] = {}

        for col in (
                INVARIANT_JOIN_COLS
                + (DEFAULT_JOIN_COLS if not drop_default_join_cols else [])
                + (list(join_cols) if join_cols else [])
        ):
            self._append_join_cols(col, join_expr_map)

        join_col_list = list(
            pl.col(col) for col in join_expr_map.keys()
        )

        for stage in self.stages:  # Add join columns used by stages
            for join_filter in stage.join_filters:
                for col in join_filter.meta.root_names():
                    self._append_join_cols(col, join_expr_map, True)

        join_col_set = functools.reduce(
            operator.or_,
            (
                set(expr.meta.root_names())
                for stage in self.stages
                for expr in stage.join_filters
            ),
            set()
        )

        if 'seg_ro' in join_col_set and join_expr_map['seg_ro'] is None:
            self.compute_seg_ro = True

        if 'weight' in join_expr_map.keys() and join_expr_map['weight'] is None:
            self.compute_weight = True

        self._append_join_cols(join_col_set, join_expr_map, True)

        self.join_cols = tuple(join_col_list)
        self.join_col_exprs = tuple(expr for expr in join_expr_map.values() if expr is not None)

    def _append_join_cols(
            self,
            exprs: Iterable[pl.Expr | str] | pl.Expr | str,
            join_expr_map: Mapping[str, Optional[pl.Expr]],
            no_replace: bool = False,
    ) -> None:
        """Append expressions to the list of columns included in the join table.

        These columns will be appended to the standard join table columns. Each expression
        should name the column it creates using ".alias()" if necessary. These columns do
        not affect the join itself, just the columns that appear in the join table.

        For example, to retain the "pos" column from df_a and df_b, then append
        "pl.col('pos_a')" and "pl.col('pos_b')". If you wanted to set a flag for whether
        the variant in df_a comes before df_b, then a new columns could be added:
        "(pl.col('pos_a') <= pl.col('pos_b')).alias('left_a')"

        :param exprs: A join column or a list of join columns where each may be defined as a string
            (known column name or a field already in the join table) or a Polars expression.
        :param join_col_exprs: A dictionary to which the expressions will be added.
        :param no_replace: Do not replace existing join column expressions if True.
        """
        if isinstance(exprs, (pl.Expr, str)):
            exprs = [exprs]

        for expr in exprs:
            if isinstance(expr, str):
                col_name = expr

                if col_name in JOIN_COL_EXPR:
                    col_expr = JOIN_COL_EXPR[col_name].alias(col_name)

                elif col_name == 'match_prop':
                    col_expr = self.match_prop_expr.alias(col_name)

                elif col_name == 'weight':
                    col_expr = None  # Set later

                elif col_name == 'seg_ro':
                    col_expr = None  # Ignore, cannot be set as a column expression, handled separately

                else:
                    raise ValueError(
                        f'Join column "{col_name} is not a known column name in '
                        f'"{", ".join(JOIN_COL_EXPR.keys())}": '
                        f'Custom columns must be Polars expressions over columns ending with "_a" and "_b" '
                        f'(for source table) and an "alias" to the desired column name.'
                    )
            else:
                col_name = expr.meta.output_name()
                col_expr = expr

            if no_replace and col_name in join_expr_map:
                continue

            join_expr_map[col_name] = col_expr

    def _get_expected_cols(self) -> frozenset[str]:
        """Extract expected columns from all stages and join columns.

        Call after join columns are set.

        :return: A set of join column names.
        """
        col_set: set[str] = {'chrom', 'pos', 'end'}

        stage_i = 0

        for stage in self.stages:
            stage_i += 1

            expr_i = 0
            for expr in stage.join_predicates:
                expr_i += 1
                for col in expr.meta.root_names():
                    col_set.add(_check_expected_col(col, 'join_predicates', stage_i, expr_i))

            expr_i = 0
            for col, bound, exprs in stage.chunk_range:
                expr_i += 1
                col_set.add(_check_expected_col(col, 'chunk_range', stage_i, expr_i))

        if self.has_seg_ro:
            col_set.add(_check_expected_col('seg_a', 'has_seg_ro', None, None))
            col_set.add(_check_expected_col('seg_b', 'has_seg_ro', None, None))
            col_set.add(_check_expected_col('qry_end_a', 'has_seg_ro', None, None))
            col_set.add(_check_expected_col('qry_end_b', 'has_seg_ro', None, None))
            col_set.add(_check_expected_col('qry_pos_a', 'has_seg_ro', None, None))
            col_set.add(_check_expected_col('qry_pos_b', 'has_seg_ro', None, None))

        for col in self.expr_match_prop.meta.root_names():
            col_set.add(_check_expected_col(col, 'expr_match_prop', None, None))

        expr_i = 0
        for expr in self.join_col_exprs:
            expr_i += 1

            for col in expr.meta.root_names():
                col_set.add(_check_expected_col(col, 'join_cols', None, expr_i))

        return frozenset(col_set - RESERVED_COLS)

    def _chunk_relative(
            self,
            df_a: pl.LazyFrame,
            df_b: pl.LazyFrame,
            chrom: str,
            chunk_range: dict[tuple[str, str], list[pl.Expr]],
    ) -> pl.LazyFrame:
        """Chunk one DataFrame relative to another.

        Chunk df_b relative to df_a choosing records in df_b that could possibly be joined with some record in df_a.
        For example, this function may determine the minimum and maximum values of pos and end, and then subset df_b
        by those values. The actual subset values are determined by the `chunk_range` attribute.

        `chunk_range` is a dictionary with keys formatted as ('column', 'limit') where "column" is a column name and
        "limit" is "min" or "max". Each value is a list of expressions to be applied to df_a, which will then determine
        the minimum or maximum value to be applied.

        For example, if ('pos', 'min') is a key in chunk_range, then chunk_range['pos', 'min'] is a list of expressions.
        In this example, assume it is a list with the single expression "pl.col('pos_a') - pl.col('varlen_a')"). For
        each record in df_a, the expression will compute the position minus the variant length producing a single value
        for each record. Since this is a minimum value, the minimum of these values (one per record in df_a) will
        be used to filter records in df_b by excluding any records with "pos_b" less than this minimum.

        The flexibility of this function is needed to support different limits. For example, when reciprocal overlap is
        used as a limit, the maximum value of pos_b is based on the maximum value of end_a (i.e. "chunk_range['pos',
        'max']" will contain "pl.col('end_a')" because if pos_b greater than any "end_a", then variants cannot overlap.

        :param df_a: Table chunk.
        :param df_b: Table to be chunked to records that may overlap with df_a.
        :param chrom: Chromosome name.

        :returns: df_b partitioned (LazyFrame).
        """
        filter_list = [
            pl.col('chrom_b') == chrom
        ]

        for (col_name, limit), expr_list in chunk_range.items():
            if limit == 'min':
                filter_list.append(
                    pl.col(col_name) >= (
                        df_a
                        .select(pl.min_horizontal(*expr_list))
                        .collect()
                        .to_series()
                        .min()
                    )
                )

            elif limit == 'max':
                filter_list.append(
                    pl.col(col_name) <= (
                        df_a
                        .select(pl.max_horizontal(*expr_list))
                        .collect()
                        .to_series()
                        .max()
                    )
                )

            else:
                raise ValueError(f'Unknown limit: "{limit}"')

        return df_b.filter(*filter_list)

    def _seg_ro(
            self,
            df_join: pl.LazyFrame,
            df_a: pl.LazyFrame,
            df_b: pl.LazyFrame,
    ) -> pl.LazyFrame:
        """Compute segment RO.

        :param df_join: Join table without segment RO.
        :param df_a: Table A.
        :param df_b: Table B.

        :return: Join table with segment RO.
        """
        df_seg_ro = (
            df_join
            .select('index_a', 'index_b')
            .join(
                (
                    df_a
                    .with_columns(
                        # Query bases aligned in segments
                        pl.col('seg_a').list.eval(
                            (
                                pl.element().struct.field('qry_end')
                                - pl.element().struct.field('qry_pos')
                            ).abs()
                        )
                        .list.sum()
                        .alias('seg_qry_len_a')
                    )
                    .explode('seg_a')
                    .with_row_index('_seg_a_index')
                ),
                left_on='index_a', right_on='_index_a', how='inner'
            )
            .join(
                (
                    df_b
                    .with_columns(
                        # Query bases aligned in segments
                        pl.col('seg_b').list.eval(
                            (
                                pl.element().struct.field('qry_end')
                                - pl.element().struct.field('qry_pos')
                            ).abs()
                        )
                        .list.sum()
                        .alias('seg_qry_len_b')
                    )
                    .explode('seg_b')
                    .with_row_index('_seg_b_index')
                ),
                left_on='index_b', right_on='_index_b', how='inner'
            )
            .with_columns(
                (
                    # Overlapping bases (reference)
                    (
                        pl.min_horizontal(
                            pl.col('seg_a').struct.field('end'), pl.col('seg_b').struct.field('end')
                        )
                        - pl.max_horizontal(
                            pl.col('seg_a').struct.field('pos'), pl.col('seg_b').struct.field('pos')
                        )
                    )
                    / (  # Reference bp to query bp
                        pl.col('seg_a').struct.field('end')
                        - pl.col('seg_a').struct.field('pos')
                    ).abs()
                    * (
                        pl.col('seg_a').struct.field('qry_end')
                        - pl.col('seg_a').struct.field('qry_pos')
                    ).abs()
                )
                .fill_null(0.0)
                .clip(0.0)
                .cast(pl.Float32)
                .alias('seg_ro_len')
            )
            .group_by('_seg_a_index', '_seg_b_index')
            .agg(
                # Resolve cross-join among segments, choose the best pairs
                pl.all().get(pl.col('seg_ro_len').arg_max()),
            )
            .group_by('index_a', 'index_b')
            .agg(
                (
                    # Compute segment RO...
                    (
                        # Sum of overlapping segments in query bp (estimated, above)
                        pl.col('seg_ro_len').sum()

                        # Unaligned segment bp
                        + pl.min_horizontal(
                            (pl.col('qry_end_a').first() - pl.col('qry_pos_a').first()).abs()
                            - pl.col('seg_qry_len_a').first(),
                            (pl.col('qry_end_b').first() - pl.col('qry_pos_b').first()).abs()
                            - pl.col('seg_qry_len_b').first()
                        )
                    )

                    # Divide by max length for RO
                    / pl.max_horizontal(
                        (pl.col('qry_end_a').first() - pl.col('qry_pos_a').first()).abs(),
                        (pl.col('qry_end_b').first() - pl.col('qry_pos_b').first()).abs(),
                    )
                )
                .clip(0.0, 1.0)
                .cast(pl.Float32)
                .alias('seg_ro')
            )
        )

        return (
            df_join
            .join(
                df_seg_ro,
                on=['index_a', 'index_b'],
                how='left'
            )
            .with_columns(
                pl.col('seg_ro').fill_null(1.0)
            )
        )

    @classmethod
    def from_definiton(
            cls,
            overlap_def,
    ):
        keys: Optional[set[str]] = None

        # Try keys
        try:
            keys = set(overlap_def.keys())

            if 'stages' not in keys:
                raise ValueError('Missing "stages" key in overlap definition')
        except AttributeError:
            pass

        if keys is None:
            try:
                stage_list = list(overlap_def)
            except TypeError:
                raise ValueError(
                    f'Definition must must be a dict-like with keys or a list-like (or itera ble) '
                    f'of stage definitions: {type(overlap_def)}'
                )

            overlap_def = {
                'stages': stage_list,
            }

        # Parse arguments
        parsed_def = {}

        for key, value in overlap_def.items():
            if key == 'stages':
                try:
                    parsed_def['stages'] = [
                        PairwiseOverlapStage(**stage_def)
                            if not isinstance(stage_def, PairwiseOverlapStage) else stage_def
                        for stage_def in value
                    ]
                except Exception as e:
                    raise ValueError(f'Error creating stages: {e}') from e

            elif key == 'join_cols':
                try:
                    parsed_def['join_cols'] = list(value)
                except Exception as e:
                    raise ValueError(f'Error creating join columns: {e}') from e

            elif key == 'drop_default_join_cols':
                try:
                    parsed_def['drop_default_join_cols'] = bool(value)
                except Exception as e:
                    raise ValueError(f'Error creating drop_default_join_cols: {e}') from e

            elif key == 'match_score_model':
                try:
                    parsed_def['match_score_model'] = (
                        MatchScoreModel(*value)
                            if not isinstance(value, MatchScoreModel) else value
                    )
                except Exception as e:
                    raise ValueError(f'Error creating match_score_model: {e}') from e

            elif key == 'weight_strategy':
                try:
                    parsed_def['weight_strategy'] = (
                        WeightStrategy(*value)
                            if not isinstance(value, WeightStrategy) else value
                    )
                except Exception as e:
                    raise ValueError(f'Error creating weight_strategy: {e}') from e

            elif key == 'force_end_ro':
                try:
                    parsed_def['force_end_ro'] = bool(value)
                except Exception as e:
                    raise ValueError(f'Error creating force_end_ro: {e}') from e

            elif key == 'chunk_size':
                try:
                    parsed_def['chunk_size'] = int(value)
                except Exception as e:
                    raise ValueError(f'Error creating chunk_size: {e}') from e

        return cls(**parsed_def)


def _check_expected_col(
        col: str,
        source: str,
        stage_i: Optional[int],
        expr_i: Optional[int],
):
    """Check a column name for expected suffixes when joining tables.

    :param col: Column name to check.
    :param source: Source of the column name.
    :param stage_i: Stage index.
    :param expr_i: Expression index.

    :returns: Column name without suffix.
    """
    if not (col.endswith('_a') or col.endswith('_b')):
        raise ValueError(
            f'Column {col} must end with "_a" or "_b" ('
            f'{source}, '
            f'stage {stage_i if stage_i is not None else "NA"}, '
            f'expression {expr_i if expr_i is not None else "NA"}'
            f')'
        )

    return col[:-2]
