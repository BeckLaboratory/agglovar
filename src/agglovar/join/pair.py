"""
Generate pairwise intersects between variant tables.

For a pair of variant tables (A and B), find all pairs of variants between A and B that meet a set of intersect
parameters. Each variant may have multiple matches.

The PairwiseIntersect class handles joins. The class is immutable once initialized, and one class can be used to
generate multiple joins.

Input is in Polars tables (DataFrame or LazyFrame). These should be loaded into memory (i.e. `pl.DataFrame`) or linked
to a Parquet file (i.e. `pl.scan_parquet()`). Output tables are LazyFrames. Columns starting with "_" may be overwritten
by the join process (i.e. "_index" or "_end_ro" are added to tables). Input table schema and general format are checked
with some missing values automatically filled in, but input is not completely validated.

Warning:
    Input tables should be sorted to avoid extremely high CPU and memory usage.

Join columns:
    * index_a: Row index in df_a.
    * index_b: Row index in df_b.
    * id_a: Variant ID in df_a.
    * id_b: Variant ID in df_b.
    * ro: Reciprocal overlap if variants intersect (0.0 if no overlap).
    * size_ro: Size reciprocal overlap (maximum RO if variants were shifted to maximally intersect).
    * offset_dist: The maximum of the start position distance and end position distance.
    * offset_prop: Offset / variant length. Variant length is the minimum of varlen_a and varlen_b.
    * match_prop: Proportion of an alignment/match score between joined sequences divided by the
        maximum match score if sequences were identical.

The init parameters passed to the PairwiseIntersect class provide powerful and flexible join control. For additional
control, several methods are provided to add join predicates, join columns, and join filters.

Joins use the following logic:
.. code-block:: python

    (
        df_a
        .join_where(
            df_b,
            *join_predicates
        )
        .select(
            *join_cols
        )
        .filter(
            *join_filters
        )
        .sort(
            ['index_a', 'index_b']
        )
    )

Input tables `df_a` and `df_b` are first pre-processed to ensure all columns are present, and some columns are added
or defaults are set for missing values. Each column is appended with "_a" and "_b" suffixes to avoid name conflicts
during joins. Join predicates control which pairs of variants are joined and operate on columns in df_a and df_b with
"_a" and "_b" suffixes on all column names. Join columns are expressions that transform df_a and df_b columns into
a join table. Join filters operate on the join columns and provide further filtering. These expressions are set by the
init parameters passed to the PairwiseIntersect class, but the class allows additional expressions to be appended.
Note that once a join is performed or `lock()` is called, no further expressions may be added.
"""

from dataclasses import dataclass
from dataclasses import field
from typing import Iterable, Generator
from warnings import warn

import polars as pl

from .. import seqmatch
from .. import schema

# Reserved columns are added automatically to input tables
RESERVED_COLUMNS = {
    '_index', '_end_ro'
}

# Default size to chunk tables before joining. A value of 10,000 works well to balance combinatorial explosion
# without exploding the number of chunks to merge when intersecting large variant tables.
DEFAULT_CHUNK_SIZE = 10_000

#
# Intersect class definition
#

@dataclass(frozen=True)
class PairwiseIntersect:
    """
    Pairwise intersect class.

    Attributes:
        ro_min: Minimum reciprocal overlap for allowed matches. If 0.0, then any overlap matches.
        size_ro_min: Reciprocal length proportion of allowed matches. If `match_prop_min` is set and the
            value of this parameter is `None` or is less than `match_prop_min`, then it is set to `match_prop_min` since
            this value represents the lower-bound of allowed match proportions.
        offset_max: Maximum offset allowed (minimum of start or end position distance).
        offset_prop_max: Maximum size-offset (offset / varlen) allowed.
        match_ref: "REF" column must match between two variants.
        match_alt: "ALT" column must match between two variants.
        match_prop_min: Minimum matched base proportion in alignment or None to not match.
        match_score_model: (Advanced) Configured model for scoring similarity between pairs of sequences. If `None` and
            `match_prop_min` is set, then a default aligner will be used.
        force_end_ro: (Advanced) By default, reciprocal overlap is calculated with the end position set to the start
            position plus the variant length. For all variants except insertions, this will typically match the end
            value in the source DataFrame. If `True`, the end position in the DataFrame is also used for reciprocal
            overlap without changes. Typically, this option should not be used and will break reciprocal overlap for
            insertions.
        chunk_size: (Advanced) Chunk df_a into partitions of this size, and for each chunk, subset df_b to include only
            variants that may intersect with variants in the chunk. If None, each chromosome is a single chunk, which
            will lead to a combinatorial explosion unless offset_max is greater than 0.
        match_seq: (Internal state) True if sequence matching is performed (i.e. match_prop_min is not None).
        is_locked: (Internal state)True if this object is locked and cannot be modified.
    """

    # Configuration Attributes
    ro_min: float = field(default=None)
    size_ro_min: float = field(default=None)
    offset_max: int = field(default=None)
    offset_prop_max: float = field(default=None)
    match_ref: bool = field(default=False)
    match_alt: bool = field(default=False)
    match_prop_min: float = field(default=None)

    # Advanced Configuration Attributes
    match_score_model: seqmatch.MatchScoreModel = field(default=None)
    force_end_ro: bool = field(default=False)
    chunk_size: int = field(default=DEFAULT_CHUNK_SIZE)

    # Table and Join Control
    _join_predicates: list[pl.Expr] = field(default_factory=list, init=False, repr=False)
    _join_filters: list[pl.Expr] = field(default_factory=list, init=False, repr=False)
    _join_cols: list[pl.Expr] = field(default_factory=list, init=False,repr=False)

    # Class Internal Attributes
    match_seq: bool = field(default=False, init=False, repr=False)
    is_locked: bool = field(default=False, init=False, repr=False)
    _expected_cols: set[str] = field(default_factory=lambda: {'chrom', 'pos', 'end'}, init=False, repr=False)
    _chunk_range: dict[tuple[str, str], list[pl.Expr]] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self):

        #
        # Join Table Expressions
        #

        expr_overlap_ro = (
            (
                (
                    pl.min_horizontal(
                        [pl.col('_end_ro_a'), pl.col('_end_ro_b')]
                    )
                    - pl.max_horizontal(
                        [pl.col('pos_a'), pl.col('pos_b')]
                    )
                ) / (
                    pl.max_horizontal([pl.col('varlen_a'), pl.col('varlen_b')])
                )
            )
            .clip(0.0, 1.0)
            .cast(pl.Float32)
        )

        expr_szro = (
            (
                (
                    pl.min_horizontal([pl.col('varlen_a'), pl.col('varlen_b')])
                ) / (
                    pl.max_horizontal([pl.col('varlen_a'), pl.col('varlen_b')])
                )
            )
            .cast(pl.Float32)
        )

        expr_offset_dist = (
            pl.max_horizontal(
                (pl.col('pos_a') - pl.col('pos_b')).abs(),
                (pl.col('end_a') - pl.col('end_b')).abs()
            )
            .cast(pl.Int32)
        )

        expr_offset_prop = (
            (
                expr_offset_dist / pl.min_horizontal([pl.col('varlen_a'), pl.col('varlen_b')])
            )
            .cast(pl.Float32)
        )

        expr_match_prop = (
            pl.struct(
                pl.col('seq_a'), pl.col('seq_b')
            )
            .map_elements(
                lambda s: self.match_score_model.match_prop(s['seq_a'], s['seq_b']),
                return_dtype=pl.Float32
            )
        ) if self.match_prop_min is not None else (
            pl.lit(None).cast(pl.Float32)
        )

        #
        # Configuration Attributes
        #

        # Check: ro_min
        if self.ro_min is not None:
            try:
                object.__setattr__(self, 'ro_min', float(self.ro_min))
            except ValueError as e:
                raise ValueError(f'Reciprocal-overlap parameter (ro_min) is not a number: {self.ro_min}') from e

            if not 0.0 <= self.ro_min <= 1.0:
                raise ValueError(
                    f'Reciprocal-overlap parameter (ro_min) must be between 0.0 and 1.0 (inclusive): {self.ro_min}'
                )

            self.append_join_predicates(
                expr_overlap_ro >= self.ro_min
            )

            # self.append_join_filters(
            #     pl.col('ro') >= ro_min
            # )

            self._append_chunk_range('pos', 'max', pl.col('_end_ro_a'))
            self._append_chunk_range('_end_ro', 'min', pl.col('pos_a'))

        # Check: size_ro_min
        if self.size_ro_min is not None:
            try:
                object.__setattr__(self, 'size_ro_min', float(self.size_ro_min))
            except ValueError as e:
                raise ValueError(
                    f'Size-reciprocal-overlap parameter (size_ro_min) is not a number: {self.size_ro_min}'
                ) from e

            if not 0.0 < self.size_ro_min <= 1.0:
                raise ValueError(
                    f'Size-reciprocal-overlap parameter (size_ro_min) must be between '
                    f'0.0 (exclusive) and 1.0 (inclusive): {self.size_ro_min}'
                )

            self.append_join_predicates(
                expr_szro >= self.size_ro_min
            )

            self._append_chunk_range('varlen', 'min', pl.col('varlen_a') * self.size_ro_min)
            self._append_chunk_range('varlen', 'max', pl.col('varlen_a') * (1 / self.size_ro_min))

        # Check: offset_max
        if self.offset_max is not None:
            try:
                object.__setattr__(self, 'offset_max', int(self.offset_max))
            except ValueError as e:
                raise ValueError(f'Offset-max parameter (offset_max) is not an integer: {self.offset_max}') from e

            if self.offset_max < 0:
                raise ValueError(f'Offset-max parameter (offset_max) must not be negative: {self.offset_max}')

            if self.offset_max == 0:
                self.append_join_predicates([  # Very fast joins on equality
                    pl.col('pos_a') == pl.col('pos_b'),
                    pl.col('end_a') == pl.col('end_b')
                ])
            else:
                self.append_join_predicates(
                    expr_offset_dist <= self.offset_max
                )

            self._append_chunk_range('pos', 'min', pl.col('pos_a') - self.offset_max)
            self._append_chunk_range('end', 'max', pl.col('end_a') + self.offset_max)

        # Check: offset_prop_max
        if self.offset_prop_max is not None:
            try:
                object.__setattr__(self, 'offset_prop_max', float(self.offset_prop_max))
            except ValueError as e:
                raise ValueError(
                    f'Size-offset-max parameter (offset_prop_max) is not a number: {self.offset_prop_max}'
                ) from e

            if self.offset_prop_max < 0.0:
                raise ValueError(
                    f'Size-offset-max parameter (offset_prop_max) must not be negative: {self.offset_prop_max}'
                )

            self.append_join_predicates(
                expr_offset_prop <= self.offset_prop_max
            )

            self._append_chunk_range('pos', 'min', pl.col('pos_a') - pl.col('varlen_a') * self.offset_prop_max)
            self._append_chunk_range('end', 'max', pl.col('end_a') + pl.col('varlen_a') * self.offset_prop_max)

        # Check: match_ref
        if not isinstance(self.match_ref, bool):
            raise ValueError(f'Match reference allele (match_ref) must be a boolean: {type(self.match_ref)}')

        if self.match_ref:
            self.append_join_predicates(
                pl.col('ref_a') == pl.col('ref_b')
            )

        # Check: match_alt
        if not isinstance(self.match_alt, bool):
            raise ValueError(f'Match alternate allele (match_alt) must be a boolean: {type(self.match_alt)}')

        if self.match_alt:
            self.append_join_predicates(
                pl.col('alt_a') == pl.col('alt_b')
            )

        # Check: match_prop_min
        if self.match_prop_min is not None:
            try:
                object.__setattr__(self, 'match_prop_min', float(self.match_prop_min))
            except ValueError as e:
                raise ValueError(
                    f'Alignment proportion (match_prop_min) must be a number: ({type(self.match_prop_min)})'
                ) from e

            if not 0.0 <= self.match_prop_min <= 1.0:
                raise ValueError(
                    f'Alignment proportion (match_prop_min) must be between 0.0 and 1.0 (inclusive): '
                    f'{self.match_prop_min}'
                )

            if self.match_prop_min > 0.0:
                self.append_join_filters(
                    pl.col('match_prop') >= self.match_prop_min
                )

        #
        # Advanced Configuration Attributes
        #

        # Check: match_score_model
        if self.match_prop_min is not None:
            if self.match_score_model is None:
                object.__setattr__(self, 'match_score_model', seqmatch.MatchScoreModel())
            else:
                if not isinstance(self.match_score_model, seqmatch.MatchScoreModel):
                    raise ValueError(
                        f'Alignment model (match_score_model) must be a seqmatch.MatchScoreModel: '
                        f'{type(self.match_score_model)}'
                    )

        # Check: force_end_ro
        if not isinstance(self.force_end_ro, bool):
            raise ValueError(f'Force end overlap (force_end_ro) must be a boolean: {type(self.force_end_ro)}')

        # Check: chunk_size
        try:
            object.__setattr__(self, 'chunk_size', int(self.chunk_size))
        except ValueError as e:
            raise ValueError(f'Chunk size (chunk_size) is not an integer: {self.chunk_size}') from e

        if self.chunk_size < 0:
            raise ValueError(f'Chunk size (chunk_size) must not be negative: {self.chunk_size}')

        #
        # Class Internal Attributes
        #

        if self.match_prop_min is not None:
            object.__setattr__(self, 'match_seq', True)

        # Set join columns
        self.append_join_cols([
            pl.col('_index_a').alias('index_a'),
            pl.col('_index_b').alias('index_b'),
            pl.col('id_a'),
            pl.col('id_b'),
            expr_overlap_ro.alias('ro'),
            expr_offset_dist.alias('offset_dist'),
            expr_offset_prop.alias('offset_prop'),
            expr_szro.alias('size_ro'),
            expr_match_prop.alias('match_prop')
        ])

    def lock(self) -> None:
        """Locks the join configuration to prevent changes."""

        # join_predicates and join_filters may be empty, and will crash the join if they are. Ensure they are not empty.
        if len(self._join_predicates) == 0:
            self.append_join_predicates(pl.lit(True))

        if len(self._join_filters) == 0:
            self.append_join_filters(pl.lit(True))

        # Lock the object
        object.__setattr__(self, 'is_locked', True)

    @property
    def join_predicates(self) -> list[pl.Expr]:
        """
        List of expressions to be applied during the table join.
        
        These expressions are arguments to pl.join_where() and operate on columns
        of df_a and df_b joined into one record where all columns from df_a have the suffix "_a" 
        and all columns from df_b have the suffix "_b". For example, if match_alt is True, the expression
        "pl.col('alt_a') == pl.col('alt_b')" will be in this list.
        """
        return self._join_predicates.copy()

    @property
    def join_filters(self) -> list[pl.Expr]:
        """
        List of expressions to be applied after the join is performed.
        
        These expressions operate on the columns of the joined table (see join table columns in the class docstring).
        """
        return self._join_filters.copy()

    @property
    def join_cols(self) -> list[pl.Expr]:
        """List of columns to include in the join table."""
        return self._join_cols.copy()

    @property
    def expected_cols(self) -> set[str]:
        """
        A list of columns expected to be found in df_a and df_b without "_a" or "_b" suffixes.
        
        This is set based on parameters needed to perform the join. For example, if sequence 
        matching is required, then "seq" will be in this list, and if "seq" does not exist 
        in both df_a and df_b, then an error is raised.
        """
        return self._expected_cols.copy()

    @property
    def chunk_range(self) -> dict[tuple[str, str], list[pl.Expr]]:
        """
        A dict of keys to a list of expressions used to subset df_b to include only variants 
        that may match variants in a df_a chunk.
        
        Keys are formatted as "field_limit" where "limit" is "min" or "max" (e.g. "pos_min" 
        is the minimum value for "pos"). The list of expressions associated with a key are 
        executed on a df_a chunk, and the minimum or maximum value from the list (one element 
        per record in df_a) is used as the limit value for a field in df_b. For example, if 
        "pos_min" is a key and [pl.col('pos_a')] is the value, then the expression takes the 
        minimum value of pos_a across all records in df_a and uses it to filter df_b such that 
        no variant in the chunked df_b table has "pos_b" less than this minimum value. If 
        multiple expressions are given, then all expressions are executed and the minimum or 
        maximum value for all is taken. This allows non-trivial chunking of df_b necessary to 
        restrict combinatorial explosion for certain parameters. For example, if reciprocal 
        overlap (ro_min) is set, the maximum position in df_b is determined by the minimum end 
        position in df_a (i.e. "pos_max" will contain "pl.col('end_ro_a'))".
        """
        return self._chunk_range.copy()

    @property
    def has_match_prop(self) -> bool:
        """`True` if "match_prop" is computed for the join table."""
        return self.match_prop_min is not None

    def append_join_predicates(self, expr: Iterable[pl.Expr]|pl.Expr) -> None:
        """
        Append expressions to a list of join predicates given as arguments to pl.join_where().
        
        This class will construct a list of join predicates from the constructor arguments, 
        but additional join control may be added here.

        Warning:
            Adding predicates may alter the join results so that they are not reproducible 
            based on join arguments. Use with caution.

        Args:
            expr: An expression or list of expressions.
        """

        if self.is_locked:
            raise AttributeError('Pairwise intersect object is locked and cannot be modified.')

        if isinstance(expr, pl.Expr):
            expr = [expr]

        self._add_expected_cols(expr)
        self._join_predicates.extend(expr)

    def append_join_filters(self, expr: Iterable[pl.Expr]|pl.Expr) -> None:
        """
        Append expressions to a list of join filters applied to the join table immediately after the join through
        pl.filter().

        Warning:
            Adding predicates may alter the join results so that they are not reproducible
            based on join arguments. Use with caution.

        Args:
            expr: An expression or list of expressions.
        """

        if self.is_locked:
            raise AttributeError('Pairwise intersect object is locked and cannot be modified.')

        if isinstance(expr, pl.Expr):
            expr = [expr]

        self._join_filters.extend(expr)

    def append_join_cols(self, expr: Iterable[pl.Expr]|pl.Expr) -> None:
        """
        Append expressions to the list of columns included in the join table.
        
        These columns will be appended to the standard join table columns. Each expression 
        should name the column it creates using ".alias()" if necessary. These columns do 
        not affect the join itself, just the columns that appear in the join table.

        For example, to retain the "pos" column from df_a and df_b, then append 
        "pl.col('pos_a')" and "pl.col('pos_b')". If you wanted to set a flag for whether 
        the variant in df_a comes before df_b, then a new columns could be added: 
        "(pl.col('pos_a') <= pl.col('pos_b')).alias('left_a')"

        Args:
            expr: Expression or list of expressions.
        """
        if self.is_locked:
            raise AttributeError('Pairwise intersect object is locked and cannot be modified.')

        if isinstance(expr, pl.Expr):
            expr = [expr]

        self._add_expected_cols(expr)
        self._join_cols.extend(expr)

    def _add_expected_cols(self, expr: Iterable[pl.Expr]|pl.Expr) -> None:
        """
        Inspect expressions and add each required column to the set of expected columns in the source dataframe.

        Args:
            expr: List of expressions.
        """

        if isinstance(expr, pl.Expr):
            expr = [expr]

        for e in expr:
            for col_name in e.meta.root_names():
                if not col_name.endswith('_a') and not col_name.endswith('_b'):
                    raise ValueError(
                        f'Expected column name to end with "_a" or "_b": Found column "{col_name}" in expression "{e}"'
                    )

                col_name = col_name[:-2]

                if col_name not in RESERVED_COLUMNS:
                    self._expected_cols.add(col_name)

    def _append_chunk_range(
            self,
            key: str,
            limit: str,
            expr: pl.Expr
    ) -> None:
        """
        Append a rule for chunking df_b based on a subset of df_a.

        Args:
            key: Column name in df_b without "_b" suffix (e.g. "pos" for "pos_a").
            limit: If the limit is "min" or "max".
            expr: An expression applied to df_a to determine a minimum or maximum value for column "key" in df_b.
        """

        if limit not in {'min', 'max'}:
            raise ValueError(f'Limit must be "min" or "max": {limit}')

        if not (key := key.strip() if key else None):
            raise ValueError('Key must not be empty')

        if key not in RESERVED_COLUMNS:
            self._expected_cols.add(key)

        self._add_expected_cols(expr)

        if (key, limit) not in self._chunk_range:
            self._chunk_range[key, limit] = []

        self._chunk_range[key, limit].append(expr)

    def join_iter(
            self,
            df_a: pl.DataFrame|pl.LazyFrame,
            df_b: pl.DataFrame|pl.LazyFrame
        ) -> Generator[pl.LazyFrame, None, None]:
        """
        Find all pairs of variants in two sources that meet a set of criteria.

        Args:
            df_a: Source dataframe.
            df_b: Target dataframe.

        Yields:
            A LazyFrame for each chunk.
        """

        # Prepare tables
        df_a, df_b = self.prepare_tables(df_a, df_b)

        # Join per chromosome
        chrom_list = df_a.select('chrom_a').unique().collect().to_series().sort().to_list()

        self.lock()

        if len(chrom_list) == 0:
            # No chromosomes found, yield empty table (otherwise, concat across iterater will fail on an empty list)
            yield (
                df_a.head(0)
                .join_where(
                    df_b.head(0),
                    *self._join_predicates
                )
                .select(
                    *self._join_cols
                )
                .filter(
                    *self._join_filters
                )
                .sort(
                    ['index_a', 'index_b']
                )
            )

        for chrom in chrom_list:
            range_a_min, range_a_max = (
                df_a
                .filter(
                    pl.col('chrom_a') == chrom
                )
                .select(
                    pl.col('_index_a').min().alias('min'),
                    (pl.col('_index_a').max() + 1).alias('max')
                )
                .collect()
                .transpose()
                .to_series()
            )

            chunk_size_chrom = self.chunk_size if self.chunk_size > 0 else range_a_max - range_a_min

            for i_a in range(range_a_min, range_a_max, chunk_size_chrom):
                i_a_end = min(i_a + chunk_size_chrom, range_a_max)

                df_a_chunk = chunk_index(df_a, i_a, i_a_end)
                df_b_chunk = self._chunk_relative(df_a_chunk, df_b, chrom)

                yield (
                    df_a_chunk
                    .join_where(
                        df_b_chunk,
                        *self._join_predicates
                    )
                    .select(
                        *self._join_cols
                    )
                    .filter(
                        *self._join_filters
                    )
                    .sort(
                        ['index_a', 'index_b']
                    )
                )

    def join(
            self,
            df_a: pl.DataFrame|pl.LazyFrame,
            df_b: pl.DataFrame|pl.LazyFrame
    ) -> pl.LazyFrame:
        """
        Find all pairs of variants in two sources that meet a set of criteria.

        This is a convenience function that calls join_iter() and concatenates the results.

        Args:
            df_a: Table A.
            df_b: Table B.

        Returns:
            A join table.
        """

        return pl.concat(
            self.join_iter(df_a, df_b)
        )

    def prepare_tables(self, df_a: pl.LazyFrame, df_b: pl.LazyFrame) -> tuple[pl.LazyFrame, pl.LazyFrame]:
        """
        Prepares tables for join. Checks for expected columns and formats, adds missing columns as needed, and
        appends "_a" and "_b" suffixes to column names.

        Args:
            df_a: Table A.
            df_b: Table B.

        Returns:
            Tuple of normalized tables (df_a, df_b).
        """

        # Columns this function automatically generates
        autogen_cols = {'varlen', '_end_ro'}

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

        missing_cols_a = sorted((self._expected_cols - columns_a) - autogen_cols)
        missing_cols_b = sorted((self._expected_cols - columns_b) - autogen_cols)

        if missing_cols_a or missing_cols_b:
            if missing_cols_a == missing_cols_b:
                raise ValueError(f'DataFrames "A" and "B" are missing expected column(s): {", ".join(missing_cols_a)}')

            raise ValueError(
                f'DataFrame "A" missing expected column(s): {", ".join(missing_cols_a)}; '
                f'DataFrame "B" missing expected column(s): {", ".join(missing_cols_b)}'
            )

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

        # Warn for reserved columns
        if reserved_cols := sorted(RESERVED_COLUMNS & columns_a):
            warn(f'Found reserved columns in table "A" (may be overwritten): {", ".join(reserved_cols)}')

        if reserved_cols := sorted(RESERVED_COLUMNS & columns_b):
            warn(f'Found reserved columns in table "B" (may be overwritten): {", ".join(reserved_cols)}')

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

        df_a = df_a.with_columns(
            pl.col('id').fill_null(
                pl.concat_str(pl.lit('VarIdx'), pl.col('_index'))
            )
        )

        df_b = df_b.with_columns(
            pl.col('id').fill_null(
                pl.concat_str(pl.lit('VarIdx'), pl.col('_index'))
            )
        )

        # Prepare REF & ALT
        if self.match_ref:
            df_a = df_a.with_columns(
                pl.col('ref').str.to_uppercase().str.strip_chars()
            )

            df_b = df_b.with_columns(
                pl.col('ref').str.to_uppercase().str.strip_chars()
            )

        if self.match_alt:
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

    def _chunk_relative(
            self,
            df_a: pl.LazyFrame,
            df_b: pl.LazyFrame,
            chrom: str
    ) -> pl.LazyFrame:
        """
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

        Args:
            df_a: Table chunk.
            df_b: Table to be chunked to records that may intersect with df_a.
            chrom: Chromosome name.

        Returns:
            df_b partitioned (LazyFrame).
        """

        filter_list = [
            pl.col('chrom_b') == chrom
        ]

        for (col_name, limit), expr_list in self._chunk_range.items():

            if limit == 'min':
                filter_list.append(
                    pl.col(col_name + '_b') >= (
                        df_a
                        .select(pl.min_horizontal(*expr_list))
                        .collect()
                        .to_series()
                        .min()
                    )
                )

            elif limit == 'max':
                filter_list.append(
                    pl.col(col_name + '_b') <= (
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


def chunk_index(
        df_a: pl.LazyFrame,
        i_a: int,
        i_a_max: int
) -> pl.LazyFrame:
    """
    Chunk df_a by a range of indices. WARNING: This function assumes the index range contains a single chromosome,
    and this is not checked.

    Args:
        df_a: Table to chunk.
        i_a: Minimum index.
        i_a_max: Maximum index.

    Returns:
        df_a chunked (LazyFrame).
    """

    if i_a < 0 or i_a_max <= i_a:
        raise ValueError(f'Invalid index range: [{i_a}, {i_a_max})')

    return (
        df_a
        .filter(pl.col('_index_a') >= i_a)
        .filter(pl.col('_index_a') < i_a_max)
    )


def join_iter(
    df_a: pl.DataFrame|pl.LazyFrame,
    df_b: pl.DataFrame|pl.LazyFrame,
    ro_min: float=None,
    size_ro_min: float=None,
    offset_max: int=None,
    offset_prop_max: float=None,
    match_ref: bool=False,
    match_alt: bool=False,
    match_prop_min: float=None
) -> Generator[pl.LazyFrame, None, None]:
    """
    A convenience wrapper for creating a PairwiseIntersect object and calling its join_iter() method.

    Args:
        df_a: Table A.
        df_b: Table B.
        ro_min: Minimum reciprocal overlap for allowed matches. If 0.0, then any overlap matches.
        size_ro_min: Reciprocal length proportion of allowed matches.
            this value represents the lower-bound of allowed match proportions.
        offset_max: Maximum offset allowed (minimum of start or end position distance).
        offset_prop_max: Maximum size-offset (offset / varlen) allowed.
        match_ref: "REF" column must match between two variants.
        match_alt: "ALT" column must match between two variants.
        match_prop_min: Minimum matched base proportion in alignment or None to not match.

    Yields:
        A LazyFrame for each chunk.
    """

    return PairwiseIntersect(
        ro_min=ro_min,
        size_ro_min=size_ro_min,
        offset_max=offset_max,
        offset_prop_max=offset_prop_max,
        match_ref=match_ref,
        match_alt=match_alt,
        match_prop_min=match_prop_min
    ).join_iter(df_a, df_b)


def join(
    df_a: pl.DataFrame|pl.LazyFrame,
    df_b: pl.DataFrame|pl.LazyFrame,
    ro_min: float=None,
    size_ro_min: float=None,
    offset_max: int=None,
    offset_prop_max: float=None,
    match_ref: bool=False,
    match_alt: bool=False,
    match_prop_min: float=None
) -> pl.LazyFrame:
    """
    A convenience wrapper for creating a PairwiseIntersect object and calling its join() method.

    Args:
        df_a: Table A.
        df_b: Table B.
        ro_min: Minimum reciprocal overlap for allowed matches. If 0.0, then any overlap matches.
        size_ro_min: Reciprocal length proportion of allowed matches.
        offset_max: Maximum offset allowed (minimum of start or end position distance).
        offset_prop_max: Maximum size-offset (offset / varlen) allowed.
        match_ref: "REF" column must match between two variants.
        match_alt: "ALT" column must match between two variants.
        match_prop_min: Minimum matched base proportion in alignment or None to not match.

    Returns:
        A join table.
    """

    return PairwiseIntersect(
        ro_min=ro_min,
        size_ro_min=size_ro_min,
        offset_max=offset_max,
        offset_prop_max=offset_prop_max,
        match_ref=match_ref,
        match_alt=match_alt,
        match_prop_min=match_prop_min,
    ).join(df_a, df_b)
