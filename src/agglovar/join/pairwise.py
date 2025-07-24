"""
Get nearest variant by SVLEN overlap. Used for merging and comparing callsets.
"""

import collections
import polars as pl

from typing import Generator

from .. import seqmatch
from .. import schema

COL_MAP_BED = {
    '#CHROM': 'chrom',
    'POS': 'pos',
    'END': 'end',
    'SVTYPE': 'svtype',
    'SVLEN': 'svlen',
    'ID': 'id',
    'REF': 'ref',
    'ALT': 'alt',
    'SEQ': 'seq',
    'MERGE_SAMPLES': 'merge_samples'
}

DEFAULT_PRIORITY_MATCH =[
    ('ro', 0.2),
    ('size_ro', 0.2),
    ('offset_prop', 0.1),
    ('match_prop', 0.5)
]

DEFAULT_PRIORITY_NOMATCH =[
    ('ro', 0.4),
    ('size_ro', 0.4),
    ('offset_prop', 0.2),
]

DEFAULT_PRIORITY = {
    'match': DEFAULT_PRIORITY_MATCH,
    'nomatch': DEFAULT_PRIORITY_NOMATCH,
    'default': DEFAULT_PRIORITY_MATCH
}

KNOWN_COL_SET = {
    'chrom', 'pos', 'end', 'id', 'svtype', 'svlen', 'ref', 'alt', 'seq'
}


#
# Intersect class definition
#

class PairwiseIntersect(object):
    ro_min: float
    size_ro_min: float
    offset_max: int
    offset_prop_max: float
    match_prop_min: float
    match_ref: bool
    match_alt: bool
    col_map: dict[str,str]
    match_score_model: seqmatch.MatchScoreModel
    force_end_ro: bool
    chunk_size: int
    chunk_range: dict[str,list[pl.Expr]]
    join_predicates: list[pl.Expr]
    join_filters: list[pl.Expr]
    expected_cols: list[str]
    expr_overlap_ro: pl.Expr
    expr_szro: pl.Expr
    expr_offset_dist: pl.Expr
    expr_offset_prop: pl.Expr
    expr_match_prop: pl.Expr
    col_list: list[pl.Expr]

    """
    Pairwise intersect class.

    This class is used to find all pairs of variants between two sources that meet a set of criteria where a variant in
    one source may have multiple matches in the other source. The implementation relies on variant tables in LazyFrames
    to efficiently run intersects.

    Intersects accept two tables, df_a and df_b. Each table is transformed to a normalized form where all columns are
    appended with "_a" (in df_a) and "_b" (in df_b). In the following documentation, column names with "_a" refer to
    a column in df_a, and column names with "_b" refer to a column in df_b.

    A join table is returned as a LazyFrame (join()) or Generator (join_iter()) with a standard set of columns:

    * index_a: Row index in df_a.
    * index_b: Row index in df_b.
    * id_a: Variant ID in df_a.
    * id_b: Variant ID in df_b.
    * ro: Reciprocal overlap if variants intersect (0.0 if no overlap).
    * size_ro: Size reciprocal overlap (maximum RO if variants were shifted to maximally intersect).
    * offset_dist: Distance between variants measured as the maximum of the start position and end position
        distances (i.e. max(abs(pos_a - pos_b), abs(end_a - end_b))).
    * offset_prop: Offset / variant length. Variant length is the minimum of svlen_a and svlen_b.
    * match_prop: Proportion of an alignment/match score between joined sequences divided by the
        maximum match score if sequences were identical. If both variants are the same size and have identical
        sequences, then this value is 1.0. If the alignment score between the two sequences is half the maximum, then
        the value is 0.5. The maximum alignment score is defined as the alignment score if all bases aligned
        (i.e. match * min(svlen_a, svlen_b)). If `match_prop_min` is None, this column contains only null value. Setting
        match_prop_min to 0.0 will compute the match_prop values, but will not filter by it.

    Attributes:
        ro_min: Minimum reciprocal overlap for allowed matches. If 0.0, then any overlap matches.
        size_ro_min: Reciprocal length proportion of allowed matches. If `match_prop_min` is set and the
            value of this parameter is `None` or is less than `match_prop_min`, then it is set to `match_prop_min` since
            this value represents the lower-bound of allowed match proportions.
        offset_max: Maximum offset allowed (minimum of start or end position distance).
        offset_prop_max: Maximum size-offset (offset / svlen) allowed.
        match_prop_min: Minimum matched base proportion in alignment or None to not match.
        match_ref: "REF" column must match between two variants.
        match_alt: "ALT" column must match between two variants.
        col_map: Map column names from input variant tables by this dictionary. May be string "bed" to use default
            BED column names.
        match_score_model: Configured model for scoring similarity between pairs of sequences. If `None` and
            `match_prop_min` is set, then a default aligner will be used.
        force_end_ro: By default, reciprocal overlap is calculated with the end position set to the start
            position plus the variant length. For all variants except insertions, this will typically match the end value
            in the source DataFrame. If `True`, the end position in the DataFrame is also used for reciprocal overlap
            without changes. Typically, this option should not be used and will break reciprocal overlap for insertions.
        chunk_size: Chunk df_a into partitions of this size, and for each chunk, subset df_b to include only
            variants that may intersect with variants in the chunk. If None, each chromosome is a single chunk, which will
            lead to a combinatorial explosion unless offset_max is 0 (the join operation is optimized without chunking in
            this case). All intersects are first chunked by chromosome, then by this chunk size.
        match_seq: Set to True if sequence matching is required (i.e. match_prop_min is not None).
        expected_cols: Expected column names in input variant tables without "_a" or "_b" suffixes appended during
            table preparation.
        chunk_range: A dict of keys to a list of expressions used to subset df_b to include only variants that may
            match variants in a df_a chunk. Keys are formatted as "field_limit" where "limit" is "min" or "max" (e.g.
            "pos_min" is the minimum value for "pos"). The list of expressions associated with a key are executed on
            a df_a chunk, and the minimum or maximum value from the list (one element per record in df_a) is used as
            the limit value for a field in df_b. For example, if "pos_min" is a key and [pl.col('pos_a')] is the value,
            then the expression takes the minimum value of pos_a across all records in df_a and uses it to filter
            df_b such that no variant in the chunked df_b table has "pos_b" less than this minimum value. If multiple
            expressions are given, then all expressions are executed and the minimum or maximum value for all is taken.
            This allows non-trivial chunking of df_b necessary to restrict combinatorial explosion for certain
            parameters. For example, if reciprocal overlap (ro_min) is set, the maximum position in df_b is
            determined by the minimum end position in df_a (i.e. "pos_max" will contain "pl.col('end_ro_a'))".
        join_predicates: List of expressions to be applied during the table join. These expressions are arguments to
            pl.join_where(). These expressions operate on columns of df_a and df_b joined into one record where all
            columns from df_a have the suffix "_a" and all columns from df_b have the suffix "_b".
        join_filters: List of expressions to be applied after the join is performed. These expressions operate on the
            columns of the joined table (see join table columns above).
        expected_cols: A list of columns expected to be found in df_a and df_b. This is set based on parameters
            needed to perform the join. For example, if sequence matching is required, then "seq" will be in this list.
        expr_overlap_ro: Expression for reciprocal overlap.
        expr_szro: Expression for reciprocal size overlap.
        expr_offset_dist: Expression for offset distance.
        expr_offset_prop: Expression for offset proportion.
        expr_match_prop: Expression for match proportion.
        col_list: List of columns in the join table.
    """

    def __init__(
            self,
            ro_min: float=None,
            size_ro_min: float=None,
            offset_max: int=None,
            offset_prop_max: float=None,
            match_prop_min: float=None,
            match_ref: bool=False,
            match_alt: bool=False,
            col_map: dict[str,str]=None,
            match_score_model: seqmatch.MatchScoreModel=None,
            force_end_ro: bool=False,
            chunk_size: int=10_000,
            join_predicates: list[pl.Expr]=None,
            join_filters: list[pl.Expr]=None,
            col_list: list[pl.Expr]=None
    ):
            """
            Find all pairs of variants in two sources that meet a set of criteria.

            The columns expected in `df_a` and `df_b` are:
            * chrom: Variant chromosome.
            * pos: Variant position.
            * end: End position. If missing, is set to pos + svlen.
            * svlen: Variant length.
            * id: Variant ID. If missing or empty, will be generated using string "VarIdx" with the record index appended.
            * ref: Optional, required if `match_ref`.
            * alt: Optional, required if `match_alt`.
            * seq: Optional, required if `match_prop_min`.

            Columns in returned DataFrame:
            * index_a: Row index in `df_a`.
            * index_b: Row index in `df_b`.
            * id_a: Record id in `df_a`.
            * id_b: Record id in `df_b`.
            * ro: Reciprocal overlap if variants intersect.
            * size_ro: Size reciprocal overlap (Max RO if variants were shifted to maximally intersect).
            * offset_dist: Minimum of start position distance and end position distance.
            * offset_prop: Offset / variant length.
            * match_prop: Proportion of an alignment/match score between two sequences (from `df_a` and `df_b`) divided by the
                maximum match score if sequences were identical. If `match_prop_min` is None, this column contains only null
                values.

            :param ro_min: Minimum reciprocal overlap for allowed matches. If 0.0, then any overlap matches.
            :param size_ro_min: Reciprocal length proportion of allowed matches. If `match_prop_min` is set and the
                value of this parameter is `None` or is less than `match_prop_min`, then it is set to `match_prop_min` since
                this value represents the lower-bound of allowed match proportions.
            :param offset_max: Maximum offset allowed (minimum of start or end position distance).
            :param offset_prop_max: Maximum size-offset (offset / svlen) allowed.
            :param match_prop_min: Minimum matched base proportion in alignment or None to not match.
            :param match_ref: "REF" column must match between two variants.
            :param match_alt: "ALT" column must match between two variants.
            :param col_map: Map column names from input dataframes by this dictionary. May be string "bed" to use default
                BED column names.
            :param match_score_model: Configured model for scoring similarity between pairs of sequences. If `None` and
                `match_prop_min` is set, then a default aligner will be used.
            :param force_end_ro: By default, reciprocal overlap is calculated with the end position set to the start
                position plus the variant length. For all variants except insertions, this will typically match the end value
                in the source DataFrame. If `True`, the end position in the DataFrame is also used for reciprocal overlap
                without changes. Typically, this option should not be used and will break reciprocal overlap for insertions.
            :param chunk_size: Chunk df_a into partitions of this size, and for each chunk, subset df_b to include only
                variants that may intersect with variants in the chunk. If None, each chromosome is a single chunk, which will
                lead to a combinatorial explosion unless offset_max is 0 (the join operation is optimized without chunking in
                this case). All intersects are first chunked by chromosome, then by this chunk size. The default chunk
                has been tested to work well joining millions of SNV variants.
            :param join_predicates: List of predicates to join on (e.g. `pl.col('pos_a') == pl.col('pos_b')`). This
                should not be set for most joins, it is automatically constructed by the parameters above.
            :param join_filters: List of predicates to filter on (e.g. `pl.col('pos_a') > pl.col('pos_b')`). This
                should not be set for most joins, it is automatically constructed by the parameters above.
            :param col_list: Append these columns to the join table. These may be used to retain columns that would
                normally be dropped (e.g. "pl.col('id_a')") or to compute new ones
                (e.g. "(pl.col('pos_b') > pl.col('pos_a').alias('greater_b')"). If custom filters in join_filters use
                columns that are not in the default join table, then they must be included in this list. This should not
                be set for most joins.

            :return: An iterator of LazyFrames for each chunk.
            """

            # Set attributes
            self.ro_min = ro_min
            self.size_ro_min = size_ro_min
            self.offset_max = offset_max
            self.offset_prop_max = offset_prop_max
            self.match_prop_min = match_prop_min
            self.match_ref = match_ref
            self.match_alt = match_alt
            self.col_map = col_map
            self.match_score_model = match_score_model
            self.force_end_ro = force_end_ro
            self.chunk_size = chunk_size
            self.chunk_range = None  # Set by _set_predicates()
            self.join_predicates = join_predicates
            self.join_filters = join_filters

            # Expected columns (set by _set_expected_cols())
            self.expected_cols = None

            # Join expressions (set by _set_join_exprs())
            self.expr_overlap_ro = None
            self.expr_szro = None
            self.expr_offset_dist = None
            self.expr_offset_prop = None
            self.expr_match_prop = None

            # Check and normalize parameters
            self._check_params()

            # Set expressions used by join
            self._set_join_exprs()

            # Set predicates and filters (post-join filters and chunk filters)
            self._set_predicates()

            # Set expected columns from input tables
            self._set_expected_cols()

            # Set expected join columns
            self.col_list = [
                pl.col('index_a'),
                pl.col('index_b'),
                pl.col('id_a'),
                pl.col('id_b'),
                self.expr_overlap_ro.alias('ro'),
                self.expr_offset_dist.alias('offset_dist'),
                self.expr_offset_prop.alias('offset_prop'),
                self.expr_szro.alias('size_ro'),
                self.expr_match_prop.alias('match_prop')
            ] + (col_list if col_list is not None else [])

    def join_iter(
            self,
            df_a: pl.DataFrame|pl.LazyFrame,
            df_b: pl.DataFrame|pl.LazyFrame
        ) -> Generator[pl.LazyFrame, None, None]:
            """
            Find all pairs of variants in two sources that meet a set of criteria.

            :param df_a: Source dataframe.
            :param df_b: Target dataframe.

            :return: An iterator of LazyFrames for each chunk.
            """

            # Prepare tables
            df_a, df_b = self.prepare_tables(df_a, df_b)


            # Join per chromosome
            chrom_list = df_a.select('chrom_a').unique().collect().to_series().sort().to_list()

            for chrom in chrom_list:
                range_a_min, range_a_max = (
                    df_a
                    .filter(
                        pl.col('chrom_a') == chrom
                    )
                    .select(
                        pl.col('index_a').min().alias('min'),
                        (pl.col('index_a').max() + 1).alias('max')
                    )
                    .collect()
                    .transpose()
                    .to_series()
                )

                chunk_size_chrom = self.chunk_size if self.chunk_size is not None else range_a_max - range_a_min

                for i_a in range(range_a_min, range_a_max, chunk_size_chrom):
                    i_a_end = min(i_a + chunk_size_chrom, range_a_max)

                    df_a_chunk = chunk_index(df_a, i_a, i_a_end)
                    df_b_chunk = chunk_relative(df_a_chunk, df_b, chrom, self.chunk_range)

                    yield (
                        df_a_chunk
                        .join_where(
                            df_b_chunk,
                            *self.join_predicates
                        )
                        .select(
                            *self.col_list
                        )
                        .filter(
                            *self.join_filters
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

        :param df_a: Source dataframe.
        :param df_b: Target dataframe.

        :return: A join table.
        """

        return pl.concat(
            self.join_iter(df_a, df_b)
        )


    def _check_params(self) -> None:
        """
        Check parameters and set defaults. This should be called at the end of the constructor.
        """

        # Column name mapping
        if isinstance(self.col_map, str) and self.col_map.strip().lower() == 'bed':
            self.col_map = COL_MAP_BED

        # Check: ro_min
        if self.ro_min is not None:
            try:
                self.ro_min = float(self.ro_min)
            except ValueError:
                raise ValueError(f'Reciprocal-overlap parameter (ro_min) is not a floating point number: {self.ro_min}')

            if self.ro_min < 0.0 or self.ro_min > 1.0:
                raise ValueError(f'Reciprocal-overlap parameter (ro_min) must be between 0.0 and 1.0 (inclusive): {self.ro_min}')

        # Check: size_ro_min
        if self.size_ro_min is not None:
            try:
                self.size_ro_min = float(self.size_ro_min)
            except ValueError:
                raise ValueError(f'Size-reciprocal-overlap parameter (size_ro_min) is not a floating point number: {self.size_ro_min}')

            if self.size_ro_min <= 0.0 or self.size_ro_min > 1.0:
                raise ValueError(f'Size-reciprocal-overlap parameter (size_ro_min) must be between 0 (exclusive) and 1 (inclusive): {self.size_ro_min}')

        # Check: offset_max
        if self.offset_max is not None:
            try:
                self.offset_max = int(self.offset_max)
            except ValueError:
                raise ValueError(f'Offset-max parameter (offset_max) is not an integer: {self.offset_max}')
            except OverflowError:
                raise ValueError(f'Offset-max parameter (offset_max) exceeds the max size (32 bits): {self.offset_max}')

            if self.offset_max < 0:
                raise ValueError(f'Offset-max parameter (offset_max) must not be negative: {self.offset_max}')

        # Check: offset_prop_max
        if self.offset_prop_max is not None:
            try:
                self.offset_prop_max = float(self.offset_prop_max)
            except ValueError:
                raise ValueError(f'Size-offset-max parameter (offset_prop_max) is not a floating point number: {self.offset_prop_max}')

            if self.offset_prop_max < float(0.0):
                raise v(f'Size-offset-max parameter (offset_prop_max) must not be negative: {self.offset_prop_max}')

        # Check: match_ref
        if self.match_ref is None:
            self.match_ref = False
        else:
            if not isinstance(self.match_ref, bool):
                raise ValueError(f'Match reference allele (match_ref) must be a boolean: {type(self.match_ref)}')

        # Check: match_alt
        if self.match_alt is None:
            self.match_alt = False
        else:
            if not isinstance(self.match_alt, bool):
                raise ValueError(f'Match alternate allele (match_alt) must be a boolean: {type(self.match_alt)}')

        # Check: match_prop_min & match_score_model
        self.match_seq = self.match_prop_min is not None

        if self.match_seq:
            if self.match_score_model is None:
                self.match_score_model = seqmatch.MatchScoreModel()
            else:
                if not isinstance(self.match_score_model, seqmatch.MatchScoreModel):
                    raise ValueError(f'Alignment model (match_score_model) must be a seqmatch.MatchScoreModel: {type(self.match_score_model)}')

            try:
                self.match_prop_min = float(self.match_prop_min)
            except ValueError:
                raise ValueError(f'Alignment proportion (match_prop_min) must be a number: {self.match_prop_min} (type {type(self.match_prop_min)})')

            if self.match_prop_min <= 0.0 or self.match_prop_min > 1.0:
                raise ValueError(f'Alignment proportion (match_prop_min) must be between 0.0 (exclusive) and 1.0 (inclusive): {self.match_prop_min}')

        # Check: chunk_range
        if self.chunk_range is None:
            self.chunk_range = dict()
        else:
            if not isinstance(self.chunk_range, dict):
                raise ValueError(f'Chunk range (chunk_range) must be a dict: {type(self.chunk_range)}')

        # Check: join_predicates
        if self.join_predicates is None:
            self.join_predicates = []
        elif not isinstance(self.join_predicates, list):
            raise ValueError(f'Join predicates (join_predicates) must be a list: {type(self.join_predicates)}')

        # Check: join_filters
        if self.join_filters is None:
            self.join_filters = []
        elif not isinstance(self.join_filters, list):
            raise ValueError(f'Join filters (join_filters) must be a list: {type(self.join_filters)}')

        # Check: chunk_size
        if self.chunk_size is not None:
            try:
                self.chunk_size = int(self.chunk_size)
            except ValueError:
                raise ValueError(f'Chunk size (chunk_size) is not an integer: {self.chunk_size}')
            except OverflowError:
                raise ValueError(f'Chunk size (chunk_size) exceeds the max size (32 bits): {self.chunk_size}')

            if self.chunk_size < 0:
                raise ValueError(f'Chunk size (chunk_size) must not be negative: {self.chunk_size}')

    def _set_join_exprs(self) -> None:
        """
        Set expressions for joining. This should only be called once by the constructor.
        """

        self.expr_overlap_ro = (
            (
                (
                    pl.min_horizontal([pl.col('end_ro_a'), pl.col('end_ro_b')]) - pl.max_horizontal([pl.col('pos_a'), pl.col('pos_b')])
                ) / (
                    pl.max_horizontal([pl.col('svlen_a'), pl.col('svlen_b')])
                )
            )
            .clip(0.0, 1.0)
            .cast(pl.Float32)
        )

        self.expr_szro = (
            (
                (
                    pl.min_horizontal([pl.col('svlen_a'), pl.col('svlen_b')])
                ) / (
                    pl.max_horizontal([pl.col('svlen_a'), pl.col('svlen_b')])
                )
            )
            .cast(pl.Float32)
        )

        self.expr_offset_dist = (
            pl.max_horizontal(
                (pl.col('pos_a') - pl.col('pos_b')).abs(),
                (pl.col('end_a') - pl.col('end_b')).abs()
            )
            .cast(pl.Int32)
        )

        self.expr_offset_prop = (
            (
                self.expr_offset_dist / pl.min_horizontal([pl.col('svlen_a'), pl.col('svlen_b')])
            )
            .cast(pl.Float32)
        )

        self.expr_match_prop = (
            pl.struct(
                pl.col('seq_a'), pl.col('seq_b')
            )
            .map_elements(
                lambda s: self.match_score_model.match_prop(s['seq_a'], s['seq_b']),
                return_dtype=pl.Float32
            )
        ) if self.match_seq else (
            pl.lit(None).cast(pl.Float32)
        )

    def _set_predicates(self) -> None:
        """
        Set predicates and expressions for joining. This should only be called once by the constructor.
        """

        chunk_range = collections.defaultdict(list)
        join_predicates = []
        join_filters = []

        # ro_min
        if self.ro_min is not None:
            chunk_range['pos_max'].append(pl.col('end_ro_a'))
            chunk_range['end_ro_min'].append(pl.col('pos_a'))

            join_predicates.extend([
                self.expr_overlap_ro >= self.ro_min
            ])

            # join_filters.append(
            #     pl.col('ro') >= ro_min
            # )

        # size_ro_min
        if self.size_ro_min is not None:
            chunk_range['svlen_min'].append(pl.col('svlen_a') * self.size_ro_min)
            chunk_range['svlen_max'].append(pl.col('svlen_a') * (1 / self.size_ro_min))

            join_predicates.append(
                self.expr_szro >= self.size_ro_min
            )

        # offset_max
        if self.offset_max is not None:
            if self.offset_max == 0:
                join_predicates.extend([  # Very fast joins on equality
                    pl.col('pos_a') == pl.col('pos_b'),
                    pl.col('end_a') == pl.col('end_b')
                ])
            else:
                join_predicates.append(
                    self.expr_offset_dist <= self.offset_max
                )

            chunk_range['pos_min'].append(pl.col('pos_a') - self.offset_max)
            chunk_range['end_max'].append(pl.col('end_a') + self.offset_max)

        # offset_prop_max
        if self.offset_prop_max is not None:
            chunk_range['pos_min'].append(pl.col('pos_a') - pl.col('svlen_a') * self.offset_prop_max)
            chunk_range['end_max'].append(pl.col('end_a') + pl.col('svlen_a') * self.offset_prop_max)

            join_predicates.append(
                self.expr_offset_prop <= self.offset_prop_max
            )

        # match_prop_min
        if self.match_prop_min is not None and self.match_prop_min > 0.0:
            join_filters.append(
                pl.col('match_prop') >= self.match_prop_min
            )

        # match_ref
        if self.match_ref:
            join_predicates.append(
                pl.col('ref_a') == pl.col('ref_b')
            )

        # match_alt
        if self.match_alt:
            join_predicates.append(
                pl.col('alt_a') == pl.col('alt_b')
            )

        # Merge chunk ranges with user-defined ranges (user-defined last)
        for key in set(self.chunk_range.keys()).union(set(chunk_range.keys())):
            self.chunk_range[key] = (
                chunk_range[key]
            ) + (
                self.chunk_range[key] if key in self.chunk_range else []
            )

        # Concatenate join predicates and join filters
        self.join_predicates = join_predicates + self.join_filters
        self.join_filters = join_filters + self.join_filters

        if len(self.join_predicates) == 0:
            self.join_predicates = [pl.lit(True)]

        if len(self.join_filters) == 0:
            self.join_filters = [pl.lit(True)]


    def _set_expected_cols(self) -> None:

        self.expected_cols = ['chrom', 'pos', 'end']

        if self.match_ref:
            self.expected_cols.append('ref')
        if self.match_alt:
            self.expected_cols.append('alt')
        if self.match_seq:
            self.expected_cols.append('seq')


    def prepare_tables(self, df_a: pl.LazyFrame, df_b: pl.LazyFrame) -> tuple[pl.LazyFrame, pl.LazyFrame]:
        """
        Prepares tables for join. Checks for expected columns and formats, adds missing columns as needed, and
        appends "_a" and "_b" suffixes to column names.

        :param df_a: Table A.
        :param df_b: Table B.

        :return: Tuple of normalized tables (df_a, df_b).
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

        # Apply column map
        if self.col_map is not None:
            df_a = df_a.rename(lambda col: self.col_map.get(col, col))
            df_b = df_b.rename(lambda col: self.col_map.get(col, col))

        # Check for expected columns
        columns_a = [col for col in df_a.collect_schema().names() if col in KNOWN_COL_SET]
        columns_b = [col for col in df_b.collect_schema().names() if col in KNOWN_COL_SET]

        missing_cols_a = [col for col in self.expected_cols if col not in columns_a]
        missing_cols_b = [col for col in self.expected_cols if col not in columns_b]

        if missing_cols_a or missing_cols_b:
            if missing_cols_a == missing_cols_b:
                raise ValueError(f'DataFrames "A" and "B" are missing expected column(s): {", ".join(missing_cols_a)}')
            else:
                raise ValueError(f'DataFrame "A" missing expected column(s): {", ".join(missing_cols_a)}; DataFrame "B" missing expected column(s): {", ".join(missing_cols_b)}')

        # Add svlen
        if 'svlen' not in columns_a:
            df_a = (
                df_a
                .with_columns(
                    pl.lit(None).cast(schema.VARIANT['svlen']).alias('svlen')
                )
            )

            columns_a.append('svlen')

        if 'svlen' not in columns_b:
            df_b = (
                df_b
                .with_columns(
                    pl.lit(None).cast(schema.VARIANT['svlen']).alias('svlen')
                )
            )

            columns_b.append('svlen')

        df_a = (
            df_a
            .with_columns(
                pl.col('svlen').fill_null(
                    pl.col('end') - pl.col('pos')
                )
            )
        )

        df_b = (
            df_b
            .with_columns(
                pl.col('svlen').fill_null(
                    pl.col('end') - pl.col('pos')
                )
            )
        )

        # Subset to known columns only (i.e. drop other columns that might confilct, such as an arbitrary "index" column).
        df_a = df_a.select(columns_a)
        df_b = df_b.select(columns_b)

        # Set index
        df_a = (
            df_a
            .with_row_index('index')
        )

        df_b = (
            df_b
            .with_row_index('index')
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
                pl.concat_str(pl.lit('VarIdx'), pl.col('index'))
            )
        )

        df_b = df_b.with_columns(
            pl.col('id').fill_null(
                pl.concat_str(pl.lit('VarIdx'), pl.col('index'))
            )
        )

        # ABS SVLEN (some sources may have negative SVLEN for DEL)
        df_a = df_a.with_columns(
            pl.col('svlen').abs()
        )

        df_b = df_b.with_columns(
            pl.col('svlen').abs()
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

        # Prepare SEQ
        if self.match_seq:
            df_a = df_a.with_columns(
                pl.col('seq').str.to_uppercase().str.strip_chars()
            )

            df_b = df_b.with_columns(
                pl.col('seq').str.to_uppercase().str.strip_chars()
            )

        # Get END for RO
        if not self.force_end_ro:
            df_a = df_a.with_columns(
                (pl.col('pos') + pl.col('svlen')).alias('end_ro')
            )

            df_b = df_b.with_columns(
                (pl.col('pos') + pl.col('svlen')).alias('end_ro')
            )
        else:
            df_a = df_a.with_columns(
                pl.col('end').alias('end_ro')
            )

            df_b = df_b.with_columns(
                pl.col('end').alias('end_ro')
            )

        # Append suffixes to all columns
        df_a = df_a.select(pl.all().name.suffix('_a'))
        df_b = df_b.select(pl.all().name.suffix('_b'))

        return df_a, df_b


def chunk_index(
        df_a: pl.LazyFrame,
        i_a: int,
        i_a_max: int
) -> pl.LazyFrame:
    """
    Chunk df_a by a range of indices. WARNING: This function assumes the index range contains a single chromosome,
    and this is not checked.

    :param df_a: Table to chunk.
    :param i_a: Minimum index.
    :param i_a_max: Maximum index.

    :return: df_a chunked (LazyFrame).
    """

    if i_a < 0 or i_a_max <= i_a:
        raise ValueError(f'Invalid index range: [{i_a}, {i_a_max})')

    return (
        df_a
        .filter(pl.col('index_a') >= i_a)
        .filter(pl.col('index_a') < i_a_max)
    )


def chunk_relative(
        df_a: pl.LazyFrame,
        df_b: pl.LazyFrame,
        chrom: str,
        chunk_range: dict[str,list[pl.Expr]],
) -> pl.LazyFrame:
    """
    Chunk df_b relative to df_a choosing records in df_b that could possibly be joined with some record in df_a.
    For example, this function may determine the minimum and maximum values of pos and end, and then subset df_b
    by those values. The actual subset values are determined by the `chunk_range` parameter.

    `chunk_range` is a dictionary with keys formatted as "column_limit" where "column" is a column name and "limit" is
    "min" or "max". Each value is a list of expressions to be applied to df_a, which will then determine the minimum
    or maximum value to be applied.

    For example, if "pos_min" is a key in chunk_range, then chunk_range['pos_min'] is a list of expressions. In this
    example, assume it is a list with the single expression "pl.col('pos_a') - pl.col('svlen_a')"). For each record in
    df_a, the expression will compute the position minus the variant length producing a single value for each record.
    Since this is a minimum value (i.e. key was "pos_min"), the minimum of these values (one per record in df_a) will
    be used to filter records in df_b by excluding any records with "pos_b" less than this minimum.

    The flexibility of this function is needed to support different limits. For example, when reciprocal overlap is
    used as a limit, the maximum value of pos_b is based on the maximum value of end_a (i.e. "chunk_range['pos_max']"
    will contain "pl.col('end_a')" because if pos_b greater than any "end_a", then variants cannot overlap.

    :param df_a: Table chunk.
    :param df_b: Table to be chunked to records that may intersect with df_a.
    :param chrom: Chromosome name.
    :param chunk_range: A dictionary of expressions to be applied to df_a to determine the minimum or maximum value for
        fields in df_b.

    :return: df_b partitioned (LazyFrame).
    """

    filter_list = [
        pl.col('chrom_b') == chrom
    ]

    for key, expr_list in chunk_range.items():
        col_name, limit = key.rsplit('_', 1)

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


def join_weight(
    df_join: pl.DataFrame|pl.LazyFrame,
    priority: list[tuple[str,float]]=None,
    offset_prop_max: float=2.0
) -> pl.DataFrame|pl.LazyFrame:
    """
    Add a "join_weight" column to an intersect DataFrame computed by summing weighted values across columns of the
    input DataFrame.

    :param df_join: Intersect DataFrame or LazyFrame.
    :param priority: A list of (column, weight) tuples.
    :param offset_prop_max: Maximum value for `offset_prop`.

    :return: A DataFrame or LazyFrame (same type as `df_join`) with a "join_weight" column.
    """

    # Fill priority presets
    if priority is None:
        priority = DEFAULT_PRIORITY['default']
    elif isinstance(priority, str):
        priority = DEFAULT_PRIORITY.get(priority.strip().lower(), None)

        if priority is None:
            raise ValueError(f'Unknown priority preset: "{priority}"')

    # Set lazy
    if isinstance(df_join, pl.DataFrame):
        df_join = df_join.lazy()
        do_collect = True
    else:
        do_collect = False

    # Compute weights
    df_join = df_join.with_columns(
        pl.lit(0.0).alias('join_weight').cast(pl.Float32)
    )

    for col, weight in priority:
        if col not in {'ro', 'size_ro', 'offset_prop', 'match_prop'}:
            raise ValueError(f'Priority column must be one of ro, size_ro, offset_prop, or match_prop: {col}')

        if col != 'offset_prop':
            df_join = df_join.with_columns(
                (
                    pl.col('join_weight') +
                    pl.col(col).cast(pl.Float32) * float(weight)
                ).alias('join_weight')
            )

        else:
            df_join = df_join.with_columns(
                (
                    pl.col('join_weight') +
                    (
                        (1 - pl.col(col).clip(0.0, offset_prop_max) / offset_prop_max).cast(pl.Float32)
                    ) * float(weight)
                ).alias('join_weight')
            )

    # Collect and return
    if do_collect:
        df_join = df_join.collect()

    return df_join

def join_iter(
    df_a: pl.DataFrame|pl.LazyFrame,
    df_b: pl.DataFrame|pl.LazyFrame,
    ro_min: float=None,
    size_ro_min: float=None,
    offset_max: int=None,
    offset_prop_max: float=None,
    match_prop_min: float=None,
    match_ref: bool=False,
    match_alt: bool=False,
    col_map: dict[str,str]=None,
    match_score_model: seqmatch.MatchScoreModel=None,
    force_end_ro: bool=False,
    chunk_size: int=10_000
) -> Generator[pl.LazyFrame, None, None]:
    """
    A convenience wrapper for creating a PairwiseIntersect object and calling its join_iter() method.

    :param df_a: Table A.
    :param df_b: Table B.
    :param ro_min: Minimum reciprocal overlap for allowed matches. If 0.0, then any overlap matches.
    :param size_ro_min: Reciprocal length proportion of allowed matches. If `match_prop_min` is set and the
        value of this parameter is `None` or is less than `match_prop_min`, then it is set to `match_prop_min` since
        this value represents the lower-bound of allowed match proportions.
    :param offset_max: Maximum offset allowed (minimum of start or end position distance).
    :param offset_prop_max: Maximum size-offset (offset / svlen) allowed.
    :param match_prop_min: Minimum matched base proportion in alignment or None to not match.
    :param match_ref: "REF" column must match between two variants.
    :param match_alt: "ALT" column must match between two variants.
    :param col_map: Map column names from input dataframes by this dictionary. May be string "bed" to use default
        BED column names.
    :param match_score_model: Configured model for scoring similarity between pairs of sequences. If `None` and
        `match_prop_min` is set, then a default aligner will be used.
    :param force_end_ro: By default, reciprocal overlap is calculated with the end position set to the start
        position plus the variant length. For all variants except insertions, this will typically match the end value
        in the source DataFrame. If `True`, the end position in the DataFrame is also used for reciprocal overlap
        without changes. Typically, this option should not be used and will break reciprocal overlap for insertions.
    :param chunk_size: Chunk df_a into partitions of this size, and for each chunk, subset df_b to include only
        variants that may intersect with variants in the chunk. If None, each chromosome is a single chunk, which will
        lead to a combinatorial explosion unless offset_max is 0 (the join operation is optimized without chunking in
        this case). All intersects are first chunked by chromosome, then by this chunk size.

    :return: An iterator of join results.
    """

    return PairwiseIntersect(
        ro_min=ro_min,
        size_ro_min=size_ro_min,
        offset_max=offset_max,
        offset_prop_max=offset_prop_max,
        match_prop_min=match_prop_min,
        match_ref=match_ref,
        match_alt=match_alt,
        col_map=col_map,
        match_score_model=match_score_model,
        force_end_ro=force_end_ro,
        chunk_size=chunk_size
    ).join_iter(df_a, df_b)


def join(
    df_a: pl.DataFrame|pl.LazyFrame,
    df_b: pl.DataFrame|pl.LazyFrame,
    ro_min: float=None,
    size_ro_min: float=None,
    offset_max: int=None,
    offset_prop_max: float=None,
    match_prop_min: float=None,
    match_ref: bool=False,
    match_alt: bool=False,
    col_map: dict[str,str]=None,
    match_score_model: seqmatch.MatchScoreModel=None,
    force_end_ro: bool=False,
    chunk_size: int=10_000
) -> pl.LazyFrame:
    """
    A convenience wrapper for creating a PairwiseIntersect object and calling its join() method.

    :param df_a: Table A.
    :param df_b: Table B.
    :param ro_min: Minimum reciprocal overlap for allowed matches. If 0.0, then any overlap matches.
    :param size_ro_min: Reciprocal length proportion of allowed matches. If `match_prop_min` is set and the
        value of this parameter is `None` or is less than `match_prop_min`, then it is set to `match_prop_min` since
        this value represents the lower-bound of allowed match proportions.
    :param offset_max: Maximum offset allowed (minimum of start or end position distance).
    :param offset_prop_max: Maximum size-offset (offset / svlen) allowed.
    :param match_prop_min: Minimum matched base proportion in alignment or None to not match.
    :param match_ref: "REF" column must match between two variants.
    :param match_alt: "ALT" column must match between two variants.
    :param col_map: Map column names from input dataframes by this dictionary. May be string "bed" to use default
        BED column names.
    :param match_score_model: Configured model for scoring similarity between pairs of sequences. If `None` and
        `match_prop_min` is set, then a default aligner will be used.
    :param force_end_ro: By default, reciprocal overlap is calculated with the end position set to the start
        position plus the variant length. For all variants except insertions, this will typically match the end value
        in the source DataFrame. If `True`, the end position in the DataFrame is also used for reciprocal overlap
        without changes. Typically, this option should not be used and will break reciprocal overlap for insertions.
    :param chunk_size: Chunk df_a into partitions of this size, and for each chunk, subset df_b to include only
        variants that may intersect with variants in the chunk. If None, each chromosome is a single chunk, which will
        lead to a combinatorial explosion unless offset_max is 0 (the join operation is optimized without chunking in
        this case). All intersects are first chunked by chromosome, then by this chunk size.

    :return: A join table.
    """

    return PairwiseIntersect(
        ro_min=ro_min,
        size_ro_min=size_ro_min,
        offset_max=offset_max,
        offset_prop_max=offset_prop_max,
        match_prop_min=match_prop_min,
        match_ref=match_ref,
        match_alt=match_alt,
        col_map=col_map,
        match_score_model=match_score_model,
        force_end_ro=force_end_ro,
        chunk_size=chunk_size
    ).join(df_a, df_b)